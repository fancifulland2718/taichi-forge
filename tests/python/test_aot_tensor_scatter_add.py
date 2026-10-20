"""Packed Field accumulation preserves algorithms, state and public indices."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.graph._native import NativeGraphExecutable, NativeGraphNode
from tests import test_utils


class _Parent(NativeGraphNode, NativeGraphExecutable):
    def __init__(self, graph, output):
        self.graph, self.bindings = graph, graph.bind(dict(out=output))

    def compile(self):
        return self

    def run(self):
        self.graph.run(self.bindings)

    def _export_aot(self, context):
        context.bound_graph(self.graph, self.bindings)


@pytest.mark.parametrize("case", ["matrix", "vector64", "wide_integer"])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, cpu_max_num_threads=4)
def test_tensor_scatter_add_cold_warm_parent_mutation_and_gradient(tmp_path, case):
    vector = case == "vector64"
    rows, cols = (3, 1) if vector else (8, 8) if case == "wide_integer" else (2, 3)
    n, bound = (5 if case == "wide_integer" else 17), 4
    dtype = ti.f64 if vector else ti.i32 if case == "wide_integer" else ti.f32
    if ti.cfg.arch == ti.vulkan and not ti.lang.impl.get_runtime().prog.vulkan_scatter_add_value_type_available(
        ti.algorithms._algorithms._scatter_add_value_type(dtype)
    ):
        pytest.skip("Vulkan target lacks this native atomic scalar type")
    shape, lanes = ((rows,) if vector else (rows, cols)), rows * cols
    scalar = np.float64 if vector else np.int32 if case == "wide_integer" else np.float32
    field = (
        (lambda size: ti.Vector.field(rows, dtype, shape=size))
        if vector
        else (lambda size: ti.Matrix.field(rows, cols, dtype, shape=size))
    )
    src, dst = field(n), field(bound)
    indices = ti.ndarray(ti.i32, n) if vector else ti.field(ti.i32, n)
    public = dict(input_indices=indices) if vector else {}
    host = (np.arange(n * lanes).reshape((n, *shape)) % 13 - 6).astype(scalar)
    initial = (np.arange(bound * lanes).reshape((bound, *shape)) % 5 + 37).astype(scalar)
    keys = np.arange(n, dtype=np.int32) % bound
    keys[0], keys[-1] = -1, bound
    src.from_numpy(host)
    indices.from_numpy(keys)
    dst.from_numpy(initial)
    output = ti.ndarray(dtype, (bound + n) * lanes)

    @ti.kernel
    def observe(out: ti.types.ndarray(dtype, 1)):
        for i, lane in ti.ndrange(bound, lanes):
            if ti.static(vector):
                out[i * lanes + lane] = dst[i][lane]
            else:
                out[i * lanes + lane] = dst[i][lane // cols, lane % cols]
        for i, lane in ti.ndrange(n, lanes):
            if ti.static(vector):
                out[(bound + i) * lanes + lane] = src[i][lane]
            else:
                out[(bound + i) * lanes + lane] = src[i][lane // cols, lane % cols]

    sequence = ti.algorithms.primitive_sequence().scatter_add(src, indices, dst)
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    builder.dispatch(observe, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", dtype, ndim=1))
    graph = builder.compile()
    caps = ["spirv_has_int64", "spirv_has_float64"] if vector and ti.cfg.arch == ti.vulkan else []
    module = ti.aot.Module(caps=caps)
    cold = module.add_execution_plan("cold", graph, source_bindings=public, snapshot_state=True)
    document = json.loads(cold.to_json())
    native = document["operations"][0]
    assert len(document["operations"]) == 2 and len(module._exported_kernels) == 1
    assert native["parameters"]["strategy"] == "packed_field"
    assert native["parameters"]["element_shape"] == list(shape)
    np.testing.assert_array_equal(dst.to_numpy(), initial)
    graph.run(dict(out=output))
    delta = np.zeros_like(initial)
    valid = (keys >= 0) & (keys < bound)
    np.add.at(delta, keys[valid], host[valid])
    np.testing.assert_array_equal(output.to_numpy(), np.concatenate([(initial + delta).ravel(), host.ravel()]))
    assert sequence._calls[0].plan.method_name == (
        "scatter_add_dense_field_packed" if vector else "scatter_add_dense_field_packed_indices_field"
    )
    dst.from_numpy(initial)
    module.add_execution_plan("warm", graph, source_bindings=public, snapshot_state=True)
    parent_builder = ti.graph.GraphBuilder()
    parent_builder.append_native(_Parent(graph, output))
    parent = parent_builder.compile()
    module.add_execution_plan("parent", parent, source_bindings=dict(public, out=output), snapshot_state=True)
    assert len(module._exported_kernels) == 1
    module.add_execution_plan("read", ti.aot.ExecutionPlan([document["operations"][-1]]))
    module.add_execution_plan(
        "reset",
        ti.aot.ExecutionPlan(
            [dict(kind="copy", source="initial_values", destination=native["bindings"]["dst"])],
            resources=document["resources"],
        ),
    )
    if not vector:
        module.add_execution_plan(
            "set_indices",
            ti.aot.ExecutionPlan(
                [dict(kind="copy", source="index_updates", destination=native["bindings"]["indices"])],
                resources=document["resources"],
            ),
        )
    prefix = json.loads(cold.to_json())
    prefix["operations"][0]["parameters"].update(size=n - 1, dst_items=bound - 1)
    module.add_execution_plan("prefix", ti.aot.ExecutionPlan.from_json(json.dumps(prefix)))
    prefix_delta = np.zeros_like(initial)
    valid = (keys[: n - 1] >= 0) & (keys[: n - 1] < bound - 1)
    np.add.at(prefix_delta, keys[: n - 1][valid], host[: n - 1][valid])
    for name, params in (
        ("bad_shape", dict(element_shape=[rows, 1] if vector else [1, lanes])),
        ("bad_extent", dict(size=n + 1)),
        ("bad_strategy", dict(strategy="packed_array")),
    ):
        bad = json.loads(cold.to_json())
        bad["operations"][0]["parameters"].update(params)
        module.add_execution_plan(name, ti.aot.ExecutionPlan.from_json(json.dumps(bad)))
    bad = json.loads(cold.to_json())
    bad["operations"][0]["bindings"]["dst"] = bad["operations"][0]["bindings"]["src"]
    module.add_execution_plan("bad_alias", ti.aot.ExecutionPlan.from_json(json.dumps(bad)))
    if dtype != ti.i32:

        @ti.kernel
        def energy(x: ti.types.ndarray(dtype, 1, needs_grad=True), y: ti.types.ndarray(dtype, 1, needs_grad=True)):
            for i in x:
                y[i] = 0.5 * x[i] * x[i]

        module.add_kernel(energy)
        module.add_kernel(energy.grad, name="reverse")
        module.add_execution_plan(
            "suffix",
            ti.aot.ExecutionPlan(
                document["operations"]
                + [
                    dict(kind="fill32", destination="dx", value=0),
                    dict(kind="kernel", entry="energy", arguments=["out", "energy"]),
                    dict(kind="kernel", entry="reverse", arguments=["out", "energy"]),
                ],
                resources=document["resources"],
                gradients=dict(out="dx", energy="dy"),
            ),
        )
    artifact = tmp_path / f"{case}.tcm"
    module.archive(str(artifact))
    artifact.with_suffix(".json").write_text(
        json.dumps(
            dict(
                n=n,
                bound=bound,
                shape=shape,
                lanes=lanes,
                dtype=str(dtype),
                public_indices=vector,
                initial=initial.ravel().tolist(),
                source=host.ravel().tolist(),
                indices=keys.tolist(),
                delta=delta.ravel().tolist(),
                prefix_delta=prefix_delta.ravel().tolist(),
            )
        )
    )
    parent.close()
    graph.close()
    sequence.clear()
    arch = "cpu" if ti.cfg.arch == ti.cpu else ti.cfg.arch.name
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                str(Path(__file__).with_name("aot_tensor_scatter_add_probe.py")),
                mode,
                arch,
                str(artifact),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        text = result.stdout + result.stderr
        assert result.returncode == 0, text
        assert not any(marker in text for marker in ("Validation Error", "SYNC-HAZARD", "cleanup failed")), text
