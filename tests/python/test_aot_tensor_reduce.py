"""Whole tensor reductions preserve lane order, shared scratch and frozen roots."""

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
def test_tensor_reduce_cold_warm_parent_prefix_and_gradient(tmp_path, case):
    vector = case == "vector64"
    rows, cols = (3, 1) if vector else (8, 8) if case == "wide_integer" else (2, 3)
    n = 5 if case == "wide_integer" else 17
    dtype = ti.f64 if vector else ti.i32 if case == "wide_integer" else ti.f32
    if ti.cfg.arch == ti.vulkan and not ti.lang.impl.get_runtime().prog.vulkan_reduce_value_type_available(
        ti.algorithms._algorithms._REDUCE_VALUE_TYPE[dtype]
    ):
        pytest.skip("Vulkan device lacks this native reduction scalar type")
    lanes = rows * cols
    shape = (rows,) if vector else (rows, cols)

    def field(index_shape):
        if vector:
            return ti.Vector.field(rows, dtype, shape=index_shape)
        return ti.Matrix.field(rows, cols, dtype, shape=index_shape)

    values = field(n)
    total, low, high = field(()), field(()), field(())
    host = (np.arange(n * lanes).reshape((n, *shape)) % 19 - 7).astype(
        np.float64 if vector else np.int32 if case == "wide_integer" else np.float32
    )
    values.from_numpy(host)
    for dst in (total, low, high):
        dst.fill(-9)
    count = (n + 3) * lanes
    output = ti.ndarray(dtype, count)

    @ti.kernel
    def observe(out: ti.types.ndarray(dtype, 1)):
        for lane in range(lanes):
            if ti.static(vector):
                out[lane] = total[None][lane]
                out[lanes + lane] = low[None][lane]
                out[2 * lanes + lane] = high[None][lane]
            else:
                out[lane] = total[None][lane // cols, lane % cols]
                out[lanes + lane] = low[None][lane // cols, lane % cols]
                out[2 * lanes + lane] = high[None][lane // cols, lane % cols]
        for i, lane in ti.ndrange(n, lanes):
            if ti.static(vector):
                out[(3 + i) * lanes + lane] = values[i][lane]
            else:
                out[(3 + i) * lanes + lane] = values[i][lane // cols, lane % cols]

    sequence = ti.algorithms.primitive_sequence()
    for op, destination in (("sum", total), ("min", low), ("max", high)):
        sequence.reduce(values, destination, op=op)
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    builder.dispatch(observe, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", dtype, ndim=1))
    graph = builder.compile()
    caps = ["spirv_has_int64", "spirv_has_float64"] if vector and ti.cfg.arch == ti.vulkan else []
    module = ti.aot.Module(caps=caps)
    cold = module.add_execution_plan("cold", graph, snapshot_state=True)
    document = json.loads(cold.to_json())
    assert len(document["operations"]) == 4 and len(module._exported_kernels) == 1
    for op in document["operations"][:3]:
        assert op["parameters"]["strategy"] == "packed_field"
        assert op["parameters"]["element_shape"] == list(shape)
    np.testing.assert_array_equal(total.to_numpy(), np.full(shape, -9))
    graph.run(dict(out=output))
    expected = np.concatenate([getattr(np, op)(host, axis=0).ravel() for op in ("sum", "min", "max")] + [host.ravel()])
    np.testing.assert_array_equal(output.to_numpy(), expected)
    assert all("reduce_dense_field_packed" in call.plan.method_name for call in sequence._calls)
    for dst in (total, low, high):
        dst.fill(-9)
    module.add_execution_plan("warm", graph, snapshot_state=True)
    parent_builder = ti.graph.GraphBuilder()
    parent_builder.append_native(_Parent(graph, output))
    parent = parent_builder.compile()
    module.add_execution_plan("parent", parent, snapshot_state=True, source_bindings=dict(out=output))
    assert len(module._exported_kernels) == 1  # Only observe; no lane/packing helper.
    module.add_execution_plan("read", ti.aot.ExecutionPlan([document["operations"][-1]]))
    prefix_count = min(11, n - 1)
    prefix = json.loads(cold.to_json())
    for op in prefix["operations"][:3]:
        op["parameters"]["size"] = prefix_count
    module.add_execution_plan("prefix", ti.aot.ExecutionPlan.from_json(json.dumps(prefix)))
    for name, parameters in (
        ("bad_shape", dict(element_shape=[lanes, 1] if vector else [1, lanes])),
        ("bad_extent", dict(size=n + 1)),
        ("bad_strategy", dict(strategy="strided_scalar")),
    ):
        bad = json.loads(cold.to_json())
        bad["operations"][0]["parameters"].update(parameters)
        module.add_execution_plan(name, ti.aot.ExecutionPlan.from_json(json.dumps(bad)))
    bad = json.loads(cold.to_json())
    bad["operations"][0]["bindings"]["output"] = bad["operations"][0]["bindings"]["values"]
    module.add_execution_plan("bad_alias", ti.aot.ExecutionPlan.from_json(json.dumps(bad)))
    if dtype != ti.i32:

        @ti.kernel
        def energy(x: ti.types.ndarray(dtype, 1, needs_grad=True), y: ti.types.ndarray(dtype, 1, needs_grad=True)):
            for i in x:
                y[i] = 0.5 * x[i] * x[i]

        module.add_kernel(energy)
        module.add_kernel(energy.grad, name="reverse")
        # dy is supplied by the consumer so f32/f64 share the same real adjoint.
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
                shape=shape,
                lanes=lanes,
                dtype=str(dtype),
                prefix=prefix_count,
                values=host.ravel().tolist(),
                expected=expected.tolist(),
                prefix_expected=np.concatenate(
                    [getattr(np, op)(host[:prefix_count], axis=0).ravel() for op in ("sum", "min", "max")]
                    + [host.ravel()]
                ).tolist(),
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
                str(Path(__file__).with_name("aot_tensor_reduce_probe.py")),
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
