"""Export the selected packed clear-and-scatter group without extra kernels."""

import copy
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
def test_tensor_grouped_reduce_cold_warm_clear_keys_parent_and_gradient(tmp_path, case):
    vector = case == "vector64"
    rows, cols = (3, 1) if vector else (8, 8) if case == "wide_integer" else (2, 3)
    n, groups = (5 if case == "wide_integer" else 17), 4
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
    values, result = field(n), field(groups)
    keys = ti.ndarray(ti.i32, n) if vector else ti.field(ti.i32, n)
    public = dict(input_keys=keys) if vector else {}
    host = (np.arange(n * lanes).reshape((n, *shape)) % 13 - 6).astype(scalar)
    initial = (np.arange(groups * lanes).reshape((groups, *shape)) % 5 + 37).astype(scalar)
    ids = np.arange(n, dtype=np.int32) % (groups - 1)
    ids[0], ids[-1] = -1, groups
    values.from_numpy(host)
    keys.from_numpy(ids)
    result.from_numpy(initial)
    output = ti.ndarray(dtype, (groups + n) * lanes)

    @ti.kernel
    def observe(out: ti.types.ndarray(dtype, 1)):
        for i, lane in ti.ndrange(groups, lanes):
            if ti.static(vector):
                out[i * lanes + lane] = result[i][lane]
            else:
                out[i * lanes + lane] = result[i][lane // cols, lane % cols]
        for i, lane in ti.ndrange(n, lanes):
            if ti.static(vector):
                out[(groups + i) * lanes + lane] = values[i][lane]
            else:
                out[(groups + i) * lanes + lane] = values[i][lane // cols, lane % cols]

    sequence = ti.algorithms.primitive_sequence().grouped_reduce(keys, values, result)
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    builder.dispatch(observe, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", dtype, ndim=1))
    graph = builder.compile()
    caps = ["spirv_has_int64", "spirv_has_float64"] if vector and ti.cfg.arch == ti.vulkan else []
    module = ti.aot.Module(caps=caps)
    cold = module.add_execution_plan("cold", graph, source_bindings=public, snapshot_state=True)
    document = json.loads(cold.to_json())
    fill, scatter, observation = document["operations"]
    assert fill == dict(kind="fill32", destination=scatter["bindings"]["dst"], value=0)
    assert scatter["operation"] == "scatter_add" and scatter["parameters"]["element_shape"] == list(shape)
    assert len(module._exported_kernels) == 1
    np.testing.assert_array_equal(result.to_numpy(), initial)
    graph.run(dict(out=output))
    expected = np.zeros_like(initial)
    valid = (ids >= 0) & (ids < groups)
    np.add.at(expected, ids[valid], host[valid])
    np.testing.assert_array_equal(output.to_numpy(), np.concatenate([expected.ravel(), host.ravel()]))
    call = sequence._calls[0]
    assert type(call.plan) is ti.algorithms._algorithms._NativePrimitivePlanGroup
    assert [p.method_name for p in call.plan.plans] == [
        "fill_dense_field_packed",
        "scatter_add_dense_field_packed" if vector else "scatter_add_dense_field_packed_indices_field",
    ]
    result.from_numpy(initial)
    module.add_execution_plan("warm", graph, source_bindings=public, snapshot_state=True)
    # Changed or incomplete cached groups must not be silently replaced.
    original = call.plan
    try:
        for index, bad_plan in enumerate((copy.copy(original), copy.copy(original))):
            if index == 0:
                bad_fill = copy.copy(original.plans[0])
                bad_fill.semantic_key = ("wrong_zeroing_rule",)
                bad_plan.plans = (bad_fill, original.plans[1])
            else:
                bad_plan.plans = (original.plans[1],)
            call.plan = bad_plan
            with pytest.raises(RuntimeError, match="selected"):
                module.add_execution_plan(f"bad_cached_{index}", graph, source_bindings=public, snapshot_state=True)
    finally:
        call.plan = original
    parent_builder = ti.graph.GraphBuilder()
    parent_builder.append_native(_Parent(graph, output))
    parent = parent_builder.compile()
    module.add_execution_plan("parent", parent, source_bindings=dict(public, out=output), snapshot_state=True)
    assert len(module._exported_kernels) == 1
    module.add_execution_plan("read", ti.aot.ExecutionPlan([observation]))
    module.add_execution_plan(
        "reset",
        ti.aot.ExecutionPlan(
            [dict(kind="copy", source="initial_values", destination=scatter["bindings"]["dst"])],
            resources=document["resources"],
        ),
    )
    if not vector:
        module.add_execution_plan(
            "set_keys",
            ti.aot.ExecutionPlan(
                [dict(kind="copy", source="key_updates", destination=scatter["bindings"]["indices"])],
                resources=document["resources"],
            ),
        )
    for name, params in (
        ("bad_shape", dict(element_shape=[rows, 1] if vector else [1, lanes])),
        ("bad_extent", dict(size=n + 1)),
    ):
        bad = json.loads(cold.to_json())
        bad["operations"][1]["parameters"].update(params)
        module.add_execution_plan(name, ti.aot.ExecutionPlan.from_json(json.dumps(bad)))
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
                groups=groups,
                shape=shape,
                lanes=lanes,
                dtype=str(dtype),
                public_keys=vector,
                initial=initial.ravel().tolist(),
                source=host.ravel().tolist(),
                keys=ids.tolist(),
                expected=expected.ravel().tolist(),
            )
        )
    )
    parent.close()
    graph.close()
    sequence.clear()
    arch = "cpu" if ti.cfg.arch == ti.cpu else ti.cfg.arch.name
    for mode in ("standalone", "borrowed"):
        completed = subprocess.run(
            [
                sys.executable,
                "-S",
                str(Path(__file__).with_name("aot_tensor_grouped_reduce_probe.py")),
                mode,
                arch,
                str(artifact),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        text = completed.stdout + completed.stderr
        assert completed.returncode == 0, text
        assert not any(marker in text for marker in ("Validation Error", "SYNC-HAZARD", "cleanup failed")), text
