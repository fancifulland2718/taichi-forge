"""Packed Field transforms retain their selected native execution and roots."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("shape,dtype,inplace", [((2, 3), ti.f32, False), ((3,), ti.u32, True)])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, cpu_max_num_threads=4)
def test_packed_field_transform_cold_warm_roots_and_gradient(tmp_path, shape, dtype, inplace):
    n = 17

    def allocate():
        if len(shape) == 1:
            return ti.Vector.field(shape[0], dtype, shape=n)
        return ti.Matrix.field(*shape, dtype, shape=n)

    src = allocate()
    dst = src if inplace else allocate()
    src.fill(37)
    dst.fill(37)
    sequence = ti.algorithms.primitive_sequence().transform(src, dst, scale=3, bias=7)
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    graph = builder.compile()
    module = ti.aot.Module()
    cold = module.add_execution_plan("cold", graph, snapshot_state=True)
    document = json.loads(cold.to_json())
    operation = document["operations"][0]
    assert len(document["operations"]) == 1 and not module._exported_kernels
    assert operation["parameters"]["strategy"] == "packed_field"
    assert operation["parameters"]["element_shape"] == list(shape)
    assert all(item["kind"] == "module_array" for item in document["resources"])
    np.testing.assert_array_equal(dst.to_numpy(), np.full((n, *shape), 37))
    host = np.arange(n * np.prod(shape), dtype=np.float32 if dtype == ti.f32 else np.uint32).reshape(n, *shape)
    if dtype == ti.u32:
        host.flat[0] = np.iinfo(np.uint32).max
    src.from_numpy(host)
    sequence.run()
    expected = host * host.dtype.type(3) + host.dtype.type(7)
    np.testing.assert_array_equal(dst.to_numpy(), expected)
    assert sequence._calls[0].plan.method_name == "transform_affine_dense_field_packed"
    src.fill(37)
    dst.fill(37)
    module.add_execution_plan("warm", graph, snapshot_state=True)
    assert not module._exported_kernels  # No transform helper or packing kernel.
    operations = [
        dict(kind="copy", source="input", destination=operation["bindings"]["src"]),
        operation,
        dict(kind="copy", source=operation["bindings"]["dst"], destination="output"),
    ]

    @ti.kernel
    def energy(x: ti.types.ndarray(ti.f32, 1, needs_grad=True), y: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in x:
            y[i] = 0.5 * x[i] * x[i]

    gradients = {}
    if dtype == ti.f32:
        module.add_kernel(energy, name="energy")
        module.add_kernel(energy.grad, name="backward")
        operations += [
            dict(kind="fill32", destination="dx", value=0),
            dict(kind="fill32", destination="dy", value=0x3F800000),
            dict(kind="kernel", entry="energy", arguments=["output", "energy"]),
            dict(kind="kernel", entry="backward", arguments=["output", "energy"]),
        ]
        gradients = dict(output="dx", energy="dy")
    module.add_execution_plan(
        "step",
        ti.aot.ExecutionPlan(
            operations,
            resources=document["resources"],
            initialization=document.get("initialization"),
            gradients=gradients,
        ),
    )
    # Observe module state without overwriting it, and exercise a prefix of the
    # same packed tensor allocation without losing the trailing scalar lanes.
    module.add_execution_plan(
        "read",
        ti.aot.ExecutionPlan(
            [dict(kind="copy", source=operation["bindings"]["dst"], destination="output")],
            resources=document["resources"],
        ),
    )
    prefix = dict(operation, parameters=dict(operation["parameters"], size=2))
    module.add_execution_plan(
        "prefix",
        ti.aot.ExecutionPlan(
            [
                dict(kind="copy", source="input", destination=operation["bindings"]["src"]),
                prefix,
                dict(kind="copy", source=operation["bindings"]["dst"], destination="output"),
            ],
            resources=document["resources"],
        ),
    )
    artifact = tmp_path / "packed-field.tcm"
    module.archive(str(artifact))
    artifact.with_suffix(".json").write_text(json.dumps(dict(dtype=str(dtype), shape=shape, size=n, inplace=inplace)))
    graph.close()
    sequence.clear()
    arch = "cpu" if ti.lang.impl.current_cfg().arch == ti.cpu else ti.lang.impl.current_cfg().arch.name
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_field_transform_probe.py")),
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
        assert "Validation Error" not in text and "SYNC-HAZARD" not in text, text


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_packed_field_transform_rejects_changed_strategy_and_layout():
    src, dst = (ti.Vector.field(3, ti.f32, shape=7) for _ in range(2))
    sequence = ti.algorithms.primitive_sequence().transform(src, dst)
    sequence.run()
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    graph = builder.compile()
    sequence._calls[0].plan.method_name = "cpu_transform_affine_packed_strided_ndarray"
    module = ti.aot.Module()
    with pytest.raises(ti.TaichiRuntimeError, match="selected cached strategy"):
        module.add_execution_plan("changed", graph, snapshot_state=True)
    assert not module._content and not module._execution_plans
    graph.close()
    # A tensor member inside an interleaved Struct is still a distinct strided
    # contract; accepting packed Fields must not silently flatten its neighbors.
    member = ti.Struct.field(dict(value=ti.types.vector(3, ti.f32), tag=ti.i32), shape=7)
    seq = ti.algorithms.primitive_sequence().transform(member.value, dst)
    builder = ti.graph.GraphBuilder()
    builder.append_native(seq)
    graph = builder.compile()
    with pytest.raises(ti.TaichiRuntimeError, match="packed"):
        module.add_execution_plan("strided", graph, snapshot_state=True)
    assert not module._content and not module._execution_plans
    graph.close()
