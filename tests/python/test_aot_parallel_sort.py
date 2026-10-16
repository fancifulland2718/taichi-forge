"""Frozen provider SPIR-V supports mixed AOT graphs without consumer DXC."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_parallel_sort_aot_capability_is_backend_specific():
    with ti.aot.borrow_runtime() as runtime:
        assert runtime.probe_native_operation("vulkan_parallel_sort_u32") == {
            "available": False,
            "capabilities": frozenset(),
            "library_version": None,
        }


@pytest.mark.parametrize("payload,fused", [(False, True), (True, False), (True, True)])
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_parallel_sort_recording_export_and_downstream_grad(tmp_path, monkeypatch, payload, fused):
    from taichi_forge.hardware import _parallel_sort

    compiler = os.environ.get("TI_TEST_PARALLEL_SORT_DXC")
    if not compiler:
        pytest.skip("set TI_TEST_PARALLEL_SORT_DXC for the producer")
    n = 513
    keys = ti.ndarray(ti.u32, n)
    values = ti.ndarray(ti.u32, n) if payload else None
    plans = [ti.hardware.sort.VulkanParallelSortPlan(keys, values, compiler_path=compiler, fuse_prefix=fused)]
    # Independent workspaces can share identical immutable shader bytes.
    if payload and fused:
        plans.append(
            ti.hardware.sort.VulkanParallelSortPlan(keys, values, compiler_path=compiler, fuse_prefix=fused)
        )
    arg = lambda name, dtype=ti.f32: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, dtype, ndim=1)
    source_arg, keys_arg = arg("source", ti.u32), arg("keys", ti.u32)
    values_arg = arg("values", ti.u32)
    w, dw, loss = (arg(name) for name in ("w", "dw", "loss"))

    @ti.kernel
    def copy(source: ti.types.ndarray(ti.u32, 1), keys: ti.types.ndarray(ti.u32, 1)):
        for i in keys:
            keys[i] = source[i]

    @ti.kernel
    def indices(values: ti.types.ndarray(ti.u32, 1)):
        for i in values:
            values[i] = ti.cast(i, ti.u32)

    @ti.kernel
    def seed(
        dw: ti.types.ndarray(ti.f32, 1), loss: ti.types.ndarray(ti.f32, 1), dl: ti.types.ndarray(ti.f32, 1)
    ):
        for i in dw:
            dw[i] = 0
        loss[0], dl[0] = 0, 1

    @ti.kernel
    def energy(
        keys: ti.types.ndarray(ti.u32, 1),
        w: ti.types.ndarray(ti.f32, 1, needs_grad=True),
        loss: ti.types.ndarray(ti.f32, 1, needs_grad=True),
    ):
        for i in keys:
            loss[0] += 0.5 * (ti.cast(keys[i] & 7, ti.f32) + 1) * w[i] * w[i]

    @ti.kernel
    def indexed_energy(
        keys: ti.types.ndarray(ti.u32, 1),
        order: ti.types.ndarray(ti.u32, 1),
        w: ti.types.ndarray(ti.f32, 1, needs_grad=True),
        loss: ti.types.ndarray(ti.f32, 1, needs_grad=True),
    ):
        for i in keys:
            j = order[i]
            loss[0] += 0.5 * (ti.cast(keys[i] & 7, ti.f32) + 1) * w[j] * w[j]

    builder = ti.graph.GraphBuilder()
    dl = builder.private_ndarray("dloss", ti.f32, 1)
    builder.dispatch(copy, source_arg, keys_arg)
    if payload:
        builder.dispatch(indices, values_arg)
    builder.dispatch(seed, dw, loss, dl)
    for plan in plans:
        builder.append_native(plan.record())
        builder.append_native(plan.record())
    if payload:
        builder.dispatch(indexed_energy, keys_arg, values_arg, w, loss)
        builder.dispatch(indexed_energy.grad, keys_arg, values_arg, w, loss)
    else:
        builder.dispatch(energy, keys_arg, w, loss)
        builder.dispatch(energy.grad, keys_arg, w, loss)
    graph = builder.compile()
    inputs = (np.arange(n, dtype=np.uint32) * 7) % 37
    inputs[0] = 0xFFFFFFFF
    # Validate the frozen sort through JIT as well. The standalone consumers
    # below exercise the downstream kernel.grad with explicit AOT bindings.
    keys.from_numpy(inputs)
    if payload:
        values.from_numpy(np.arange(n, dtype=np.uint32))
    for plan in plans:
        plan.run()
    order = np.argsort(inputs, kind="stable")
    np.testing.assert_array_equal(keys.to_numpy(), inputs[order])
    if payload:
        np.testing.assert_array_equal(values.to_numpy(), order.astype(np.uint32))
    snapshot = keys.to_numpy()

    def no_compile(*args, **kwargs):
        raise AssertionError("AOT export repeated shader compilation")

    monkeypatch.setattr(_parallel_sort, "compile_shaders", no_compile)
    module = ti.aot.Module()
    description = json.loads(
        module.add_execution_plan("step", graph, gradients=dict(w="dw", loss="dloss")).to_json()
    )
    assert "compiler_path" not in json.dumps(description)
    native = [r for r in description["resources"] if r["kind"] == "native"]
    blobs = [r for r in description["resources"] if r["kind"] == "host_blob"]
    assert len(native) == len(plans) and len(blobs) == 1
    assert all(r["bindings"]["code"] == blobs[0]["name"] for r in native)
    assert blobs[0]["byte_size"] == len(plans[0]._aot_shader_payload)
    for plan in plans:
        assert plan.memory_report().known_resident_requested_bytes == (
            plan.statistics()["workspace_bytes"] + len(plan._aot_shader_payload)
        )
    np.testing.assert_array_equal(keys.to_numpy(), snapshot)
    graph.close()
    for plan in plans:
        plan.close()
        assert plan._aot_shader_payload == b""
        assert plan.memory_report().known_resident_requested_bytes == 0
    artifact = tmp_path / "parallel-sort.tcm"
    module.archive(str(artifact))
    environment = os.environ.copy()
    environment["TI_TEST_PARALLEL_SORT_DXC"] = "consumer-has-no-dxc.exe"
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_parallel_sort_probe.py")),
                mode,
                str(artifact),
            ],
            env=environment,
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Unable to release execution resource" not in result.stdout + result.stderr
