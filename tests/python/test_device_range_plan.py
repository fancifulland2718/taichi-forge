"""Selected task strategies survive JIT recording and AOT code sharing."""
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.lang._offload_execution_plan import (
    _OffloadExecutionPlan, _bind_offload_execution_plan,
)
from tests import test_utils


def select(kernel, *arrays):
    tasks = kernel.task_manifest(*arrays)
    baseline = _OffloadExecutionPlan.from_task_manifests(tasks)
    selected = baseline
    for task in tasks:
        if task.device_range_dispatch_version == 1:
            selected = selected.replace_task(task.task_index, range_dispatch="device")
    assert selected.device_range_tasks
    assert selected.compilation_identity == baseline.compilation_identity
    assert selected.identity != baseline.identity
    return baseline, selected


def graph(kernel, plan, symbols):
    builder = ti.graph.GraphBuilder()
    builder.dispatch(_bind_offload_execution_plan(kernel, plan), *symbols)
    return builder.compile()


def consume(artifact):
    result = subprocess.run(
        [sys.executable, "-I", str(Path(__file__).with_name("aot_device_range_probe.py")), str(artifact)],
        env=os.environ.copy(), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "VUID-" not in result.stdout + result.stderr
    assert "SYNC-HAZARD" not in result.stdout + result.stderr
    print(result.stdout)


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_range_plan_keeps_code_shared_and_snapshots_per_task(tmp_path):
    @ti.kernel
    def sequence(out: ti.types.ndarray(ti.i32, 1), limits: ti.types.ndarray(ti.i32, 1)):
        out[65] = 0
        ti.loop_config(block_dim=32)
        for i in range(limits[0]):
            out[i] = i + 1
        limits[0] = limits[0] // 2
        ti.loop_config(block_dim=32)
        for i in range(limits[0]):
            out[i] += 100
        out[65] = 17

    out = ti.ndarray(ti.i32, shape=66)
    limits = ti.ndarray(ti.i32, shape=1)
    baseline, selected = select(sequence, out, limits)
    assert len(selected.device_range_tasks) == 2
    symbols = [ti.graph.Arg(ti.graph.ArgKind.NDARRAY, n, ti.i32, ndim=1) for n in ("out", "limits")]
    program = ti.lang.impl.get_runtime()
    program.set_kernel_executable_lifecycle_telemetry_enabled(True)
    program.debug_kernel_executable_lifecycle_stats(True)
    # select() already compiled the ordinary specialization to obtain its
    # proof. Neither a baseline plan nor its launch-only selection needs new code.
    fixed = graph(sequence, baseline, symbols)
    device = graph(sequence, selected, symbols)
    assert program.debug_kernel_executable_lifecycle_stats()["compiler_invocations"] == 0
    program.set_kernel_executable_lifecycle_telemetry_enabled(False)
    with pytest.raises(ti.TaichiRuntimeError, match="Graph recording"):
        _bind_offload_execution_plan(sequence, selected)(out, limits)

    def run():
        for size in (0, 5, 65, 0):
            out.fill(-1)
            limits.from_numpy(np.array([size], dtype=np.int32))
            device.run(dict(out=out, limits=limits))
            expected = np.full(66, -1, dtype=np.int32)
            expected[:size] = np.arange(1, size + 1)
            expected[:size // 2] += 100
            expected[65] = 17
            np.testing.assert_array_equal(out.to_numpy(), expected)
    run()
    module = ti.aot.Module()
    module.add_execution_plan("fixed", fixed)
    module.add_execution_plan("device", device)
    module.add_execution_plan("again", device)
    for name in ("one", "two"):
        artifact = tmp_path / (name + ".tcm")
        module.archive(str(artifact))
        with zipfile.ZipFile(artifact) as archive:
            metadata = json.loads(archive.read("metadata.json"))
        assert metadata["metadata_version"] == 14
        assert len(metadata["kernels"]) == 1, "Launch-only selection duplicated shader payload"
        assert len(metadata["device_range_dispatch"]) == 1
        assert metadata["device_range_dispatch"][0]["tasks"] == list(selected.device_range_tasks)
        from taichi_forge.aot.conventions.gfxruntime140 import dr, sr
        decoded = sr.from_dr_metadata(dr.from_json_metadata(metadata))
        assert list(decoded.device_range_dispatch) == metadata["device_range_dispatch"]
        consume(artifact)
        run()  # Export must not modify the producer or its selected strategy.
    fixed.close()
    device.close()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_selected_range_primal_and_explicit_adjoint_in_borrowed_domain(tmp_path):
    @ti.kernel
    def square(x: ti.types.ndarray(ti.f32, 1, needs_grad=True),
               y: ti.types.ndarray(ti.f32, 1, needs_grad=True),
               limits: ti.types.ndarray(ti.i32, 1)):
        ti.loop_config(block_dim=32)
        for i in range(limits[0]):
            y[i] = x[i] * x[i]

    x = ti.ndarray(ti.f32, shape=65, needs_grad=True)
    y = ti.ndarray(ti.f32, shape=65, needs_grad=True)
    limits = ti.ndarray(ti.i32, shape=1)
    symbols = [ti.graph.Arg(ti.graph.ArgKind.NDARRAY, n, dt, ndim=1)
               for n, dt in (("x", ti.f32), ("y", ti.f32), ("limits", ti.i32))]
    _, forward = select(square, x, y, limits)
    _, reverse = select(square.grad, x, y, limits)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(_bind_offload_execution_plan(square, forward), *symbols)
    builder.dispatch(_bind_offload_execution_plan(square.grad, reverse), *symbols)
    selected = builder.compile()
    x.from_numpy(np.arange(65, dtype=np.float32) + 1)
    with ti.aot.borrow_runtime() as runtime:
        domain = runtime.execution_domain(instance_capacity=1)
        prepared = selected.prepare(dict(x=x, y=y, limits=limits), execution_domain=domain)
        prepared.initialize()
        for size in (0, 5, 65, 0):
            limits.from_numpy(np.array([size], dtype=np.int32))
            y.fill(-1)
            x.grad.fill(0)
            y.grad.fill(1)
            # Test input writes use the default queue; complete that producer
            # before submitting to the independent execution domain.
            ti.sync()
            prepared.submit().wait()
            expected_y = np.full(65, -1, dtype=np.float32)
            expected_dx = np.zeros(65, dtype=np.float32)
            expected_y[:size] = (np.arange(size) + 1) ** 2
            expected_dx[:size] = 2 * (np.arange(size) + 1)
            np.testing.assert_array_equal(y.to_numpy(), expected_y)
            np.testing.assert_array_equal(x.grad.to_numpy(), expected_dx)
        prepared.close()
        domain.close()
    module = ti.aot.Module()
    module.add_execution_plan("gradient", selected, gradients=dict(x="dx", y="dy"))
    artifact = tmp_path / "gradient.tcm"
    module.archive(str(artifact))
    with zipfile.ZipFile(artifact) as archive:
        metadata = json.loads(archive.read("metadata.json"))
    assert metadata["metadata_version"] == 14
    assert len(metadata["device_range_dispatch"]) == 2
    consume(artifact)
    selected.close()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_range_selection_rejects_unproven_task():
    @ti.kernel
    def constant(out: ti.types.ndarray(ti.i32, 1)):
        for i in range(5):
            out[i] = i

    out = ti.ndarray(ti.i32, shape=5)
    tasks = constant.task_manifest(out)
    plan = _OffloadExecutionPlan.from_task_manifests(tasks)
    index = next(t.task_index for t in tasks if t.task_type == "range_for")
    plan = plan.replace_task(index, range_dispatch="device")
    with pytest.raises(ValueError, match="no device range dispatch proof"):
        graph(constant, plan, [ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1)])
