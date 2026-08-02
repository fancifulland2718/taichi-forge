from dataclasses import FrozenInstanceError
import threading

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge._lib import core as ti_core
from taichi_forge.lang import impl
from tests import test_utils


def _range_task(tasks):
    selected = [task for task in tasks if task.task_type == "range_for"]
    assert len(selected) == 1
    return selected[0]


def test_task_launch_policy_value_validation():
    assert ti.TaskLaunchPolicy() == ti.TaskLaunchPolicy.auto()
    assert ti.TaskLaunchPolicy.block(256, mode="require").block_dim == 256
    with pytest.raises(ValueError, match="does not accept block_dim"):
        ti.TaskLaunchPolicy(block_dim=64)
    with pytest.raises(ValueError, match="mode"):
        ti.TaskLaunchPolicy.block(64, mode="force")
    with pytest.raises(TypeError, match="integer"):
        ti.TaskLaunchPolicy.block(True)
    with pytest.raises(ValueError, match=r"\[1, 1024\]"):
        ti.TaskLaunchPolicy.block(2048)
    with pytest.raises(ValueError, match="power of two or a multiple of 32"):
        ti.TaskLaunchPolicy.block(48)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_task_launch_resource_report_is_read_only_and_no_submit():
    values = ti.ndarray(ti.i32, shape=257)

    @ti.kernel
    def fill(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(257):
            out[i] = i + 3

    policy = ti.TaskLaunchPolicy.block(256)
    launch = fill.with_launch_policy(policy)
    program = impl.get_runtime().prog
    report = launch.report(values)
    before = program._runtime_statistics_snapshot()
    assert launch.report(values) == report
    after = program._runtime_statistics_snapshot()
    assert after["submission"] == before["submission"]
    assert after["transfer"] == before["transfer"]
    assert after["memory"] == before["memory"]
    assert len(report.resources) == len(report.tasks)

    task = _range_task(report.tasks)
    resource = next(item for item in report.resources if item.task_id == task.task_id)
    assert resource.observation_kind == "compile_time_no_submit"
    assert resource.selected_block_size == task.selected_block_size
    assert resource.static_shared_bytes == task.static_shared_bytes
    assert resource.dynamic_shared_bytes == task.dynamic_shared_bytes
    assert resource.registers_per_thread is None
    assert resource.local_memory_bytes_per_thread is None
    with pytest.raises(FrozenInstanceError):
        resource.static_shared_bytes = 1

    if impl.current_cfg().arch == ti_core.Arch.x64:
        assert report.status == "fallback_auto"
        assert resource.max_threads_per_block is None
        assert resource.representative_legal_block_sizes == ()
        assert resource.rejected_candidates[0].block_dim == 256
        assert "worker" in resource.rejected_candidates[0].reason
    else:
        assert report.status == "applied"
        assert 256 in resource.representative_legal_block_sizes
        assert resource.rejected_candidates == ()
        if impl.current_cfg().arch == ti_core.Arch.cuda:
            assert resource.max_threads_per_block == impl.current_cfg().max_block_dim
            assert "materialized native function" in resource.registers_reason
        else:
            assert resource.max_threads_per_block is None
            assert "not exposed" in resource.max_threads_reason
            assert "driver owned" in resource.registers_reason


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_task_launch_policy_cross_backend_correctness_and_report():
    count = 4099
    values = ti.ndarray(ti.i32, shape=count)

    @ti.kernel
    def fill(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(count):
            out[i] = i * 3 + 7

    auto_manifest = fill.task_manifest(values)
    hint = fill.with_launch_policy(ti.TaskLaunchPolicy.block(256))
    report = hint.report(values)
    assert report.policy == hint.policy
    assert report.backend == ti_core.arch_name(impl.current_cfg().arch)

    program = impl.get_runtime().prog
    before_report = program._runtime_statistics_snapshot()
    assert hint.report(values) == report
    after_report = program._runtime_statistics_snapshot()
    assert after_report["submission"] == before_report["submission"]
    assert after_report["transfer"] == before_report["transfer"]

    hint(values)
    ti.sync()
    np.testing.assert_array_equal(
        values.to_numpy(), np.arange(count, dtype=np.int32) * 3 + 7
    )

    if impl.current_cfg().arch == ti_core.Arch.x64:
        assert report.status == "fallback_auto"
        assert report.tasks == auto_manifest
        required = fill.with_launch_policy(
            ti.TaskLaunchPolicy.block(256, mode="require")
        )
        before = program._runtime_statistics_snapshot()["submission"]
        with pytest.raises(RuntimeError, match="unavailable on CPU"):
            required(values)
        assert program._runtime_statistics_snapshot()["submission"] == before
    else:
        assert report.status == "applied"
        task = _range_task(report.tasks)
        assert task.requested_block_size == 256
        assert task.selected_block_size == 256
        assert task.actual_block_size == 256
        assert task.task_id != _range_task(auto_manifest).task_id

        required = fill.with_launch_policy(
            ti.TaskLaunchPolicy.block(256, mode="require")
        )
        required(values)
        ti.sync()
        assert required.report(values).status == "applied"



@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_task_launch_policy_supports_safe_dynamic_range_preamble():
    capacity = 1024
    values = ti.ndarray(ti.i32, shape=capacity)

    @ti.kernel
    def fill(count: ti.i32, out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(count):
            out[i] = i + 11

    launch = fill.with_launch_policy(ti.TaskLaunchPolicy.block(64, mode="require"))
    launch(777, values)
    ti.sync()
    actual = values.to_numpy()
    np.testing.assert_array_equal(actual[:777], np.arange(777, dtype=np.int32) + 11)
    assert np.all(actual[777:] == 0)
    report = launch.report(777, values)
    assert _range_task(report.tasks).actual_block_size == 64
    assert all(task.task_type in ("serial", "range_for") for task in report.tasks)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_task_launch_policy_binds_data_oriented_methods():
    count = 257
    values = ti.ndarray(ti.i32, shape=count)

    @ti.data_oriented
    class Updater:
        def __init__(self, bias):
            self.bias = bias

        @ti.kernel
        def run(self, out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
            for i in range(count):
                out[i] = i + self.bias

    updater = Updater(9)
    mode = "hint" if impl.current_cfg().arch == ti_core.Arch.x64 else "require"
    launch = updater.run.with_launch_policy(ti.TaskLaunchPolicy.block(96, mode=mode))
    report = launch.report(values)
    assert report.status == (
        "fallback_auto" if impl.current_cfg().arch == ti_core.Arch.x64 else "applied"
    )
    launch(values)
    ti.sync()
    np.testing.assert_array_equal(
        values.to_numpy(), np.arange(count, dtype=np.int32) + 9
    )


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_task_launch_policy_preserves_source_block_and_rejects_unsafe_changes():
    values = ti.ndarray(ti.i32, shape=256)

    @ti.kernel
    def explicit(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        ti.loop_config(block_dim=64)
        for i in range(256):
            out[i] = i

    hinted = explicit.with_launch_policy(ti.TaskLaunchPolicy.block(256))
    report = hinted.report(values)
    assert report.status == "hint_not_applied"
    range_resource = next(
        resource for resource in report.resources if resource.task_type == "range_for"
    )
    assert range_resource.rejected_candidates[0].block_dim == 256
    assert _range_task(report.tasks).actual_block_size == 64
    hinted(values)
    ti.sync()
    np.testing.assert_array_equal(values.to_numpy(), np.arange(256, dtype=np.int32))

    required = explicit.with_launch_policy(
        ti.TaskLaunchPolicy.block(256, mode="require")
    )
    program = impl.get_runtime().prog
    before = program._runtime_statistics_snapshot()["submission"]
    with pytest.raises(RuntimeError, match="conflicts with.*loop_config"):
        required(values)
    assert program._runtime_statistics_snapshot()["submission"] == before

    @ti.kernel
    def shared(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(256):
            scratch = ti.simt.block.SharedArray((256,), ti.i32)
            scratch[i % 256] = i
            ti.simt.block.sync()
            out[i] = scratch[i % 256]

    before = program._runtime_statistics_snapshot()["submission"]
    with pytest.raises(RuntimeError, match="SharedArray|block-sensitive"):
        shared.with_launch_policy(ti.TaskLaunchPolicy.block(256)).report(values)
    assert program._runtime_statistics_snapshot()["submission"] == before

    @ti.kernel
    def explicit_shared(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        ti.loop_config(block_dim=64)
        for i in range(256):
            scratch = ti.simt.block.SharedArray((64,), ti.i32)
            scratch[i % 64] = i + 5
            ti.simt.block.sync()
            out[i] = scratch[i % 64]

    source_owned = explicit_shared.with_launch_policy(
        ti.TaskLaunchPolicy.block(64, mode="require")
    )
    assert _range_task(source_owned.report(values).tasks).actual_block_size == 64
    source_owned(values)
    ti.sync()
    np.testing.assert_array_equal(values.to_numpy(), np.arange(256, dtype=np.int32) + 5)

    if impl.current_cfg().arch == ti_core.Arch.cuda:

        @ti.kernel
        def block_intrinsic(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
            for i in range(256):
                out[i] = ti.global_thread_idx()

    else:

        @ti.kernel
        def block_intrinsic(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
            for i in range(256):
                out[i] = ti.simt.block.thread_idx()

    before = program._runtime_statistics_snapshot()["submission"]
    with pytest.raises(RuntimeError, match="block-sensitive intrinsic"):
        block_intrinsic.with_launch_policy(
            ti.TaskLaunchPolicy.block(256)
        ).report(values)
    assert program._runtime_statistics_snapshot()["submission"] == before

    @ti.kernel
    def two_ranges(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(256):
            out[i] = i
        for i in range(256):
            out[i] += 1

    before = program._runtime_statistics_snapshot()["submission"]
    with pytest.raises(RuntimeError, match="exactly one top-level"):
        two_ranges.with_launch_policy(ti.TaskLaunchPolicy.block(64)).report(values)
    assert program._runtime_statistics_snapshot()["submission"] == before


@pytest.mark.run_in_serial
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_task_launch_policy_concurrency_reset_and_resource_stability():
    arch = impl.current_cfg().arch
    count = 8192
    values = ti.ndarray(ti.i32, shape=count)

    @ti.kernel
    def increment(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(count):
            out[i] += 1

    policy = ti.TaskLaunchPolicy.block(
        128,
        mode="hint" if impl.current_cfg().arch == ti_core.Arch.x64 else "require",
    )
    launch = increment.with_launch_policy(policy)
    if arch in (ti_core.Arch.cuda, ti_core.Arch.vulkan):
        cold_errors = []

        def cold_worker():
            try:
                launch(values)
            except Exception as exc:  # pragma: no cover - asserted below
                cold_errors.append(exc)

        cold_thread = threading.Thread(target=cold_worker)
        cold_thread.start()
        cold_thread.join()
        assert len(cold_errors) == 1
        assert "main thread" in str(cold_errors[0])

    # Compilation remains a main-thread operation. Warm execution is the
    # concurrency contract and must not mutate the policy specialization.
    launch.report(values)
    barrier = threading.Barrier(4)
    errors = []

    def worker():
        try:
            barrier.wait()
            for _ in range(8):
                launch(values)
        except Exception as exc:  # pragma: no cover - reported by assertion
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    ti.sync()
    assert not errors
    assert np.all(values.to_numpy() == 32)

    for _ in range(8):
        launch(values)
    ti.sync()
    program = impl.get_runtime().prog
    runtime_before = program._runtime_statistics_snapshot()["memory"]
    host_before = dict(ti_core.get_host_memory_pool_stats())
    device_before = dict(ti_core.get_device_memory_pool_stats())
    report = launch.report(values)
    for _ in range(500):
        assert launch.report(values) == report
    for _ in range(64):
        launch(values)
    ti.sync()
    assert program._runtime_statistics_snapshot()["memory"] == runtime_before
    assert dict(ti_core.get_host_memory_pool_stats()) == host_before
    assert dict(ti_core.get_device_memory_pool_stats()) == device_before

    ti.reset()
    ti.init(arch=arch, offline_cache=False)
    values = ti.ndarray(ti.i32, shape=count)
    launch(values)
    ti.sync()
    assert np.all(values.to_numpy() == 1)
