import numpy as np

import taichi_forge as ti
from taichi_forge._lib import core as ti_core
from taichi_forge.lang import impl
from taichi_forge.lang._kernel_optimization import (
    _BackendCodegenOptions,
    _KernelOptimizationSpec,
    _LaunchOptions,
    _bind_kernel_optimization_spec,
)
from tests import test_utils


_TELEMETRY_KEYS = (
    "cuda_grid_residency_resolution_calls",
    "cuda_grid_residency_resolution_failures",
    "cuda_grid_residency_last_requested_waves",
    "cuda_grid_residency_last_baseline_grid",
    "cuda_grid_residency_last_resolved_grid",
    "cuda_grid_residency_last_active_blocks_per_multiprocessor",
    "cuda_grid_residency_last_multiprocessor_count",
)


def _telemetry():
    return {key: int(ti_core.query_int64(key)) for key in _TELEMETRY_KEYS}


def _spec(waves):
    return _KernelOptimizationSpec(
        backend=_BackendCodegenOptions(workgroup_size=256),
        launch=_LaunchOptions(block_mode="require", grid_residency_waves=waves),
    )


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_grid_residency_is_cached_per_cufunction_and_preserves_results():
    count = 1 << 20
    values = ti.ndarray(ti.i32, shape=count)

    @ti.kernel
    def fill(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(count):
            out[i] = i * 3 + 7

    bindings = tuple(
        _bind_kernel_optimization_spec(fill, _spec(waves)) for waves in (1, 2, 4)
    )
    reports = tuple(binding.report(values) for binding in bindings)
    assert len({binding._fast_key for binding in bindings}) == 1
    assert len({id(binding._fast_kernel_cpp) for binding in bindings}) == 1
    assert (
        len(
            {
                tuple(task.optimization_spec_id for task in report.tasks)
                for report in reports
            }
        )
        == 1
    )

    baseline = fill.with_launch_policy(ti.TaskLaunchPolicy.block(256, mode="require"))
    before = _telemetry()
    baseline(values)
    ti.sync()
    assert (
        _telemetry()["cuda_grid_residency_resolution_calls"]
        == before["cuda_grid_residency_resolution_calls"]
    )
    program = impl.get_runtime().prog

    for waves, binding in zip((1, 2, 4), bindings):
        for _ in range(4):
            binding(values)
        ti.sync()
        evidence = _telemetry()
        assert (
            evidence["cuda_grid_residency_resolution_failures"]
            == before["cuda_grid_residency_resolution_failures"]
        )
        assert evidence["cuda_grid_residency_last_requested_waves"] == waves
        assert evidence["cuda_grid_residency_last_resolved_grid"] == min(
            evidence["cuda_grid_residency_last_baseline_grid"],
            evidence["cuda_grid_residency_last_active_blocks_per_multiprocessor"]
            * evidence["cuda_grid_residency_last_multiprocessor_count"]
            * waves,
        )

    after = _telemetry()
    assert (
        after["cuda_grid_residency_resolution_calls"]
        - before["cuda_grid_residency_resolution_calls"]
        == 3
    )
    np.testing.assert_array_equal(
        values.to_numpy(), np.arange(count, dtype=np.int32) * 3 + 7
    )

    runtime_memory = program._runtime_statistics_snapshot()["memory"]
    host_memory = dict(ti_core.get_host_memory_pool_stats())
    device_memory = dict(ti_core.get_device_memory_pool_stats())
    for _ in range(64):
        bindings[0](values)
    ti.sync()
    assert (
        _telemetry()["cuda_grid_residency_resolution_calls"]
        == after["cuda_grid_residency_resolution_calls"]
    )
    assert program._runtime_statistics_snapshot()["memory"] == runtime_memory
    assert dict(ti_core.get_host_memory_pool_stats()) == host_memory
    assert dict(ti_core.get_device_memory_pool_stats()) == device_memory
