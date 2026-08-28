import numpy as np
import pytest

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
    "cuda_range_coarsening_resolution_calls",
    "cuda_range_coarsening_resolution_failures",
    "cuda_range_coarsening_last_work_per_thread_target",
    "cuda_range_coarsening_last_baseline_grid",
    "cuda_range_coarsening_last_resolved_grid",
    "cuda_grid_residency_resolution_calls",
    "cuda_grid_residency_last_resolved_grid",
)


def _telemetry():
    return {key: int(ti_core.query_int64(key)) for key in _TELEMETRY_KEYS}


def _spec(target, *, waves=None):
    return _KernelOptimizationSpec(
        backend=_BackendCodegenOptions(workgroup_size=256),
        launch=_LaunchOptions(
            block_mode="require",
            grid_residency_waves=waves,
            range_work_per_thread_target=target,
        ),
    )


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_range_coarsening_reuses_artifact_and_caches_launch_plans():
    count = 1 << 20
    values = ti.ndarray(ti.i32, shape=count)

    @ti.kernel
    def fill(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(count):
            out[i] = i * 5 + 11

    bindings = tuple(
        _bind_kernel_optimization_spec(fill, _spec(target))
        for target in (1, 2, 4, 8)
    )
    reports = tuple(binding.report(values) for binding in bindings)
    assert len({binding._fast_key for binding in bindings}) == 1
    assert len({id(binding._fast_kernel_cpp) for binding in bindings}) == 1
    assert all(
        next(task for task in report.tasks if task.task_type == "range_for")
        .constant_range_size
        == count
        for report in reports
    )

    before = _telemetry()
    for target, binding in zip((1, 2, 4, 8), bindings):
        binding(values)
        ti.sync()
        evidence = _telemetry()
        if target == 1:
            assert (
                evidence["cuda_range_coarsening_resolution_calls"]
                == before["cuda_range_coarsening_resolution_calls"]
            )
            continue
        assert evidence["cuda_range_coarsening_resolution_failures"] == before[
            "cuda_range_coarsening_resolution_failures"
        ]
        assert (
            evidence["cuda_range_coarsening_last_work_per_thread_target"]
            == target
        )
        expected = min(
            evidence["cuda_range_coarsening_last_baseline_grid"],
            (count + 256 * target - 1) // (256 * target),
        )
        assert evidence["cuda_range_coarsening_last_resolved_grid"] == expected

    after = _telemetry()
    assert (
        after["cuda_range_coarsening_resolution_calls"]
        - before["cuda_range_coarsening_resolution_calls"]
        == 3
    )
    np.testing.assert_array_equal(
        values.to_numpy(), np.arange(count, dtype=np.int32) * 5 + 11
    )

    program = impl.get_runtime().prog
    runtime_memory = program._runtime_statistics_snapshot()["memory"]
    host_memory = dict(ti_core.get_host_memory_pool_stats())
    device_memory = dict(ti_core.get_device_memory_pool_stats())
    for _ in range(64):
        bindings[-1](values)
    ti.sync()
    assert _telemetry() == after
    assert program._runtime_statistics_snapshot()["memory"] == runtime_memory
    assert dict(ti_core.get_host_memory_pool_stats()) == host_memory
    assert dict(ti_core.get_device_memory_pool_stats()) == device_memory


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_range_coarsening_composes_with_residency():
    count = 1 << 20
    values = ti.ndarray(ti.i32, shape=count)

    @ti.kernel
    def reduce(inp: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(count):
            inp[i] += 1

    bound = _bind_kernel_optimization_spec(reduce, _spec(4, waves=1))
    before = _telemetry()
    bound(values)
    ti.sync()
    after = _telemetry()
    assert after["cuda_range_coarsening_resolution_calls"] == (
        before["cuda_range_coarsening_resolution_calls"] + 1
    )
    assert after["cuda_grid_residency_resolution_calls"] == (
        before["cuda_grid_residency_resolution_calls"] + 1
    )
    assert after["cuda_grid_residency_last_resolved_grid"] <= after[
        "cuda_range_coarsening_last_resolved_grid"
    ]
    np.testing.assert_array_equal(values.to_numpy(), np.ones(count, dtype=np.int32))


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_range_coarsening_rejects_dynamic_range():
    values = ti.ndarray(ti.i32, shape=32)

    @ti.kernel
    def fill(count: ti.i32, out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(count):
            out[i] = i

    bound = _bind_kernel_optimization_spec(fill, _spec(2))
    with pytest.raises(RuntimeError, match="requires a constant range"):
        bound(32, values)
