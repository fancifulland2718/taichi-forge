import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge._lib import core as ti_core
from taichi_forge.lang import impl
from taichi_forge.lang._offload_execution_plan import (
    _OffloadExecutionPlan,
    _bind_offload_execution_plan,
)
from tests import test_utils


@test_utils.test(arch=ti.cuda, offline_cache=False, kernel_profiler=True)
def test_task_indexed_plan_materializes_two_distinct_range_tasks():
    count = 1 << 20
    values = ti.ndarray(ti.i32, shape=count)

    @ti.kernel
    def two_stage(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(count):
            out[i] = i * 2
        for i in range(count):
            out[i] = out[i] * 3 + 1

    baseline = _OffloadExecutionPlan.from_task_manifests(two_stage.task_manifest(values))
    range_indices = tuple(task.task_index for task in baseline.tasks if task.task_kind == "range_for")
    assert len(range_indices) == 2
    plan = baseline.replace_task(
        range_indices[0],
        workgroup_size=64,
        range_work_per_thread_target=2,
    ).replace_task(
        range_indices[1],
        workgroup_size=256,
        range_work_per_thread_target=4,
    )
    bound = _bind_offload_execution_plan(two_stage, plan)
    report = bound.report(values)
    ranges = tuple(task for task in report.tasks if task.task_type == "range_for")

    assert tuple(task.selected_block_size for task in ranges) == (64, 256)
    assert tuple(task.requested_range_work_per_thread_target for task in ranges) == (2, 4)
    assert all(task.optimization_spec_id == plan.compilation_identity for task in report.tasks)

    before = int(ti_core.query_int64("cuda_range_coarsening_resolution_calls"))
    ti.profiler.clear_kernel_profiler_info()
    bound(values)
    ti.sync()
    after = int(ti_core.query_int64("cuda_range_coarsening_resolution_calls"))
    assert after == before + 1
    program = impl.get_runtime().prog
    program.sync_kernel_profiler()
    program.update_kernel_profiler()
    launches = tuple(
        record
        for record in program.get_kernel_profiler_records()
        if "two_stage" in record.name and "range_for" in record.name
    )
    assert tuple(record.block_size for record in launches) == (64, 256)
    assert tuple(record.grid_size for record in launches) == (
        min(ranges[0].selected_grid_size, count // (64 * 2)),
        min(ranges[1].selected_grid_size, count // (256 * 4)),
    )
    np.testing.assert_array_equal(values.to_numpy(), np.arange(count, dtype=np.int32) * 6 + 1)

    for _ in range(16):
        bound(values)
    ti.sync()
    assert int(ti_core.query_int64("cuda_range_coarsening_resolution_calls")) == after


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_task_indexed_plan_materializes_tls_only_for_selected_reduction_task():
    count = 1 << 16
    values = ti.ndarray(ti.i32, shape=count)
    total = ti.field(ti.i32, shape=())

    @ti.kernel
    def fill_then_reduce(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(count):
            out[i] = i % 17
        for i in range(count):
            total[None] += out[i]

    baseline = _OffloadExecutionPlan.from_task_manifests(fill_then_reduce.task_manifest(values))
    ranges = tuple(task for task in baseline.tasks if task.task_kind == "range_for")
    assert len(ranges) == 2
    plan = baseline.replace_task(
        ranges[0].task_index,
        workgroup_size=128,
        thread_local="off",
    ).replace_task(
        ranges[1].task_index,
        workgroup_size=256,
        thread_local="on",
    )
    bound = _bind_offload_execution_plan(fill_then_reduce, plan)
    materialized = tuple(task for task in bound.task_manifest(values) if task.task_type == "range_for")

    assert tuple(task.requested_thread_local_mode for task in materialized) == (
        "off",
        "on",
    )
    assert materialized[0].thread_local_bytes == 0
    assert materialized[1].thread_local_bytes > 0
    total[None] = 0
    bound(values)
    ti.sync()
    assert total[None] == int(np.sum(np.arange(count, dtype=np.int64) % 17))


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_task_indexed_plan_fails_closed_for_missing_stale_and_source_owned_specs():
    values = ti.ndarray(ti.i32, shape=1 << 10)

    @ti.kernel
    def two_ranges(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in out:
            out[i] = i
        for i in out:
            out[i] += 1

    baseline = _OffloadExecutionPlan.from_task_manifests(two_ranges.task_manifest(values))
    missing = _OffloadExecutionPlan(baseline.semantic_kernel_identity, baseline.tasks[:-1])
    with pytest.raises(RuntimeError, match="topology mismatch"):
        _bind_offload_execution_plan(two_ranges, missing).report(values)

    @ti.kernel
    def different_kernel(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in out:
            out[i] = i * 3
        for i in out:
            out[i] += 7

    with pytest.raises(ValueError, match="topology does not match"):
        _bind_offload_execution_plan(different_kernel, baseline).report(values)

    @ti.kernel
    def source_owned(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        ti.loop_config(block_dim=128)
        for i in out:
            out[i] = i

    source_plan = _OffloadExecutionPlan.from_task_manifests(source_owned.task_manifest(values))
    source_range = next(task for task in source_plan.tasks if task.task_kind == "range_for")
    source_plan = source_plan.replace_task(source_range.task_index, workgroup_size=64)
    with pytest.raises(RuntimeError, match="source-owned block_dim"):
        _bind_offload_execution_plan(source_owned, source_plan).report(values)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_task_indexed_plan_survives_cuda_graph_capture_and_exact_replay():
    count = 1 << 16

    @ti.kernel
    def two_stage(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(count):
            out[i] = i + 2
        for i in range(count):
            out[i] *= 5

    probe = ti.ndarray(ti.i32, shape=count)
    baseline = _OffloadExecutionPlan.from_task_manifests(two_stage.task_manifest(probe))
    ranges = tuple(task for task in baseline.tasks if task.task_kind == "range_for")
    plan = baseline.replace_task(
        ranges[0].task_index,
        workgroup_size=64,
        range_work_per_thread_target=2,
    ).replace_task(
        ranges[1].task_index,
        workgroup_size=256,
        range_work_per_thread_target=4,
    )
    bound = _bind_offload_execution_plan(two_stage, plan)
    symbolic = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1)
    builder = ti.graph.GraphBuilder()
    before = int(ti_core.query_int64("cuda_range_coarsening_resolution_calls"))
    builder.dispatch(bound, symbolic)
    graph = builder.compile()
    output = ti.ndarray(ti.i32, shape=count)
    graph._graph_stats

    graph.run({"out": output})
    ti.sync()
    first = graph._graph_stats[0]
    resolved = int(ti_core.query_int64("cuda_range_coarsening_resolution_calls"))
    graph.run({"out": output})
    ti.sync()
    second = graph._graph_stats[0]

    assert resolved == before + 1
    assert int(ti_core.query_int64("cuda_range_coarsening_resolution_calls")) == resolved
    assert first["captures"] == 1
    assert second["exact_replays"] == 1
    assert second["last_path"] == "cuda_exact_replay"
    np.testing.assert_array_equal(output.to_numpy(), (np.arange(count, dtype=np.int32) + 2) * 5)
    manifests = tuple(task for task in graph.task_manifest() if task.task_type == "range_for")
    assert tuple(task.selected_block_size for task in manifests) == (64, 256)
    assert tuple(task.requested_range_work_per_thread_target for task in manifests) == (2, 4)
    ti.reset()
    with pytest.raises(RuntimeError, match="before ti.reset|runtime.*invalid|retired"):
        graph.run({"out": output})
