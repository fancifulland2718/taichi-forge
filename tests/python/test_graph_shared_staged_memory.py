import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge._lib import core as ti_core
from taichi_forge.lang import impl
from taichi_forge.lang.exception import TaichiCompilationError
from taichi_forge.lang._offload_execution_plan import (
    _OffloadExecutionPlan,
    _bind_offload_execution_plan,
)
from taichi_forge.graph import _graph as graph_impl
from tests import test_utils


def _shared_staged_plan(kernel, *probe_args, block_dim=128):
    baseline = _OffloadExecutionPlan.from_task_manifests(
        kernel.task_manifest(*probe_args)
    )
    ranges = tuple(task for task in baseline.tasks if task.task_kind == "range_for")
    assert len(ranges) == 1
    return baseline.replace_task(
        ranges[0].task_index,
        workgroup_size=block_dim,
        memory_strategy="shared_staged_1d",
    )


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_private_graph_shared_staged_recipe_materializes_and_replays_exactly(
    monkeypatch,
):
    count = 1027

    @ti.kernel
    def stencil(
        source: ti.types.ndarray(dtype=ti.f32, ndim=1),
        output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in range(1, count - 1):
            output[i] = source[i - 1] + source[i] * 2.0 + source[i + 1]

    source = ti.ndarray(ti.f32, shape=count)
    output = ti.ndarray(ti.f32, shape=count)
    values = np.arange(count, dtype=np.float32) * 0.25
    source.from_numpy(values)
    plan = _shared_staged_plan(stencil, source, output)
    bound = _bind_offload_execution_plan(stencil, plan)

    manifest = next(
        task
        for task in bound.task_manifest(source, output)
        if task.task_type == "range_for"
    )
    assert manifest.requested_memory_strategy == "shared_staged_1d"
    assert manifest.range_mapping == "shared_tiled_one_to_one"
    assert manifest.selected_block_size == 128
    assert manifest.selected_grid_size == (count - 2 + 127) // 128
    assert manifest.staged_external_arg_index == 0
    assert (manifest.staged_halo_low, manifest.staged_halo_high) == (-1, 1)
    assert manifest.static_shared_bytes == (128 + 2) * 4

    source_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "source", ti.f32, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder._dispatch_shared_staged_1d(bound, source_arg, output_arg)
    graph = builder.compile()
    graph._graph_stats

    alias_checks = 0
    original_alias_check = graph_impl.analyze_storage_alias

    def counted_alias_check(*args, **kwargs):
        nonlocal alias_checks
        alias_checks += 1
        return original_alias_check(*args, **kwargs)

    monkeypatch.setattr(graph_impl, "analyze_storage_alias", counted_alias_check)

    graph.run({"source": source, "output": output})
    ti.sync()
    expected = np.zeros(count, dtype=np.float32)
    expected[1:-1] = values[:-2] + values[1:-1] * 2.0 + values[2:]
    np.testing.assert_allclose(output.to_numpy(), expected, rtol=0, atol=0)

    first = graph._graph_stats[0]
    output.fill(0)
    graph.run({"source": source, "output": output})
    ti.sync()
    second = graph._graph_stats[0]
    assert first["captures"] == 1
    assert second["exact_replays"] == 1
    np.testing.assert_allclose(output.to_numpy(), expected, rtol=0, atol=0)

    for _ in range(16):
        graph.run({"source": source, "output": output})
    ti.sync()
    graph_identity = graph._instance_debug_info
    program = impl.get_runtime().prog
    runtime_before = program._runtime_statistics_snapshot()["memory"]
    host_before = dict(ti_core.get_host_memory_pool_stats())
    device_before = dict(ti_core.get_device_memory_pool_stats())
    for _ in range(10_000):
        graph.run({"source": source, "output": output})
    ti.sync()
    assert alias_checks == 1
    assert graph._instance_debug_info == graph_identity
    assert graph._graph_stats[0]["exact_replays"] >= 10_001
    runtime_after = program._runtime_statistics_snapshot()["memory"]
    for key in (
        "host_requested_live_bytes",
        "host_raw_bytes",
        "device_requested_live_bytes",
        "device_raw_bytes",
        "device_cached_bytes",
    ):
        if runtime_before[key] is not None and runtime_after[key] is not None:
            assert runtime_after[key] <= runtime_before[key]
    for before, after in (
        (host_before, dict(ti_core.get_host_memory_pool_stats())),
        (device_before, dict(ti_core.get_device_memory_pool_stats())),
    ):
        for key in (
            "raw_chunks",
            "requested_live_bytes",
            "raw_bytes",
            "reserved_bytes",
            "committed_bytes",
            "used_bytes",
            "cached_blocks",
            "cached_bytes",
        ):
            if key in before and key in after:
                assert after[key] <= before[key]

    ti.reset()
    with pytest.raises(RuntimeError, match="compiled before ti.reset"):
        graph.run({"source": source, "output": output})


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_private_graph_shared_staged_recipe_is_graph_owned_and_alias_safe():
    count = 257

    @ti.kernel
    def stencil(
        source: ti.types.ndarray(dtype=ti.f32, ndim=1),
        output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in range(1, count - 1):
            output[i] = source[i - 1] + source[i + 1]

    source = ti.ndarray(ti.f32, shape=count)
    output = ti.ndarray(ti.f32, shape=count)
    bound = _bind_offload_execution_plan(
        stencil, _shared_staged_plan(stencil, source, output)
    )
    source_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "source", ti.f32, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1)

    with pytest.raises(RuntimeError, match="Graph-owned"):
        bound(source, output)
    with pytest.raises(TaichiCompilationError, match="Graph-owned memory recipe"):
        ti.graph.GraphBuilder().dispatch(bound, source_arg, output_arg)

    builder = ti.graph.GraphBuilder()
    builder._dispatch_shared_staged_1d(bound, source_arg, output_arg)
    graph = builder.compile()
    with pytest.raises(RuntimeError, match="requires proven disjoint storage"):
        graph.run({"source": source, "output": source})

    short_source = ti.ndarray(ti.f32, shape=count - 1)
    short_output = ti.ndarray(ti.f32, shape=count - 1)
    with pytest.raises(RuntimeError, match="at least 257 scalar elements"):
        graph.run({"source": short_source, "output": short_output})


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_private_graph_shared_staged_recipe_rejects_pointwise_input():
    count = 256

    @ti.kernel
    def pointwise(
        source: ti.types.ndarray(dtype=ti.f32, ndim=1),
        output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in range(count):
            output[i] = source[i] * 2.0

    source = ti.ndarray(ti.f32, shape=count)
    output = ti.ndarray(ti.f32, shape=count)
    plan = _shared_staged_plan(pointwise, source, output)
    with pytest.raises(RuntimeError, match="at least two distinct affine offsets"):
        _bind_offload_execution_plan(pointwise, plan).task_manifest(source, output)
