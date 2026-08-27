from dataclasses import FrozenInstanceError

import pytest

import taichi_forge as ti
from taichi_forge._lib import core as ti_core
from taichi_forge.lang import impl
from tests import test_utils


def _profile_names():
    program = impl.get_runtime().prog
    program.sync_kernel_profiler()
    program.update_kernel_profiler()
    return [record.name for record in program.get_kernel_profiler_records()]


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
)
def test_task_manifest_is_stable_read_only_and_does_not_launch():
    values = ti.field(dtype=ti.i32, shape=257)

    @ti.kernel
    def fill():
        ti.loop_config(block_dim=64)
        for i in values:
            values[i] = i + 1

    first = fill.task_manifest()
    second = fill.task_manifest()

    assert first
    assert first == second
    assert len({task.task_id for task in first}) == len(first)
    assert all(task.task_id.startswith("tf:") for task in first)
    assert len({task.logical_task_id for task in first}) == len(first)
    assert all(task.logical_task_id.startswith("tfl:") for task in first)
    assert all(task.optimization_spec_id == "" for task in first)
    assert all(
        task.backend == ti_core.arch_name(impl.current_cfg().arch) for task in first
    )

    program = impl.get_runtime().prog
    runtime_before = program._runtime_statistics_snapshot()
    host_pool_before = dict(ti_core.get_host_memory_pool_stats())
    device_pool_before = dict(ti_core.get_device_memory_pool_stats())
    for _ in range(1000):
        assert fill.task_manifest() == first
    runtime_after = program._runtime_statistics_snapshot()
    assert runtime_after["submission"] == runtime_before["submission"]
    assert runtime_after["transfer"] == runtime_before["transfer"]
    assert runtime_after["synchronization"] == runtime_before["synchronization"]
    assert runtime_after["memory"] == runtime_before["memory"]
    assert dict(ti_core.get_host_memory_pool_stats()) == host_pool_before
    assert dict(ti_core.get_device_memory_pool_stats()) == device_pool_before
    assert values.to_numpy().sum() == 0

    with pytest.raises(FrozenInstanceError):
        first[0].selected_block_size = 1

    if impl.current_cfg().arch == ti_core.Arch.x64:
        assert all(
            task.range_mapping
            == (
                "cpu_scheduler"
                if task.task_type == "range_for"
                else "not_applicable"
            )
            for task in first
        )
        assert all(task.selected_grid_size is None for task in first)
        assert all(task.selected_block_size is None for task in first)
        assert all(task.actual_grid_size is None for task in first)
        assert all(task.actual_block_size is None for task in first)
        assert all(task.static_shared_bytes == 0 for task in first)
        assert all(task.dynamic_shared_bytes == 0 for task in first)
        assert all(
            task.actual_geometry_kind == "cpu_runtime_scheduler" for task in first
        )
    else:
        parallel = [task for task in first if task.task_type == "range_for"]
        assert parallel
        assert parallel[0].range_mapping == "grid_stride"
        assert all(
            task.range_mapping == "not_applicable"
            for task in first
            if task.task_type != "range_for"
        )
        assert parallel[0].requested_block_size == 64
        assert parallel[0].selected_grid_size > 0
        assert parallel[0].selected_block_size > 0
        assert parallel[0].actual_grid_size == parallel[0].selected_grid_size
        assert parallel[0].actual_block_size == parallel[0].selected_block_size
        assert parallel[0].actual_geometry_kind == "static_direct"


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_task_manifest_identity_survives_runtime_recompilation():
    @ti.kernel
    def increment(values: ti.types.ndarray()):
        for i in values:
            values[i] += 1

    values = ti.ndarray(ti.i32, shape=32)
    first = increment.task_manifest(values)
    first_ids = tuple(task.task_id for task in first)
    first_logical_ids = tuple(task.logical_task_id for task in first)

    ti.reset()
    ti.init(arch=ti.cpu, offline_cache=False)
    values = ti.ndarray(ti.i32, shape=32)
    second = increment.task_manifest(values)
    assert tuple(task.task_id for task in second) == first_ids
    assert tuple(task.logical_task_id for task in second) == first_logical_ids


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_logical_task_identity_separates_multi_offload_ordinals():
    values = ti.field(dtype=ti.i32, shape=32)

    @ti.kernel
    def two_ranges():
        for i in range(32):
            values[i] = i
        for i in range(16):
            values[i] += 1

    manifest = two_ranges.task_manifest()
    ranges = tuple(task for task in manifest if task.task_type == "range_for")
    assert len(ranges) == 2
    assert ranges[0].task_index != ranges[1].task_index
    assert ranges[0].logical_task_id != ranges[1].logical_task_id


@test_utils.test(
    arch=[ti.cuda, ti.vulkan],
    offline_cache=False,
)
def test_task_manifest_reports_static_shared_memory():
    values = ti.field(dtype=ti.i32, shape=64)

    @ti.kernel
    def use_shared():
        ti.loop_config(block_dim=64)
        for i in range(64):
            shared = ti.simt.block.SharedArray((64,), ti.i32)
            shared[i % 64] = values[i]
            ti.simt.block.sync()
            values[i] = shared[(i + 1) % 64]

    manifest = use_shared.task_manifest()
    parallel = [task for task in manifest if task.task_type == "range_for"]
    assert len(parallel) == 1
    assert parallel[0].static_shared_bytes >= 64 * 4
    assert parallel[0].dynamic_shared_bytes == 0


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_static_shared_limit_applies_to_the_complete_task():
    values = ti.field(dtype=ti.i32, shape=64)

    @ti.kernel
    def over_limit():
        ti.loop_config(block_dim=64)
        for i in range(64):
            first = ti.simt.block.SharedArray((8192,), ti.i32)
            second = ti.simt.block.SharedArray((8192,), ti.i32)
            first[i] = values[i]
            second[i] = values[(i + 1) % 64]
            ti.simt.block.sync()
            values[i] = first[(i + 1) % 64] + second[(i + 2) % 64]

    with pytest.raises(RuntimeError, match=r"aggregate bytes.*48 KiB.*IMA"):
        over_limit.task_manifest()


@test_utils.test(
    arch=ti.vulkan,
    offline_cache=False,
    kernel_profiler=False,
    vulkan_dispatch_cache=False,
)
def test_graph_manifest_marks_device_indirect_geometry_as_invocation_specific():
    @ti.kernel
    def consume(out: ti.types.ndarray()):
        ti.loop_config(block_dim=32)
        for i in range(64):
            out[i] += 1

    packet = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "packet", ti.u32, ndim=1)
    out = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch_indirect(
        consume,
        out,
        dispatch_packet=packet,
        label="phase=device-worklist",
    )
    manifest = builder.compile().task_manifest()

    assert manifest
    assert all(task.indirect for task in manifest)
    assert all(task.dispatch_label == "phase=device-worklist" for task in manifest)
    assert all(task.selected_grid_size > 0 for task in manifest)
    assert all(task.selected_block_size > 0 for task in manifest)
    assert all(task.actual_grid_size is None for task in manifest)
    assert all(task.actual_block_size is None for task in manifest)
    assert all(task.actual_geometry_kind == "runtime_indirect" for task in manifest)


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
    kernel_profiler=True,
)
def test_graph_dispatch_labels_map_manifest_and_profiler_events():
    value = ti.field(dtype=ti.i32, shape=())

    @ti.kernel
    def increment():
        value[None] += 1

    builder = ti.graph.GraphBuilder()
    builder.dispatch(increment, label="sweep=3/color=red")
    builder.dispatch(increment, label="sweep=3/color=black")
    graph = builder.compile()

    manifest = graph.task_manifest()
    assert {task.dispatch_index for task in manifest} == {0, 1}
    assert {task.dispatch_label for task in manifest} == {
        "sweep=3/color=red",
        "sweep=3/color=black",
    }
    assert all(task.source_dispatch_count == 1 for task in manifest)
    by_dispatch = {}
    for task in manifest:
        by_dispatch.setdefault(task.dispatch_index, []).append(task.task_id)
    assert by_dispatch[0] == by_dispatch[1]

    module = ti.aot.Module()
    with pytest.raises(RuntimeError, match="JIT-only dispatch labels"):
        module.add_graph("labeled", graph)

    if impl.current_cfg().arch == ti_core.Arch.cuda:
        assert graph._graph_stats[0]["attempts"] == 0
    ti.profiler.clear_kernel_profiler_info()
    graph.run({})
    assert value[None] == 2
    if impl.current_cfg().arch == ti_core.Arch.cuda:
        stats = graph._graph_stats[0]
        assert stats["attempts"] == 0
        assert stats["captures"] == 0
        assert stats["exact_replays"] == 0
    names = _profile_names()
    assert any(
        "tf.task=" in name and "label=sweep=3/color=red" in name for name in names
    )
    assert any(
        "tf.task=" in name and "label=sweep=3/color=black" in name for name in names
    )


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
    kernel_profiler=True,
)
def test_scoped_dispatch_label_is_nested_and_default_path_stays_plain():
    value = ti.field(dtype=ti.i32, shape=())

    @ti.kernel
    def increment():
        value[None] += 1

    increment.task_manifest()
    ti.profiler.clear_kernel_profiler_info()
    with ti.profiler.dispatch_label("phase=outer"):
        with ti.profiler.dispatch_label("phase=inner"):
            increment()
        increment()
    increment()
    ti.sync()

    names = _profile_names()
    assert any(
        "tf.task=" in name and "label=phase=inner" in name for name in names
    ), names
    assert any(
        "tf.task=" in name and "label=phase=outer" in name for name in names
    ), names
    assert any("tf.task=" not in name and "increment" in name for name in names), names


def test_dispatch_label_validation():
    with pytest.raises(RuntimeError, match="at most 128 UTF-8 bytes"):
        with ti.profiler.dispatch_label("x" * 129):
            pass
    with pytest.raises(RuntimeError, match="line breaks"):
        with ti.profiler.dispatch_label("phase=one\nphase=two"):
            pass
    with pytest.raises(TypeError, match="expects a string"):
        with ti.profiler.dispatch_label(3):
            pass


@test_utils.test(
    arch=[ti.cuda, ti.vulkan],
    offline_cache=False,
    kernel_profiler=True,
)
def test_structured_graph_dispatch_labels_reach_profiler_events():
    @ti.kernel
    def condition(counter: ti.types.ndarray(), predicate: ti.types.ndarray()):
        predicate[None] = counter[None] < 3

    @ti.kernel
    def body(counter: ti.types.ndarray()):
        counter[None] += 1

    counter_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "counter", ti.i32, ndim=0)
    predicate_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "predicate", ti.i32, ndim=0
    )
    builder = ti.graph.GraphBuilder()
    condition_seq = builder.create_sequential()
    condition_seq.dispatch(
        condition, counter_arg, predicate_arg, label="role=condition"
    )
    body_seq = builder.create_sequential()
    body_seq.dispatch(body, counter_arg, label="role=body")
    builder.while_loop(
        condition_seq,
        body_seq,
        predicate=predicate_arg,
        carried_state=(counter_arg,),
        counter=counter_arg,
        max_iterations=8,
    )
    graph = builder.compile()
    counter = ti.ndarray(ti.i32, shape=())
    predicate = ti.ndarray(ti.i32, shape=())
    counter.fill(0)
    predicate.fill(0)

    ti.profiler.clear_kernel_profiler_info()
    graph.run({"counter": counter, "predicate": predicate})
    ti.sync()
    names = _profile_names()
    assert any("label=role=condition" in name for name in names), names
    assert any("label=role=body" in name for name in names), names
