import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge._lib import core as ti_core
from taichi_forge.lang import impl
from tests import test_utils


@ti.kernel
def _append_range(
    values: ti.types.ndarray(dtype=ti.i32, ndim=1),
    extent_state: ti.types.ndarray(dtype=ti.i32, ndim=1),
    generated: ti.types.ndarray(dtype=ti.i32, ndim=0),
    overflow: ti.types.ndarray(dtype=ti.i32, ndim=0),
    capacity: ti.i32,
    count: ti.i32,
):
    for i in range(count):
        ti.algorithms.device_worklist_append(
            values,
            extent_state,
            generated,
            overflow,
            capacity,
            i * 3 + 1,
        )


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_device_worklist_atomic_append_clamps_and_reports_overflow():
    capacity = 17
    worklist = ti.algorithms.DeviceWorklist(capacity, ti.i32)
    worklist.prepare_next()
    _append_range(*worklist.append_arguments(), capacity + 6)
    worklist.commit_next()

    snapshot = worklist.snapshot()
    assert snapshot.extent.count == capacity
    assert snapshot.extent.overflow
    assert snapshot.statistics.generated == capacity + 6
    assert snapshot.statistics.accepted == capacity
    assert snapshot.statistics.rejected == 6
    assert snapshot.statistics.winners == capacity
    assert snapshot.statistics.overflow
    np.testing.assert_array_equal(
        np.sort(snapshot.values),
        np.arange(capacity, dtype=np.int32) * 3 + 1,
    )
    report = worklist.execution_report()
    assert report.route == "not_attached"
    assert report.useful_count == capacity
    assert report.executed_count is None
    assert report.generated == capacity + 6
    assert report.overflow


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_device_worklist_atomic_append_count_boundaries():
    capacity = 9
    worklist = ti.algorithms.DeviceWorklist(capacity, ti.i32)
    for requested in (0, 1, capacity - 1, capacity, capacity + 1):
        worklist.prepare_next()
        _append_range(*worklist.append_arguments(), requested)
        worklist.commit_next()
        snapshot = worklist.snapshot()
        expected = min(requested, capacity)
        assert snapshot.extent.count == expected
        assert snapshot.extent.overflow == (requested > capacity)
        assert snapshot.statistics.generated == requested
        assert snapshot.statistics.accepted == expected
        assert snapshot.statistics.rejected == requested - expected
        assert snapshot.statistics.overflow == (requested > capacity)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_device_worklist_atomic_append_rejects_capacity_mismatch():
    capacity = 11
    worklist = ti.algorithms.DeviceWorklist(capacity, ti.i32)
    worklist.prepare_next()
    arguments = list(worklist.append_arguments())
    arguments[-1] = capacity + 1
    _append_range(*arguments, 5)
    worklist.commit_next()

    snapshot = worklist.snapshot()
    assert snapshot.extent.count == 0
    assert snapshot.extent.overflow
    assert snapshot.statistics.generated == 0
    assert snapshot.statistics.accepted == 0
    assert snapshot.statistics.rejected == 0
    assert snapshot.statistics.overflow


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_device_worklist_stable_selection_stays_device_resident():
    capacity = 64
    count = 37
    values_host = np.arange(capacity, dtype=np.int32) * 5 + 2
    flags_host = ((np.arange(capacity) % 4) != 1).astype(np.int32)
    flags = ti.ndarray(ti.i32, shape=capacity)
    flags.from_numpy(flags_host)
    worklist = ti.algorithms.DeviceWorklist(capacity, ti.i32)
    worklist.values.from_numpy(values_host)
    worklist.extent.set(count)

    program = impl.get_runtime().prog
    before = program._runtime_statistics_snapshot()
    worklist.select(flags)
    after_enqueue = program._runtime_statistics_snapshot()
    assert after_enqueue["transfer"] == before["transfer"]
    for counter in ("program_syncs", "completion_waits"):
        assert (
            after_enqueue["synchronization"][counter]
            == before["synchronization"][counter]
        )

    expected = values_host[:count][flags_host[:count] != 0]
    snapshot = worklist.snapshot()
    np.testing.assert_array_equal(snapshot.values, expected)
    assert snapshot.statistics.generated == count
    assert snapshot.statistics.accepted == expected.size
    assert snapshot.statistics.rejected == count - expected.size
    assert snapshot.statistics.conflicts == 0
    assert snapshot.statistics.winners == expected.size
    assert not snapshot.statistics.overflow


@pytest.mark.parametrize(
    "policy,priorities,expected_values,expected_priorities",
    [
        (
            "first",
            None,
            np.array([11, 20, 35], dtype=np.int32),
            np.array([0, 0, 0], dtype=np.int32),
        ),
        (
            "min_priority",
            np.array([5, 2, 1, 2, 1, 0], dtype=np.int32),
            np.array([11, 22, 35], dtype=np.int32),
            np.array([2, 1, 0], dtype=np.int32),
        ),
        (
            "max_priority",
            np.array([5, 2, 1, 2, 1, 0], dtype=np.int32),
            np.array([11, 20, 35], dtype=np.int32),
            np.array([2, 5, 0], dtype=np.int32),
        ),
    ],
)
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_device_worklist_deterministic_keyed_claim(
    policy, priorities, expected_values, expected_priorities
):
    capacity = 32
    count = 6
    keys_host = np.array([2, 1, 2, 1, 2, 3], dtype=np.int32)
    values_host = np.array([20, 11, 22, 13, 24, 35], dtype=np.int32)
    keys = ti.ndarray(ti.i32, shape=capacity)
    keys.from_numpy(np.pad(keys_host, (0, capacity - count)))
    priority_array = None
    if priorities is not None:
        priority_array = ti.ndarray(ti.i32, shape=capacity)
        priority_array.from_numpy(np.pad(priorities, (0, capacity - count)))
    worklist = ti.algorithms.DeviceWorklist(capacity, ti.i32)
    worklist.values.from_numpy(np.pad(values_host, (0, capacity - count)))
    worklist.extent.set(count)

    result = worklist.resolve_conflicts(
        keys,
        priorities=priority_array,
        policy=policy,
    )
    snapshot = worklist.snapshot()
    np.testing.assert_array_equal(result.keys.to_numpy()[:3], [1, 2, 3])
    np.testing.assert_array_equal(snapshot.values, expected_values)
    np.testing.assert_array_equal(result.priorities.to_numpy()[:3], expected_priorities)
    np.testing.assert_array_equal(
        result.ordinals.to_numpy()[:3], [1, 0 if policy != "min_priority" else 2, 5]
    )
    assert snapshot.statistics.generated == count
    assert snapshot.statistics.accepted == 3
    assert snapshot.statistics.rejected == 3
    assert snapshot.statistics.conflicts == 3
    assert snapshot.statistics.winners == 3
    assert not snapshot.statistics.overflow


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_device_worklist_graph_sequence_and_completion_observation(monkeypatch):
    monkeypatch.setenv("TI_GRAPH_OBSERVATION_SLOTS", "2")
    capacity = 48
    count = 31
    values_host = np.arange(capacity, dtype=np.int32) + 9
    flags_host = ((np.arange(capacity) % 3) == 0).astype(np.int32)
    flags = ti.ndarray(ti.i32, shape=capacity)
    flags.from_numpy(flags_host)
    worklist = ti.algorithms.DeviceWorklist(capacity, ti.i32)
    worklist.values.from_numpy(values_host)
    worklist.extent.set(count)

    args = worklist.graph_args("active")
    flags_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "flags", ti.i32, ndim=1)
    sequence = ti.algorithms.DeviceWorklistSequence(args)
    sequence.select(flags_arg)
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    args.observe(builder, name="worklist")
    graph = builder.compile()
    runtime_args = worklist.runtime_arguments("active")
    runtime_args["flags"] = flags

    ticket = graph.submit(runtime_args)
    observed = ticket.observations()["worklist"]
    statistics = args.decode_observation(observed)
    expected = values_host[:count][flags_host[:count] != 0]
    assert observed["active_generated"] == count
    assert observed["active_accepted"] == expected.size
    assert observed["active_rejected"] == count - expected.size
    assert observed["active_conflicts"] == 0
    assert observed["active_winners"] == expected.size
    assert observed["active_overflow"] == 0
    assert statistics.generated == count
    assert statistics.accepted == expected.size
    assert not statistics.overflow
    assert worklist.next_extent.check() == expected.size
    np.testing.assert_array_equal(
        worklist.next_values.to_numpy()[: expected.size], expected
    )
    memory = graph.execution_stats().memory
    assert memory.observation_completion_attached
    assert memory.observation_readback_mode == (
        "completion_attached_pinned_copy"
        if impl.current_cfg().arch == ti_core.Arch.cuda
        else "completion_attached_host_visible"
    )


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_device_worklist_graph_atomic_producer_uses_recorded_reset(monkeypatch):
    monkeypatch.setenv("TI_GRAPH_OBSERVATION_SLOTS", "2")
    capacity = 23
    produced = capacity + 4
    worklist = ti.algorithms.DeviceWorklist(capacity, ti.i32)
    args = worklist.graph_args("frontier")
    count_arg = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "count", ti.i32)
    reset = ti.algorithms.DeviceWorklistSequence(args).prepare_next()
    finalize = ti.algorithms.DeviceWorklistSequence(args).finalize_next()
    builder = ti.graph.GraphBuilder()
    builder.append_native(reset)
    builder.dispatch(_append_range, *args.append_arguments(), count_arg)
    builder.append_native(finalize)
    args.observe(builder, name="frontier")
    graph = builder.compile()
    runtime_args = worklist.runtime_arguments("frontier", include_capacity=True)
    runtime_args["count"] = produced

    observed = graph.submit(runtime_args).observations()["frontier"]
    assert observed["frontier_generated"] == produced
    assert observed["frontier_accepted"] == capacity
    assert observed["frontier_rejected"] == produced - capacity
    assert observed["frontier_winners"] == capacity
    assert observed["frontier_overflow"] == 1
    assert worklist.next_extent.snapshot().overflow
    assert worklist.next_extent.snapshot().count == capacity


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_device_worklist_atomic_finalize_feeds_bounded_dispatch_and_report():
    capacity = 65
    produced = 19
    block_dim = 32

    @ti.kernel
    def consume(
        values: ti.types.ndarray(dtype=ti.i32, ndim=1),
        extent_state: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        ti.loop_config(block_dim=block_dim)
        for i in range(capacity):
            if i < ti.device_extent_count(extent_state):
                output[i] = values[i] * 2

    worklist = ti.algorithms.DeviceWorklist(capacity, ti.i32)
    args = worklist.graph_args("bounded_frontier")
    count_arg = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "count", ti.i32)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1)
    launch_state = worklist.next_extent.dispatch_state(block_dim)
    reset = ti.algorithms.DeviceWorklistSequence(args).prepare_next()
    finalize = ti.algorithms.DeviceWorklistSequence(args).finalize_next(
        dispatch_state=launch_state
    )
    builder = ti.graph.GraphBuilder()
    builder.append_native(reset)
    builder.dispatch(_append_range, *args.append_arguments(), count_arg)
    builder.append_native(finalize)
    handle = builder.dispatch_bounded(
        consume,
        args.next_values,
        args.next_extent,
        output_arg,
        extent=args.next_extent,
        capacity=capacity,
        block_dim=block_dim,
        launch_state=launch_state,
    )
    graph = builder.compile()
    assert all(
        type(node).__name__ != "_CompiledNativeGraphNode"
        for node in graph._spec.nodes
    )
    assert sum(node.source_native_count for node in graph._spec.nodes) == 2
    output = ti.ndarray(ti.i32, shape=capacity)
    output.fill(-1)
    runtime_args = worklist.runtime_arguments("bounded_frontier", include_capacity=True)
    runtime_args.update(count=produced, output=output)
    graph.run(runtime_args)

    expected = (np.arange(produced, dtype=np.int32) * 3 + 1) * 2
    np.testing.assert_array_equal(np.sort(output.to_numpy()[:produced]), expected)
    report = worklist.execution_report(handle, target="next")
    assert report.useful_count == produced
    assert report.generated == produced
    assert report.accepted == produced
    assert report.overflow is False
    if impl.current_cfg().arch == ti_core.Arch.vulkan:
        assert handle.capabilities.producer_owned_launch_state
        assert handle.capabilities.preparation_dispatches == 0
        assert report.exact_physical_grid
        assert report.executed_count == block_dim
    elif impl.current_cfg().arch in (ti_core.Arch.x64, ti_core.Arch.arm64):
        assert not handle.capabilities.producer_owned_launch_state
        assert report.exact_physical_grid
        assert report.executed_count == produced
    else:
        assert not handle.capabilities.producer_owned_launch_state
        assert not report.exact_physical_grid
        assert report.executed_count == capacity


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_worklist_finalize_fuses_graph_owned_bounded_publication():
    capacity = 65
    block_dim = 32

    @ti.kernel
    def consume(
        extent_state: ti.types.ndarray(dtype=ti.i32, ndim=1),
        visited: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        ti.loop_config(block_dim=block_dim)
        for i in range(capacity):
            ti.atomic_add(visited[0], 1)
            if i < ti.device_extent_count(extent_state):
                ti.atomic_add(visited[1], 1)

    worklist = ti.algorithms.DeviceWorklist(capacity, ti.i32)
    args = worklist.graph_args("vulkan_owned_frontier")
    count_arg = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "count", ti.i32)
    first_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "first_visited", ti.i32, ndim=1
    )
    second_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "second_visited", ti.i32, ndim=1
    )
    builder = ti.graph.GraphBuilder()
    builder.append_native(
        ti.algorithms.DeviceWorklistSequence(args).prepare_next()
    )
    builder.dispatch(_append_range, *args.append_arguments(), count_arg)
    builder.append_native(
        ti.algorithms.DeviceWorklistSequence(args).finalize_next()
    )
    handles = (
        builder.dispatch_bounded(
            consume,
            args.next_extent,
            first_arg,
            extent=args.next_extent,
            capacity=capacity,
            block_dim=block_dim,
        ),
        builder.dispatch_bounded(
            consume,
            args.next_extent,
            second_arg,
            extent=args.next_extent,
            capacity=capacity,
            block_dim=block_dim,
        ),
    )
    graph = builder.compile()

    assert all(
        type(node).__name__ == "_CompiledCGraphNode"
        for node in graph._spec.nodes
    )
    assert sum(node.source_native_count for node in graph._spec.nodes) == 2
    assert graph._spec.execution_definition["internal_storage_bytes"] == 16
    assert all(
        not handle.capabilities.producer_owned_launch_state for handle in handles
    )
    assert all(handle.capabilities.preparation_dispatches == 0 for handle in handles)
    assert [handle.workspace_bytes for handle in handles] == [16, 0]
    assert [handle.workspace_allocation_count for handle in handles] == [1, 0]

    first = ti.ndarray(ti.i32, shape=2)
    second = ti.ndarray(ti.i32, shape=2)
    runtime_args = worklist.runtime_arguments(
        "vulkan_owned_frontier", include_capacity=True
    )
    runtime_args.update(count=0, first_visited=first, second_visited=second)
    for requested in (0, 1, 19, capacity, capacity + 7):
        runtime_args["count"] = requested
        first.fill(0)
        second.fill(0)
        graph.run(runtime_args)
        useful = min(requested, capacity)
        expected_lanes = (
            0
            if useful == 0
            else min(
                capacity,
                ((useful + block_dim - 1) // block_dim) * block_dim,
            )
        )
        for visited in (first, second):
            np.testing.assert_array_equal(
                visited.to_numpy(), np.array([expected_lanes, useful])
            )
    assert graph.execution_stats().memory.persistent_argument_bytes >= 16


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_bounded_publication_falls_back_across_intervening_dispatch():
    capacity = 17
    block_dim = 16

    @ti.kernel
    def preserve_extent(extent_state: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for _ in range(1):
            extent_state[0] = extent_state[0]

    @ti.kernel
    def consume(
        extent_state: ti.types.ndarray(dtype=ti.i32, ndim=1),
        visited: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        ti.loop_config(block_dim=block_dim)
        for i in range(capacity):
            if i < ti.device_extent_count(extent_state):
                ti.atomic_add(visited[0], 1)

    worklist = ti.algorithms.DeviceWorklist(capacity, ti.i32)
    args = worklist.graph_args("vulkan_separated_frontier")
    visited_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "separated_visited", ti.i32, ndim=1
    )
    builder = ti.graph.GraphBuilder()
    builder.append_native(
        ti.algorithms.DeviceWorklistSequence(args).finalize_next()
    )
    builder.dispatch(preserve_extent, args.next_extent)
    handle = builder.dispatch_bounded(
        consume,
        args.next_extent,
        visited_arg,
        extent=args.next_extent,
        capacity=capacity,
        block_dim=block_dim,
    )
    graph = builder.compile()

    assert handle.capabilities.preparation_dispatches == 1
    assert handle.workspace_bytes == 12
    assert graph._spec.execution_definition["internal_storage_bytes"] == 12


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_worklist_finalize_keeps_bounded_state_graph_owned(monkeypatch):
    probe = dict(ti_core.cuda_bounded_dispatch_probe())
    if not probe["exact_device_grid_available"]:
        pytest.skip(probe["unavailable_reason"])
    monkeypatch.setenv("TI_CUDA_BOUNDED_DISPATCH_MODE", "device_update")
    monkeypatch.setenv("TI_GRAPH_CUDA_BOUNDED_UPDATE_POLICY", "per_node")
    capacity = 65
    block_dim = 32

    @ti.kernel
    def consume(
        extent_state: ti.types.ndarray(dtype=ti.i32, ndim=1),
        visited: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        ti.loop_config(block_dim=block_dim)
        for i in range(capacity):
            ti.atomic_add(visited[0], 1)
            if i < ti.device_extent_count(extent_state):
                ti.atomic_add(visited[1], 1)

    worklist = ti.algorithms.DeviceWorklist(capacity, ti.i32)
    args = worklist.graph_args("owned_frontier")
    count_arg = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "count", ti.i32)
    first_visited_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "first_visited", ti.i32, ndim=1
    )
    second_visited_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "second_visited", ti.i32, ndim=1
    )
    launch_state = worklist.next_extent.dispatch_state(block_dim)
    reset = ti.algorithms.DeviceWorklistSequence(args).prepare_next()
    finalize = ti.algorithms.DeviceWorklistSequence(args).finalize_next()
    builder = ti.graph.GraphBuilder()
    builder.append_native(reset)
    builder.dispatch(_append_range, *args.append_arguments(), count_arg)
    builder.append_native(finalize)
    handles = (
        builder.dispatch_bounded(
            consume,
            args.next_extent,
            first_visited_arg,
            extent=args.next_extent,
            capacity=capacity,
            block_dim=block_dim,
            launch_state=launch_state,
        ),
        builder.dispatch_bounded(
            consume,
            args.next_extent,
            second_visited_arg,
            extent=args.next_extent,
            capacity=capacity,
            block_dim=block_dim,
            launch_state=launch_state,
        ),
    )
    graph = builder.compile()
    first_visited = ti.ndarray(ti.i32, shape=2)
    second_visited = ti.ndarray(ti.i32, shape=2)
    runtime_args = worklist.runtime_arguments(
        "owned_frontier", include_capacity=True
    )
    runtime_args.update(
        count=0,
        first_visited=first_visited,
        second_visited=second_visited,
    )
    graph.execution_stats()

    for requested in (1, 0, capacity, 19, capacity + 7, capacity):
        runtime_args["count"] = requested
        first_visited.fill(0)
        second_visited.fill(0)
        graph.run(runtime_args)
        useful = min(requested, capacity)
        expected_lanes = (
            0
            if useful == 0
            else min(
                capacity,
                ((useful + block_dim - 1) // block_dim) * block_dim,
            )
        )
        for visited in (first_visited, second_visited):
            observed = visited.to_numpy()
            segment = next(
                item
                for item in graph.execution_stats().segments
                if item.kind == "cgraph"
            )
            assert int(observed[0]) == expected_lanes, (
                segment.last_path,
                segment.fallback_reason,
                segment.last_driver_error,
                segment.bounded_update_groups,
                segment.bounded_updater_dispatches,
                segment.bounded_grouped_payloads,
                segment.bounded_producer_fused_groups,
            )
            assert int(observed[1]) == useful
        snapshot = worklist.next_extent.snapshot()
        assert snapshot.count == useful
        assert snapshot.overflow is (requested > capacity)
    assert all(
        not handle.capabilities.producer_owned_launch_state for handle in handles
    )
    assert all(handle.capabilities.preparation_dispatches == 1 for handle in handles)
    report = graph.execution_stats()
    segment = next(item for item in report.segments if item.kind == "cgraph")
    assert segment.last_driver_error == 0
    assert segment.bounded_update_groups == 2
    assert segment.bounded_updater_dispatches == 2
    assert segment.bounded_grouped_payloads == 0
    assert segment.bounded_producer_fused_groups == 0
    assert segment.persistent_bounded_control_bytes == 64


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_cpu_worklist_finalize_preserves_exact_scheduler_lowering(monkeypatch):
    monkeypatch.setenv("TI_CPU_BOUNDED_DISPATCH_MODE", "exact_scheduler")
    capacity = 65

    @ti.kernel
    def consume(
        extent_state: ti.types.ndarray(dtype=ti.i32, ndim=1),
        visited: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in range(capacity):
            ti.atomic_add(visited[0], 1)
            if i < ti.device_extent_count(extent_state):
                ti.atomic_add(visited[1], 1)

    worklist = ti.algorithms.DeviceWorklist(capacity, ti.i32)
    args = worklist.graph_args("cpu_exact_frontier")
    count_arg = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "count", ti.i32)
    visited_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "visited", ti.i32, ndim=1
    )
    builder = ti.graph.GraphBuilder()
    builder.append_native(
        ti.algorithms.DeviceWorklistSequence(args).prepare_next()
    )
    builder.dispatch(_append_range, *args.append_arguments(), count_arg)
    builder.append_native(
        ti.algorithms.DeviceWorklistSequence(args).finalize_next()
    )
    handle = builder.dispatch_bounded(
        consume,
        args.next_extent,
        visited_arg,
        extent=args.next_extent,
        capacity=capacity,
    )
    graph = builder.compile()
    assert [type(node).__name__ for node in graph._spec.nodes] == [
        "_CompiledCGraphNode"
    ]
    visited = ti.ndarray(ti.i32, shape=2)
    runtime_args = worklist.runtime_arguments(
        "cpu_exact_frontier", include_capacity=True
    )
    runtime_args.update(count=19, visited=visited)
    graph.run(runtime_args)
    np.testing.assert_array_equal(visited.to_numpy(), np.array([19, 19]))
    report = worklist.execution_report(handle, target="next")
    assert report.exact_physical_grid
    assert report.executed_count == 19


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_device_worklist_graph_deterministic_claim_is_recordable():
    capacity = 24
    count = 7
    worklist = ti.algorithms.DeviceWorklist(capacity, ti.i32)
    values_host = np.array([90, 40, 71, 32, 83, 64, 55], dtype=np.int32)
    keys_host = np.array([4, 2, 4, 2, 4, 6, 5], dtype=np.int32)
    priorities_host = np.array([3, 4, 1, 2, 1, 0, 5], dtype=np.int32)
    worklist.values.from_numpy(np.pad(values_host, (0, capacity - count)))
    worklist.extent.set(count)
    keys = ti.ndarray(ti.i32, shape=capacity)
    priorities = ti.ndarray(ti.i32, shape=capacity)
    output_keys = ti.ndarray(ti.i32, shape=capacity)
    output_priorities = ti.ndarray(ti.i32, shape=capacity)
    output_ordinals = ti.ndarray(ti.i32, shape=capacity)
    keys.from_numpy(np.pad(keys_host, (0, capacity - count)))
    priorities.from_numpy(np.pad(priorities_host, (0, capacity - count)))

    args = worklist.graph_args("claims")
    keys_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "keys", ti.i32, ndim=1)
    priorities_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "priorities", ti.i32, ndim=1
    )
    output_keys_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "output_keys", ti.i32, ndim=1
    )
    output_priorities_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "output_priorities", ti.i32, ndim=1
    )
    output_ordinals_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "output_ordinals", ti.i32, ndim=1
    )
    sequence = ti.algorithms.DeviceWorklistSequence(args)
    sequence.resolve_conflicts(
        keys_arg,
        output_keys_arg,
        output_priorities_arg,
        output_ordinals_arg,
        priorities=priorities_arg,
        policy="min_priority",
    )
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    args.observe(builder, name="claims")
    graph = builder.compile()
    runtime_args = worklist.runtime_arguments("claims")
    runtime_args.update(
        keys=keys,
        priorities=priorities,
        output_keys=output_keys,
        output_priorities=output_priorities,
        output_ordinals=output_ordinals,
    )

    observed = None
    for _ in range(16):
        observed = graph.submit(runtime_args).observations()["claims"]
        np.testing.assert_array_equal(output_keys.to_numpy()[:4], [2, 4, 5, 6])
        np.testing.assert_array_equal(
            worklist.next_values.to_numpy()[:4], [32, 71, 55, 64]
        )
        np.testing.assert_array_equal(output_priorities.to_numpy()[:4], [2, 1, 5, 0])
        np.testing.assert_array_equal(output_ordinals.to_numpy()[:4], [3, 2, 6, 5])
    assert observed["claims_generated"] == count
    assert observed["claims_accepted"] == 4
    assert observed["claims_conflicts"] == 3
    assert observed["claims_winners"] == 4


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_device_worklist_workspace_and_runtime_memory_are_stable():
    capacity = 96
    values = np.arange(capacity, dtype=np.int32)
    flags = ti.ndarray(ti.i32, shape=capacity)
    flags.from_numpy(np.ones(capacity, dtype=np.int32))
    worklist = ti.algorithms.DeviceWorklist(capacity, ti.i32)
    worklist.values.from_numpy(values)
    worklist.extent.set(capacity)
    worklist.select(flags)
    worklist.commit_next()
    ti.sync()
    report_before = worklist.memory_report()
    assert report_before["total_bytes_current"] == (
        report_before["front_back_value_bytes"]
        + report_before["extent_bytes"]
        + report_before["counter_bytes"]
        + report_before["workspace_bytes_current"]
    )
    runtime_before = impl.get_runtime().prog._runtime_statistics_snapshot()["memory"]
    host_before = dict(ti_core.get_host_memory_pool_stats())
    device_before = dict(ti_core.get_device_memory_pool_stats())

    for iteration in range(32):
        worklist.extent.set(1 + iteration % capacity)
        worklist.select(flags)
    ti.sync()

    report_after = worklist.memory_report()
    assert report_after == report_before
    assert (
        impl.get_runtime().prog._runtime_statistics_snapshot()["memory"]
        == runtime_before
    )
    assert dict(ti_core.get_host_memory_pool_stats()) == host_before
    assert dict(ti_core.get_device_memory_pool_stats()) == device_before


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_device_worklist_rejects_stale_generation():
    worklist = ti.algorithms.DeviceWorklist(8, ti.i32)
    ti.reset()
    ti.init(arch=ti.cpu)
    with pytest.raises(ti.TaichiRuntimeError, match="stale"):
        worklist.prepare_next()
