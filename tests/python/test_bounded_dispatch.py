import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge._lib import core as ti_core
from taichi_forge.lang import impl
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_dynamic_work_capabilities_separate_launch_and_iteration_semantics():
    capabilities = ti.graph.dynamic_work_capabilities()
    bounded = capabilities["bounded_dispatch"]
    iteration = capabilities["structured_iteration"]
    observation = capabilities["observation"]
    arch = ti.lang.impl.current_cfg().arch

    assert capabilities["schema_version"] == 3
    assert capabilities["count_contract"] == {
        "owner": "DeviceExtent",
        "state_words": 2,
        "fixed_capacity": True,
        "device_published_count": True,
        "sticky_overflow": True,
        "runtime_generation_qualified": True,
        "replay_host_readback": False,
    }
    assert capabilities["worklist"]["available"]
    assert capabilities["worklist"]["deterministic_keyed_claim"]
    assert capabilities["worklist"]["capacity_mismatch_fail_closed"]
    assert capabilities["worklist"]["atomic_append_order"] == "unspecified"
    assert (
        capabilities["worklist"]["physical_launch_semantics"]
        == bounded["execution_semantics"]
    )
    assert observation["worklist_counters"]
    assert bounded["producer_owned_launch_state"]
    assert bounded["no_host_readback"]
    assert bounded["selected_route"] == bounded["route"]
    assert bounded["range_mapping"] in (
        "one_to_one",
        "grid_stride",
        "cpu_scheduler",
    )
    assert observation["completion_attached"]
    assert observation["readback_mode"] == (
        "completion_attached_pinned_copy"
        if arch == ti.cuda
        else "completion_attached_host_visible"
    )
    if arch == ti.vulkan:
        assert bounded["execution_semantics"] == "exact_device_grid"
        assert bounded["exact_physical_grid"]
        assert bounded["producer_packet_consumed"]
        assert bounded["default_preparation_dispatches"] == 1
        assert bounded["requested_route"] == "not_applicable"
        assert bounded["setup_probe_passed"]
        assert bounded["fallback_reason"] == "none"
        assert not iteration["command_termination_exact"]
    elif arch == ti.cuda:
        assert bounded["execution_semantics"] == "masked_capacity"
        assert bounded["masked_capacity"]
        assert not bounded["producer_packet_consumed"]
        assert bounded["default_preparation_dispatches"] == 0
        assert bounded["requested_route"] == "auto"
        assert bounded["minimum_driver_api_version"] == 12040
        assert not bounded["setup_probe_passed"]
        assert bounded["fallback_reason"] in (
            "cuda_driver_api_below_12040",
            "cuda_device_update_symbols_unavailable",
            "cuda_device_update_lowering_not_compiled",
            "cuda_device_update_probe_not_run",
            "cuda_bounded_runtime_path_not_compiled",
        )
        assert iteration["command_termination_exact"] == (
            iteration["execution_semantics"] == "exact_dynamic_termination"
        )
    else:
        assert bounded["execution_semantics"] == "masked_capacity"
        assert bounded["requested_route"] == "auto"
        assert bounded["fallback_reason"] == (
            "cpu_exact_scheduler_lowering_not_qualified"
        )
        assert iteration["execution_semantics"] == "portable_host_control"


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_cpu_bounded_route_selection_is_fail_closed(monkeypatch):
    monkeypatch.setenv("TI_CPU_BOUNDED_DISPATCH_MODE", "masked_capacity")
    capabilities = ti.graph.bounded_dispatch_capabilities()
    assert capabilities["requested_route"] == "masked_capacity"
    assert capabilities["selected_route"] == "masked_capacity"
    assert capabilities["fallback_reason"] == "forced_masked_capacity"

    monkeypatch.setenv("TI_CPU_BOUNDED_DISPATCH_MODE", "exact_scheduler")
    with pytest.raises(RuntimeError, match="exact_scheduler"):
        ti.graph.bounded_dispatch_capabilities()

    monkeypatch.setenv("TI_CPU_BOUNDED_DISPATCH_MODE", "invalid")
    with pytest.raises(RuntimeError, match="TI_CPU_BOUNDED_DISPATCH_MODE"):
        ti.graph.bounded_dispatch_capabilities()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_bounded_route_selection_is_fail_closed(monkeypatch):
    monkeypatch.setenv("TI_CUDA_BOUNDED_DISPATCH_MODE", "masked_capacity")
    capabilities = ti.graph.bounded_dispatch_capabilities()
    assert capabilities["requested_route"] == "masked_capacity"
    assert capabilities["selected_route"] == "masked_capacity"
    assert capabilities["fallback_reason"] == "forced_masked_capacity"

    monkeypatch.setenv("TI_CUDA_BOUNDED_DISPATCH_MODE", "device_update")
    with pytest.raises(RuntimeError, match="device_update"):
        ti.graph.bounded_dispatch_capabilities()

    monkeypatch.setenv("TI_CUDA_BOUNDED_DISPATCH_MODE", "invalid")
    with pytest.raises(RuntimeError, match="TI_CUDA_BOUNDED_DISPATCH_MODE"):
        ti.graph.bounded_dispatch_capabilities()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_bounded_device_update_driver_probe():
    probe = dict(ti_core.cuda_bounded_dispatch_probe())
    assert probe["setup_probe_attempted"]
    if probe["driver_version_eligible"] and probe["required_symbols_loaded"]:
        assert probe["device_update_ptx_compiled"]
        assert probe["device_update_ptx_linked"]
        assert probe["setup_probe_passed"], probe
        assert probe["zero_count_command_skip_qualified"]
        assert probe["probe_sparse_visited"] == 7
        assert probe["probe_zero_visited"] == 7
        assert probe["probe_rebound_visited"] == 10
        assert probe["probe_baseline_visited"] == 74
        assert probe["probe_reason"] == "none"
    else:
        assert not probe["setup_probe_passed"]
        assert probe["probe_reason"] in (
            "cuda_driver_api_below_12040",
            "cuda_device_update_symbols_unavailable",
        )


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_graph_host_known_bounded_dispatch_is_exact_and_clamped():
    capacity = 97

    @ti.kernel
    def consume(count: ti.i32, output: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(count):
            output[i] = i + 5

    count_arg = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "count", ti.i32)
    output_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1
    )
    builder = ti.graph.GraphBuilder()
    handle = builder.dispatch_bounded(
        consume,
        count_arg,
        output_arg,
        count=count_arg,
        capacity=capacity,
    )
    graph = builder.compile()
    output = ti.ndarray(ti.i32, shape=capacity)

    assert handle.capabilities.route == "exact_host_range"
    assert handle.capabilities.exact_grid
    assert ti.graph.bounded_dispatch_capabilities()["host_known_route"] == (
        "exact_host_range"
    )
    for requested, expected_count, overflow in (
        (-1, 0, True),
        (0, 0, False),
        (1, 1, False),
        (capacity, capacity, False),
        (capacity + 7, capacity, True),
    ):
        output.fill(-1)
        graph.run({"count": requested, "output": output})
        expected = np.full(capacity, -1, dtype=np.int32)
        expected[:expected_count] = np.arange(expected_count, dtype=np.int32) + 5
        np.testing.assert_array_equal(output.to_numpy(), expected)
        snapshot = handle.snapshot(requested)
        assert snapshot.useful_count == expected_count
        assert snapshot.executed_count == expected_count
        assert snapshot.overflow == overflow


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_graph_device_bounded_dispatch_routes_and_boundaries():
    capacity = 65
    block_dim = 32

    @ti.kernel
    def produce(
        requested: ti.i32,
        state: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        ti.device_extent_publish(state, capacity, requested)

    @ti.kernel
    def consume(
        state: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
        visited: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        ti.loop_config(block_dim=block_dim)
        for i in range(capacity):
            ti.atomic_add(visited[0], 1)
            if i < ti.device_extent_count(state):
                output[i] = i + 11

    requested_arg = ti.graph.Arg(
        ti.graph.ArgKind.SCALAR, "requested", ti.i32
    )
    extent_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "extent", ti.i32, ndim=1
    )
    output_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1
    )
    visited_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "visited", ti.i32, ndim=1
    )
    builder = ti.graph.GraphBuilder()
    builder.dispatch(produce, requested_arg, extent_arg)
    handle = builder.dispatch_bounded(
        consume,
        extent_arg,
        output_arg,
        visited_arg,
        extent=extent_arg,
        capacity=capacity,
        block_dim=block_dim,
    )
    graph = builder.compile()
    extent = ti.DeviceExtent(capacity)
    output = ti.ndarray(ti.i32, shape=capacity)
    visited = ti.ndarray(ti.i32, shape=1)
    capabilities = ti.graph.bounded_dispatch_capabilities()
    manifest = graph.task_manifest()

    assert capabilities["available"]
    assert capabilities["no_host_readback"]
    assert handle.capabilities.route == capabilities["route"]
    if ti.lang.impl.current_cfg().arch == ti.vulkan:
        assert handle.capabilities.exact_grid
        assert handle.capabilities.zero_count_command_skip
        assert handle.workspace_bytes == 12
        assert sum(task.indirect for task in manifest) == 1
        payload = next(task for task in manifest if "consume" in task.kernel_name)
        assert payload.range_mapping == "one_to_one"
    else:
        assert not handle.capabilities.exact_grid
        assert handle.capabilities.route == "masked_capacity"
        assert handle.workspace_bytes == 0
        assert not any(task.indirect for task in manifest)
        payload = next(task for task in manifest if "consume" in task.kernel_name)
        assert payload.range_mapping in ("cpu_scheduler", "grid_stride")

    for requested, count, overflow in (
        (0, 0, False),
        (1, 1, False),
        (capacity - 1, capacity - 1, False),
        (capacity, capacity, False),
        (capacity + 1, capacity, True),
    ):
        output.fill(-1)
        visited.fill(0)
        before = impl.get_runtime().prog._runtime_statistics_snapshot()
        graph.run(
            {
                "requested": requested,
                "extent": extent,
                "output": output,
                "visited": visited,
            }
        )
        after_enqueue = impl.get_runtime().prog._runtime_statistics_snapshot()
        assert after_enqueue["transfer"] == before["transfer"]
        assert (
            after_enqueue["synchronization"]["program_syncs"]
            == before["synchronization"]["program_syncs"]
        )
        expected = np.full(capacity, -1, dtype=np.int32)
        expected[:count] = np.arange(count, dtype=np.int32) + 11
        np.testing.assert_array_equal(output.to_numpy(), expected)
        snapshot = handle.snapshot(extent)
        assert snapshot.useful_count == count
        assert snapshot.capacity == capacity
        assert snapshot.overflow == overflow
        assert snapshot.skipped_count == snapshot.executed_count - count
        if handle.capabilities.exact_grid:
            assert snapshot.executed_count == min(
                capacity, ((count + block_dim - 1) // block_dim) * block_dim
            )
        else:
            assert snapshot.executed_count == capacity
        assert int(visited.to_numpy()[0]) == snapshot.executed_count


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_graph_device_prefix_sequence_publishes_bounded_launch_state():
    capacity = 96
    block_dim = 32

    @ti.kernel
    def consume(
        values: ti.types.ndarray(dtype=ti.i32, ndim=1),
        extent: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
        visited: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        ti.loop_config(block_dim=block_dim)
        for i in range(capacity):
            ti.atomic_add(visited[0], 1)
            if i < ti.device_extent_count(extent):
                output[i] = values[i] * 3 + 1

    values_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "values", ti.i32, ndim=1
    )
    input_extent_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "input_extent", ti.i32, ndim=1
    )
    flags_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "flags", ti.i32, ndim=1
    )
    compacted_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "compacted", ti.i32, ndim=1
    )
    compact_extent_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "compact_extent", ti.i32, ndim=1
    )
    output_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1
    )
    visited_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "visited", ti.i32, ndim=1
    )

    input_extent = ti.DeviceExtent(capacity)
    compact_extent = ti.DeviceExtent(capacity)
    launch_state = compact_extent.dispatch_state(block_dim)
    sequence = ti.algorithms.DevicePrefixSequence(capacity)
    sequence.input(values_arg, input_extent_arg).compact(
        flags_arg,
        compacted_arg,
        compact_extent_arg,
        dispatch_state=launch_state,
    )
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    handle = builder.dispatch_bounded(
        consume,
        compacted_arg,
        compact_extent_arg,
        output_arg,
        visited_arg,
        extent=compact_extent_arg,
        capacity=capacity,
        block_dim=block_dim,
        launch_state=launch_state,
    )
    graph = builder.compile()

    values_host = np.arange(capacity, dtype=np.int32) + 2
    flags_host = ((np.arange(capacity) % 4) != 1).astype(np.int32)
    values = ti.ndarray(ti.i32, shape=capacity)
    flags = ti.ndarray(ti.i32, shape=capacity)
    compacted = ti.ndarray(ti.i32, shape=capacity)
    output = ti.ndarray(ti.i32, shape=capacity)
    visited = ti.ndarray(ti.i32, shape=1)
    values.from_numpy(values_host)
    flags.from_numpy(flags_host)

    assert handle.capabilities.producer_owned_launch_state == (
        ti.lang.impl.current_cfg().arch == ti.vulkan
    )
    assert handle.capabilities.preparation_dispatches == 0
    assert handle.workspace_allocation_count == 0
    assert handle.workspace_bytes == 0
    assert graph._debug_info["nodes"][0]["kind"] == "device_prefix_sequence"
    assert graph._debug_info["nodes"][0]["operation_count"] == 1

    runtime_args = {
        "values": values,
        "input_extent": input_extent,
        "flags": flags,
        "compacted": compacted,
        "compact_extent": compact_extent,
        "output": output,
        "visited": visited,
    }
    for requested in (0, 1, capacity // 3, capacity, capacity + 1):
        input_extent.set(requested)
        output.fill(-1)
        visited.fill(0)
        graph.run(runtime_args)
        active = min(max(requested, 0), capacity)
        expected_values = values_host[:active][flags_host[:active] != 0]
        expected = np.full(capacity, -1, dtype=np.int32)
        expected[: expected_values.size] = expected_values * 3 + 1
        np.testing.assert_array_equal(output.to_numpy(), expected)
        snapshot = handle.snapshot(compact_extent)
        assert snapshot.useful_count == expected_values.size
        if handle.capabilities.exact_grid:
            assert int(visited.to_numpy()[0]) == min(
                capacity,
                ((expected_values.size + block_dim - 1) // block_dim) * block_dim,
            )
        else:
            assert int(visited.to_numpy()[0]) == capacity


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_graph_ordered_segmented_dispatch_reuses_payload_and_orders_ranges():
    capacity = 24
    block_dim = 32
    offsets_host = np.array([0, 5, 5, 11, 17], dtype=np.int32)
    segment_count = offsets_host.size - 1

    @ti.kernel
    def consume_segment(
        offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
        extent: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
        segment_state: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        ti.loop_config(block_dim=block_dim)
        for local in range(capacity):
            if local < ti.graph.segmented_dispatch_count(segment_state):
                index = ti.graph.segmented_dispatch_begin(segment_state) + local
                output[index] = ti.graph.segmented_dispatch_index(segment_state) + 1

    offsets_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "offsets", ti.i32, ndim=1
    )
    extent_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "extent", ti.i32, ndim=1
    )
    output_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1
    )
    builder = ti.graph.GraphBuilder()
    handle = builder.dispatch_ordered_segments(
        consume_segment,
        offsets_arg,
        extent_arg,
        output_arg,
        offsets=offsets_arg,
        extent=extent_arg,
        capacity=capacity,
        segment_count=segment_count,
        block_dim=block_dim,
    )
    graph = builder.compile()
    manifest = graph.task_manifest()
    offsets = ti.ndarray(ti.i32, shape=segment_count + 1)
    extent = ti.DeviceExtent(capacity)
    output = ti.ndarray(ti.i32, shape=capacity)
    offsets.from_numpy(offsets_host)
    extent.set(int(offsets_host[-1]))
    output.fill(-1)

    graph.run({"offsets": offsets, "extent": extent, "output": output})
    expected = np.full(capacity, -1, dtype=np.int32)
    for segment in range(segment_count):
        expected[offsets_host[segment] : offsets_host[segment + 1]] = segment + 1
    np.testing.assert_array_equal(output.to_numpy(), expected)
    snapshot = handle.snapshot(extent, offsets)
    assert snapshot.useful_count == int(offsets_host[-1])
    assert not snapshot.overflow
    assert len(snapshot.segments) == segment_count
    assert snapshot.segments[1].useful_count == 0
    if ti.lang.impl.current_cfg().arch == ti.vulkan:
        assert handle.workspace_allocation_count == 2
        assert handle.workspace_bytes == 32
    else:
        assert handle.workspace_allocation_count == 1
        assert handle.workspace_bytes == 20
    payload_tasks = [
        task for task in manifest if "consume_segment" in task.kernel_name
    ]
    assert len(payload_tasks) == segment_count
    assert len({task.kernel_name for task in payload_tasks}) == 1


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_graph_ordered_segmented_dispatch_clamps_invalid_offsets():
    capacity = 16
    segment_count = 3

    @ti.kernel
    def consume_segment(
        offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
        extent: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
        segment_state: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for local in range(capacity):
            if local < ti.graph.segmented_dispatch_count(segment_state):
                index = ti.graph.segmented_dispatch_begin(segment_state) + local
                output[index] += 1

    offsets_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "offsets", ti.i32, ndim=1
    )
    extent_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "extent", ti.i32, ndim=1
    )
    output_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1
    )
    builder = ti.graph.GraphBuilder()
    handle = builder.dispatch_ordered_segments(
        consume_segment,
        offsets_arg,
        extent_arg,
        output_arg,
        offsets=offsets_arg,
        extent=extent_arg,
        capacity=capacity,
        segment_count=segment_count,
    )
    graph = builder.compile()
    offsets = ti.ndarray(ti.i32, shape=segment_count + 1)
    offsets.from_numpy(np.array([-3, 5, 4, 99], dtype=np.int32))
    extent = ti.DeviceExtent(capacity)
    extent.set(10)
    output = ti.ndarray(ti.i32, shape=capacity + 2)
    output.fill(0)

    graph.run({"offsets": offsets, "extent": extent, "output": output})
    snapshot = handle.snapshot(extent, offsets)
    assert snapshot.overflow
    assert any(segment.invalid_offsets for segment in snapshot.segments)
    # Device clamping guarantees no writes outside the fixed capacity.
    np.testing.assert_array_equal(output.to_numpy()[capacity:], np.zeros(2, np.int32))


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_graph_bounded_dispatch_validates_binding_and_aot_boundary():
    capacity = 8

    @ti.kernel
    def consume(
        extent: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in range(capacity):
            if i < ti.device_extent_count(extent):
                output[i] += 1

    extent_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "extent", ti.i32, ndim=1
    )
    output_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1
    )
    builder = ti.graph.GraphBuilder()
    builder.dispatch_bounded(
        consume,
        extent_arg,
        output_arg,
        extent=extent_arg,
        capacity=capacity,
    )
    graph = builder.compile()
    output = ti.ndarray(ti.i32, shape=capacity)
    raw_extent = ti.ndarray(ti.i32, shape=2)
    with pytest.raises(RuntimeError, match="must be the DeviceExtent"):
        graph.run({"extent": raw_extent, "output": output})

    module = ti.aot.Module()
    with pytest.raises(RuntimeError, match="JIT-only internal fixed bindings"):
        module.add_graph("bounded", graph)

    count_arg = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "count", ti.i32)

    @ti.kernel
    def wrong_host_range(
        count: ti.i32, result: ti.types.ndarray(dtype=ti.i32, ndim=1)
    ):
        for i in range(capacity):
            if i < count:
                result[i] += 1

    invalid_builder = ti.graph.GraphBuilder()
    with pytest.raises(RuntimeError, match="sole range domain"):
        invalid_builder.dispatch_bounded(
            wrong_host_range,
            count_arg,
            output_arg,
            count=count_arg,
            capacity=capacity,
        )


@pytest.mark.run_in_serial
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_graph_bounded_dispatch_replay_memory_is_stable():
    capacity = 128

    @ti.kernel
    def consume(
        extent: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in range(capacity):
            if i < ti.device_extent_count(extent):
                output[i] += 1

    extent_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "extent", ti.i32, ndim=1
    )
    output_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1
    )
    builder = ti.graph.GraphBuilder()
    handle = builder.dispatch_bounded(
        consume,
        extent_arg,
        output_arg,
        extent=extent_arg,
        capacity=capacity,
    )
    graph = builder.compile()
    extent = ti.DeviceExtent(capacity)
    output = ti.ndarray(ti.i32, shape=capacity)
    extent.set(1)
    graph.run({"extent": extent, "output": output})
    ti.sync()
    graph.execution_stats()
    graph.run({"extent": extent, "output": output})
    ti.sync()

    graph_identity = graph._instance_debug_info
    memory_before = impl.get_runtime().prog._runtime_statistics_snapshot()["memory"]
    host_before = dict(ti_core.get_host_memory_pool_stats())
    device_before = dict(ti_core.get_device_memory_pool_stats())
    allocations = handle.workspace_allocation_count
    for count in range(1000):
        extent.set(count % (capacity + 1))
        graph.run({"extent": extent, "output": output})
    ti.sync()

    assert graph._instance_debug_info == graph_identity
    assert handle.workspace_allocation_count == allocations
    memory_after = impl.get_runtime().prog._runtime_statistics_snapshot()["memory"]
    assert memory_after["live_resources"] <= memory_before["live_resources"]
    assert memory_after["retiring_resources"] <= memory_before["retiring_resources"]
    for key in (
        "host_requested_live_bytes",
        "host_raw_bytes",
        "device_requested_live_bytes",
        "device_raw_bytes",
        "device_cached_bytes",
    ):
        before_value = memory_before[key]
        after_value = memory_after[key]
        if before_value is not None and after_value is not None:
            assert after_value <= before_value
    assert dict(ti_core.get_host_memory_pool_stats()) == host_before
    assert dict(ti_core.get_device_memory_pool_stats()) == device_before
