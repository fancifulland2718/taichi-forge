import concurrent.futures
import threading

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

    assert capabilities["schema_version"] == 4
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
    assert bounded["producer_owned_launch_state"] == (arch == ti.vulkan)
    assert bounded["no_host_readback"]
    assert bounded["selected_route"] == bounded["route"]
    assert bounded["range_mapping"] in (
        "one_to_one",
        "device_bounded_grid_stride",
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
        assert bounded["logical_iteration_exact"]
        assert bounded["physical_launch_kind"] == "indirect_one_to_one"
        assert bounded["execution_semantics"] == "exact_device_grid"
        assert bounded["exact_physical_grid"]
        assert bounded["producer_packet_consumed"]
        assert bounded["forge_producer_fusion_supported"]
        assert bounded["default_preparation_dispatches"] == 1
        assert bounded["requested_route"] == "not_applicable"
        assert bounded["setup_probe_passed"]
        assert bounded["fallback_reason"] == "none"
        assert not iteration["command_termination_exact"]
    elif arch == ti.cuda:
        assert bounded["execution_semantics"] == "exact_device_range"
        assert bounded["logical_iteration_exact"]
        assert bounded["physical_launch_kind"] == "saturated_grid_stride"
        assert bounded["range_mapping"] == "device_bounded_grid_stride"
        assert not bounded["masked_capacity"]
        assert not bounded["producer_packet_consumed"]
        assert not bounded["forge_producer_fusion_supported"]
        assert bounded["default_preparation_dispatches"] == 0
        assert bounded["requested_route"] == "auto"
        assert bounded["minimum_driver_api_version"] is None
        assert bounded["updater_dispatches"] == 0
        assert bounded["fallback_reason"] == "none"
        assert iteration["command_termination_exact"] == (
            iteration["execution_semantics"] == "exact_dynamic_termination"
        )
    else:
        assert bounded["logical_iteration_exact"]
        assert bounded["physical_launch_kind"] == "cpu_dynamic_chunks"
        assert not bounded["forge_producer_fusion_supported"]
        assert bounded["requested_route"] == "auto"
        assert bounded["execution_semantics"] == "exact_cpu_scheduler"
        assert bounded["exact_physical_grid"]
        assert bounded["fallback_reason"] == "none"
        assert iteration["execution_semantics"] == "portable_host_control"


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_bounded_internal_launch_storage_is_graph_instance_owned():
    capacity = 17

    @ti.kernel
    def consume(
        extent_state: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in range(capacity):
            if i < ti.device_extent_count(extent_state):
                output[i] = i + 1

    extent_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "owned_extent", ti.i32, ndim=1
    )
    output_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "owned_output", ti.i32, ndim=1
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
    extent.set(5)
    output = ti.ndarray(ti.i32, shape=capacity)
    output.fill(0)

    if ti.lang.impl.current_cfg().arch == ti.vulkan:
        assert type(handle._packet).__name__ == "_GraphInternalNdarraySpec"
        assert len(graph._instance._internal_storages) == 1
        storage = graph._instance._internal_storages[0]
        assert all(
            type(value).__name__ != "_GraphInternalNdarraySpec"
            for value in graph._instance._fixed_runtime_args.values()
        )
        assert handle.workspace_bytes == 12
    else:
        assert handle._packet is None
        assert graph._instance._internal_storages == ()
        storage = None

    args = {"owned_extent": extent, "owned_output": output}
    graph.run(args)
    graph.run(args)
    np.testing.assert_array_equal(
        output.to_numpy()[:5], np.arange(1, 6, dtype=np.int32)
    )
    if storage is not None:
        assert graph._instance._internal_storages[0] is storage


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_cpu_bounded_route_selection_is_explicit_and_fail_closed(monkeypatch):
    monkeypatch.delenv("TI_CPU_BOUNDED_DISPATCH_MODE", raising=False)
    capabilities = ti.graph.bounded_dispatch_capabilities()
    assert capabilities["requested_route"] == "auto"
    assert capabilities["selected_route"] == "exact_cpu_scheduler"
    assert capabilities["fallback_reason"] == "none"

    monkeypatch.setenv("TI_CPU_BOUNDED_DISPATCH_MODE", "masked_capacity")
    capabilities = ti.graph.bounded_dispatch_capabilities()
    assert capabilities["requested_route"] == "masked_capacity"
    assert capabilities["selected_route"] == "masked_capacity"
    assert capabilities["fallback_reason"] == "forced_masked_capacity"

    monkeypatch.setenv("TI_CPU_BOUNDED_DISPATCH_MODE", "exact_scheduler")
    capabilities = ti.graph.bounded_dispatch_capabilities()
    assert capabilities["requested_route"] == "exact_scheduler"
    assert capabilities["selected_route"] == "exact_cpu_scheduler"
    assert capabilities["execution_semantics"] == "exact_cpu_scheduler"
    assert capabilities["exact_physical_grid"]
    assert capabilities["zero_count_command_skip"]
    assert capabilities["range_mapping"] == "cpu_scheduler"
    assert capabilities["fallback_reason"] == "none"

    monkeypatch.setenv("TI_CPU_BOUNDED_DISPATCH_MODE", "invalid")
    with pytest.raises(RuntimeError, match="TI_CPU_BOUNDED_DISPATCH_MODE"):
        ti.graph.bounded_dispatch_capabilities()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_cpu_exact_bounded_dispatch_schedules_useful_range(monkeypatch):
    monkeypatch.setenv("TI_CPU_BOUNDED_DISPATCH_MODE", "exact_scheduler")
    capacity = 97

    @ti.kernel
    def produce(
        requested: ti.i32,
        extent: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        ti.device_extent_publish(extent, capacity, requested)

    @ti.kernel
    def consume(
        extent: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
        visited: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in range(capacity):
            ti.atomic_add(visited[0], 1)
            if i < ti.device_extent_count(extent):
                output[i] = i + 7

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
    )
    graph = builder.compile()
    output = ti.ndarray(ti.i32, shape=capacity)
    visited = ti.ndarray(ti.i32, shape=1)
    extents = (ti.DeviceExtent(capacity), ti.DeviceExtent(capacity))

    assert handle.capabilities.route == "exact_cpu_scheduler"
    assert handle.capabilities.no_host_readback
    assert handle.workspace_bytes == 0
    payload = next(
        task for task in graph.task_manifest() if "consume" in task.kernel_name
    )
    assert payload.range_mapping == "one_to_one"

    def run_case(extent, requested, expected_count, overflow):
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
        after = impl.get_runtime().prog._runtime_statistics_snapshot()
        assert after["transfer"] == before["transfer"]
        assert (
            after["synchronization"]["program_syncs"]
            == before["synchronization"]["program_syncs"]
        )
        expected = np.full(capacity, -1, dtype=np.int32)
        expected[:expected_count] = np.arange(expected_count) + 7
        np.testing.assert_array_equal(output.to_numpy(), expected)
        assert int(visited.to_numpy()[0]) == expected_count
        snapshot = handle.snapshot(extent)
        assert snapshot.useful_count == expected_count
        assert snapshot.executed_count == expected_count
        assert snapshot.skipped_count == 0
        assert snapshot.overflow == overflow

    for requested, expected_count, overflow in (
        (-1, 0, True),
        (0, 0, False),
        (1, 1, False),
        (33, 33, False),
        (capacity, capacity, False),
        (capacity + 11, capacity, True),
    ):
        run_case(extents[0], requested, expected_count, overflow)
    run_case(extents[1], 17, 17, False)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_cpu_exact_bounded_dispatch_preserves_tls_and_continue(monkeypatch):
    monkeypatch.setenv("TI_CPU_BOUNDED_DISPATCH_MODE", "exact_scheduler")
    capacity = 4097
    total = ti.field(ti.i64, shape=())

    @ti.kernel
    def produce(
        requested: ti.i32,
        extent: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        ti.device_extent_publish(extent, capacity, requested)

    @ti.kernel
    def consume(extent: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        # GPU launch geometry must not become CPU scheduler grain. The scalar
        # field reduction also exercises the per-chunk TLS prologue/epilogue.
        ti.loop_config(block_dim=1)
        for i in range(capacity):
            if i >= ti.device_extent_count(extent) or i % 3 == 0:
                continue
            total[None] += i % 11 + 1

    requested_arg = ti.graph.Arg(
        ti.graph.ArgKind.SCALAR, "requested", ti.i32
    )
    extent_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "extent", ti.i32, ndim=1
    )
    builder = ti.graph.GraphBuilder()
    builder.dispatch(produce, requested_arg, extent_arg)
    handle = builder.dispatch_bounded(
        consume,
        extent_arg,
        extent=extent_arg,
        capacity=capacity,
        block_dim=256,
    )
    graph = builder.compile()
    extent = ti.DeviceExtent(capacity)

    for requested in (0, 1, 511, 512, 513, capacity, capacity + 1):
        total[None] = 0
        graph.run({"requested": requested, "extent": extent})
        expected_count = min(requested, capacity)
        expected = sum(
            index % 11 + 1
            for index in range(expected_count)
            if index % 3 != 0
        )
        assert total[None] == expected
        snapshot = handle.snapshot(extent)
        assert snapshot.executed_count == expected_count
        assert snapshot.overflow == (requested > capacity)


@test_utils.test(arch=ti.cpu, offline_cache=False, debug=True)
def test_cpu_exact_bounded_dispatch_debug_chunk_preserves_continue(monkeypatch):
    monkeypatch.delenv("TI_CPU_BOUNDED_DISPATCH_MODE", raising=False)
    capacity = 513

    @ti.kernel
    def produce(
        requested: ti.i32,
        extent: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        ti.device_extent_publish(extent, capacity, requested)

    @ti.kernel
    def consume(
        extent: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        ti.loop_config(block_dim=1)
        for i in range(capacity):
            if i >= ti.device_extent_count(extent) or i % 5 == 0:
                continue
            output[i] = i + 1

    requested_arg = ti.graph.Arg(
        ti.graph.ArgKind.SCALAR, "requested", ti.i32
    )
    extent_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "extent", ti.i32, ndim=1
    )
    output_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1
    )
    builder = ti.graph.GraphBuilder()
    builder.dispatch(produce, requested_arg, extent_arg)
    builder.dispatch_bounded(
        consume,
        extent_arg,
        output_arg,
        extent=extent_arg,
        capacity=capacity,
    )
    graph = builder.compile()
    extent = ti.DeviceExtent(capacity)
    output = ti.ndarray(ti.i32, shape=capacity)
    output.fill(-1)
    graph.run({"requested": capacity, "extent": extent, "output": output})
    expected = np.arange(1, capacity + 1, dtype=np.int32)
    expected[::5] = -1
    np.testing.assert_array_equal(output.to_numpy(), expected)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_bounded_route_selection_is_fail_closed(monkeypatch):
    monkeypatch.delenv("TI_CUDA_BOUNDED_DISPATCH_MODE", raising=False)
    capabilities = ti.graph.bounded_dispatch_capabilities()
    assert capabilities["requested_route"] == "auto"
    assert capabilities["selected_route"] == "device_bounded_grid_stride"
    assert capabilities["logical_iteration_exact"]
    assert capabilities["physical_launch_kind"] == "saturated_grid_stride"
    assert capabilities["range_mapping"] == "device_bounded_grid_stride"
    assert not capabilities["exact_physical_grid"]
    assert capabilities["updater_dispatches"] == 0
    assert capabilities["minimum_driver_api_version"] is None

    monkeypatch.setenv("TI_CUDA_BOUNDED_DISPATCH_MODE", "masked_capacity")
    capabilities = ti.graph.bounded_dispatch_capabilities()
    assert capabilities["requested_route"] == "masked_capacity"
    assert capabilities["selected_route"] == "masked_capacity"
    assert capabilities["fallback_reason"] == "forced_masked_capacity"

    monkeypatch.setenv("TI_CUDA_BOUNDED_DISPATCH_MODE", "device_update")
    probe = dict(ti_core.cuda_bounded_dispatch_probe())
    if probe["exact_device_grid_available"]:
        capabilities = ti.graph.bounded_dispatch_capabilities()
        assert capabilities["requested_route"] == "device_update"
        assert capabilities["selected_route"] == "adaptive_device_grid_update"
        assert capabilities["logical_iteration_exact"]
        assert capabilities["physical_launch_kind"] == (
            "adaptive_saturated_grid_stride"
        )
        assert not capabilities["exact_physical_grid"]
        assert capabilities["zero_count_command_skip"]
        assert capabilities["range_mapping"] == "device_bounded_grid_stride"
        assert capabilities["updater_dispatches"] == 1
        assert capabilities["fallback_reason"] == "none"
        assert capabilities["producer_update_policy_requested"] == "auto"
        assert capabilities["producer_update_policy"] == "grouped_stateful"
        assert capabilities["grouped_updates_supported"]
        assert not capabilities["forge_producer_fusion_supported"]
        assert not capabilities["producer_packet_consumed"]
    else:
        with pytest.raises(RuntimeError, match="device_update"):
            ti.graph.bounded_dispatch_capabilities()

    monkeypatch.setenv("TI_CUDA_BOUNDED_DISPATCH_MODE", "invalid")
    with pytest.raises(RuntimeError, match="TI_CUDA_BOUNDED_DISPATCH_MODE"):
        ti.graph.bounded_dispatch_capabilities()

    monkeypatch.setenv("TI_CUDA_BOUNDED_DISPATCH_MODE", "device_update")
    monkeypatch.setenv("TI_GRAPH_CUDA_BOUNDED_UPDATE_POLICY", "invalid")
    with pytest.raises(RuntimeError, match="TI_GRAPH_CUDA_BOUNDED_UPDATE_POLICY"):
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
        assert probe["launch_update_persists"]
        assert probe["external_update_persists"]
        assert probe["partial_failure_capacity_safe"]
        assert probe["probe_sparse_visited"] == 7
        assert probe["probe_zero_visited"] == 7
        assert probe["probe_rebound_visited"] == 10
        assert probe["probe_baseline_visited"] == 74
        assert probe["probe_persistent_sparse_visited"] == 14
        assert probe["probe_persistent_disabled_visited"] == 0
        assert probe["probe_external_update_visited"] == 5
        assert probe["probe_external_reset_visited"] == 10
        assert probe["probe_partial_failure_visited"] == 14
        assert probe["probe_reason"] == "none"
    else:
        assert not probe["setup_probe_passed"]
        assert probe["probe_reason"] in (
            "cuda_driver_api_below_12040",
            "cuda_device_update_symbols_unavailable",
        )


@test_utils.test(arch=ti.cuda, offline_cache=False, saturating_grid_dim=2)
def test_cuda_exact_bounded_dispatch_replay_and_rebind(monkeypatch):
    probe = dict(ti_core.cuda_bounded_dispatch_probe())
    if not probe["exact_device_grid_available"]:
        pytest.skip(probe["unavailable_reason"])
    monkeypatch.setenv("TI_CUDA_BOUNDED_DISPATCH_MODE", "device_update")

    capacity = 97
    block_dim = 32

    @ti.kernel
    def produce(
        requested: ti.i32,
        extent: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        ti.device_extent_publish(extent, capacity, requested)

    @ti.kernel
    def consume(
        extent: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
        visited: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        ti.loop_config(block_dim=block_dim)
        for i in range(capacity):
            ti.atomic_add(visited[0], 1)
            if i < ti.device_extent_count(extent):
                output[i] = i + 3

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
    output = ti.ndarray(ti.i32, shape=capacity)
    visited = ti.ndarray(ti.i32, shape=1)
    extents = (ti.DeviceExtent(capacity), ti.DeviceExtent(capacity))
    graph.execution_stats()

    def run_case(extent, requested, expected_count, overflow):
        output.fill(-1)
        visited.fill(0)
        graph.run(
            {
                "requested": requested,
                "extent": extent,
                "output": output,
                "visited": visited,
            }
        )
        expected_visited = expected_count
        expected = np.full(capacity, -1, dtype=np.int32)
        expected[:expected_count] = np.arange(expected_count) + 3
        np.testing.assert_array_equal(output.to_numpy(), expected)
        assert int(visited.to_numpy()[0]) == expected_visited
        snapshot = handle.snapshot(extent)
        assert snapshot.useful_count == expected_count
        assert snapshot.executed_count == expected_visited
        assert snapshot.overflow == overflow
        report = graph.execution_stats()
        assert report.ordinary_fallback_segments == 0
        assert report.segments[0].last_driver_error == 0
        assert report.segments[0].last_path in (
            "cuda_capture",
            "cuda_exact_replay",
            "cuda_patched_replay",
        )

    for requested, expected_count, overflow in (
        (0, 0, False),
        (1, 1, False),
        (33, 33, False),
        (capacity, capacity, False),
        (capacity + 11, capacity, True),
    ):
        run_case(extents[0], requested, expected_count, overflow)
    run_case(extents[1], 17, 17, False)

    report = graph.execution_stats()
    assert report.memory.persistent_argument_bytes >= 32
    assert report.memory.persistent_bounded_control_bytes == 32


@test_utils.test(arch=ti.cuda, offline_cache=False, saturating_grid_dim=2)
def test_cuda_auto_bounded_dispatch_labeled_fallback_is_exact(monkeypatch):
    monkeypatch.delenv("TI_CUDA_BOUNDED_DISPATCH_MODE", raising=False)
    capacity = 257
    block_dim = 32

    @ti.kernel
    def produce(
        requested: ti.i32,
        extent: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        ti.device_extent_publish(extent, capacity, requested)

    @ti.kernel
    def consume(
        extent: ti.types.ndarray(dtype=ti.i32, ndim=1),
        visited: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        ti.loop_config(block_dim=block_dim)
        for i in range(capacity):
            ti.atomic_add(visited[0], 1)
            ti.atomic_add(visited[1], i + 1)

    requested_arg = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "requested", ti.i32)
    extent_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "extent", ti.i32, ndim=1)
    visited_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "visited", ti.i32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(produce, requested_arg, extent_arg)
    handle = builder.dispatch_bounded(
        consume,
        extent_arg,
        visited_arg,
        extent=extent_arg,
        capacity=capacity,
        block_dim=block_dim,
        label="bounded_fallback",
    )
    graph = builder.compile()
    extent = ti.DeviceExtent(capacity)
    visited = ti.ndarray(ti.i32, shape=2)

    for requested in (0, 1, 63, 64, 65, capacity, capacity + 1):
        visited.fill(0)
        graph.run(
            {
                "requested": requested,
                "extent": extent,
                "visited": visited,
            }
        )
        count = min(requested, capacity)
        observed = visited.to_numpy()
        assert int(observed[0]) == count
        assert int(observed[1]) == count * (count + 1) // 2
        snapshot = handle.snapshot(extent)
        assert snapshot.executed_count == count

    report = graph.execution_stats()
    assert report.ordinary_fallback_segments > 0
    assert report.segments[0].last_path == "ordinary"


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_grouped_stateful_bounded_update_shares_one_updater(monkeypatch):
    probe = dict(ti_core.cuda_bounded_dispatch_probe())
    if not probe["exact_device_grid_available"]:
        pytest.skip(probe["unavailable_reason"])
    monkeypatch.setenv("TI_CUDA_BOUNDED_DISPATCH_MODE", "device_update")
    monkeypatch.setenv("TI_GRAPH_CUDA_BOUNDED_UPDATE_POLICY", "grouped_stateful")

    capacity = 257
    block_dim = 32

    @ti.kernel
    def publish(
        requested: ti.i32,
        extent: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        ti.device_extent_publish(extent, capacity, requested)

    @ti.kernel
    def consume(
        extent: ti.types.ndarray(dtype=ti.i32, ndim=1),
        visited: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        ti.loop_config(block_dim=block_dim)
        for i in range(capacity):
            ti.atomic_add(visited[0], 1)
            if i < ti.device_extent_count(extent):
                ti.atomic_add(visited[1], i + 1)

    requested_arg = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "requested", ti.i32)
    extent_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "extent", ti.i32, ndim=1)
    first_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "first_visited", ti.i32, ndim=1)
    second_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "second_visited", ti.i32, ndim=1
    )
    builder = ti.graph.GraphBuilder()
    builder.dispatch(publish, requested_arg, extent_arg)
    handles = (
        builder.dispatch_bounded(
            consume,
            extent_arg,
            first_arg,
            extent=extent_arg,
            capacity=capacity,
            block_dim=block_dim,
        ),
        builder.dispatch_bounded(
            consume,
            extent_arg,
            second_arg,
            extent=extent_arg,
            capacity=capacity,
            block_dim=block_dim,
        ),
    )
    graph = builder.compile()
    extent = ti.DeviceExtent(capacity)
    first = ti.ndarray(ti.i32, shape=2)
    second = ti.ndarray(ti.i32, shape=2)
    args = {
        "requested": 0,
        "extent": extent,
        "first_visited": first,
        "second_visited": second,
    }
    graph.execution_stats()

    # Repeat states as well as changing them so the persistent-state fast path
    # is exercised without weakening boundary coverage.
    for requested, expected_count in (
        (-1, 0),
        (0, 0),
        (0, 0),
        (1, 1),
        (1, 1),
        (65, 65),
        (65, 65),
        (capacity, capacity),
        (capacity, capacity),
        (capacity + 9, capacity),
        (17, 17),
        (0, 0),
    ):
        args["requested"] = requested
        first.fill(0)
        second.fill(0)
        graph.run(args)
        expected = np.array(
            [
                expected_count,
                expected_count * (expected_count + 1) // 2,
            ],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(first.to_numpy(), expected)
        np.testing.assert_array_equal(second.to_numpy(), expected)

    stable_report = graph.execution_stats()
    stable_segment = stable_report.segments[0]
    assert stable_segment.bounded_update_replays == 12
    assert stable_segment.bounded_update_state_changes == 6
    assert stable_segment.bounded_update_cache_hits == 6
    assert stable_segment.bounded_node_api_calls == 14
    assert stable_segment.bounded_max_group_size == 2

    rebound_extent = ti.DeviceExtent(capacity)
    args["extent"] = rebound_extent
    args["requested"] = 33
    first.fill(0)
    second.fill(0)
    graph.run(args)
    rebound_expected = np.array([33, 33 * 34 // 2], dtype=np.int32)
    np.testing.assert_array_equal(first.to_numpy(), rebound_expected)
    np.testing.assert_array_equal(second.to_numpy(), rebound_expected)

    assert all(handle.capabilities.logical_iteration_exact for handle in handles)
    report = graph.execution_stats()
    segment = report.segments[0]
    assert segment.last_driver_error == 0
    assert segment.bounded_update_groups == 1
    assert segment.bounded_updater_dispatches == 1
    assert segment.bounded_grouped_payloads == 2
    assert segment.bounded_producer_fused_groups == 0
    assert segment.bounded_max_group_size == 2
    assert segment.bounded_update_replays == 1
    assert segment.bounded_update_state_changes == 1
    assert segment.bounded_update_cache_hits == 0
    assert segment.bounded_node_api_calls == 4
    assert report.memory.persistent_bounded_control_bytes == 96
    capabilities = ti.graph.bounded_dispatch_capabilities()
    assert capabilities["grouped_updates_supported"]
    assert capabilities["producer_update_policy"] == "grouped_stateful"


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_grouped_updater_telemetry_is_lazily_enabled(monkeypatch):
    probe = dict(ti_core.cuda_bounded_dispatch_probe())
    if not probe["exact_device_grid_available"]:
        pytest.skip(probe["unavailable_reason"])
    monkeypatch.setenv("TI_CUDA_BOUNDED_DISPATCH_MODE", "device_update")
    monkeypatch.setenv(
        "TI_GRAPH_CUDA_BOUNDED_UPDATE_POLICY", "grouped_stateful"
    )

    capacity = 65
    block_dim = 32

    @ti.kernel
    def publish(
        requested: ti.i32,
        extent: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        ti.device_extent_publish(extent, capacity, requested)

    @ti.kernel
    def consume(
        extent: ti.types.ndarray(dtype=ti.i32, ndim=1),
        observed: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        ti.loop_config(block_dim=block_dim)
        for i in range(capacity):
            if i < ti.device_extent_count(extent):
                ti.atomic_add(observed[0], 1)

    requested_arg = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "requested", ti.i32)
    extent_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "extent", ti.i32, ndim=1)
    observed_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "observed", ti.i32, ndim=1
    )
    builder = ti.graph.GraphBuilder()
    builder.dispatch(publish, requested_arg, extent_arg)
    for _ in range(2):
        builder.dispatch_bounded(
            consume,
            extent_arg,
            observed_arg,
            extent=extent_arg,
            capacity=capacity,
            block_dim=block_dim,
        )
    graph = builder.compile()
    extent = ti.DeviceExtent(capacity)
    observed = ti.ndarray(ti.i32, shape=1)
    args = {"requested": 17, "extent": extent, "observed": observed}

    # Materialize and initialize the stateful updater before diagnostics are
    # requested. The first snapshot enables device counters for later replays
    # without rewriting the cached launch state.
    graph.run(args)
    first = graph.execution_stats().segments[0]
    assert first.bounded_update_replays == 0
    assert first.bounded_update_state_changes == 0
    assert first.bounded_update_cache_hits == 0
    assert first.bounded_node_api_calls == 0

    graph.run(args)
    graph.run(args)
    report = graph.execution_stats()
    segment = report.segments[0]
    assert segment.bounded_max_group_size == 2
    assert segment.bounded_update_replays == 2
    assert segment.bounded_update_state_changes == 0
    assert segment.bounded_update_cache_hits == 2
    assert segment.bounded_node_api_calls == 0
    assert report.memory.persistent_bounded_control_bytes == 96
    assert int(observed.to_numpy()[0]) == 3 * 2 * 17


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_launch_state_compatibility_keeps_per_node_ownership(monkeypatch):
    probe = dict(ti_core.cuda_bounded_dispatch_probe())
    if not probe["exact_device_grid_available"]:
        pytest.skip(probe["unavailable_reason"])
    monkeypatch.setenv("TI_CUDA_BOUNDED_DISPATCH_MODE", "device_update")
    monkeypatch.setenv("TI_GRAPH_CUDA_BOUNDED_UPDATE_POLICY", "per_node")

    capacity = 97
    block_dim = 32

    @ti.kernel
    def publish(
        requested: ti.i32,
        extent: ti.types.ndarray(dtype=ti.i32, ndim=1),
        packet: ti.types.ndarray(dtype=ti.u32, ndim=1),
    ):
        ti.device_dispatch_state_publish(extent, packet, capacity, requested)

    @ti.kernel
    def consume(
        extent: ti.types.ndarray(dtype=ti.i32, ndim=1),
        visited: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        ti.loop_config(block_dim=block_dim)
        for i in range(capacity):
            ti.atomic_add(visited[0], 1)
            if i < ti.device_extent_count(extent):
                ti.atomic_add(visited[1], 1)

    requested_arg = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "requested", ti.i32)
    extent_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "extent", ti.i32, ndim=1
    )
    packet_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "packet", ti.u32, ndim=1
    )
    first_visited_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "first_visited", ti.i32, ndim=1
    )
    second_visited_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "second_visited", ti.i32, ndim=1
    )
    extent = ti.DeviceExtent(capacity)
    launch_state = extent.dispatch_state(block_dim)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(publish, requested_arg, extent_arg, packet_arg)
    handles = (
        builder.dispatch_bounded(
            consume,
            extent_arg,
            first_visited_arg,
            extent=extent_arg,
            capacity=capacity,
            block_dim=block_dim,
            launch_state=launch_state,
        ),
        builder.dispatch_bounded(
            consume,
            extent_arg,
            second_visited_arg,
            extent=extent_arg,
            capacity=capacity,
            block_dim=block_dim,
            launch_state=launch_state,
        ),
    )
    graph = builder.compile()
    first_visited = ti.ndarray(ti.i32, shape=2)
    second_visited = ti.ndarray(ti.i32, shape=2)
    args = {
        "requested": 0,
        "extent": extent,
        "packet": launch_state.packet,
        "first_visited": first_visited,
        "second_visited": second_visited,
    }
    graph.execution_stats()

    for requested in (1, 0, capacity, 33, capacity):
        args["requested"] = requested
        first_visited.fill(0)
        second_visited.fill(0)
        graph.run(args)
        expected_grid_lanes = min(requested, capacity)
        for visited in (first_visited, second_visited):
            observed = visited.to_numpy()
            segment = graph.execution_stats().segments[0]
            assert int(observed[0]) == expected_grid_lanes, (
                segment.last_path,
                segment.fallback_reason,
                segment.last_driver_error,
                segment.bounded_update_groups,
                segment.bounded_updater_dispatches,
                segment.bounded_grouped_payloads,
                segment.persistent_bounded_control_bytes,
            )
            assert int(observed[1]) == requested

    assert all(
        not handle.capabilities.producer_owned_launch_state for handle in handles
    )
    assert all(handle.capabilities.preparation_dispatches == 1 for handle in handles)
    report = graph.execution_stats()
    segment = report.segments[0]
    assert segment.last_driver_error == 0
    assert segment.bounded_update_groups == 2
    assert segment.bounded_updater_dispatches == 2
    assert segment.bounded_grouped_payloads == 0
    assert segment.bounded_producer_fused_groups == 0
    assert segment.bounded_update_replays == 0
    assert segment.bounded_update_state_changes == 0
    assert segment.bounded_update_cache_hits == 0
    assert segment.bounded_node_api_calls == 0
    assert segment.bounded_max_group_size == 1
    assert report.memory.persistent_bounded_control_bytes == 64


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_exact_bounded_dispatch_two_graph_concurrent_replay(monkeypatch):
    arch = ti.lang.impl.current_cfg().arch
    if arch == ti.cuda:
        probe = dict(ti_core.cuda_bounded_dispatch_probe())
        if not probe["exact_device_grid_available"]:
            pytest.skip(probe["unavailable_reason"])
        monkeypatch.setenv("TI_CUDA_BOUNDED_DISPATCH_MODE", "device_update")
    else:
        monkeypatch.setenv("TI_CPU_BOUNDED_DISPATCH_MODE", "exact_scheduler")

    capacity = 257
    block_dim = 64

    @ti.kernel
    def clear(
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
        visited: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        visited[0] = 0
        for i in range(capacity):
            output[i] = -1

    @ti.kernel
    def produce(
        requested: ti.i32,
        extent: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        ti.device_extent_publish(extent, capacity, requested)

    @ti.kernel
    def consume(
        extent: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
        visited: ti.types.ndarray(dtype=ti.i32, ndim=1),
        token: ti.i32,
    ):
        ti.loop_config(block_dim=block_dim)
        for i in range(capacity):
            ti.atomic_add(visited[0], 1)
            if i < ti.device_extent_count(extent):
                output[i] = token + i

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
    token_arg = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "token", ti.i32)

    def make_graph():
        builder = ti.graph.GraphBuilder()
        builder.dispatch(clear, output_arg, visited_arg)
        builder.dispatch(produce, requested_arg, extent_arg)
        handle = builder.dispatch_bounded(
            consume,
            extent_arg,
            output_arg,
            visited_arg,
            token_arg,
            extent=extent_arg,
            capacity=capacity,
            block_dim=block_dim,
        )
        return builder.compile(), handle

    graphs_and_handles = (make_graph(), make_graph())
    extents = (ti.DeviceExtent(capacity), ti.DeviceExtent(capacity))
    outputs = (
        ti.ndarray(ti.i32, shape=capacity),
        ti.ndarray(ti.i32, shape=capacity),
    )
    visited = (ti.ndarray(ti.i32, shape=1), ti.ndarray(ti.i32, shape=1))
    tokens = (1000, 2000)
    runtime_args = tuple(
        {
            "requested": 0,
            "extent": extents[index],
            "output": outputs[index],
            "visited": visited[index],
            "token": tokens[index],
        }
        for index in range(2)
    )
    for index, (graph, _) in enumerate(graphs_and_handles):
        graph.run(runtime_args[index])
    ti.sync()

    counts = (0, 1, 65, capacity, 17, capacity - 1)
    barrier = threading.Barrier(2)

    def replay(index):
        graph, _ = graphs_and_handles[index]
        args = runtime_args[index]
        barrier.wait()
        for replay_index in range(64):
            args["requested"] = counts[replay_index % len(counts)]
            graph.run(args)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = tuple(executor.submit(replay, index) for index in range(2))
        for future in futures:
            future.result()
    ti.sync()

    final_count = counts[63 % len(counts)]
    for index, (_, handle) in enumerate(graphs_and_handles):
        snapshot = handle.snapshot(extents[index])
        assert snapshot.useful_count == final_count
        assert int(visited[index].to_numpy()[0]) == snapshot.executed_count
        expected = np.full(capacity, -1, dtype=np.int32)
        expected[:final_count] = tokens[index] + np.arange(final_count)
        np.testing.assert_array_equal(outputs[index].to_numpy(), expected)


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_bounded_same_graph_inflight_submissions_preserve_internal_state(
    monkeypatch,
):
    if ti.lang.impl.current_cfg().arch == ti.cuda:
        probe = dict(ti_core.cuda_bounded_dispatch_probe())
        if not probe["exact_device_grid_available"]:
            pytest.skip(probe["unavailable_reason"])
        monkeypatch.setenv("TI_CUDA_BOUNDED_DISPATCH_MODE", "device_update")

    capacity = 65
    block_dim = 32

    @ti.kernel
    def clear(output: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(capacity):
            output[i] = -1

    @ti.kernel
    def publish(
        requested: ti.i32,
        extent: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        ti.device_extent_publish(extent, capacity, requested)

    @ti.kernel
    def consume(
        extent: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
        token: ti.i32,
    ):
        ti.loop_config(block_dim=block_dim)
        for i in range(capacity):
            if i < ti.device_extent_count(extent):
                output[i] = token + i

    requested_arg = ti.graph.Arg(
        ti.graph.ArgKind.SCALAR, "requested", ti.i32
    )
    extent_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "extent", ti.i32, ndim=1
    )
    output_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1
    )
    token_arg = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "token", ti.i32)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(clear, output_arg)
    builder.dispatch(publish, requested_arg, extent_arg)
    builder.dispatch_bounded(
        consume,
        extent_arg,
        output_arg,
        token_arg,
        extent=extent_arg,
        capacity=capacity,
        block_dim=block_dim,
    )
    graph = builder.compile()
    # Enable replay counters before the first submission so this regression
    # also proves that fixed-slot backpressure does not escape through an
    # ordinary fallback, which cannot preserve indirect dispatch semantics.
    if ti.lang.impl.current_cfg().arch == ti.vulkan:
        graph.execution_stats()

    counts = (0, 1, 33, capacity, 19, capacity - 1) * 2
    extents = tuple(ti.DeviceExtent(capacity) for _ in counts)
    outputs = tuple(ti.ndarray(ti.i32, shape=capacity) for _ in counts)
    tickets = []
    for index, count in enumerate(counts):
        tickets.append(
            graph.submit(
                {
                    "requested": count,
                    "extent": extents[index],
                    "output": outputs[index],
                    "token": 1000 * (index + 1),
                }
            )
        )
    for ticket in reversed(tickets):
        ticket.wait()

    if ti.lang.impl.current_cfg().arch == ti.vulkan:
        replay_stats = graph._graph_stats[0]
        assert replay_stats["attempts"] == len(tickets)
        assert replay_stats["ordinary_fallbacks"] == 0
        assert replay_stats["replay_slot_saturation_fallbacks"] == 0
        assert replay_stats["records"] + replay_stats["replays"] == len(tickets)

    for index, count in enumerate(counts):
        expected = np.full(capacity, -1, dtype=np.int32)
        expected[:count] = 1000 * (index + 1) + np.arange(count)
        np.testing.assert_array_equal(outputs[index].to_numpy(), expected)


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
    if handle.capabilities.logical_iteration_exact:
        assert handle.workspace_bytes == (
            12 if ti.lang.impl.current_cfg().arch == ti.vulkan else 0
        )
        assert sum(task.indirect for task in manifest) == (
            1 if ti.lang.impl.current_cfg().arch == ti.vulkan else 0
        )
        payload = next(task for task in manifest if "consume" in task.kernel_name)
        assert (
            payload.range_mapping
            == {
                ti.cpu: "one_to_one",
                ti.cuda: "device_bounded_grid_stride",
                ti.vulkan: "one_to_one",
            }[ti.lang.impl.current_cfg().arch]
        )
    else:
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
        if handle.capabilities.logical_iteration_exact:
            if ti.lang.impl.current_cfg().arch == ti.vulkan:
                assert snapshot.executed_count == min(
                    capacity, ((count + block_dim - 1) // block_dim) * block_dim
                )
            else:
                assert snapshot.executed_count == count
        else:
            assert snapshot.executed_count == capacity
        assert int(visited.to_numpy()[0]) == snapshot.executed_count, (
            graph.execution_stats().segments[0].last_driver_error
        )


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
        or (
            ti.lang.impl.current_cfg().arch == ti.cuda
            and handle.capabilities.exact_grid
        )
    )
    assert handle.capabilities.preparation_dispatches == (
        1
        if ti.lang.impl.current_cfg().arch == ti.cuda
        and handle.capabilities.exact_grid
        else 0
    )
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
        if handle.capabilities.logical_iteration_exact:
            if ti.lang.impl.current_cfg().arch == ti.vulkan:
                assert int(visited.to_numpy()[0]) == min(
                    capacity,
                    ((expected_values.size + block_dim - 1) // block_dim)
                    * block_dim,
                )
            else:
                assert int(visited.to_numpy()[0]) == expected_values.size
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


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_graph_bounded_consumers_share_one_extent_contract():
    capacity = 16

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
    first_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "first", ti.i32, ndim=1
    )
    second_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "second", ti.i32, ndim=1
    )
    builder = ti.graph.GraphBuilder()
    handles = (
        builder.dispatch_bounded(
            consume,
            extent_arg,
            first_arg,
            extent=extent_arg,
            capacity=capacity,
        ),
        builder.dispatch_bounded(
            consume,
            extent_arg,
            second_arg,
            extent=extent_arg,
            capacity=capacity,
        ),
    )
    graph = builder.compile()
    contracts = tuple(
        lease
        for lease in graph._spec.lifetime_leases
        if type(lease).__name__ == "_DeviceExtentGraphContract"
    )
    assert len(contracts) == 1
    assert all(handle not in graph._spec.lifetime_leases for handle in handles)

    extent = ti.DeviceExtent(capacity)
    extent.set(7)
    first = ti.ndarray(ti.i32, shape=capacity)
    second = ti.ndarray(ti.i32, shape=capacity)
    graph.run({"extent": extent, "first": first, "second": second})
    np.testing.assert_array_equal(first.to_numpy()[:7], np.ones(7, np.int32))
    np.testing.assert_array_equal(second.to_numpy()[:7], np.ones(7, np.int32))
    assert handles[0].snapshot(extent).useful_count == 7


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_consecutive_bounded_consumers_share_prepared_packet(monkeypatch):
    capacity = 65
    block_dim = 32

    @ti.kernel
    def consume(
        extent: ti.types.ndarray(dtype=ti.i32, ndim=1),
        visited: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        ti.loop_config(block_dim=block_dim)
        for i in range(capacity):
            ti.atomic_add(visited[0], 1)
            if i < ti.device_extent_count(extent):
                ti.atomic_add(visited[1], 1)

    extent_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "extent", ti.i32, ndim=1)
    first_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "first_visited", ti.i32, ndim=1)
    second_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "second_visited", ti.i32, ndim=1
    )
    builder = ti.graph.GraphBuilder()
    handles = (
        builder.dispatch_bounded(
            consume,
            extent_arg,
            first_arg,
            extent=extent_arg,
            capacity=capacity,
            block_dim=block_dim,
        ),
        builder.dispatch_bounded(
            consume,
            extent_arg,
            second_arg,
            extent=extent_arg,
            capacity=capacity,
            block_dim=block_dim,
        ),
    )
    graph = builder.compile()
    assert [handle.capabilities.preparation_dispatches for handle in handles] == [
        1,
        0,
    ]
    assert [handle.workspace_bytes for handle in handles] == [12, 0]
    assert graph._spec.execution_definition["internal_storage_bytes"] == 12

    extent = ti.DeviceExtent(capacity)
    first = ti.ndarray(ti.i32, shape=2)
    second = ti.ndarray(ti.i32, shape=2)
    for requested in (0, 1, 33, capacity):
        extent.set(requested)
        first.fill(0)
        second.fill(0)
        graph.run({"extent": extent, "first_visited": first, "second_visited": second})
        encoded = (
            0
            if requested == 0
            else min(
                capacity,
                ((requested + block_dim - 1) // block_dim) * block_dim,
            )
        )
        expected = np.array([encoded, requested], dtype=np.int32)
        np.testing.assert_array_equal(first.to_numpy(), expected)
        np.testing.assert_array_equal(second.to_numpy(), expected)

    monkeypatch.setenv("TI_GRAPH_VULKAN_BOUNDED_PACKET_POLICY", "per_consumer")
    baseline_builder = ti.graph.GraphBuilder()
    baseline_handles = tuple(
        baseline_builder.dispatch_bounded(
            consume,
            extent_arg,
            target,
            extent=extent_arg,
            capacity=capacity,
            block_dim=block_dim,
        )
        for target in (first_arg, second_arg)
    )
    baseline_graph = baseline_builder.compile()
    assert [
        handle.capabilities.preparation_dispatches for handle in baseline_handles
    ] == [1, 1]
    assert baseline_graph._spec.execution_definition["internal_storage_bytes"] == 24


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_prepared_packet_is_invalidated_by_intervening_dispatch():
    capacity = 17

    @ti.kernel
    def consume(
        extent: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in range(capacity):
            if i < ti.device_extent_count(extent):
                output[i] += 1

    @ti.kernel
    def possible_extent_writer(
        extent: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        extent[0] = extent[0]

    extent_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "extent", ti.i32, ndim=1)
    first_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "first", ti.i32, ndim=1)
    second_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "second", ti.i32, ndim=1)
    builder = ti.graph.GraphBuilder()
    first_handle = builder.dispatch_bounded(
        consume,
        extent_arg,
        first_arg,
        extent=extent_arg,
        capacity=capacity,
    )
    builder.dispatch(possible_extent_writer, extent_arg)
    second_handle = builder.dispatch_bounded(
        consume,
        extent_arg,
        second_arg,
        extent=extent_arg,
        capacity=capacity,
    )
    graph = builder.compile()

    assert first_handle.capabilities.preparation_dispatches == 1
    assert second_handle.capabilities.preparation_dispatches == 1
    assert first_handle.workspace_bytes == 12
    assert second_handle.workspace_bytes == 12
    assert graph._spec.execution_definition["internal_storage_bytes"] == 24


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


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_bounded_pipeline_telemetry_correlates_labels_geometry_and_work():
    capacity = 257
    block_dim = 64

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
    for color in ("red", "black"):
        builder.dispatch_bounded(
            consume,
            extent_arg,
            output_arg,
            extent=extent_arg,
            capacity=capacity,
            block_dim=block_dim,
            label=f"sweep=7/color={color}",
        )
    graph = builder.compile()
    extent = ti.DeviceExtent(capacity)
    output = ti.ndarray(ti.i32, shape=capacity)
    args = {"extent": extent, "output": output}

    extent.set(73)
    output.fill(0)
    ordinary = graph.submit(args)
    ordinary.wait()
    assert ordinary.pipeline_report() is None
    assert not graph._instance.structured_telemetry_arena_stats["materialized"]

    extent.set(73)
    output.fill(0)
    ticket = graph.submit(args, telemetry=True)
    # This write is ordered after the ticket's tail snapshot. The report must
    # retain 73 instead of observing the later reusable DeviceExtent value.
    extent.set(5)
    pipeline = ticket.pipeline_report()
    assert pipeline.schema_version == 2
    assert pipeline.stage_count == 1
    assert pipeline.bounded_dispatch_count == 2
    stage = pipeline.stages[0]
    assert stage.task_mapping_status == "available"
    assert stage.bounded_mapping_status == "available"
    assert stage.tasks
    assert {task.dispatch_label for task in stage.tasks if task.dispatch_label} == {
        "sweep=7/color=red",
        "sweep=7/color=black",
    }
    assert all(task.task_id.startswith("tf:") for task in stage.tasks)
    reports = stage.bounded_dispatches
    assert {report.label for report in reports} == {
        "sweep=7/color=red",
        "sweep=7/color=black",
    }
    assert all(report.physical_dispatch_index is not None for report in reports)
    assert all(report.count_source == "device_extent" for report in reports)
    assert all(report.source_count == 73 for report in reports)
    assert all(report.useful_count == 73 for report in reports)
    assert all(not report.overflow for report in reports)
    assert all(
        report.snapshot_status == "ticket_device_snapshot" for report in reports
    )
    arch = ti.lang.impl.current_cfg().arch
    if arch == ti.vulkan:
        assert all(report.selected_route == "exact_indirect" for report in reports)
        assert all(report.encoded_lanes == 128 for report in reports)
        assert all(report.executed_count == 128 for report in reports)
        assert all(report.skipped_count == 55 for report in reports)
    else:
        assert all(report.logical_iteration_exact for report in reports)
        assert all(report.executed_count == 73 for report in reports)
        assert all(report.encoded_lanes == 73 for report in reports)
        assert all(report.skipped_count == 0 for report in reports)
    telemetry_stats = graph._instance.structured_telemetry_arena_stats
    assert telemetry_stats["reserved_bytes"] == 8
    assert telemetry_stats["host_readback_bytes"] == 8


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_host_bounded_pipeline_telemetry_reports_raw_clamped_count_without_buffer():
    capacity = 17

    @ti.kernel
    def consume(count: ti.i32, output: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(count):
            output[i] = i + 1

    count_arg = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "count", ti.i32)
    output_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1
    )
    builder = ti.graph.GraphBuilder()
    builder.dispatch_bounded(
        consume,
        count_arg,
        output_arg,
        count=count_arg,
        capacity=capacity,
        block_dim=8,
        label="host-count",
    )
    graph = builder.compile()
    output = ti.ndarray(ti.i32, shape=capacity)
    report = graph.submit(
        {"count": capacity + 9, "output": output}, telemetry=True
    ).pipeline_report()
    bounded = report.stages[0].bounded_dispatches[0]
    assert bounded.count_source == "host_scalar"
    assert bounded.source_count == capacity + 9
    assert bounded.useful_count == capacity
    assert bounded.executed_count == capacity
    assert bounded.encoded_lanes == (
        capacity if ti.lang.impl.current_cfg().arch == ti.cpu else 24
    )
    assert bounded.skipped_count == 0
    assert bounded.overflow
    assert bounded.snapshot_status == "host_argument"
    assert graph._instance.structured_telemetry_arena_stats["reserved_bytes"] == 0
