import threading

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.algorithms import _algorithms as alg_impl
from taichi_forge._lib import core as _ti_core
from tests import test_utils


@test_utils.test(arch=[ti.cpu])
def test_primitive_workspace_control_surface_is_exact_and_safe_to_clear():
    prog = ti.lang.impl.get_runtime().prog
    before = prog._primitive_workspace_stats()
    assert set(before) == {
        "budget_bytes",
        "reserved_bytes",
        "in_use_bytes",
        "persistent_bytes",
        "reclaimable_bytes",
        "over_budget_bytes",
        "peak_reserved_bytes",
        "peak_in_use_bytes",
        "entries",
        "active_leases",
        "acquisitions",
        "cache_hits",
        "cache_misses",
        "growth_events",
        "clear_calls",
        "cleared_entries",
        "trim_calls",
        "evictions",
        "lock_samples",
        "lock_contentions",
        "lock_wait_ns",
    }
    assert before["entries"] == 0
    assert before["reserved_bytes"] == 0

    prog._primitive_workspace_set_budget_bytes(8192)
    prog._primitive_workspace_clear()
    after = prog._primitive_workspace_stats()
    assert after["budget_bytes"] == 8192
    assert after["entries"] == 0
    assert after["active_leases"] == 0
    assert after["clear_calls"] == before["clear_calls"] + 1


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_runtime_statistics_kernel_completion_sync_and_memory():
    @ti.kernel
    def increment(value: ti.types.ndarray()):
        value[0] += 1

    value = ti.ndarray(ti.i32, shape=1)
    increment(value)
    ti.sync()

    prog = ti.lang.impl.get_runtime().prog
    before = prog._runtime_statistics_snapshot()
    assert before["schema_version"] == 3
    assert (
        before["submission"]["backend_graph_launches"]
        == before["submission"]["graph_backend_submissions"]
    )
    assert before["backend"] in ("x64", "cuda", "vulkan")
    assert before["program_domain"] > 0
    assert before["memory"]["live_resources"] >= 1
    assert isinstance(before["memory"]["host_raw_bytes"], int)
    legacy_host = dict(_ti_core.get_host_memory_pool_stats())
    assert before["memory"]["host_raw_bytes"] == legacy_host["raw_bytes"]
    host_allocator = before["memory"]["host_allocator"]
    assert host_allocator["requested_live_bytes"] == legacy_host[
        "requested_live_bytes"
    ]
    assert host_allocator["reserved_bytes"] == legacy_host["reserved_bytes"]
    assert host_allocator["capacity_bytes"] == legacy_host["capacity_bytes"]
    assert host_allocator["used_bytes"] == legacy_host["used_bytes"]
    assert host_allocator["available_bytes"] == legacy_host["available_bytes"]
    assert host_allocator["wasted_bytes"] == legacy_host["wasted_bytes"]
    assert host_allocator["chunk_count"] == legacy_host["unified_chunks"]
    assert (
        host_allocator["committed_bytes"]
        == legacy_host["committed_bytes"]
    )
    resource_snapshots = (
        prog._debug_argpack_resource_stats(),
        prog._debug_ndarray_resource_stats(),
        prog._debug_texture_resource_stats(),
        prog._debug_dense_field_staging_stats(),
    )
    assert before["memory"]["live_resources"] == sum(
        snapshot["live"] for snapshot in resource_snapshots
    )
    assert before["memory"]["retiring_resources"] == sum(
        snapshot["retiring"] for snapshot in resource_snapshots
    )
    if ti.lang.impl.current_cfg().arch in (ti.cpu, ti.cuda):
        assert isinstance(before["memory"]["device_raw_bytes"], int)
        legacy_device = dict(_ti_core.get_device_memory_pool_stats())
        assert before["memory"]["device_raw_bytes"] == legacy_device["raw_bytes"]
        assert (
            before["memory"]["device_cached_bytes"]
            == legacy_device["cached_bytes"]
        )
    else:
        assert before["memory"]["device_raw_bytes"] is None

    increment(value)
    after_kernel = prog._runtime_statistics_snapshot()
    assert (
        after_kernel["submission"]["kernel_submissions"]
        == before["submission"]["kernel_submissions"] + 1
    )

    completion = prog._record_runtime_completion()
    completion.done()
    completion.wait()
    after_completion = prog._runtime_statistics_snapshot()
    # GPU completion tracking may poll once internally before the explicit
    # public done() call. The unified counter intentionally reports both.
    assert (
        after_completion["synchronization"]["completion_polls"]
        >= after_kernel["synchronization"]["completion_polls"] + 1
    )
    assert (
        after_completion["synchronization"]["completion_waits"]
        == after_kernel["synchronization"]["completion_waits"] + 1
    )
    assert isinstance(
        after_completion["synchronization"]["completion_wait_ns"], int
    )

    ti.sync()
    after_sync = prog._runtime_statistics_snapshot()
    assert (
        after_sync["synchronization"]["program_syncs"]
        == after_completion["synchronization"]["program_syncs"] + 1
    )
    assert isinstance(after_sync["synchronization"]["program_sync_wait_ns"], int)
    assert after_sync["fault"]["state"] == "healthy"
    assert after_sync["fault"]["first_fault"] is None


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_runtime_statistics_backend_wait_and_lock_adapter():
    value = ti.field(dtype=ti.i32, shape=())

    @ti.kernel
    def increment():
        value[None] += 1

    # Keep compilation and first materialization outside the observation
    # window. The adapter reports runtime synchronization, not JIT setup.
    increment()
    ti.sync()
    value[None] = 0
    ti.sync()

    prog = ti.lang.impl.get_runtime().prog
    before = prog._runtime_statistics_snapshot()["synchronization"]
    arch = ti.lang.impl.current_cfg().arch
    optional_keys = (
        "backend_waits",
        "backend_wait_ns",
        "backend_lock_samples",
        "backend_lock_contentions",
        "backend_lock_sampled_wait_ns",
    )
    if arch == ti.cpu:
        assert all(before[key] is None for key in optional_keys)
        return

    assert all(isinstance(before[key], int) for key in optional_keys)
    legacy_before = None
    if arch == ti.cuda:
        legacy_before = {
            key: int(_ti_core.query_int64(key))
            for key in (
                "cuda_driver_lock_sampled_acquisitions",
                "cuda_driver_lock_contended_acquisitions",
                "cuda_context_lock_sampled_acquisitions",
                "cuda_context_lock_contended_acquisitions",
            )
        }

    # More than one sampling period guarantees at least one observed backend
    # lock without changing the 1/64 steady-state policy. Vulkan batches
    # ordinary kernels into one command list, so explicitly flush there to
    # exercise real queue acquisitions rather than counting kernel calls.
    iterations = 72
    for _ in range(iterations):
        increment()
        if arch == ti.vulkan:
            ti.sync()
    ti.sync()

    legacy_after = None
    if arch == ti.cuda:
        legacy_after = {
            key: int(_ti_core.query_int64(key)) for key in legacy_before
        }
    after = prog._runtime_statistics_snapshot()["synchronization"]
    assert after["backend_waits"] >= before["backend_waits"] + 1
    assert after["backend_wait_ns"] >= before["backend_wait_ns"]
    assert after["backend_lock_samples"] > before["backend_lock_samples"]
    assert (
        after["backend_lock_contentions"]
        >= before["backend_lock_contentions"]
    )
    assert (
        after["backend_lock_contentions"]
        <= after["backend_lock_samples"]
    )
    assert (
        after["backend_lock_sampled_wait_ns"]
        >= before["backend_lock_sampled_wait_ns"]
    )
    assert value[None] == iterations

    if arch == ti.cuda:
        legacy_sample_delta = (
            legacy_after["cuda_driver_lock_sampled_acquisitions"]
            - legacy_before["cuda_driver_lock_sampled_acquisitions"]
            + legacy_after["cuda_context_lock_sampled_acquisitions"]
            - legacy_before["cuda_context_lock_sampled_acquisitions"]
        )
        legacy_contention_delta = (
            legacy_after["cuda_driver_lock_contended_acquisitions"]
            - legacy_before["cuda_driver_lock_contended_acquisitions"]
            + legacy_after["cuda_context_lock_contended_acquisitions"]
            - legacy_before["cuda_context_lock_contended_acquisitions"]
        )
        # Unified CUDA telemetry additionally includes the submission mutex.
        assert (
            after["backend_lock_samples"] - before["backend_lock_samples"]
            >= legacy_sample_delta
        )
        assert (
            after["backend_lock_contentions"]
            - before["backend_lock_contentions"]
            >= legacy_contention_delta
        )


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
)
def test_runtime_statistics_graph_adapter_matches_execution_report():
    values = ti.field(dtype=ti.i32, shape=64)

    @ti.kernel
    def advance():
        for i in values:
            values[i] += i + 1

    @ti.kernel
    def scale():
        for i in values:
            values[i] *= 2

    builder = ti.graph.GraphBuilder()
    builder.dispatch(advance)
    builder.dispatch(scale)
    graph = builder.compile()
    # execution_stats() is a passive structural snapshot. Runtime replay
    # counters live in the always-on Program statistics unless a submission
    # explicitly requests ticket-owned telemetry.
    structural_before = graph.execution_stats()

    prog = ti.lang.impl.get_runtime().prog
    before = prog._runtime_statistics_snapshot()
    arch = ti.lang.impl.current_cfg().arch
    run_count = 9 if arch == ti.vulkan else 2
    for run_index in range(run_count):
        if arch == ti.vulkan and run_index == 8:
            ti.sync()
        graph.run({})
    ti.sync()

    report = graph.execution_stats()
    segment = report.segments[0]
    assert segment.counters == structural_before.segments[0].counters
    after = prog._runtime_statistics_snapshot()
    submission_delta = {
        key: after["submission"][key] - before["submission"][key]
        for key in after["submission"]
    }
    assert (
        submission_delta["backend_graph_launches"]
        == submission_delta["graph_backend_submissions"]
    )
    graph_delta = {
        key: after["graph"][key] - before["graph"][key]
        for key in after["graph"]
    }
    assert submission_delta["graph_submissions"] == run_count

    if arch == ti.cuda:
        assert graph_delta["captures"] >= 1
        assert graph_delta["recaptures"] == 0
        assert submission_delta["graph_backend_submissions"] == (
            graph_delta["captures"] + graph_delta["replays"]
        )
        assert graph_delta["ordinary_fallbacks"] == 0
    elif arch == ti.vulkan:
        assert graph_delta["captures"] >= 1
        assert submission_delta["graph_backend_submissions"] == (
            graph_delta["captures"] + graph_delta["replays"]
        )
        assert graph_delta["replay_slot_saturation_fallbacks"] == 0
        assert graph_delta["ordinary_fallbacks"] == 0
    else:
        assert report.execution_path == "ordinary"
        assert submission_delta["graph_backend_submissions"] == 0
        assert graph_delta["captures"] == 0
        assert graph_delta["recaptures"] == 0
        assert graph_delta["replays"] == 0
        assert graph_delta["ordinary_fallbacks"] == 0


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_runtime_statistics_native_adapter_matches_legacy_diagnostics():
    n = 64
    src_np = np.arange(n, dtype=np.int32)
    src = ti.ndarray(dtype=ti.i32, shape=n)
    dst = ti.ndarray(dtype=ti.i32, shape=n)
    src.from_numpy(src_np)
    method = {
        ti.cpu: "cpu_native",
        ti.cuda: "cuda_device",
        ti.vulkan: "vulkan_native",
    }[ti.lang.impl.current_cfg().arch]
    workspace = alg_impl.TransformWorkspace(max_items=n)
    prog = ti.lang.impl.get_runtime().prog

    # Prewarm the same plan that the legacy diagnostics observe. The unified
    # counter also covers this cold call, but the double-read window below is
    # intentionally identical for both sources.
    cold_before = prog._runtime_statistics_snapshot()
    alg_impl.experimental_transform(
        src,
        dst,
        scale=3,
        bias=-2,
        method=method,
        workspace=workspace,
    )
    cold_after = prog._runtime_statistics_snapshot()
    assert (
        cold_after["submission"]["native_submissions"]
        == cold_before["submission"]["native_submissions"] + 1
    )
    alg_impl.set_primitive_diagnostics_enabled(True, clear=True)
    try:
        before = prog._runtime_statistics_snapshot()
        workspace._native_transform_plan.invoke(prog)
        middle = prog._runtime_statistics_snapshot()
        first_diagnostics = alg_impl.get_primitive_diagnostics()
        assert (
            middle["submission"]["native_submissions"]
            - before["submission"]["native_submissions"]
            == first_diagnostics["native_plan.invoke.calls"]
            == 1
        )

        workspace._native_transform_plan.invoke(prog)
        after = prog._runtime_statistics_snapshot()
        diagnostics = alg_impl.get_primitive_diagnostics()
        assert (
            after["submission"]["native_submissions"]
            - before["submission"]["native_submissions"]
            == diagnostics["native_plan.invoke.calls"]
            == 2
        )
        assert np.array_equal(dst.to_numpy(), src_np * 3 - 2)
    finally:
        alg_impl.set_primitive_diagnostics_enabled(False, clear=True)

    method_name = {
        ti.cpu: "cpu_transform_affine_ndarray",
        ti.cuda: "cuda_device_transform_affine_ndarray",
        ti.vulkan: "vulkan_transform_affine_ndarray",
    }[ti.lang.impl.current_cfg().arch]
    before_failure = prog._runtime_statistics_snapshot()
    with pytest.raises(RuntimeError):
        getattr(prog, method_name)(None, None, 0, 1.0, 0.0)
    after_failure = prog._runtime_statistics_snapshot()
    assert (
        after_failure["submission"]["native_submissions"]
        == before_failure["submission"]["native_submissions"]
    )
    assert (
        after_failure["submission"]["failed_submissions"]
        == before_failure["submission"]["failed_submissions"] + 1
    )


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_primitive_runtime_diagnostics_explain_cached_provider_without_sync():
    n = 64
    src_np = np.arange(n, dtype=np.int32)
    src = ti.ndarray(ti.i32, shape=n)
    dst = ti.ndarray(ti.i32, shape=n)
    sort_keys = ti.ndarray(ti.i32, shape=n)
    sort_values = ti.ndarray(ti.i32, shape=n)
    src.from_numpy(src_np)
    sort_keys_np = ((src_np * 17) % 41 - 20).astype(np.int32)
    sort_values_np = np.arange(n, dtype=np.int32)
    sort_keys.from_numpy(sort_keys_np)
    sort_values.from_numpy(sort_values_np)
    workspace = alg_impl.TransformWorkspace(max_items=n)
    sort_workspace = alg_impl.SortWorkspace(max_items=n)
    expected_provider = {
        ti.cpu: "cpu_transform_affine_ndarray",
        ti.cuda: "cuda_device_transform_affine_ndarray",
        ti.vulkan: "vulkan_transform_affine_ndarray_trusted",
    }[ti.lang.impl.current_cfg().arch]
    expected_dependency = {
        ti.cpu: "none",
        ti.cuda: "driver_only",
        ti.vulkan: "none",
    }[ti.lang.impl.current_cfg().arch]
    expected_sort_provider = {
        ti.cpu: "cpu_stable_sort_ndarray",
        ti.cuda: "cuda_device_radix_sort_ndarray",
        ti.vulkan: "vulkan_radix_sort_u32_ndarray",
    }[ti.lang.impl.current_cfg().arch]

    # Prewarm and record the native replay plan before opening the diagnostic
    # interval. This keeps the assertion independent of cold descriptor probes.
    alg_impl.experimental_transform(
        src, dst, scale=5, bias=-3, method="auto", workspace=workspace
    )
    alg_impl.set_primitive_diagnostics_enabled(True, clear=True)
    try:
        prog = ti.lang.impl.get_runtime().prog
        before_sync = prog._runtime_statistics_snapshot()["synchronization"]
        alg_impl.experimental_transform(
            src, dst, scale=5, bias=-3, method="auto", workspace=workspace
        )
        alg_impl.sort(
            sort_keys,
            sort_values,
            method="auto",
            workspace=sort_workspace,
        )
        diagnostics = alg_impl.get_primitive_runtime_diagnostics()
        after_sync = prog._runtime_statistics_snapshot()["synchronization"]
    finally:
        alg_impl.set_primitive_diagnostics_enabled(False, clear=True)

    providers = {item["provider"]: item for item in diagnostics["providers"]}
    assert diagnostics["schema_version"] == 1
    assert diagnostics["enabled"] is True
    assert providers[expected_provider]["dependency_class"] == expected_dependency
    assert providers[expected_provider]["count"] == 1
    assert (
        providers[expected_sort_provider]["dependency_class"]
        == expected_dependency
    )
    assert providers[expected_sort_provider]["count"] == 1
    assert diagnostics["fallbacks"] == ()
    assert diagnostics["workspace"]["schema_version"] == 1
    assert (
        diagnostics["workspace"]["default_cache"]["ownership"]
        == "per_python_thread"
    )
    workspace_statistics = diagnostics["workspace"]
    assert workspace_statistics["program_provider_bytes_total"] == sum(
        workspace_statistics["program_provider_bytes"].values()
    )
    if ti.lang.impl.current_cfg().arch == ti.cuda:
        assert workspace_statistics["program_provider_aliases"] == {
            "cuda_cub_check_count": "cuda_device_check_count",
            "cuda_cub_histogram": "cuda_device_histogram",
            "cuda_cub_metric_reduce": "cuda_device_metric_reduce",
            "cuda_cub_radix_sort": "cuda_device_radix_sort",
            "cuda_cub_reduce": "cuda_device_reduce",
            "cuda_cub_scan": "cuda_device_scan",
            "cuda_cub_select": "cuda_device_compact",
        }
        assert not any(
            name.startswith("cuda_cub_")
            for name in workspace_statistics["program_provider_bytes"]
        )
    assert after_sync["program_syncs"] == before_sync["program_syncs"]
    assert after_sync["completion_waits"] == before_sync["completion_waits"]
    ti.sync()
    np.testing.assert_array_equal(dst.to_numpy(), src_np * 5 - 3)
    order = np.argsort(sort_keys_np, kind="stable")
    np.testing.assert_array_equal(sort_keys.to_numpy(), sort_keys_np[order])
    np.testing.assert_array_equal(sort_values.to_numpy(), sort_values_np[order])


def test_primitive_diagnostic_counter_is_exact_under_host_concurrency():
    thread_count = 4
    iterations = 2000

    def record_many():
        for _ in range(iterations):
            alg_impl._record_primitive_diagnostic("test.concurrent")

    alg_impl.set_primitive_diagnostics_enabled(True, clear=True)
    try:
        threads = [
            threading.Thread(target=record_many) for _ in range(thread_count)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        assert all(not thread.is_alive() for thread in threads)
        assert alg_impl.get_primitive_diagnostics()["test.concurrent"] == (
            thread_count * iterations
        )
    finally:
        alg_impl.set_primitive_diagnostics_enabled(False, clear=True)


@test_utils.test(arch=ti.cpu)
def test_primitive_runtime_diagnostics_reports_legacy_fallback_route():
    alg_impl.set_primitive_diagnostics_enabled(True, clear=True)
    try:
        alg_impl._record_legacy_helper_fallback(
            "test_fallback()", "auto", "kernel"
        )
        diagnostics = alg_impl.get_primitive_runtime_diagnostics()
    finally:
        alg_impl.set_primitive_diagnostics_enabled(False, clear=True)
    assert diagnostics["fallbacks"] == (
        {
            "operation": "test_fallback()",
            "requested_method": "auto",
            "selected_fallback": "kernel",
            "count": 1,
        },
    )


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_runtime_statistics_bulk_transfer_bytes_are_exact():
    n = 128
    payload_bytes = n * np.dtype(np.int32).itemsize
    src_np = np.arange(n, dtype=np.int32)
    ndarray_src = ti.ndarray(dtype=ti.i32, shape=n)
    ndarray_dst = ti.ndarray(dtype=ti.i32, shape=n)
    field_src = ti.field(dtype=ti.i32, shape=n)
    field_dst = ti.field(dtype=ti.i32, shape=n)
    prog = ti.lang.impl.get_runtime().prog
    before = prog._runtime_statistics_snapshot()

    ndarray_src.from_numpy(src_np)
    prog.copy_ndarray(ndarray_dst.arr, ndarray_src.arr)
    ndarray_result = ndarray_dst.to_numpy()
    field_src.from_numpy(src_np)
    field_dst.copy_from(field_src)
    field_result = field_dst.to_numpy()

    after = prog._runtime_statistics_snapshot()
    assert (
        after["transfer"]["host_to_device_bytes"]
        - before["transfer"]["host_to_device_bytes"]
        == 2 * payload_bytes
    )
    assert (
        after["transfer"]["device_to_host_bytes"]
        - before["transfer"]["device_to_host_bytes"]
        == 2 * payload_bytes
    )
    assert (
        after["transfer"]["device_to_device_bytes"]
        - before["transfer"]["device_to_device_bytes"]
        == 2 * payload_bytes
    )
    assert np.array_equal(ndarray_result, src_np)
    assert np.array_equal(field_result, src_np)
