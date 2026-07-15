import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.algorithms import _algorithms as alg_impl
from taichi_forge._lib import core as _ti_core
from tests import test_utils


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
    assert before["schema_version"] == 1
    assert before["backend"] in ("x64", "cuda", "vulkan")
    assert before["program_domain"] > 0
    assert before["memory"]["live_resources"] >= 1
    assert isinstance(before["memory"]["host_raw_bytes"], int)
    legacy_host = dict(_ti_core.get_host_memory_pool_stats())
    assert before["memory"]["host_raw_bytes"] == legacy_host["raw_bytes"]
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
    # The existing detailed report is opt-in. Enable it before both snapshots
    # so its counters and the always-on unified counters cover the same runs.
    graph.execution_stats()

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
    after = prog._runtime_statistics_snapshot()
    submission_delta = {
        key: after["submission"][key] - before["submission"][key]
        for key in after["submission"]
    }
    graph_delta = {
        key: after["graph"][key] - before["graph"][key]
        for key in after["graph"]
    }
    assert submission_delta["graph_submissions"] == run_count

    if arch == ti.cuda:
        replay_count = (
            segment.counters.exact_replays
            + segment.counters.patched_replays
        )
        assert graph_delta["captures"] == segment.counters.captures
        assert graph_delta["recaptures"] == segment.counters.recaptures
        assert graph_delta["replays"] == replay_count
        assert submission_delta["graph_backend_submissions"] == (
            segment.counters.captures + replay_count
        )
        assert graph_delta["ordinary_fallbacks"] == 0
    elif arch == ti.vulkan:
        assert graph_delta["captures"] == segment.counters.records
        assert graph_delta["replays"] == segment.counters.replays
        assert submission_delta["graph_backend_submissions"] == (
            segment.counters.records + segment.counters.replays
        )
        assert graph_delta["replay_slot_saturation_fallbacks"] == (
            segment.counters.replay_slot_saturation_fallbacks
        )
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
