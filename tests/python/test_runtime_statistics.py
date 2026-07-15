import taichi_forge as ti
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
