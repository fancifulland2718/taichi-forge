import numpy as np

import taichi_forge as ti
from taichi_forge._lib import core as ti_core
from taichi_forge.lang import impl
from tests import test_utils


@ti.kernel
def _publish_extent_kernel(
    state: ti.types.ndarray(dtype=ti.i32, ndim=1),
    capacity: ti.i32,
    value: ti.i32,
):
    ti.device_extent_publish(state, capacity, value)


def _publish_extent(extent, count):
    _publish_extent_kernel(extent.state, extent.capacity, count)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_device_prefix_compact_scan_reduce_without_host_count():
    capacity = 64
    count = 19
    values_host = np.arange(capacity, dtype=np.int32) + 1
    flags_host = ((np.arange(capacity) % 3) != 1).astype(np.int32)
    values = ti.ndarray(ti.i32, shape=capacity)
    flags = ti.ndarray(ti.i32, shape=capacity)
    compacted = ti.ndarray(ti.i32, shape=capacity)
    scanned = ti.ndarray(ti.i32, shape=capacity)
    reduced = ti.ndarray(ti.i32, shape=1)
    extent = ti.DeviceExtent(capacity)
    compact_extent = ti.DeviceExtent(capacity)
    workspace = ti.algorithms.DevicePrefixWorkspace(capacity)
    values.from_numpy(values_host)
    flags.from_numpy(flags_host)
    _publish_extent(extent, count)

    program = impl.get_runtime().prog
    before = program._runtime_statistics_snapshot()
    compact_prefix = ti.algorithms.device_prefix(
        values, extent, workspace=workspace
    ).compact(flags, compacted, compact_extent)
    compact_prefix.scan(scanned)
    compact_prefix.reduce(reduced)
    after_enqueue = program._runtime_statistics_snapshot()

    assert after_enqueue["transfer"] == before["transfer"]
    for counter in ("program_syncs", "completion_waits"):
        assert (
            after_enqueue["synchronization"][counter]
            == before["synchronization"][counter]
        )
    expected = values_host[:count][flags_host[:count] != 0]
    assert compact_extent.check() == expected.size
    np.testing.assert_array_equal(compacted.to_numpy()[: expected.size], expected)
    np.testing.assert_array_equal(scanned.to_numpy()[: expected.size], np.cumsum(expected))
    assert int(reduced.to_numpy()[0]) == int(expected.sum())


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_device_prefix_sort_unique_rle_chain():
    capacity = 48
    count = 15
    keys_host = np.array(
        [7, 2, 7, 1, 9, 2, 2, 5, 1, 8, 5, 3, 3, 7, 4]
        + [123] * (capacity - count),
        dtype=np.int32,
    )
    keys = ti.ndarray(ti.i32, shape=capacity)
    unique = ti.ndarray(ti.i32, shape=capacity)
    rle_keys = ti.ndarray(ti.i32, shape=capacity)
    run_lengths = ti.ndarray(ti.i32, shape=capacity)
    extent = ti.DeviceExtent(capacity)
    unique_extent = ti.DeviceExtent(capacity)
    rle_extent = ti.DeviceExtent(capacity)
    workspace = ti.algorithms.DevicePrefixWorkspace(capacity)
    keys.from_numpy(keys_host)
    _publish_extent(extent, count)

    prefix = ti.algorithms.device_prefix(keys, extent, workspace=workspace)
    prefix.sort()
    prefix.unique(unique, unique_extent)
    prefix.run_length_encode(rle_keys, run_lengths, rle_extent)
    ti.sync()

    expected_keys, expected_lengths = np.unique(
        keys_host[:count], return_counts=True
    )
    assert unique_extent.check() == expected_keys.size
    assert rle_extent.check() == expected_keys.size
    np.testing.assert_array_equal(unique.to_numpy()[: expected_keys.size], expected_keys)
    np.testing.assert_array_equal(rle_keys.to_numpy()[: expected_keys.size], expected_keys)
    np.testing.assert_array_equal(
        run_lengths.to_numpy()[: expected_keys.size], expected_lengths.astype(np.int32)
    )


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_device_prefix_grouped_reduce_and_bucket_builder():
    capacity = 40
    count = 23
    groups = 5
    keys_host = (np.arange(capacity, dtype=np.int32) * 7 + 2) % groups
    values_host = np.arange(capacity, dtype=np.int32) + 3
    keys = ti.ndarray(ti.i32, shape=capacity)
    values = ti.ndarray(ti.i32, shape=capacity)
    grouped = ti.ndarray(ti.i32, shape=groups)
    offsets = ti.ndarray(ti.i32, shape=groups + 1)
    bucketed = ti.ndarray(ti.i32, shape=capacity)
    extent = ti.DeviceExtent(capacity)
    workspace = ti.algorithms.DevicePrefixWorkspace(capacity)
    keys.from_numpy(keys_host)
    values.from_numpy(values_host)
    _publish_extent(extent, count)

    prefix = ti.algorithms.device_prefix(values, extent, workspace=workspace)
    prefix.grouped_reduce(keys, grouped)
    bucket_prefix = prefix.bucket_builder(keys, offsets, bucketed)
    ti.sync()

    expected_grouped = np.bincount(
        keys_host[:count], weights=values_host[:count], minlength=groups
    ).astype(np.int32)
    np.testing.assert_array_equal(grouped.to_numpy(), expected_grouped)
    expected_order = np.argsort(keys_host[:count], kind="stable")
    np.testing.assert_array_equal(
        bucketed.to_numpy()[:count], values_host[:count][expected_order]
    )
    np.testing.assert_array_equal(
        offsets.to_numpy(),
        np.concatenate(
            ([0], np.cumsum(np.bincount(keys_host[:count], minlength=groups)))
        ).astype(np.int32),
    )
    assert bucket_prefix.extent is extent


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_device_prefix_workspace_reuses_allocations_across_count_churn():
    capacity = 32
    values = ti.ndarray(ti.i32, shape=capacity)
    flags = ti.ndarray(ti.i32, shape=capacity)
    output = ti.ndarray(ti.i32, shape=capacity)
    extent = ti.DeviceExtent(capacity)
    output_extent = ti.DeviceExtent(capacity)
    workspace = ti.algorithms.DevicePrefixWorkspace(capacity)
    values.from_numpy(np.arange(capacity, dtype=np.int32))
    flags.from_numpy(np.ones(capacity, dtype=np.int32))
    prefix = ti.algorithms.device_prefix(values, extent, workspace=workspace)

    _publish_extent(extent, 1)
    prefix.compact(flags, output, output_extent)
    ti.sync()
    allocation_count = workspace.allocation_count
    workspace_bytes = workspace.workspace_bytes_current
    memory_before = impl.get_runtime().prog._runtime_statistics_snapshot()["memory"]
    host_before = dict(ti_core.get_host_memory_pool_stats())
    device_before = dict(ti_core.get_device_memory_pool_stats())

    for count in range(1, capacity + 1):
        _publish_extent(extent, count)
        prefix.compact(flags, output, output_extent)
    ti.sync()

    assert workspace.allocation_count == allocation_count
    assert workspace.workspace_bytes_current == workspace_bytes
    assert impl.get_runtime().prog._runtime_statistics_snapshot()["memory"] == memory_before
    assert dict(ti_core.get_host_memory_pool_stats()) == host_before
    assert dict(ti_core.get_device_memory_pool_stats()) == device_before
    assert output_extent.check() == capacity
