import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.algorithms._sorted_key_buckets import (
    _SortedKeyBucketBuilder,
)
from taichi_forge.lang.exception import TaichiRuntimeError
from tests import test_utils


_CAPACITY = 16
_NUM_ITEMS = 12
_LOGICAL_KEY_LIMIT = 16
_MAX_ITEMS_PER_BUCKET = 4
_KEYS_A = np.asarray(
    [5, 2, 5, 9, 2, 5, 7, 9, 2, 7, 7, 7],
    dtype=np.int32,
)
_KEYS_B = np.asarray(
    [1, 8, 1, 4, 8, 4, 1, 12, 4, 8, 12, 12],
    dtype=np.int32,
)


def _device_source(keys_np):
    source = ti.ndarray(ti.i32, shape=_CAPACITY)
    keys = ti.ndarray(ti.i32, shape=_CAPACITY)
    values = ti.ndarray(ti.i32, shape=_CAPACITY)
    source_host = np.full(_CAPACITY, -77, dtype=np.int32)
    source_host[: keys_np.size] = keys_np
    source.from_numpy(source_host)

    @ti.kernel
    def produce(
        source_arr: ti.types.ndarray(dtype=ti.i32, ndim=1),
        key_arr: ti.types.ndarray(dtype=ti.i32, ndim=1),
        value_arr: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for index in range(_CAPACITY):
            key_arr[index] = source_arr[index]
            value_arr[index] = index

    produce(source, keys, values)
    return source, keys, values


def _expected_generation(keys_np):
    order = np.argsort(keys_np, kind="stable")
    sorted_keys = keys_np[order]
    unique_keys, counts = np.unique(sorted_keys, return_counts=True)
    offsets = np.zeros(unique_keys.size + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(counts, dtype=np.int64).astype(np.int32)
    values = np.full(_CAPACITY, -1, dtype=np.int32)
    values[: keys_np.size] = order.astype(np.int32)
    return (
        unique_keys.astype(np.int32),
        offsets,
        values,
        counts.astype(np.int32),
    )


def _assert_snapshot(snapshot, keys_np):
    unique_keys, offsets, values, counts = _expected_generation(keys_np)
    np.testing.assert_array_equal(
        snapshot._unique_keys.to_numpy()[: snapshot.num_buckets],
        unique_keys,
    )
    np.testing.assert_array_equal(snapshot._offsets.to_numpy(), offsets)
    np.testing.assert_array_equal(snapshot._values.to_numpy(), values)
    assert snapshot.num_items == keys_np.size
    assert snapshot.num_buckets == unique_keys.size
    assert snapshot.max_bucket_size == int(np.max(counts, initial=0))


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
    hash_snode_experimental=True,
    vulkan_sparse_experimental=True,
    vulkan_listgen_dynamic_size=True,
    cuda_sparse_pool_auto_size=True,
    cuda_sparse_per_snode_pool=True,
)
def test_sorted_key_buckets_match_particle_cell_hash_occupancy():
    _source, source_keys, source_values = _device_source(_KEYS_A)
    bucket_builder = _SortedKeyBucketBuilder(
        capacity=_CAPACITY,
        logical_key_limit=_LOGICAL_KEY_LIMIT,
        max_items_per_bucket=_MAX_ITEMS_PER_BUCKET,
    )
    snapshot = bucket_builder.build(
        source_keys, source_values, num_items=_NUM_ITEMS
    )
    _assert_snapshot(snapshot, _KEYS_A)

    stats = snapshot.debug_runtime_stats()
    assert stats["schema_version"] == 1
    assert stats["identity"] == {
        "backend_family": {
            ti.cpu: "cpu",
            ti.cuda: "cuda",
            ti.vulkan: "vulkan",
        }[ti.lang.impl.current_cfg().arch],
        "generation": 1,
        "capacity": _CAPACITY,
        "num_items": _NUM_ITEMS,
        "logical_key_limit": _LOGICAL_KEY_LIMIT,
        "num_buckets": 4,
        "max_bucket_size": 4,
        "ordering": "key_then_stable_source_ordinal",
    }
    resources = stats["resources"]
    assert resources["borrowed_source_payload_bytes"] == 128
    assert resources["unique_keys_reserved_payload_bytes"] == 16
    assert resources["offsets_reserved_payload_bytes"] == 20
    assert resources["sorted_values_reserved_payload_bytes"] == 64
    assert resources["generation_reserved_payload_bytes"] == 100
    assert resources["generation_active_payload_bytes"] == 84
    assert resources["builder_core_staging_payload_bytes"] == 276
    assert resources["builder_rle_scratch_payload_bytes"] == 192
    assert (
        resources["builder_persistent_explicit_array_payload_bytes"] == 468
    )
    assert (
        resources["candidate_build_explicit_array_peak_payload_bytes"] == 568
    )
    assert resources["live_prior_generation_payload_bytes_at_build"] == 0
    assert (
        resources["build_peak_with_live_generations_payload_bytes"] == 568
    )
    assert (
        resources[
            "build_peak_with_live_generations_and_borrowed_source_payload_bytes"
        ]
        == 696
    )
    assert resources["sort_workspace_reported_bytes"] >= 0
    assert resources["rle_workspace_reported_bytes"] >= 192
    assert resources["shared_sort_scan_workspace_bytes_at_publish"] >= 0
    transfers = stats["transfers"]
    assert transfers["device_to_host_control_bytes"] == 20
    assert transfers["device_to_host_payload_bytes"] == 0
    assert transfers["host_observation_sync_count"] == 2
    assert transfers["device_kernel_generation_payload_bytes"] == 100
    contract = stats["contract"]
    assert contract["source_ownership"] == "borrowed_during_build"
    assert not contract["source_retained_after_build"]
    assert contract["generation_ownership"] == "snapshot"
    assert contract["stable_equal_key_source_order"]
    assert contract["invalid_key_policy"] == "transactional_failure"
    assert contract["overflow_policy"] == "transactional_failure"
    assert contract["empty_generation_supported"]
    assert not contract["shared_workspace_in_explicit_capacity"]
    assert not contract["workspace_total_bytes_reported"]
    assert contract["explicit_array_bytes_are_logical_payload"]
    assert not contract["runtime_allocator_overhead_reported"]
    assert not contract["total_owned_bytes_reported"]
    assert not contract["public_api"]

    hash_counts = ti.field(ti.i32)
    fields_builder = ti.FieldsBuilder()
    hash_node = fields_builder.hash(
        ti.i, _LOGICAL_KEY_LIMIT, expected_active=4
    )
    hash_node.place(hash_counts)
    tree = fields_builder.finalize()
    active_count = ti.ndarray(ti.i32, shape=1)
    active_keys = ti.ndarray(ti.i32, shape=8)
    active_values = ti.ndarray(ti.i32, shape=8)

    @ti.kernel
    def fill_hash_counts(
        keys: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for index in range(_NUM_ITEMS):
            ti.atomic_add(hash_counts[keys[index]], 1)

    @ti.kernel
    def collect_hash_counts(
        count: ti.types.ndarray(dtype=ti.i32, ndim=1),
        keys: ti.types.ndarray(dtype=ti.i32, ndim=1),
        values: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for key in hash_counts:
            output = ti.atomic_add(count[0], 1)
            keys[output] = key
            values[output] = hash_counts[key]

    @ti.kernel
    def sum_inactive_hash_reads() -> ti.i32:
        total = 0
        for key in range(_LOGICAL_KEY_LIMIT):
            total += hash_counts[key]
        return total

    active_count.fill(0)
    active_keys.fill(-1)
    active_values.fill(-1)
    fill_hash_counts(source_keys)
    collect_hash_counts(active_count, active_keys, active_values)
    count_host = int(active_count.to_numpy()[0])
    keys_host = active_keys.to_numpy()[:count_host]
    values_host = active_values.to_numpy()[:count_host]
    order = np.argsort(keys_host, kind="stable")
    expected_keys, _offsets, _values, expected_counts = (
        _expected_generation(_KEYS_A)
    )
    np.testing.assert_array_equal(keys_host[order], expected_keys)
    np.testing.assert_array_equal(values_host[order], expected_counts)

    hash_node.deactivate_all()
    active_count.fill(0)
    collect_hash_counts(active_count, active_keys, active_values)
    assert int(active_count.to_numpy()[0]) == 0
    assert sum_inactive_hash_reads() == 0
    tree.destroy()


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
)
def test_sorted_key_bucket_rebuild_failure_preserves_old_generation():
    _source, source_keys, source_values = _device_source(_KEYS_A)
    bucket_builder = _SortedKeyBucketBuilder(
        capacity=_CAPACITY,
        logical_key_limit=_LOGICAL_KEY_LIMIT,
        max_items_per_bucket=_MAX_ITEMS_PER_BUCKET,
    )
    old_snapshot = bucket_builder.build(
        source_keys, source_values, num_items=_NUM_ITEMS
    )

    invalid_keys = np.full(_CAPACITY, -1, dtype=np.int32)
    invalid_keys[:_NUM_ITEMS] = _KEYS_A
    invalid_keys[3] = _LOGICAL_KEY_LIMIT
    source_keys.from_numpy(invalid_keys)
    with pytest.raises(
        TaichiRuntimeError,
        match="source keys must satisfy.*no generation was published",
    ):
        bucket_builder.build(
            source_keys, source_values, num_items=_NUM_ITEMS
        )

    overflow_keys = np.full(_CAPACITY, -1, dtype=np.int32)
    overflow_keys[:_NUM_ITEMS] = np.asarray(
        [3, 3, 3, 3, 3, 1, 2, 4, 5, 6, 7, 8], dtype=np.int32
    )
    source_keys.from_numpy(overflow_keys)
    with pytest.raises(
        TaichiRuntimeError,
        match="run length exceeds max_items_per_bucket=4.*no generation was published",
    ):
        bucket_builder.build(
            source_keys, source_values, num_items=_NUM_ITEMS
        )

    replacement_keys = np.full(_CAPACITY, -1, dtype=np.int32)
    replacement_keys[:_NUM_ITEMS] = _KEYS_B
    source_keys.from_numpy(replacement_keys)
    new_snapshot = bucket_builder.build(
        source_keys, source_values, num_items=_NUM_ITEMS
    )
    _assert_snapshot(old_snapshot, _KEYS_A)
    _assert_snapshot(new_snapshot, _KEYS_B)
    assert old_snapshot.generation == 1
    assert new_snapshot.generation == 2

    empty_snapshot = bucket_builder.build(
        source_keys, source_values, num_items=0
    )
    assert empty_snapshot.generation == 3
    assert empty_snapshot.num_items == 0
    assert empty_snapshot.num_buckets == 0
    assert empty_snapshot.max_bucket_size == 0
    np.testing.assert_array_equal(
        empty_snapshot._unique_keys.to_numpy(),
        np.asarray([-1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        empty_snapshot._offsets.to_numpy(),
        np.asarray([0], dtype=np.int32),
    )
    assert np.all(empty_snapshot._values.to_numpy() == -1)

    builder_stats = bucket_builder.debug_runtime_stats()
    assert builder_stats["operations"] == {
        "build_attempts": 5,
        "published_generations": 3,
        "failed_builds": 2,
        "live_generations": 3,
    }
    assert builder_stats["resources"]["live_generation_payload_bytes"] == 272


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_sorted_key_buckets_reject_runtime_rebind():
    _source, source_keys, source_values = _device_source(_KEYS_A)
    bucket_builder = _SortedKeyBucketBuilder(
        capacity=_CAPACITY,
        logical_key_limit=_LOGICAL_KEY_LIMIT,
        max_items_per_bucket=_MAX_ITEMS_PER_BUCKET,
    )
    snapshot = bucket_builder.build(
        source_keys, source_values, num_items=_NUM_ITEMS
    )

    ti.reset()
    ti.init(arch=ti.cpu, offline_cache=False)
    with pytest.raises(
        TaichiRuntimeError,
        match="bucket builder cannot be used after.*runtime.*reset",
    ):
        bucket_builder.debug_runtime_stats()
    with pytest.raises(
        TaichiRuntimeError,
        match="bucket snapshot cannot be used after.*runtime.*reset",
    ):
        snapshot.debug_runtime_stats()
