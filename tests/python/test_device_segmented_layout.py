import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.lang.exception import TaichiRuntimeError
from tests import test_utils


_ROWS = 5
_MAX_CONTACTS = 4
_NUM_CONTACTS = 8
_CAPACITY = 10
_CONTACT_SLOTS = np.asarray(
    [
        [1, 4, -1, -1],
        [-1, -1, -1, -1],
        [0, 3, 4, -1],
        [2, -1, -1, -1],
        [0, 2, -1, -1],
    ],
    dtype=np.int32,
)
_COUNTS = np.asarray([2, 0, 3, 1, 2], dtype=np.int32)
_OFFSETS = np.asarray([0, 2, 2, 5, 6, 8], dtype=np.int32)
_COLUMNS = np.asarray([1, 4, 0, 3, 4, 2, 0, 2], dtype=np.int32)


def _stage_contact_candidates():
    source = ti.ndarray(ti.i32, shape=(_ROWS, _MAX_CONTACTS))
    staged = ti.ndarray(ti.i32, shape=(_ROWS, _MAX_CONTACTS))
    counts = ti.ndarray(ti.i32, shape=_ROWS)
    source.from_numpy(_CONTACT_SLOTS)

    @ti.kernel
    def stage(
        source_arr: ti.types.ndarray(dtype=ti.i32, ndim=2),
        staged_arr: ti.types.ndarray(dtype=ti.i32, ndim=2),
        count_arr: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for row in range(_ROWS):
            count = 0
            for slot in ti.static(range(_MAX_CONTACTS)):
                column = source_arr[row, slot]
                staged_arr[row, slot] = column
                if column >= 0:
                    count += 1
            count_arr[row] = count

    stage(source, staged, counts)
    return source, staged, counts


def _make_layout(counts):
    return ti.algorithms.SegmentedLayout._from_device_counts(
        counts,
        num_items=_NUM_CONTACTS,
        max_segment_length=_MAX_CONTACTS,
        capacity=_CAPACITY,
    )


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
    vulkan_sparse_experimental=True,
    cuda_sparse_pool_auto_size=True,
    cuda_sparse_per_snode_pool=True,
)
def test_device_count_layout_matches_contact_csr_and_dynamic_snode():
    _source, staged, counts = _stage_contact_candidates()
    layout = _make_layout(counts)
    columns = ti.ndarray(ti.i32, shape=_CAPACITY)
    weights = ti.ndarray(ti.i32, shape=_CAPACITY)
    row_sums = ti.ndarray(ti.i32, shape=_ROWS)
    scanned = ti.ndarray(ti.i32, shape=_CAPACITY)
    columns.fill(-1)
    weights.fill(777)
    scanned.fill(-999)

    @ti.kernel
    def fill_contact_csr(
        staged_arr: ti.types.ndarray(dtype=ti.i32, ndim=2),
        count_arr: ti.types.ndarray(dtype=ti.i32, ndim=1),
        offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output_columns: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output_weights: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for row in range(_ROWS):
            begin = offsets[row]
            for local_index in range(count_arr[row]):
                column = staged_arr[row, local_index]
                output_columns[begin + local_index] = column
                output_weights[begin + local_index] = (row + 1) * 10 + column

    fill_contact_csr(
        staged,
        counts,
        layout._offsets,
        columns,
        weights,
    )
    ti.algorithms.experimental_segmented_reduce(weights, layout, row_sums)
    ti.algorithms.experimental_segmented_scan(
        weights, layout, scanned, inclusive=False
    )

    expected_columns = np.full(_CAPACITY, -1, dtype=np.int32)
    expected_columns[:_NUM_CONTACTS] = _COLUMNS
    expected_segment_ids = np.full(_CAPACITY, -1, dtype=np.int32)
    expected_segment_ids[:_NUM_CONTACTS] = np.repeat(
        np.arange(_ROWS, dtype=np.int32), _COUNTS
    )
    expected_row_sums = np.asarray(
        [
            sum((row + 1) * 10 + int(column) for column in _CONTACT_SLOTS[row] if column >= 0)
            for row in range(_ROWS)
        ],
        dtype=np.int32,
    )
    expected_scanned = np.full(_CAPACITY, -999, dtype=np.int32)
    for row in range(_ROWS):
        running = 0
        for index in range(_OFFSETS[row], _OFFSETS[row + 1]):
            expected_scanned[index] = running
            running += (row + 1) * 10 + int(expected_columns[index])
    np.testing.assert_array_equal(counts.to_numpy(), _COUNTS)
    np.testing.assert_array_equal(layout._offsets.to_numpy(), _OFFSETS)
    np.testing.assert_array_equal(
        layout._segment_ids.to_numpy(), expected_segment_ids
    )
    np.testing.assert_array_equal(columns.to_numpy(), expected_columns)
    np.testing.assert_array_equal(row_sums.to_numpy(), expected_row_sums)
    np.testing.assert_array_equal(scanned.to_numpy(), expected_scanned)

    assert layout.encoding == "device_counts"
    assert layout.num_items == _NUM_CONTACTS
    assert layout.capacity == _CAPACITY
    assert layout.num_segments == _ROWS
    assert layout.max_segment_length == 3
    assert layout.topology_bytes == 64
    stats = layout._debug_device_generation_stats()
    assert stats["schema_version"] == 1
    assert stats["generation"] == {
        "encoding": "device_counts",
        "num_segments": _ROWS,
        "num_items": _NUM_CONTACTS,
        "capacity": _CAPACITY,
        "max_segment_length": 3,
    }
    resources = stats["resources"]
    assert resources["borrowed_counts_bytes"] == 20
    assert resources["owned_offsets_bytes"] == 24
    assert resources["owned_segment_ids_bytes"] == 40
    assert resources["owned_topology_bytes"] == 64
    assert resources["construction_control_bytes"] == 12
    assert (
        resources[
            "construction_explicit_array_peak_bytes_excluding_shared_workspace"
        ]
        == 76
    )
    assert resources["device_to_host_control_bytes"] == 12
    assert resources["device_to_host_payload_bytes"] == 0
    assert resources["device_kernel_published_topology_bytes"] == 64
    shared_scan_bytes = resources["shared_scan_workspace_bytes_at_publish"]
    assert shared_scan_bytes is None or shared_scan_bytes >= 0
    contract = stats["contract"]
    assert contract["counts_ownership"] == "borrowed_during_build"
    assert not contract["counts_retained_after_build"]
    assert contract["topology_ownership"] == "layout_generation"
    assert contract["construction_sync_count"] == 1
    assert not contract["payload_device_to_host"]
    assert not contract["failure_publishes_partial_layout"]
    assert contract["row_local_order"] == "caller_fill_order"
    assert (
        contract["shared_scan_workspace_ownership_scope"]
        == "program_scan_arena"
    )
    assert not contract["shared_scan_workspace_in_explicit_capacity"]
    assert contract["explicit_array_bytes_are_logical_payload"]
    assert not contract["runtime_allocator_overhead_reported"]
    assert not contract["total_owned_bytes_reported"]

    dynamic_values = ti.field(ti.i32)
    builder = ti.FieldsBuilder()
    dynamic = builder.dense(ti.i, _ROWS).dynamic(
        ti.j, _MAX_CONTACTS, chunk_size=_MAX_CONTACTS
    )
    dynamic.place(dynamic_values)
    tree = builder.finalize()
    dynamic_lengths = ti.ndarray(ti.i32, shape=_ROWS)
    dynamic_payload = ti.ndarray(ti.i32, shape=(_ROWS, _MAX_CONTACTS))
    inactive_reads = ti.ndarray(ti.i32, shape=(_ROWS, _MAX_CONTACTS))
    dynamic_payload.fill(-1)

    @ti.kernel
    def append_dynamic_rows(
        staged_arr: ti.types.ndarray(dtype=ti.i32, ndim=2),
        count_arr: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for row in range(_ROWS):
            for local_index in range(count_arr[row]):
                ti.append(dynamic, row, staged_arr[row, local_index])

    @ti.kernel
    def snapshot_dynamic_rows(
        lengths: ti.types.ndarray(dtype=ti.i32, ndim=1),
        payload: ti.types.ndarray(dtype=ti.i32, ndim=2),
    ):
        for row in range(_ROWS):
            length = ti.length(dynamic, row)
            lengths[row] = length
            for local_index in range(length):
                payload[row, local_index] = dynamic_values[row, local_index]

    @ti.kernel
    def read_all_dynamic_slots(
        output: ti.types.ndarray(dtype=ti.i32, ndim=2),
    ):
        for row, local_index in ti.ndrange(_ROWS, _MAX_CONTACTS):
            output[row, local_index] = dynamic_values[row, local_index]

    append_dynamic_rows(staged, counts)
    snapshot_dynamic_rows(dynamic_lengths, dynamic_payload)
    np.testing.assert_array_equal(dynamic_lengths.to_numpy(), _COUNTS)
    np.testing.assert_array_equal(dynamic_payload.to_numpy(), _CONTACT_SLOTS)

    dynamic.deactivate_all()
    dynamic_payload.fill(-1)
    snapshot_dynamic_rows(dynamic_lengths, dynamic_payload)
    read_all_dynamic_slots(inactive_reads)
    np.testing.assert_array_equal(
        dynamic_lengths.to_numpy(), np.zeros(_ROWS, dtype=np.int32)
    )
    assert np.all(dynamic_payload.to_numpy() == -1)
    assert np.all(inactive_reads.to_numpy() == 0)
    tree.destroy()


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
)
def test_device_count_layout_failure_preserves_published_generation():
    counts = ti.ndarray(ti.i32, shape=_ROWS)
    counts.from_numpy(_COUNTS)
    published = _make_layout(counts)
    values = ti.ndarray(ti.i32, shape=_CAPACITY)
    reduced = ti.ndarray(ti.i32, shape=_ROWS)
    values.from_numpy(np.arange(_CAPACITY, dtype=np.int32))

    counts.from_numpy(np.asarray([2, 0, 2, 1, 2], dtype=np.int32))
    with pytest.raises(
        TaichiRuntimeError,
        match="summed to 7, expected exact num_items=8.*no layout was published",
    ):
        _make_layout(counts)

    counts.from_numpy(np.asarray([2, 0, 4, 0, 2], dtype=np.int32))
    with pytest.raises(
        TaichiRuntimeError,
        match="device counts must satisfy.*no layout was published",
    ):
        ti.algorithms.SegmentedLayout._from_device_counts(
            counts,
            num_items=_NUM_CONTACTS,
            max_segment_length=3,
            capacity=_CAPACITY,
        )
    with pytest.raises(
        ValueError,
        match=r"num_segments \* max_segment_length.*prefix sums cannot overflow",
    ):
        ti.algorithms.SegmentedLayout._from_device_counts(
            counts,
            num_items=_NUM_CONTACTS,
            max_segment_length=0x40000000,
            capacity=_CAPACITY,
        )

    ti.algorithms.experimental_segmented_reduce(values, published, reduced)
    expected = np.asarray(
        [
            np.sum(np.arange(_OFFSETS[row], _OFFSETS[row + 1]))
            for row in range(_ROWS)
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(published._offsets.to_numpy(), _OFFSETS)
    np.testing.assert_array_equal(reduced.to_numpy(), expected)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_device_count_layout_rejects_runtime_rebind():
    counts = ti.ndarray(ti.i32, shape=_ROWS)
    counts.from_numpy(_COUNTS)
    layout = _make_layout(counts)

    ti.reset()
    ti.init(arch=ti.cpu, offline_cache=False)
    values = ti.ndarray(ti.i32, shape=_CAPACITY)
    reduced = ti.ndarray(ti.i32, shape=_ROWS)
    with pytest.raises(
        TaichiRuntimeError,
        match="SegmentedLayout cannot be used after its Taichi runtime has been reset",
    ):
        layout._debug_device_generation_stats()
    with pytest.raises(
        TaichiRuntimeError,
        match="SegmentedLayout cannot be used after its Taichi runtime has been reset",
    ):
        ti.algorithms.experimental_segmented_reduce(values, layout, reduced)
