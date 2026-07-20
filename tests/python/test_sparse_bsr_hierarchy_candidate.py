import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.linalg._sparse_bsr_graph_operator import (
    _DeviceBsrSnapshot,
)
from taichi_forge.linalg._sparse_bsr_hierarchy_candidate import (
    _CallerCoarsenedBsrHierarchyBuilder,
)
from tests import test_utils


_FINE_BLOCK_ROWS = 8
_BLOCK_SIZE = 3
_FIRST_MAP = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)
_SECOND_MAP = np.asarray([0, 0, 1, 1], dtype=np.int32)


def _i32_array(values):
    result = ti.ndarray(ti.i32, shape=len(values))
    result.from_numpy(np.asarray(values, dtype=np.int32))
    return result


def _f32_array(values):
    result = ti.ndarray(ti.f32, shape=len(values))
    result.from_numpy(np.asarray(values, dtype=np.float32))
    return result


def _irregular_block_spd_operator(block_size):
    matrix = np.zeros(
        (_FINE_BLOCK_ROWS * block_size, _FINE_BLOCK_ROWS * block_size),
        dtype=np.float32,
    )
    coordinates = np.arange(1, block_size + 1, dtype=np.float32)
    for node in range(_FINE_BLOCK_ROWS):
        begin = node * block_size
        direction = coordinates + 0.125 * node
        mass = np.diag(0.4 + 0.015 * coordinates)
        mass += 0.01 * np.outer(direction, direction) / block_size
        matrix[
            begin : begin + block_size, begin : begin + block_size
        ] += mass

    edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (0, 3),
        (1, 5),
        (2, 6),
        (4, 7),
    )
    for ordinal, (left, right) in enumerate(edges):
        direction = coordinates + 0.07 * (ordinal + 1)
        weight = np.diag(
            0.7 + 0.025 * coordinates + 0.01 * ordinal
        )
        weight += 0.035 * np.outer(direction, direction) / block_size
        left_begin = left * block_size
        right_begin = right * block_size
        left_slice = slice(left_begin, left_begin + block_size)
        right_slice = slice(right_begin, right_begin + block_size)
        matrix[left_slice, left_slice] += weight
        matrix[right_slice, right_slice] += weight
        matrix[left_slice, right_slice] -= weight
        matrix[right_slice, left_slice] -= weight
    return matrix


def _compress_dense_blocks(dense, block_rows, block_size):
    row_offsets = [0]
    columns = []
    values = []
    for block_row in range(block_rows):
        row_begin = block_row * block_size
        for block_column in range(block_rows):
            column_begin = block_column * block_size
            block = dense[
                row_begin : row_begin + block_size,
                column_begin : column_begin + block_size,
            ]
            if np.any(np.abs(block) > 1e-7):
                columns.append(block_column)
                values.extend(block.reshape(-1))
        row_offsets.append(len(columns))
    return (
        np.asarray(row_offsets, dtype=np.int32),
        np.asarray(columns, dtype=np.int32),
        np.asarray(values, dtype=np.float32),
    )


def _source():
    dense = _irregular_block_spd_operator(_BLOCK_SIZE)
    row_offsets, columns, values = _compress_dense_blocks(
        dense, _FINE_BLOCK_ROWS, _BLOCK_SIZE
    )
    assert len(columns) == 30
    source = _DeviceBsrSnapshot.copy_validated(
        block_rows=_FINE_BLOCK_ROWS,
        block_cols=_FINE_BLOCK_ROWS,
        block_size=_BLOCK_SIZE,
        row_offsets=_i32_array(row_offsets),
        column_indices=_i32_array(columns),
        values=_f32_array(values),
        topology_version=7,
        numeric_version=11,
    )
    return source, dense


def _prolongation(aggregate, coarse_block_rows):
    fine_block_rows = len(aggregate)
    result = np.zeros(
        (
            fine_block_rows * _BLOCK_SIZE,
            coarse_block_rows * _BLOCK_SIZE,
        ),
        dtype=np.float32,
    )
    identity = np.eye(_BLOCK_SIZE, dtype=np.float32)
    for fine_block_row, coarse_block_row in enumerate(aggregate):
        fine_begin = fine_block_row * _BLOCK_SIZE
        coarse_begin = int(coarse_block_row) * _BLOCK_SIZE
        result[
            fine_begin : fine_begin + _BLOCK_SIZE,
            coarse_begin : coarse_begin + _BLOCK_SIZE,
        ] = identity
    return result


def _snapshot_payload(snapshot):
    return (
        snapshot._row_offsets.to_numpy(),
        snapshot._column_indices.to_numpy(),
        snapshot._values.to_numpy(),
    )


def _snapshot_dense(snapshot):
    row_offsets, columns, values = _snapshot_payload(snapshot)
    dense = np.zeros((snapshot.rows, snapshot.cols), dtype=np.float32)
    for block_row in range(snapshot.block_rows):
        row_begin = block_row * snapshot.block_size
        for offset in range(
            int(row_offsets[block_row]), int(row_offsets[block_row + 1])
        ):
            block_column = int(columns[offset])
            column_begin = block_column * snapshot.block_size
            value_begin = offset * snapshot.block_size * snapshot.block_size
            dense[
                row_begin : row_begin + snapshot.block_size,
                column_begin : column_begin + snapshot.block_size,
            ] = values[
                value_begin : value_begin
                + snapshot.block_size * snapshot.block_size
            ].reshape(snapshot.block_size, snapshot.block_size)
    return dense


def _level_specs():
    return (
        _i32_array(_FIRST_MAP),
        _i32_array(_SECOND_MAP),
    )


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
    vulkan_sparse_experimental=True,
)
def test_private_caller_coarsened_bsr_hierarchy_owns_exact_levels():
    source, fine_dense = _source()
    first_map, second_map = _level_specs()
    builder = _CallerCoarsenedBsrHierarchyBuilder(
        explicit_array_capacity_bytes=7800,
        bottom_scalar_size_cap=6,
    )
    hierarchy = builder.build(
        source,
        [(4, first_map), (2, second_map)],
        topology_version=7,
        numeric_version=11,
    )

    stats = hierarchy.debug_runtime_stats()
    assert stats["identity"] == {
        "backend_family": source._backend,
        "storage_format": "recursive_bsr",
        "dtype": "f32",
        "index_dtype": "i32",
        "block_size": 3,
        "topology_version": 7,
        "numeric_version": 11,
        "level_count": 3,
        "level_block_rows": (8, 4, 2),
        "level_scalar_rows": (24, 12, 6),
        "level_block_nnz": (30, 14, 4),
        "bottom_scalar_size_cap": 6,
    }
    assert stats["resources"] == {
        "operator_reserved_bytes": 1988,
        "aggregate_map_reserved_bytes": 48,
        "restriction_schedule_reserved_bytes": 80,
        "steady_reserved_bytes": 2116,
        "level_operator_reserved_bytes": (1236, 580, 172),
        "level_pattern_reserved_bytes": (156, 76, 28),
        "level_value_reserved_bytes": (1080, 504, 144),
        "level_map_reserved_bytes": (32, 16),
        "level_restriction_schedule_reserved_bytes": (52, 28),
    }
    report = stats["build"]
    assert report["preflight"]["steady_exact_upper_bytes"] == 3796
    assert report["preflight"][
        "build_peak_excluding_workspace_upper_bytes"
    ] == 7800
    assert report["steady_exact_bytes"] == 2116
    assert report[
        "actual_build_peak_excluding_workspace_bytes"
    ] == 5096
    assert report["staging_overlap_peak_bytes"] == 3008
    assert report["control_readbacks"] == 2
    assert report["final_completion_synchronizations"] == 1
    assert report["device_to_host_bytes"] == 16
    assert report["device_to_device_bytes"] == 848
    assert report["device_kernel_publish_bytes"] == 80
    assert report["device_payload_readback_bytes"] == 0
    assert report["retired_plan_count"] == 4
    assert report["live_plan_count_after_publish"] == 0
    assert not report["workspace_total_bytes_reported"]
    assert tuple(
        level["galerkin_staging_bytes"] for level in report["levels"]
    ) == (2000, 944)
    assert tuple(
        level["restriction_schedule_staging_bytes"]
        for level in report["levels"]
    ) == (64, 32)
    assert tuple(
        level["output_bytes"] for level in report["levels"]
    ) == (580, 172)
    assert tuple(
        level["stable_key_sort_passes"] for level in report["levels"]
    ) == (1, 1)

    first_prolongation = _prolongation(_FIRST_MAP, 4)
    first_oracle = first_prolongation.T @ fine_dense @ first_prolongation
    second_prolongation = _prolongation(_SECOND_MAP, 2)
    second_oracle = (
        second_prolongation.T @ first_oracle @ second_prolongation
    )
    np.testing.assert_allclose(
        _snapshot_dense(hierarchy._levels[1]),
        first_oracle,
        rtol=0.0,
        atol=5e-5,
    )
    np.testing.assert_allclose(
        _snapshot_dense(hierarchy._levels[2]),
        second_oracle,
        rtol=0.0,
        atol=8e-5,
    )

    # Caller maps are copied and schedules preserve block-row source order.
    first_map.fill(3)
    second_map.fill(1)
    np.testing.assert_array_equal(
        hierarchy._aggregate_maps[0].to_numpy(), _FIRST_MAP
    )
    np.testing.assert_array_equal(
        hierarchy._aggregate_maps[1].to_numpy(), _SECOND_MAP
    )
    np.testing.assert_array_equal(
        hierarchy._restriction_schedules[0]._coarse_offsets.to_numpy(),
        np.asarray([0, 2, 4, 6, 8], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        hierarchy._restriction_schedules[0]._ordered_fine_rows.to_numpy(),
        np.arange(8, dtype=np.int32),
    )
    np.testing.assert_array_equal(
        hierarchy._restriction_schedules[1]._coarse_offsets.to_numpy(),
        np.asarray([0, 2, 4], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        hierarchy._restriction_schedules[1]._ordered_fine_rows.to_numpy(),
        np.arange(4, dtype=np.int32),
    )

    builder_stats = builder.debug_runtime_stats()
    assert builder_stats["operations"] == {
        "build_attempts": 1,
        "successful_builds": 1,
        "rejected_builds": 0,
        "failed_builds": 0,
        "host_control_readbacks": 2,
        "final_host_synchronizations": 1,
    }
    assert builder_stats["resources"]["live_snapshot_count"] == 1
    assert builder_stats["resources"][
        "live_snapshot_reserved_bytes"
    ] == 2116
    assert builder_stats["transfers"] == {
        "device_to_host_bytes": 16,
        "device_to_device_bytes": 848,
        "device_kernel_publish_bytes": 80,
        "device_payload_readback_bytes": 0,
    }
    assert builder_stats["contract"][
        "one_key_sort_per_level_independent_of_block_size"
    ]
    assert builder_stats["contract"][
        "restriction_schedule_shared_across_block_components"
    ]
    assert builder_stats["contract"]["bottom_cap_counts_scalar_rows"]
    assert not builder_stats["contract"][
        "workspace_total_bytes_reported"
    ]


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_private_caller_coarsened_bsr_hierarchy_rejects_atomically():
    source, _ = _source()
    first_map, second_map = _level_specs()
    level_specs = [(4, first_map), (2, second_map)]

    too_small = _CallerCoarsenedBsrHierarchyBuilder(
        explicit_array_capacity_bytes=7799,
        bottom_scalar_size_cap=6,
    )
    with pytest.raises(TaichiRuntimeError, match="before build"):
        too_small.build(
            source,
            level_specs,
            topology_version=7,
            numeric_version=11,
        )
    rejected_stats = too_small.debug_runtime_stats()
    assert rejected_stats["identity"]["last_status"] == (
        "capacity_overflow"
    )
    assert rejected_stats["operations"] == {
        "build_attempts": 1,
        "successful_builds": 0,
        "rejected_builds": 1,
        "failed_builds": 0,
        "host_control_readbacks": 0,
        "final_host_synchronizations": 0,
    }
    assert rejected_stats["transfers"] == {
        "device_to_host_bytes": 0,
        "device_to_device_bytes": 0,
        "device_kernel_publish_bytes": 0,
        "device_payload_readback_bytes": 0,
    }

    builder = _CallerCoarsenedBsrHierarchyBuilder(
        explicit_array_capacity_bytes=7800,
        bottom_scalar_size_cap=6,
    )
    published = builder.build(
        source,
        level_specs,
        topology_version=7,
        numeric_version=11,
    )
    published_bottom = _snapshot_payload(published._levels[-1])
    invalid_second_map = _i32_array(
        np.asarray([0, 0, 1, 2], dtype=np.int32)
    )
    with pytest.raises(TaichiRuntimeError, match="before publish"):
        builder.build(
            source,
            [(4, first_map), (2, invalid_second_map)],
            topology_version=7,
            numeric_version=11,
        )
    for actual, expected in zip(
        _snapshot_payload(published._levels[-1]), published_bottom
    ):
        np.testing.assert_array_equal(actual, expected)

    failed_stats = builder.debug_runtime_stats()
    assert failed_stats["identity"]["last_status"] == "build_failed"
    assert failed_stats["operations"] == {
        "build_attempts": 2,
        "successful_builds": 1,
        "rejected_builds": 0,
        "failed_builds": 1,
        "host_control_readbacks": 4,
        "final_host_synchronizations": 2,
    }
    assert failed_stats["resources"]["live_snapshot_count"] == 1
    assert failed_stats["resources"][
        "live_snapshot_reserved_bytes"
    ] == 2116
    assert failed_stats["transfers"] == {
        "device_to_host_bytes": 32,
        "device_to_device_bytes": 1524,
        "device_kernel_publish_bytes": 132,
        "device_payload_readback_bytes": 0,
    }

    with pytest.raises(TaichiRuntimeError, match="versions must match"):
        builder.build(
            source,
            level_specs,
            topology_version=8,
            numeric_version=11,
        )
    with pytest.raises(TaichiRuntimeError, match="decrease strictly"):
        builder.build(
            source,
            [(8, first_map)],
            topology_version=7,
            numeric_version=11,
        )
    bottom_limited = _CallerCoarsenedBsrHierarchyBuilder(
        explicit_array_capacity_bytes=7800,
        bottom_scalar_size_cap=5,
    )
    with pytest.raises(TaichiRuntimeError, match="bottom_scalar_size_cap"):
        bottom_limited.build(
            source,
            level_specs,
            topology_version=7,
            numeric_version=11,
        )


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_private_caller_coarsened_bsr_hierarchy_rejects_after_reset():
    source, _ = _source()
    first_map, second_map = _level_specs()
    builder = _CallerCoarsenedBsrHierarchyBuilder(
        explicit_array_capacity_bytes=7800,
        bottom_scalar_size_cap=6,
    )
    hierarchy = builder.build(
        source,
        [(4, first_map), (2, second_map)],
        topology_version=7,
        numeric_version=11,
    )
    ti.reset()
    ti.init(arch=ti.cpu, enable_fallback=False)

    with pytest.raises(TaichiRuntimeError, match="runtime has been reset"):
        hierarchy.debug_runtime_stats()
    with pytest.raises(TaichiRuntimeError, match="runtime has been reset"):
        builder.debug_runtime_stats()
