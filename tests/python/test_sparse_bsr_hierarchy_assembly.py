import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.linalg._sparse_bsr_graph_operator import (
    _DeviceBsrSnapshot,
    _SparseBsrGraphOperatorPlan,
)
from taichi_forge.linalg._sparse_bsr_hierarchy_assembly import (
    _SparseGalerkinBsrAssemblyPlan,
)
from tests import test_utils


_BLOCK_ROWS = 4
_ROW_OFFSETS = np.asarray([0, 3, 6, 9, 12], dtype=np.int32)
_COLUMN_INDICES = np.asarray(
    [0, 1, 3, 0, 1, 2, 1, 2, 3, 0, 2, 3], dtype=np.int32
)
_AGGREGATE = np.asarray([0, 0, 1, 1], dtype=np.int32)
_COARSE_ROW_OFFSETS = np.asarray([0, 2, 4], dtype=np.int32)
_COARSE_COLUMN_INDICES = np.asarray([0, 1, 0, 1], dtype=np.int32)


def _i32_array(values):
    result = ti.ndarray(ti.i32, shape=len(values))
    result.from_numpy(np.asarray(values, dtype=np.int32))
    return result


def _f32_array(values):
    result = ti.ndarray(ti.f32, shape=len(values))
    result.from_numpy(np.asarray(values, dtype=np.float32))
    return result


def _dense_block_operator(block_size, scale=1.0):
    dense = np.zeros(
        (_BLOCK_ROWS * block_size, _BLOCK_ROWS * block_size),
        dtype=np.float32,
    )
    coordinates = np.arange(1, block_size + 1, dtype=np.float32)
    for block_row in range(_BLOCK_ROWS):
        begin = block_row * block_size
        mass = np.diag(0.5 + 0.03 * coordinates)
        mass += 0.02 * np.outer(coordinates, coordinates) / block_size
        dense[
            begin : begin + block_size, begin : begin + block_size
        ] += mass
    for ordinal, (left, right) in enumerate(
        ((0, 1), (1, 2), (2, 3), (0, 3))
    ):
        direction = coordinates + 0.1 * ordinal
        weight = np.diag(
            0.8 + 0.04 * coordinates + 0.02 * ordinal
        )
        weight += 0.03 * np.outer(direction, direction) / block_size
        left_begin = left * block_size
        right_begin = right * block_size
        left_slice = slice(left_begin, left_begin + block_size)
        right_slice = slice(right_begin, right_begin + block_size)
        dense[left_slice, left_slice] += weight
        dense[right_slice, right_slice] += weight
        dense[left_slice, right_slice] -= weight
        dense[right_slice, left_slice] -= weight
    return np.asarray(scale * dense, dtype=np.float32)


def _flat_block_values(dense, block_size):
    values = []
    for block_row in range(_BLOCK_ROWS):
        row_begin = block_row * block_size
        for offset in range(
            _ROW_OFFSETS[block_row], _ROW_OFFSETS[block_row + 1]
        ):
            block_column = int(_COLUMN_INDICES[offset])
            column_begin = block_column * block_size
            values.extend(
                dense[
                    row_begin : row_begin + block_size,
                    column_begin : column_begin + block_size,
                ].reshape(-1)
            )
    return np.asarray(values, dtype=np.float32)


def _source(block_size, *, values=None, numeric_version=1):
    dense = _dense_block_operator(block_size)
    if values is None:
        values = _flat_block_values(dense, block_size)
    source = _DeviceBsrSnapshot.copy_validated(
        block_rows=_BLOCK_ROWS,
        block_cols=_BLOCK_ROWS,
        block_size=block_size,
        row_offsets=_i32_array(_ROW_OFFSETS),
        column_indices=_i32_array(_COLUMN_INDICES),
        values=_f32_array(values),
        topology_version=1,
        numeric_version=numeric_version,
    )
    return source, dense


def _coarse_dense_oracle(fine_dense, block_size):
    prolongation = np.zeros(
        (_BLOCK_ROWS * block_size, 2 * block_size), dtype=np.float32
    )
    identity = np.eye(block_size, dtype=np.float32)
    for fine_block_row, coarse_block_row in enumerate(_AGGREGATE):
        fine_begin = fine_block_row * block_size
        coarse_begin = int(coarse_block_row) * block_size
        prolongation[
            fine_begin : fine_begin + block_size,
            coarse_begin : coarse_begin + block_size,
        ] = identity
    return prolongation.T @ fine_dense @ prolongation


def _snapshot_payload(snapshot):
    return (
        snapshot._row_offsets.to_numpy(),
        snapshot._column_indices.to_numpy(),
        snapshot._values.to_numpy(),
    )


def _snapshot_dense(snapshot):
    row_offsets, columns, values = _snapshot_payload(snapshot)
    block_size = snapshot.block_size
    dense = np.zeros((snapshot.rows, snapshot.cols), dtype=np.float32)
    for block_row in range(snapshot.block_rows):
        row_begin = block_row * block_size
        for offset in range(
            int(row_offsets[block_row]), int(row_offsets[block_row + 1])
        ):
            block_column = int(columns[offset])
            column_begin = block_column * block_size
            dense[
                row_begin : row_begin + block_size,
                column_begin : column_begin + block_size,
            ] = values[
                offset * block_size * block_size :
                (offset + 1) * block_size * block_size
            ].reshape(block_size, block_size)
    return dense


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
    vulkan_sparse_experimental=True,
)
def test_private_block_galerkin_publishes_exact_bsr_with_one_key_sort():
    source, dense = _source(3)
    aggregate = _i32_array(_AGGREGATE)
    plan = _SparseGalerkinBsrAssemblyPlan(
        fine_block_rows=4,
        coarse_block_rows=2,
        block_size=3,
        capacity=12,
    )
    coarse = plan.build(
        source,
        aggregate,
        topology_version=2,
        numeric_version=1,
    )

    coarse_stats = coarse.debug_runtime_stats()
    assert coarse_stats["identity"] == {
        "backend_family": source._backend,
        "storage_format": "bsr",
        "dtype": "f32",
        "index_dtype": "i32",
        "block_rows": 2,
        "block_cols": 2,
        "block_size": 3,
        "block_nnz": 4,
        "rows": 6,
        "cols": 6,
        "stored_scalar_values": 36,
        "topology_version": 2,
        "numeric_version": 1,
        "construction": "block_galerkin_exact_prefix_publish",
    }
    assert coarse_stats["resources"] == {
        "pattern_reserved_bytes": 28,
        "value_reserved_bytes": 144,
        "total_reserved_bytes": 172,
    }
    assert coarse_stats["transfers"] == {
        "device_to_host_bytes": 0,
        "device_to_device_bytes": 172,
        "device_payload_readback_bytes": 0,
    }

    row_offsets, columns, values = _snapshot_payload(coarse)
    np.testing.assert_array_equal(row_offsets, _COARSE_ROW_OFFSETS)
    np.testing.assert_array_equal(columns, _COARSE_COLUMN_INDICES)
    np.testing.assert_allclose(
        _snapshot_dense(coarse),
        _coarse_dense_oracle(dense, 3),
        rtol=0.0,
        atol=3e-5,
    )
    coarse_dense = _coarse_dense_oracle(dense, 3)
    operator_plan = _SparseBsrGraphOperatorPlan(
        coarse, explicit_array_capacity_bytes=344
    )
    input_numpy = np.linspace(-0.75, 1.0, 6, dtype=np.float32)
    input_array = _f32_array(input_numpy)
    output_array = ti.ndarray(ti.f32, shape=6)
    output_array.fill(0.0)
    operator_plan.apply(input_array, output_array)
    ti.sync()
    np.testing.assert_allclose(
        output_array.to_numpy(),
        coarse_dense @ input_numpy,
        rtol=0.0,
        atol=3e-5,
    )

    # A failed staging reuse must not mutate the already-published arrays.
    published_payload = (row_offsets.copy(), columns.copy(), values.copy())
    invalid_aggregate = _AGGREGATE.copy()
    invalid_aggregate[-1] = 2
    with pytest.raises(TaichiRuntimeError, match="before publish"):
        plan.build(
            source,
            _i32_array(invalid_aggregate),
            topology_version=3,
            numeric_version=1,
        )
    for actual, expected in zip(
        _snapshot_payload(coarse), published_payload
    ):
        np.testing.assert_array_equal(actual, expected)

    stats = plan.debug_runtime_stats()
    assert stats["identity"] == {
        "backend_family": source._backend,
        "method": "stable_key_ordinal_sort_rle_block_segment_sum_bsr",
        "fine_block_rows": 4,
        "coarse_block_rows": 2,
        "block_size": 3,
        "block_elements": 9,
        "block_edge_capacity": 12,
    }
    assert stats["status"] == {
        "last_status": 1,
        "last_unique_block_nnz": 0,
        "last_duplicate_block_edges": 0,
    }
    assert stats["operations"] == {
        "build_calls": 2,
        "successful_builds": 1,
        "failed_builds": 1,
        "stable_key_sort_passes_per_build": 1,
        "workspace_builds": 1,
        "workspace_reuses": 1,
        "host_control_readbacks": 2,
    }
    resources = stats["resources"]
    assert resources["persistent_staging_reserved_bytes"] == 816
    assert resources["key_reserved_bytes"] == 96
    assert resources["source_ordinal_reserved_bytes"] == 48
    assert resources["block_payload_staging_reserved_bytes"] == 432
    assert resources["last_output_pattern_bytes"] == 28
    assert resources["last_output_value_bytes"] == 144
    assert resources["last_build_peak_excluding_workspace_bytes"] == 988
    assert resources["live_snapshot_count"] == 1
    assert resources["live_snapshot_reserved_bytes"] == 172
    assert resources["known_workspace_reported_bytes"] >= 48
    detailed_workspace = (
        ti.lang.impl.get_runtime().prog._primitive_workspace_detailed_stats()
    )
    expected_scan_bytes = sum(
        group["reserved_bytes"]
        for group in detailed_workspace["groups"]
        if group["family"] == "scan"
    )
    assert resources["shared_scan_workspace_bytes"] == expected_scan_bytes
    assert resources[
        "shared_scan_workspace_ownership_scope"
    ] == "program_scan_arena"
    assert stats["transfers"] == {
        "device_to_host_bytes": 16,
        "device_to_device_bytes": 204,
        "device_payload_readback_bytes": 0,
        "control_readback_bytes_per_build": 8,
    }
    assert stats["contract"]["one_key_sort_independent_of_block_size"]
    assert stats["contract"][
        "block_components_reduce_from_one_shared_permutation"
    ]
    assert stats["contract"][
        "failed_build_does_not_mutate_returned_snapshots"
    ]
    assert not stats["contract"]["coarsening_policy_selected"]
    assert not stats["contract"]["workspace_total_bytes_reported"]
    assert stats["contract"][
        "shared_scan_workspace_current_bytes_reported"
    ]
    assert not stats["contract"][
        "shared_scan_workspace_in_plan_owned_bytes"
    ]


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_private_block_galerkin_supports_qualified_physics_block_sizes():
    for block_size in (2, 6, 12):
        source, dense = _source(block_size)
        plan = _SparseGalerkinBsrAssemblyPlan(
            fine_block_rows=4,
            coarse_block_rows=2,
            block_size=block_size,
            capacity=12,
        )
        coarse = plan.build(
            source,
            _i32_array(_AGGREGATE),
            topology_version=2,
            numeric_version=1,
        )
        np.testing.assert_array_equal(
            coarse._row_offsets.to_numpy(), _COARSE_ROW_OFFSETS
        )
        np.testing.assert_array_equal(
            coarse._column_indices.to_numpy(), _COARSE_COLUMN_INDICES
        )
        np.testing.assert_allclose(
            _snapshot_dense(coarse),
            _coarse_dense_oracle(dense, block_size),
            rtol=0.0,
            atol=5e-5,
        )
        assert plan.debug_runtime_stats()["operations"][
            "stable_key_sort_passes_per_build"
        ] == 1


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_private_block_galerkin_rejects_nonfinite_duplicate_sum():
    block_size = 2
    values = np.zeros(
        len(_COLUMN_INDICES) * block_size * block_size,
        dtype=np.float32,
    )
    values[0] = np.float32(3.0e38)
    values[block_size * block_size] = np.float32(3.0e38)
    source, _ = _source(block_size, values=values, numeric_version=2)
    plan = _SparseGalerkinBsrAssemblyPlan(
        fine_block_rows=4,
        coarse_block_rows=2,
        block_size=block_size,
        capacity=12,
    )
    with pytest.raises(
        TaichiRuntimeError, match="duplicate dense-block sum"
    ):
        plan.build(
            source,
            _i32_array(_AGGREGATE),
            topology_version=2,
            numeric_version=2,
        )
    assert plan.debug_runtime_stats()["status"] == {
        "last_status": 3,
        "last_unique_block_nnz": 0,
        "last_duplicate_block_edges": 0,
    }


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_private_block_galerkin_plan_rejects_after_reset():
    plan = _SparseGalerkinBsrAssemblyPlan(
        fine_block_rows=4,
        coarse_block_rows=2,
        block_size=3,
        capacity=12,
    )
    ti.reset()
    ti.init(arch=ti.cpu, enable_fallback=False)

    with pytest.raises(TaichiRuntimeError, match="runtime has been reset"):
        plan.debug_runtime_stats()
