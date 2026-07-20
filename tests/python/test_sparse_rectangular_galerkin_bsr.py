import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils

from taichi_forge.linalg._sparse_block_transfer_graph import (
    _DeviceBlockTransferSnapshot,
)
from taichi_forge.linalg._sparse_bsr_graph_operator import (
    _DeviceBsrSnapshot,
    _SparseBsrGraphOperatorPlan,
)
from taichi_forge.linalg._sparse_rectangular_galerkin_bsr import (
    _SparseRectangularGalerkinBsrBuilder,
)


_FINE_BLOCK_ROWS = 5
_COARSE_BLOCK_ROWS = 2
_FINE_BLOCK_SIZE = 3
_COARSE_BLOCK_SIZE = 6
_TRANSFER_ROW_OFFSETS = np.asarray([0, 1, 3, 4, 6, 7], dtype=np.int32)
_TRANSFER_COLUMNS = np.asarray([0, 0, 1, 1, 0, 1, 1], dtype=np.int32)


def _i32_array(values):
    result = ti.ndarray(ti.i32, shape=len(values))
    result.from_numpy(np.asarray(values, dtype=np.int32))
    return result


def _f32_array(values):
    result = ti.ndarray(ti.f32, shape=len(values))
    result.from_numpy(np.asarray(values, dtype=np.float32))
    return result


def _fine_dense_operator():
    rows = _FINE_BLOCK_ROWS * _FINE_BLOCK_SIZE
    row = np.arange(rows, dtype=np.float64)[:, None]
    column = np.arange(rows, dtype=np.float64)[None, :]
    generator = 0.075 * np.sin(0.17 * (row + 1.0) * (column + 2.0))
    generator += 0.035 * np.cos(0.11 * (row + column + 3.0))
    dense = generator.T @ generator
    dense += np.eye(rows, dtype=np.float64) * 0.8
    return dense.astype(np.float32)


def _transfer_values():
    blocks = []
    for ordinal in range(len(_TRANSFER_COLUMNS)):
        block = np.arange(18, dtype=np.float32).reshape(3, 6)
        block = 0.01 * block + np.float32(0.125 * (ordinal + 1))
        blocks.append(block)
    return np.asarray(blocks, dtype=np.float32).reshape(-1)


def _dense_transfer(values=None):
    if values is None:
        values = _transfer_values()
    blocks = np.asarray(values, dtype=np.float32).reshape(-1, 3, 6)
    dense = np.zeros((15, 12), dtype=np.float32)
    for fine_row in range(_FINE_BLOCK_ROWS):
        begin = _TRANSFER_ROW_OFFSETS[fine_row]
        end = _TRANSFER_ROW_OFFSETS[fine_row + 1]
        for offset in range(begin, end):
            coarse_row = int(_TRANSFER_COLUMNS[offset])
            dense[
                3 * fine_row : 3 * fine_row + 3,
                6 * coarse_row : 6 * coarse_row + 6,
            ] = blocks[offset]
    return dense


def _compress_square_blocks(dense, block_size):
    block_rows = dense.shape[0] // block_size
    row_offsets = [0]
    columns = []
    values = []
    for block_row in range(block_rows):
        row_begin = block_row * block_size
        for block_col in range(block_rows):
            col_begin = block_col * block_size
            block = dense[
                row_begin : row_begin + block_size,
                col_begin : col_begin + block_size,
            ]
            if np.any(np.abs(block) > 1e-12):
                columns.append(block_col)
                values.extend(block.reshape(-1))
        row_offsets.append(len(columns))
    return (
        np.asarray(row_offsets, dtype=np.int32),
        np.asarray(columns, dtype=np.int32),
        np.asarray(values, dtype=np.float32),
    )


def _source(dense=None):
    if dense is None:
        dense = _fine_dense_operator()
    row_offsets, columns, values = _compress_square_blocks(dense, 3)
    return _DeviceBsrSnapshot.copy_validated(
        block_rows=5,
        block_cols=5,
        block_size=3,
        row_offsets=_i32_array(row_offsets),
        column_indices=_i32_array(columns),
        values=_f32_array(values),
        topology_version=7,
        numeric_version=11,
    )


def _transfer(values=None):
    if values is None:
        values = _transfer_values()
    return _DeviceBlockTransferSnapshot.copy_validated(
        fine_block_rows=5,
        coarse_block_rows=2,
        fine_block_size=3,
        coarse_block_size=6,
        row_offsets=_i32_array(_TRANSFER_ROW_OFFSETS),
        column_indices=_i32_array(_TRANSFER_COLUMNS),
        values=_f32_array(values),
        topology_version=13,
        numeric_version=17,
    )


def _snapshot_dense(snapshot):
    row_offsets = snapshot._row_offsets.to_numpy()
    columns = snapshot._column_indices.to_numpy()
    blocks = snapshot._values.to_numpy().reshape(
        snapshot.block_nnz, snapshot.block_size, snapshot.block_size
    )
    dense = np.zeros((snapshot.rows, snapshot.cols), dtype=np.float32)
    for block_row in range(snapshot.block_rows):
        for offset in range(row_offsets[block_row], row_offsets[block_row + 1]):
            block_col = int(columns[offset])
            row_begin = block_row * snapshot.block_size
            col_begin = block_col * snapshot.block_size
            dense[
                row_begin : row_begin + snapshot.block_size,
                col_begin : col_begin + snapshot.block_size,
            ] = blocks[offset]
    return dense


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
)
def test_private_rectangular_galerkin_builds_exact_device_bsr():
    fine_dense = _fine_dense_operator()
    transfer_dense = _dense_transfer()
    source = _source(fine_dense)
    transfer = _transfer()
    builder = _SparseRectangularGalerkinBsrBuilder(
        explicit_array_capacity_bytes=3576
    )
    coarse = builder.build(
        source,
        transfer,
        contribution_count=49,
        max_transfer_blocks_per_row=2,
        topology_version=19,
        numeric_version=23,
    )

    oracle = transfer_dense.T @ fine_dense @ transfer_dense
    np.testing.assert_allclose(
        _snapshot_dense(coarse), oracle, rtol=3e-4, atol=3e-4
    )
    np.testing.assert_array_equal(
        coarse._row_offsets.to_numpy(), np.asarray([0, 2, 4], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        coarse._column_indices.to_numpy(),
        np.asarray([0, 1, 0, 1], dtype=np.int32),
    )
    coarse_stats = coarse.debug_runtime_stats()
    assert coarse_stats["identity"]["block_rows"] == 2
    assert coarse_stats["identity"]["block_size"] == 6
    assert coarse_stats["identity"]["block_nnz"] == 4
    assert coarse_stats["identity"]["topology_version"] == 19
    assert coarse_stats["identity"]["numeric_version"] == 23
    assert coarse_stats["identity"]["construction"] == (
        "rectangular_galerkin_exact_device_publish"
    )
    assert coarse_stats["resources"] == {
        "pattern_reserved_bytes": 28,
        "value_reserved_bytes": 576,
        "total_reserved_bytes": 604,
    }
    assert coarse_stats["transfers"] == {
        "device_to_host_bytes": 0,
        "device_to_device_bytes": 0,
        "device_payload_readback_bytes": 0,
    }

    stats = builder.debug_runtime_stats()
    assert stats["identity"] == {
        "backend_family": stats["identity"]["backend_family"],
        "method": "exact_contribution_prefix_stable_ordinal_sort_gather_ptap",
        "last_contribution_count": 49,
        "last_unique_block_nnz": 4,
        "last_observed_max_transfer_row_degree": 2,
        "last_source_ordinal_degree_bits": 1,
    }
    assert stats["status"] == {"last_status": 0}
    assert stats["operations"] == {
        "build_calls": 1,
        "successful_builds": 1,
        "failed_builds": 0,
        "stable_key_sort_passes": 1,
        "control_readbacks_per_successful_build": 3,
    }
    resources = stats["resources"]
    assert resources["borrowed_source_operator_reserved_bytes"] == 1024
    assert resources["borrowed_transfer_reserved_bytes"] == 556
    assert resources[
        "edge_row_and_contribution_offset_staging_bytes"
    ] == 204
    assert resources["key_and_ordinal_sort_staging_bytes"] == 1176
    assert resources["control_and_run_count_staging_bytes"] == 12
    assert resources["retired_builder_staging_reserved_bytes"] == 1392
    assert resources["materialized_contribution_payload_bytes"] == 0
    assert resources["avoided_contribution_payload_bytes"] == 7056
    assert resources["last_output_pattern_bytes"] == 28
    assert resources["last_output_value_bytes"] == 576
    assert resources["last_steady_generation_bytes"] == 2184
    assert resources["last_build_peak_excluding_workspace_bytes"] == 3576
    assert resources["explicit_array_capacity_bytes"] == 3576
    detailed_workspace = (
        ti.lang.impl.get_runtime().prog._primitive_workspace_detailed_stats()
    )
    expected_sort_scan_bytes = sum(
        group["reserved_bytes"]
        for group in detailed_workspace["groups"]
        if group["family"] in ("ordering", "ordering_aux", "scan")
    )
    assert (
        resources["shared_sort_scan_workspace_bytes"]
        == expected_sort_scan_bytes
    )
    assert resources[
        "shared_sort_scan_workspace_ownership_scope"
    ] == "program_ordering_ordering_aux_scan_arena"
    assert resources["live_snapshot_count"] == 1
    assert resources["live_snapshot_reserved_bytes"] == 604
    assert stats["transfers"] == {
        "device_to_host_bytes": 24,
        "device_to_device_bytes": 0,
        "device_kernel_publish_bytes": 604,
        "device_payload_readback_bytes": 0,
        "control_readback_bytes_per_successful_build": 24,
    }
    assert stats["contract"]["single_i32_source_triple_ordinal"]
    assert stats["contract"][
        "one_stable_key_sort_independent_of_block_components"
    ]
    assert not stats["contract"]["contribution_dense_payload_materialized"]
    assert stats["contract"]["exact_output_allocated_after_unique_count"]
    assert stats["contract"][
        "shared_sort_scan_workspace_current_bytes_reported"
    ]
    assert not stats["contract"][
        "shared_sort_scan_workspace_in_explicit_capacity"
    ]
    assert not stats["contract"]["workspace_total_bytes_reported"]
    assert not stats["contract"]["device_payload_readback_required"]

    graph = _SparseBsrGraphOperatorPlan(
        coarse, explicit_array_capacity_bytes=1208
    )
    vector_host = np.linspace(-0.6, 0.9, 12, dtype=np.float32)
    vector = _f32_array(vector_host)
    output = ti.ndarray(ti.f32, shape=12)
    graph.apply(vector, output)
    np.testing.assert_allclose(
        output.to_numpy(), oracle @ vector_host, rtol=4e-4, atol=4e-4
    )


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_private_rectangular_galerkin_rejects_topology_and_capacity_atomically():
    source = _source()
    transfer = _transfer()
    source_before = source._values.to_numpy()
    transfer_before = transfer._values.to_numpy()

    count_builder = _SparseRectangularGalerkinBsrBuilder(
        explicit_array_capacity_bytes=3576
    )
    with pytest.raises(RuntimeError, match="contribution count"):
        count_builder.build(
            source,
            transfer,
            contribution_count=48,
            max_transfer_blocks_per_row=2,
            topology_version=19,
            numeric_version=23,
        )
    assert count_builder.debug_runtime_stats()["operations"] == {
        "build_calls": 1,
        "successful_builds": 0,
        "failed_builds": 1,
        "stable_key_sort_passes": 0,
        "control_readbacks_per_successful_build": 3,
    }

    degree_builder = _SparseRectangularGalerkinBsrBuilder(
        explicit_array_capacity_bytes=3576
    )
    with pytest.raises(RuntimeError, match="degree exceeds"):
        degree_builder.build(
            source,
            transfer,
            contribution_count=49,
            max_transfer_blocks_per_row=1,
            topology_version=19,
            numeric_version=23,
        )

    capacity_builder = _SparseRectangularGalerkinBsrBuilder(
        explicit_array_capacity_bytes=3575
    )
    with pytest.raises(RuntimeError, match="exact output allocation"):
        capacity_builder.build(
            source,
            transfer,
            contribution_count=49,
            max_transfer_blocks_per_row=2,
            topology_version=19,
            numeric_version=23,
        )
    capacity_stats = capacity_builder.debug_runtime_stats()
    assert capacity_stats["transfers"]["device_to_host_bytes"] == 16
    assert capacity_stats["resources"][
        "last_build_peak_excluding_workspace_bytes"
    ] == 3576
    assert capacity_stats["resources"]["live_snapshot_count"] == 0

    np.testing.assert_array_equal(source._values.to_numpy(), source_before)
    np.testing.assert_array_equal(transfer._values.to_numpy(), transfer_before)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_private_rectangular_galerkin_rejects_nonfinite_output_and_reset():
    huge_dense = np.full((15, 15), np.float32(3.0e38), dtype=np.float32)
    source = _source(huge_dense)
    transfer = _transfer(np.ones(7 * 18, dtype=np.float32))
    builder = _SparseRectangularGalerkinBsrBuilder(
        explicit_array_capacity_bytes=3576
    )
    with pytest.raises(RuntimeError, match="not finite"):
        builder.build(
            source,
            transfer,
            contribution_count=49,
            max_transfer_blocks_per_row=2,
            topology_version=29,
            numeric_version=31,
        )
    assert builder.debug_runtime_stats()["status"] == {"last_status": 4}

    ti.reset()
    with pytest.raises(RuntimeError, match="runtime has been reset"):
        builder.debug_runtime_stats()
