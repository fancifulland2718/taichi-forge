import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils
from tests.sparse_runtime_stats import (
    assert_sparse_graph_cache_attribution,
)

from taichi_forge.linalg._sparse_block_transfer_graph import (
    _DeviceBlockTransferSnapshot,
    _SparseBlockTransferGraphPlan,
)


_FINE_BLOCK_ROWS = 5
_COARSE_BLOCK_ROWS = 2
_FINE_BLOCK_SIZE = 3
_COARSE_BLOCK_SIZE = 6
_ROW_OFFSETS = np.asarray([0, 1, 3, 4, 6, 7], dtype=np.int32)
_COLUMNS = np.asarray([0, 0, 1, 1, 0, 1, 1], dtype=np.int32)


def _host_values():
    blocks = []
    for ordinal in range(len(_COLUMNS)):
        block = np.arange(18, dtype=np.float32).reshape(3, 6)
        block = 0.01 * block + np.float32(0.125 * (ordinal + 1))
        blocks.append(block)
    return np.asarray(blocks, dtype=np.float32).reshape(-1)


def _dense_transfer(values=None):
    if values is None:
        values = _host_values()
    blocks = np.asarray(values, dtype=np.float32).reshape(-1, 3, 6)
    dense = np.zeros((15, 12), dtype=np.float32)
    for fine_row in range(_FINE_BLOCK_ROWS):
        for offset in range(_ROW_OFFSETS[fine_row], _ROW_OFFSETS[fine_row + 1]):
            coarse_row = int(_COLUMNS[offset])
            dense[
                3 * fine_row : 3 * fine_row + 3,
                6 * coarse_row : 6 * coarse_row + 6,
            ] = blocks[offset]
    return dense


def _i32_array(values):
    result = ti.ndarray(ti.i32, shape=len(values))
    result.from_numpy(np.asarray(values, dtype=np.int32))
    return result


def _f32_array(values):
    result = ti.ndarray(ti.f32, shape=len(values))
    result.from_numpy(np.asarray(values, dtype=np.float32))
    return result


def _snapshot(*, row_offsets=None, columns=None, values=None):
    if row_offsets is None:
        row_offsets = _ROW_OFFSETS
    if columns is None:
        columns = _COLUMNS
    if values is None:
        values = _host_values()
    return _DeviceBlockTransferSnapshot.copy_validated(
        fine_block_rows=_FINE_BLOCK_ROWS,
        coarse_block_rows=_COARSE_BLOCK_ROWS,
        fine_block_size=_FINE_BLOCK_SIZE,
        coarse_block_size=_COARSE_BLOCK_SIZE,
        row_offsets=_i32_array(row_offsets),
        column_indices=_i32_array(columns),
        values=_f32_array(values),
        topology_version=7,
        numeric_version=11,
    )


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
)
def test_private_rectangular_block_transfer_graph_matches_dense_transpose():
    row_offsets = _i32_array(_ROW_OFFSETS)
    columns = _i32_array(_COLUMNS)
    values_host = _host_values()
    values = _f32_array(values_host)
    snapshot = _DeviceBlockTransferSnapshot.copy_validated(
        fine_block_rows=_FINE_BLOCK_ROWS,
        coarse_block_rows=_COARSE_BLOCK_ROWS,
        fine_block_size=_FINE_BLOCK_SIZE,
        coarse_block_size=_COARSE_BLOCK_SIZE,
        row_offsets=row_offsets,
        column_indices=columns,
        values=values,
        topology_version=7,
        numeric_version=11,
    )
    row_offsets.fill(0)
    columns.fill(0)
    values.fill(0.0)

    snapshot_stats = snapshot.debug_runtime_stats()
    assert snapshot_stats["identity"] == {
        "backend_family": snapshot_stats["identity"]["backend_family"],
        "storage_format": "rectangular_block_csr",
        "dtype": "f32",
        "index_dtype": "i32",
        "fine_block_rows": 5,
        "coarse_block_rows": 2,
        "fine_block_size": 3,
        "coarse_block_size": 6,
        "block_nnz": 7,
        "fine_scalar_rows": 15,
        "coarse_scalar_rows": 12,
        "stored_scalar_values": 126,
        "topology_version": 7,
        "numeric_version": 11,
    }
    assert snapshot_stats["resources"] == {
        "pattern_reserved_bytes": 52,
        "value_reserved_bytes": 504,
        "total_reserved_bytes": 556,
        "borrowed_source_bytes_during_copy": 556,
        "validation_control_peak_bytes": 8,
        "construction_peak_explicit_array_bytes": 1120,
    }
    assert snapshot_stats["transfers"] == {
        "device_to_host_bytes": 8,
        "device_to_device_bytes": 556,
        "device_payload_readback_bytes": 0,
    }
    assert snapshot_stats["contract"][
        "fine_and_coarse_block_sizes_are_independent"
    ]
    assert snapshot_stats["contract"]["caller_source_not_retained"]

    plan = _SparseBlockTransferGraphPlan(
        snapshot, explicit_array_capacity_bytes=708
    )
    dense = _dense_transfer(values_host)
    coarse_host = np.linspace(-0.8, 1.1, 12, dtype=np.float32)
    fine_host = np.linspace(0.7, -0.6, 15, dtype=np.float32)
    coarse_input = _f32_array(coarse_host)
    fine_output = ti.ndarray(ti.f32, shape=15)
    plan.prolongate(coarse_input, fine_output)
    np.testing.assert_allclose(
        fine_output.to_numpy(), dense @ coarse_host, rtol=2e-5, atol=2e-5
    )

    fine_input = _f32_array(fine_host)
    coarse_output = ti.ndarray(ti.f32, shape=12)
    plan.restrict(fine_input, coarse_output)
    np.testing.assert_allclose(
        coarse_output.to_numpy(), dense.T @ fine_host, rtol=2e-5, atol=2e-5
    )

    schedule = plan._schedule
    np.testing.assert_array_equal(
        schedule._coarse_offsets.to_numpy(),
        np.asarray([0, 3, 7], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        schedule._ordered_fine_rows.to_numpy(),
        np.asarray([0, 1, 3, 1, 2, 3, 4], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        schedule._ordered_block_ordinals.to_numpy(),
        np.asarray([0, 1, 4, 2, 3, 5, 6], dtype=np.int32),
    )
    assert schedule.debug_runtime_stats()["resources"] == {
        "coarse_offsets_reserved_bytes": 12,
        "ordered_fine_rows_reserved_bytes": 28,
        "ordered_block_ordinals_reserved_bytes": 28,
        "total_reserved_bytes": 68,
    }

    stats = plan.debug_runtime_stats()
    assert stats["operations"] == {
        "prolongate_calls": 1,
        "restrict_calls": 1,
        "rejected_apply_calls": 0,
        "prolongate_graph_node_count": 1,
        "prolongate_graph_dispatch_count": 1,
        "restrict_graph_node_count": 1,
        "restrict_graph_dispatch_count": 1,
        "host_graph_submissions_per_apply": 1,
        "explicit_apply_host_synchronizations": 0,
        "construction_host_synchronizations": 1,
    }
    assert_sparse_graph_cache_attribution(
        stats, expected_cache_object_count=2
    )
    assert stats["resources"][
        "borrowed_transfer_snapshot_reserved_bytes"
    ] == 556
    assert stats["resources"]["transpose_schedule_reserved_bytes"] == 68
    assert stats["resources"]["steady_explicit_array_bytes"] == 624
    assert stats["resources"][
        "retired_schedule_staging_reserved_bytes"
    ] == 84
    assert stats["resources"]["build_peak_explicit_array_bytes"] == 708
    assert stats["resources"]["explicit_array_capacity_bytes"] == 708
    detailed_workspace = (
        ti.lang.impl.get_runtime().prog._primitive_workspace_detailed_stats()
    )
    expected_scan_bytes = sum(
        group["reserved_bytes"]
        for group in detailed_workspace["groups"]
        if group["family"] == "scan"
    )
    assert stats["resources"]["shared_scan_workspace_bytes"] == (
        expected_scan_bytes
    )
    assert stats["resources"][
        "shared_scan_workspace_ownership_scope"
    ] == "program_scan_arena"
    assert stats["transfers"] == {
        "device_to_host_bytes": 0,
        "device_to_device_bytes": 0,
        "device_kernel_publish_bytes": 68,
        "device_payload_readback_bytes": 0,
    }
    assert stats["contract"]["transpose_schedule_owned"]
    assert stats["contract"]["schedule_staging_retired_after_sync"]
    assert stats["contract"]["build_peak_excludes_provider_workspace"]
    assert stats["contract"][
        "shared_scan_workspace_current_bytes_reported"
    ]
    assert not stats["contract"][
        "shared_scan_workspace_in_plan_owned_bytes"
    ]
    assert stats["contract"]["restriction_uses_deterministic_coarse_gather"]
    assert not stats["contract"]["floating_atomic_transfer_required"]
    assert not stats["contract"]["native_square_linear_operator_published"]


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
)
def test_private_rectangular_block_transfer_rejects_before_publish():
    duplicate_columns = _COLUMNS.copy()
    duplicate_columns[2] = 0
    with pytest.raises(RuntimeError, match="strictly increasing and unique"):
        _snapshot(columns=duplicate_columns)

    out_of_range = _COLUMNS.copy()
    out_of_range[-1] = 2
    with pytest.raises(RuntimeError, match="outside the transfer dimensions"):
        _snapshot(columns=out_of_range)

    nonfinite = _host_values()
    nonfinite[17] = np.nan
    with pytest.raises(RuntimeError, match="not finite"):
        _snapshot(values=nonfinite)

    snapshot = _snapshot()
    with pytest.raises(RuntimeError, match="capacity overflow"):
        _SparseBlockTransferGraphPlan(
            snapshot, explicit_array_capacity_bytes=707
        )


@pytest.mark.parametrize(
    "fine_block_size,coarse_block_size",
    [(2, 3), (3, 6), (6, 12), (12, 2)],
)
@test_utils.test(arch=[ti.cpu], offline_cache=False)
def test_private_rectangular_block_transfer_qualified_block_sizes(
    fine_block_size, coarse_block_size
):
    row_offsets_host = np.asarray([0, 1, 2], dtype=np.int32)
    columns_host = np.asarray([0, 0], dtype=np.int32)
    value_count = 2 * fine_block_size * coarse_block_size
    values_host = (
        np.arange(value_count, dtype=np.float32) * 0.01 + 0.125
    )
    snapshot = _DeviceBlockTransferSnapshot.copy_validated(
        fine_block_rows=2,
        coarse_block_rows=1,
        fine_block_size=fine_block_size,
        coarse_block_size=coarse_block_size,
        row_offsets=_i32_array(row_offsets_host),
        column_indices=_i32_array(columns_host),
        values=_f32_array(values_host),
        topology_version=13,
        numeric_version=17,
    )
    capacity = snapshot.total_reserved_bytes + 24 + 24
    plan = _SparseBlockTransferGraphPlan(
        snapshot, explicit_array_capacity_bytes=capacity
    )
    dense = values_host.reshape(2 * fine_block_size, coarse_block_size)
    coarse_host = np.linspace(
        -0.4, 0.8, coarse_block_size, dtype=np.float32
    )
    fine_host = np.linspace(
        0.7, -0.3, 2 * fine_block_size, dtype=np.float32
    )
    coarse_input = _f32_array(coarse_host)
    fine_output = ti.ndarray(ti.f32, shape=2 * fine_block_size)
    plan.prolongate(coarse_input, fine_output)
    np.testing.assert_allclose(
        fine_output.to_numpy(), dense @ coarse_host, rtol=2e-5, atol=2e-5
    )
    fine_input = _f32_array(fine_host)
    coarse_output = ti.ndarray(ti.f32, shape=coarse_block_size)
    plan.restrict(fine_input, coarse_output)
    np.testing.assert_allclose(
        coarse_output.to_numpy(), dense.T @ fine_host, rtol=2e-5, atol=2e-5
    )
    assert plan.debug_runtime_stats()["resources"][
        "build_peak_explicit_array_bytes"
    ] == capacity


@test_utils.test(arch=[ti.cpu], offline_cache=False)
def test_private_rectangular_block_transfer_alias_and_reset_guards():
    row_offsets = np.asarray([0, 1, 2, 3, 4], dtype=np.int32)
    columns = np.asarray([0, 0, 1, 1], dtype=np.int32)
    values = np.arange(4 * 18, dtype=np.float32) * 0.01 + 0.25
    snapshot = _DeviceBlockTransferSnapshot.copy_validated(
        fine_block_rows=4,
        coarse_block_rows=2,
        fine_block_size=3,
        coarse_block_size=6,
        row_offsets=_i32_array(row_offsets),
        column_indices=_i32_array(columns),
        values=_f32_array(values),
        topology_version=3,
        numeric_version=5,
    )
    plan = _SparseBlockTransferGraphPlan(
        snapshot, explicit_array_capacity_bytes=432
    )
    shared = ti.ndarray(ti.f32, shape=12)
    with pytest.raises(RuntimeError, match="alias is unsupported"):
        plan.prolongate(shared, shared)
    with pytest.raises(RuntimeError, match="alias is unsupported"):
        plan.restrict(shared, shared)
    assert plan.debug_runtime_stats()["operations"][
        "rejected_apply_calls"
    ] == 2

    ti.reset()
    with pytest.raises(RuntimeError, match="runtime has been reset"):
        snapshot.debug_runtime_stats()
    with pytest.raises(RuntimeError, match="runtime has been reset"):
        plan.debug_runtime_stats()
