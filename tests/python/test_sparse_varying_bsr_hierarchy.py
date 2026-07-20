import gc
import weakref

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils

from taichi_forge.linalg._sparse_block_transfer_graph import (
    _DeviceBlockTransferSnapshot,
)
from taichi_forge.linalg._sparse_bsr_graph_operator import _DeviceBsrSnapshot
from taichi_forge.linalg._sparse_varying_bsr_hierarchy import (
    _CallerProvidedVaryingBsrHierarchyBuilder,
    _VaryingBsrHierarchyLevelSpec,
)


_FIRST_ROW_OFFSETS = np.asarray([0, 1, 3, 4, 6, 7], dtype=np.int32)
_FIRST_COLUMNS = np.asarray([0, 0, 1, 1, 0, 1, 1], dtype=np.int32)


def _i32_array(values):
    result = ti.ndarray(ti.i32, shape=len(values))
    result.from_numpy(np.asarray(values, dtype=np.int32))
    return result


def _f32_array(values):
    result = ti.ndarray(ti.f32, shape=len(values))
    result.from_numpy(np.asarray(values, dtype=np.float32))
    return result


def _fine_dense_operator():
    rows = 15
    row = np.arange(rows, dtype=np.float64)[:, None]
    column = np.arange(rows, dtype=np.float64)[None, :]
    generator = 0.075 * np.sin(0.17 * (row + 1.0) * (column + 2.0))
    generator += 0.035 * np.cos(0.11 * (row + column + 3.0))
    dense = generator.T @ generator
    dense += np.eye(rows, dtype=np.float64) * 0.8
    return dense.astype(np.float32)


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


def _source():
    row_offsets, columns, values = _compress_square_blocks(
        _fine_dense_operator(), 3
    )
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


def _first_transfer_values():
    identity = np.eye(3, dtype=np.float32)
    blocks = np.zeros((len(_FIRST_COLUMNS), 3, 6), dtype=np.float32)
    blocks[0, :, :3] = identity
    blocks[1, :, 3:] = identity
    blocks[2, :, :3] = 0.05 * identity
    blocks[2, :, 3:] = -0.025 * identity
    blocks[3, :, :3] = identity
    blocks[4, :, :3] = 0.075 * identity
    blocks[4, :, 3:] = 0.025 * identity
    blocks[5, :, :3] = -0.04 * identity
    blocks[5, :, 3:] = 0.06 * identity
    blocks[6, :, 3:] = identity
    return blocks.reshape(-1)


def _first_transfer():
    return _DeviceBlockTransferSnapshot.copy_validated(
        fine_block_rows=5,
        coarse_block_rows=2,
        fine_block_size=3,
        coarse_block_size=6,
        row_offsets=_i32_array(_FIRST_ROW_OFFSETS),
        column_indices=_i32_array(_FIRST_COLUMNS),
        values=_f32_array(_first_transfer_values()),
        topology_version=13,
        numeric_version=17,
    )


def _second_transfer():
    values = np.concatenate(
        [
            np.eye(6, dtype=np.float32).reshape(-1),
            (0.75 * np.eye(6, dtype=np.float32)).reshape(-1),
        ]
    )
    return _DeviceBlockTransferSnapshot.copy_validated(
        fine_block_rows=2,
        coarse_block_rows=1,
        fine_block_size=6,
        coarse_block_size=6,
        row_offsets=_i32_array([0, 1, 2]),
        column_indices=_i32_array([0, 0]),
        values=_f32_array(values),
        topology_version=29,
        numeric_version=31,
    )


def _dense_first_transfer():
    blocks = _first_transfer_values().reshape(-1, 3, 6)
    dense = np.zeros((15, 12), dtype=np.float32)
    for fine_row in range(5):
        for offset in range(
            _FIRST_ROW_OFFSETS[fine_row], _FIRST_ROW_OFFSETS[fine_row + 1]
        ):
            coarse_row = int(_FIRST_COLUMNS[offset])
            dense[
                3 * fine_row : 3 * fine_row + 3,
                6 * coarse_row : 6 * coarse_row + 6,
            ] = blocks[offset]
    return dense


def _dense_second_transfer():
    return np.vstack(
        [
            np.eye(6, dtype=np.float32),
            0.75 * np.eye(6, dtype=np.float32),
        ]
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


def _specs(first, second, *, second_contributions=4):
    return (
        _VaryingBsrHierarchyLevelSpec(
            transfer=first,
            contribution_count=49,
            max_transfer_blocks_per_row=2,
            coarse_topology_version=19,
            coarse_numeric_version=23,
        ),
        _VaryingBsrHierarchyLevelSpec(
            transfer=second,
            contribution_count=second_contributions,
            max_transfer_blocks_per_row=1,
            coarse_topology_version=37,
            coarse_numeric_version=41,
        ),
    )


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
)
def test_private_varying_bsr_hierarchy_owns_exact_levels_and_transfers():
    source = _source()
    first = _first_transfer()
    second = _second_transfer()
    source_ref = weakref.ref(source)
    first_ref = weakref.ref(first)
    second_ref = weakref.ref(second)
    builder = _CallerProvidedVaryingBsrHierarchyBuilder(
        explicit_array_capacity_bytes=3884,
        bottom_scalar_size_cap=6,
    )
    hierarchy = builder.build(
        source,
        _specs(first, second),
        topology_version=43,
        numeric_version=47,
    )

    first_oracle = (
        _dense_first_transfer().T
        @ _fine_dense_operator()
        @ _dense_first_transfer()
    )
    second_oracle = (
        _dense_second_transfer().T
        @ first_oracle
        @ _dense_second_transfer()
    )
    np.testing.assert_allclose(
        _snapshot_dense(hierarchy._levels[1]),
        first_oracle,
        rtol=3e-4,
        atol=3e-4,
    )
    np.testing.assert_allclose(
        _snapshot_dense(hierarchy._levels[2]),
        second_oracle,
        rtol=5e-4,
        atol=5e-4,
    )

    stats = hierarchy.debug_runtime_stats()
    assert stats["identity"] == {
        "backend_family": stats["identity"]["backend_family"],
        "level_count": 3,
        "transition_count": 2,
        "level_block_rows": (5, 2, 1),
        "level_block_sizes": (3, 6, 6),
        "level_block_nnz": (25, 4, 1),
        "level_scalar_rows": (15, 12, 6),
        "transfer_block_nnz": (7, 2),
        "contribution_counts": (49, 4),
        "max_transfer_row_degrees": (2, 1),
        "topology_version": 43,
        "numeric_version": 47,
    }
    resources = dict(stats["resources"])
    shared_sort_scan_bytes = resources.pop(
        "shared_sort_scan_workspace_bytes"
    )
    shared_sort_scan_scope = resources.pop(
        "shared_sort_scan_workspace_ownership_scope"
    )
    detailed_workspace = (
        ti.lang.impl.get_runtime().prog._primitive_workspace_detailed_stats()
    )
    expected_sort_scan_bytes = sum(
        group["reserved_bytes"]
        for group in detailed_workspace["groups"]
        if group["family"] in ("ordering", "ordering_aux", "scan")
    )
    assert shared_sort_scan_bytes == expected_sort_scan_bytes
    assert (
        shared_sort_scan_scope
        == "program_ordering_ordering_aux_scan_arena"
    )
    assert resources == {
        "level_operator_reserved_bytes": (1024, 604, 156),
        "transfer_reserved_bytes_by_level": (556, 308),
        "operator_reserved_bytes": 1784,
        "transfer_reserved_bytes": 864,
        "steady_reserved_bytes": 2648,
        "preflight_steady_upper_bytes": 2648,
        "preflight_build_peak_upper_bytes": 3884,
        "actual_build_peak_excluding_workspace_bytes": 3884,
        "peak_retired_builder_staging_bytes": 1392,
    }
    assert stats["transfers"] == {
        "device_to_host_bytes": 48,
        "device_to_device_bytes": 0,
        "device_kernel_publish_bytes": 760,
        "device_payload_readback_bytes": 0,
    }
    assert stats["per_transition"][0][
        "phase_peak_excluding_workspace_bytes"
    ] == 3884
    assert stats["per_transition"][1][
        "phase_peak_excluding_workspace_bytes"
    ] == 2792
    assert stats["contract"]["fine_and_coarse_block_sizes_may_differ"]
    assert stats["contract"][
        "shared_sort_scan_workspace_current_bytes_reported"
    ]
    assert not stats["contract"][
        "shared_sort_scan_workspace_in_explicit_capacity"
    ]
    assert not stats["contract"]["workspace_total_bytes_reported"]
    assert stats["contract"][
        "rectangular_transfers_replace_implicit_aggregate_maps"
    ]
    assert not stats["contract"]["transfer_graph_plans_constructed"]
    assert not stats["contract"]["recursive_vcycle_constructed"]

    del source, first, second
    gc.collect()
    assert source_ref() is hierarchy._levels[0]
    assert first_ref() is hierarchy._transfers[0]
    assert second_ref() is hierarchy._transfers[1]

    builder_stats = builder.debug_runtime_stats()
    assert builder_stats["operations"] == {
        "build_calls": 1,
        "successful_builds": 1,
        "failed_builds": 0,
        "last_retired_transition_builder_count": 2,
    }
    assert builder_stats["resources"][
        "last_actual_build_peak_excluding_workspace_bytes"
    ] == 3884
    assert builder_stats["resources"]["last_steady_reserved_bytes"] == 2648
    assert builder_stats["resources"]["live_snapshot_count"] == 1
    assert (
        builder_stats["resources"]["shared_sort_scan_workspace_bytes"]
        == expected_sort_scan_bytes
    )
    assert builder_stats["resources"][
        "shared_sort_scan_workspace_ownership_scope"
    ] == "program_ordering_ordering_aux_scan_arena"
    assert builder_stats["contract"][
        "shared_sort_scan_workspace_current_bytes_reported"
    ]
    assert not builder_stats["contract"][
        "shared_sort_scan_workspace_in_explicit_capacity"
    ]
    assert not builder_stats["contract"]["workspace_total_bytes_reported"]


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_private_varying_bsr_hierarchy_rejects_atomically_and_preserves_old():
    source = _source()
    first = _first_transfer()
    second = _second_transfer()
    builder = _CallerProvidedVaryingBsrHierarchyBuilder(
        explicit_array_capacity_bytes=3884,
        bottom_scalar_size_cap=6,
    )
    old = builder.build(
        source,
        _specs(first, second),
        topology_version=43,
        numeric_version=47,
    )
    old_bottom = _snapshot_dense(old._levels[-1]).copy()
    with pytest.raises(RuntimeError, match="contribution count"):
        builder.build(
            source,
            _specs(first, second, second_contributions=3),
            topology_version=53,
            numeric_version=59,
        )
    np.testing.assert_array_equal(_snapshot_dense(old._levels[-1]), old_bottom)
    stats = builder.debug_runtime_stats()
    assert stats["operations"]["build_calls"] == 2
    assert stats["operations"]["successful_builds"] == 1
    assert stats["operations"]["failed_builds"] == 1
    assert stats["operations"]["last_retired_transition_builder_count"] == 1
    assert stats["resources"]["live_snapshot_count"] == 1
    assert len(stats["last_per_transition"]) == 1

    capacity_builder = _CallerProvidedVaryingBsrHierarchyBuilder(
        explicit_array_capacity_bytes=3883,
        bottom_scalar_size_cap=6,
    )
    with pytest.raises(RuntimeError, match="capacity overflow during preflight"):
        capacity_builder.build(
            source,
            _specs(first, second),
            topology_version=43,
            numeric_version=47,
        )
    assert capacity_builder.debug_runtime_stats()["operations"] == {
        "build_calls": 1,
        "successful_builds": 0,
        "failed_builds": 1,
        "last_retired_transition_builder_count": 0,
    }

    bottom_builder = _CallerProvidedVaryingBsrHierarchyBuilder(
        explicit_array_capacity_bytes=3884,
        bottom_scalar_size_cap=5,
    )
    with pytest.raises(RuntimeError, match="bottom scalar size exceeds cap"):
        bottom_builder.build(
            source,
            _specs(first, second),
            topology_version=43,
            numeric_version=47,
        )


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_private_varying_bsr_hierarchy_rejects_after_reset():
    source = _source()
    first = _first_transfer()
    second = _second_transfer()
    builder = _CallerProvidedVaryingBsrHierarchyBuilder(
        explicit_array_capacity_bytes=3884,
        bottom_scalar_size_cap=6,
    )
    hierarchy = builder.build(
        source,
        _specs(first, second),
        topology_version=43,
        numeric_version=47,
    )
    ti.reset()
    with pytest.raises(RuntimeError, match="runtime has been reset"):
        hierarchy.debug_runtime_stats()
    with pytest.raises(RuntimeError, match="runtime has been reset"):
        builder.debug_runtime_stats()
