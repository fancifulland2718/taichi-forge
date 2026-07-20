import gc

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.linalg._sparse_hierarchy_assembly import (
    _DeviceCsrSnapshot,
    _SparseGalerkinCsrAssemblyPlan,
)
from taichi_forge.linalg._sparse_csr_graph_operator import (
    _SparseCsrGraphOperatorPlan,
)
from taichi_forge.linalg._sparse_hierarchy_candidate import (
    _AggregateRestrictionSchedulePlan,
    _CallerCoarsenedSparseHierarchyBuilder,
)
from taichi_forge.linalg._sparse_solve_publication import (
    _SparseSolvePublicationRegistry,
)
from taichi_forge.linalg._sparse_vcycle_graph import (
    _SparseRecursiveVcycleGraphPlan,
    _SparseVcycleNumericSnapshot,
)
from taichi_forge.linalg._sparse_vcycle_solve import (
    _SparseVcycleSolvePublicationBuilder,
)
from tests import test_utils
from tests.sparse_runtime_stats import (
    assert_sparse_graph_cache_attribution,
)


_ROW_OFFSETS = np.asarray(
    [0, 3, 6, 9, 12, 15, 18, 21, 24], dtype=np.int32
)
_COLUMN_INDICES = np.asarray(
    [
        0,
        2,
        3,
        1,
        4,
        7,
        0,
        2,
        5,
        0,
        3,
        6,
        1,
        4,
        6,
        2,
        5,
        7,
        3,
        4,
        6,
        1,
        5,
        7,
    ],
    dtype=np.int32,
)
_VALUES = np.asarray(
    [
        2.2,
        -0.7,
        -0.5,
        3.1,
        -0.8,
        -1.3,
        -0.7,
        2.8,
        -1.1,
        -0.5,
        2.4,
        -0.9,
        -0.8,
        3.0,
        -1.2,
        -1.1,
        2.7,
        -0.6,
        -0.9,
        -1.2,
        3.1,
        -1.3,
        -0.6,
        2.9,
    ],
    dtype=np.float32,
)
_AGGREGATE = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)
_COARSE_ROW_OFFSETS = np.asarray([0, 4, 8, 12, 16], dtype=np.int32)
_COARSE_COLUMN_INDICES = np.tile(
    np.arange(4, dtype=np.int32), 4
)
_COARSE_VALUES = np.asarray(
    [
        5.3,
        -1.2,
        -0.8,
        -1.3,
        -1.2,
        5.2,
        -1.1,
        -0.9,
        -0.8,
        -1.1,
        5.7,
        -1.8,
        -1.3,
        -0.9,
        -1.8,
        6.0,
    ],
    dtype=np.float32,
)
_BOTTOM_ROW_OFFSETS = np.asarray([0, 2, 4], dtype=np.int32)
_BOTTOM_COLUMN_INDICES = np.asarray([0, 1, 0, 1], dtype=np.int32)
_BOTTOM_VALUES = np.asarray([8.1, -4.1, -4.1, 8.1], dtype=np.float32)


def _source_arrays(source_values=_VALUES):
    row_offsets = ti.ndarray(ti.i32, shape=_ROW_OFFSETS.size)
    column_indices = ti.ndarray(ti.i32, shape=_COLUMN_INDICES.size)
    values = ti.ndarray(ti.f32, shape=_VALUES.size)
    row_offsets.from_numpy(_ROW_OFFSETS)
    column_indices.from_numpy(_COLUMN_INDICES)
    values.from_numpy(np.asarray(source_values, dtype=np.float32))
    return row_offsets, column_indices, values


def _aggregate_array(values=_AGGREGATE):
    aggregate = ti.ndarray(ti.i32, shape=len(values))
    aggregate.from_numpy(np.asarray(values, dtype=np.int32))
    return aggregate


def _f32_array(values):
    values = np.asarray(values, dtype=np.float32)
    result = ti.ndarray(ti.f32, shape=values.size)
    result.from_numpy(values.reshape(-1))
    return result


def _dense_csr(row_offsets, column_indices, values):
    size = len(row_offsets) - 1
    dense = np.zeros((size, size), dtype=np.float64)
    for row in range(size):
        for offset in range(row_offsets[row], row_offsets[row + 1]):
            dense[row, column_indices[offset]] = values[offset]
    return dense


def _vcycle_numeric_numpy(scale=1.0):
    fine = _dense_csr(
        _ROW_OFFSETS, _COLUMN_INDICES, scale * _VALUES
    )
    coarse = _dense_csr(
        _COARSE_ROW_OFFSETS,
        _COARSE_COLUMN_INDICES,
        scale * _COARSE_VALUES,
    )
    inverse_diagonal_numpy = []
    damping_numpy = []
    for dense in (fine, coarse):
        diagonal = np.diag(dense).copy()
        inverse_diagonal = (1.0 / diagonal).astype(np.float32)
        inverse_sqrt = 1.0 / np.sqrt(diagonal)
        normalized = inverse_sqrt[:, None] * dense * inverse_sqrt[None, :]
        damping = np.asarray(
            [1.0 / np.max(np.sum(np.abs(normalized), axis=1))],
            dtype=np.float32,
        )
        inverse_diagonal_numpy.append(inverse_diagonal)
        damping_numpy.append(damping)
    bottom = _dense_csr(
        _BOTTOM_ROW_OFFSETS,
        _BOTTOM_COLUMN_INDICES,
        scale * _BOTTOM_VALUES,
    )
    bottom_inverse_numpy = np.linalg.inv(bottom)
    bottom_inverse_numpy = (
        0.5 * (bottom_inverse_numpy + bottom_inverse_numpy.T)
    ).astype(np.float32)
    return (
        inverse_diagonal_numpy,
        damping_numpy,
        bottom_inverse_numpy.reshape(-1),
    )


def _vcycle_numeric_sources(scale=1.0):
    inverse_diagonal_numpy, damping_numpy, bottom_inverse_numpy = (
        _vcycle_numeric_numpy(scale)
    )
    inverse_diagonals = [
        _f32_array(value) for value in inverse_diagonal_numpy
    ]
    dampings = [_f32_array(value) for value in damping_numpy]
    return (
        inverse_diagonals,
        dampings,
        _f32_array(bottom_inverse_numpy),
        inverse_diagonal_numpy,
        damping_numpy,
        bottom_inverse_numpy,
    )


def _vcycle_reference(rhs, scale=1.0):
    fine = _dense_csr(
        _ROW_OFFSETS, _COLUMN_INDICES, scale * _VALUES
    )
    coarse = _dense_csr(
        _COARSE_ROW_OFFSETS,
        _COARSE_COLUMN_INDICES,
        scale * _COARSE_VALUES,
    )
    bottom = _dense_csr(
        _BOTTOM_ROW_OFFSETS,
        _BOTTOM_COLUMN_INDICES,
        scale * _BOTTOM_VALUES,
    )
    inverse_diagonals, dampings, bottom_inverse = (
        _vcycle_numeric_numpy(scale)
    )
    matrices = (fine, coarse)
    maps = (
        _AGGREGATE,
        np.asarray([0, 0, 1, 1], dtype=np.int32),
    )

    def apply(level_index, level_rhs):
        if level_index == len(matrices):
            return bottom_inverse.reshape(bottom.shape) @ level_rhs
        matrix = matrices[level_index]
        inverse_diagonal = inverse_diagonals[level_index].astype(np.float64)
        damping = float(dampings[level_index][0])
        pre_solution = damping * inverse_diagonal * level_rhs
        residual = level_rhs - matrix @ pre_solution
        fine_to_coarse = maps[level_index]
        coarse_rhs = np.zeros(matrix.shape[0] // 2, dtype=np.float64)
        for fine_row, coarse_row in enumerate(fine_to_coarse):
            coarse_rhs[coarse_row] += residual[fine_row]
        coarse_solution = apply(level_index + 1, coarse_rhs)
        corrected = pre_solution + coarse_solution[fine_to_coarse]
        residual = level_rhs - matrix @ corrected
        return corrected + damping * inverse_diagonal * residual

    return apply(0, np.asarray(rhs, dtype=np.float64))


def _snapshot_payload(snapshot):
    return (
        snapshot._row_offsets.to_numpy(),
        snapshot._column_indices.to_numpy(),
        snapshot._values.to_numpy(),
    )


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
)
def test_private_galerkin_plan_publishes_exact_owned_csr_transactionally():
    row_offsets, column_indices, values = _source_arrays()
    invalid_row_offsets = _ROW_OFFSETS.copy()
    invalid_row_offsets[-1] -= 1
    row_offsets.from_numpy(invalid_row_offsets)
    with pytest.raises(TaichiRuntimeError, match="canonical"):
        _DeviceCsrSnapshot.copy_validated(
            rows=8,
            cols=8,
            row_offsets=row_offsets,
            column_indices=column_indices,
            values=values,
            topology_version=1,
            numeric_version=1,
        )
    row_offsets.from_numpy(_ROW_OFFSETS)

    source = _DeviceCsrSnapshot.copy_validated(
        rows=8,
        cols=8,
        row_offsets=row_offsets,
        column_indices=column_indices,
        values=values,
        topology_version=1,
        numeric_version=1,
    )
    source_stats = source.debug_runtime_stats()
    assert source_stats["resources"] == {
        "pattern_reserved_bytes": 132,
        "value_reserved_bytes": 96,
        "total_reserved_bytes": 228,
    }
    assert source_stats["transfers"] == {
        "device_to_host_bytes": 8,
        "device_to_device_bytes": 228,
        "device_payload_readback_bytes": 0,
    }

    # The validated source owns its copy and is not changed by caller reuse.
    values.fill(123.0)
    aggregate = _aggregate_array()
    plan = _SparseGalerkinCsrAssemblyPlan(
        fine_rows=8, coarse_rows=4, capacity=24
    )
    first = plan.build(
        source,
        aggregate,
        topology_version=2,
        numeric_version=1,
    )
    first_stats = first.debug_runtime_stats()
    assert first_stats["identity"]["construction"] == (
        "galerkin_exact_prefix_publish"
    )
    assert first_stats["resources"] == {
        "pattern_reserved_bytes": 84,
        "value_reserved_bytes": 64,
        "total_reserved_bytes": 148,
    }
    assert first_stats["transfers"] == {
        "device_to_host_bytes": 0,
        "device_to_device_bytes": 148,
        "device_payload_readback_bytes": 0,
    }

    # Reuse staging before reading the first output. Program submission order
    # must finish its exact-prefix copy before the next build mutates staging.
    second = plan.build(
        source,
        aggregate,
        topology_version=2,
        numeric_version=1,
    )
    first_payload = _snapshot_payload(first)
    second_payload = _snapshot_payload(second)
    np.testing.assert_array_equal(first_payload[0], _COARSE_ROW_OFFSETS)
    np.testing.assert_array_equal(first_payload[1], _COARSE_COLUMN_INDICES)
    np.testing.assert_allclose(
        first_payload[2], _COARSE_VALUES, rtol=0.0, atol=1e-6
    )
    for actual, expected in zip(second_payload, first_payload):
        np.testing.assert_array_equal(actual, expected)

    next_plan = _SparseGalerkinCsrAssemblyPlan(
        fine_rows=4, coarse_rows=2, capacity=16
    )
    bottom = next_plan.build(
        first,
        _aggregate_array(np.asarray([0, 0, 1, 1], dtype=np.int32)),
        topology_version=3,
        numeric_version=1,
    )
    bottom_payload = _snapshot_payload(bottom)
    np.testing.assert_array_equal(bottom_payload[0], _BOTTOM_ROW_OFFSETS)
    np.testing.assert_array_equal(
        bottom_payload[1], _BOTTOM_COLUMN_INDICES
    )
    np.testing.assert_allclose(
        bottom_payload[2], _BOTTOM_VALUES, rtol=0.0, atol=1e-6
    )
    next_stats = next_plan.debug_runtime_stats()
    assert next_stats["resources"]["persistent_staging_reserved_bytes"] == 560
    assert next_stats["resources"]["last_output_pattern_bytes"] == 28
    assert next_stats["resources"]["last_output_value_bytes"] == 16
    assert next_stats["resources"][
        "last_build_peak_excluding_workspace_bytes"
    ] == 604
    assert next_stats["transfers"] == {
        "device_to_host_bytes": 8,
        "device_to_device_bytes": 60,
        "device_payload_readback_bytes": 0,
        "control_readback_bytes_per_build": 8,
    }

    invalid_aggregate = _AGGREGATE.copy()
    invalid_aggregate[-1] = 4
    with pytest.raises(TaichiRuntimeError, match="before publish"):
        plan.build(
            source,
            _aggregate_array(invalid_aggregate),
            topology_version=3,
            numeric_version=1,
        )
    for actual, expected in zip(_snapshot_payload(first), first_payload):
        np.testing.assert_array_equal(actual, expected)

    overflow_values = _VALUES.copy()
    overflow_values[0] = np.float32(3.0e38)
    overflow_values[3] = np.float32(3.0e38)
    overflow_rows, overflow_columns, overflow_data = _source_arrays(
        overflow_values
    )
    overflow_source = _DeviceCsrSnapshot.copy_validated(
        rows=8,
        cols=8,
        row_offsets=overflow_rows,
        column_indices=overflow_columns,
        values=overflow_data,
        topology_version=1,
        numeric_version=2,
    )
    with pytest.raises(TaichiRuntimeError, match="duplicate sum"):
        plan.build(
            overflow_source,
            aggregate,
            topology_version=2,
            numeric_version=2,
        )
    for actual, expected in zip(_snapshot_payload(first), first_payload):
        np.testing.assert_array_equal(actual, expected)

    stats = plan.debug_runtime_stats()
    assert stats["status"] == {
        "last_status": 3,
        "last_unique_nnz": 0,
        "last_duplicate_triplets": 0,
    }
    assert stats["operations"] == {
        "build_calls": 4,
        "successful_builds": 2,
        "failed_builds": 2,
        "workspace_builds": 1,
        "workspace_reuses": 3,
        "host_control_readbacks": 4,
    }
    resources = stats["resources"]
    assert resources["persistent_staging_reserved_bytes"] == 848
    assert resources["last_output_pattern_bytes"] == 84
    assert resources["last_output_value_bytes"] == 64
    assert resources["last_build_peak_excluding_workspace_bytes"] == 996
    assert resources["live_snapshot_count"] == 2
    assert resources["live_snapshot_reserved_bytes"] == 296
    assert resources["known_workspace_reported_bytes"] >= 24 * 4
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
        "device_to_host_bytes": 32,
        "device_to_device_bytes": 424,
        "device_payload_readback_bytes": 0,
        "control_readback_bytes_per_build": 8,
    }
    assert stats["contract"]["exact_sized_snapshot_publish"]
    assert stats["contract"][
        "failed_build_does_not_mutate_returned_snapshots"
    ]
    assert stats["contract"]["native_provider_required_without_host_fallback"]
    assert not stats["contract"]["workspace_total_bytes_reported"]
    assert stats["contract"][
        "shared_scan_workspace_current_bytes_reported"
    ]
    assert not stats["contract"][
        "shared_scan_workspace_in_plan_owned_bytes"
    ]

    del second
    gc.collect()
    assert plan.debug_runtime_stats()["resources"]["live_snapshot_count"] == 1


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_private_galerkin_snapshot_and_plan_reject_after_reset():
    row_offsets, column_indices, values = _source_arrays()
    source = _DeviceCsrSnapshot.copy_validated(
        rows=8,
        cols=8,
        row_offsets=row_offsets,
        column_indices=column_indices,
        values=values,
        topology_version=1,
        numeric_version=1,
    )
    plan = _SparseGalerkinCsrAssemblyPlan(
        fine_rows=8, coarse_rows=4, capacity=24
    )
    ti.reset()
    ti.init(arch=ti.cpu, enable_fallback=False)

    with pytest.raises(TaichiRuntimeError, match="runtime has been reset"):
        source.debug_runtime_stats()
    with pytest.raises(TaichiRuntimeError, match="runtime has been reset"):
        plan.debug_runtime_stats()


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
    vulkan_sparse_experimental=True,
)
def test_private_csr_graph_target_publishes_and_refreshes_without_host_pack():
    row_offsets, column_indices, values = _source_arrays()
    source = _DeviceCsrSnapshot.copy_validated(
        rows=8,
        cols=8,
        row_offsets=row_offsets,
        column_indices=column_indices,
        values=values,
        topology_version=7,
        numeric_version=11,
    )
    with pytest.raises(TaichiRuntimeError, match="capacity overflow"):
        _SparseCsrGraphOperatorPlan(
            source,
            explicit_array_capacity_bytes=455,
        )
    plan = _SparseCsrGraphOperatorPlan(
        source,
        explicit_array_capacity_bytes=500,
    )
    stats = plan.debug_runtime_stats()
    assert stats["identity"] == {
        "backend_family": source._backend,
        "method": "single_dispatch_csr_graph",
        "size": 8,
        "nnz": 24,
        "topology_version": 7,
        "numeric_version": 11,
    }
    assert stats["operations"] == {
        "apply_calls": 0,
        "rejected_apply_calls": 0,
        "graph_node_count": 1,
        "graph_dispatch_count": 1,
        "host_graph_submissions_per_apply": 1,
        "explicit_apply_host_synchronizations": 0,
        "native_operator_publishes": 0,
    }
    assert assert_sparse_graph_cache_attribution(
        stats, expected_cache_object_count=1
    ) == {
        "borrowed_snapshot_reserved_bytes": 228,
        "topology_argument_reserved_bytes": 132,
        "numeric_argument_reserved_bytes": 96,
        "native_operator_reserved_bytes": 228,
        "build_peak_explicit_array_bytes": 456,
        "explicit_array_capacity_bytes": 500,
    }
    assert stats["contract"]["no_host_pattern_pack"]
    assert stats["contract"]["no_host_payload_readback"]

    input_numpy = np.asarray(
        [1.0, -0.5, 2.0, -1.5, 0.75, -2.5, 1.25, -0.25],
        dtype=np.float32,
    )
    input_array = _f32_array(input_numpy)
    output = ti.ndarray(ti.f32, shape=8)
    output.fill(0.0)
    plan.apply(input_array, output)
    ti.sync()
    expected = _dense_csr(
        _ROW_OFFSETS, _COLUMN_INDICES, _VALUES
    ) @ input_numpy
    np.testing.assert_allclose(
        output.to_numpy(), expected, rtol=0.0, atol=1e-6
    )
    with pytest.raises(TaichiRuntimeError, match="alias"):
        plan.apply(input_array, input_array)

    publisher = plan.create_numeric_publisher()
    publisher_stats = publisher.debug_runtime_stats()
    assert publisher_stats["host_topology_metadata_bytes"] == 12
    assert publisher_stats["device_reserved_bytes"] == 0
    assert publisher_stats["numeric_role_count"] == 1
    assert publisher_stats["numeric_payload_bytes"] == 96
    native = plan.create_native_operator()
    with pytest.raises(TaichiRuntimeError, match="at most one"):
        plan.create_native_operator()
    published_stats = plan.debug_runtime_stats()
    assert published_stats["operations"]["apply_calls"] == 1
    assert published_stats["operations"]["rejected_apply_calls"] == 1
    assert published_stats["operations"]["native_operator_publishes"] == 1
    assert published_stats["transfers"] == {
        "device_to_host_bytes": 0,
        "device_to_device_bytes": 228,
        "device_payload_readback_bytes": 0,
    }

    program = ti.lang.impl.get_runtime().prog
    output.fill(0.0)
    native.spmv(program, input_array.arr, output.arr)
    ti.sync()
    np.testing.assert_allclose(
        output.to_numpy(), expected, rtol=0.0, atol=1e-6
    )
    native_stats = native._debug_runtime_stats()
    assert native_stats["resources"]["pattern_reserved_bytes"] == 132
    assert native_stats["resources"]["values_reserved_bytes"] == 96
    assert native_stats["resources"]["spmv_workspace_reserved_bytes"] == 0
    assert native_stats["resources"]["operator_owned_reserved_bytes"] == 228

    scaled_rows, scaled_columns, scaled_values = _source_arrays(2.0 * _VALUES)
    scaled_source = _DeviceCsrSnapshot.copy_validated(
        rows=8,
        cols=8,
        row_offsets=scaled_rows,
        column_indices=scaled_columns,
        values=scaled_values,
        topology_version=7,
        numeric_version=12,
    )
    with pytest.raises(TaichiRuntimeError, match="topology version"):
        publisher.bind_source(
            scaled_source,
            expected_topology_version=8,
            expected_numeric_version=11,
        )
    numeric_source = publisher.bind_source(
        scaled_source,
        expected_topology_version=7,
        expected_numeric_version=11,
    )
    native.update_numeric_data(program, {"values": numeric_source}, 7, 11)
    output.fill(0.0)
    native.spmv(program, input_array.arr, output.arr)
    ti.sync()
    np.testing.assert_allclose(
        output.to_numpy(), 2.0 * expected, rtol=0.0, atol=2e-6
    )
    refreshed_stats = native._debug_runtime_stats()
    assert refreshed_stats["identity"]["numeric_version"] == 12
    assert refreshed_stats["operations"]["numeric_updates"] == 1
    assert refreshed_stats["resources"][
        "numeric_update_peak_temporary_bytes"
    ] == 96
    assert refreshed_stats["resources"][
        "operator_owned_reserved_bytes"
    ] == 228
    assert refreshed_stats["transfers"]["device_to_device_bytes"] == 324
    final_publisher_stats = publisher.debug_runtime_stats()
    assert final_publisher_stats["operations"] == {
        "bind_calls": 1,
        "rejected_bind_calls": 1,
    }


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
)
def test_private_restriction_schedule_stably_inverts_unordered_map():
    unordered = np.asarray([2, 0, 1, 2, 0, 1, 3, 0], dtype=np.int32)
    aggregate = _aggregate_array(unordered)
    plan = _AggregateRestrictionSchedulePlan(
        fine_rows=8,
        coarse_rows=4,
    )
    first = plan.build(aggregate)
    second = plan.build(aggregate)
    expected_offsets = np.asarray([0, 3, 5, 7, 8], dtype=np.int32)
    expected_rows = np.asarray([1, 4, 7, 2, 5, 0, 3, 6], dtype=np.int32)
    for schedule in (first, second):
        np.testing.assert_array_equal(
            schedule._coarse_offsets.to_numpy(), expected_offsets
        )
        np.testing.assert_array_equal(
            schedule._ordered_fine_rows.to_numpy(), expected_rows
        )
        stats = schedule.debug_runtime_stats()
        assert stats["resources"] == {
            "coarse_offsets_reserved_bytes": 20,
            "ordered_fine_rows_reserved_bytes": 32,
            "total_reserved_bytes": 52,
        }
        assert stats["contract"][
            "deterministic_gather_within_each_aggregate"
        ]
        assert not stats["contract"][
            "floating_atomic_restriction_required"
        ]

    plan_stats = plan.debug_runtime_stats()
    assert plan_stats["operations"]["build_calls"] == 2
    assert plan_stats["resources"][
        "persistent_staging_reserved_bytes"
    ] == 64
    detailed_workspace = (
        ti.lang.impl.get_runtime().prog._primitive_workspace_detailed_stats()
    )
    expected_scan_bytes = sum(
        group["reserved_bytes"]
        for group in detailed_workspace["groups"]
        if group["family"] == "scan"
    )
    assert plan_stats["resources"][
        "shared_scan_workspace_bytes"
    ] == expected_scan_bytes
    assert plan_stats["resources"][
        "shared_scan_workspace_ownership_scope"
    ] == "program_scan_arena"
    assert plan_stats["transfers"] == {
        "device_to_host_bytes": 0,
        "device_to_device_bytes": 0,
        "device_kernel_publish_bytes": 104,
    }
    assert plan_stats["contract"][
        "native_provider_required_without_host_fallback"
    ]
    assert plan_stats["contract"][
        "shared_scan_workspace_current_bytes_reported"
    ]
    assert not plan_stats["contract"][
        "shared_scan_workspace_in_plan_owned_bytes"
    ]


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
)
def test_private_caller_coarsened_hierarchy_owns_bounded_exact_levels():
    row_offsets, column_indices, values = _source_arrays()
    source = _DeviceCsrSnapshot.copy_validated(
        rows=8,
        cols=8,
        row_offsets=row_offsets,
        column_indices=column_indices,
        values=values,
        topology_version=7,
        numeric_version=11,
    )
    first_map = _aggregate_array()
    second_map_values = np.asarray([0, 0, 1, 1], dtype=np.int32)
    second_map = _aggregate_array(second_map_values)
    builder = _CallerCoarsenedSparseHierarchyBuilder(
        explicit_array_capacity_bytes=2500,
        bottom_size_cap=2,
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
        "storage_format": "recursive_csr",
        "dtype": "f32",
        "index_dtype": "i32",
        "topology_version": 7,
        "numeric_version": 11,
        "level_count": 3,
        "level_sizes": (8, 4, 2),
        "level_nnz": (24, 16, 4),
        "bottom_size_cap": 2,
    }
    assert stats["resources"] == {
        "operator_reserved_bytes": 420,
        "aggregate_map_reserved_bytes": 48,
        "restriction_schedule_reserved_bytes": 80,
        "steady_reserved_bytes": 548,
        "level_operator_reserved_bytes": (228, 148, 44),
        "level_map_reserved_bytes": (32, 16),
        "level_restriction_schedule_reserved_bytes": (52, 28),
    }
    report = stats["build"]
    assert report["preflight"]["steady_exact_upper_bytes"] == 772
    assert report["preflight"][
        "build_peak_excluding_workspace_upper_bytes"
    ] == 2472
    assert report["steady_exact_bytes"] == 548
    assert report[
        "actual_build_peak_excluding_workspace_bytes"
    ] == 1992
    assert report["staging_overlap_peak_bytes"] == 1472
    assert report["control_readbacks"] == 2
    assert report["final_completion_synchronizations"] == 1
    assert report["device_to_host_bytes"] == 16
    assert report["device_to_device_bytes"] == 288
    assert report["device_kernel_publish_bytes"] == 80
    assert report["device_payload_readback_bytes"] == 0
    assert report["retired_plan_count"] == 4
    assert report["live_plan_count_after_publish"] == 0
    assert not report["workspace_total_bytes_reported"]
    assert tuple(
        level["galerkin_staging_bytes"] for level in report["levels"]
    ) == (
        848,
        560,
    )
    assert tuple(
        level["restriction_schedule_staging_bytes"]
        for level in report["levels"]
    ) == (64, 32)
    assert tuple(level["output_bytes"] for level in report["levels"]) == (
        148,
        44,
    )
    assert tuple(
        level["restriction_schedule_bytes"] for level in report["levels"]
    ) == (52, 28)

    # The published hierarchy owns aggregate maps even if callers reuse the
    # input ndarrays after publication.
    first_map.fill(3)
    second_map.fill(1)
    np.testing.assert_array_equal(
        hierarchy._aggregate_maps[0].to_numpy(), _AGGREGATE
    )
    np.testing.assert_array_equal(
        hierarchy._aggregate_maps[1].to_numpy(), second_map_values
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
    bottom_payload = _snapshot_payload(hierarchy._levels[-1])
    np.testing.assert_array_equal(bottom_payload[0], _BOTTOM_ROW_OFFSETS)
    np.testing.assert_array_equal(
        bottom_payload[1], _BOTTOM_COLUMN_INDICES
    )
    np.testing.assert_allclose(
        bottom_payload[2], _BOTTOM_VALUES, rtol=0.0, atol=1e-6
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
    ] == 548
    assert builder_stats["transfers"] == {
        "device_to_host_bytes": 16,
        "device_to_device_bytes": 288,
        "device_kernel_publish_bytes": 80,
        "device_payload_readback_bytes": 0,
    }
    assert builder_stats["contract"][
        "at_most_two_neighbor_staging_plans"
    ]
    assert builder_stats["contract"]["one_final_completion_sync"]
    assert builder_stats["contract"][
        "failed_build_publishes_no_partial_hierarchy"
    ]
    assert builder_stats["contract"][
        "deterministic_restriction_without_float_atomics"
    ]


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
)
def test_private_vcycle_numeric_snapshot_owns_validated_device_resources():
    row_offsets, column_indices, values = _source_arrays()
    source = _DeviceCsrSnapshot.copy_validated(
        rows=8,
        cols=8,
        row_offsets=row_offsets,
        column_indices=column_indices,
        values=values,
        topology_version=7,
        numeric_version=11,
    )
    hierarchy = _CallerCoarsenedSparseHierarchyBuilder(
        explicit_array_capacity_bytes=2500,
        bottom_size_cap=2,
    ).build(
        source,
        [
            (4, _aggregate_array()),
            (
                2,
                _aggregate_array(
                    np.asarray([0, 0, 1, 1], dtype=np.int32)
                ),
            ),
        ],
        topology_version=7,
        numeric_version=11,
    )
    (
        inverse_diagonals,
        dampings,
        bottom_inverse,
        inverse_diagonal_numpy,
        damping_numpy,
        bottom_inverse_numpy,
    ) = _vcycle_numeric_sources()
    numeric = _SparseVcycleNumericSnapshot.copy_validated(
        hierarchy,
        inverse_diagonals=inverse_diagonals,
        dampings=dampings,
        bottom_inverse=bottom_inverse,
        topology_version=7,
        numeric_version=11,
    )
    stats = numeric.debug_runtime_stats()
    assert stats["identity"] == {
        "backend_family": source._backend,
        "method": "scalar_jacobi_dense_bottom_inverse",
        "topology_version": 7,
        "numeric_version": 11,
        "level_sizes": (8, 4, 2),
        "level_nnz": (24, 16, 4),
        "nonbottom_level_count": 2,
        "bottom_size": 2,
    }
    assert stats["resources"] == {
        "inverse_diagonal_reserved_bytes": 48,
        "damping_reserved_bytes": 8,
        "bottom_inverse_reserved_bytes": 16,
        "total_reserved_bytes": 72,
    }
    assert stats["transfers"] == {
        "device_to_host_bytes": 8,
        "device_to_device_bytes": 72,
        "device_payload_readback_bytes": 0,
    }
    assert stats["contract"]["all_numeric_arrays_owned"]
    assert not stats["contract"]["host_matrix_payload_required"]

    for value in inverse_diagonals + dampings + [bottom_inverse]:
        value.fill(99.0)
    for actual, expected in zip(
        numeric._inverse_diagonals, inverse_diagonal_numpy
    ):
        np.testing.assert_array_equal(actual.to_numpy(), expected)
    for actual, expected in zip(numeric._dampings, damping_numpy):
        np.testing.assert_array_equal(actual.to_numpy(), expected)
    np.testing.assert_array_equal(
        numeric._bottom_inverse.to_numpy(), bottom_inverse_numpy
    )

    (
        invalid_inverse_diagonals,
        valid_dampings,
        valid_bottom_inverse,
        _,
        _,
        _,
    ) = _vcycle_numeric_sources()
    invalid_inverse_diagonals[0].fill(0.0)
    with pytest.raises(TaichiRuntimeError, match="inverse diagonal"):
        _SparseVcycleNumericSnapshot.copy_validated(
            hierarchy,
            inverse_diagonals=invalid_inverse_diagonals,
            dampings=valid_dampings,
            bottom_inverse=valid_bottom_inverse,
            topology_version=7,
            numeric_version=11,
        )

    (
        valid_inverse_diagonals,
        valid_dampings,
        _,
        _,
        _,
        valid_bottom_numpy,
    ) = _vcycle_numeric_sources()
    asymmetric_bottom = valid_bottom_numpy.copy()
    asymmetric_bottom[1] += np.float32(0.125)
    with pytest.raises(TaichiRuntimeError, match="exactly symmetric"):
        _SparseVcycleNumericSnapshot.copy_validated(
            hierarchy,
            inverse_diagonals=valid_inverse_diagonals,
            dampings=valid_dampings,
            bottom_inverse=_f32_array(asymmetric_bottom),
            topology_version=7,
            numeric_version=11,
        )


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
    vulkan_sparse_experimental=True,
)
def test_private_recursive_vcycle_graph_consumes_hierarchy_and_refreshes():
    row_offsets, column_indices, values = _source_arrays()
    source = _DeviceCsrSnapshot.copy_validated(
        rows=8,
        cols=8,
        row_offsets=row_offsets,
        column_indices=column_indices,
        values=values,
        topology_version=7,
        numeric_version=11,
    )
    level_specs = [
        (4, _aggregate_array()),
        (
            2,
            _aggregate_array(
                np.asarray([0, 0, 1, 1], dtype=np.int32)
            ),
        ),
    ]
    hierarchy = _CallerCoarsenedSparseHierarchyBuilder(
        explicit_array_capacity_bytes=2500,
        bottom_size_cap=2,
    ).build(
        source,
        level_specs,
        topology_version=7,
        numeric_version=11,
    )
    inverse_diagonals, dampings, bottom_inverse, _, _, _ = (
        _vcycle_numeric_sources()
    )
    numeric = _SparseVcycleNumericSnapshot.copy_validated(
        hierarchy,
        inverse_diagonals=inverse_diagonals,
        dampings=dampings,
        bottom_inverse=bottom_inverse,
        topology_version=7,
        numeric_version=11,
    )
    with pytest.raises(TaichiRuntimeError, match="capacity overflow"):
        _SparseRecursiveVcycleGraphPlan(
            hierarchy,
            numeric,
            explicit_array_capacity_bytes=1387,
        )
    plan = _SparseRecursiveVcycleGraphPlan(
        hierarchy,
        numeric,
        explicit_array_capacity_bytes=1400,
    )
    initial_stats = plan.debug_runtime_stats()
    assert initial_stats["identity"] == {
        "backend_family": source._backend,
        "method": "recursive_symmetric_vcycle",
        "size": 8,
        "level_count": 3,
        "topology_version": 7,
        "numeric_version": 11,
    }
    assert initial_stats["operations"] == {
        "apply_calls": 0,
        "rejected_apply_calls": 0,
        "graph_node_count": 1,
        "graph_dispatch_count": 7,
        "kernel_dispatches_per_apply": 7,
        "host_graph_submissions_per_apply": 1,
        "explicit_apply_host_synchronizations": 0,
        "native_operator_publishes": 0,
    }
    assert assert_sparse_graph_cache_attribution(
        initial_stats, expected_cache_object_count=1
    ) == {
        "borrowed_hierarchy_reserved_bytes": 548,
        "borrowed_numeric_setup_reserved_bytes": 72,
        "topology_argument_reserved_bytes": 344,
        "numeric_argument_reserved_bytes": 232,
        "plan_workspace_reserved_bytes": 96,
        "native_operator_reserved_bytes": 672,
        "build_peak_explicit_array_bytes": 1388,
        "explicit_array_capacity_bytes": 1400,
    }
    assert initial_stats["contract"][
        "deterministic_restriction_gather"
    ]
    assert not initial_stats["contract"][
        "floating_atomic_restriction_required"
    ]

    rhs_numpy = np.asarray(
        [1.0, -0.5, 2.0, -1.5, 0.75, -2.5, 1.25, -0.25],
        dtype=np.float32,
    )
    rhs = _f32_array(rhs_numpy)
    output = ti.ndarray(ti.f32, shape=8)
    output.fill(0.0)
    plan.apply(rhs, output)
    ti.sync()
    expected = _vcycle_reference(rhs_numpy)
    np.testing.assert_allclose(
        output.to_numpy(), expected, rtol=0.0, atol=2e-5
    )
    with pytest.raises(TaichiRuntimeError, match="alias"):
        plan.apply(rhs, rhs)

    publisher = plan.create_numeric_publisher()
    publisher_stats = publisher.debug_runtime_stats()
    assert publisher_stats["host_topology_metadata_bytes"] == 24
    assert publisher_stats["device_reserved_bytes"] == 0
    assert publisher_stats["numeric_role_count"] == 7
    assert publisher_stats["numeric_payload_bytes"] == 232
    native = plan.create_native_operator()
    with pytest.raises(TaichiRuntimeError, match="at most one"):
        plan.create_native_operator()
    published_stats = plan.debug_runtime_stats()
    assert published_stats["operations"]["apply_calls"] == 1
    assert published_stats["operations"]["rejected_apply_calls"] == 1
    assert published_stats["operations"]["native_operator_publishes"] == 1
    assert published_stats["transfers"] == {
        "device_to_host_bytes": 0,
        "device_to_device_bytes": 672,
        "device_kernel_workspace_initialization_bytes": 96,
        "device_payload_readback_bytes": 0,
    }

    output.fill(0.0)
    native.spmv(ti.lang.impl.get_runtime().prog, rhs.arr, output.arr)
    ti.sync()
    np.testing.assert_allclose(
        output.to_numpy(), expected, rtol=0.0, atol=2e-5
    )
    native_stats = native._debug_runtime_stats()
    assert native_stats["resources"]["pattern_reserved_bytes"] == 344
    assert native_stats["resources"]["values_reserved_bytes"] == 232
    assert native_stats["resources"][
        "spmv_workspace_reserved_bytes"
    ] == 96
    assert native_stats["resources"]["operator_owned_reserved_bytes"] == 672
    assert native_stats["transfers"]["device_to_device_bytes"] == 672

    scaled_rows, scaled_columns, scaled_values = _source_arrays(2.0 * _VALUES)
    scaled_source = _DeviceCsrSnapshot.copy_validated(
        rows=8,
        cols=8,
        row_offsets=scaled_rows,
        column_indices=scaled_columns,
        values=scaled_values,
        topology_version=7,
        numeric_version=12,
    )
    scaled_hierarchy = _CallerCoarsenedSparseHierarchyBuilder(
        explicit_array_capacity_bytes=2500,
        bottom_size_cap=2,
    ).build(
        scaled_source,
        [
            (4, _aggregate_array()),
            (
                2,
                _aggregate_array(
                    np.asarray([0, 0, 1, 1], dtype=np.int32)
                ),
            ),
        ],
        topology_version=7,
        numeric_version=12,
    )
    (
        scaled_inverse_diagonals,
        scaled_dampings,
        scaled_bottom_inverse,
        _,
        _,
        _,
    ) = _vcycle_numeric_sources(scale=2.0)
    scaled_numeric = _SparseVcycleNumericSnapshot.copy_validated(
        scaled_hierarchy,
        inverse_diagonals=scaled_inverse_diagonals,
        dampings=scaled_dampings,
        bottom_inverse=scaled_bottom_inverse,
        topology_version=7,
        numeric_version=12,
    )
    with pytest.raises(TaichiRuntimeError, match="topology version"):
        publisher.bind_sources(
            scaled_hierarchy,
            scaled_numeric,
            expected_topology_version=8,
            expected_numeric_version=11,
        )
    numeric_sources = publisher.bind_sources(
        scaled_hierarchy,
        scaled_numeric,
        expected_topology_version=7,
        expected_numeric_version=11,
    )
    assert len(numeric_sources) == 7
    native.update_numeric_data(
        ti.lang.impl.get_runtime().prog,
        numeric_sources,
        7,
        11,
    )
    numeric_sources = None
    output.fill(0.0)
    native.spmv(ti.lang.impl.get_runtime().prog, rhs.arr, output.arr)
    ti.sync()
    scaled_expected = _vcycle_reference(rhs_numpy, scale=2.0)
    np.testing.assert_allclose(
        output.to_numpy(), scaled_expected, rtol=0.0, atol=2e-5
    )
    refreshed_stats = native._debug_runtime_stats()
    assert refreshed_stats["identity"]["numeric_version"] == 12
    assert refreshed_stats["operations"]["numeric_updates"] == 1
    assert refreshed_stats["resources"][
        "numeric_update_peak_temporary_bytes"
    ] == 232
    assert refreshed_stats["resources"][
        "operator_owned_reserved_bytes"
    ] == 672
    assert refreshed_stats["transfers"]["device_to_device_bytes"] == 904
    final_publisher_stats = publisher.debug_runtime_stats()
    assert final_publisher_stats["operations"] == {
        "bind_calls": 1,
        "rejected_bind_calls": 1,
    }


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
    vulkan_sparse_experimental=True,
)
def test_private_vcycle_solve_publication_rebuilds_numeric_generation():
    def make_builder(scale, numeric_version):
        row_offsets, column_indices, values = _source_arrays(scale * _VALUES)
        source = _DeviceCsrSnapshot.copy_validated(
            rows=8,
            cols=8,
            row_offsets=row_offsets,
            column_indices=column_indices,
            values=values,
            topology_version=7,
            numeric_version=numeric_version,
        )
        hierarchy = _CallerCoarsenedSparseHierarchyBuilder(
            explicit_array_capacity_bytes=2500,
            bottom_size_cap=2,
        ).build(
            source,
            [
                (4, _aggregate_array()),
                (
                    2,
                    _aggregate_array(
                        np.asarray([0, 0, 1, 1], dtype=np.int32)
                    ),
                ),
            ],
            topology_version=7,
            numeric_version=numeric_version,
        )
        inverse_diagonals, dampings, bottom_inverse, _, _, _ = (
            _vcycle_numeric_sources(scale)
        )
        numeric = _SparseVcycleNumericSnapshot.copy_validated(
            hierarchy,
            inverse_diagonals=inverse_diagonals,
            dampings=dampings,
            bottom_inverse=bottom_inverse,
            topology_version=7,
            numeric_version=numeric_version,
        )
        return _SparseVcycleSolvePublicationBuilder(
            hierarchy,
            numeric,
            max_iterations=32,
            absolute_tolerance=1e-5,
            explicit_array_capacity_bytes=2000,
        )

    def solve_and_check(lease, scale):
        exact_numpy = np.asarray(
            [1.0, -0.5, 2.0, -1.5, 0.75, -2.5, 1.25, -0.25],
            dtype=np.float32,
        )
        rhs_numpy = (
            _dense_csr(
                _ROW_OFFSETS,
                _COLUMN_INDICES,
                scale * _VALUES,
            )
            @ exact_numpy
        ).astype(np.float32)
        solution = ti.ndarray(ti.f32, shape=8)
        solution.fill(0.0)
        rhs = _f32_array(rhs_numpy)
        lease.solve(solution, rhs)
        ti.sync()
        assert lease._publication.solver.is_success()
        solver_stats = lease._publication.solver._debug_runtime_stats()
        expected_method = (
            "pcg_compiled_graph_bounded_masked_probe"
            if ti.lang.impl.current_cfg().arch == ti.vulkan
            else "pcg_compiled_graph"
        )
        assert solver_stats["identity"]["method"] == expected_method
        assert solver_stats["identity"]["preconditioner_method"] == (
            "compiled_graph_inverse_apply"
        )
        np.testing.assert_allclose(
            solution.to_numpy(), exact_numpy, rtol=0.0, atol=2e-4
        )

    initial_builder = make_builder(1.0, 11)
    backend = ti.lang.impl.current_cfg().arch
    expected_solver_bytes = 168 if backend == ti.vulkan else 128
    expected_steady_bytes = 1068 if backend == ti.vulkan else 1028
    expected_build_peak = 1784 if backend == ti.vulkan else 1744
    assert initial_builder.estimated_steady_device_bytes == (
        expected_steady_bytes
    )
    assert initial_builder.estimated_build_peak_device_bytes == (
        expected_build_peak
    )
    with pytest.raises(TaichiRuntimeError, match="capacity overflow"):
        _SparseVcycleSolvePublicationBuilder(
            initial_builder._hierarchy,
            initial_builder._numeric,
            max_iterations=32,
            absolute_tolerance=1e-5,
            explicit_array_capacity_bytes=expected_build_peak - 1,
        )

    registry = _SparseSolvePublicationRegistry(
        ti.lang.impl.get_runtime().prog,
        capacity_bytes=2900,
    )
    initial_result = registry.publish(
        expected_generation=0,
        topology_version=7,
        numeric_version=11,
        estimated_steady_device_bytes=(
            initial_builder.estimated_steady_device_bytes
        ),
        estimated_build_peak_device_bytes=(
            initial_builder.estimated_build_peak_device_bytes
        ),
        builder=initial_builder.build,
    )
    assert initial_result["published"], initial_result
    assert initial_result["generation"] == 1
    assert initial_result["numeric_version"] == 11
    initial_stats = initial_builder.debug_runtime_stats()
    assert initial_stats["operations"] == {
        "build_attempts": 1,
        "successful_builds": 1,
        "failed_builds": 0,
    }
    assert initial_stats["resources"][
        "target_operator_reservation_bytes"
    ] == 228
    assert initial_stats["resources"][
        "inverse_operator_reservation_bytes"
    ] == 672
    assert initial_stats["resources"][
        "solver_workspace_reservation_bytes"
    ] == expected_solver_bytes
    expected_materialized = 0 if backend == ti.cuda else expected_solver_bytes
    assert initial_stats["resources"]["last_report"][
        "solver_workspace_materialized_bytes"
    ] == expected_materialized

    old_lease = registry.acquire()
    solve_and_check(old_lease, 1.0)
    replacement_builder = make_builder(2.0, 12)
    replacement_result = registry.publish(
        expected_generation=1,
        topology_version=7,
        numeric_version=12,
        estimated_steady_device_bytes=(
            replacement_builder.estimated_steady_device_bytes
        ),
        estimated_build_peak_device_bytes=(
            replacement_builder.estimated_build_peak_device_bytes
        ),
        builder=replacement_builder.build,
    )
    assert replacement_result["published"], replacement_result
    assert replacement_result["generation"] == 2
    assert replacement_result["numeric_version"] == 12
    assert replacement_result[
        "old_plus_new_steady_device_bytes"
    ] == 2 * expected_steady_bytes
    new_lease = registry.acquire()
    assert old_lease.numeric_version == 11
    assert new_lease.numeric_version == 12
    solve_and_check(old_lease, 1.0)
    solve_and_check(new_lease, 2.0)

    registry_stats = registry.debug_runtime_stats()
    assert registry_stats["identity"] == {
        "generation": 2,
        "topology_version": 7,
        "numeric_version": 12,
    }
    assert registry_stats["resources"][
        "live_generation_steady_device_bytes"
    ] == 2 * expected_steady_bytes
    assert registry_stats["resources"][
        "retired_lease_steady_device_bytes"
    ] == expected_steady_bytes
    assert registry_stats["contract"][
        "same_topology_numeric_generations_supported"
    ]
    old_lease.release()
    new_lease.release()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_private_caller_coarsened_hierarchy_rejects_capacity_atomically():
    row_offsets, column_indices, values = _source_arrays()
    source = _DeviceCsrSnapshot.copy_validated(
        rows=8,
        cols=8,
        row_offsets=row_offsets,
        column_indices=column_indices,
        values=values,
        topology_version=7,
        numeric_version=11,
    )
    first_map = _aggregate_array()
    second_map_values = np.asarray([0, 0, 1, 1], dtype=np.int32)
    second_map = _aggregate_array(second_map_values)
    level_specs = [(4, first_map), (2, second_map)]

    too_small = _CallerCoarsenedSparseHierarchyBuilder(
        explicit_array_capacity_bytes=2471,
        bottom_size_cap=2,
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

    builder = _CallerCoarsenedSparseHierarchyBuilder(
        explicit_array_capacity_bytes=2500,
        bottom_size_cap=2,
    )
    published = builder.build(
        source,
        level_specs,
        topology_version=7,
        numeric_version=11,
    )
    published_bottom = _snapshot_payload(published._levels[-1])
    invalid_second_map = _aggregate_array(
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
    ] == 548
    assert failed_stats["transfers"] == {
        "device_to_host_bytes": 32,
        "device_to_device_bytes": 532,
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
            [(8, _aggregate_array())],
            topology_version=7,
            numeric_version=11,
        )
    with pytest.raises(TaichiRuntimeError, match="bottom_size_cap"):
        builder.build(
            source,
            [(4, _aggregate_array())],
            topology_version=7,
            numeric_version=11,
        )


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_private_caller_coarsened_hierarchy_rejects_after_reset():
    row_offsets, column_indices, values = _source_arrays()
    source = _DeviceCsrSnapshot.copy_validated(
        rows=8,
        cols=8,
        row_offsets=row_offsets,
        column_indices=column_indices,
        values=values,
        topology_version=7,
        numeric_version=11,
    )
    target_plan = _SparseCsrGraphOperatorPlan(
        source,
        explicit_array_capacity_bytes=500,
    )
    target_publisher = target_plan.create_numeric_publisher()
    builder = _CallerCoarsenedSparseHierarchyBuilder(
        explicit_array_capacity_bytes=2500,
        bottom_size_cap=2,
    )
    hierarchy = builder.build(
        source,
        [
            (4, _aggregate_array()),
            (
                2,
                _aggregate_array(
                    np.asarray([0, 0, 1, 1], dtype=np.int32)
                ),
            ),
        ],
        topology_version=7,
        numeric_version=11,
    )
    inverse_diagonals, dampings, bottom_inverse, _, _, _ = (
        _vcycle_numeric_sources()
    )
    numeric = _SparseVcycleNumericSnapshot.copy_validated(
        hierarchy,
        inverse_diagonals=inverse_diagonals,
        dampings=dampings,
        bottom_inverse=bottom_inverse,
        topology_version=7,
        numeric_version=11,
    )
    graph_plan = _SparseRecursiveVcycleGraphPlan(
        hierarchy,
        numeric,
        explicit_array_capacity_bytes=1400,
    )
    numeric_publisher = graph_plan.create_numeric_publisher()
    ti.reset()
    ti.init(arch=ti.cpu, enable_fallback=False)

    with pytest.raises(TaichiRuntimeError, match="runtime has been reset"):
        hierarchy.debug_runtime_stats()
    with pytest.raises(TaichiRuntimeError, match="runtime has been reset"):
        builder.debug_runtime_stats()
    with pytest.raises(TaichiRuntimeError, match="runtime has been reset"):
        numeric.debug_runtime_stats()
    with pytest.raises(TaichiRuntimeError, match="runtime has been reset"):
        graph_plan.debug_runtime_stats()
    with pytest.raises(TaichiRuntimeError, match="runtime has been reset"):
        numeric_publisher.debug_runtime_stats()
    with pytest.raises(TaichiRuntimeError, match="runtime has been reset"):
        target_plan.debug_runtime_stats()
    with pytest.raises(TaichiRuntimeError, match="runtime has been reset"):
        target_publisher.debug_runtime_stats()
