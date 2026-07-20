import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.linalg._sparse_block_vcycle_numeric import (
    _SparseBlockVcycleNumericSnapshot,
)
from taichi_forge.linalg._sparse_block_vcycle_graph import (
    _SparseRecursiveBlockVcycleGraphPlan,
)
from taichi_forge.linalg._sparse_block_vcycle_solve import (
    _SparseBlockVcycleSolvePublicationBuilder,
)
from taichi_forge.linalg._sparse_bsr_graph_operator import (
    _DeviceBsrSnapshot,
)
from taichi_forge.linalg._sparse_bsr_hierarchy_candidate import (
    _CallerCoarsenedBsrHierarchyBuilder,
)
from tests import test_utils
from tests.sparse_runtime_stats import (
    assert_sparse_graph_cache_attribution,
)


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


def _path_source():
    block_rows = 8
    coordinates = np.arange(1, _BLOCK_SIZE + 1, dtype=np.float32)
    diagonal = 4.0 * np.eye(_BLOCK_SIZE, dtype=np.float32)
    diagonal += 0.01 * np.outer(coordinates, coordinates)
    coupling = -0.5 * np.eye(_BLOCK_SIZE, dtype=np.float32)
    row_offsets = [0]
    columns = []
    values = []
    for block_row in range(block_rows):
        if block_row > 0:
            columns.append(block_row - 1)
            values.extend(coupling.reshape(-1))
        columns.append(block_row)
        values.extend(diagonal.reshape(-1))
        if block_row + 1 < block_rows:
            columns.append(block_row + 1)
            values.extend(coupling.reshape(-1))
        row_offsets.append(len(columns))
    source = _DeviceBsrSnapshot.copy_validated(
        block_rows=block_rows,
        block_cols=block_rows,
        block_size=_BLOCK_SIZE,
        row_offsets=_i32_array(row_offsets),
        column_indices=_i32_array(columns),
        values=_f32_array(values),
        topology_version=13,
        numeric_version=19,
    )
    assert source.block_nnz == 22
    return source


def _hierarchy():
    source = _path_source()
    hierarchy = _CallerCoarsenedBsrHierarchyBuilder(
        explicit_array_capacity_bytes=6000,
        bottom_scalar_size_cap=6,
    ).build(
        source,
        [
            (4, _i32_array(_FIRST_MAP)),
            (2, _i32_array(_SECOND_MAP)),
        ],
        topology_version=13,
        numeric_version=19,
    )
    assert hierarchy.level_block_nnz == (22, 10, 4)
    return hierarchy


def _snapshot_dense(snapshot):
    row_offsets = snapshot._row_offsets.to_numpy()
    columns = snapshot._column_indices.to_numpy()
    values = snapshot._values.to_numpy()
    dense = np.zeros((snapshot.rows, snapshot.cols), dtype=np.float64)
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


def _numeric_sources(hierarchy):
    inverse_arrays = []
    inverse_numpy = []
    for level in hierarchy._levels[:-1]:
        row_offsets = level._row_offsets.to_numpy()
        columns = level._column_indices.to_numpy()
        values = level._values.to_numpy().reshape(
            level.block_nnz, _BLOCK_SIZE, _BLOCK_SIZE
        )
        level_inverse = []
        for block_row in range(level.block_rows):
            begin = int(row_offsets[block_row])
            end = int(row_offsets[block_row + 1])
            diagonal_offsets = np.flatnonzero(
                columns[begin:end] == block_row
            )
            assert len(diagonal_offsets) == 1
            block = values[begin + int(diagonal_offsets[0])].astype(
                np.float64
            )
            inverse = np.linalg.inv(block)
            inverse = 0.5 * (inverse + inverse.T)
            level_inverse.extend(inverse.reshape(-1))
        level_inverse = np.asarray(level_inverse, dtype=np.float32)
        inverse_numpy.append(level_inverse)
        inverse_arrays.append(_f32_array(level_inverse))

    damping_numpy = np.asarray([0.72, 0.68], dtype=np.float32)
    dampings = _f32_array(damping_numpy)
    bottom_dense = _snapshot_dense(hierarchy._levels[-1])
    bottom_inverse_numpy = np.linalg.inv(bottom_dense)
    bottom_inverse_numpy = 0.5 * (
        bottom_inverse_numpy + bottom_inverse_numpy.T
    )
    bottom_inverse_numpy = np.asarray(
        bottom_inverse_numpy, dtype=np.float32
    ).reshape(-1)
    bottom_inverse = _f32_array(bottom_inverse_numpy)
    return (
        inverse_arrays,
        dampings,
        bottom_inverse,
        inverse_numpy,
        damping_numpy,
        bottom_inverse_numpy,
    )


def _block_inverse_apply(inverse_values, vector, block_rows):
    blocks = inverse_values.reshape(
        block_rows, _BLOCK_SIZE, _BLOCK_SIZE
    ).astype(np.float64)
    vector_blocks = np.asarray(vector, dtype=np.float64).reshape(
        block_rows, _BLOCK_SIZE
    )
    return np.einsum("bij,bj->bi", blocks, vector_blocks).reshape(-1)


def _block_vcycle_oracle(
    hierarchy,
    inverse_numpy,
    damping_numpy,
    bottom_inverse_numpy,
    rhs,
):
    matrices = [
        _snapshot_dense(level) for level in hierarchy._levels
    ]
    aggregate_maps = [
        mapping.to_numpy() for mapping in hierarchy._aggregate_maps
    ]

    def apply(level_index, level_rhs):
        if level_index + 1 == hierarchy.level_count:
            bottom_size = hierarchy.level_scalar_rows[-1]
            bottom = bottom_inverse_numpy.reshape(
                bottom_size, bottom_size
            ).astype(np.float64)
            return bottom @ level_rhs
        block_rows = hierarchy.level_block_rows[level_index]
        damping = float(damping_numpy[level_index])
        pre = damping * _block_inverse_apply(
            inverse_numpy[level_index], level_rhs, block_rows
        )
        aggregate = aggregate_maps[level_index]
        coarse_block_rows = hierarchy.level_block_rows[level_index + 1]
        prolongation = np.zeros(
            (
                block_rows * _BLOCK_SIZE,
                coarse_block_rows * _BLOCK_SIZE,
            ),
            dtype=np.float64,
        )
        identity = np.eye(_BLOCK_SIZE, dtype=np.float64)
        for fine_block_row, coarse_block_row in enumerate(aggregate):
            fine_begin = fine_block_row * _BLOCK_SIZE
            coarse_begin = int(coarse_block_row) * _BLOCK_SIZE
            prolongation[
                fine_begin : fine_begin + _BLOCK_SIZE,
                coarse_begin : coarse_begin + _BLOCK_SIZE,
            ] = identity
        residual = level_rhs - matrices[level_index] @ pre
        coarse_rhs = prolongation.T @ residual
        coarse_solution = apply(level_index + 1, coarse_rhs)
        corrected = pre + prolongation @ coarse_solution
        post_residual = level_rhs - matrices[level_index] @ corrected
        return corrected + damping * _block_inverse_apply(
            inverse_numpy[level_index], post_residual, block_rows
        )

    return apply(0, np.asarray(rhs, dtype=np.float64))


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
    vulkan_sparse_experimental=True,
)
def test_private_block_vcycle_numeric_owns_one_packed_generation():
    hierarchy = _hierarchy()
    (
        inverse_arrays,
        dampings,
        bottom_inverse,
        inverse_numpy,
        damping_numpy,
        bottom_inverse_numpy,
    ) = _numeric_sources(hierarchy)
    expected_inverse = np.concatenate(
        (*inverse_numpy, bottom_inverse_numpy)
    )
    numeric = _SparseBlockVcycleNumericSnapshot.copy_validated(
        hierarchy,
        block_inverses=inverse_arrays,
        dampings=dampings,
        bottom_inverse=bottom_inverse,
        topology_version=13,
        numeric_version=19,
    )
    stats = numeric.debug_runtime_stats()
    assert stats["identity"] == {
        "backend_family": hierarchy._backend,
        "method": "packed_block_jacobi_dense_bottom_inverse",
        "block_size": 3,
        "level_block_rows": (8, 4, 2),
        "level_block_nnz": (22, 10, 4),
        "nonbottom_level_count": 2,
        "block_inverse_offsets": (0, 72),
        "bottom_inverse_offset": 108,
        "bottom_scalar_size": 6,
        "topology_version": 13,
        "numeric_version": 19,
    }
    assert stats["resources"] == {
        "block_inverse_reserved_bytes": 432,
        "bottom_inverse_reserved_bytes": 144,
        "packed_inverse_reserved_bytes": 576,
        "damping_reserved_bytes": 8,
        "total_reserved_bytes": 584,
    }
    assert stats["transfers"] == {
        "device_to_host_bytes": 8,
        "device_to_device_bytes": 584,
        "device_payload_readback_bytes": 0,
    }

    # Caller buffers can be reused after the owned numeric generation exists.
    for inverse in inverse_arrays:
        inverse.fill(0.0)
    dampings.fill(0.0)
    bottom_inverse.fill(0.0)
    np.testing.assert_allclose(
        numeric._inverse_values.to_numpy(),
        expected_inverse,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        numeric._dampings.to_numpy(), damping_numpy
    )
    assert stats["contract"]["one_packed_inverse_numeric_role"]
    assert stats["contract"]["one_packed_damping_numeric_role"]
    assert stats["contract"]["single_validation_control_readback"]
    assert stats["contract"][
        "full_spd_qualification_is_caller_responsibility"
    ]
    assert not stats["contract"]["host_inversion_performed"]
    assert not stats["contract"]["graph_or_solver_constructed"]


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
    vulkan_sparse_experimental=True,
)
def test_private_recursive_block_vcycle_graph_matches_host_oracle():
    hierarchy = _hierarchy()
    (
        inverse_arrays,
        dampings,
        bottom_inverse,
        inverse_numpy,
        damping_numpy,
        bottom_inverse_numpy,
    ) = _numeric_sources(hierarchy)
    numeric = _SparseBlockVcycleNumericSnapshot.copy_validated(
        hierarchy,
        block_inverses=inverse_arrays,
        dampings=dampings,
        bottom_inverse=bottom_inverse,
        topology_version=13,
        numeric_version=19,
    )
    with pytest.raises(TaichiRuntimeError, match="capacity overflow"):
        _SparseRecursiveBlockVcycleGraphPlan(
            hierarchy,
            numeric,
            explicit_array_capacity_bytes=5131,
        )
    plan = _SparseRecursiveBlockVcycleGraphPlan(
        hierarchy,
        numeric,
        explicit_array_capacity_bytes=5132,
    )
    initial_stats = plan.debug_runtime_stats()
    assert initial_stats["identity"] == {
        "backend_family": hierarchy._backend,
        "method": "recursive_symmetric_block_vcycle",
        "size": 24,
        "block_size": 3,
        "level_count": 3,
        "level_block_rows": (8, 4, 2),
        "topology_version": 13,
        "numeric_version": 19,
    }
    assert assert_sparse_graph_cache_attribution(
        initial_stats, expected_cache_object_count=1
    ) == {
        "borrowed_hierarchy_reserved_bytes": 1636,
        "borrowed_numeric_setup_reserved_bytes": 584,
        "topology_argument_reserved_bytes": 312,
        "numeric_argument_reserved_bytes": 1736,
        "plan_workspace_reserved_bytes": 432,
        "native_operator_reserved_bytes": 2480,
        "build_peak_explicit_array_bytes": 5132,
        "explicit_array_capacity_bytes": 5132,
    }

    rhs_numpy = np.linspace(-1.0, 1.3, 24, dtype=np.float32)
    expected = _block_vcycle_oracle(
        hierarchy,
        inverse_numpy,
        damping_numpy,
        bottom_inverse_numpy,
        rhs_numpy,
    )
    rhs = _f32_array(rhs_numpy)
    output = ti.ndarray(ti.f32, shape=24)
    output.fill(0.0)
    plan.apply(rhs, output)
    ti.sync()
    np.testing.assert_allclose(
        output.to_numpy(), expected, rtol=0.0, atol=2e-4
    )
    with pytest.raises(TaichiRuntimeError, match="alias"):
        plan.apply(rhs, rhs)

    native = plan.create_native_operator()
    with pytest.raises(TaichiRuntimeError, match="at most one"):
        plan.create_native_operator()
    native_output = ti.ndarray(ti.f32, shape=24)
    native_output.fill(0.0)
    program = ti.lang.impl.get_runtime().prog
    native.spmv(program, rhs.arr, native_output.arr)
    ti.sync()
    np.testing.assert_allclose(
        native_output.to_numpy(), expected, rtol=0.0, atol=2e-4
    )
    native_stats = native._debug_runtime_stats()
    assert native_stats["resources"]["pattern_reserved_bytes"] == 312
    assert native_stats["resources"]["values_reserved_bytes"] == 1736
    assert native_stats["resources"][
        "spmv_workspace_reserved_bytes"
    ] == 432
    assert native_stats["resources"][
        "operator_owned_reserved_bytes"
    ] == 2480

    stats = plan.debug_runtime_stats()
    assert stats["operations"] == {
        "apply_calls": 1,
        "rejected_apply_calls": 1,
        "graph_node_count": 1,
        "graph_dispatch_count": 11,
        "kernel_dispatches_per_apply": 11,
        "pre_restrict_dispatches_per_level": 2,
        "post_dispatches_per_level": 3,
        "host_graph_submissions_per_apply": 1,
        "explicit_apply_host_synchronizations": 0,
        "native_operator_publishes": 1,
    }
    assert stats["transfers"] == {
        "device_to_host_bytes": 0,
        "device_to_device_bytes": 2480,
        "device_kernel_workspace_initialization_bytes": 432,
        "device_payload_readback_bytes": 0,
    }
    assert stats["contract"][
        "deterministic_block_restriction_gather"
    ]
    assert stats["contract"][
        "post_spmv_workspace_avoids_repeated_block_residual"
    ]
    assert not stats["contract"]["floating_atomic_restriction_required"]
    assert not stats["contract"]["workspace_total_bytes_reported"]


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
    vulkan_sparse_experimental=True,
)
def test_private_block_vcycle_pcg_publishes_immutable_generation():
    hierarchy = _hierarchy()
    inverse_arrays, dampings, bottom_inverse, _, _, _ = _numeric_sources(
        hierarchy
    )
    numeric = _SparseBlockVcycleNumericSnapshot.copy_validated(
        hierarchy,
        block_inverses=inverse_arrays,
        dampings=dampings,
        bottom_inverse=bottom_inverse,
        topology_version=13,
        numeric_version=19,
    )
    backend = ti.lang.impl.current_cfg().arch
    expected_solver_bytes = 424 if backend == ti.vulkan else 384
    expected_steady_bytes = 3820 if backend == ti.vulkan else 3780
    expected_build_peak = 6472 if backend == ti.vulkan else 6432
    with pytest.raises(TaichiRuntimeError, match="capacity overflow"):
        _SparseBlockVcycleSolvePublicationBuilder(
            hierarchy,
            numeric,
            max_iterations=32,
            absolute_tolerance=1e-5,
            explicit_array_capacity_bytes=expected_build_peak - 1,
        )
    builder = _SparseBlockVcycleSolvePublicationBuilder(
        hierarchy,
        numeric,
        max_iterations=32,
        absolute_tolerance=1e-5,
        explicit_array_capacity_bytes=expected_build_peak,
    )
    assert builder.estimated_steady_device_bytes == expected_steady_bytes
    assert builder.estimated_build_peak_device_bytes == expected_build_peak
    publication = builder.build()

    exact = np.linspace(-0.8, 1.15, 24, dtype=np.float32)
    fine_dense = _snapshot_dense(hierarchy._levels[0])
    rhs = _f32_array((fine_dense @ exact).astype(np.float32))
    solution = ti.ndarray(ti.f32, shape=24)
    solution.fill(0.0)
    publication.solve(solution, rhs)
    ti.sync()
    assert publication.solver.is_success()
    np.testing.assert_allclose(
        solution.to_numpy(), exact, rtol=0.0, atol=3e-4
    )
    solver_stats = publication.solver._debug_runtime_stats()
    expected_method = (
        "pcg_compiled_graph_bounded_masked_probe"
        if backend == ti.vulkan
        else "pcg_compiled_graph"
    )
    assert solver_stats["identity"]["method"] == expected_method
    assert solver_stats["identity"]["preconditioner_method"] == (
        "compiled_graph_inverse_apply"
    )
    target_stats = publication.target._debug_runtime_stats()
    inverse_stats = publication.inverse._debug_runtime_stats()
    assert target_stats["resources"]["pattern_reserved_bytes"] == 124
    assert target_stats["resources"]["values_reserved_bytes"] == 792
    assert target_stats["resources"][
        "operator_owned_reserved_bytes"
    ] == 916
    assert inverse_stats["resources"]["pattern_reserved_bytes"] == 312
    assert inverse_stats["resources"]["values_reserved_bytes"] == 1736
    assert inverse_stats["resources"][
        "spmv_workspace_reserved_bytes"
    ] == 432
    assert inverse_stats["resources"][
        "operator_owned_reserved_bytes"
    ] == 2480

    stats = builder.debug_runtime_stats()
    assert stats["operations"] == {
        "build_attempts": 1,
        "successful_builds": 1,
        "failed_builds": 0,
    }
    resources = stats["resources"]
    assert resources["target_operator_reservation_bytes"] == 916
    assert resources["inverse_operator_reservation_bytes"] == 2480
    assert resources["inverse_workspace_reservation_bytes"] == 432
    assert resources["solver_workspace_reservation_bytes"] == (
        expected_solver_bytes
    )
    assert resources["estimated_steady_device_bytes"] == (
        expected_steady_bytes
    )
    assert resources["estimated_build_peak_device_bytes"] == (
        expected_build_peak
    )
    expected_materialized = 0 if backend == ti.cuda else expected_solver_bytes
    assert resources["last_report"][
        "solver_workspace_materialized_bytes"
    ] == expected_materialized
    assert resources["live_publication_count"] == 1
    assert resources["live_publication_steady_device_bytes"] == (
        expected_steady_bytes
    )
    assert stats["contract"][
        "immutable_target_inverse_solver_generation"
    ]
    assert stats["contract"]["compiled_graph_target_and_inverse"]
    assert not stats["contract"]["host_inversion_performed"]


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_private_block_vcycle_numeric_rejects_unqualified_sources():
    hierarchy = _hierarchy()
    (
        inverse_arrays,
        dampings,
        bottom_inverse,
        inverse_numpy,
        _,
        bottom_inverse_numpy,
    ) = _numeric_sources(hierarchy)

    asymmetric = inverse_numpy[0].copy()
    asymmetric[1] += np.float32(0.25)
    with pytest.raises(TaichiRuntimeError, match="not symmetric"):
        _SparseBlockVcycleNumericSnapshot.copy_validated(
            hierarchy,
            block_inverses=[_f32_array(asymmetric), inverse_arrays[1]],
            dampings=dampings,
            bottom_inverse=bottom_inverse,
            topology_version=13,
            numeric_version=19,
        )

    nonpositive = inverse_numpy[0].copy()
    nonpositive[0] = np.float32(-1.0)
    with pytest.raises(TaichiRuntimeError, match="diagonal is not positive"):
        _SparseBlockVcycleNumericSnapshot.copy_validated(
            hierarchy,
            block_inverses=[_f32_array(nonpositive), inverse_arrays[1]],
            dampings=dampings,
            bottom_inverse=bottom_inverse,
            topology_version=13,
            numeric_version=19,
        )

    with pytest.raises(TaichiRuntimeError, match="damping"):
        _SparseBlockVcycleNumericSnapshot.copy_validated(
            hierarchy,
            block_inverses=inverse_arrays,
            dampings=_f32_array([0.0, 0.68]),
            bottom_inverse=bottom_inverse,
            topology_version=13,
            numeric_version=19,
        )

    asymmetric_bottom = bottom_inverse_numpy.copy()
    asymmetric_bottom[1] += np.float32(0.125)
    with pytest.raises(TaichiRuntimeError, match="bottom inverse"):
        _SparseBlockVcycleNumericSnapshot.copy_validated(
            hierarchy,
            block_inverses=inverse_arrays,
            dampings=dampings,
            bottom_inverse=_f32_array(asymmetric_bottom),
            topology_version=13,
            numeric_version=19,
        )

    with pytest.raises(TaichiRuntimeError, match="one block inverse"):
        _SparseBlockVcycleNumericSnapshot.copy_validated(
            hierarchy,
            block_inverses=inverse_arrays[:1],
            dampings=dampings,
            bottom_inverse=bottom_inverse,
            topology_version=13,
            numeric_version=19,
        )
    with pytest.raises(TaichiRuntimeError, match="versions must match"):
        _SparseBlockVcycleNumericSnapshot.copy_validated(
            hierarchy,
            block_inverses=inverse_arrays,
            dampings=dampings,
            bottom_inverse=bottom_inverse,
            topology_version=13,
            numeric_version=20,
        )


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_private_block_vcycle_numeric_rejects_after_reset():
    hierarchy = _hierarchy()
    inverse_arrays, dampings, bottom_inverse, _, _, _ = _numeric_sources(
        hierarchy
    )
    numeric = _SparseBlockVcycleNumericSnapshot.copy_validated(
        hierarchy,
        block_inverses=inverse_arrays,
        dampings=dampings,
        bottom_inverse=bottom_inverse,
        topology_version=13,
        numeric_version=19,
    )
    graph_plan = _SparseRecursiveBlockVcycleGraphPlan(
        hierarchy,
        numeric,
        explicit_array_capacity_bytes=5132,
    )
    ti.reset()
    ti.init(arch=ti.cpu, enable_fallback=False)

    with pytest.raises(TaichiRuntimeError, match="runtime has been reset"):
        numeric.debug_runtime_stats()
    with pytest.raises(TaichiRuntimeError, match="runtime has been reset"):
        graph_plan.debug_runtime_stats()
