import gc

import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


def _make_tridiagonal_pattern():
    row_offsets = ti.ndarray(dtype=ti.i32, shape=4)
    column_indices = ti.ndarray(dtype=ti.i32, shape=7)
    row_offsets.from_numpy(np.asarray([0, 2, 5, 7], dtype=np.int32))
    column_indices.from_numpy(np.asarray([0, 1, 0, 1, 2, 1, 2], dtype=np.int32))
    pattern = ti.lang.impl.get_runtime().prog._create_csr_pattern(3, 3, row_offsets.arr, column_indices.arr)
    return pattern, row_offsets, column_indices


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
)
def test_shared_csr_pattern_spmv_update_and_ownership():
    pattern, source_rows, source_columns = _make_tridiagonal_pattern()
    values_host = np.asarray([2, -1, -1, 2, -1, -1, 2], dtype=np.float32)
    values_a = ti.ndarray(dtype=ti.f32, shape=7)
    values_b = ti.ndarray(dtype=ti.f32, shape=7)
    values_a.from_numpy(values_host)
    values_b.from_numpy(2 * values_host)

    # Pattern creation takes an immutable snapshot. Later source mutation must
    # not alter operators created from the pattern.
    source_rows.from_numpy(np.zeros(4, dtype=np.int32))
    source_columns.from_numpy(np.zeros(7, dtype=np.int32))

    prog = ti.lang.impl.get_runtime().prog
    matrix_a = ti.linalg.SparseMatrix(sm=prog._create_csr_matrix_from_pattern(pattern, values_a.arr))
    matrix_b = ti.linalg.SparseMatrix(sm=prog._create_csr_matrix_from_pattern(pattern, values_b.arr))
    vector_host = np.asarray([1, 2, 3], dtype=np.float32)
    vector = ti.ndarray(dtype=ti.f32, shape=3)
    vector.from_numpy(vector_host)
    np.testing.assert_allclose((matrix_a @ vector).to_numpy(), [0, 0, 4])
    np.testing.assert_allclose((matrix_b @ vector).to_numpy(), [0, 0, 8])

    matrix_a.update_values(values_b)
    np.testing.assert_allclose((matrix_a @ vector).to_numpy(), [0, 0, 8])
    pattern_stats = pattern._debug_runtime_stats()
    stats_a = matrix_a._debug_runtime_stats()
    stats_b = matrix_b._debug_runtime_stats()
    assert pattern_stats["identity"]["storage_format"] == "csr"
    assert pattern_stats["identity"]["value_order"] == "row_major_compressed"
    assert pattern_stats["identity"]["nnz"] == 7
    assert pattern_stats["lifecycle"]["immutable"]
    assert pattern_stats["lifecycle"]["operator_references"] == 2
    assert pattern_stats["resources"]["pattern_reserved_bytes"] == 44
    assert stats_a["identity"]["pattern_id"] == pattern_stats["identity"]["pattern_id"]
    assert stats_b["identity"]["pattern_id"] == pattern_stats["identity"]["pattern_id"]
    assert stats_a["identity"]["numeric_version"] == 2
    assert stats_b["identity"]["numeric_version"] == 1
    assert stats_a["resources"]["pattern_storage_shared"]
    assert not stats_a["resources"]["sum_operator_owned_bytes_across_operators_safe"]
    assert stats_a["resources"]["operator_exclusive_reserved_bytes"] >= 28
    contract = matrix_a._get_format_contract()
    assert contract["pattern"]["ownership"] == "shared_immutable"
    assert contract["pattern"]["mutability"] == "fixed"
    assert not contract["pattern"]["empty_supported"]
    if ti.lang.impl.current_cfg().arch == ti.cpu:
        assert not contract["operations"]["public_direct_solver"]
        assert contract["operations"]["public_cg"]
        assert contract["operations"]["public_jacobi_selection"]

    del matrix_a
    gc.collect()
    assert pattern._debug_runtime_stats()["lifecycle"]["operator_references"] == 1
    del matrix_b
    gc.collect()
    assert pattern._debug_runtime_stats()["lifecycle"]["operator_references"] == 0


@test_utils.test(arch=[ti.cpu], offline_cache=False)
def test_shared_cpu_csr_pattern_supports_f64_values():
    pattern, _, _ = _make_tridiagonal_pattern()
    values = ti.ndarray(dtype=ti.f64, shape=7)
    values.from_numpy(np.asarray([2, -1, -1, 2, -1, -1, 2], dtype=np.float64))
    prog = ti.lang.impl.get_runtime().prog
    matrix = ti.linalg.SparseMatrix(sm=prog._create_csr_matrix_from_pattern(pattern, values.arr))
    vector = ti.ndarray(dtype=ti.f64, shape=3)
    vector.from_numpy(np.asarray([1, 2, 3], dtype=np.float64))
    np.testing.assert_allclose((matrix @ vector).to_numpy(), [0, 0, 4])
    assert matrix.dtype == ti.f64


@test_utils.test(arch=[ti.cpu], offline_cache=False)
def test_shared_csr_pattern_rejects_noncanonical_columns():
    row_offsets = ti.ndarray(dtype=ti.i32, shape=3)
    duplicate_columns = ti.ndarray(dtype=ti.i32, shape=2)
    row_offsets.from_numpy(np.asarray([0, 2, 2], dtype=np.int32))
    duplicate_columns.from_numpy(np.asarray([0, 0], dtype=np.int32))
    with pytest.raises(RuntimeError, match="strictly increasing and unique"):
        ti.lang.impl.get_runtime().prog._create_csr_pattern(2, 2, row_offsets.arr, duplicate_columns.arr)


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
)
def test_shared_csr_pattern_rejects_runtime_rebind():
    pattern, _, _ = _make_tridiagonal_pattern()
    values = ti.ndarray(dtype=ti.f32, shape=7)
    values.from_numpy(np.asarray([2, -1, -1, 2, -1, -1, 2], dtype=np.float32))
    prog = ti.lang.impl.get_runtime().prog
    matrix = ti.linalg.SparseMatrix(sm=prog._create_csr_matrix_from_pattern(pattern, values.arr))
    arch = ti.lang.impl.current_cfg().arch

    ti.reset()
    ti.init(arch=arch)
    replacement_values = ti.ndarray(dtype=ti.f32, shape=7)
    with pytest.raises(RuntimeError, match="pattern owned by the same Program"):
        ti.lang.impl.get_runtime().prog._create_csr_matrix_from_pattern(pattern, replacement_values.arr)
    with pytest.raises(
        RuntimeError,
        match="SparseMatrix cannot be used after its Taichi runtime has been reset",
    ):
        matrix.update_values(replacement_values)

    del matrix, pattern
    gc.collect()


@test_utils.test(arch=[ti.cuda], offline_cache=False)
def test_shared_cuda_csr_pattern_keeps_cg_compatible_across_value_update():
    pattern, _, _ = _make_tridiagonal_pattern()
    values_host = np.asarray([2, -1, -1, 2, -1, -1, 2], dtype=np.float32)
    values = ti.ndarray(dtype=ti.f32, shape=7)
    scaled_values = ti.ndarray(dtype=ti.f32, shape=7)
    values.from_numpy(values_host)
    scaled_values.from_numpy(2 * values_host)
    prog = ti.lang.impl.get_runtime().prog
    matrix = ti.linalg.SparseMatrix(sm=prog._create_csr_matrix_from_pattern(pattern, values.arr))
    rhs = ti.ndarray(dtype=ti.f32, shape=3)
    rhs.from_numpy(np.asarray([0, 0, 4], dtype=np.float32))
    cg = ti.linalg.SparseCG(matrix, rhs, max_iter=20, atol=1e-5)

    solution, converged = cg.solve()
    assert converged
    np.testing.assert_allclose(solution.to_numpy(), [1, 2, 3], rtol=1e-5, atol=1e-5)
    matrix.update_values(scaled_values)
    updated_solution, updated_converged = cg.solve()
    assert updated_converged
    np.testing.assert_allclose(updated_solution.to_numpy(), [0.5, 1, 1.5], rtol=1e-5, atol=1e-5)
    stats = cg._debug_runtime_stats()
    assert stats["operations"]["workspace_builds"] == 1
    assert stats["operations"]["workspace_reuses"] == 1


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
)
def test_public_csr_pattern_matrix_spmv_update_and_solver_contract():
    row_offsets = ti.ndarray(dtype=ti.i32, shape=4)
    column_indices = ti.ndarray(dtype=ti.i32, shape=7)
    row_offsets.from_numpy(np.asarray([0, 2, 5, 7], dtype=np.int32))
    column_indices.from_numpy(np.asarray([0, 1, 0, 1, 2, 1, 2], dtype=np.int32))
    pattern = ti.linalg.SparsePattern.csr(
        rows=3,
        cols=3,
        row_offsets=row_offsets,
        column_indices=column_indices,
    )
    assert pattern.shape == (3, 3)
    assert pattern.storage_format == "csr"
    assert pattern.num_nonzeros == 7
    with pytest.raises(RuntimeError, match="available for BSR patterns only"):
        _ = pattern.block_size

    values_host = np.asarray([2, -1, -1, 2, -1, -1, 2], dtype=np.float32)
    values = ti.ndarray(dtype=ti.f32, shape=7)
    scaled_values = ti.ndarray(dtype=ti.f32, shape=7)
    values.from_numpy(values_host)
    scaled_values.from_numpy(2 * values_host)
    matrix = pattern.matrix(values)
    vector = ti.ndarray(dtype=ti.f32, shape=3)
    vector.from_numpy(np.asarray([1, 2, 3], dtype=np.float32))
    np.testing.assert_allclose((matrix @ vector).to_numpy(), [0, 0, 4])
    matrix.update_values(scaled_values)
    np.testing.assert_allclose((matrix @ vector).to_numpy(), [0, 0, 8])

    contract = matrix._get_format_contract()
    arch = ti.lang.impl.current_cfg().arch
    assert contract["pattern"]["ownership"] == "shared_immutable"
    assert contract["pattern"]["mutability"] == "fixed"
    assert not contract["pattern"]["empty_supported"]
    supports_public_cg = arch in (ti.cpu, ti.cuda)
    assert contract["operations"]["public_cg"] == supports_public_cg
    assert contract["operations"]["public_direct_solver"] == (arch == ti.cuda)
    assert contract["operations"]["public_jacobi_selection"] == supports_public_cg
    with pytest.raises(RuntimeError, match="no NumPy or host fallback"):
        pattern.matrix(2 * values_host)

    rhs = ti.ndarray(dtype=ti.f32, shape=3)
    rhs.from_numpy(np.asarray([0, 0, 8], dtype=np.float32))
    if supports_public_cg:
        cg = ti.linalg.SparseCG(matrix, rhs, max_iter=20, atol=1e-5)
        solution, converged = cg.solve()
        assert converged
        np.testing.assert_allclose(solution.to_numpy(), [1, 2, 3], rtol=1e-5, atol=1e-5)
        if arch == ti.cpu:
            solve_stats = cg._debug_runtime_stats()
            assert solve_stats["identity"]["method"] == "pcg_jacobi"
            assert solve_stats["identity"]["preconditioner_selection"] == "jacobi"
            assert solve_stats["preconditioner"]["identity"]["method"] == "jacobi"
    else:
        with pytest.raises(
            RuntimeError,
            match="operation 'public_cg'.*no fallback was performed",
        ):
            ti.linalg.SparseCG(matrix, rhs, max_iter=20, atol=1e-5)
    if arch == ti.cpu:
        solver = ti.linalg.SparseSolver(dtype=ti.f32)
        with pytest.raises(
            RuntimeError,
            match="operation 'public_direct_solver'.*no fallback was performed",
        ):
            solver.compute(matrix)


@test_utils.test(arch=[ti.cpu], offline_cache=False)
def test_public_cpu_csr_pattern_supports_f64_values():
    row_offsets = ti.ndarray(dtype=ti.i32, shape=3)
    column_indices = ti.ndarray(dtype=ti.i32, shape=4)
    row_offsets.from_numpy(np.asarray([0, 2, 4], dtype=np.int32))
    column_indices.from_numpy(np.asarray([0, 1, 0, 1], dtype=np.int32))
    pattern = ti.linalg.SparsePattern.csr(2, 2, row_offsets, column_indices)
    values = ti.ndarray(dtype=ti.f64, shape=4)
    values.from_numpy(np.asarray([2, -1, -1, 2], dtype=np.float64))
    matrix = ti.linalg.SparseMatrix.from_pattern(pattern, values)
    vector = ti.ndarray(dtype=ti.f64, shape=2)
    vector.from_numpy(np.asarray([1, 3], dtype=np.float64))
    np.testing.assert_allclose((matrix @ vector).to_numpy(), [-1, 5])
    assert matrix.dtype == ti.f64

    rhs = ti.ndarray(dtype=ti.f64, shape=2)
    rhs.from_numpy(np.asarray([-1, 5], dtype=np.float64))
    cg = ti.linalg.SparseCG(
        matrix, rhs, max_iter=16, atol=1e-12, preconditioner="jacobi"
    )
    solution, converged = cg.solve()
    assert converged
    np.testing.assert_allclose(
        solution.to_numpy(), [1, 3], rtol=1e-12, atol=1e-12
    )

    scaled_values = ti.ndarray(dtype=ti.f64, shape=4)
    scaled_values.from_numpy(np.asarray([4, -2, -2, 4], dtype=np.float64))
    matrix.update_values(scaled_values)
    updated_solution, updated_converged = cg.solve()
    assert updated_converged
    np.testing.assert_allclose(
        updated_solution.to_numpy(), [0.5, 1.5], rtol=1e-12, atol=1e-12
    )
    stats = cg._debug_runtime_stats()
    assert stats["identity"]["method"] == "pcg_jacobi"
    assert stats["identity"]["dtype"] == "f64"
    assert stats["operations"]["workspace_builds"] == 1
    assert stats["operations"]["workspace_reuses"] == 1
    assert stats["operations"]["preconditioner_auto_refresh_attempts"] == 1
    assert stats["operations"]["preconditioner_auto_refresh_successes"] == 1

    invalid_values = ti.ndarray(dtype=ti.f64, shape=4)
    invalid_values.from_numpy(np.asarray([4, -2, -2, 0], dtype=np.float64))
    matrix.update_values(invalid_values)
    with pytest.raises(RuntimeError, match="diagonal at row 1 is zero"):
        cg.solve()
    failed = cg._debug_runtime_stats()
    assert failed["operations"]["solve_calls"] == 2
    assert failed["operations"]["preconditioner_auto_refresh_attempts"] == 2
    assert failed["operations"]["preconditioner_auto_refresh_successes"] == 1
    assert failed["preconditioner"]["operations"]["numeric_refresh_failures"] == 1

    recovered_values = ti.ndarray(dtype=ti.f64, shape=4)
    recovered_values.from_numpy(np.asarray([6, -3, -3, 6], dtype=np.float64))
    matrix.update_values(recovered_values)
    recovered_solution, recovered_converged = cg.solve()
    assert recovered_converged
    np.testing.assert_allclose(
        recovered_solution.to_numpy(), [1.0 / 3.0, 1.0], rtol=1e-12, atol=1e-12
    )
    recovered = cg._debug_runtime_stats()
    assert recovered["operations"]["solve_calls"] == 3
    assert recovered["operations"]["workspace_builds"] == 1
    assert recovered["operations"]["workspace_reuses"] == 2
    assert recovered["operations"]["preconditioner_auto_refresh_attempts"] == 3
    assert recovered["operations"]["preconditioner_auto_refresh_successes"] == 2


@test_utils.test(arch=[ti.cpu], offline_cache=False)
def test_public_cpu_fixed_csr_cg_preserves_zero_iteration_contract():
    row_offsets = ti.ndarray(dtype=ti.i32, shape=3)
    column_indices = ti.ndarray(dtype=ti.i32, shape=2)
    values = ti.ndarray(dtype=ti.f32, shape=2)
    row_offsets.from_numpy(np.asarray([0, 1, 2], dtype=np.int32))
    column_indices.from_numpy(np.asarray([0, 1], dtype=np.int32))
    values.from_numpy(np.asarray([2, 3], dtype=np.float32))
    matrix = ti.linalg.SparsePattern.csr(
        2, 2, row_offsets, column_indices
    ).matrix(values)
    rhs = np.asarray([2, 3], dtype=np.float32)
    initial = np.zeros(2, dtype=np.float32)

    cg = ti.linalg.SparseCG(
        matrix, rhs, initial, max_iter=0, atol=1e-6
    )
    solution, converged = cg.solve()
    assert not converged
    assert cg._last_solve_result.status_code == 0
    assert cg._last_solve_result.termination_reason == "max_iterations"
    assert cg._last_solve_result.iterations == 0
    np.testing.assert_array_equal(solution.to_numpy(), initial)

    with pytest.raises(RuntimeError, match="non-negative max iterations"):
        ti.linalg.SparseCG(matrix, rhs, max_iter=-1, atol=1e-6)


@test_utils.test(arch=[ti.cpu], offline_cache=False)
def test_public_cpu_fixed_csr_cg_supports_relative_tolerance():
    row_offsets = ti.ndarray(dtype=ti.i32, shape=3)
    column_indices = ti.ndarray(dtype=ti.i32, shape=2)
    values = ti.ndarray(dtype=ti.f32, shape=2)
    row_offsets.from_numpy(np.asarray([0, 1, 2], dtype=np.int32))
    column_indices.from_numpy(np.asarray([0, 1], dtype=np.int32))
    values.from_numpy(np.ones(2, dtype=np.float32))
    matrix = ti.linalg.SparsePattern.csr(
        2, 2, row_offsets, column_indices
    ).matrix(values)
    rhs = np.asarray([1000.0, 2000.0], dtype=np.float32)
    initial = rhs * 0.95
    cg = ti.linalg.SparseCG(
        matrix,
        rhs,
        initial,
        max_iter=0,
        atol=0.0,
        rtol=0.1,
    )

    solution, converged = cg.solve()
    assert converged
    np.testing.assert_array_equal(solution.to_numpy(), initial)
    result = cg._last_solve_result
    rhs_norm = np.linalg.norm(rhs)
    assert result.relative_reference_norm == pytest.approx(rhs_norm)
    assert result.effective_tolerance == pytest.approx(0.1 * rhs_norm)
    stats = cg._debug_runtime_stats()
    assert stats["identity"]["relative_tolerance"] == pytest.approx(0.1)
    assert stats["identity"]["last_effective_tolerance"] == pytest.approx(
        0.1 * rhs_norm
    )
    assert stats["operations"]["host_scalar_reductions"] == 2


@test_utils.test(arch=[ti.cpu], offline_cache=False)
def test_public_csr_pattern_rejects_host_and_invalid_geometry():
    with pytest.raises(RuntimeError, match="cannot be constructed directly"):
        ti.linalg.SparsePattern()
    with pytest.raises(RuntimeError, match="no NumPy or host fallback"):
        ti.linalg.SparsePattern.csr(
            2,
            2,
            np.asarray([0, 1, 2], dtype=np.int32),
            np.asarray([0, 1], dtype=np.int32),
        )

    row_offsets = ti.ndarray(dtype=ti.i32, shape=3)
    column_indices = ti.ndarray(dtype=ti.i32, shape=2)
    with pytest.raises(RuntimeError, match="must be an integer"):
        ti.linalg.SparsePattern.csr(2.5, 2, row_offsets, column_indices)
    with pytest.raises(RuntimeError, match="rows and cols must be positive"):
        ti.linalg.SparsePattern.csr(0, 2, row_offsets, column_indices)


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
)
def test_public_csr_pattern_rejects_runtime_rebind():
    row_offsets = ti.ndarray(dtype=ti.i32, shape=3)
    column_indices = ti.ndarray(dtype=ti.i32, shape=4)
    row_offsets.from_numpy(np.asarray([0, 2, 4], dtype=np.int32))
    column_indices.from_numpy(np.asarray([0, 1, 0, 1], dtype=np.int32))
    pattern = ti.linalg.SparsePattern.csr(2, 2, row_offsets, column_indices)
    values = ti.ndarray(dtype=ti.f32, shape=4)
    values.from_numpy(np.asarray([2, -1, -1, 2], dtype=np.float32))
    matrix = pattern.matrix(values)
    arch = ti.lang.impl.current_cfg().arch

    ti.reset()
    ti.init(arch=arch)
    replacement_values = ti.ndarray(dtype=ti.f32, shape=4)
    with pytest.raises(
        RuntimeError,
        match="SparsePattern cannot be used after its Taichi runtime has been reset",
    ):
        pattern.matrix(replacement_values)
    with pytest.raises(
        RuntimeError,
        match="SparseMatrix cannot be used after its Taichi runtime has been reset",
    ):
        matrix.update_values(replacement_values)
