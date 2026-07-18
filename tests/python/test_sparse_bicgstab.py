import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


def _numpy_dtype(ti_dtype):
    return np.float32 if ti_dtype == ti.f32 else np.float64


def _build_matrix(dense, ti_dtype, storage_format="row_major"):
    n, m = dense.shape
    builder = ti.linalg.SparseMatrixBuilder(
        n,
        m,
        max_num_triplets=n * m,
        dtype=ti_dtype,
        storage_format=storage_format,
    )
    host = np.asarray(dense, dtype=_numpy_dtype(ti_dtype))

    @ti.kernel
    def assemble(
        matrix: ti.types.sparse_matrix_builder(),
        values: ti.types.ndarray(),
    ):
        for i, j in ti.ndrange(n, m):
            if values[i, j] != 0:
                matrix[i, j] += values[i, j]

    assemble(builder, host)
    return builder.build()


def _compressed_nonzeros(dense, storage_format):
    coordinates = list(zip(*np.nonzero(dense)))
    if storage_format == "row_major":
        coordinates.sort(key=lambda item: (item[0], item[1]))
    else:
        coordinates.sort(key=lambda item: (item[1], item[0]))
    return np.asarray([dense[row, col] for row, col in coordinates])


def _fixed_csr_matrix(dense, ti_dtype):
    rows, cols = dense.shape
    row_offsets_host = [0]
    column_indices_host = []
    values_host = []
    for row in range(rows):
        for col in range(cols):
            if dense[row, col] != 0:
                column_indices_host.append(col)
                values_host.append(dense[row, col])
        row_offsets_host.append(len(column_indices_host))
    row_offsets = ti.ndarray(dtype=ti.i32, shape=len(row_offsets_host))
    column_indices = ti.ndarray(
        dtype=ti.i32, shape=len(column_indices_host)
    )
    values = ti.ndarray(dtype=ti_dtype, shape=len(values_host))
    row_offsets.from_numpy(np.asarray(row_offsets_host, dtype=np.int32))
    column_indices.from_numpy(
        np.asarray(column_indices_host, dtype=np.int32)
    )
    values.from_numpy(
        np.asarray(values_host, dtype=_numpy_dtype(ti_dtype))
    )
    pattern = ti.linalg.SparsePattern.csr(
        rows, cols, row_offsets, column_indices
    )
    return pattern.matrix(values)


def _fixed_bsr_matrix(ti_dtype):
    np_dtype = _numpy_dtype(ti_dtype)
    blocks = np.asarray(
        [
            [[4.0, -1.0], [2.0, 5.0]],
            [[0.0, 1.0], [1.0, 0.0]],
            [[1.0, 0.0], [0.0, -2.0]],
            [[6.0, 1.0], [1.0, 7.0]],
        ],
        dtype=np_dtype,
    )
    row_offsets = ti.ndarray(dtype=ti.i32, shape=3)
    column_indices = ti.ndarray(dtype=ti.i32, shape=4)
    values = ti.ndarray(dtype=ti_dtype, shape=16)
    row_offsets.from_numpy(np.asarray([0, 2, 4], dtype=np.int32))
    column_indices.from_numpy(np.asarray([0, 1, 0, 1], dtype=np.int32))
    values.from_numpy(blocks.reshape(-1))
    pattern = ti.linalg.SparsePattern.bsr(
        2, 2, 2, row_offsets, column_indices
    )
    dense = np.block(
        [[blocks[0], blocks[1]], [blocks[2], blocks[3]]]
    )
    return pattern.matrix(values), dense, blocks.reshape(-1)


@pytest.mark.parametrize("ti_dtype", [ti.f32, ti.f64])
@pytest.mark.parametrize("storage_format", ["row_major", "col_major"])
@test_utils.test(arch=[ti.cpu], offline_cache=False)
def test_sparse_bicgstab_solves_and_reuses_provider_state(
    ti_dtype, storage_format
):
    dense = np.asarray(
        [
            [4.0, -1.0, 0.0, 0.0],
            [2.0, 5.0, 1.0, 0.0],
            [0.0, -2.0, 6.0, 1.0],
            [1.0, 0.0, 1.0, 7.0],
        ],
        dtype=_numpy_dtype(ti_dtype),
    )
    matrix = _build_matrix(dense, ti_dtype, storage_format)
    expected = np.asarray([-0.5, 1.25, 0.75, -1.0], dtype=dense.dtype)
    rhs = dense @ expected
    initial = np.asarray([0.25, -0.5, 0.5, 0.25], dtype=dense.dtype)
    atol = 2e-5 if ti_dtype == ti.f32 else 1e-11
    solver = ti.linalg.SparseBiCGSTAB(
        matrix,
        rhs,
        initial,
        max_iter=40,
        atol=atol,
        rtol=0.0,
    )

    solution, converged = solver.solve()
    assert converged
    np.testing.assert_allclose(solution, expected, rtol=atol, atol=atol)
    result = solver._last_solve_result
    assert result.termination_reason == "converged"
    assert result.residual_norm <= result.effective_tolerance
    first = solver._debug_runtime_stats()
    assert first["identity"]["method"] == "bicgstab"
    assert first["identity"]["preconditioner_method"] == "jacobi"
    assert first["operations"]["workspace_builds"] == 1
    assert first["operations"]["workspace_reuses"] == 0
    assert first["operations"]["operator_apply_calls"] is None
    assert first["operations"]["preconditioner_apply_calls"] is None
    assert first["resources"]["persistent_vector_count"] == 2
    assert first["resources"]["preconditioner_ownership_scope"] == (
        "provider_state"
    )

    expected_second = np.asarray(
        [1.5, -0.25, 0.5, 0.75], dtype=dense.dtype
    )
    solver.b = dense @ expected_second
    solver.x0 = np.zeros(4, dtype=dense.dtype)
    solution, converged = solver.solve()
    assert converged
    np.testing.assert_allclose(
        solution, expected_second, rtol=atol, atol=atol
    )
    second = solver._debug_runtime_stats()
    assert second["operations"]["solve_calls"] == 2
    assert second["operations"]["workspace_builds"] == 1
    assert second["operations"]["workspace_reuses"] == 1

    updated = ti.ndarray(dtype=ti_dtype, shape=matrix._num_nonzero())
    updated.from_numpy(
        _compressed_nonzeros(dense * 2, storage_format).astype(dense.dtype)
    )
    matrix.update_values(updated)
    solver.x0 = np.zeros(4, dtype=dense.dtype)
    solution, converged = solver.solve()
    assert converged
    np.testing.assert_allclose(
        solution, expected_second * 0.5, rtol=atol, atol=atol
    )
    third = solver._debug_runtime_stats()
    assert third["operations"]["solve_calls"] == 3
    assert third["operations"]["workspace_builds"] == 2
    assert third["operations"]["workspace_reuses"] == 1
    assert third["identity"]["operator_numeric_version"] == 2


@test_utils.test(arch=[ti.cpu], offline_cache=False)
def test_sparse_bicgstab_zero_iteration_relative_tolerance_contract():
    dense = np.asarray(
        [[3.0, -1.0, 0.0], [1.0, 4.0, 1.0], [0.0, -2.0, 5.0]],
        dtype=np.float32,
    )
    matrix = _build_matrix(dense, ti.f32)
    expected = np.asarray([1.0, -0.5, 2.0], dtype=np.float32)
    rhs = dense @ expected
    perturbation = np.asarray([0.01, -0.01, 0.005], dtype=np.float32)
    initial = expected + perturbation
    relative_residual = np.linalg.norm(dense @ initial - rhs) / np.linalg.norm(
        rhs
    )
    solver = ti.linalg.SparseBiCGSTAB(
        matrix,
        rhs,
        initial,
        max_iter=0,
        atol=0.0,
        rtol=float(relative_residual * 1.01),
    )

    solution, converged = solver.solve()
    assert converged
    np.testing.assert_array_equal(solution, initial)
    first = solver._last_solve_result
    assert first.iterations == 0
    assert first.residual_norm <= first.effective_tolerance
    assert solver._debug_runtime_stats()["operations"]["workspace_builds"] == 0

    scale = np.float32(1000.0)
    solver.b = rhs * scale
    solver.x0 = initial * scale
    _, converged = solver.solve()
    assert converged
    second = solver._last_solve_result
    assert second.relative_reference_norm == pytest.approx(
        first.relative_reference_norm * scale, rel=2e-6
    )
    assert second.effective_tolerance == pytest.approx(
        first.effective_tolerance * scale, rel=2e-6
    )

    solver.x0 = np.zeros(3, dtype=np.float32)
    _, converged = solver.solve()
    assert not converged
    assert solver._last_solve_result.termination_reason == "max_iterations"
    assert solver._last_solve_result.iterations == 0


@test_utils.test(arch=[ti.cpu], offline_cache=False)
def test_sparse_bicgstab_reports_nonfinite_and_zero_rhs_without_provider_run():
    dense = np.asarray(
        [[2.0, 1.0], [-1.0, 3.0]], dtype=np.float32
    )
    matrix = _build_matrix(dense, ti.f32)
    solver = ti.linalg.SparseBiCGSTAB(
        matrix,
        np.asarray([np.nan, 1.0], dtype=np.float32),
        np.zeros(2, dtype=np.float32),
        max_iter=8,
        atol=1e-6,
    )

    _, converged = solver.solve()
    assert not converged
    assert solver._last_solve_result.termination_reason == "breakdown"
    assert solver._last_solve_result.breakdown
    assert solver._debug_runtime_stats()["operations"]["workspace_builds"] == 0

    solver.b = np.zeros(2, dtype=np.float32)
    solver.x0 = np.ones(2, dtype=np.float32)
    solution, converged = solver.solve()
    assert converged
    np.testing.assert_array_equal(solution, np.zeros(2, dtype=np.float32))
    assert solver._last_solve_result.iterations == 0
    assert solver._last_solve_result.residual_norm == 0.0
    assert solver._debug_runtime_stats()["operations"]["workspace_builds"] == 0


@test_utils.test(arch=[ti.cpu], offline_cache=False)
def test_sparse_bicgstab_classifies_nonfinite_solution_as_breakdown():
    # Sparse multiplication by a structurally empty matrix does not read the
    # NaN entries that Eigen's recurrence can produce, so the true residual can
    # remain finite. Solution finiteness is therefore an independent part of
    # the public breakdown contract.
    matrix = _build_matrix(np.zeros((2, 2), dtype=np.float32), ti.f32)
    solver = ti.linalg.SparseBiCGSTAB(
        matrix,
        np.ones(2, dtype=np.float32),
        np.zeros(2, dtype=np.float32),
        max_iter=4,
        atol=1e-6,
    )

    solution, converged = solver.solve()
    assert not converged
    assert not np.isfinite(solution).all()
    assert np.isfinite(solver._last_solve_result.residual_norm)
    assert solver._last_solve_result.termination_reason == "breakdown"
    assert solver._last_solve_result.breakdown


@test_utils.test(arch=[ti.cpu], offline_cache=False)
def test_sparse_bicgstab_rejects_invalid_controls_and_geometry():
    matrix = _build_matrix(np.eye(2, dtype=np.float32), ti.f32)
    rhs = np.ones(2, dtype=np.float32)
    cases = [
        ({"max_iter": -1}, "non-negative max iterations"),
        ({"max_iter": 1.5}, "non-negative max iterations"),
        ({"max_iter": True}, "non-negative max iterations"),
        ({"atol": -1.0}, "atol must be finite and non-negative"),
        ({"atol": float("inf")}, "atol must be finite and non-negative"),
        ({"rtol": float("nan")}, "rtol must be finite and non-negative"),
        ({"rtol": True}, "rtol must be finite and non-negative"),
        ({"atol": 0.0, "rtol": 0.0}, "atol > 0 or rtol > 0"),
    ]
    for kwargs, message in cases:
        with pytest.raises(RuntimeError, match=message):
            ti.linalg.SparseBiCGSTAB(matrix, rhs, **kwargs)

    solver = ti.linalg.SparseBiCGSTAB(
        matrix, np.ones(3, dtype=np.float32)
    )
    with pytest.raises(RuntimeError, match=r"RHS must have shape \(2,\)"):
        solver.solve()
    solver.b = np.ones(2, dtype=np.int32)
    with pytest.raises(RuntimeError, match="RHS must have a floating dtype"):
        solver.solve()

    rectangular = _build_matrix(
        np.asarray([[1.0, 2.0, 0.0], [0.0, 3.0, 4.0]], dtype=np.float32),
        ti.f32,
    )
    with pytest.raises(RuntimeError, match="non-empty square matrix"):
        ti.linalg.SparseBiCGSTAB(
            rectangular, np.ones(2, dtype=np.float32)
        )
    fixed_rectangular = _fixed_csr_matrix(
        np.asarray([[1.0, 2.0, 0.0], [0.0, 3.0, 4.0]], dtype=np.float32),
        ti.f32,
    )
    assert not fixed_rectangular._get_format_contract()["operations"][
        "public_bicgstab"
    ]
    with pytest.raises(
        RuntimeError,
        match="operation 'public_bicgstab'.*no fallback was performed",
    ):
        ti.linalg.SparseBiCGSTAB(
            fixed_rectangular, np.ones(2, dtype=np.float32)
        )


@pytest.mark.parametrize("ti_dtype", [ti.f32, ti.f64])
@test_utils.test(arch=[ti.cpu], offline_cache=False)
def test_sparse_bicgstab_solves_fixed_csr_without_eigen_shadow(ti_dtype):
    np_dtype = _numpy_dtype(ti_dtype)
    dense = np.asarray(
        [
            [4.0, -1.0, 0.0, 0.0],
            [2.0, 5.0, 1.0, 0.0],
            [0.0, -2.0, 6.0, 1.0],
            [1.0, 0.0, 1.0, 7.0],
        ],
        dtype=np_dtype,
    )
    matrix = _fixed_csr_matrix(dense, ti_dtype)
    contract = matrix._get_format_contract()
    assert contract["operations"]["public_bicgstab"]
    expected = np.asarray([-0.5, 1.25, 0.75, -1.0], dtype=np_dtype)
    rhs = dense @ expected
    initial = np.asarray([0.25, -0.5, 0.5, 0.25], dtype=np_dtype)
    atol = 2e-4 if ti_dtype == ti.f32 else 1e-10
    solver = ti.linalg.SparseBiCGSTAB(
        matrix, rhs, initial, max_iter=40, atol=atol
    )

    solution, converged = solver.solve()
    assert converged
    np.testing.assert_allclose(solution, expected, rtol=atol, atol=atol)
    first = solver._debug_runtime_stats()
    matrix_first = matrix._debug_runtime_stats()
    assert first["identity"]["method"] == "bicgstab"
    assert first["identity"]["preconditioner_method"] == "identity"
    assert first["operations"]["workspace_builds"] == 1
    assert first["operations"]["workspace_reuses"] == 0
    assert first["operations"]["operator_apply_calls"] >= 3
    assert first["operations"]["host_scalar_reductions"] > 0
    assert first["operations"]["preconditioner_apply_calls"] == 0
    assert first["resources"]["persistent_vector_count"] == 8
    assert first["resources"]["transient_solver_workspace_bytes"] == 0
    assert matrix_first["operations"]["pattern_builds"] == 0
    assert matrix_first["operations"]["spmv_plan_builds"] == 1

    updated = ti.ndarray(dtype=ti_dtype, shape=matrix._num_nonzero())
    updated.from_numpy(
        _compressed_nonzeros(dense * 2, "row_major").astype(np_dtype)
    )
    matrix.update_values(updated)
    solver.x0 = np.zeros(4, dtype=np_dtype)
    solution, converged = solver.solve()
    assert converged
    np.testing.assert_allclose(
        solution, expected * 0.5, rtol=atol, atol=atol
    )
    second = solver._debug_runtime_stats()
    assert second["operations"]["workspace_builds"] == 1
    assert second["operations"]["workspace_reuses"] == 1
    assert second["identity"]["operator_numeric_version"] == 2


@pytest.mark.parametrize("ti_dtype", [ti.f32, ti.f64])
@test_utils.test(arch=[ti.cpu], offline_cache=False)
def test_sparse_bicgstab_solves_fixed_bsr_without_scalar_expansion(ti_dtype):
    matrix, dense, values_host = _fixed_bsr_matrix(ti_dtype)
    contract = matrix._get_format_contract()
    assert contract["operations"]["public_bicgstab"]
    np_dtype = _numpy_dtype(ti_dtype)
    expected = np.asarray([-0.5, 1.25, 0.75, -1.0], dtype=np_dtype)
    rhs = dense @ expected
    atol = 2e-4 if ti_dtype == ti.f32 else 1e-10
    solver = ti.linalg.SparseBiCGSTAB(
        matrix, rhs, max_iter=40, atol=atol
    )

    solution, converged = solver.solve()
    assert converged
    np.testing.assert_allclose(solution, expected, rtol=atol, atol=atol)
    matrix_first = matrix._debug_runtime_stats()
    assert matrix_first["identity"]["storage_format"] == "bsr"
    assert matrix_first["identity"]["block_size"] == 2
    assert matrix_first["operations"]["pattern_builds"] == 0

    updated = ti.ndarray(dtype=ti_dtype, shape=values_host.size)
    updated.from_numpy(values_host * 2)
    matrix.update_values(updated)
    solver.x0 = np.zeros(4, dtype=np_dtype)
    solution, converged = solver.solve()
    assert converged
    np.testing.assert_allclose(
        solution, expected * 0.5, rtol=atol, atol=atol
    )
    stats = solver._debug_runtime_stats()
    assert stats["operations"]["workspace_builds"] == 1
    assert stats["operations"]["workspace_reuses"] == 1


@test_utils.test(arch=[ti.cpu], offline_cache=False)
def test_sparse_bicgstab_fixed_zero_operator_reports_finite_breakdown():
    row_offsets = ti.ndarray(dtype=ti.i32, shape=3)
    column_indices = ti.ndarray(dtype=ti.i32, shape=2)
    values = ti.ndarray(dtype=ti.f32, shape=2)
    row_offsets.from_numpy(np.asarray([0, 1, 2], dtype=np.int32))
    column_indices.from_numpy(np.asarray([0, 1], dtype=np.int32))
    values.from_numpy(np.zeros(2, dtype=np.float32))
    pattern = ti.linalg.SparsePattern.csr(
        2, 2, row_offsets, column_indices
    )
    matrix = pattern.matrix(values)
    solver = ti.linalg.SparseBiCGSTAB(
        matrix,
        np.ones(2, dtype=np.float32),
        np.zeros(2, dtype=np.float32),
        max_iter=4,
        atol=1e-6,
    )

    solution, converged = solver.solve()
    assert not converged
    np.testing.assert_array_equal(solution, np.zeros(2, dtype=np.float32))
    assert solver._last_solve_result.termination_reason == "breakdown"
    assert solver._last_solve_result.breakdown
    assert np.isfinite(solver._last_solve_result.residual_norm)
    assert solver._last_solve_result.iterations == 0


@test_utils.test(arch=[ti.cpu], offline_cache=False)
def test_sparse_bicgstab_rejects_operator_owned_bsr_adapter():
    row_offsets = ti.ndarray(dtype=ti.i32, shape=3)
    column_indices = ti.ndarray(dtype=ti.i32, shape=2)
    values = ti.ndarray(dtype=ti.f32, shape=8)
    row_offsets.from_numpy(np.asarray([0, 1, 2], dtype=np.int32))
    column_indices.from_numpy(np.asarray([0, 1], dtype=np.int32))
    values.from_numpy(
        np.asarray([1.0, 0.0, 0.0, 1.0] * 2, dtype=np.float32)
    )
    program = ti.lang.impl.get_runtime().prog
    core = program._create_cpu_bsr_matrix(
        2, 2, 2, row_offsets.arr, column_indices.arr, values.arr
    )
    matrix = ti.linalg.SparseMatrix(sm=core)

    contract = matrix._get_format_contract()
    assert not contract["operations"]["public_bicgstab"]
    with pytest.raises(
        RuntimeError,
        match="operation 'public_bicgstab'.*no fallback was performed",
    ):
        ti.linalg.SparseBiCGSTAB(matrix, np.ones(4, dtype=np.float32))


@pytest.mark.parametrize("storage_format", ["csr", "bsr"])
@test_utils.test(arch=[ti.cuda], offline_cache=False)
def test_private_cuda_fixed_bicgstab_solves_and_reuses_workspace(
    storage_format,
):
    if storage_format == "csr":
        dense = np.asarray(
            [
                [4.0, -1.0, 0.0, 0.0],
                [2.0, 5.0, 1.0, 0.0],
                [0.0, -2.0, 6.0, 1.0],
                [1.0, 0.0, 1.0, 7.0],
            ],
            dtype=np.float32,
        )
        matrix = _fixed_csr_matrix(dense, ti.f32)
        updated_host = _compressed_nonzeros(dense * 2, "row_major").astype(
            np.float32
        )
    else:
        try:
            matrix, dense, values_host = _fixed_bsr_matrix(ti.f32)
        except RuntimeError as exc:
            if "does not support generic BSR SpMV" in str(exc):
                pytest.skip("loaded cuSPARSE provider lacks generic BSR SpMV")
            raise
        updated_host = values_host * 2

    contract = matrix._get_format_contract()
    assert not contract["operations"]["public_bicgstab"]
    expected = np.asarray([-0.5, 1.25, 0.75, -1.0], dtype=np.float32)
    rhs_host = dense @ expected
    initial_host = np.asarray(
        [0.25, -0.5, 0.5, 0.25], dtype=np.float32
    )
    rhs = ti.ndarray(dtype=ti.f32, shape=4)
    solution = ti.ndarray(dtype=ti.f32, shape=4)
    rhs.from_numpy(rhs_host)
    solution.from_numpy(initial_host)
    prog = ti.lang.impl.get_runtime().prog
    plan = ti._lib.core._make_cuda_fixed_sparse_bicgstab_plan(
        prog, matrix.matrix, 40, 5e-5, False
    )

    plan.solve(prog, solution.arr, rhs.arr)
    assert plan.get_status() == 2
    assert dict(plan._get_last_result())["termination_reason"] == "converged"
    np.testing.assert_allclose(
        solution.to_numpy(), expected, rtol=5e-5, atol=5e-5
    )
    first = plan._debug_runtime_stats()
    matrix_first = matrix._debug_runtime_stats()
    assert first["identity"]["method"] == (
        "bicgstab_identity_host_scalar_probe"
    )
    assert first["identity"]["preconditioner_method"] == "identity"
    assert first["operations"]["workspace_builds"] == 1
    assert first["operations"]["workspace_reuses"] == 0
    assert first["operations"]["operator_apply_calls"] >= 3
    assert first["operations"]["host_scalar_reductions"] > 0
    assert first["operations"]["host_scalar_readbacks"] == first[
        "operations"
    ]["host_scalar_reductions"]
    assert first["operations"]["host_synchronizations"] == first[
        "operations"
    ]["host_scalar_reductions"]
    assert first["operations"]["host_synchronizations_exact"]
    assert first["operations"]["host_synchronization_scope"] == (
        "cublas_host_pointer_reductions"
    )
    assert first["operations"]["device_scalar_operations"] == 0
    assert not first["operations"]["bounded_masked_execution"]
    assert first["resources"]["persistent_vector_count"] == 6
    assert first["resources"]["persistent_vector_reserved_bytes"] == 96
    assert first["resources"]["persistent_scalar_count"] == 0
    assert first["resources"]["cublas_handle_count"] == 1
    assert first["resources"]["transient_solver_workspace_bytes"] == 0
    assert first["resources"]["shared_primitive_workspace_bytes"] is None
    assert first["transfers"]["device_to_host_bytes"] == (
        4 * first["operations"]["host_scalar_readbacks"]
    )
    assert matrix_first["operations"]["pattern_builds"] == 0
    assert matrix_first["resources"]["pattern_storage_shared"]

    updated = ti.ndarray(dtype=ti.f32, shape=updated_host.size)
    updated.from_numpy(updated_host)
    matrix.update_values(updated)
    solution.fill(0)
    plan.solve(prog, solution.arr, rhs.arr)
    assert plan.get_status() == 2
    np.testing.assert_allclose(
        solution.to_numpy(), expected * 0.5, rtol=5e-5, atol=5e-5
    )
    second = plan._debug_runtime_stats()
    assert second["operations"]["workspace_builds"] == 1
    assert second["operations"]["workspace_reuses"] == 1
    assert second["identity"]["operator_numeric_version"] == 2

    solution.from_numpy(expected * 0.5)
    plan.solve(prog, solution.arr, rhs.arr)
    assert plan.get_status() == 2
    assert plan.get_iterations() == 0
    np.testing.assert_array_equal(solution.to_numpy(), expected * 0.5)

    rhs.fill(0)
    solution.fill(1)
    plan.solve(prog, solution.arr, rhs.arr)
    assert plan.get_status() == 2
    assert plan.get_iterations() == 0
    np.testing.assert_array_equal(
        solution.to_numpy(), np.zeros(4, dtype=np.float32)
    )
    final = plan._debug_runtime_stats()
    assert final["operations"]["solve_calls"] == 4
    assert final["operations"]["workspace_builds"] == 1
    assert final["operations"]["workspace_reuses"] == 3


@test_utils.test(arch=[ti.cuda], offline_cache=False)
def test_private_cuda_fixed_bicgstab_breakdown_and_pattern_boundary():
    row_offsets = ti.ndarray(dtype=ti.i32, shape=3)
    column_indices = ti.ndarray(dtype=ti.i32, shape=2)
    values = ti.ndarray(dtype=ti.f32, shape=2)
    row_offsets.from_numpy(np.asarray([0, 1, 2], dtype=np.int32))
    column_indices.from_numpy(np.asarray([0, 1], dtype=np.int32))
    values.from_numpy(np.zeros(2, dtype=np.float32))
    pattern = ti.linalg.SparsePattern.csr(
        2, 2, row_offsets, column_indices
    )
    matrix = pattern.matrix(values)
    rhs = ti.ndarray(dtype=ti.f32, shape=2)
    solution = ti.ndarray(dtype=ti.f32, shape=2)
    rhs.fill(1)
    solution.fill(0)
    prog = ti.lang.impl.get_runtime().prog
    plan = ti._lib.core._make_cuda_fixed_sparse_bicgstab_plan(
        prog, matrix.matrix, 4, 1e-6, False
    )

    plan.solve(prog, solution.arr, rhs.arr)
    assert plan.get_status() == 1
    result = dict(plan._get_last_result())
    assert result["termination_reason"] == "breakdown"
    assert result["breakdown"]
    assert np.isfinite(result["residual_norm"])
    assert result["iterations"] == 0
    np.testing.assert_array_equal(
        solution.to_numpy(), np.zeros(2, dtype=np.float32)
    )

    builder_matrix = _build_matrix(np.eye(2, dtype=np.float32), ti.f32)
    with pytest.raises(
        RuntimeError,
        match="caller-owned shared CSR/BSR pattern with pattern_builds=0",
    ):
        ti._lib.core._make_cuda_fixed_sparse_bicgstab_plan(
            prog, builder_matrix.matrix, 4, 1e-6, False
        )


@pytest.mark.parametrize("storage_format", ["csr", "bsr"])
@test_utils.test(arch=[ti.vulkan], offline_cache=False)
def test_private_vulkan_fixed_bicgstab_solves_and_reuses_workspace(
    storage_format,
):
    if storage_format == "csr":
        dense = np.asarray(
            [
                [4.0, -1.0, 0.0, 0.0],
                [2.0, 5.0, 1.0, 0.0],
                [0.0, -2.0, 6.0, 1.0],
                [1.0, 0.0, 1.0, 7.0],
            ],
            dtype=np.float32,
        )
        matrix = _fixed_csr_matrix(dense, ti.f32)
        updated_host = _compressed_nonzeros(dense * 2, "row_major").astype(
            np.float32
        )
    else:
        matrix, dense, values_host = _fixed_bsr_matrix(ti.f32)
        updated_host = values_host * 2

    contract = matrix._get_format_contract()
    assert not contract["operations"]["public_bicgstab"]
    expected = np.asarray([-0.5, 1.25, 0.75, -1.0], dtype=np.float32)
    rhs_host = dense @ expected
    initial_host = np.asarray(
        [0.25, -0.5, 0.5, 0.25], dtype=np.float32
    )
    rhs = ti.ndarray(dtype=ti.f32, shape=4)
    solution = ti.ndarray(dtype=ti.f32, shape=4)
    rhs.from_numpy(rhs_host)
    solution.from_numpy(initial_host)
    prog = ti.lang.impl.get_runtime().prog
    max_iterations = 12
    plan = ti._lib.core._make_vulkan_fixed_sparse_bicgstab_plan(
        prog, matrix.matrix, max_iterations, 5e-5, False
    )

    plan.solve(prog, solution.arr, rhs.arr)
    assert plan.get_status() == 2
    assert dict(plan._get_last_result())["termination_reason"] == "converged"
    np.testing.assert_allclose(
        solution.to_numpy(), expected, rtol=5e-5, atol=5e-5
    )
    first = plan._debug_runtime_stats()
    matrix_first = matrix._debug_runtime_stats()
    assert first["identity"]["method"] == (
        "bicgstab_identity_bounded_true_residual_probe"
    )
    assert first["identity"]["preconditioner_method"] == "identity"
    assert first["operations"]["workspace_builds"] == 1
    assert first["operations"]["workspace_reuses"] == 0
    assert first["operations"]["operator_apply_calls"] == (
        2 + 4 * max_iterations
    )
    assert first["operations"]["operator_apply_call_scope"] == (
        "scheduled_dispatches"
    )
    assert first["operations"]["masked_operator_dispatches"] == (
        4 * max_iterations
    )
    assert first["operations"]["host_scalar_reductions"] == 0
    assert first["operations"]["device_scalar_operations"] == (
        7 + 14 * max_iterations
    )
    assert first["operations"]["host_scalar_readbacks"] == 20
    assert first["operations"]["host_synchronizations"] == 1
    assert not first["operations"]["host_synchronizations_exact"]
    assert first["operations"]["host_synchronization_scope"] == (
        "explicit_plan_only"
    )
    assert first["operations"]["bounded_masked_execution"]
    assert not first["operations"]["fixed_iteration_only"]
    assert first["resources"]["persistent_vector_count"] == 8
    assert first["resources"]["persistent_vector_reserved_bytes"] == 128
    assert first["resources"]["persistent_scalar_count"] == 20
    assert first["resources"]["persistent_scalar_reserved_bytes"] == 80
    assert first["resources"]["cublas_handle_count"] == 0
    assert first["resources"]["transient_solver_workspace_bytes"] == 0
    assert first["resources"]["shared_primitive_workspace_bytes"] == (
        prog._vulkan_sparse_algebra_workspace_bytes()
        + prog.vulkan_reduce_workspace_bytes()
    )
    assert first["resources"][
        "shared_primitive_workspace_ownership_scope"
    ] == "program_sparse_algebra_and_reduce_cache"
    assert first["transfers"]["device_to_host_bytes"] == 80
    assert first["transfers"]["device_to_device_bytes"] == (
        (2 * max_iterations + 2) * 4 * 4
    )
    assert matrix_first["operations"]["pattern_builds"] == 0
    assert matrix_first["resources"]["pattern_storage_shared"]

    updated = ti.ndarray(dtype=ti.f32, shape=updated_host.size)
    updated.from_numpy(updated_host)
    matrix.update_values(updated)
    solution.fill(0)
    plan.solve(prog, solution.arr, rhs.arr)
    assert plan.get_status() == 2
    np.testing.assert_allclose(
        solution.to_numpy(), expected * 0.5, rtol=5e-5, atol=5e-5
    )
    second = plan._debug_runtime_stats()
    assert second["operations"]["workspace_builds"] == 1
    assert second["operations"]["workspace_reuses"] == 1
    assert second["identity"]["operator_numeric_version"] == 2

    solution.from_numpy(expected * 0.5)
    plan.solve(prog, solution.arr, rhs.arr)
    assert plan.get_status() == 2
    assert plan.get_iterations() == 0
    np.testing.assert_array_equal(solution.to_numpy(), expected * 0.5)

    rhs.fill(0)
    solution.fill(1)
    plan.solve(prog, solution.arr, rhs.arr)
    assert plan.get_status() == 2
    assert plan.get_iterations() == 0
    np.testing.assert_array_equal(
        solution.to_numpy(), np.zeros(4, dtype=np.float32)
    )
    final = plan._debug_runtime_stats()
    assert final["operations"]["solve_calls"] == 4
    assert final["operations"]["workspace_builds"] == 1
    assert final["operations"]["workspace_reuses"] == 3
    assert final["operations"]["host_synchronizations"] == 4


@test_utils.test(arch=[ti.vulkan], offline_cache=False)
def test_private_vulkan_fixed_bicgstab_breakdown_and_pattern_boundary():
    row_offsets = ti.ndarray(dtype=ti.i32, shape=3)
    column_indices = ti.ndarray(dtype=ti.i32, shape=2)
    values = ti.ndarray(dtype=ti.f32, shape=2)
    row_offsets.from_numpy(np.asarray([0, 1, 2], dtype=np.int32))
    column_indices.from_numpy(np.asarray([0, 1], dtype=np.int32))
    values.from_numpy(np.zeros(2, dtype=np.float32))
    pattern = ti.linalg.SparsePattern.csr(
        2, 2, row_offsets, column_indices
    )
    matrix = pattern.matrix(values)
    rhs = ti.ndarray(dtype=ti.f32, shape=2)
    solution = ti.ndarray(dtype=ti.f32, shape=2)
    rhs.fill(1)
    solution.fill(0)
    prog = ti.lang.impl.get_runtime().prog
    plan = ti._lib.core._make_vulkan_fixed_sparse_bicgstab_plan(
        prog, matrix.matrix, 4, 1e-6, False
    )

    plan.solve(prog, solution.arr, rhs.arr)
    assert plan.get_status() == 1
    result = dict(plan._get_last_result())
    assert result["termination_reason"] == "breakdown"
    assert result["breakdown"]
    assert np.isfinite(result["residual_norm"])
    assert result["iterations"] == 0
    np.testing.assert_array_equal(
        solution.to_numpy(), np.zeros(2, dtype=np.float32)
    )

    zero_iteration = ti._lib.core._make_vulkan_fixed_sparse_bicgstab_plan(
        prog, matrix.matrix, 0, 1e-6, False
    )
    zero_iteration.solve(prog, solution.arr, rhs.arr)
    assert zero_iteration.get_status() == 0
    assert zero_iteration.get_iterations() == 0
    np.testing.assert_array_equal(
        solution.to_numpy(), np.zeros(2, dtype=np.float32)
    )

    builder_matrix = _build_matrix(np.eye(2, dtype=np.float32), ti.f32)
    with pytest.raises(
        RuntimeError,
        match="caller-owned shared CSR/BSR pattern with pattern_builds=0",
    ):
        ti._lib.core._make_vulkan_fixed_sparse_bicgstab_plan(
            prog, builder_matrix.matrix, 4, 1e-6, False
        )


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_sparse_bicgstab_rejects_gpu_without_host_fallback():
    matrix = _build_matrix(
        np.asarray([[2.0, 1.0], [-1.0, 3.0]], dtype=np.float32), ti.f32
    )
    contract = matrix._get_format_contract()
    assert not contract["operations"]["public_bicgstab"]
    with pytest.raises(
        RuntimeError,
        match="operation 'public_bicgstab'.*no fallback was performed",
    ):
        ti.linalg.SparseBiCGSTAB(
            matrix, np.ones(2, dtype=np.float32)
        )
