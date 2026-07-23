import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


def _numpy_dtype(dtype):
    return np.float64 if dtype == ti.f64 else np.float32


def _vector(values, dtype):
    values = np.asarray(values, dtype=_numpy_dtype(dtype))
    result = ti.ndarray(dtype, shape=values.size)
    result.from_numpy(values)
    return result


def _fixed_csr(dense, dtype):
    dense = np.asarray(dense, dtype=_numpy_dtype(dtype))
    rows, columns = dense.shape
    row_offsets = [0]
    column_indices = []
    values = []
    for row in range(rows):
        for column in range(columns):
            if dense[row, column] != 0:
                column_indices.append(column)
                values.append(dense[row, column])
        row_offsets.append(len(values))
    offsets = ti.ndarray(ti.i32, shape=len(row_offsets))
    indices = ti.ndarray(ti.i32, shape=len(column_indices))
    numeric = ti.ndarray(dtype, shape=len(values))
    offsets.from_numpy(np.asarray(row_offsets, dtype=np.int32))
    indices.from_numpy(np.asarray(column_indices, dtype=np.int32))
    numeric.from_numpy(np.asarray(values, dtype=_numpy_dtype(dtype)))
    pattern = ti.linalg.SparsePattern.csr(
        rows, columns, offsets, indices
    )
    return pattern.matrix(numeric)


def _operator(dense, dtype):
    return ti.linalg.experimental.LinearOperator.from_sparse_matrix(
        _fixed_csr(dense, dtype),
        traits=ti.linalg.experimental.OperatorTraits(singular=False),
    )


@pytest.mark.parametrize("dtype", [ti.f32, ti.f64])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_cpu_gmres_fixed_storage_reuse_and_exact_work(dtype):
    np_dtype = _numpy_dtype(dtype)
    dense = np.asarray(
        [
            [4.0, 1.5, 0.0, 0.0, -0.25, 0.0],
            [-0.5, 3.0, 0.75, 0.0, 0.0, 0.0],
            [0.0, -1.0, 2.5, 0.5, 0.0, 0.0],
            [0.25, 0.0, -0.75, 2.0, 0.5, 0.0],
            [0.0, 0.0, 0.25, -0.5, 3.5, 1.0],
            [0.1, 0.0, 0.0, 0.25, -0.75, 2.25],
        ],
        dtype=np_dtype,
    )
    exact = np.asarray([1.0, -0.5, 2.0, 0.25, -1.0, 0.75], dtype=np_dtype)
    rhs = _vector(dense @ exact, dtype)
    tolerance = 2e-5 if dtype == ti.f32 else 1e-11
    plan = ti.linalg.experimental.SolvePlan(
        _operator(dense, dtype),
        method="gmres",
        restart=8,
        max_iterations=32,
        atol=tolerance,
        rtol=tolerance,
    )

    first = plan.solve(rhs)
    second = plan.solve(rhs)
    assert first.converged and second.converged
    np.testing.assert_allclose(
        second.solution.to_numpy(), exact, rtol=10 * tolerance, atol=10 * tolerance
    )
    true_residual = np.linalg.norm(
        dense.astype(np.float64)
        @ second.solution.to_numpy().astype(np.float64)
        - rhs.to_numpy().astype(np.float64)
    )
    assert second.residual_norm == pytest.approx(
        true_residual, rel=5e-4 if dtype == ti.f32 else 1e-8, abs=tolerance
    )

    stats = plan.statistics()
    identity = stats["identity"]
    operations = stats["operations"]
    resources = stats["resources"]
    assert identity["method"] == "gmres"
    assert identity["preconditioning_side"] == "none"
    assert operations["restart"] == 8
    assert operations["orthogonalization_strategy"] == (
        "cgs2_always_reorthogonalize"
    )
    assert operations["orthogonalization_passes"] == 2
    assert operations["workspace_reuses"] == 1
    assert operations["multi_dot_calls"] == 2 * operations["total_iterations"]
    assert operations["operator_apply_calls"] == (
        operations["solve_calls"]
        + operations["total_iterations"]
        + operations["restart_cycles"]
    )
    assert resources["basis_vector_count"] == 9
    assert resources["basis_reserved_bytes"] == 9 * dense.shape[0] * np_dtype().nbytes
    assert resources["persistent_vector_count"] == 12
    assert resources["persistent_scalar_count"] == 113
    assert resources["transient_solver_workspace_bytes"] == 0


@pytest.mark.parametrize("dtype", [ti.f32, ti.f64])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_cpu_gmres_fixed_linear_right_preconditioner(dtype):
    np_dtype = _numpy_dtype(dtype)
    dense = np.asarray(
        [
            [8.0, 2.0, 0.0, 0.0],
            [-1.0, 4.0, 1.0, 0.0],
            [0.0, -0.5, 2.0, 0.5],
            [0.25, 0.0, -1.0, 1.0],
        ],
        dtype=np_dtype,
    )
    inverse_diagonal = np.diag(1.0 / np.diag(dense)).astype(np_dtype)
    exact = np.asarray([1.0, -2.0, 0.5, 1.5], dtype=np_dtype)
    operator = _operator(dense, dtype)
    preconditioner = _operator(inverse_diagonal, dtype)
    tolerance = 2e-5 if dtype == ti.f32 else 1e-11
    plan = ti.linalg.experimental.SolvePlan(
        operator,
        method="gmres",
        preconditioner=preconditioner,
        restart=8,
        max_iterations=24,
        atol=tolerance,
        rtol=tolerance,
    )
    result = plan.solve(_vector(dense @ exact, dtype))
    assert result.converged
    np.testing.assert_allclose(
        result.solution.to_numpy(), exact, rtol=10 * tolerance, atol=10 * tolerance
    )
    stats = plan.statistics()
    assert stats["identity"]["preconditioning_side"] == "right"
    assert stats["operations"]["preconditioner_apply_calls"] == (
        stats["operations"]["total_iterations"]
        + stats["operations"]["restart_cycles"]
    )
    assert stats["resources"]["persistent_vector_count"] == 13


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_cpu_gmres_happy_breakdown_and_bicgstab_separation():
    identity = _operator(np.eye(3, dtype=np.float32) * 2.0, ti.f32)
    rhs = _vector([2.0, -4.0, 1.0], ti.f32)
    plan = ti.linalg.experimental.SolvePlan(
        identity,
        method="gmres",
        restart=8,
        max_iterations=8,
        atol=1e-6,
    )
    result = plan.solve(rhs)
    assert result.converged and result.iterations == 1
    assert result.breakdown_reason == "none"
    assert plan.statistics()["operations"]["happy_breakdowns"] == 1

    skew = np.asarray([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    skew_operator = _operator(skew, ti.f32)
    skew_rhs = _vector([1.0, 0.0], ti.f32)
    bicgstab = ti.linalg.experimental.SolvePlan(
        skew_operator,
        method="bicgstab",
        max_iterations=8,
        atol=1e-6,
    ).solve(skew_rhs)
    gmres = ti.linalg.experimental.SolvePlan(
        skew_operator,
        method="gmres",
        restart=8,
        max_iterations=8,
        atol=1e-6,
    ).solve(skew_rhs)
    assert bicgstab.breakdown
    assert bicgstab.breakdown_reason == "alpha_denominator"
    assert gmres.converged
    np.testing.assert_allclose(gmres.solution.to_numpy(), [0.0, 1.0], atol=2e-6)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_gmres_restart_and_provider_boundaries():
    operator = _operator(np.eye(3, dtype=np.float32), ti.f32)
    with pytest.raises(Exception, match="restart must be one of"):
        ti.linalg.experimental.SolvePlan(
            operator, method="gmres", restart=4, atol=1e-6
        )
    with pytest.raises(Exception, match="accepted only"):
        ti.linalg.experimental.SolvePlan(
            operator, method="bicgstab", restart=8, atol=1e-6
        )
    singular_pc = ti.linalg.experimental.LinearOperator.from_sparse_matrix(
        _fixed_csr(np.eye(3, dtype=np.float32), ti.f32),
        traits=ti.linalg.experimental.OperatorTraits(singular=True),
    )
    with pytest.raises(Exception, match="GMRES rejects singular"):
        ti.linalg.experimental.SolvePlan(
            operator,
            method="gmres",
            preconditioner=singular_pc,
            restart=8,
            atol=1e-6,
        )
