"""Physical method selection and factor validity before persistent export."""

import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


def _matrix(dense, dtype):
    dense = np.asarray(dense, dtype=np.float32 if dtype == ti.f32 else np.float64)
    builder = ti.linalg.SparseMatrixBuilder(3, 3, max_num_triplets=9, dtype=dtype)

    @ti.kernel
    def fill(builder: ti.types.sparse_matrix_builder(), values: ti.types.ndarray()):
        for i, j in ti.ndrange(3, 3):
            if values[i, j] != 0:
                builder[i, j] += values[i, j]

    fill(builder, dense)
    return builder.build()


@pytest.mark.parametrize("dtype", [ti.f32, ti.f64])
@pytest.mark.parametrize("ordering", ["AMD", "COLAMD"])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_cpu_sparse_lu_uses_requested_ordering(dtype, ordering):
    dense = np.asarray([[4.0, 2.0, 0.0], [1.0, 3.0, 1.0], [0.0, 0.5, 2.0]], dtype=np.float64)
    solver = ti.linalg.SparseSolver(dtype=dtype, solver_type="LU", ordering=ordering)
    # Numerical equality alone cannot detect a silently ignored ordering.
    assert type(solver.solver).__name__ == f"EigenSparseSolverfloat{32 if dtype == ti.f32 else 64}LU{ordering}"
    assert not solver.info()
    matrix = _matrix(dense, dtype)
    solver.analyze_pattern(matrix)
    assert not solver.info()
    solver.factorize(matrix)
    assert solver.info()
    rhs = ti.ndarray(dtype, 3)
    rhs.from_numpy(np.asarray([1.0, 2.0, 3.0], dtype=np.float32 if dtype == ti.f32 else np.float64))
    np.testing.assert_allclose(solver.solve(rhs).to_numpy(), np.linalg.solve(dense, [1.0, 2.0, 3.0]), rtol=1e-6)


@pytest.mark.parametrize("route", ["compute", "factorize"])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_cpu_sparse_numeric_failure_invalidates_info_and_public_state(route):
    good = _matrix([[4.0, 1.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 2.0]], ti.f32)
    bad = _matrix([[2.0, 3.0, 0.0], [3.0, 2.0, 0.0], [0.0, 0.0, -2.0]], ti.f32)
    solver = ti.linalg.SparseSolver(solver_type="LLT")
    rhs = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    solver.compute(good)
    expected = solver.solve(rhs)
    assert solver.info()
    assert solver.provider_status()["reuse"]["factorization_active"]
    # Both matrices have identical stored patterns. A numeric failure must
    # invalidate the old factor, unlike rejection before numeric work starts.
    getattr(solver, route)(bad)
    assert not solver.info()
    assert not solver.provider_status()["reuse"]["factorization_active"]
    with pytest.raises(RuntimeError, match="no successful numeric factorization"):
        solver.solve(rhs)
    getattr(solver, route)(good)
    assert solver.info()
    assert solver.provider_status()["reuse"]["factorization_active"]
    np.testing.assert_allclose(solver.solve(rhs), expected, rtol=1e-6)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_cpu_sparse_solve_rejects_wrong_ndarray_layout_before_reading():
    matrix = _matrix([[4.0, 1.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 2.0]], ti.f32)
    solver = ti.linalg.SparseSolver(solver_type="LLT")
    solver.compute(matrix)
    for rhs, message in (
        (ti.ndarray(ti.f64, 3), "factorization dtype"),
        (ti.ndarray(ti.f32, 2), "RHS and"),
        (ti.ndarray(ti.f32, (3, 1)), "one-dimensional scalar"),
        (ti.Vector.ndarray(2, ti.f32, 3), "one-dimensional scalar"),
        (np.zeros(2, dtype=np.float32), "RHS and"),
    ):
        with pytest.raises(RuntimeError, match=message):
            solver.solve(rhs)
    # Rejected binding must not invalidate or corrupt the numeric factor.
    assert solver.info()
    rhs = np.asarray([1, 2, 3], dtype=np.float32)
    np.testing.assert_allclose(
        solver.solve(rhs),
        np.linalg.solve(np.asarray([[4, 1, 0], [1, 3, 0], [0, 0, 2]]), rhs),
        rtol=1e-6,
    )
