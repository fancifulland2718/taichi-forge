import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("ti_dtype", [ti.f32, ti.f64])
@test_utils.test(arch=[ti.cpu])
def test_cg(ti_dtype):
    n = 10
    random = np.random.default_rng(0).random((n, n))
    A_psd = np.dot(random, random.transpose()) + n * np.eye(n)
    atol = 1e-4 if ti_dtype == ti.f32 else 1e-10
    Abuilder = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=300, dtype=ti_dtype)
    b = ti.ndarray(dtype=ti_dtype, shape=n)
    x0 = ti.ndarray(dtype=ti_dtype, shape=n)

    @ti.kernel
    def fill(
        Abuilder: ti.types.sparse_matrix_builder(),
        InputArray: ti.types.ndarray(),
        b: ti.types.ndarray(),
    ):
        for i, j in ti.ndrange(n, n):
            Abuilder[i, j] += InputArray[i, j]
        for i in range(n):
            b[i] = i + 1

    fill(Abuilder, A_psd, b)
    A = Abuilder.build(dtype=ti_dtype)
    cg = ti.linalg.SparseCG(A, b, x0, max_iter=50, atol=atol)
    x, exit_code = cg.solve()
    res = np.linalg.solve(A_psd, b.to_numpy())
    assert exit_code == True
    assert cg._last_solve_info.converged
    assert 0 <= cg._last_solve_info.iterations <= 50
    assert cg._last_solve_info.residual_norm <= atol
    for i in range(n):
        assert x[i] == test_utils.approx(res[i], rel=1.0)

@pytest.mark.parametrize("ti_dtype", [ti.f32])
@test_utils.test(arch=[ti.cuda])
def test_cg_cuda(ti_dtype):
    n = 10
    random = np.random.default_rng(0).random((n, n))
    A_psd = np.dot(random, random.transpose()) + n * np.eye(n)
    atol = 1e-4
    Abuilder = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=300, dtype=ti_dtype)
    b = ti.ndarray(dtype=ti_dtype, shape=n)
    x0 = ti.ndarray(dtype=ti_dtype, shape=n)

    @ti.kernel
    def fill(
        Abuilder: ti.types.sparse_matrix_builder(),
        InputArray: ti.types.ndarray(),
        b: ti.types.ndarray(),
    ):
        for i, j in ti.ndrange(n, n):
            Abuilder[i, j] += InputArray[i, j]
        for i in range(n):
            b[i] = i + 1

    fill(Abuilder, A_psd, b)
    A = Abuilder.build(dtype=ti_dtype)
    cg = ti.linalg.SparseCG(A, b, x0, max_iter=50, atol=atol)
    x, exit_code = cg.solve()
    res = np.linalg.solve(A_psd, b.to_numpy())
    assert exit_code == True
    assert cg._last_solve_info.converged
    assert 0 <= cg._last_solve_info.iterations <= 50
    assert cg._last_solve_info.residual_norm <= atol
    for i in range(n):
        assert x[i] == test_utils.approx(res[i], rel=1.0)

    # A second solve reuses the solver-owned CUDA workspace.
    x_repeated, repeated_exit_code = cg.solve()
    assert repeated_exit_code
    assert cg._last_solve_info.converged
    for i in range(n):
        assert x_repeated[i] == test_utils.approx(res[i], rel=1.0)


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_cg_reports_non_convergence():
    n = 4
    builder = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=n, dtype=ti.f32)
    b = ti.ndarray(dtype=ti.f32, shape=n)
    x0 = ti.ndarray(dtype=ti.f32, shape=n)

    @ti.kernel
    def fill(
        builder: ti.types.sparse_matrix_builder(),
        b: ti.types.ndarray(),
    ):
        for i in range(n):
            builder[i, i] += i + 2
            b[i] = i + 1

    fill(builder, b)
    matrix = builder.build()
    cg = ti.linalg.SparseCG(matrix, b, x0, max_iter=0, atol=1e-6)
    _, converged = cg.solve()

    assert not converged
    assert not cg._last_solve_info.converged
    assert cg._last_solve_info.iterations == 0
    assert cg._last_solve_info.residual_norm == pytest.approx(cg._last_solve_info.initial_residual_norm)
