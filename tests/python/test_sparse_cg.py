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


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_cg_solve_plan_runtime_statistics():
    n = 4
    builder = ti.linalg.SparseMatrixBuilder(
        n,
        n,
        max_num_triplets=n,
        dtype=ti.f32,
        storage_format="row_major",
    )
    b = ti.ndarray(dtype=ti.f32, shape=n)
    x0 = ti.ndarray(dtype=ti.f32, shape=n)

    @ti.kernel
    def fill(
        matrix: ti.types.sparse_matrix_builder(),
        rhs: ti.types.ndarray(),
    ):
        for i in range(n):
            matrix[i, i] += i + 2
            rhs[i] = i + 1

    fill(builder, b)
    matrix = builder.build()
    cg = ti.linalg.SparseCG(matrix, b, x0, max_iter=20, atol=1e-5)
    before = cg._debug_runtime_stats()
    assert before["schema_version"] == 1
    assert before["identity"]["method"] == "cg"
    assert before["identity"]["operator_pattern_version"] == 1
    assert before["identity"]["operator_numeric_version"] == 1
    assert before["operations"]["solve_calls"] == 0

    _, converged = cg.solve()
    assert converged
    first = cg._debug_runtime_stats()
    assert first["operations"]["solve_calls"] == 1
    assert first["identity"]["last_solve_pattern_version"] == 1
    assert first["identity"]["last_solve_numeric_version"] == 1
    assert not first["identity"]["operator_pattern_changed_since_last_solve"]
    assert not first["identity"]["operator_numeric_changed_since_last_solve"]

    if first["identity"]["backend_family"] == "cpu":
        assert first["resources"]["persistent_vector_count"] == 2
        assert first["resources"]["persistent_vector_reserved_bytes"] == 2 * n * 4
        assert not first["resources"]["solver_state_rebuilt_each_solve"]
        assert first["resources"]["transient_solver_workspace_bytes"] is None
        assert first["operations"]["operator_apply_calls"] is None
        assert first["operations"]["workspace_builds"] == 1
        assert first["operations"]["workspace_reuses"] == 0
    else:
        assert first["resources"]["persistent_vector_count"] == 3
        assert first["resources"]["persistent_vector_reserved_bytes"] == 3 * n * 4
        assert first["resources"]["cublas_handle_count"] == 1
        assert first["operations"]["workspace_builds"] == 1
        assert first["operations"]["workspace_reuses"] == 0
        assert first["operations"]["operator_apply_calls"] > 0
        assert first["operations"]["host_scalar_reductions"] > 0
        assert first["transfers"]["device_to_device_bytes"] > 0

    values = ti.ndarray(dtype=ti.f32, shape=n)
    values.fill(3)
    matrix._update_values(values)
    stale = cg._debug_runtime_stats()
    assert stale["identity"]["operator_pattern_version"] == 1
    assert stale["identity"]["operator_numeric_version"] == 2
    assert not stale["identity"]["operator_pattern_changed_since_last_solve"]
    assert stale["identity"]["operator_numeric_changed_since_last_solve"]

    _, converged = cg.solve()
    assert converged
    second = cg._debug_runtime_stats()
    assert second["operations"]["solve_calls"] == 2
    assert second["identity"]["last_solve_pattern_version"] == 1
    assert second["identity"]["last_solve_numeric_version"] == 2
    assert not second["identity"]["operator_numeric_changed_since_last_solve"]
    if second["identity"]["backend_family"] == "cuda":
        assert second["operations"]["workspace_builds"] == 1
        assert second["operations"]["workspace_reuses"] == 1
    else:
        assert second["operations"]["workspace_builds"] == 2
        assert second["operations"]["workspace_reuses"] == 0


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_cg_reloads_rhs_and_initial_guess_each_solve():
    n = 16
    builder = ti.linalg.SparseMatrixBuilder(
        n,
        n,
        max_num_triplets=3 * n - 2,
        dtype=ti.f32,
        storage_format="row_major",
    )

    @ti.kernel
    def assemble(matrix: ti.types.sparse_matrix_builder()):
        for i in range(n):
            matrix[i, i] += 2.0
            if i > 0:
                matrix[i, i - 1] += -1.0
            if i + 1 < n:
                matrix[i, i + 1] += -1.0

    def apply_poisson(x):
        y = 2.0 * x.copy()
        y[1:] -= x[:-1]
        y[:-1] -= x[1:]
        return y.astype(np.float32)

    assemble(builder)
    matrix = builder.build()
    rhs = ti.ndarray(dtype=ti.f32, shape=n)
    initial = ti.ndarray(dtype=ti.f32, shape=n)
    first_expected = np.linspace(0.25, 1.0, n, dtype=np.float32)
    second_expected = np.linspace(1.0, -0.5, n, dtype=np.float32)
    rhs.from_numpy(apply_poisson(first_expected))
    initial.fill(0)
    cg = ti.linalg.SparseCG(
        matrix, rhs, initial, max_iter=64, atol=1e-5
    )

    first, first_ok = cg.solve()
    assert first_ok
    np.testing.assert_allclose(
        first.to_numpy() if hasattr(first, "to_numpy") else first,
        first_expected,
        rtol=1e-5,
        atol=1e-5,
    )

    second_initial = np.full(n, 0.125, dtype=np.float32)
    second_rhs = apply_poisson(second_expected)
    rhs.from_numpy(second_rhs)
    initial.from_numpy(second_initial)
    second, second_ok = cg.solve()
    assert second_ok
    np.testing.assert_allclose(
        second.to_numpy() if hasattr(second, "to_numpy") else second,
        second_expected,
        rtol=1e-5,
        atol=1e-5,
    )
    expected_initial_residual = np.linalg.norm(
        apply_poisson(second_initial) - second_rhs
    )
    assert cg._last_solve_info.initial_residual_norm == pytest.approx(
        expected_initial_residual, rel=2e-5, abs=2e-5
    )

    stats = cg._debug_runtime_stats()
    assert stats["operations"]["solve_calls"] == 2
    assert stats["identity"]["operator_pattern_version"] == 1
    assert stats["identity"]["operator_numeric_version"] == 1
    if stats["identity"]["backend_family"] == "cuda":
        assert stats["operations"]["workspace_builds"] == 1
        assert stats["operations"]["workspace_reuses"] == 1
    else:
        assert stats["operations"]["workspace_builds"] == 1
        assert stats["operations"]["workspace_reuses"] == 1
