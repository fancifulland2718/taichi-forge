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
    cg = ti.linalg.SparseCG(
        A,
        b,
        x0,
        max_iter=50,
        atol=atol,
        preconditioner="jacobi",
    )
    assert cg.cg_solver.get_status() == -1
    x, exit_code = cg.solve()
    res = np.linalg.solve(A_psd, b.to_numpy())
    assert exit_code == True
    assert cg._last_solve_info.converged
    assert cg._last_solve_result is cg._last_solve_info
    assert cg._last_solve_result.status_code == 2
    assert cg._last_solve_result.termination_reason == "converged"
    assert not cg._last_solve_result.breakdown
    assert not cg._last_solve_result.reached_max_iterations
    assert 0 <= cg._last_solve_info.iterations <= 50
    assert cg._last_solve_info.residual_norm <= atol
    for i in range(n):
        assert x[i] == test_utils.approx(res[i], rel=1.0)


@test_utils.test(arch=[ti.cpu])
def test_cpu_operator_action_rejects_legacy_eigen_provider_at_construction():
    builder = ti.linalg.SparseMatrixBuilder(
        2, 2, max_num_triplets=2, dtype=ti.f32
    )

    @ti.kernel
    def fill(matrix: ti.types.sparse_matrix_builder()):
        for i in range(2):
            matrix[i, i] += i + 2

    fill(builder)
    matrix = builder.build()
    program = ti.lang.impl.get_runtime().prog
    with pytest.raises(
        RuntimeError,
        match="does not support backend 'cpu' with storage format 'cs[rc]'",
    ):
        ti._lib.core._make_cpu_operator_cg_solver(
            program, matrix.matrix, 8, 1e-6, 0.0
        )


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
    assert cg.cg_solver.get_status() == -1
    x, exit_code = cg.solve()
    res = np.linalg.solve(A_psd, b.to_numpy())
    assert exit_code == True
    assert cg._last_solve_info.converged
    assert cg._last_solve_result.status_code == 2
    assert cg._last_solve_result.termination_reason == "converged"
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
    assert cg._last_solve_result.status_code == 0
    assert cg._last_solve_result.termination_reason == "max_iterations"
    assert not cg._last_solve_result.breakdown
    assert cg._last_solve_result.reached_max_iterations
    assert cg._last_solve_info.iterations == 0
    assert cg._last_solve_info.residual_norm == pytest.approx(cg._last_solve_info.initial_residual_norm)


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_cg_reports_nonfinite_breakdown():
    n = 4
    builder = ti.linalg.SparseMatrixBuilder(
        n, n, max_num_triplets=n, dtype=ti.f32
    )
    b = ti.ndarray(dtype=ti.f32, shape=n)
    x0 = ti.ndarray(dtype=ti.f32, shape=n)

    @ti.kernel
    def fill(builder: ti.types.sparse_matrix_builder()):
        for i in range(n):
            builder[i, i] += i + 2

    fill(builder)
    b.from_numpy(np.asarray([np.nan, 1.0, 2.0, 3.0], dtype=np.float32))
    x0.fill(0)
    matrix = builder.build()
    cg = ti.linalg.SparseCG(matrix, b, x0, max_iter=4, atol=1e-6)
    _, converged = cg.solve()

    assert not converged
    assert cg._last_solve_result.status_code == 1
    assert cg._last_solve_result.termination_reason == "breakdown"
    assert cg._last_solve_result.breakdown
    assert not cg._last_solve_result.reached_max_iterations
    assert not cg._last_solve_result.converged


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
    assert before["identity"]["preconditioner_selection"] == "legacy"
    assert before["preconditioner"] is None
    assert before["operations"]["preconditioner_auto_refresh_attempts"] == 0
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
        assert first["identity"]["preconditioner_method"] == "jacobi"
        assert first["operations"]["preconditioner_apply_calls"] is None
        assert first["resources"]["preconditioner_ownership_scope"] == (
            "provider_state"
        )
        assert not first["resources"]["external_preconditioner"]
        assert first["resources"]["persistent_vector_count"] == 2
        assert first["resources"]["persistent_vector_reserved_bytes"] == 2 * n * 4
        assert not first["resources"]["solver_state_rebuilt_each_solve"]
        assert first["resources"]["transient_solver_workspace_bytes"] is None
        assert first["operations"]["operator_apply_calls"] is None
        assert first["operations"]["workspace_builds"] == 1
        assert first["operations"]["workspace_reuses"] == 0
    else:
        assert first["identity"]["preconditioner_method"] == "identity"
        assert first["operations"]["preconditioner_apply_calls"] == 0
        assert first["resources"]["preconditioner_ownership_scope"] == (
            "none"
        )
        assert not first["resources"]["external_preconditioner"]
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


@pytest.mark.parametrize("preconditioner", [None, "jacobi"])
@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_sparse_cg_relative_tolerance_tracks_each_rhs_scale(
    preconditioner,
):
    n = 4
    builder = ti.linalg.SparseMatrixBuilder(
        n,
        n,
        max_num_triplets=n,
        dtype=ti.f32,
        storage_format="row_major",
    )

    @ti.kernel
    def assemble(matrix: ti.types.sparse_matrix_builder()):
        for i in range(n):
            matrix[i, i] += 1.0

    assemble(builder)
    matrix = builder.build()
    rhs = ti.ndarray(dtype=ti.f32, shape=n)
    initial = ti.ndarray(dtype=ti.f32, shape=n)
    base = np.linspace(1.0, 2.5, n, dtype=np.float32)

    def load(scale):
        rhs_host = base * scale
        initial_host = rhs_host * 0.95
        rhs.from_numpy(rhs_host)
        initial.from_numpy(initial_host)
        return rhs_host, initial_host

    first_rhs, first_initial = load(1e-3)
    cg = ti.linalg.SparseCG(
        matrix,
        rhs,
        initial,
        max_iter=0,
        atol=0.0,
        rtol=0.1,
        preconditioner=preconditioner,
    )
    first_solution, first_converged = cg.solve()
    assert first_converged
    first = cg._last_solve_result
    first_rhs_norm = np.linalg.norm(first_rhs)
    assert first.absolute_tolerance == 0.0
    assert first.relative_tolerance == pytest.approx(0.1)
    assert first.relative_reference_norm == pytest.approx(
        first_rhs_norm, rel=2e-5, abs=1e-8
    )
    assert first.effective_tolerance == pytest.approx(
        0.1 * first_rhs_norm, rel=2e-5, abs=1e-8
    )
    assert first.initial_residual_norm <= first.effective_tolerance
    np.testing.assert_allclose(
        first_solution.to_numpy()
        if hasattr(first_solution, "to_numpy")
        else first_solution,
        first_initial,
        rtol=0,
        atol=1e-8,
    )

    second_rhs, second_initial = load(1e3)
    second_solution, second_converged = cg.solve()
    assert second_converged
    second = cg._last_solve_result
    second_rhs_norm = np.linalg.norm(second_rhs)
    assert second.relative_reference_norm == pytest.approx(
        second_rhs_norm, rel=2e-5
    )
    assert second.effective_tolerance == pytest.approx(
        0.1 * second_rhs_norm, rel=2e-5
    )
    assert second.initial_residual_norm <= second.effective_tolerance
    np.testing.assert_allclose(
        second_solution.to_numpy()
        if hasattr(second_solution, "to_numpy")
        else second_solution,
        second_initial,
        rtol=0,
        atol=1e-3,
    )
    stats = cg._debug_runtime_stats()
    assert stats["identity"]["absolute_tolerance"] == 0.0
    assert stats["identity"]["relative_tolerance"] == pytest.approx(0.1)
    assert stats["identity"][
        "last_relative_reference_norm"
    ] == pytest.approx(second_rhs_norm, rel=2e-5)
    assert stats["identity"]["last_effective_tolerance"] == pytest.approx(
        0.1 * second_rhs_norm, rel=2e-5
    )
    if stats["identity"]["backend_family"] == "cuda":
        assert stats["operations"]["host_scalar_reductions"] == 4


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_sparse_cg_relative_only_zero_rhs_requires_exact_residual():
    n = 2
    builder = ti.linalg.SparseMatrixBuilder(
        n, n, max_num_triplets=n, dtype=ti.f32
    )

    @ti.kernel
    def assemble(matrix: ti.types.sparse_matrix_builder()):
        for i in range(n):
            matrix[i, i] += 1.0

    assemble(builder)
    matrix = builder.build()
    rhs = ti.ndarray(dtype=ti.f32, shape=n)
    initial = ti.ndarray(dtype=ti.f32, shape=n)
    rhs.fill(0)
    initial.fill(1)
    cg = ti.linalg.SparseCG(
        matrix, rhs, initial, max_iter=4, atol=0.0, rtol=0.1
    )
    solution, converged = cg.solve()
    assert converged
    result = cg._last_solve_result
    assert result.relative_reference_norm == 0.0
    assert result.effective_tolerance == 0.0
    assert result.residual_norm == 0.0
    np.testing.assert_array_equal(
        solution.to_numpy() if hasattr(solution, "to_numpy") else solution,
        np.zeros(n, dtype=np.float32),
    )


@test_utils.test(arch=[ti.cpu], offline_cache=False)
def test_sparse_cg_rejects_invalid_solver_controls():
    builder = ti.linalg.SparseMatrixBuilder(
        1, 1, max_num_triplets=1, dtype=ti.f32
    )

    @ti.kernel
    def assemble(matrix: ti.types.sparse_matrix_builder()):
        matrix[0, 0] += 1.0

    assemble(builder)
    matrix = builder.build()
    rhs = np.ones(1, dtype=np.float32)
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
            ti.linalg.SparseCG(matrix, rhs, **kwargs)


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_public_sparse_cg_explicit_jacobi_refreshes_value_updates():
    n = 6
    diagonal = np.linspace(2.0, 5.0, n, dtype=np.float32)
    builder = ti.linalg.SparseMatrixBuilder(
        n,
        n,
        max_num_triplets=n,
        dtype=ti.f32,
        storage_format="row_major",
    )

    @ti.kernel
    def assemble(matrix: ti.types.sparse_matrix_builder()):
        for i in range(n):
            matrix[i, i] += 2.0 + 0.6 * i

    assemble(builder)
    matrix = builder.build()
    expected = np.linspace(-0.75, 1.25, n, dtype=np.float32)
    rhs_host = diagonal * expected
    rhs = ti.ndarray(dtype=ti.f32, shape=n)
    initial = ti.ndarray(dtype=ti.f32, shape=n)
    rhs.from_numpy(rhs_host)
    initial.fill(0)
    cg = ti.linalg.SparseCG(
        matrix,
        rhs,
        initial,
        max_iter=16,
        atol=1e-5,
        preconditioner="JaCoBi",
    )

    first, first_ok = cg.solve()
    assert first_ok
    first_host = first.to_numpy() if hasattr(first, "to_numpy") else first
    np.testing.assert_allclose(first_host, expected, rtol=2e-5, atol=2e-5)
    first_stats = cg._debug_runtime_stats()
    assert first_stats["identity"]["preconditioner_selection"] == "jacobi"
    assert first_stats["identity"]["preconditioner_method"] == "jacobi"
    assert first_stats["operations"]["preconditioner_auto_refresh_attempts"] == 0

    updated_values = ti.ndarray(dtype=ti.f32, shape=n)
    updated_values.from_numpy(diagonal * 2.0)
    matrix._update_values(updated_values)
    second, second_ok = cg.solve()
    assert second_ok
    second_host = second.to_numpy() if hasattr(second, "to_numpy") else second
    np.testing.assert_allclose(
        second_host, expected * 0.5, rtol=2e-5, atol=2e-5
    )
    second_stats = cg._debug_runtime_stats()
    assert second_stats["operations"]["solve_calls"] == 2
    assert second_stats["identity"]["operator_numeric_version"] == 2
    if second_stats["identity"]["backend_family"] == "cpu":
        assert second_stats["preconditioner"] is None
        assert (
            second_stats["resources"]["preconditioner_ownership_scope"]
            == "provider_state"
        )
        assert second_stats["operations"][
            "preconditioner_auto_refresh_attempts"
        ] == 0
        assert second_stats["operations"]["workspace_builds"] == 2
    else:
        assert second_stats["identity"]["method"] == "pcg_jacobi"
        assert second_stats["operations"][
            "preconditioner_auto_refresh_attempts"
        ] == 1
        assert second_stats["operations"][
            "preconditioner_auto_refresh_successes"
        ] == 1
        assert second_stats["operations"]["workspace_builds"] == 1
        assert second_stats["operations"]["workspace_reuses"] == 1
        assert second_stats["preconditioner"]["schema_version"] == 2
        assert second_stats["preconditioner"]["operations"][
            "numeric_refresh_successes"
        ] == 1
        assert not second_stats["preconditioner"]["identity"][
            "operator_stale"
        ]

        invalid_diagonal = diagonal * 2.0
        invalid_diagonal[2] = 0.0
        invalid_values = ti.ndarray(dtype=ti.f32, shape=n)
        invalid_values.from_numpy(invalid_diagonal)
        matrix._update_values(invalid_values)
        sentinel = np.linspace(3.0, 4.0, n, dtype=np.float32)
        initial.from_numpy(sentinel)
        with pytest.raises(RuntimeError, match="diagonal at row 2 is zero"):
            cg.solve()
        np.testing.assert_array_equal(initial.to_numpy(), sentinel)
        failed_stats = cg._debug_runtime_stats()
        assert failed_stats["operations"]["solve_calls"] == 2
        assert failed_stats["operations"][
            "preconditioner_auto_refresh_attempts"
        ] == 2
        assert failed_stats["operations"][
            "preconditioner_auto_refresh_successes"
        ] == 1
        assert failed_stats["preconditioner"]["operations"][
            "numeric_refresh_failures"
        ] == 1
        assert failed_stats["preconditioner"]["identity"][
            "operator_stale"
        ]


@test_utils.test(arch=[ti.cpu], offline_cache=False)
def test_public_sparse_cg_rejects_unsupported_preconditioner_selection():
    builder = ti.linalg.SparseMatrixBuilder(
        1, 1, max_num_triplets=1, dtype=ti.f32, storage_format="row_major"
    )

    @ti.kernel
    def assemble(matrix: ti.types.sparse_matrix_builder()):
        matrix[0, 0] += 1.0

    assemble(builder)
    matrix = builder.build()
    rhs = np.ones(1, dtype=np.float32)
    with pytest.raises(
        RuntimeError, match="None, 'jacobi', or 'block_jacobi'"
    ):
        ti.linalg.SparseCG(matrix, rhs, preconditioner="identity")
    with pytest.raises(
        RuntimeError,
        match="operation 'public_block_jacobi_selection'.*cpu csr",
    ):
        ti.linalg.SparseCG(matrix, rhs, preconditioner="block_jacobi")
