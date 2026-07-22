import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_fixed_csr_jacobi_plan_apply_reuse_and_stale_version():
    n = 4
    diagonal = np.asarray([2.0, 4.0, 8.0, 16.0], dtype=np.float32)
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
            matrix[i, i] += 2 ** (i + 1)

    assemble(builder)
    matrix = builder.build()
    prog = ti.lang.impl.get_runtime().prog
    plan = ti._lib.core._make_sparse_jacobi_preconditioner_plan(
        prog, matrix.matrix
    )
    rhs_host = np.asarray([1.0, -2.0, 6.0, 8.0], dtype=np.float32)
    rhs = ti.ndarray(dtype=ti.f32, shape=n)
    output = ti.ndarray(dtype=ti.f32, shape=n)
    rhs.from_numpy(rhs_host)

    plan.apply(prog, rhs.arr, output.arr)
    np.testing.assert_allclose(output.to_numpy(), rhs_host / diagonal)

    rhs.from_numpy(rhs_host)
    plan.apply(prog, rhs.arr, rhs.arr)
    np.testing.assert_allclose(rhs.to_numpy(), rhs_host / diagonal)

    stats = plan._debug_runtime_stats()
    assert stats["schema_version"] == 2
    assert stats["identity"]["method"] == "jacobi"
    assert stats["identity"]["dtype"] == "f32"
    assert stats["identity"]["rows"] == n
    assert not stats["identity"]["operator_stale"]
    assert stats["operations"]["apply_calls"] == 2
    assert stats["resources"]["persistent_inverse_count"] == 1
    assert stats["resources"]["persistent_inverse_reserved_bytes"] == n * 4
    assert stats["transfers"]["apply_host_transfer_bytes"] == 0
    assert stats["contract"]["in_place_apply_supported"]
    assert stats["contract"]["numeric_refresh_supported"]
    assert stats["contract"]["numeric_update_requires_refresh"]
    assert not stats["contract"]["numeric_update_requires_rebuild"]
    assert stats["contract"]["pattern_update_requires_rebuild"]
    if stats["identity"]["backend_family"] == "cpu":
        assert stats["transfers"]["construction_device_to_host_bytes"] == 0
        assert stats["transfers"]["construction_host_to_device_bytes"] == 0
        assert stats["transfers"]["construction_host_synchronizations"] == 0
    else:
        expected_readback = ((n + 1) + n + n) * 4
        assert (
            stats["transfers"]["construction_device_to_host_bytes"]
            == expected_readback
        )
        assert stats["transfers"]["construction_host_to_device_bytes"] == n * 4
        expected_syncs = (
            1 if stats["identity"]["backend_family"] == "vulkan" else 0
        )
        assert (
            stats["transfers"]["construction_host_synchronizations"]
            == expected_syncs
        )

    updated_diagonal = diagonal * 2.0
    updated_values = ti.ndarray(dtype=ti.f32, shape=n)
    updated_values.from_numpy(updated_diagonal)
    matrix._update_values(updated_values)
    stale = plan._debug_runtime_stats()
    assert stale["identity"]["operator_stale"]
    with pytest.raises(RuntimeError, match="plan is stale"):
        plan.apply(prog, output.arr, rhs.arr)
    assert plan._debug_runtime_stats()["operations"]["apply_calls"] == 2

    plan._refresh_numeric(prog)
    refreshed = plan._debug_runtime_stats()
    assert not refreshed["identity"]["operator_stale"]
    assert refreshed["operations"]["numeric_refresh_calls"] == 1
    assert refreshed["operations"]["numeric_refresh_successes"] == 1
    assert refreshed["operations"]["numeric_refresh_noops"] == 0
    assert refreshed["operations"]["numeric_refresh_failures"] == 0
    backend = refreshed["identity"]["backend_family"]
    if backend == "cpu":
        assert (
            refreshed["resources"]["refresh_peak_temporary_host_bytes"]
            == n * 4
        )
        assert (
            refreshed["resources"]["refresh_peak_temporary_device_bytes"]
            == 0
        )
        assert refreshed["transfers"]["refresh_device_to_host_bytes"] == 0
        assert refreshed["transfers"]["refresh_host_to_device_bytes"] == 0
        assert refreshed["transfers"]["refresh_host_synchronizations"] == 0
    else:
        assert (
            refreshed["transfers"]["refresh_device_to_host_bytes"]
            == n * 4
        )
        assert (
            refreshed["transfers"]["refresh_host_to_device_bytes"]
            == n * 4
        )
        assert refreshed["transfers"]["refresh_host_synchronizations"] == (
            1 if backend == "vulkan" else 0
        )
        assert (
            refreshed["resources"]["refresh_peak_temporary_host_bytes"]
            == 2 * n * 4
        )
        assert (
            refreshed["resources"]["refresh_peak_temporary_device_bytes"]
            == n * 4
        )
    rhs.from_numpy(rhs_host)
    plan.apply(prog, rhs.arr, output.arr)
    np.testing.assert_allclose(output.to_numpy(), rhs_host / updated_diagonal)

    plan._refresh_numeric(prog)
    noop = plan._debug_runtime_stats()
    assert noop["operations"]["numeric_refresh_calls"] == 2
    assert noop["operations"]["numeric_refresh_successes"] == 1
    assert noop["operations"]["numeric_refresh_noops"] == 1
    assert noop["transfers"] == refreshed["transfers"]

    invalid_diagonal = updated_diagonal.copy()
    invalid_diagonal[1] = 0.0
    invalid_values = ti.ndarray(dtype=ti.f32, shape=n)
    invalid_values.from_numpy(invalid_diagonal)
    matrix._update_values(invalid_values)
    with pytest.raises(RuntimeError, match="diagonal at row 1 is zero"):
        plan._refresh_numeric(prog)
    failed = plan._debug_runtime_stats()
    assert failed["identity"]["operator_stale"]
    assert failed["operations"]["numeric_refresh_calls"] == 3
    assert failed["operations"]["numeric_refresh_successes"] == 1
    assert failed["operations"]["numeric_refresh_noops"] == 1
    assert failed["operations"]["numeric_refresh_failures"] == 1
    if backend != "cpu":
        assert (
            failed["transfers"]["refresh_device_to_host_bytes"]
            == 2 * n * 4
        )
        assert (
            failed["transfers"]["refresh_host_to_device_bytes"]
            == n * 4
        )
        assert failed["transfers"]["refresh_host_synchronizations"] == (
            2 if backend == "vulkan" else 0
        )
    sentinel = np.linspace(-3.0, 3.0, n, dtype=np.float32)
    output.from_numpy(sentinel)
    with pytest.raises(RuntimeError, match="plan is stale"):
        plan.apply(prog, rhs.arr, output.arr)
    np.testing.assert_array_equal(output.to_numpy(), sentinel)

    recovered_diagonal = diagonal * 4.0
    recovered_values = ti.ndarray(dtype=ti.f32, shape=n)
    recovered_values.from_numpy(recovered_diagonal)
    matrix._update_values(recovered_values)
    plan._refresh_numeric(prog)
    recovered = plan._debug_runtime_stats()
    assert not recovered["identity"]["operator_stale"]
    assert recovered["operations"]["numeric_refresh_calls"] == 4
    assert recovered["operations"]["numeric_refresh_successes"] == 2
    assert recovered["operations"]["numeric_refresh_failures"] == 1
    rhs.from_numpy(rhs_host)
    plan.apply(prog, rhs.arr, output.arr)
    np.testing.assert_allclose(
        output.to_numpy(), rhs_host / recovered_diagonal
    )


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_fixed_csr_jacobi_plan_rejects_missing_diagonal_transactionally():
    n = 3
    builder = ti.linalg.SparseMatrixBuilder(
        n,
        n,
        max_num_triplets=n,
        dtype=ti.f32,
        storage_format="row_major",
    )

    @ti.kernel
    def assemble(matrix: ti.types.sparse_matrix_builder()):
        matrix[0, 0] += 2.0
        matrix[1, 0] += -1.0
        matrix[2, 2] += 3.0

    assemble(builder)
    matrix = builder.build()
    before = matrix._debug_runtime_stats()
    prog = ti.lang.impl.get_runtime().prog
    with pytest.raises(RuntimeError, match="row 1 has no stored diagonal"):
        ti._lib.core._make_sparse_jacobi_preconditioner_plan(
            prog, matrix.matrix
        )
    after = matrix._debug_runtime_stats()
    assert after["identity"]["pattern_version"] == before["identity"][
        "pattern_version"
    ]
    assert after["identity"]["numeric_version"] == before["identity"][
        "numeric_version"
    ]
    assert after["resources"]["operator_owned_reserved_bytes"] == before[
        "resources"
    ]["operator_owned_reserved_bytes"]


@pytest.mark.parametrize(
    "ti_dtype,np_dtype,dtype_name,item_size",
    [
        (ti.f32, np.float32, "f32", 4),
        (ti.f64, np.float64, "f64", 8),
    ],
)
@test_utils.test(arch=[ti.cpu], offline_cache=False)
def test_shared_cpu_csr_jacobi_plan_supports_dtype_reuse_and_refresh(
    ti_dtype, np_dtype, dtype_name, item_size
):
    n = 4
    row_offsets = ti.ndarray(dtype=ti.i32, shape=n + 1)
    column_indices = ti.ndarray(dtype=ti.i32, shape=10)
    row_offsets.from_numpy(np.asarray([0, 2, 5, 8, 10], dtype=np.int32))
    column_indices.from_numpy(
        np.asarray([0, 1, 0, 1, 2, 1, 2, 3, 2, 3], dtype=np.int32)
    )
    pattern = ti.linalg.SparsePattern.csr(
        n, n, row_offsets, column_indices
    )
    values_host = np.asarray(
        [2.0, -1.0, -1.0, 4.0, -1.0, -1.0, 8.0, -1.0, -1.0, 16.0],
        dtype=np_dtype,
    )
    diagonal = values_host[[0, 3, 6, 9]]
    values = ti.ndarray(dtype=ti_dtype, shape=10)
    values.from_numpy(values_host)
    matrix = pattern.matrix(values)
    prog = ti.lang.impl.get_runtime().prog
    plan = ti._lib.core._make_sparse_jacobi_preconditioner_plan(
        prog, matrix.matrix
    )
    rhs_host = np.asarray([1.0, -2.0, 6.0, 8.0], dtype=np_dtype)
    rhs = ti.ndarray(dtype=ti_dtype, shape=n)
    output = ti.ndarray(dtype=ti_dtype, shape=n)
    rhs.from_numpy(rhs_host)

    plan.apply(prog, rhs.arr, output.arr)
    np.testing.assert_allclose(output.to_numpy(), rhs_host / diagonal)
    rhs.from_numpy(rhs_host)
    plan.apply(prog, rhs.arr, rhs.arr)
    np.testing.assert_allclose(rhs.to_numpy(), rhs_host / diagonal)

    initial = plan._debug_runtime_stats()
    assert initial["identity"]["backend_family"] == "cpu"
    assert initial["identity"]["dtype"] == dtype_name
    assert initial["operations"]["apply_calls"] == 2
    assert initial["resources"]["persistent_inverse_reserved_bytes"] == (
        n * item_size
    )
    assert initial["transfers"]["construction_device_to_host_bytes"] == 0
    assert initial["transfers"]["construction_host_to_device_bytes"] == 0

    updated_values = ti.ndarray(dtype=ti_dtype, shape=10)
    updated_values.from_numpy(2 * values_host)
    matrix.update_values(updated_values)
    with pytest.raises(RuntimeError, match="plan is stale"):
        plan.apply(prog, rhs.arr, output.arr)
    plan._refresh_numeric(prog)
    rhs.from_numpy(rhs_host)
    plan.apply(prog, rhs.arr, output.arr)
    np.testing.assert_allclose(output.to_numpy(), rhs_host / (2 * diagonal))

    refreshed = plan._debug_runtime_stats()
    assert refreshed["operations"]["numeric_refresh_successes"] == 1
    assert refreshed["operations"]["apply_calls"] == 3
    assert refreshed["resources"]["refresh_peak_temporary_host_bytes"] == (
        n * item_size
    )
    assert refreshed["resources"]["refresh_peak_temporary_device_bytes"] == 0
    assert refreshed["transfers"]["refresh_device_to_host_bytes"] == 0
    assert refreshed["transfers"]["refresh_host_to_device_bytes"] == 0


@pytest.mark.parametrize(
    "ti_dtype,np_dtype,tolerance,rtol",
    [
        (ti.f32, np.float32, 1e-5, 2e-4),
        (ti.f64, np.float64, 1e-10, 2e-9),
    ],
)
@test_utils.test(arch=[ti.cpu], offline_cache=False)
def test_shared_cpu_csr_jacobi_pcg_reuses_workspace_and_numeric_refresh(
    ti_dtype, np_dtype, tolerance, rtol
):
    n = 8
    row_offsets_host = [0]
    column_indices_host = []
    values_host = []
    dense = np.zeros((n, n), dtype=np_dtype)
    for row in range(n):
        if row > 0:
            column_indices_host.append(row - 1)
            values_host.append(-1.0)
            dense[row, row - 1] = -1.0
        diagonal = 4.0 + 0.125 * row
        column_indices_host.append(row)
        values_host.append(diagonal)
        dense[row, row] = diagonal
        if row + 1 < n:
            column_indices_host.append(row + 1)
            values_host.append(-1.0)
            dense[row, row + 1] = -1.0
        row_offsets_host.append(len(column_indices_host))
    values_host = np.asarray(values_host, dtype=np_dtype)

    row_offsets = ti.ndarray(dtype=ti.i32, shape=n + 1)
    column_indices = ti.ndarray(dtype=ti.i32, shape=len(column_indices_host))
    values = ti.ndarray(dtype=ti_dtype, shape=values_host.size)
    row_offsets.from_numpy(np.asarray(row_offsets_host, dtype=np.int32))
    column_indices.from_numpy(
        np.asarray(column_indices_host, dtype=np.int32)
    )
    values.from_numpy(values_host)
    pattern = ti.linalg.SparsePattern.csr(
        n, n, row_offsets, column_indices
    )
    matrix = pattern.matrix(values)
    prog = ti.lang.impl.get_runtime().prog
    preconditioner = (
        ti._lib.core._make_sparse_jacobi_preconditioner_plan(
            prog, matrix.matrix
        )
    )
    max_iterations = 64
    solver = ti._lib.core._make_cpu_jacobi_pcg_solver(
        prog,
        matrix.matrix,
        preconditioner,
        max_iterations,
        tolerance,
    )
    rhs = ti.ndarray(dtype=ti_dtype, shape=n)
    solution = ti.ndarray(dtype=ti_dtype, shape=n)

    exact_solutions = [
        np.linspace(-0.75, 1.0, n, dtype=np_dtype),
        np.cos(np.linspace(0.2, 1.4, n, dtype=np_dtype)).astype(np_dtype),
    ]
    for exact in exact_solutions:
        rhs.from_numpy(dense @ exact)
        solution.from_numpy(np.zeros(n, dtype=np_dtype))
        solver.solve(prog, solution.arr, rhs.arr)
        assert solver.get_status() == 2
        assert 0 < solver.get_iterations() <= max_iterations
        np.testing.assert_allclose(
            solution.to_numpy(), exact, rtol=rtol, atol=rtol
        )

    stats = solver._debug_runtime_stats()
    total_iterations = stats["operations"]["total_iterations"]
    assert stats["identity"]["method"] == "pcg_jacobi"
    assert stats["identity"]["preconditioner_method"] == "jacobi"
    assert stats["operations"]["solve_calls"] == 2
    assert stats["operations"]["workspace_builds"] == 1
    assert stats["operations"]["workspace_reuses"] == 1
    assert stats["identity"]["operator_action_provider"] == "forge_cpu_native"
    assert stats["identity"]["preconditioner_action_provider"] == "cpu_jacobi"
    assert not stats["identity"]["operator_asynchronous_submit"]
    assert not stats["identity"]["preconditioner_asynchronous_submit"]
    assert stats["identity"]["preconditioner_behavior"] == "fixed_linear"
    assert stats["operations"]["operator_generation_pins"] == 3
    assert stats["operations"]["preconditioner_generation_pins"] == 3
    assert stats["operations"]["preconditioner_setup_calls"] == 1
    assert stats["operations"]["preconditioner_update_calls"] == 2
    assert stats["operations"]["preconditioner_update_noops"] == 2
    assert stats["operations"]["operator_apply_calls"] == (
        2 + total_iterations
    )
    assert stats["operations"]["preconditioner_apply_calls"] == (
        total_iterations
    )
    assert stats["operations"]["host_scalar_readbacks"] == 0
    assert stats["operations"]["host_synchronizations"] == 0
    assert stats["resources"]["persistent_vector_count"] == 4
    assert stats["resources"]["persistent_vector_reserved_bytes"] == (
        4 * n * np.dtype(np_dtype).itemsize
    )

    scaled_values = ti.ndarray(dtype=ti_dtype, shape=values_host.size)
    scaled_values.from_numpy(values_host * np_dtype(1.5))
    matrix.update_values(scaled_values)
    sentinel = np.full(n, -7.0, dtype=np_dtype)
    solution.from_numpy(sentinel)
    with pytest.raises(RuntimeError, match="plan is stale"):
        solver.solve(prog, solution.arr, rhs.arr)
    np.testing.assert_array_equal(solution.to_numpy(), sentinel)
    assert solver._debug_runtime_stats()["operations"]["solve_calls"] == 2

    preconditioner._refresh_numeric(prog)
    exact = exact_solutions[0]
    rhs.from_numpy((dense * np_dtype(1.5)) @ exact)
    solution.from_numpy(np.zeros(n, dtype=np_dtype))
    solver.solve(prog, solution.arr, rhs.arr)
    assert solver.get_status() == 2
    np.testing.assert_allclose(
        solution.to_numpy(), exact, rtol=rtol, atol=rtol
    )
    refreshed = solver._debug_runtime_stats()
    assert refreshed["operations"]["solve_calls"] == 3
    assert refreshed["operations"]["workspace_builds"] == 1
    assert refreshed["operations"]["workspace_reuses"] == 2
    assert refreshed["operations"]["operator_generation_pins"] == 5
    assert refreshed["operations"]["preconditioner_generation_pins"] == 4
    assert refreshed["operations"]["preconditioner_setup_calls"] == 1
    assert refreshed["operations"]["preconditioner_update_calls"] == 4
    assert refreshed["operations"]["preconditioner_update_successes"] == 1
    assert refreshed["operations"]["preconditioner_update_noops"] == 2
    assert refreshed["operations"]["preconditioner_update_failures"] == 1
    assert refreshed["operations"]["operator_generation_changes"] == 1
    assert (
        refreshed["operations"]["preconditioner_generation_changes"] == 1
    )
    assert refreshed["operations"]["operator_plan_invalidations"] == 0
    assert refreshed["operations"]["preconditioner_plan_invalidations"] == 0
    assert refreshed["operations"]["preconditioner_apply_calls"] == (
        preconditioner._debug_runtime_stats()["operations"]["apply_calls"]
    )


@test_utils.test(arch=[ti.cpu], offline_cache=False)
def test_fixed_csr_jacobi_plan_rejects_unsupported_layout():
    n = 2
    prog = ti.lang.impl.get_runtime().prog

    csc_builder = ti.linalg.SparseMatrixBuilder(
        n, n, max_num_triplets=n, dtype=ti.f32, storage_format="col_major"
    )

    @ti.kernel
    def assemble_csc(matrix: ti.types.sparse_matrix_builder()):
        for i in range(n):
            matrix[i, i] += i + 2

    assemble_csc(csc_builder)
    csc_matrix = csc_builder.build()
    with pytest.raises(RuntimeError, match="fixed CSR storage"):
        ti._lib.core._make_sparse_jacobi_preconditioner_plan(
            prog, csc_matrix.matrix
        )


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_device_fixed_csr_jacobi_pcg_composition_and_stale_version():
    n = 8
    max_iterations = 16
    tolerance = 1e-4
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
            matrix[i, i] += 2.5 + 0.05 * i
            if i > 0:
                matrix[i, i - 1] += -1.0
            if i + 1 < n:
                matrix[i, i + 1] += -1.0

    assemble(builder)
    matrix = builder.build()
    exact = np.sin(
        np.linspace(0.15, 2.0, n, dtype=np.float32)
    ).astype(np.float32)
    diagonal = 2.5 + 0.05 * np.arange(n, dtype=np.float32)
    rhs_host = diagonal * exact
    rhs_host[1:] -= exact[:-1]
    rhs_host[:-1] -= exact[1:]
    rhs = ti.ndarray(dtype=ti.f32, shape=n)
    identity_solution = ti.ndarray(dtype=ti.f32, shape=n)
    pcg_solution = ti.ndarray(dtype=ti.f32, shape=n)
    rhs.from_numpy(rhs_host)
    identity_solution.from_numpy(np.zeros(n, dtype=np.float32))
    pcg_solution.from_numpy(np.zeros(n, dtype=np.float32))
    prog = ti.lang.impl.get_runtime().prog
    preconditioner = (
        ti._lib.core._make_sparse_jacobi_preconditioner_plan(
            prog, matrix.matrix
        )
    )
    is_cuda = prog.config().arch == ti._lib.core.Arch.cuda
    if is_cuda:
        identity = ti._lib.core.make_cucg_solver(
            matrix.matrix, max_iterations, tolerance, False
        )
        pcg = ti._lib.core._make_cuda_jacobi_pcg_solver(
            prog,
            matrix.matrix,
            preconditioner,
            max_iterations,
            tolerance,
            False,
        )
    else:
        identity = ti._lib.core._make_vulkan_cg_convergence_plan(
            prog, matrix.matrix, max_iterations, tolerance
        )
        pcg = ti._lib.core._make_vulkan_jacobi_pcg_convergence_plan(
            prog,
            matrix.matrix,
            preconditioner,
            max_iterations,
            tolerance,
        )

    identity.solve(prog, identity_solution.arr, rhs.arr)
    pcg.solve(prog, pcg_solution.arr, rhs.arr)


    assert identity.get_status() == 2
    assert pcg.get_status() == 2
    assert dict(identity._get_last_result())["termination_reason"] == (
        "converged"
    )
    assert dict(pcg._get_last_result())["termination_reason"] == (
        "converged"
    )
    np.testing.assert_allclose(
        identity_solution.to_numpy(), exact, rtol=2e-3, atol=2e-3
    )
    np.testing.assert_allclose(
        pcg_solution.to_numpy(), exact, rtol=2e-3, atol=2e-3
    )

    identity_stats = identity._debug_runtime_stats()
    assert identity_stats["identity"]["preconditioner_method"] == (
        "identity"
    )
    assert identity_stats["operations"]["preconditioner_apply_calls"] == 0
    assert not identity_stats["resources"]["external_preconditioner"]
    assert identity_stats["resources"]["persistent_vector_count"] == 3
    if is_cuda:
        assert identity_stats["operations"]["host_scalar_reductions"] == (
            1 + 2 * identity.get_iterations()
        )

    pcg_stats = pcg._debug_runtime_stats()
    assert pcg_stats["identity"]["preconditioner_method"] == "jacobi"
    if is_cuda:
        assert pcg_stats["identity"]["method"] == "pcg_jacobi"
        expected_apply_calls = pcg.get_iterations()
    else:
        assert pcg_stats["identity"]["method"] == (
            "pcg_jacobi_bounded_masked_probe"
        )
        expected_apply_calls = max_iterations + 1
    assert (
        pcg_stats["operations"]["preconditioner_apply_calls"]
        == expected_apply_calls
    )
    if is_cuda:
        assert pcg_stats["operations"]["host_synchronizations"] == 0
        assert pcg_stats["operations"]["host_scalar_readbacks"] == 0
        assert pcg_stats["operations"]["device_scalar_operations"] == 0
        assert pcg_stats["operations"]["host_scalar_reductions"] == (
            1 + 2 * pcg.get_iterations() + expected_apply_calls
        )
    else:
        assert pcg_stats["operations"]["host_synchronizations"] == 1
        assert pcg_stats["operations"]["host_scalar_readbacks"] == 4
        assert pcg_stats["operations"]["device_scalar_operations"] == 97
    assert pcg_stats["resources"]["external_preconditioner"]
    assert pcg_stats["resources"]["preconditioner_ownership_scope"] == (
        "external_plan"
    )
    assert pcg_stats["resources"]["persistent_vector_count"] == 4
    assert (
        pcg_stats["resources"]["persistent_vector_reserved_bytes"]
        == 4 * n * 4
    )
    preconditioner_stats = preconditioner._debug_runtime_stats()
    assert (
        preconditioner_stats["operations"]["apply_calls"]
        == expected_apply_calls
    )

    if is_cuda:
        rhs.from_numpy(np.zeros(n, dtype=np.float32))
        pcg_solution.from_numpy(np.zeros(n, dtype=np.float32))
        pcg.solve(prog, pcg_solution.arr, rhs.arr)
        assert pcg.get_status() == 2
        assert pcg.get_iterations() == 0
        zero_stats = pcg._debug_runtime_stats()
        assert zero_stats["operations"]["solve_calls"] == 2
        assert zero_stats["operations"]["workspace_builds"] == 1
        assert zero_stats["operations"]["workspace_reuses"] == 1
        assert (
            zero_stats["operations"]["preconditioner_apply_calls"]
            == expected_apply_calls
        )
        assert (
            preconditioner._debug_runtime_stats()["operations"][
                "apply_calls"
            ]
            == expected_apply_calls
        )

    compressed_values = []
    for row in range(n):
        if row > 0:
            compressed_values.append(-1.0)
        compressed_values.append(float(diagonal[row]))
        if row + 1 < n:
            compressed_values.append(-1.0)
    updated_values = ti.ndarray(dtype=ti.f32, shape=3 * n - 2)
    updated_values.from_numpy(
        np.asarray(compressed_values, dtype=np.float32) * 1.25
    )
    matrix._update_values(updated_values)
    rhs.from_numpy(rhs_host)
    sentinel = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    pcg_solution.from_numpy(sentinel)
    with pytest.raises(RuntimeError, match="plan is stale"):
        pcg.solve(prog, pcg_solution.arr, rhs.arr)
    np.testing.assert_array_equal(pcg_solution.to_numpy(), sentinel)
    assert (
        preconditioner._debug_runtime_stats()["operations"]["apply_calls"]
        == expected_apply_calls
    )

    preconditioner._refresh_numeric(prog)
    refreshed_preconditioner = preconditioner._debug_runtime_stats()
    assert not refreshed_preconditioner["identity"]["operator_stale"]
    assert (
        refreshed_preconditioner["operations"]["numeric_refresh_successes"]
        == 1
    )
    pcg_solution.from_numpy(np.zeros(n, dtype=np.float32))
    pcg.solve(prog, pcg_solution.arr, rhs.arr)
    assert pcg.get_status() == 2
    np.testing.assert_allclose(
        pcg_solution.to_numpy(), exact / 1.25, rtol=2e-3, atol=2e-3
    )
    refreshed_pcg = pcg._debug_runtime_stats()
    assert refreshed_pcg["operations"]["workspace_builds"] == 1
    assert refreshed_pcg["operations"]["workspace_reuses"] == (
        2 if is_cuda else 1
    )
    assert (
        refreshed_pcg["operations"]["preconditioner_apply_calls"]
        == preconditioner._debug_runtime_stats()["operations"][
            "apply_calls"
        ]
    )


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_device_native_stored_csr_pcg_chunk_replay():
    n = 16
    max_iterations = 16
    tolerance = 1e-5
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
            matrix[i, i] += 3.0 + 0.02 * i
            if i > 0:
                matrix[i, i - 1] += -1.0
            if i + 1 < n:
                matrix[i, i + 1] += -1.0

    assemble(builder)
    matrix = builder.build()
    exact = np.sin(
        np.linspace(0.1, 2.4, n, dtype=np.float32)
    ).astype(np.float32)
    diagonal = 3.0 + 0.02 * np.arange(n, dtype=np.float32)
    rhs_host = diagonal * exact
    rhs_host[1:] -= exact[:-1]
    rhs_host[:-1] -= exact[1:]
    rhs = ti.ndarray(dtype=ti.f32, shape=n)
    solution = ti.ndarray(dtype=ti.f32, shape=n)
    rhs.from_numpy(rhs_host)
    solution.from_numpy(np.zeros(n, dtype=np.float32))
    prog = ti.lang.impl.get_runtime().prog
    if prog.config().arch == ti._lib.core.Arch.cuda:
        identity_plan = ti._lib.core.make_cucg_solver(
            matrix.matrix, max_iterations, tolerance, False
        )
    else:
        identity_plan = ti._lib.core._make_vulkan_cg_convergence_plan(
            prog, matrix.matrix, max_iterations, tolerance
        )
    identity_plan._configure_execution_policy("host_check_every_k", 4)
    identity_plan.solve(prog, solution.arr, rhs.arr)
    assert identity_plan.get_status() == 2
    identity_first = identity_plan._debug_runtime_stats()
    assert identity_first["identity"]["solver_graph_enabled"]
    assert identity_first["operations"]["solver_chunk_builds"] >= 1
    solution.from_numpy(np.zeros(n, dtype=np.float32))
    identity_plan.solve(prog, solution.arr, rhs.arr)
    identity_second = identity_plan._debug_runtime_stats()
    assert identity_second["operations"]["solver_chunk_builds"] == (
        identity_first["operations"]["solver_chunk_builds"]
    )
    assert identity_second["operations"]["solver_chunk_replays"] > (
        identity_first["operations"]["solver_chunk_replays"]
    )
    np.testing.assert_allclose(
        solution.to_numpy(), exact, rtol=3e-3, atol=3e-3
    )
    solution.from_numpy(np.zeros(n, dtype=np.float32))
    preconditioner = ti._lib.core._make_sparse_jacobi_preconditioner_plan(
        prog, matrix.matrix
    )
    if prog.config().arch == ti._lib.core.Arch.cuda:
        plan = ti._lib.core._make_cuda_jacobi_pcg_solver(
            prog,
            matrix.matrix,
            preconditioner,
            max_iterations,
            tolerance,
            False,
        )
    else:
        plan = ti._lib.core._make_vulkan_jacobi_pcg_convergence_plan(
            prog,
            matrix.matrix,
            preconditioner,
            max_iterations,
            tolerance,
        )
    plan._configure_execution_policy("host_check_every_k", 4)

    plan.solve(prog, solution.arr, rhs.arr)
    assert plan.get_status() == 2
    np.testing.assert_allclose(
        solution.to_numpy(), exact, rtol=3e-3, atol=3e-3
    )
    first = plan._debug_runtime_stats()
    assert first["identity"]["solver_graph_enabled"]
    assert first["identity"]["solver_replay_unavailable_reason"] == "none"
    assert first["operations"]["solver_chunk_builds"] >= 1
    assert first["operations"]["solver_chunk_direct_submissions"] == 0

    solution.from_numpy(np.zeros(n, dtype=np.float32))
    plan.solve(prog, solution.arr, rhs.arr)
    assert plan.get_status() == 2
    np.testing.assert_allclose(
        solution.to_numpy(), exact, rtol=3e-3, atol=3e-3
    )
    second = plan._debug_runtime_stats()
    assert second["operations"]["solver_chunk_builds"] == first[
        "operations"
    ]["solver_chunk_builds"]
    assert second["operations"]["solver_chunk_replays"] > first[
        "operations"
    ]["solver_chunk_replays"]
    assert second["operations"]["solver_chunk_direct_submissions"] == 0
    assert second["operations"]["solver_chunk_invalidations"] == 0

    compressed_values = []
    for row in range(n):
        if row > 0:
            compressed_values.append(-1.0)
        compressed_values.append(float(diagonal[row]))
        if row + 1 < n:
            compressed_values.append(-1.0)
    updated_values = ti.ndarray(dtype=ti.f32, shape=3 * n - 2)
    updated_values.from_numpy(
        np.asarray(compressed_values, dtype=np.float32) * 1.25
    )
    matrix._update_values(updated_values)
    preconditioner._refresh_numeric(prog)
    solution.from_numpy(np.zeros(n, dtype=np.float32))
    plan.solve(prog, solution.arr, rhs.arr)
    np.testing.assert_allclose(
        solution.to_numpy(), exact / 1.25, rtol=3e-3, atol=3e-3
    )
    rebound = plan._debug_runtime_stats()
    assert rebound["operations"]["solver_chunk_builds"] == second[
        "operations"
    ]["solver_chunk_builds"]
    assert rebound["operations"]["solver_chunk_rebinds"] > second[
        "operations"
    ]["solver_chunk_rebinds"]
    assert rebound["operations"]["solver_chunk_invalidations"] == 0

    replacement_solution = ti.ndarray(dtype=ti.f32, shape=n)
    replacement_solution.from_numpy(np.zeros(n, dtype=np.float32))
    plan.solve(prog, replacement_solution.arr, rhs.arr)
    np.testing.assert_allclose(
        replacement_solution.to_numpy(),
        exact / 1.25,
        rtol=3e-3,
        atol=3e-3,
    )
    rebound_address = plan._debug_runtime_stats()
    assert rebound_address["operations"]["solver_chunk_builds"] > rebound[
        "operations"
    ]["solver_chunk_builds"]
    assert rebound_address["operations"]["solver_chunk_rebinds"] > rebound[
        "operations"
    ]["solver_chunk_rebinds"]
    assert rebound_address["operations"][
        "solver_chunk_invalidations"
    ] > rebound["operations"]["solver_chunk_invalidations"]


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_device_native_stored_bsr_block_jacobi_pcg_chunk_replay():
    block_size = 2
    block_rows = 4
    n = block_size * block_rows
    row_offsets = ti.ndarray(dtype=ti.i32, shape=block_rows + 1)
    column_indices = ti.ndarray(dtype=ti.i32, shape=block_rows)
    row_offsets.from_numpy(np.arange(block_rows + 1, dtype=np.int32))
    column_indices.from_numpy(np.arange(block_rows, dtype=np.int32))
    pattern = ti.linalg.SparsePattern.bsr(
        block_rows=block_rows,
        block_cols=block_rows,
        block_size=block_size,
        row_offsets=row_offsets,
        column_indices=column_indices,
    )
    blocks = np.asarray(
        [
            [[3.0, 0.25], [0.25, 2.0]],
            [[2.5, -0.2], [-0.2, 4.0]],
            [[4.0, 0.3], [0.3, 3.0]],
            [[3.5, -0.15], [-0.15, 2.5]],
        ],
        dtype=np.float32,
    )
    values = ti.ndarray(dtype=ti.f32, shape=blocks.size)
    values.from_numpy(blocks.reshape(-1))
    matrix = pattern.matrix(values)
    exact = np.linspace(-0.8, 1.1, n, dtype=np.float32)
    rhs_host = np.concatenate(
        [blocks[i] @ exact[2 * i : 2 * i + 2] for i in range(block_rows)]
    ).astype(np.float32)
    rhs = ti.ndarray(dtype=ti.f32, shape=n)
    solution = ti.ndarray(dtype=ti.f32, shape=n)
    rhs.from_numpy(rhs_host)
    solution.fill(0.0)
    prog = ti.lang.impl.get_runtime().prog
    preconditioner = (
        ti._lib.core._make_sparse_block_jacobi_preconditioner_plan(
            prog, matrix.matrix
        )
    )
    if prog.config().arch == ti._lib.core.Arch.cuda:
        plan = ti._lib.core._make_cuda_block_jacobi_pcg_solver(
            prog,
            matrix.matrix,
            preconditioner,
            8,
            1e-6,
            False,
        )
    else:
        plan = (
            ti._lib.core._make_vulkan_block_jacobi_pcg_convergence_plan(
                prog, matrix.matrix, preconditioner, 8, 1e-6
            )
        )
    plan._configure_execution_policy("host_check_every_k", 4)

    plan.solve(prog, solution.arr, rhs.arr)
    assert plan.get_status() == 2
    np.testing.assert_allclose(
        solution.to_numpy(), exact, rtol=3e-4, atol=3e-4
    )
    first = plan._debug_runtime_stats()
    assert first["identity"]["solver_graph_enabled"]
    assert first["operations"]["solver_chunk_builds"] >= 1
    assert first["operations"]["solver_chunk_direct_submissions"] == 0

    solution.fill(0.0)
    plan.solve(prog, solution.arr, rhs.arr)
    assert plan.get_status() == 2
    np.testing.assert_allclose(
        solution.to_numpy(), exact, rtol=3e-4, atol=3e-4
    )
    second = plan._debug_runtime_stats()
    assert second["operations"]["solver_chunk_builds"] == first[
        "operations"
    ]["solver_chunk_builds"]
    assert second["operations"]["solver_chunk_replays"] > first[
        "operations"
    ]["solver_chunk_replays"]
    assert second["operations"]["solver_chunk_direct_submissions"] == 0
    assert second["operations"]["solver_chunk_invalidations"] == 0
