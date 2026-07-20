import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils


def _make_compiled_diagonal_operator(diagonal):
    diagonal = np.asarray(diagonal, dtype=np.float32)
    size = diagonal.size
    topology = ti.ndarray(ti.i32, shape=size)
    numeric = ti.ndarray(ti.f32, shape=size)
    compile_input = ti.ndarray(ti.f32, shape=size)
    compile_output = ti.ndarray(ti.f32, shape=size)
    topology.from_numpy(np.arange(size, dtype=np.int32))
    numeric.from_numpy(diagonal)

    @ti.kernel
    def apply_diagonal(
        active_size: ti.i32,
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        numeric_data: ti.types.ndarray(dtype=ti.f32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(active_size):
            y[index] = numeric_data[index] * x[topology_data[index]]

    primal = apply_diagonal._primal
    key = primal.ensure_compiled(size, topology, numeric, compile_input, compile_output)
    kernel_cpp = primal.compiled_kernels[key]
    program = impl.get_runtime().prog
    operator = program._create_compiled_kernel_linear_operator_with_numeric_data(
        kernel_cpp,
        size,
        1,
        1,
        topology.arr,
        numeric.arr,
    )
    return program, operator


def _make_compiled_diagonal_preconditioner(program, target, diagonal):
    inverse_program, inverse_operator = _make_compiled_diagonal_operator(
        1.0 / np.asarray(diagonal, dtype=np.float32)
    )
    assert inverse_program is program
    preconditioner = ti._lib.core._make_compiled_kernel_preconditioner_plan(
        program,
        target,
        inverse_operator,
        True,
    )
    return preconditioner, inverse_operator


def _make_compiled_pcg_plan(program, operator, preconditioner):
    arch = impl.current_cfg().arch
    if arch == ti.cpu:
        return (
            ti._lib.core._make_cpu_compiled_kernel_pcg_solver(
                program, operator, preconditioner, 16, 1e-6, 0.0
            ),
            "pcg_compiled_kernel",
        )
    if arch == ti.cuda:
        return (
            ti._lib.core._make_cuda_compiled_kernel_pcg_solver(
                program, operator, preconditioner, 16, 1e-5, False, 0.0
            ),
            "pcg_compiled_kernel",
        )
    return (
        ti._lib.core._make_vulkan_compiled_kernel_pcg_convergence_plan(
            program, operator, preconditioner, 16, 1e-5
        ),
        "pcg_compiled_kernel_bounded_masked_probe",
    )


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_compiled_kernel_cg_reuses_persistent_workspace():
    diagonal = np.asarray([2.0, 3.0, 5.0, 7.0], dtype=np.float32)
    program, operator = _make_compiled_diagonal_operator(diagonal)
    plan = ti._lib.core._make_cuda_compiled_kernel_cg_solver(
        program, operator, 16, 1e-5, False, 0.0
    )
    solution = ti.ndarray(ti.f32, shape=diagonal.size)
    rhs = ti.ndarray(ti.f32, shape=diagonal.size)
    exact_solutions = (
        np.asarray([0.5, -1.0, 2.0, 1.5], dtype=np.float32),
        np.asarray([-2.0, 0.25, 1.25, 3.0], dtype=np.float32),
    )
    for exact in exact_solutions:
        solution.fill(0.0)
        rhs.from_numpy(diagonal * exact)
        plan.solve(program, solution.arr, rhs.arr)
        assert plan.is_success()
        np.testing.assert_allclose(solution.to_numpy(), exact, rtol=2e-4, atol=2e-4)

    stats = plan._debug_runtime_stats()
    assert stats["identity"]["method"] == "cg_compiled_kernel"
    assert stats["identity"]["preconditioner_method"] == "identity"
    assert stats["operations"]["solve_calls"] == 2
    assert stats["operations"]["workspace_builds"] == 1
    assert stats["operations"]["workspace_reuses"] == 1
    assert stats["operations"]["preconditioner_apply_calls"] == 0
    assert stats["resources"]["persistent_vector_count"] == 3
    assert stats["resources"]["persistent_vector_reserved_bytes"] == (
        diagonal.size * 12
    )
    assert stats["resources"]["cublas_handle_count"] == 1
    assert not stats["resources"]["external_preconditioner"]
    assert not stats["resources"]["solver_state_rebuilt_each_solve"]


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_cpu_compiled_kernel_cg_uses_operator_action():
    diagonal = np.asarray([2.0, 3.0, 5.0, 7.0], dtype=np.float32)
    program, operator = _make_compiled_diagonal_operator(diagonal)
    plan = ti._lib.core._make_cpu_operator_cg_solver(program, operator, 16, 1e-6, 0.0)
    solution = ti.ndarray(ti.f32, shape=diagonal.size)
    rhs = ti.ndarray(ti.f32, shape=diagonal.size)
    exact_solutions = (
        np.asarray([0.5, -1.0, 2.0, 1.5], dtype=np.float32),
        np.asarray([-2.0, 0.25, 1.25, 3.0], dtype=np.float32),
    )
    for exact in exact_solutions:
        solution.fill(0.0)
        rhs.from_numpy(diagonal * exact)
        plan.solve(program, solution.arr, rhs.arr)
        assert plan.is_success()
        np.testing.assert_allclose(solution.to_numpy(), exact, rtol=2e-4, atol=2e-4)

    updated_diagonal = diagonal * 1.5
    updated_numeric = ti.ndarray(ti.f32, shape=diagonal.size)
    updated_numeric.from_numpy(updated_diagonal)
    operator.update_numeric_data(program, updated_numeric.arr, 1, 1)
    updated_exact = np.asarray([1.25, -0.5, 0.75, 2.0], dtype=np.float32)
    solution.fill(0.0)
    rhs.from_numpy(updated_diagonal * updated_exact)
    plan.solve(program, solution.arr, rhs.arr)
    assert plan.is_success()
    np.testing.assert_allclose(solution.to_numpy(), updated_exact, rtol=2e-4, atol=2e-4)

    row_offsets = ti.ndarray(ti.i32, shape=diagonal.size + 1)
    column_indices = ti.ndarray(ti.i32, shape=diagonal.size)
    csr_values = ti.ndarray(ti.f32, shape=diagonal.size)
    row_offsets.from_numpy(np.arange(diagonal.size + 1, dtype=np.int32))
    column_indices.from_numpy(np.arange(diagonal.size, dtype=np.int32))
    csr_values.from_numpy(updated_diagonal)
    pattern = program._create_csr_pattern(
        diagonal.size,
        diagonal.size,
        row_offsets.arr,
        column_indices.arr,
    )
    csr_operator = program._create_csr_matrix_from_pattern(pattern, csr_values.arr)
    csr_plan = ti._lib.core._make_cpu_operator_cg_solver(
        program, csr_operator, 16, 1e-6, 0.0
    )
    csr_solution = ti.ndarray(ti.f32, shape=diagonal.size)
    csr_solution.fill(0.0)
    csr_plan.solve(program, csr_solution.arr, rhs.arr)
    assert csr_plan.is_success()
    dense_solution = np.linalg.solve(
        np.diag(updated_diagonal.astype(np.float64)),
        rhs.to_numpy().astype(np.float64),
    )
    np.testing.assert_allclose(
        csr_solution.to_numpy(), dense_solution, rtol=2e-4, atol=2e-4
    )
    np.testing.assert_allclose(
        csr_solution.to_numpy(), solution.to_numpy(), rtol=2e-4, atol=2e-4
    )

    stats = plan._debug_runtime_stats()
    operator_stats = operator._debug_runtime_stats()
    csr_stats = csr_operator._debug_runtime_stats()
    csr_plan_stats = csr_plan._debug_runtime_stats()
    assert stats["identity"]["method"] == "cg_operator_action"
    assert stats["identity"]["preconditioner_method"] == "identity"
    assert stats["operations"]["solve_calls"] == 3
    assert stats["operations"]["workspace_builds"] == 1
    assert stats["operations"]["workspace_reuses"] == 2
    assert stats["operations"]["operator_apply_calls"] > 0
    assert stats["operations"]["preconditioner_apply_calls"] == 0
    assert stats["resources"]["persistent_vector_count"] == 4
    assert stats["resources"]["persistent_vector_reserved_bytes"] == (
        diagonal.size * 16
    )
    assert not stats["resources"]["external_preconditioner"]
    assert stats["resources"]["preconditioner_ownership_scope"] == "none"
    assert operator_stats["operations"]["spmv_calls"] == (
        stats["operations"]["operator_apply_calls"]
    )
    assert operator_stats["identity"]["numeric_version"] == 2
    assert csr_plan_stats["identity"]["method"] == "cg_operator_action"
    assert csr_stats["operations"]["spmv_calls"] == (
        csr_plan_stats["operations"]["operator_apply_calls"]
    )


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_cpu_operator_cg_rejects_reset_without_mutation():
    diagonal = np.asarray([2.0, 3.0, 5.0, 7.0], dtype=np.float32)
    program, operator = _make_compiled_diagonal_operator(diagonal)
    plan = ti._lib.core._make_cpu_operator_cg_solver(program, operator, 16, 1e-6, 0.0)

    ti.reset()
    ti.init(arch=ti.cpu, enable_fallback=False, offline_cache=False)
    replacement_program = impl.get_runtime().prog
    solution = ti.ndarray(ti.f32, shape=diagonal.size)
    rhs = ti.ndarray(ti.f32, shape=diagonal.size)
    solution.fill(31.0)
    rhs.fill(1.0)
    with pytest.raises(RuntimeError, match="construction Program"):
        plan.solve(replacement_program, solution.arr, rhs.arr)
    np.testing.assert_array_equal(
        solution.to_numpy(),
        np.full(diagonal.size, 31.0, dtype=np.float32),
    )


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
)
def test_compiled_kernel_pcg_reuses_persistent_workspace():
    diagonal = np.asarray([2.0, 3.0, 5.0, 7.0], dtype=np.float32)
    program, operator = _make_compiled_diagonal_operator(diagonal)
    preconditioner, inverse_operator = _make_compiled_diagonal_preconditioner(
        program, operator, diagonal
    )
    plan, expected_method = _make_compiled_pcg_plan(program, operator, preconditioner)

    solution = ti.ndarray(ti.f32, shape=diagonal.size)
    rhs = ti.ndarray(ti.f32, shape=diagonal.size)
    exact_solutions = (
        np.asarray([0.5, -1.0, 2.0, 1.5], dtype=np.float32),
        np.asarray([-2.0, 0.25, 1.25, 3.0], dtype=np.float32),
    )
    for exact in exact_solutions:
        solution.fill(0.0)
        rhs.from_numpy(diagonal * exact)
        plan.solve(program, solution.arr, rhs.arr)
        assert plan.is_success()
        np.testing.assert_allclose(solution.to_numpy(), exact, rtol=2e-4, atol=2e-4)

    stats = plan._debug_runtime_stats()
    preconditioner_stats = preconditioner._debug_runtime_stats()
    inverse_stats = inverse_operator._debug_runtime_stats()
    assert stats["identity"]["method"] == expected_method
    assert stats["identity"]["preconditioner_method"] == (
        "compiled_kernel_inverse_apply"
    )
    assert stats["operations"]["solve_calls"] == 2
    assert stats["operations"]["workspace_builds"] == 1
    assert stats["operations"]["workspace_reuses"] == 1
    assert stats["operations"]["preconditioner_apply_calls"] > 0
    assert stats["resources"]["persistent_vector_count"] == 4
    assert stats["resources"]["persistent_vector_reserved_bytes"] == (
        diagonal.size * 16
    )
    assert not stats["resources"]["solver_state_rebuilt_each_solve"]
    assert preconditioner_stats["operations"]["apply_calls"] == (
        stats["operations"]["preconditioner_apply_calls"]
    )
    assert not preconditioner_stats["identity"]["operator_stale"]
    assert not preconditioner_stats["identity"]["preconditioner_stale"]
    assert inverse_stats["operations"]["spmv_calls"] == (
        stats["operations"]["preconditioner_apply_calls"]
    )


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
)
def test_compiled_kernel_pcg_rejects_stale_target_generation():
    diagonal = np.asarray([2.0, 3.0, 5.0, 7.0], dtype=np.float32)
    program, operator = _make_compiled_diagonal_operator(diagonal)
    preconditioner, _ = _make_compiled_diagonal_preconditioner(
        program, operator, diagonal
    )
    plan, _ = _make_compiled_pcg_plan(program, operator, preconditioner)
    updated_numeric = ti.ndarray(ti.f32, shape=diagonal.size)
    updated_numeric.from_numpy(diagonal * 2.0)
    operator.update_numeric_data(program, updated_numeric.arr, 1, 1)

    solution = ti.ndarray(ti.f32, shape=diagonal.size)
    rhs = ti.ndarray(ti.f32, shape=diagonal.size)
    solution.fill(29.0)
    rhs.fill(1.0)
    with pytest.raises(RuntimeError, match="preconditioner is stale"):
        plan.solve(program, solution.arr, rhs.arr)
    np.testing.assert_array_equal(
        solution.to_numpy(),
        np.full(diagonal.size, 29.0, dtype=np.float32),
    )

    plan_stats = plan._debug_runtime_stats()
    preconditioner_stats = preconditioner._debug_runtime_stats()
    assert plan_stats["operations"]["solve_calls"] == 0
    assert plan_stats["operations"]["preconditioner_apply_calls"] == 0
    assert preconditioner_stats["identity"]["operator_stale"]
    assert not preconditioner_stats["identity"]["preconditioner_stale"]
    assert preconditioner_stats["operations"]["apply_calls"] == 0


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
)
def test_compiled_kernel_numeric_update_preserves_inflight_generation():
    diagonal = np.asarray([2.0, 3.0, 5.0, 7.0], dtype=np.float32)
    updated_diagonal = diagonal * 1.5
    program, operator = _make_compiled_diagonal_operator(diagonal)
    input_array = ti.ndarray(ti.f32, shape=diagonal.size)
    first_output = ti.ndarray(ti.f32, shape=diagonal.size)
    second_output = ti.ndarray(ti.f32, shape=diagonal.size)
    updated_numeric = ti.ndarray(ti.f32, shape=diagonal.size)
    input_host = np.asarray([1.0, -2.0, 0.5, 3.0], dtype=np.float32)
    input_array.from_numpy(input_host)
    updated_numeric.from_numpy(updated_diagonal)

    operator.spmv(program, input_array.arr, first_output.arr)
    operator.update_numeric_data(program, updated_numeric.arr, 1, 1)
    operator.spmv(program, input_array.arr, second_output.arr)
    ti.sync()

    np.testing.assert_allclose(first_output.to_numpy(), diagonal * input_host)
    np.testing.assert_allclose(second_output.to_numpy(), updated_diagonal * input_host)
    stats = operator._debug_runtime_stats()
    contract = ti.linalg.SparseMatrix(sm=operator)._get_format_contract()
    assert contract["constraints"]["matrix_free_provider_private"]
    assert not contract["constraints"]["silent_format_fallback"]
    assert stats["identity"]["pattern_version"] == 1
    assert stats["identity"]["numeric_version"] == 2
    assert stats["operations"]["numeric_updates"] == 1
    assert stats["operations"]["numeric_update_bytes"] == diagonal.size * 4
    assert stats["operations"]["spmv_calls"] == 2
    assert stats["operations"]["spmv_plan_builds"] == 1
    assert stats["operations"]["spmv_plan_reuses"] == 2
    assert stats["resources"]["operator_owned_reserved_bytes"] == (diagonal.size * 8)
    assert stats["transfers"]["device_to_host_bytes"] == 0
    assert stats["transfers"]["device_to_device_bytes"] == diagonal.size * 12


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
)
def test_compiled_kernel_operator_rejects_reset_without_mutation():
    diagonal = np.asarray([2.0, 3.0, 5.0, 7.0], dtype=np.float32)
    program, operator = _make_compiled_diagonal_operator(diagonal)
    input_array = ti.ndarray(ti.f32, shape=diagonal.size)
    output_array = ti.ndarray(ti.f32, shape=diagonal.size)
    input_array.fill(1.0)
    output_array.fill(0.0)
    operator.spmv(program, input_array.arr, output_array.arr)
    ti.sync()
    np.testing.assert_allclose(output_array.to_numpy(), diagonal)

    arch = impl.current_cfg().arch
    ti.reset()
    ti.init(arch=arch, enable_fallback=False, offline_cache=False)
    replacement_input = ti.ndarray(ti.f32, shape=diagonal.size)
    replacement_output = ti.ndarray(ti.f32, shape=diagonal.size)
    replacement_input.fill(1.0)
    replacement_output.fill(23.0)
    with pytest.raises(RuntimeError, match="requires its owning Program"):
        operator.spmv(
            impl.get_runtime().prog,
            replacement_input.arr,
            replacement_output.arr,
        )
    ti.sync()
    np.testing.assert_array_equal(
        replacement_output.to_numpy(),
        np.full(diagonal.size, 23.0, dtype=np.float32),
    )


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_compiled_kernel_cg_applies_operator_end_to_end():
    diagonal = np.asarray([2.0, 3.0, 5.0, 7.0], dtype=np.float32)
    exact = np.asarray([0.5, -1.0, 2.0, 1.5], dtype=np.float32)
    program, operator = _make_compiled_diagonal_operator(diagonal)
    solution = ti.ndarray(ti.f32, shape=diagonal.size)
    rhs = ti.ndarray(ti.f32, shape=diagonal.size)
    solution.fill(0.0)
    rhs.from_numpy(diagonal * exact)

    plan = ti._lib.core._make_vulkan_compiled_kernel_cg_convergence_plan(
        program, operator, 16, 1e-5
    )
    plan.solve(program, solution.arr, rhs.arr)

    assert plan.is_success()
    np.testing.assert_allclose(solution.to_numpy(), exact, rtol=2e-4, atol=2e-4)
    plan_stats = plan._debug_runtime_stats()
    operator_stats = operator._debug_runtime_stats()
    assert plan_stats["identity"]["method"] == (
        "cg_compiled_kernel_bounded_masked_probe"
    )
    assert plan_stats["operations"]["operator_apply_calls"] == 17
    assert operator_stats["operations"]["spmv_calls"] == 17
    assert operator_stats["transfers"]["device_to_host_bytes"] == 0
