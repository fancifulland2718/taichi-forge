import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


def _vector(values, dtype=ti.f32):
    np_dtype = np.float64 if dtype == ti.f64 else np.float32
    values = np.asarray(values, dtype=np_dtype)
    result = ti.ndarray(dtype=dtype, shape=values.size)
    result.from_numpy(values)
    return result


def _fixed_csr(matrix, dtype):
    np_dtype = np.float64 if dtype == ti.f64 else np.float32
    matrix = np.asarray(matrix, dtype=np_dtype)
    rows, cols = matrix.shape
    row_offsets = [0]
    column_indices = []
    values = []
    for row in range(rows):
        for column in range(cols):
            if matrix[row, column] != 0:
                column_indices.append(column)
                values.append(matrix[row, column])
        row_offsets.append(len(values))
    offsets_array = ti.ndarray(ti.i32, shape=len(row_offsets))
    columns_array = ti.ndarray(ti.i32, shape=len(column_indices))
    values_array = ti.ndarray(dtype, shape=len(values))
    offsets_array.from_numpy(np.asarray(row_offsets, dtype=np.int32))
    columns_array.from_numpy(np.asarray(column_indices, dtype=np.int32))
    values_array.from_numpy(np.asarray(values, dtype=np_dtype))
    pattern = ti.linalg.SparsePattern.csr(
        rows, cols, offsets_array, columns_array
    )
    return pattern.matrix(values_array)


@test_utils.test(arch=ti.cpu, offline_cache=False)
@pytest.mark.parametrize("dtype", [ti.f32, ti.f64])
def test_provider_neutral_minres_fixed_csr_reuses_workspace_and_generation(dtype):
    experimental = ti.linalg.experimental
    matrix_host = np.asarray(
        [
            [4.0, 1.0, 0.0, 0.0],
            [1.0, -3.0, 0.5, 0.0],
            [0.0, 0.5, 2.0, 1.0],
            [0.0, 0.0, 1.0, -2.0],
        ],
        dtype=np.float64 if dtype == ti.f64 else np.float32,
    )
    matrix = _fixed_csr(matrix_host, dtype)
    operator = ti.linalg.LinearOperator.from_sparse_matrix(
        matrix,
        traits=ti.linalg.OperatorTraits(
            self_adjoint=True, positive_definite=False, singular=False
        ),
    )
    exact = np.asarray(
        [1.0, -0.5, 2.0, 0.25],
        dtype=np.float64 if dtype == ti.f64 else np.float32,
    )
    rhs = _vector(matrix_host @ exact, dtype)
    plan = experimental.SolvePlan(
        operator, method="minres", max_iterations=20, atol=1e-10, rtol=1e-6
    )

    first = plan.solve(rhs)
    second = plan.solve(rhs)
    assert first.converged and second.converged
    np.testing.assert_allclose(
        second.solution.to_numpy(),
        exact,
        rtol=2e-5 if dtype == ti.f32 else 2e-12,
        atol=2e-5 if dtype == ti.f32 else 2e-12,
    )
    stats = plan.statistics()
    assert stats["identity"]["method"] == "minres"
    assert stats["identity"]["preconditioner_method"] == "identity"
    assert stats["identity"]["operator_action_provider"] == "forge_cpu_native"
    assert stats["operations"]["solve_calls"] == 2
    assert stats["operations"]["workspace_reuses"] == 1
    assert stats["operations"]["operator_generation_pins"] == 2
    assert stats["resources"]["persistent_vector_count"] == 11
    assert stats["resources"]["transient_solver_workspace_bytes"] == 0


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_provider_neutral_minres_compiled_kernel_and_terminal_contracts():
    experimental = ti.linalg.experimental
    topology = ti.ndarray(ti.i32, shape=1)
    topology.from_numpy(np.asarray([0], dtype=np.int32))
    numeric = _vector([2.0, 1.0, 1.0, -1.0])

    @ti.kernel
    def symmetric_indefinite(
        active_size: ti.i32,
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        numeric_data: ti.types.ndarray(dtype=ti.f32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        y[0] = numeric_data[0] * x[0] + numeric_data[1] * x[1]
        y[1] = numeric_data[2] * x[0] + numeric_data[3] * x[1]

    operator = ti.linalg.LinearOperator.from_kernel(
        symmetric_indefinite,
        2,
        topology,
        numeric=numeric,
        traits=ti.linalg.OperatorTraits(
            self_adjoint=True, positive_definite=False, singular=False
        ),
    )
    exact = np.asarray([1.5, -0.25], dtype=np.float32)
    rhs = _vector(np.asarray([[2.0, 1.0], [1.0, -1.0]]) @ exact)
    plan = experimental.SolvePlan(
        operator, method="minres", max_iterations=8, atol=1e-6
    )
    result = plan.solve(rhs)
    assert result.converged
    np.testing.assert_allclose(result.solution.to_numpy(), exact, atol=2e-5)
    assert plan.statistics()["identity"]["operator_action_provider"] == (
        "forge_compiled_taichi_kernel"
    )

    exact_initial = _vector(exact)
    initial = plan.solve(rhs, initial_guess=exact_initial)
    assert initial.converged and initial.iterations == 0
    zero = plan.solve(_vector([0.0, 0.0]))
    assert zero.converged and zero.iterations == 0
    limited = experimental.SolvePlan(
        operator, method="minres", max_iterations=0, atol=1e-8
    ).solve(rhs)
    assert limited.reached_max_iterations and limited.iterations == 0


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_provider_neutral_minres_trait_and_preconditioner_gates():
    experimental = ti.linalg.experimental
    matrix = _fixed_csr(np.diag([1.0, -2.0]).astype(np.float32), ti.f32)
    unknown = ti.linalg.LinearOperator.from_sparse_matrix(matrix)
    with pytest.raises(RuntimeError, match="self_adjoint=True"):
        experimental.SolvePlan(unknown, method="minres")

    singular = ti.linalg.LinearOperator.from_sparse_matrix(
        matrix,
        traits=ti.linalg.OperatorTraits(self_adjoint=True, singular=True),
    )
    with pytest.raises(RuntimeError, match="minimum-length"):
        experimental.SolvePlan(singular, method="minres")

    valid = ti.linalg.LinearOperator.from_sparse_matrix(
        matrix,
        traits=ti.linalg.OperatorTraits(
            self_adjoint=True, positive_definite=False, singular=False
        ),
    )
    with pytest.raises(RuntimeError, match="identity preconditioning only"):
        experimental.SolvePlan(
            valid,
            method="minres",
            preconditioner=ti.linalg.identity(2),
        )
