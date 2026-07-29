import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


def _vector(values):
    values = np.asarray(values, dtype=np.float32)
    result = ti.ndarray(dtype=ti.f32, shape=values.size)
    result.from_numpy(values)
    return result


def _fixed_csr(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
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
    values_array = ti.ndarray(ti.f32, shape=len(values))
    offsets_array.from_numpy(np.asarray(row_offsets, dtype=np.int32))
    columns_array.from_numpy(np.asarray(column_indices, dtype=np.int32))
    values_array.from_numpy(np.asarray(values, dtype=np.float32))
    pattern = ti.linalg.SparsePattern.csr(
        rows, cols, offsets_array, columns_array
    )
    return pattern.matrix(values_array)


def _fixed_bsr(block_size, row_offsets, column_indices, blocks):
    blocks = np.asarray(blocks, dtype=np.float32)
    row_offsets_array = ti.ndarray(ti.i32, shape=len(row_offsets))
    columns_array = ti.ndarray(ti.i32, shape=len(column_indices))
    values_array = ti.ndarray(ti.f32, shape=blocks.size)
    row_offsets_array.from_numpy(np.asarray(row_offsets, dtype=np.int32))
    columns_array.from_numpy(np.asarray(column_indices, dtype=np.int32))
    values_array.from_numpy(blocks.reshape(-1))
    block_rows = len(row_offsets) - 1
    pattern = ti.linalg.SparsePattern.bsr(
        block_rows,
        block_rows,
        block_size,
        row_offsets_array,
        columns_array,
    )
    return pattern.matrix(values_array)


def _indefinite_operator(matrix):
    return ti.linalg.LinearOperator.from_sparse_matrix(
        matrix,
        traits=ti.linalg.OperatorTraits(
            self_adjoint=True, positive_definite=False, singular=False
        ),
    )


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_device_minres_stored_identity_replay_and_terminal_contracts():
    dense = np.asarray(
        [
            [4.0, 1.0, 0.0, 0.0],
            [1.0, -3.0, 0.5, 0.0],
            [0.0, 0.5, 2.0, 1.0],
            [0.0, 0.0, 1.0, -2.0],
        ],
        dtype=np.float32,
    )
    operator = _indefinite_operator(_fixed_csr(dense))
    exact = np.asarray([1.0, -0.5, 2.0, 0.25], dtype=np.float32)
    rhs = _vector(dense @ exact)
    output = _vector(np.zeros(exact.size, dtype=np.float32))
    plan = ti.linalg.experimental.SolvePlan(
        operator,
        method="minres",
        max_iterations=20,
        atol=1e-6,
        rtol=1e-6,
    )
    capabilities = plan.execution_capabilities()
    assert capabilities["default_execution_policy"] == "host_check_every_k"
    assert capabilities["automatic_policy_change"]
    assert capabilities["automatic_solver_replay"]["selected"]

    first = plan.solve(rhs, out=output)
    output.fill(0)
    second = plan.solve(rhs, out=output)
    assert first.converged and second.converged
    np.testing.assert_allclose(output.to_numpy(), exact, rtol=3e-5, atol=3e-5)
    measured_residual = np.linalg.norm(dense @ output.to_numpy() - dense @ exact)
    assert second.residual_norm == pytest.approx(
        measured_residual, rel=8e-2, abs=1e-6
    )

    stats = plan.statistics()
    identity = stats["identity"]
    operations = stats["operations"]
    resources = stats["resources"]
    assert identity["method"] == "minres"
    assert identity["solver_scalar_location"] == "device"
    assert identity["solver_graph_enabled"]
    assert identity["solver_replay_unavailable_reason"] == "none"
    assert operations["host_scalar_reductions"] == 0
    assert operations["host_scalar_readbacks"] == 4
    assert operations["host_synchronizations"] == 4
    assert operations["operator_apply_calls"] == 12
    assert operations["preconditioner_apply_calls"] == 0
    assert operations["solver_chunk_builds"] == 1
    assert operations["solver_chunk_reuses"] == 1
    assert operations["solver_chunk_replays"] == 1
    assert operations["solver_chunk_direct_submissions"] == 0
    assert operations["solver_chunk_invalidations"] == 0
    assert resources["persistent_vector_count"] == 9
    assert resources["persistent_vector_reserved_bytes"] == 9 * 4 * 4
    assert resources["persistent_scalar_count"] == 36
    assert resources["persistent_scalar_reserved_bytes"] == 144
    assert resources["transient_solver_workspace_bytes"] == 0
    expected_d2d = 2 * (4 + 3 * 4 * 4)
    if identity["backend_family"] == "vulkan":
        expected_d2d += 2 * 2 * 4 * 4
    assert stats["transfers"]["device_to_device_bytes"] == expected_d2d

    exact_initial = _vector(exact)
    initial = plan.solve(rhs, initial_guess=exact_initial, out=exact_initial)
    assert initial.converged and initial.iterations == 0
    zero = plan.solve(_vector(np.zeros(4, dtype=np.float32)))
    assert zero.converged and zero.iterations == 0

    limited = ti.linalg.experimental.SolvePlan(
        operator,
        method="minres",
        max_iterations=0,
        atol=1e-7,
        execution_policy="host_check_every_k",
        check_interval=4,
    ).solve(rhs)
    assert limited.reached_max_iterations and limited.iterations == 0

    nonfinite = plan.solve(_vector([np.nan, 0.0, 0.0, 0.0]))
    assert nonfinite.breakdown and not nonfinite.converged

    scalar_dense = np.eye(3, dtype=np.float32) * np.float32(2.0)
    scalar_operator = ti.linalg.LinearOperator.from_sparse_matrix(
        _fixed_csr(scalar_dense),
        traits=ti.linalg.OperatorTraits(
            self_adjoint=True, singular=False
        ),
    )
    happy = ti.linalg.experimental.SolvePlan(
        scalar_operator,
        method="minres",
        max_iterations=4,
        atol=1e-6,
        execution_policy="host_check_every_k",
        check_interval=4,
    ).solve(_vector([2.0, -4.0, 1.0]))
    assert happy.converged and 1 <= happy.iterations <= 4


@pytest.mark.parametrize(
    "spectrum,rtol",
    [
        ([-9.0, -2.0, 0.5, 3.0, 12.0], 2e-5),
        ([-10.0, -0.1, 0.01, 0.001, 1.0], 2e-4),
    ],
)
@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_device_minres_positive_negative_spectrum_and_conditioning(
    spectrum, rtol
):
    dense = np.diag(np.asarray(spectrum, dtype=np.float32))
    operator = _indefinite_operator(_fixed_csr(dense))
    exact = np.asarray([0.25, -1.0, 2.0, -0.75, 1.5], dtype=np.float32)
    rhs_host = dense @ exact
    result = ti.linalg.experimental.SolvePlan(
        operator,
        method="minres",
        max_iterations=32,
        atol=1e-6,
        rtol=rtol,
        execution_policy="host_check_every_k",
        check_interval=8,
    ).solve(_vector(rhs_host))
    assert result.converged
    solution = result.solution.to_numpy()
    true_residual = np.linalg.norm(dense.astype(np.float64) @ solution - rhs_host)
    assert result.residual_norm == pytest.approx(
        true_residual, rel=1e-2, abs=3e-6
    )
    assert true_residual <= result.effective_tolerance * 1.05


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_device_minres_builtin_jacobi_and_block_jacobi():
    dense = np.asarray([[2.0, 3.0], [3.0, 2.0]], dtype=np.float32)
    matrix = _fixed_csr(dense)
    operator = _indefinite_operator(matrix)
    exact = np.asarray([1.25, -0.75], dtype=np.float32)
    rhs = _vector(dense @ exact)
    output = _vector([0.0, 0.0])
    plan = ti.linalg.experimental.SolvePlan(
        operator,
        method="minres",
        preconditioner="jacobi",
        max_iterations=8,
        atol=1e-6,
        execution_policy="host_check_every_k",
        check_interval=4,
    )
    first = plan.solve(rhs, out=output)
    output.fill(0)
    second = plan.solve(rhs, out=output)
    assert first.converged and second.converged
    np.testing.assert_allclose(output.to_numpy(), exact, rtol=3e-5, atol=3e-5)
    first_stats = plan.statistics()
    assert first_stats["identity"]["preconditioner_method"] == "jacobi"
    assert first_stats["identity"]["preconditioner_behavior"] == "fixed_linear"
    assert first_stats["identity"]["preconditioner_action_provider"] == (
        first_stats["identity"]["backend_family"] + "_jacobi"
    )
    assert first_stats["operations"]["preconditioner_apply_calls"] == 10
    assert first_stats["operations"]["solver_chunk_builds"] == 1
    assert first_stats["operations"]["solver_chunk_replays"] == 1

    scaled_values = _vector((dense * np.float32(1.25)).reshape(-1))
    matrix.update_values(scaled_values)
    output.fill(0)
    refreshed = plan.solve(rhs, out=output)
    assert refreshed.converged
    np.testing.assert_allclose(
        output.to_numpy(), exact / np.float32(1.25), rtol=4e-5, atol=4e-5
    )
    refreshed_stats = plan.statistics()
    assert refreshed_stats["operations"]["solver_chunk_builds"] == 1
    assert refreshed_stats["operations"]["solver_chunk_rebinds"] >= 1
    assert refreshed_stats["operations"]["solver_chunk_invalidations"] == 0
    assert refreshed_stats["operations"]["preconditioner_update_successes"] == 1

    identity = np.eye(2, dtype=np.float32)
    diagonal = np.eye(2, dtype=np.float32) * np.float32(2.0)
    coupling = identity * np.float32(3.0)
    blocks = np.asarray([diagonal, coupling, coupling, diagonal])
    block_matrix = _fixed_bsr(2, [0, 2, 4], [0, 1, 0, 1], blocks)
    block_dense = np.block([[diagonal, coupling], [coupling, diagonal]])
    block_operator = _indefinite_operator(block_matrix)
    block_exact = np.asarray([0.5, -1.0, 1.5, 0.25], dtype=np.float32)
    block_rhs = _vector(block_dense @ block_exact)
    block_output = _vector(np.zeros(4, dtype=np.float32))
    block_plan = ti.linalg.experimental.SolvePlan(
        block_operator,
        method="minres",
        preconditioner="block_jacobi",
        max_iterations=8,
        atol=2e-5,
        execution_policy="host_check_every_k",
        check_interval=4,
    )
    block_first = block_plan.solve(block_rhs, out=block_output)
    block_output.fill(0)
    block_second = block_plan.solve(block_rhs, out=block_output)
    assert block_first.converged and block_second.converged
    np.testing.assert_allclose(
        block_output.to_numpy(), block_exact, rtol=4e-5, atol=4e-5
    )
    block_stats = block_plan.statistics()
    assert block_stats["identity"]["preconditioner_method"] == "block_jacobi"
    assert block_stats["operations"]["preconditioner_apply_calls"] > 0
    assert block_stats["operations"]["solver_chunk_builds"] == 1
    assert block_stats["operations"]["solver_chunk_replays"] == 1


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_device_minres_compiled_kernel_graph_and_fixed_linear_plan():
    experimental = ti.linalg.experimental
    topology = ti.ndarray(ti.i32, shape=2)
    topology.from_numpy(np.asarray([0, 1], dtype=np.int32))
    numeric = _vector([2.0, 3.0, 3.0, 2.0])

    @ti.kernel
    def symmetric_indefinite(
        active_size: ti.i32,
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        numeric_data: ti.types.ndarray(dtype=ti.f32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        y[0] = numeric_data[0] * x[topology_data[0]] + numeric_data[1] * x[
            topology_data[1]
        ]
        y[1] = numeric_data[2] * x[topology_data[0]] + numeric_data[3] * x[
            topology_data[1]
        ]

    traits = ti.linalg.OperatorTraits(
        self_adjoint=True, positive_definite=False, singular=False
    )
    kernel_operator = ti.linalg.LinearOperator.from_kernel(
        symmetric_indefinite, 2, topology, numeric=numeric, traits=traits
    )
    exact = np.asarray([1.25, -0.75], dtype=np.float32)
    rhs = _vector(np.asarray([[2.0, 3.0], [3.0, 2.0]]) @ exact)
    kernel_plan = experimental.SolvePlan(
        kernel_operator,
        method="minres",
        max_iterations=8,
        atol=1e-6,
        execution_policy="host_check_every_k",
        check_interval=4,
    )
    kernel_result = kernel_plan.solve(rhs)
    assert kernel_result.converged
    np.testing.assert_allclose(
        kernel_result.solution.to_numpy(), exact, rtol=4e-5, atol=4e-5
    )
    kernel_stats = kernel_plan.statistics()
    assert not kernel_stats["identity"]["solver_graph_enabled"]
    assert (
        kernel_stats["identity"]["solver_replay_unavailable_reason"]
        == "provider_not_capture_composable"
    )
    assert kernel_stats["operations"]["solver_chunk_direct_submissions"] == 1

    active_arg = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "active_size", ti.i32)
    topology_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "topology", ti.i32, ndim=1
    )
    numeric_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "numeric", ti.f32, ndim=1
    )
    input_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "input", ti.f32, ndim=1)
    output_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1
    )
    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        symmetric_indefinite,
        active_arg,
        topology_arg,
        numeric_arg,
        input_arg,
        output_arg,
    )
    graph_operator = ti.linalg.LinearOperator.from_graph(
        builder.compile(),
        2,
        fixed_i32={"active_size": 2},
        topology={"topology": topology},
        numeric={"numeric": numeric},
        traits=traits,
    )
    graph_plan = experimental.SolvePlan(
        graph_operator,
        method="minres",
        max_iterations=8,
        atol=1e-6,
        execution_policy="host_check_every_k",
        check_interval=4,
    )
    graph_result = graph_plan.solve(rhs)
    assert graph_result.converged
    np.testing.assert_allclose(
        graph_result.solution.to_numpy(), exact, rtol=4e-5, atol=4e-5
    )
    assert graph_operator.execution_kind == "compiled_graph"

    inverse_numeric = _vector([0.5, 0.5])

    @ti.kernel
    def inverse_diagonal(
        active_size: ti.i32,
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        numeric_data: ti.types.ndarray(dtype=ti.f32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(active_size):
            y[index] = numeric_data[index] * x[topology_data[index]]

    preconditioner_action = ti.linalg.LinearOperator.from_kernel(
        inverse_diagonal,
        2,
        topology,
        numeric=inverse_numeric,
        traits=ti.linalg.OperatorTraits.spd(),
    )
    preconditioner = experimental.PreconditionerPlan(
        kernel_operator,
        preconditioner_action,
        method="external_diagonal",
    ).setup()
    preconditioned_plan = experimental.SolvePlan(
        kernel_operator,
        method="minres",
        preconditioner=preconditioner,
        max_iterations=8,
        atol=1e-6,
        execution_policy="host_check_every_k",
        check_interval=4,
    )
    preconditioned = preconditioned_plan.solve(rhs)
    assert preconditioned.converged
    np.testing.assert_allclose(
        preconditioned.solution.to_numpy(), exact, rtol=4e-5, atol=4e-5
    )
    preconditioned_stats = preconditioned_plan.statistics()
    assert preconditioned_stats["identity"]["preconditioner_method"] == (
        "linear_operator"
    )
    assert preconditioned_stats["resources"]["external_preconditioner"]
    assert preconditioned_stats["operations"]["preconditioner_apply_calls"] > 0
    assert preconditioned_stats["operations"]["preconditioner_generation_pins"] > 0

    untrusted = ti.linalg.LinearOperator.from_kernel(
        inverse_diagonal,
        2,
        topology,
        numeric=inverse_numeric,
        traits=ti.linalg.OperatorTraits(self_adjoint=True),
    )
    with pytest.raises(RuntimeError, match="positive_definite=True"):
        experimental.SolvePlan(
            kernel_operator,
            method="minres",
            preconditioner=untrusted,
            max_iterations=8,
            execution_policy="host_check_every_k",
            check_interval=4,
        )


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_device_minres_vulkan_default_uses_command_replay_chunks():
    dense = np.asarray([[2.0, 3.0], [3.0, 2.0]], dtype=np.float32)
    operator = _indefinite_operator(_fixed_csr(dense))
    exact = np.asarray([1.25, -0.75], dtype=np.float32)
    result_plan = ti.linalg.experimental.SolvePlan(
        operator, method="minres", max_iterations=16, atol=1e-6
    )
    result = result_plan.solve(_vector(dense @ exact))
    assert result.converged
    stats = result_plan.statistics()
    assert stats["identity"]["solver_execution_policy"] == (
        "host_check_every_k"
    )
    assert stats["identity"]["host_check_interval"] == 4
    assert stats["identity"]["solver_graph_enabled"]
    assert stats["operations"]["host_scalar_readbacks"] == 2
    assert stats["operations"]["host_synchronizations"] == 2
    assert stats["operations"]["executed_iterations"] == 4
    assert stats["operations"]["logical_iterations"] <= 4
