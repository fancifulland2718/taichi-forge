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
    offsets = ti.ndarray(ti.i32, shape=len(row_offsets))
    columns = ti.ndarray(ti.i32, shape=len(column_indices))
    numeric = ti.ndarray(ti.f32, shape=len(values))
    offsets.from_numpy(np.asarray(row_offsets, dtype=np.int32))
    columns.from_numpy(np.asarray(column_indices, dtype=np.int32))
    numeric.from_numpy(np.asarray(values, dtype=np.float32))
    return ti.linalg.SparsePattern.csr(rows, cols, offsets, columns).matrix(
        numeric
    )


def _fixed_bsr(block_size, row_offsets, column_indices, blocks):
    blocks = np.asarray(blocks, dtype=np.float32)
    offsets = ti.ndarray(ti.i32, shape=len(row_offsets))
    columns = ti.ndarray(ti.i32, shape=len(column_indices))
    numeric = ti.ndarray(ti.f32, shape=blocks.size)
    offsets.from_numpy(np.asarray(row_offsets, dtype=np.int32))
    columns.from_numpy(np.asarray(column_indices, dtype=np.int32))
    numeric.from_numpy(blocks.reshape(-1))
    block_rows = len(row_offsets) - 1
    return ti.linalg.SparsePattern.bsr(
        block_rows,
        block_rows,
        block_size,
        offsets,
        columns,
    ).matrix(numeric)


def _operator(matrix):
    return ti.linalg.LinearOperator.from_sparse_matrix(
        _fixed_csr(matrix),
        traits=ti.linalg.OperatorTraits(singular=False),
    )


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_device_gmres_stored_replay_counts_and_terminal_contracts():
    dense = np.asarray(
        [
            [4.0, 1.5, 0.0, -0.25],
            [-0.5, 3.0, 0.75, 0.0],
            [0.0, -1.0, 2.5, 0.5],
            [0.25, 0.0, -0.75, 2.0],
        ],
        dtype=np.float32,
    )
    exact = np.asarray([1.0, -0.5, 2.0, 0.25], dtype=np.float32)
    rhs_host = dense @ exact
    rhs = _vector(rhs_host)
    output = _vector(np.zeros_like(exact))
    matrix = _fixed_csr(dense)
    plan = ti.linalg.experimental.SolvePlan(
        ti.linalg.LinearOperator.from_sparse_matrix(
            matrix,
            traits=ti.linalg.OperatorTraits(singular=False),
        ),
        method="gmres",
        restart=8,
        max_iterations=16,
        atol=1e-6,
        rtol=1e-6,
    )
    capabilities = plan.execution_capabilities()
    assert capabilities["default_execution_policy"] == "host_check_every_k"
    assert capabilities["automatic_policy_change"]
    assert capabilities["automatic_solver_replay"]["selected"]

    first = plan.solve(rhs, out=output)
    second = plan.solve(rhs, out=output)
    assert first.converged and second.converged
    assert first.iterations == second.iterations == 4
    assert first.breakdown_reason == second.breakdown_reason == "none"
    np.testing.assert_allclose(output.to_numpy(), exact, rtol=4e-5, atol=4e-5)
    true_residual = np.linalg.norm(dense @ output.to_numpy() - rhs_host)
    assert second.residual_norm == pytest.approx(
        true_residual, rel=8e-2, abs=1e-6
    )

    stats = plan.statistics()
    identity = stats["identity"]
    operations = stats["operations"]
    resources = stats["resources"]
    assert identity["method"] == "gmres"
    assert identity["solver_scalar_location"] == "device"
    assert identity["solver_graph_enabled"]
    assert identity["solver_replay_unavailable_reason"] == "none"
    assert identity["preconditioning_side"] == "none"
    assert operations["restart"] == 8
    assert operations["orthogonalization_strategy"] == "cgs2"
    assert operations["orthogonalization_passes"] == 2
    assert operations["logical_iterations"] == 8
    assert operations["executed_iterations"] == 16
    assert operations["wasted_iterations"] == 8
    assert operations["restart_cycles"] == 2
    assert operations["operator_apply_calls"] == 20
    assert operations["preconditioner_apply_calls"] == 0
    assert operations["dot_product_calls"] == 38
    assert operations["multi_dot_calls"] == 32
    assert operations["vector_update_calls"] == 58
    assert operations["solver_chunk_builds"] == 1
    assert operations["solver_chunk_reuses"] == 1
    assert operations["solver_chunk_replays"] == 1
    assert operations["solver_chunk_direct_submissions"] == 0
    assert resources["basis_vector_count"] == 9
    assert resources["basis_reserved_bytes"] == 9 * 4 * 4
    assert resources["persistent_vector_count"] == 13
    assert resources["persistent_vector_reserved_bytes"] == 13 * 4 * 4
    assert resources["persistent_scalar_count"] == 157
    assert resources["persistent_scalar_reserved_bytes"] == 157 * 4
    assert resources["transient_solver_workspace_bytes"] == 0

    matrix.update_values(_vector((dense[dense != 0]) * np.float32(1.25)))
    refreshed = plan.solve(rhs, out=output)
    assert refreshed.converged
    np.testing.assert_allclose(
        output.to_numpy(), exact / np.float32(1.25), rtol=5e-5, atol=5e-5
    )
    refreshed_ops = plan.statistics()["operations"]
    assert refreshed_ops["solver_chunk_builds"] == 1
    assert refreshed_ops["solver_chunk_rebinds"] >= 1
    assert refreshed_ops["solver_chunk_invalidations"] == 0
    assert refreshed_ops["solver_chunk_replays"] == 2
    matrix.update_values(_vector(dense[dense != 0]))

    exact_initial = _vector(exact)
    initial = plan.solve(rhs, initial_guess=exact_initial, out=exact_initial)
    assert initial.converged and initial.iterations == 0
    zero = plan.solve(_vector(np.zeros(4, dtype=np.float32)))
    assert zero.converged and zero.iterations == 0
    limited = ti.linalg.experimental.SolvePlan(
        _operator(dense),
        method="gmres",
        restart=8,
        max_iterations=0,
        atol=1e-7,
        execution_policy="host_check_every_k",
        check_interval=8,
    ).solve(rhs)
    assert limited.reached_max_iterations and limited.iterations == 0
    nonfinite = plan.solve(_vector([np.nan, 0.0, 0.0, 0.0]))
    assert nonfinite.breakdown
    assert nonfinite.breakdown_reason == "nonfinite"


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_device_gmres_fixed_bsr_identity():
    blocks = np.asarray(
        [
            [[4.0, 1.0], [-0.5, 3.0]],
            [[0.2, 0.0], [0.0, 0.4]],
            [[-0.1, 0.3], [0.0, -0.2]],
            [[2.5, 0.5], [-0.75, 2.0]],
        ],
        dtype=np.float32,
    )
    dense = np.block([[blocks[0], blocks[1]], [blocks[2], blocks[3]]])
    matrix = _fixed_bsr(2, [0, 2, 4], [0, 1, 0, 1], blocks)
    operator = ti.linalg.LinearOperator.from_sparse_matrix(
        matrix,
        traits=ti.linalg.OperatorTraits(singular=False),
    )
    exact = np.asarray([1.0, -0.5, 2.0, 0.25], dtype=np.float32)
    plan = ti.linalg.experimental.SolvePlan(
        operator,
        method="gmres",
        restart=8,
        max_iterations=16,
        atol=1e-6,
        rtol=1e-6,
        execution_policy="host_check_every_k",
        check_interval=8,
    )
    result = plan.solve(_vector(dense @ exact))
    assert result.converged
    np.testing.assert_allclose(
        result.solution.to_numpy(), exact, rtol=5e-5, atol=5e-5
    )
    stats = plan.statistics()
    assert stats["identity"]["operator_action_provider"] in (
        "cusparse",
        "forge_vulkan_native",
    )
    assert stats["operations"]["solver_chunk_builds"] == 1


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_device_gmres_happy_breakdown_and_restarted_true_residual():
    scaled_identity = _operator(np.eye(3, dtype=np.float32) * 2.0)
    happy_plan = ti.linalg.experimental.SolvePlan(
        scaled_identity,
        method="gmres",
        restart=8,
        max_iterations=8,
        atol=1e-6,
        execution_policy="host_check_every_k",
        check_interval=8,
    )
    happy = happy_plan.solve(_vector([2.0, -4.0, 1.0]))
    assert happy.converged and happy.iterations == 1
    assert happy.breakdown_reason == "none"
    np.testing.assert_allclose(
        happy.solution.to_numpy(), [1.0, -2.0, 0.5], rtol=2e-6, atol=2e-6
    )
    happy_ops = happy_plan.statistics()["operations"]
    assert happy_ops["happy_breakdowns"] == 1
    assert happy_ops["logical_iterations"] == 1
    assert happy_ops["executed_iterations"] == 8

    diagonal = np.linspace(1.0, 4.0, 12, dtype=np.float32)
    dense = np.diag(diagonal)
    exact = np.linspace(-1.0, 1.0, 12, dtype=np.float32)
    rhs_host = dense @ exact
    restarted_plan = ti.linalg.experimental.SolvePlan(
        _operator(dense),
        method="gmres",
        restart=8,
        max_iterations=24,
        atol=1e-6,
        rtol=1e-6,
        execution_policy="host_check_every_k",
        check_interval=8,
    )
    restarted = restarted_plan.solve(_vector(rhs_host))
    assert restarted.converged
    output = restarted.solution.to_numpy()
    np.testing.assert_allclose(output, exact, rtol=4e-5, atol=4e-5)
    true_residual = np.linalg.norm(dense @ output - rhs_host)
    assert restarted.residual_norm == pytest.approx(
        true_residual, rel=1e-1, abs=1e-6
    )
    operations = restarted_plan.statistics()["operations"]
    assert operations["restart_cycles"] >= 2
    assert operations["executed_iterations"] % 8 == 0
    assert (
        operations["logical_iterations"] <= operations["executed_iterations"]
    )


@pytest.mark.parametrize("provider", ["kernel", "graph"])
@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_device_gmres_compiled_a_m_right_preconditioner(provider):
    experimental = ti.linalg.experimental
    dense = np.asarray(
        [[5.0, 1.0, -0.5], [-1.0, 4.0, 0.75], [0.25, -1.5, 3.0]],
        dtype=np.float32,
    )
    right_inverse = np.asarray(
        [[0.2, 0.0, 0.0], [0.04, 0.25, 0.0], [0.0, 0.08, 1.0 / 3.0]],
        dtype=np.float32,
    )
    topology = ti.ndarray(ti.i32, shape=3)
    topology.from_numpy(np.arange(3, dtype=np.int32))

    @ti.kernel
    def matrix_action(
        active_size: ti.i32,
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        numeric_data: ti.types.ndarray(dtype=ti.f32, ndim=1),
        input: ti.types.ndarray(dtype=ti.f32, ndim=1),
        output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for row in range(active_size):
            total = 0.0
            for column in range(active_size):
                total += (
                    numeric_data[row * active_size + column]
                    * input[topology_data[column]]
                )
            output[row] = total

    def compiled(values, traits):
        numeric = _vector(np.asarray(values, dtype=np.float32).reshape(-1))
        if provider == "kernel":
            return ti.linalg.LinearOperator.from_kernel(
                matrix_action,
                3,
                topology,
                numeric=numeric,
                traits=traits,
            )
        active_arg = ti.graph.Arg(
            ti.graph.ArgKind.SCALAR, "active_size", ti.i32
        )
        topology_arg = ti.graph.Arg(
            ti.graph.ArgKind.NDARRAY, "topology", ti.i32, ndim=1
        )
        numeric_arg = ti.graph.Arg(
            ti.graph.ArgKind.NDARRAY, "numeric", ti.f32, ndim=1
        )
        input_arg = ti.graph.Arg(
            ti.graph.ArgKind.NDARRAY, "input", ti.f32, ndim=1
        )
        output_arg = ti.graph.Arg(
            ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1
        )
        builder = ti.graph.GraphBuilder()
        builder.dispatch(
            matrix_action,
            active_arg,
            topology_arg,
            numeric_arg,
            input_arg,
            output_arg,
        )
        return ti.linalg.LinearOperator.from_graph(
            builder.compile(),
            3,
            fixed_i32={"active_size": 3},
            topology={"topology": topology},
            numeric={"numeric": numeric},
            traits=traits,
        )

    operator = compiled(dense, ti.linalg.OperatorTraits(singular=False))
    action = compiled(
        right_inverse,
        ti.linalg.OperatorTraits(self_adjoint=False, singular=False),
    )
    preconditioner = experimental.PreconditionerPlan(
        operator, action, method="external_right"
    ).setup()
    exact = np.asarray([1.0, -0.75, 0.5], dtype=np.float32)
    rhs_host = dense @ exact
    plan = experimental.SolvePlan(
        operator,
        method="gmres",
        preconditioner=preconditioner,
        restart=8,
        max_iterations=16,
        atol=1e-6,
        rtol=1e-6,
    )
    capabilities = plan.execution_capabilities()
    batching = capabilities["automatic_solver_batching"]
    assert capabilities["default_execution_policy"] == "host_check_every_k"
    assert capabilities["automatic_policy_change"]
    assert batching["selected"] and batching["qualified"]
    assert batching["default_check_interval"] == "restart"
    expected_provider_execution = (
        "compiled_kernel_direct_apply"
        if provider == "kernel"
        else "compiled_graph_plan_per_apply"
    )
    assert batching["provider_execution"] == expected_provider_execution
    assert not capabilities["automatic_solver_replay"]["selected"]
    result = plan.solve(_vector(rhs_host))
    assert result.converged
    np.testing.assert_allclose(
        result.solution.to_numpy(), exact, rtol=5e-5, atol=5e-5
    )
    stats = plan.statistics()
    expected_execution_kind = (
        "direct" if provider == "kernel" else "compiled_graph"
    )
    assert stats["identity"]["operator_execution_kind"] == (
        expected_execution_kind
    )
    if provider == "graph":
        assert stats["operations"]["operator_compiled_graph_submissions"] > 0
        if ti.lang.impl.current_cfg().arch == ti.cuda:
            assert stats["operations"]["operator_backend_replays"] > 0
        else:
            assert stats["operations"]["operator_backend_replays"] == 0
            assert (
                stats["identity"]["operator_backend_execution_path"]
                == "ordinary_graph_fallback"
            )
            assert stats["operations"]["operator_ordinary_fallbacks"] > 0

    assert stats["identity"]["preconditioning_side"] == "right"
    assert stats["identity"]["preconditioner_method"] == "linear_operator"
    assert not stats["identity"]["solver_graph_enabled"]
    assert stats["identity"]["solver_replay_unavailable_reason"] == (
        "provider_not_capture_composable"
    )
    assert stats["operations"]["solver_chunk_direct_submissions"] > 0
    assert stats["operations"]["preconditioner_apply_calls"] == (
        stats["operations"]["executed_iterations"]
        + stats["operations"]["restart_cycles"]
    )
    assert stats["resources"]["external_preconditioner"]
    assert stats["resources"]["persistent_vector_count"] == 14
    assert stats["operations"]["preconditioner_update_noops"] == 1


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_device_gmres_direct_and_replay_completion_equivalence(monkeypatch):
    env_name = (
        "TI_CUDA_SOLVER_CHUNK_REPLAY"
        if ti.lang.impl.current_cfg().arch == ti.cuda
        else "TI_VULKAN_SOLVER_CHUNK_REPLAY"
    )
    dense = np.asarray(
        [[4.0, 1.0, 0.0], [-0.75, 3.0, 0.5], [0.25, -1.0, 2.0]],
        dtype=np.float32,
    )
    exact = np.asarray([1.0, -0.5, 2.0], dtype=np.float32)
    operator = _operator(dense)
    rhs = _vector(dense @ exact)

    monkeypatch.setenv(env_name, "0")
    direct_plan = ti.linalg.experimental.SolvePlan(
        operator,
        method="gmres",
        restart=8,
        max_iterations=16,
        atol=1e-6,
        rtol=1e-6,
        execution_policy="host_check_every_k",
        check_interval=8,
    )
    direct = direct_plan.solve(rhs)
    direct_stats = direct_plan.statistics()
    assert not direct_stats["identity"]["solver_graph_enabled"]
    assert direct_stats["identity"]["solver_replay_unavailable_reason"] == (
        "disabled_by_environment"
    )
    assert direct_stats["operations"]["solver_chunk_direct_submissions"] > 0

    monkeypatch.setenv(env_name, "1")
    replay_plan = ti.linalg.experimental.SolvePlan(
        operator,
        method="gmres",
        restart=8,
        max_iterations=16,
        atol=1e-6,
        rtol=1e-6,
        execution_policy="host_check_every_k",
        check_interval=8,
    )
    replay_output = _vector(np.zeros(3, dtype=np.float32))
    replay_plan.solve(rhs, out=replay_output)
    replay = replay_plan.solve(rhs, out=replay_output)
    replay_stats = replay_plan.statistics()
    assert replay_stats["identity"]["solver_graph_enabled"]
    assert replay_stats["operations"]["solver_chunk_replays"] > 0

    assert direct.status_code == replay.status_code
    assert direct.iterations == replay.iterations
    assert direct.breakdown_reason == replay.breakdown_reason
    assert direct.residual_norm == pytest.approx(
        replay.residual_norm, rel=1e-5, abs=1e-7
    )
    np.testing.assert_allclose(
        direct.solution.to_numpy(),
        replay.solution.to_numpy(),
        rtol=1e-6,
        atol=1e-6,
    )


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_device_gmres_qualified_execution_policies_and_boundaries():
    dense = np.asarray(
        [[4.0, 1.0, 0.0], [-0.5, 3.0, 0.75], [0.25, -1.0, 2.5]],
        dtype=np.float32,
    )
    operator = _operator(dense)
    exact = np.asarray([1.0, -0.5, 2.0], dtype=np.float32)
    rhs = _vector(dense @ exact)
    policies = ["host_check_every_k"]
    if ti.lang.impl.current_cfg().arch == ti.vulkan:
        policies.append("fixed_budget_masked")
    for policy in policies:
        arguments = {
            "method": "gmres",
            "restart": 8,
            "max_iterations": 8,
            "atol": 1e-6,
            "rtol": 1e-6,
            "execution_policy": policy,
        }
        if policy == "host_check_every_k":
            arguments["check_interval"] = 8
        plan = ti.linalg.experimental.SolvePlan(operator, **arguments)
        result = plan.solve(rhs)
        assert result.converged
        np.testing.assert_allclose(
            result.solution.to_numpy(), exact, rtol=5e-5, atol=5e-5
        )
        stats = plan.statistics()
        assert stats["identity"]["solver_execution_policy"] == policy
        if policy == "fixed_budget_masked":
            assert stats["operations"]["bounded_masked_execution"]
            assert stats["operations"]["executed_iterations"] == 8

    with pytest.raises(Exception, match="check_interval == restart"):
        ti.linalg.experimental.SolvePlan(
            operator,
            method="gmres",
            restart=8,
            max_iterations=8,
            atol=1e-6,
            execution_policy="host_check_every_k",
            check_interval=4,
        )
