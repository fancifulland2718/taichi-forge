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


def _nonsymmetric_operator(matrix):
    return ti.linalg.LinearOperator.from_sparse_matrix(
        matrix,
        traits=ti.linalg.OperatorTraits(singular=False),
    )


def _gmres_reference(matrix, rhs, tolerance=1e-12):
    matrix = np.asarray(matrix, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    basis = np.zeros((rhs.size, rhs.size + 1), dtype=np.float64)
    hessenberg = np.zeros((rhs.size + 1, rhs.size), dtype=np.float64)
    beta = np.linalg.norm(rhs)
    basis[:, 0] = rhs / beta
    target = np.zeros(rhs.size + 1, dtype=np.float64)
    target[0] = beta
    candidate = np.zeros(rhs.size, dtype=np.float64)
    for column in range(rhs.size):
        work = matrix @ basis[:, column]
        for row in range(column + 1):
            hessenberg[row, column] = np.dot(basis[:, row], work)
            work -= hessenberg[row, column] * basis[:, row]
        hessenberg[column + 1, column] = np.linalg.norm(work)
        if hessenberg[column + 1, column] != 0:
            basis[:, column + 1] = work / hessenberg[column + 1, column]
        coefficients = np.linalg.lstsq(
            hessenberg[: column + 2, : column + 1],
            target[: column + 2],
            rcond=None,
        )[0]
        candidate = basis[:, : column + 1] @ coefficients
        if np.linalg.norm(matrix @ candidate - rhs) <= tolerance:
            return candidate
    return candidate


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_device_bicgstab_stored_replay_counts_and_terminal_contracts():
    dense = np.asarray(
        [
            [4.0, 1.5, 0.0, -0.25],
            [-0.5, 3.0, 0.75, 0.0],
            [0.0, -1.0, 2.5, 0.5],
            [0.25, 0.0, -0.75, 2.0],
        ],
        dtype=np.float32,
    )
    operator = _nonsymmetric_operator(_fixed_csr(dense))
    exact = np.asarray([1.0, -0.5, 2.0, 0.25], dtype=np.float32)
    rhs_host = dense @ exact
    rhs = _vector(rhs_host)
    output = _vector(np.zeros(exact.size, dtype=np.float32))
    plan = ti.linalg.experimental.SolvePlan(
        operator,
        method="bicgstab",
        max_iterations=24,
        atol=1e-6,
        rtol=1e-6,
        execution_policy="host_check_every_k",
        check_interval=4,
    )

    first = plan.solve(rhs, out=output)
    output.fill(0)
    second = plan.solve(rhs, out=output)
    assert first.converged and second.converged
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
    assert identity["method"] == "bicgstab"
    assert identity["solver_scalar_location"] == "device"
    assert identity["solver_graph_enabled"]
    assert identity["solver_replay_unavailable_reason"] == "none"
    assert identity["preconditioning_side"] == "none"
    assert operations["logical_iterations"] == 8
    assert operations["executed_iterations"] == 8
    assert operations["wasted_iterations"] == 0
    assert operations["operator_apply_calls"] == 20
    assert operations["preconditioner_apply_calls"] == 0
    assert operations["dot_product_calls"] == 54
    assert operations["vector_update_calls"] == 28
    assert operations["solver_chunk_builds"] == 1
    assert operations["solver_chunk_reuses"] == 1
    assert operations["solver_chunk_replays"] == 1
    assert operations["solver_chunk_direct_submissions"] == 0
    assert resources["persistent_vector_count"] == 6
    assert resources["persistent_vector_reserved_bytes"] == 6 * 4 * 4
    assert resources["persistent_scalar_count"] == 28
    assert resources["persistent_scalar_reserved_bytes"] == 112
    assert resources["transient_solver_workspace_bytes"] == 0

    exact_initial = _vector(exact)
    initial = plan.solve(rhs, initial_guess=exact_initial, out=exact_initial)
    assert initial.converged and initial.iterations == 0
    zero = plan.solve(_vector(np.zeros(4, dtype=np.float32)))
    assert zero.converged and zero.iterations == 0
    limited = ti.linalg.experimental.SolvePlan(
        operator,
        method="bicgstab",
        max_iterations=0,
        atol=1e-7,
        execution_policy="host_check_every_k",
        check_interval=4,
    ).solve(rhs)
    assert limited.reached_max_iterations and limited.iterations == 0
    nonfinite = plan.solve(_vector([np.nan, 0.0, 0.0, 0.0]))
    assert nonfinite.breakdown
    assert nonfinite.breakdown_reason == "nonfinite"


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_device_bicgstab_mid_s_convergence_and_structured_breakdown():
    scaled_identity = np.eye(3, dtype=np.float32) * np.float32(2.0)
    operator = _nonsymmetric_operator(_fixed_csr(scaled_identity))
    rhs = _vector([2.0, -4.0, 1.0])
    plan = ti.linalg.experimental.SolvePlan(
        operator,
        method="bicgstab",
        max_iterations=8,
        atol=1e-6,
        execution_policy="host_check_every_k",
        check_interval=4,
    )
    result = plan.solve(rhs)
    assert result.converged and result.iterations == 1
    np.testing.assert_allclose(
        result.solution.to_numpy(), [1.0, -2.0, 0.5], rtol=2e-6, atol=2e-6
    )
    operations = plan.statistics()["operations"]
    assert operations["logical_iterations"] == 1
    assert operations["executed_iterations"] == 4
    assert operations["wasted_iterations"] == 3
    assert operations["operator_apply_calls"] == 10
    assert operations["dot_product_calls"] == 27
    assert operations["vector_update_calls"] == 14

    skew = np.asarray([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    skew_operator = _nonsymmetric_operator(_fixed_csr(skew))
    skew_rhs = np.asarray([1.0, 0.0], dtype=np.float32)
    breakdown = ti.linalg.experimental.SolvePlan(
        skew_operator,
        method="bicgstab",
        max_iterations=4,
        atol=1e-7,
        execution_policy="host_check_every_k",
        check_interval=4,
    ).solve(_vector(skew_rhs))
    assert breakdown.breakdown and not breakdown.converged
    assert breakdown.breakdown_reason == "alpha_denominator"
    gmres = _gmres_reference(skew, skew_rhs)
    assert np.linalg.norm(skew.astype(np.float64) @ gmres - skew_rhs) < 1e-12


@pytest.mark.parametrize("provider", ["kernel", "graph"])
@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_device_bicgstab_compiled_a_m_right_preconditioner(provider):
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
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for row in range(active_size):
            total = 0.0
            for column in range(active_size):
                total += numeric_data[row * active_size + column] * x[
                    topology_data[column]
                ]
            y[row] = total

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

    operator = compiled(
        dense, ti.linalg.OperatorTraits(singular=False)
    )
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
        method="bicgstab",
        preconditioner=preconditioner,
        max_iterations=20,
        atol=1e-6,
        rtol=1e-6,
        execution_policy="host_check_every_k",
        check_interval=4,
    )
    result = plan.solve(_vector(rhs_host))
    assert result.converged
    np.testing.assert_allclose(
        result.solution.to_numpy(), exact, rtol=5e-5, atol=5e-5
    )
    stats = plan.statistics()
    assert stats["identity"]["preconditioning_side"] == "right"
    assert stats["identity"]["preconditioner_method"] == "linear_operator"
    assert not stats["identity"]["solver_graph_enabled"]
    assert stats["identity"]["solver_replay_unavailable_reason"] == (
        "provider_not_capture_composable"
    )
    assert stats["operations"]["solver_chunk_direct_submissions"] > 0
    assert stats["operations"]["preconditioner_apply_calls"] == (
        2 * stats["operations"]["executed_iterations"]
    )
    assert stats["resources"]["external_preconditioner"]
    assert stats["resources"]["persistent_vector_count"] == 8
    assert stats["operations"]["preconditioner_update_noops"] == 1


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_device_bicgstab_direct_and_replay_completion_equivalence(
    monkeypatch,
):
    env_name = (
        "TI_CUDA_SOLVER_CHUNK_REPLAY"
        if ti.lang.impl.current_cfg().arch == ti.cuda
        else "TI_VULKAN_SOLVER_CHUNK_REPLAY"
    )
    dense = np.asarray(
        [[4.0, 1.0, 0.0], [-0.75, 3.0, 0.5], [0.25, -1.0, 2.0]],
        dtype=np.float32,
    )
    operator = _nonsymmetric_operator(_fixed_csr(dense))
    exact = np.asarray([1.0, -0.5, 2.0], dtype=np.float32)
    rhs = _vector(dense @ exact)

    monkeypatch.setenv(env_name, "0")
    direct_plan = ti.linalg.experimental.SolvePlan(
        operator,
        method="bicgstab",
        max_iterations=16,
        atol=1e-6,
        rtol=1e-6,
        execution_policy="host_check_every_k",
        check_interval=4,
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
        method="bicgstab",
        max_iterations=16,
        atol=1e-6,
        rtol=1e-6,
        execution_policy="host_check_every_k",
        check_interval=4,
    )
    replay_output = _vector(np.zeros(3, dtype=np.float32))
    replay_plan.solve(rhs, out=replay_output)
    replay_output.fill(0)
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
def test_device_bicgstab_qualified_execution_policies():
    dense = np.asarray(
        [[4.0, 1.0, 0.0], [-0.5, 3.0, 0.75], [0.25, -1.0, 2.5]],
        dtype=np.float32,
    )
    operator = _nonsymmetric_operator(_fixed_csr(dense))
    exact = np.asarray([1.0, -0.5, 2.0], dtype=np.float32)
    rhs = _vector(dense @ exact)
    if ti.lang.impl.current_cfg().arch == ti.cuda:
        policies = (
            ("host_each_iteration", None),
            ("host_check_every_k", 8),
        )
    else:
        policies = (
            ("host_check_every_k", 8),
            ("fixed_budget_masked", None),
        )
    for policy, interval in policies:
        arguments = {
            "method": "bicgstab",
            "max_iterations": 8,
            "atol": 1e-6,
            "rtol": 1e-6,
            "execution_policy": policy,
        }
        if interval is not None:
            arguments["check_interval"] = interval
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
