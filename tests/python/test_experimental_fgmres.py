import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


def _numpy_dtype(dtype):
    return np.float64 if dtype == ti.f64 else np.float32


def _vector(values, dtype):
    values = np.asarray(values, dtype=_numpy_dtype(dtype))
    result = ti.ndarray(dtype, shape=values.size)
    result.from_numpy(values)
    return result


def _operator(dense, dtype):
    dense = np.asarray(dense, dtype=_numpy_dtype(dtype))
    rows, columns = dense.shape
    row_offsets = [0]
    column_indices = []
    values = []
    for row in range(rows):
        for column in range(columns):
            if dense[row, column] != 0:
                column_indices.append(column)
                values.append(dense[row, column])
        row_offsets.append(len(values))
    offsets = ti.ndarray(ti.i32, shape=len(row_offsets))
    indices = ti.ndarray(ti.i32, shape=len(column_indices))
    numeric = ti.ndarray(dtype, shape=len(values))
    offsets.from_numpy(np.asarray(row_offsets, dtype=np.int32))
    indices.from_numpy(np.asarray(column_indices, dtype=np.int32))
    numeric.from_numpy(np.asarray(values, dtype=_numpy_dtype(dtype)))
    matrix = ti.linalg.SparsePattern.csr(
        rows, columns, offsets, indices
    ).matrix(numeric)
    return ti.linalg.LinearOperator.from_sparse_matrix(
        matrix,
        traits=ti.linalg.OperatorTraits(singular=False),
    )


@pytest.mark.parametrize("dtype", [ti.f32, ti.f64])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_cpu_fgmres_cyclic_actions_z_basis_and_reuse(dtype):
    np_dtype = _numpy_dtype(dtype)
    dense = np.asarray(
        [
            [6.0, 1.0, 0.0, -0.25, 0.0, 0.0],
            [-0.5, 5.0, 0.75, 0.0, 0.0, 0.0],
            [0.0, -1.0, 4.0, 0.5, 0.0, 0.0],
            [0.25, 0.0, -0.75, 3.5, 0.5, 0.0],
            [0.0, 0.0, 0.25, -0.5, 3.0, 1.0],
            [0.1, 0.0, 0.0, 0.25, -0.75, 2.5],
        ],
        dtype=np_dtype,
    )
    inverse_diagonal = 1.0 / np.diag(dense)
    action0 = np.diag(inverse_diagonal).astype(np_dtype)
    action1 = np.diag(
        inverse_diagonal
        * np.asarray([0.65, 1.15, 0.8, 1.25, 0.7, 1.1], dtype=np_dtype)
    )
    operator = _operator(dense, dtype)
    variable = ti.linalg.experimental.PreconditionerPlan(
        operator,
        (_operator(action0, dtype), _operator(action1, dtype)),
        method="alternating_diagonal",
        behavior="variable_linear",
    ).setup()

    metadata = variable.metadata
    assert metadata["supported"]
    assert metadata["behavior"] == "variable_linear"
    assert metadata["selection"] == "cyclic"
    assert metadata["period"] == 2
    session = variable.pin()
    residual = _vector(np.ones(6, dtype=np_dtype), dtype)
    np.testing.assert_allclose(
        session.apply(residual, iteration=0).to_numpy(),
        np.diag(action0),
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        session.apply(residual, iteration=1).to_numpy(),
        np.diag(action1),
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        session.apply(residual, iteration=2).to_numpy(),
        np.diag(action0),
        rtol=2e-5,
        atol=2e-5,
    )

    tolerance = 3e-5 if dtype == ti.f32 else 1e-11
    exact = np.asarray(
        [1.0, -0.5, 2.0, 0.25, -1.0, 0.75], dtype=np_dtype
    )
    plan = ti.linalg.experimental.SolvePlan(
        operator,
        method="fgmres",
        preconditioner=variable,
        restart=8,
        max_iterations=24,
        atol=tolerance,
        rtol=tolerance,
    )
    first = plan.solve(_vector(dense @ exact, dtype))
    second = plan.solve(_vector(dense @ exact, dtype))
    assert first.converged and second.converged
    np.testing.assert_allclose(
        second.solution.to_numpy(),
        exact,
        rtol=10 * tolerance,
        atol=10 * tolerance,
    )

    stats = plan.statistics()
    identity = stats["identity"]
    operations = stats["operations"]
    resources = stats["resources"]
    assert identity["method"] == "fgmres"
    assert identity["preconditioner_behavior"] == "variable_linear"
    assert identity["preconditioner_action_count"] == 2
    assert identity["preconditioner_action_selection"] == (
        "solve_global_scheduled_inner_iteration_mod_period"
    )
    assert operations["preconditioner_apply_calls"] == (
        operations["total_iterations"]
    )
    assert operations["preconditioner_action_selections"] == (
        operations["total_iterations"]
    )
    assert operations["preconditioner_schedule_wraps"] > 0
    assert resources["preconditioned_basis_vector_count"] == 8
    assert resources["preconditioned_basis_reserved_bytes"] == (
        8 * dense.shape[0] * np_dtype().nbytes
    )
    assert resources["persistent_vector_count"] == 20
    assert resources["preconditioner_ownership_scope"] == (
        "solve_plan_action_table_snapshot"
    )
    lifecycle = stats["preconditioner_lifecycle"]
    assert lifecycle["period"] == 2
    assert len(lifecycle["actions"]) == 2


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_fgmres_behavior_boundaries_fail_closed():
    operator = _operator(np.eye(3, dtype=np.float32) * 2.0, ti.f32)
    action = _operator(np.eye(3, dtype=np.float32) * 0.5, ti.f32)
    fixed = ti.linalg.experimental.PreconditionerPlan(
        operator, action
    ).setup()
    with pytest.raises(RuntimeError, match="variable_linear"):
        ti.linalg.experimental.SolvePlan(
            operator,
            method="fgmres",
            preconditioner=fixed,
            restart=8,
            atol=1e-6,
        )

    variable = ti.linalg.experimental.PreconditionerPlan(
        operator,
        (action,),
        behavior="variable_linear",
    ).setup()
    with pytest.raises(RuntimeError, match="only by FGMRES"):
        ti.linalg.experimental.SolvePlan(
            operator,
            method="gmres",
            preconditioner=variable,
            restart=8,
            atol=1e-6,
        )
    with pytest.raises(RuntimeError, match="selection='cyclic'"):
        ti.linalg.experimental.PreconditionerPlan(
            operator,
            (action,),
            behavior="variable_linear",
            selection="callback",
        )
    unsupported = ti.linalg.experimental.PreconditionerPlan(
        operator, action, behavior="nonlinear"
    )
    assert not unsupported.metadata["supported"]
    with pytest.raises(RuntimeError, match="no qualified solver"):
        unsupported.setup()


@pytest.mark.parametrize("provider", ["kernel", "graph"])
@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_device_fgmres_compiled_action_table_and_z_basis(provider):
    experimental = ti.linalg.experimental
    size = 12
    dense = np.diag(np.linspace(2.0, 5.0, size, dtype=np.float32))
    for row in range(size):
        if row + 1 < size:
            dense[row, row + 1] = np.float32(0.35)
        if row > 0:
            dense[row, row - 1] = np.float32(-0.2)
    inverse_diagonal = 1.0 / np.diag(dense)
    scales = (
        np.linspace(0.7, 1.1, size, dtype=np.float32),
        np.linspace(1.2, 0.8, size, dtype=np.float32),
        np.asarray(
            [0.9 if index % 2 == 0 else 1.15 for index in range(size)],
            dtype=np.float32,
        ),
    )
    action_values = tuple(
        np.diag(inverse_diagonal * scale).astype(np.float32)
        for scale in scales
    )
    topology = ti.ndarray(ti.i32, shape=size)
    topology.from_numpy(np.arange(size, dtype=np.int32))

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
                total += numeric_data[row * active_size + column] * input[
                    topology_data[column]
                ]
            output[row] = total

    def compiled(values):
        numeric = _vector(np.asarray(values).reshape(-1), ti.f32)
        traits = ti.linalg.OperatorTraits(singular=False)
        if provider == "kernel":
            return ti.linalg.LinearOperator.from_kernel(
                matrix_action,
                size,
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
            size,
            fixed_i32={"active_size": size},
            topology={"topology": topology},
            numeric={"numeric": numeric},
            traits=traits,
        )

    operator = compiled(dense)
    variable = experimental.PreconditionerPlan(
        operator,
        tuple(compiled(values) for values in action_values),
        method="three_phase_diagonal",
        behavior="variable_linear",
    ).setup()
    exact = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    plan = experimental.SolvePlan(
        operator,
        method="fgmres",
        preconditioner=variable,
        restart=8,
        max_iterations=24,
        atol=1e-5,
        rtol=1e-5,
        execution_policy=(
            "host_check_every_k"
            if ti.lang.impl.current_cfg().arch == ti.cuda
            else "fixed_budget_masked"
        ),
        check_interval=(
            8
            if ti.lang.impl.current_cfg().arch == ti.cuda
            else None
        ),
    )
    result = plan.solve(_vector(dense @ exact, ti.f32))
    assert result.converged, (
        result.residual_norm,
        result.effective_tolerance,
        result.iterations,
    )
    np.testing.assert_allclose(
        result.solution.to_numpy(), exact, rtol=8e-5, atol=8e-5
    )

    stats = plan.statistics()
    identity = stats["identity"]
    operations = stats["operations"]
    resources = stats["resources"]
    assert identity["method"] == "fgmres"
    assert identity["preconditioner_behavior"] == "variable_linear"
    assert identity["preconditioner_action_count"] == 3
    assert identity["solver_replay_unavailable_reason"] == (
        "variable_action_capture_contract_unavailable"
    )
    assert not identity["solver_graph_enabled"]
    assert operations["solver_chunk_direct_submissions"] > 0
    assert operations["preconditioner_apply_calls"] == (
        operations["executed_iterations"]
    )
    assert operations["preconditioner_action_selections"] == (
        operations["executed_iterations"]
    )
    assert resources["preconditioned_basis_vector_count"] == 8
    assert resources["persistent_vector_count"] == 22
    assert resources["preconditioner_ownership_scope"] == (
        "solve_plan_action_table_snapshot"
    )
