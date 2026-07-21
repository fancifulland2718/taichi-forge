import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils


def _vector(values):
    values = np.asarray(values, dtype=np.float32)
    result = ti.ndarray(ti.f32, shape=values.size)
    result.from_numpy(values)
    return result


def _fixed_diagonal(values):
    values = np.asarray(values, dtype=np.float32)
    size = values.size
    row_offsets = ti.ndarray(ti.i32, shape=size + 1)
    column_indices = ti.ndarray(ti.i32, shape=size)
    numeric = ti.ndarray(ti.f32, shape=size)
    row_offsets.from_numpy(np.arange(size + 1, dtype=np.int32))
    column_indices.from_numpy(np.arange(size, dtype=np.int32))
    numeric.from_numpy(values)
    pattern = ti.linalg.SparsePattern.csr(
        size, size, row_offsets, column_indices
    )
    return pattern.matrix(numeric)


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False
)
def test_independent_batched_cg_status_isolation_and_reuse():
    experimental = ti.linalg.experimental
    batch_size = 4
    system_size = 8
    total_size = batch_size * system_size
    topology = ti.ndarray(ti.i32, shape=total_size)
    topology.from_numpy(np.arange(total_size, dtype=np.int32))
    diagonal_host = np.concatenate(
        (
            np.ones(system_size, dtype=np.float32),
            np.ones(system_size, dtype=np.float32),
            np.resize(
                np.asarray([2.0, 3.0], dtype=np.float32), system_size
            ),
            np.resize(
                np.asarray([2.0, 3.0, 5.0], dtype=np.float32),
                system_size,
            ),
        )
    )
    diagonal = _vector(diagonal_host)

    @ti.kernel
    def diagonal_apply(
        active_size: ti.i32,
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        numeric_data: ti.types.ndarray(dtype=ti.f32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(active_size):
            y[index] = numeric_data[index] * x[topology_data[index]]

    operator = experimental.LinearOperator.from_kernel(
        diagonal_apply,
        total_size,
        topology,
        numeric=diagonal,
        traits=experimental.OperatorTraits.spd(),
    )
    exact = np.linspace(-1.0, 1.0, total_size, dtype=np.float32)
    exact[:system_size] = 0.0
    rhs = _vector(diagonal_host * exact)
    plan = experimental.BatchedSolvePlan(
        operator,
        batch_size,
        independent_systems=True,
        max_iterations=12,
        atol=[1e-6, 1e-6, 2e-6, 3e-6],
        rtol=[0.0, 1e-6, 0.0, 1e-6],
    )

    first = plan.solve(rhs)
    second = plan.solve(rhs)
    assert first.all_converged and second.all_converged
    assert first.iterations[0] == 0
    assert first.iterations[1] == 1
    assert first.iterations[2] <= 2
    assert first.iterations[3] <= 3
    np.testing.assert_allclose(
        second.solution.to_numpy(), exact, rtol=3e-4, atol=3e-4
    )
    stats = plan.statistics()
    assert stats["operations"]["solve_calls"] == 2
    assert stats["resources"]["workspace_builds"] == 1
    assert stats["resources"]["workspace_reuses"] == 1
    assert stats["operations"]["last_masked_provider_system_iterations"] > 0
    assert 0.0 < stats["operations"]["last_active_efficiency"] < 1.0
    assert stats["contract"]["independent_systems"]
    assert stats["contract"]["recurrence_masking"]
    assert not stats["contract"]["provider_apply_masking"]

    negative = diagonal_host.copy()
    negative[2 * system_size : 3 * system_size] *= -1.0
    operator.update_numeric(
        _vector(negative),
        expected_topology_version=1,
        expected_numeric_version=1,
    )
    isolated = plan.solve(_vector(negative * exact))
    assert isolated.status_codes == (2, 2, 1, 2)
    assert isolated.iterations[2] == 0
    isolated_solution = isolated.solution.to_numpy()
    np.testing.assert_allclose(
        isolated_solution[: 2 * system_size],
        exact[: 2 * system_size],
        rtol=3e-4,
        atol=3e-4,
    )
    np.testing.assert_allclose(
        isolated_solution[3 * system_size :],
        exact[3 * system_size :],
        rtol=3e-4,
        atol=3e-4,
    )


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False
)
def test_independent_batched_fixed_operator_pcg():
    experimental = ti.linalg.experimental
    batch_size = 3
    system_size = 8
    total_size = batch_size * system_size
    topology = ti.ndarray(ti.i32, shape=total_size)
    topology.from_numpy(np.arange(total_size, dtype=np.int32))
    diagonal_host = np.resize(
        np.asarray([2.0, 3.0, 5.0, 7.0], dtype=np.float32),
        total_size,
    )
    diagonal = _vector(diagonal_host)
    inverse = _vector(1.0 / diagonal_host)

    @ti.kernel
    def diagonal_apply(
        active_size: ti.i32,
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        numeric_data: ti.types.ndarray(dtype=ti.f32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(active_size):
            y[index] = numeric_data[index] * x[topology_data[index]]

    traits = experimental.OperatorTraits.spd()
    operator = experimental.LinearOperator.from_kernel(
        diagonal_apply,
        total_size,
        topology,
        numeric=diagonal,
        traits=traits,
    )
    preconditioner = experimental.LinearOperator.from_kernel(
        diagonal_apply,
        total_size,
        topology,
        numeric=inverse,
        traits=traits,
    )
    exact = np.linspace(-0.75, 1.25, total_size, dtype=np.float32)
    plan = experimental.BatchedSolvePlan(
        operator,
        batch_size,
        independent_systems=True,
        method="pcg",
        preconditioner=preconditioner,
        max_iterations=8,
        atol=1e-6,
    )
    result = plan.solve(_vector(diagonal_host * exact))

    assert result.all_converged
    assert result.iterations == (1, 1, 1)
    np.testing.assert_allclose(
        result.solution.to_numpy(), exact, rtol=2e-4, atol=2e-4
    )
    stats = plan.statistics()
    assert stats["operations"]["preconditioner_apply_calls"] > 0
    assert stats["resources"]["workspace_vectors"] == 4
    assert stats["resources"]["state_bytes"] == 68 * batch_size + 8
    assert (
        stats["operations"]["host_synchronizations"]
        > stats["operations"]["host_checks"]
    )
    assert stats["contract"]["per_system_status"]


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False
)
def test_independent_batched_stored_operator_and_numeric_update():
    experimental = ti.linalg.experimental
    batch_size = 3
    system_size = 4
    total_size = batch_size * system_size
    diagonal_host = np.resize(
        np.asarray([2.0, 3.0, 5.0], dtype=np.float32), total_size
    )
    inverse_host = 1.0 / diagonal_host
    operator = experimental.aslinearoperator(
        _fixed_diagonal(diagonal_host),
        traits=experimental.OperatorTraits.spd(),
    )
    preconditioner = experimental.aslinearoperator(
        _fixed_diagonal(inverse_host),
        traits=experimental.OperatorTraits.spd(),
    )
    plan = experimental.BatchedSolvePlan(
        operator,
        batch_size,
        independent_systems=True,
        method="pcg",
        preconditioner=preconditioner,
        max_iterations=4,
        atol=1e-6,
    )
    exact = np.linspace(-1.0, 1.0, total_size, dtype=np.float32)
    first = plan.solve(_vector(diagonal_host * exact))
    assert first.all_converged
    assert first.iterations == (1, 1, 1)

    updated_diagonal = diagonal_host * 2.0
    operator.update_numeric(_vector(updated_diagonal))
    preconditioner.update_numeric(_vector(1.0 / updated_diagonal))
    second = plan.solve(_vector(updated_diagonal * exact))
    assert second.all_converged
    np.testing.assert_allclose(
        second.solution.to_numpy(), exact, rtol=2e-4, atol=2e-4
    )


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False
)
def test_independent_batch_one_matches_single_solve_plan():
    experimental = ti.linalg.experimental
    size = 8
    topology = ti.ndarray(ti.i32, shape=size)
    topology.from_numpy(np.arange(size, dtype=np.int32))
    diagonal_host = np.resize(
        np.asarray([2.0, 3.0, 5.0], dtype=np.float32), size
    )
    diagonal = _vector(diagonal_host)

    @ti.kernel
    def diagonal_apply(
        active_size: ti.i32,
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        numeric_data: ti.types.ndarray(dtype=ti.f32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(active_size):
            y[index] = numeric_data[index] * x[topology_data[index]]

    operator = experimental.LinearOperator.from_kernel(
        diagonal_apply,
        size,
        topology,
        numeric=diagonal,
        traits=experimental.OperatorTraits.spd(),
    )
    options = {}
    if impl.current_cfg().arch in (ti.cuda, ti.vulkan):
        options.update(
            execution_policy="host_check_every_k", check_interval=4
        )
    single = experimental.SolvePlan(
        operator, max_iterations=8, atol=1e-6, **options
    )
    batched = experimental.BatchedSolvePlan(
        operator,
        1,
        independent_systems=True,
        max_iterations=8,
        atol=1e-6,
        **options,
    )
    exact = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    rhs = _vector(diagonal_host * exact)
    single_result = single.solve(rhs)
    batched_result = batched.solve(rhs)
    assert single_result.converged
    assert batched_result.converged == (True,)
    assert batched_result.iterations == (single_result.iterations,)
    assert batched_result.residual_norms[0] == pytest.approx(
        single_result.residual_norm, rel=2e-4, abs=5e-7
    )
    np.testing.assert_allclose(
        batched_result.solution.to_numpy(), exact, rtol=3e-4, atol=3e-4
    )


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_independent_batched_contract_and_zero_budget():
    experimental = ti.linalg.experimental
    identity = experimental.identity(4)
    with pytest.raises(RuntimeError, match="independent_systems=True"):
        experimental.BatchedSolvePlan(
            identity, 2, independent_systems=False
        )
    with pytest.raises(RuntimeError, match="divisible"):
        experimental.BatchedSolvePlan(
            experimental.identity(5), 2, independent_systems=True
        )
    with pytest.raises(RuntimeError, match=r"shape \(2,"):
        experimental.BatchedSolvePlan(
            identity,
            2,
            independent_systems=True,
            atol=[1e-6],
        )
    with pytest.raises(RuntimeError, match="fixed LinearOperator"):
        experimental.BatchedSolvePlan(
            identity,
            2,
            independent_systems=True,
            method="pcg",
        )

    plan = experimental.BatchedSolvePlan(
        identity,
        2,
        independent_systems=True,
        max_iterations=0,
        atol=1e-6,
    )
    result = plan.solve(_vector([1.0, 2.0, 3.0, 4.0]))
    assert result.status_codes == (0, 0)
    assert result.iterations == (0, 0)
    np.testing.assert_allclose(
        result.residual_norms, [np.sqrt(5.0), 5.0], rtol=1e-6
    )
