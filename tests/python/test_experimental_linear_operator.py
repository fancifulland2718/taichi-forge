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


def _fixed_csr(dense):
    dense = np.asarray(dense, dtype=np.float32)
    rows, columns = dense.shape
    row_offsets = ti.ndarray(ti.i32, shape=rows + 1)
    column_indices = ti.ndarray(ti.i32, shape=rows * columns)
    numeric = ti.ndarray(ti.f32, shape=rows * columns)
    row_offsets.from_numpy(
        np.arange(0, rows * columns + 1, columns, dtype=np.int32)
    )
    column_indices.from_numpy(
        np.tile(np.arange(columns, dtype=np.int32), rows)
    )
    numeric.from_numpy(dense.reshape(-1))
    return ti.linalg.SparsePattern.csr(
        rows, columns, row_offsets, column_indices
    ).matrix(numeric)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_experimental_identity_composition_and_apply():
    experimental = ti.linalg.experimental
    identity = experimental.identity(4)
    values = _vector([1.0, -2.0, 3.0, 0.5])

    np.testing.assert_allclose(identity.apply(values).to_numpy(), values.to_numpy())
    np.testing.assert_allclose(
        identity.adjoint().apply(values).to_numpy(), values.to_numpy()
    )
    np.testing.assert_allclose(
        (2.0 * identity + identity).apply(values).to_numpy(),
        3.0 * values.to_numpy(),
    )
    np.testing.assert_allclose(
        (2.0 * identity).compose(identity).apply(values).to_numpy(),
        2.0 * values.to_numpy(),
    )

    blocks = experimental.block_diagonal(
        (experimental.identity(2), experimental.identity(3))
    )
    block_input = _vector([1.0, 2.0, -3.0, 4.0, 5.0])
    np.testing.assert_allclose(
        blocks.apply(block_input).to_numpy(), block_input.to_numpy()
    )
    assert identity.shape == (4, 4)
    assert identity.dtype == ti.f32
    assert identity.capabilities.adjoint_apply
    assert identity.traits["positive_definite"]["value"]
    with pytest.raises(TypeError):
        identity.traits["positive_definite"]["value"] = False


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_experimental_cg_and_bicgstab_reuse_plans():
    experimental = ti.linalg.experimental
    operator = 2.0 * experimental.identity(4)
    rhs = _vector([2.0, -4.0, 6.0, 1.0])
    expected = rhs.to_numpy() / 2.0

    cg = experimental.SolvePlan(
        operator, method="cg", max_iterations=8, atol=1e-6
    )
    first = cg.solve(rhs)
    second = cg.solve(rhs)
    assert first.converged and second.converged
    np.testing.assert_allclose(second.solution.to_numpy(), expected)
    cg_stats = cg.statistics()
    assert cg_stats["operations"]["solve_calls"] == 2
    assert cg_stats["operations"]["workspace_reuses"] == 1

    bicgstab = experimental.SolvePlan(
        operator, method="bicgstab", max_iterations=8, atol=1e-6
    )
    result = bicgstab.solve(rhs)
    assert result.converged
    np.testing.assert_allclose(result.solution.to_numpy(), expected)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_experimental_bicgstab_fixed_linear_right_preconditioner():
    experimental = ti.linalg.experimental
    dense = np.asarray(
        [[40.0, 2.0, 0.0], [-3.0, 5.0, 1.0], [0.0, -1.0, 0.5]],
        dtype=np.float32,
    )
    inverse_action = np.diag(1.0 / np.diag(dense)).astype(np.float32)
    inverse_action[0, 1] = np.float32(-0.01)
    operator = experimental.aslinearoperator(_fixed_csr(dense))
    preconditioner = experimental.aslinearoperator(
        _fixed_csr(inverse_action),
        traits=experimental.OperatorTraits(
            self_adjoint=False, singular=False
        ),
    )
    exact = np.asarray([0.75, -1.25, 2.0], dtype=np.float32)
    plan = experimental.SolvePlan(
        operator,
        method="bicgstab",
        preconditioner=preconditioner,
        max_iterations=12,
        atol=2e-5,
    )

    first = plan.solve(_vector(dense @ exact))
    second = plan.solve(_vector(dense @ exact))
    assert first.converged and second.converged
    assert first.breakdown_reason == "none"
    np.testing.assert_allclose(
        second.solution.to_numpy(), exact, rtol=3e-5, atol=3e-5
    )
    stats = plan.statistics()
    assert stats["identity"]["preconditioning_side"] == "right"
    assert stats["identity"]["preconditioner_method"] == "linear_operator"
    assert stats["identity"]["preconditioner_behavior"] == "fixed_linear"
    assert stats["identity"]["last_breakdown_reason"] == "none"
    assert stats["operations"]["preconditioner_apply_calls"] > 0
    assert stats["operations"]["preconditioner_update_noops"] == 2
    assert stats["resources"]["persistent_vector_count"] == 10
    assert stats["resources"]["transient_solver_workspace_bytes"] == 0


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_experimental_kernel_traits_numeric_update_and_cg():
    experimental = ti.linalg.experimental
    size = 4
    topology = ti.ndarray(ti.i32, shape=size)
    numeric = _vector([2.0, 3.0, 5.0, 7.0])
    topology.from_numpy(np.arange(size, dtype=np.int32))

    @ti.kernel
    def diagonal(
        active_size: ti.i32,
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        numeric_data: ti.types.ndarray(dtype=ti.f32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(active_size):
            y[index] = numeric_data[index] * x[topology_data[index]]

    unknown = experimental.LinearOperator.from_kernel(
        diagonal, size, topology, numeric=numeric
    )
    with pytest.raises(RuntimeError, match="self_adjoint=True"):
        experimental.SolvePlan(unknown, method="cg")

    operator = experimental.LinearOperator.from_kernel(
        diagonal,
        size,
        topology,
        numeric=numeric,
        traits=experimental.OperatorTraits.spd(),
    )
    exact = np.asarray([0.5, -1.0, 2.0, 1.5], dtype=np.float32)
    rhs = _vector(numeric.to_numpy() * exact)
    result = experimental.SolvePlan(operator, max_iterations=16).solve(rhs)
    assert result.converged
    np.testing.assert_allclose(result.solution.to_numpy(), exact, rtol=2e-4)

    updated = _vector([4.0, 6.0, 10.0, 14.0])
    operator.update_numeric(
        updated,
        expected_topology_version=1,
        expected_numeric_version=1,
    )
    np.testing.assert_allclose(
        operator.apply(_vector(np.ones(size))).to_numpy(), updated.to_numpy()
    )


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_experimental_stored_jacobi_pcg():
    experimental = ti.linalg.experimental
    diagonal = np.asarray([2.0, 3.0, 5.0, 7.0], dtype=np.float32)
    matrix = _fixed_diagonal(diagonal)
    operator = experimental.aslinearoperator(
        matrix, traits=experimental.OperatorTraits.spd()
    )
    exact = np.asarray([0.5, -1.0, 2.0, 1.5], dtype=np.float32)
    plan = experimental.SolvePlan(
        operator,
        method="pcg",
        preconditioner="jacobi",
        max_iterations=8,
        atol=1e-6,
    )
    result = plan.solve(_vector(diagonal * exact))
    assert result.converged
    np.testing.assert_allclose(result.solution.to_numpy(), exact, rtol=2e-4)
    updated_diagonal = diagonal * 2.0
    updated_values = _vector(updated_diagonal)
    operator.update_numeric(updated_values)
    repeated = plan.solve(_vector(updated_diagonal * exact))
    assert repeated.converged
    np.testing.assert_allclose(repeated.solution.to_numpy(), exact, rtol=2e-4)
    stats = plan.statistics()
    assert stats["identity"]["preconditioner_method"] == "jacobi"
    assert stats["operations"]["solve_calls"] == 2
    assert stats["operations"]["workspace_reuses"] == 1


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_experimental_fixed_linear_operator_pcg():
    experimental = ti.linalg.experimental
    size = 4
    topology = ti.ndarray(ti.i32, shape=size)
    topology.from_numpy(np.arange(size, dtype=np.int32))
    diagonal = _vector([2.0, 3.0, 5.0, 7.0])
    inverse = _vector(1.0 / diagonal.to_numpy())

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
        diagonal_apply, size, topology, numeric=diagonal, traits=traits
    )
    preconditioner = experimental.LinearOperator.from_kernel(
        diagonal_apply, size, topology, numeric=inverse, traits=traits
    )
    exact = np.asarray([0.5, -1.0, 2.0, 1.5], dtype=np.float32)
    rhs = _vector(diagonal.to_numpy() * exact)
    plan_options = {}
    if impl.current_cfg().arch in (ti.cuda, ti.vulkan):
        plan_options.update(
            execution_policy="host_check_every_k", check_interval=4
        )
    plan = experimental.SolvePlan(
        operator,
        method="pcg",
        preconditioner=preconditioner,
        max_iterations=8,
        atol=1e-5,
        **plan_options,
    )

    first = plan.solve(rhs)
    second = plan.solve(rhs)
    assert first.converged and second.converged
    np.testing.assert_allclose(second.solution.to_numpy(), exact, rtol=2e-4)
    stats = plan.statistics()
    assert stats["identity"]["preconditioner_method"] == "linear_operator"
    assert stats["identity"]["preconditioner_behavior"] == "fixed_linear"
    assert stats["operations"]["preconditioner_apply_calls"] > 0
    assert stats["operations"]["preconditioner_update_noops"] == 2
    assert stats["resources"]["external_preconditioner"]
    if impl.current_cfg().arch in (ti.cuda, ti.vulkan):
        assert stats["operations"]["host_scalar_reductions"] == 0
        assert stats["operations"]["wasted_iterations"] == 6
        assert stats["operations"]["executed_iterations"] == 8
        assert stats["operations"]["logical_iterations"] == 2
    if impl.current_cfg().arch == ti.cuda:
        assert stats["resources"]["cublas_device_pointer_mode"]
    if impl.current_cfg().arch == ti.vulkan:
        assert (
            stats["identity"]["solver_execution_policy"]
            == "host_check_every_k"
        )
        assert stats["operations"]["solver_chunk_direct_submissions"] == 2

    operator.update_numeric(
        _vector(2.0 * diagonal.to_numpy()),
        expected_topology_version=1,
        expected_numeric_version=1,
    )
    with pytest.raises(RuntimeError, match="generation does not match"):
        plan.solve(_vector(2.0 * diagonal.to_numpy() * exact))


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_experimental_host_check_cg_chunk_contract():
    experimental = ti.linalg.experimental
    size = 24
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
    exact = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    rhs = _vector(diagonal_host * exact)
    host_result = experimental.SolvePlan(
        operator, max_iterations=16, atol=1e-6
    ).solve(rhs)
    assert host_result.converged
    for check_interval in (4, 8):
        plan = experimental.SolvePlan(
            operator,
            max_iterations=16,
            atol=1e-6,
            execution_policy="host_check_every_k",
            check_interval=check_interval,
        )
        result = plan.solve(rhs)
        assert result.converged
        assert result.iterations == host_result.iterations
        assert result.residual_norm == pytest.approx(
            host_result.residual_norm, rel=2e-5, abs=2e-7
        )
        np.testing.assert_allclose(
            result.solution.to_numpy(), exact, rtol=2e-4, atol=2e-4
        )
        stats = plan.statistics()
        operations = stats["operations"]
        assert operations["host_scalar_reductions"] == 0
        assert operations["host_scalar_readbacks"] == (
            1 + operations["solver_chunk_direct_submissions"]
        )
        assert operations["operator_apply_calls"] == (
            1 + operations["executed_iterations"]
        )
        assert operations["executed_iterations"] >= result.iterations
        assert operations["wasted_iterations"] <= check_interval - 1
        assert stats["identity"]["solver_scalar_location"] == "device"
        assert not stats["identity"]["solver_graph_enabled"]
        assert stats["resources"]["persistent_scalar_reserved_bytes"] == 92

        zero = plan.solve(_vector(np.zeros(size, dtype=np.float32)))
        assert zero.converged and zero.iterations == 0

    operator.update_numeric(
        _vector(-diagonal_host),
        expected_topology_version=1,
        expected_numeric_version=1,
    )
    breakdown_plan = experimental.SolvePlan(
        operator,
        max_iterations=8,
        atol=1e-6,
        execution_policy="host_check_every_k",
        check_interval=4,
    )
    breakdown = breakdown_plan.solve(rhs)
    assert breakdown.breakdown and breakdown.iterations == 0
    breakdown_stats = breakdown_plan.statistics()["operations"]
    assert breakdown_stats["logical_iterations"] == 0
    assert breakdown_stats["executed_iterations"] == 4


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_experimental_host_check_relative_cg_chunk_contract():
    experimental = ti.linalg.experimental
    size = 24
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
    exact = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    rhs_host = diagonal_host * exact
    rhs = _vector(rhs_host)
    relative_tolerance = 1e-5

    for check_interval in (4, 8):
        plan = experimental.SolvePlan(
            operator,
            max_iterations=16,
            atol=0.0,
            rtol=relative_tolerance,
            execution_policy="host_check_every_k",
            check_interval=check_interval,
        )
        result = plan.solve(rhs)
        assert result.converged
        np.testing.assert_allclose(
            result.solution.to_numpy(), exact, rtol=2e-4, atol=2e-4
        )
        stats = plan.statistics()
        identity = stats["identity"]
        operations = stats["operations"]
        chunks = operations["solver_chunk_direct_submissions"]
        expected_reference = np.linalg.norm(rhs_host)
        assert identity["solver_execution_policy"] == "host_check_every_k"
        assert identity["host_check_interval"] == check_interval
        assert identity["relative_tolerance"] == pytest.approx(
            relative_tolerance
        )
        assert identity["last_relative_reference_norm"] == pytest.approx(
            expected_reference, rel=2e-5
        )
        assert identity["last_effective_tolerance"] == pytest.approx(
            relative_tolerance * expected_reference, rel=2e-5
        )
        assert operations["operator_apply_calls"] == (
            1 + operations["executed_iterations"]
        )
        assert operations["executed_iterations"] >= result.iterations
        assert operations["wasted_iterations"] <= check_interval - 1
        assert operations["host_synchronizations"] == 2 + chunks
        assert operations["host_scalar_readbacks"] == 7 + 2 * chunks
        assert not identity["solver_graph_enabled"]
        assert (
            identity["solver_replay_unavailable_reason"]
            == "provider_not_record_composable"
        )


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_experimental_graph_provider_apply_and_cg():
    experimental = ti.linalg.experimental
    size = 4
    topology = ti.ndarray(ti.i32, shape=size)
    numeric = _vector([2.0, 3.0, 5.0, 7.0])
    topology.from_numpy(np.arange(size, dtype=np.int32))

    @ti.kernel
    def diagonal(
        active_size: ti.i32,
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        numeric_data: ti.types.ndarray(dtype=ti.f32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(active_size):
            y[index] = numeric_data[index] * x[topology_data[index]]

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
        diagonal,
        active_arg,
        topology_arg,
        numeric_arg,
        input_arg,
        output_arg,
    )
    graph = builder.compile()
    operator = experimental.LinearOperator.from_graph(
        graph,
        size,
        fixed_i32={"active_size": size},
        topology={"topology": topology},
        numeric={"numeric": numeric},
        traits=experimental.OperatorTraits.spd(),
    )
    exact = np.asarray([0.5, -1.0, 2.0, 1.5], dtype=np.float32)
    np.testing.assert_allclose(
        operator.apply(_vector(exact)).to_numpy(),
        numeric.to_numpy() * exact,
    )
    result = experimental.SolvePlan(operator, max_iterations=16).solve(
        _vector(numeric.to_numpy() * exact)
    )
    assert result.converged
    np.testing.assert_allclose(result.solution.to_numpy(), exact, rtol=2e-4)
    expected_execution = (
        "explicit_sequence"
        if impl.current_cfg().arch == ti.cpu
        else "compiled_graph"
    )
    assert operator.execution_kind == expected_execution


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_experimental_operator_rejects_alias_and_runtime_reset():
    experimental = ti.linalg.experimental
    operator = experimental.identity(3)
    plan = experimental.SolvePlan(operator, max_iterations=4, atol=1e-6)
    values = _vector([1.0, 2.0, 3.0])
    with pytest.raises(RuntimeError, match="aliasing"):
        operator.apply(values, out=values)

    ti.reset()
    ti.init(arch=ti.cpu, enable_fallback=False, offline_cache=False)
    with pytest.raises(RuntimeError, match="after ti.reset"):
        operator.apply(_vector([1.0, 2.0, 3.0]))
    with pytest.raises(RuntimeError, match="after ti.reset"):
        plan.solve(_vector([1.0, 2.0, 3.0]))
