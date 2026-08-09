import gc

import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


def _diagonal_operator(values, *, traits=None):
    values = np.asarray(values, dtype=np.float32)
    size = values.size
    topology = ti.ndarray(ti.i32, shape=size)
    numeric = ti.ndarray(ti.f32, shape=size)
    topology.from_numpy(np.arange(size, dtype=np.int32))
    numeric.from_numpy(values)

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

    return ti.linalg.LinearOperator.from_kernel(
        apply_diagonal,
        size,
        topology,
        numeric=numeric,
        traits=traits,
    )


def _operator_graph(operator, *, adjoint=False):
    input_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "input", ti.f32, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.append_native(
        operator.graph_action(input_arg, output_arg, adjoint=adjoint)
    )
    return builder.compile()


def _rectangular_operator(values):
    values = np.asarray(values, dtype=np.float32).reshape(2, 3)
    topology = ti.ndarray(ti.i32, shape=values.size)
    numeric = ti.ndarray(ti.f32, shape=values.size)
    topology.from_numpy(np.arange(values.size, dtype=np.int32))
    numeric.from_numpy(values.reshape(-1))

    @ti.kernel
    def apply_rectangular(
        active_size: ti.i32,
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        numeric_data: ti.types.ndarray(dtype=ti.f32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for row in range(active_size):
            value = 0.0
            for column in ti.static(range(3)):
                offset = topology_data[row * 3 + column]
                value += numeric_data[offset] * x[column]
            y[row] = value

    @ti.kernel
    def apply_adjoint(
        active_size: ti.i32,
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        numeric_data: ti.types.ndarray(dtype=ti.f32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for column in range(active_size):
            value = 0.0
            for row in ti.static(range(2)):
                offset = topology_data[row * 3 + column]
                value += numeric_data[offset] * x[row]
            y[column] = value

    return ti.linalg.LinearOperator.from_kernel(
        apply_rectangular,
        (2, 3),
        topology,
        adjoint=apply_adjoint,
        numeric=numeric,
    )


def _dense_2x2_operator(values):
    values = np.asarray(values, dtype=np.float32).reshape(2, 2)
    topology = ti.ndarray(ti.i32, shape=1)
    numeric = ti.ndarray(ti.f32, shape=4)
    topology.fill(0)
    numeric.from_numpy(values.reshape(-1))

    @ti.kernel
    def apply_dense(
        active_size: ti.i32,
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        numeric_data: ti.types.ndarray(dtype=ti.f32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for row in range(active_size):
            y[row] = (
                numeric_data[row * 2] * x[0]
                + numeric_data[row * 2 + 1] * x[1]
            )

    @ti.kernel
    def apply_dense_adjoint(
        active_size: ti.i32,
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        numeric_data: ti.types.ndarray(dtype=ti.f32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for column in range(active_size):
            y[column] = (
                numeric_data[column] * x[0]
                + numeric_data[2 + column] * x[1]
            )

    return ti.linalg.LinearOperator.from_kernel(
        apply_dense,
        (2, 2),
        topology,
        adjoint=apply_dense_adjoint,
        numeric=numeric,
    )


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_linear_operator_graph_action_reuses_provider_generation_and_dense_storage():
    operator = _diagonal_operator([2.0, 3.0, 5.0, 7.0])
    first_record = operator._handle._recordable_kernel()
    second_record = operator._handle._recordable_kernel()
    assert (
        first_record._topology.device_allocation().alloc_id
        == second_record._topology.device_allocation().alloc_id
    )
    assert (
        first_record._numeric.device_allocation().alloc_id
        == second_record._numeric.device_allocation().alloc_id
    )

    graph = _operator_graph(operator)
    values = np.asarray([0.5, -1.0, 2.0, 1.5], dtype=np.float32)
    input_array = ti.ndarray(ti.f32, shape=4)
    output_array = ti.ndarray(ti.f32, shape=4)
    input_array.from_numpy(values)
    graph.run({"input": input_array, "output": output_array})
    np.testing.assert_allclose(
        output_array.to_numpy(),
        values * np.asarray([2.0, 3.0, 5.0, 7.0], dtype=np.float32),
    )
    assert graph._instance_debug_info == {"kind": "mixed_backend_region"}
    assert graph._debug_info["native_count"] == 1
    assert graph._debug_info["nodes"][0]["kind"] == "recordable_provider"

    input_field = ti.field(ti.f32, shape=(2, 2))
    output_field = ti.field(ti.f32, shape=(2, 2))
    input_field.from_numpy(values.reshape(2, 2))
    graph.run({"input": input_field, "output": output_field})
    np.testing.assert_allclose(
        output_field.to_numpy().reshape(-1),
        values * np.asarray([2.0, 3.0, 5.0, 7.0], dtype=np.float32),
    )

    with pytest.raises(ti.TaichiRuntimeError, match="proven disjoint"):
        graph.run({"input": input_array, "output": input_array})

    updated = ti.ndarray(ti.f32, shape=4)
    updated.from_numpy(np.asarray([4.0, 6.0, 10.0, 14.0], dtype=np.float32))
    operator.update_numeric(
        updated,
        expected_topology_version=1,
        expected_numeric_version=1,
    )
    graph.run({"input": input_array, "output": output_array})
    np.testing.assert_allclose(
        output_array.to_numpy(),
        values * np.asarray([4.0, 6.0, 10.0, 14.0], dtype=np.float32),
    )


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_linear_operator_graph_action_records_rectangular_adjoint_for_dense_storage():
    matrix = np.asarray(
        [[2.0, -1.0, 0.5], [3.0, 4.0, -2.0]], dtype=np.float32
    )
    operator = _rectangular_operator(matrix)
    assert operator.shape == (2, 3)
    assert operator.capabilities.adjoint_apply

    forward = _operator_graph(operator)
    adjoint = _operator_graph(operator, adjoint=True)

    forward_input = ti.ndarray(ti.f32, shape=3)
    forward_output = ti.ndarray(ti.f32, shape=2)
    forward_values = np.asarray([0.5, -2.0, 1.5], dtype=np.float32)
    forward_input.from_numpy(forward_values)
    forward.run({"input": forward_input, "output": forward_output})
    np.testing.assert_allclose(
        forward_output.to_numpy(), matrix @ forward_values
    )

    adjoint_input = ti.field(ti.f32, shape=(1, 2))
    adjoint_output = ti.field(ti.f32, shape=(3, 1))
    adjoint_values = np.asarray([1.25, -0.75], dtype=np.float32)
    adjoint_input.from_numpy(adjoint_values.reshape(1, 2))
    adjoint.run({"input": adjoint_input, "output": adjoint_output})
    np.testing.assert_allclose(
        adjoint_output.to_numpy().reshape(-1), matrix.T @ adjoint_values
    )

    updated = ti.ndarray(ti.f32, shape=matrix.size)
    updated.from_numpy((2.0 * matrix).reshape(-1))
    operator.update_numeric(
        updated,
        expected_topology_version=1,
        expected_numeric_version=1,
    )
    forward.run({"input": forward_input, "output": forward_output})
    adjoint.run(
        {"input": adjoint_input, "output": adjoint_output}
    )
    np.testing.assert_allclose(
        forward_output.to_numpy(), 2.0 * matrix @ forward_values
    )
    np.testing.assert_allclose(
        adjoint_output.to_numpy().reshape(-1),
        2.0 * matrix.T @ adjoint_values,
    )


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_linear_operator_graph_action_pins_rebound_submission_generations():
    operator = _diagonal_operator([2.0, 3.0, 5.0, 7.0])
    graph = _operator_graph(operator)
    values = np.asarray([0.5, -1.0, 2.0, 1.5], dtype=np.float32)
    source = ti.ndarray(ti.f32, shape=4)
    first_output = ti.ndarray(ti.f32, shape=4)
    second_output = ti.ndarray(ti.f32, shape=4)
    source.from_numpy(values)

    second_generation = ti.ndarray(ti.f32, shape=4)
    second_values = np.asarray([4.0, 6.0, 10.0, 14.0], dtype=np.float32)
    second_generation.from_numpy(second_values)
    operator.update_numeric(
        second_generation,
        expected_topology_version=1,
        expected_numeric_version=1,
    )
    first = graph.submit({"input": source, "output": first_output})

    third_generation = ti.ndarray(ti.f32, shape=4)
    third_values = np.asarray([8.0, 12.0, 20.0, 28.0], dtype=np.float32)
    third_generation.from_numpy(third_values)
    operator.update_numeric(
        third_generation,
        expected_topology_version=1,
        expected_numeric_version=2,
    )
    second = graph.submit({"input": source, "output": second_output})

    assert tuple(first._submission_owners[0]._resource_stamp())[4] == 2
    assert tuple(second._submission_owners[0]._resource_stamp())[4] == 3
    first.wait()
    second.wait()
    assert first._submission_owners == ()
    assert second._submission_owners == ()
    np.testing.assert_allclose(first_output.to_numpy(), second_values * values)
    np.testing.assert_allclose(second_output.to_numpy(), third_values * values)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_graph_numeric_rebind_churn_releases_retired_provider_generations():
    operator = _diagonal_operator([2.0, 3.0, 5.0, 7.0])
    graph = _operator_graph(operator)
    source = ti.ndarray(ti.f32, shape=4)
    output = ti.ndarray(ti.f32, shape=4)
    source.fill(1.0)
    graph.run({"input": source, "output": output})

    for generation in range(1, 33):
        values = np.asarray([2.0, 3.0, 5.0, 7.0], np.float32) * (
            generation + 1
        )
        numeric = ti.ndarray(ti.f32, shape=4)
        numeric.from_numpy(values)
        operator.update_numeric(
            numeric,
            expected_topology_version=1,
            expected_numeric_version=generation,
        )
        graph.run({"input": source, "output": output})
    np.testing.assert_allclose(output.to_numpy(), values)
    provider = operator._provider_core._debug_runtime_stats()
    operations = provider["operations"]
    resources = provider["resources"]
    assert operations["resource_generations_published"] == 33
    assert operations["resource_generations_retired"] == 32
    assert operations["resource_generations_released"] == 32
    assert resources["resource_generation_active_leases"] == 0
    assert resources["operator_owned_reserved_bytes"] == 32


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_solve_plan_submit_rebinds_operator_and_preconditioner_generations():
    diagonal = np.asarray([2.0, 3.0, 5.0, 7.0], dtype=np.float32)
    traits = ti.linalg.OperatorTraits.spd()
    operator = _diagonal_operator(diagonal, traits=traits)
    preconditioner = _diagonal_operator(1.0 / diagonal, traits=traits)
    plan = ti.linalg.experimental.SolvePlan(
        operator,
        method="pcg",
        preconditioner=preconditioner,
        max_iterations=16,
        atol=1e-6,
        execution_policy="device_convergent",
    )
    exact = np.asarray([0.5, -1.0, 2.0, 1.5], dtype=np.float32)
    rhs = ti.ndarray(ti.f32, shape=4)
    rhs.from_numpy(diagonal * exact)
    first = plan.submit(rhs)
    first_result = first.result()
    assert first_result.converged
    np.testing.assert_allclose(first_result.solution.to_numpy(), exact)
    cached_graph = next(iter(plan._submission_graphs.values()))["graph"]

    rebound_diagonal = 1.75 * diagonal
    next_operator = ti.ndarray(ti.f32, shape=4)
    next_preconditioner = ti.ndarray(ti.f32, shape=4)
    next_operator.from_numpy(rebound_diagonal)
    next_preconditioner.from_numpy(1.0 / rebound_diagonal)
    operator.update_numeric(
        next_operator,
        expected_topology_version=1,
        expected_numeric_version=1,
    )
    preconditioner.update_numeric(
        next_preconditioner,
        expected_topology_version=1,
        expected_numeric_version=1,
    )
    rebound_rhs = ti.ndarray(ti.f32, shape=4)
    rebound_rhs.from_numpy(rebound_diagonal * exact)
    second_result = plan.submit(rebound_rhs).result()
    assert second_result.converged
    np.testing.assert_allclose(second_result.solution.to_numpy(), exact)
    assert next(iter(plan._submission_graphs.values()))["graph"] is cached_graph


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_shifted_operator_graph_action_fuses_identity_term_and_rebinds():
    base = _diagonal_operator([2.0, 3.0, 5.0, 7.0])
    assert base.shifted(0.0) is base
    with pytest.raises(RuntimeError, match="finite"):
        base.shifted(float("nan"))
    shifted = base.shifted(1.5)
    graph = _operator_graph(shifted)
    values = np.asarray([0.5, -1.0, 2.0, 1.5], dtype=np.float32)
    source = ti.ndarray(ti.f32, shape=4)
    output = ti.ndarray(ti.f32, shape=4)
    source.from_numpy(values)

    direct = shifted.apply(source)
    np.testing.assert_allclose(
        direct.to_numpy(),
        values * np.asarray([3.5, 4.5, 6.5, 8.5], dtype=np.float32),
    )
    graph.run({"input": source, "output": output})
    np.testing.assert_allclose(
        output.to_numpy(),
        values * np.asarray([3.5, 4.5, 6.5, 8.5], dtype=np.float32),
    )
    node = graph._debug_info["nodes"][0]
    assert node["dispatch_count"] == 2
    assert node.get("temporary_bytes", 0) == 0

    updated = ti.ndarray(ti.f32, shape=4)
    updated.from_numpy(np.asarray([4.0, 6.0, 10.0, 14.0], np.float32))
    base.update_numeric(
        updated,
        expected_topology_version=1,
        expected_numeric_version=1,
    )
    graph.run({"input": source, "output": output})
    np.testing.assert_allclose(
        output.to_numpy(),
        values * np.asarray([5.5, 7.5, 11.5, 15.5], dtype=np.float32),
    )


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_shifted_operator_rejects_rectangular_shape():
    operator = _rectangular_operator(
        [[1.0, 0.0, 2.0], [0.0, 3.0, -1.0]]
    )
    with pytest.raises(RuntimeError, match="square"):
        operator.shifted(1.0)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_parameterized_affine_rebinds_one_atomic_generation_without_rebuild():
    left_values = np.asarray([2.0, 3.0, 5.0, 7.0], np.float32)
    right_values = np.asarray([1.0, 4.0, 2.0, 6.0], np.float32)
    left = _diagonal_operator(left_values, traits=ti.linalg.OperatorTraits.spd())
    right = _diagonal_operator(right_values, traits=ti.linalg.OperatorTraits.spd())
    operator = left.parameterized_affine(
        right,
        alpha=1.0,
        beta=0.25,
        alpha_range=(0.5, 2.0),
        beta_range=(0.0, 1.0),
    )
    assert operator.parameters["version"] == 1
    assert operator.traits["positive_definite"]["value"]
    graph = _operator_graph(operator)
    input_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "parameter_debug_input", ti.f32, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "parameter_debug_output", ti.f32, ndim=1)
    debug = operator.graph_action(input_arg, output_arg).compile().debug_info
    values = np.asarray([0.5, -1.0, 2.0, 1.5], np.float32)
    source = ti.ndarray(ti.f32, shape=4)
    first_output = ti.ndarray(ti.f32, shape=4)
    second_output = ti.ndarray(ti.f32, shape=4)
    source.from_numpy(values)

    first = graph.submit({"input": source, "output": first_output})
    version = operator.update_parameters(alpha=1.5, beta=0.75, expected_version=1)
    assert version == 2
    second = graph.submit({"input": source, "output": second_output})
    first.wait()
    second.wait()
    np.testing.assert_allclose(
        first_output.to_numpy(),
        values * (left_values + 0.25 * right_values),
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        second_output.to_numpy(),
        values * (1.5 * left_values + 0.75 * right_values),
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        operator.apply(source).to_numpy(),
        second_output.to_numpy(),
        rtol=2e-6,
        atol=2e-6,
    )
    assert debug["dispatch_count"] == 3
    assert debug["temporary_bytes"] == values.nbytes
    with pytest.raises(RuntimeError, match="generation changed"):
        operator.update_parameters(alpha=1.0, beta=0.5, expected_version=1)
    with pytest.raises(RuntimeError, match="outside"):
        operator.update_parameters(alpha=-1.0, beta=0.5, expected_version=2)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_parameterized_identity_shift_uses_no_composition_temporary():
    base_values = np.asarray([2.0, 3.0, 5.0, 7.0], np.float32)
    base = _diagonal_operator(base_values, traits=ti.linalg.OperatorTraits.spd())
    operator = base.parameterized_affine(
        alpha=1.0,
        beta=0.5,
        alpha_range=(1.0, 1.0),
        beta_range=(0.0, 2.0),
    )
    graph = _operator_graph(operator)
    input_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "shift_debug_input", ti.f32, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "shift_debug_output", ti.f32, ndim=1)
    debug = operator.graph_action(input_arg, output_arg).compile().debug_info
    assert debug["dispatch_count"] == 2
    assert debug["temporary_bytes"] == 0
    values = np.asarray([1.0, -2.0, 0.25, 3.0], np.float32)
    source = ti.ndarray(ti.f32, shape=4)
    output = ti.ndarray(ti.f32, shape=4)
    source.from_numpy(values)
    graph.run({"input": source, "output": output})
    np.testing.assert_allclose(output.to_numpy(), values * (base_values + 0.5))
    operator.update_parameters(alpha=1.0, beta=1.25, expected_version=1)
    graph.run({"input": source, "output": output})
    np.testing.assert_allclose(output.to_numpy(), values * (base_values + 1.25))


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_deep_composition_reuses_one_temporary_for_forward_and_adjoint():
    matrices = [
        np.asarray([[1.0 + 0.1 * index, 0.2], [-0.1, 0.9]], np.float32)
        for index in range(8)
    ]
    operators = [_dense_2x2_operator(matrix) for matrix in matrices]
    composed = operators[0]
    expected_matrix = matrices[0]
    for operator, matrix in zip(operators[1:], matrices[1:]):
        composed = operator.compose(composed)
        expected_matrix = matrix @ expected_matrix

    input_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "input", ti.f32, ndim=1
    )
    output_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1
    )
    executable = composed.graph_action(input_arg, output_arg).compile()
    assert executable.debug_info["dispatch_count"] == len(matrices)
    assert executable.debug_info["composition_chain_length"] == len(matrices)
    assert executable.debug_info["reuses_composition_temporary"]
    assert executable.debug_info["temporary_bytes"] == 2 * 4

    values = np.asarray([0.75, -1.25], dtype=np.float32)
    source = ti.ndarray(ti.f32, shape=2)
    output = ti.ndarray(ti.f32, shape=2)
    source.from_numpy(values)
    forward = _operator_graph(composed)
    forward.run({"input": source, "output": output})
    np.testing.assert_allclose(
        output.to_numpy(), expected_matrix @ values, rtol=2e-5, atol=2e-5
    )
    assert forward.execution_stats().memory.transient_temporary_bytes == 2 * 4

    adjoint = _operator_graph(composed, adjoint=True)
    adjoint.run({"input": source, "output": output})
    np.testing.assert_allclose(
        output.to_numpy(), expected_matrix.T @ values, rtol=2e-5, atol=2e-5
    )
    adjoint_executable = composed.graph_action(
        input_arg, output_arg, adjoint=True
    ).compile()
    assert adjoint_executable.debug_info["composition_chain_length"] == len(
        matrices
    )
    assert adjoint_executable.debug_info["reuses_composition_temporary"]
    assert adjoint_executable.debug_info["temporary_bytes"] == 2 * 4

    rebound_matrix = np.asarray([[0.75, -0.3], [0.4, 1.25]], np.float32)
    rebound = ti.ndarray(ti.f32, shape=4)
    rebound.from_numpy(rebound_matrix.reshape(-1))
    operators[3].update_numeric(
        rebound,
        expected_topology_version=1,
        expected_numeric_version=1,
    )
    matrices[3] = rebound_matrix
    expected_matrix = matrices[0]
    for matrix in matrices[1:]:
        expected_matrix = matrix @ expected_matrix
    forward.run({"input": source, "output": output})
    np.testing.assert_allclose(
        output.to_numpy(), expected_matrix @ values, rtol=2e-5, atol=2e-5
    )


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_inverse_block_diagonal_is_recordable_and_numerically_rebindable():
    inverse_blocks = np.asarray(
        [
            [2.0, 0.25, 0.25, 1.0],
            [1.5, -0.125, -0.125, 0.75],
        ],
        dtype=np.float32,
    ).reshape(-1)
    numeric = ti.ndarray(ti.f32, shape=inverse_blocks.size)
    numeric.from_numpy(inverse_blocks)
    action = ti.linalg.inverse_block_diagonal(
        numeric, 2, assume_spd=True
    )
    resources = action._provider_core._debug_runtime_stats()["resources"]
    assert resources["pattern_reserved_bytes"] == np.dtype(np.int32).itemsize
    assert resources["values_reserved_bytes"] == inverse_blocks.nbytes
    assert resources["operator_owned_reserved_bytes"] == (
        np.dtype(np.int32).itemsize + inverse_blocks.nbytes
    )
    graph = _operator_graph(action)
    values = np.asarray([1.0, -2.0, 0.5, 3.0], dtype=np.float32)
    source = ti.ndarray(ti.f32, shape=4)
    output = ti.ndarray(ti.f32, shape=4)
    source.from_numpy(values)
    expected = inverse_blocks.reshape(2, 2, 2) @ values.reshape(2, 2, 1)
    graph.run({"input": source, "output": output})
    np.testing.assert_allclose(output.to_numpy(), expected.reshape(-1))
    assert action.traits["self_adjoint"]["value"]
    assert action.traits["positive_definite"]["value"]

    updated_host = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.5],
            [0.75, 0.0, 0.0, 0.25],
        ],
        dtype=np.float32,
    ).reshape(-1)
    updated = ti.ndarray(ti.f32, shape=updated_host.size)
    updated.from_numpy(updated_host)
    action.update_numeric(
        updated,
        expected_topology_version=1,
        expected_numeric_version=1,
    )
    graph.run({"input": source, "output": output})
    expected = updated_host.reshape(2, 2, 2) @ values.reshape(2, 2, 1)
    np.testing.assert_allclose(output.to_numpy(), expected.reshape(-1))

    with pytest.raises(RuntimeError, match="assume_spd=True"):
        ti.linalg.inverse_block_diagonal(updated, 2, assume_spd=False)


@pytest.mark.parametrize("block_size", [1, 2, 3, 4])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_inverse_block_diagonal_specializes_each_supported_block_size(block_size):
    block_count = 3
    blocks = np.empty((block_count, block_size, block_size), dtype=np.float32)
    for block in range(block_count):
        matrix = np.eye(block_size, dtype=np.float32) * (1.5 + block)
        matrix += np.ones((block_size, block_size), dtype=np.float32) * 0.05
        blocks[block] = matrix
    flattened = blocks.reshape(-1)
    numeric = ti.ndarray(ti.f32, shape=flattened.size)
    numeric.from_numpy(flattened)
    action = ti.linalg.inverse_block_diagonal(
        numeric, block_size, assume_spd=True
    )
    graph = _operator_graph(action)
    values = np.linspace(
        -1.0, 2.0, block_count * block_size, dtype=np.float32
    )
    source = ti.ndarray(ti.f32, shape=values.size)
    output = ti.ndarray(ti.f32, shape=values.size)
    source.from_numpy(values)
    graph.run({"input": source, "output": output})
    expected = blocks @ values.reshape(block_count, block_size, 1)
    np.testing.assert_allclose(
        output.to_numpy(), expected.reshape(-1), rtol=1e-6, atol=1e-6
    )
    resources = action._provider_core._debug_runtime_stats()["resources"]
    assert resources["pattern_reserved_bytes"] == np.dtype(np.int32).itemsize
    assert resources["values_reserved_bytes"] == flattened.nbytes


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_inverse_block_diagonal_submission_pins_its_numeric_generation():
    first_values = np.asarray(
        [2.0, 0.25, 0.25, 1.0, 1.5, 0.0, 0.0, 0.75],
        dtype=np.float32,
    )
    second_values = np.asarray(
        [0.75, 0.0, 0.0, 0.5, 2.0, -0.25, -0.25, 1.25],
        dtype=np.float32,
    )
    numeric = ti.ndarray(ti.f32, shape=first_values.size)
    numeric.from_numpy(first_values)
    action = ti.linalg.inverse_block_diagonal(
        numeric, 2, assume_spd=True
    )
    graph = _operator_graph(action)
    source_values = np.asarray([1.0, -2.0, 0.5, 3.0], np.float32)
    source = ti.ndarray(ti.f32, shape=4)
    first_output = ti.ndarray(ti.f32, shape=4)
    second_output = ti.ndarray(ti.f32, shape=4)
    source.from_numpy(source_values)

    first = graph.submit({"input": source, "output": first_output})
    replacement = ti.ndarray(ti.f32, shape=second_values.size)
    replacement.from_numpy(second_values)
    action.update_numeric(
        replacement,
        expected_topology_version=1,
        expected_numeric_version=1,
    )
    second = graph.submit({"input": source, "output": second_output})
    first.wait()
    second.wait()

    expected_first = first_values.reshape(2, 2, 2) @ source_values.reshape(
        2, 2, 1
    )
    expected_second = second_values.reshape(2, 2, 2) @ source_values.reshape(
        2, 2, 1
    )
    np.testing.assert_allclose(first_output.to_numpy(), expected_first.reshape(-1))
    np.testing.assert_allclose(second_output.to_numpy(), expected_second.reshape(-1))
    resources = action._provider_core._debug_runtime_stats()["resources"]
    assert resources["resource_generation_active_leases"] == 0


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_linear_operator_graph_action_embeds_in_multi_condition_while():
    operator = _diagonal_operator([1.0, 1.0, 1.0, 1.0])

    @ti.kernel
    def evaluate_condition(
        state: ti.types.ndarray(dtype=ti.f32, ndim=1),
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
        enabled: ti.types.ndarray(dtype=ti.i32, ndim=0),
        target: ti.i32,
    ):
        predicate[None] = int(enabled[None] != 0 and counter[None] < target)

    @ti.kernel
    def commit_step(
        candidate: ti.types.ndarray(dtype=ti.f32, ndim=1),
        state: ti.types.ndarray(dtype=ti.f32, ndim=1),
        counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
    ):
        if predicate[None] != 0:
            for index in state:
                state[index] = candidate[index] + 1.0
            counter[None] += 1

    state_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "state", ti.f32, ndim=1)
    candidate_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "candidate", ti.f32, ndim=1)
    predicate_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "predicate", ti.i32, ndim=0)
    counter_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "counter", ti.i32, ndim=0)
    enabled_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "enabled", ti.i32, ndim=0)
    target_arg = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "target", ti.i32)

    builder = ti.graph.GraphBuilder()
    condition = builder.create_sequential()
    condition.dispatch(
        evaluate_condition,
        state_arg,
        predicate_arg,
        counter_arg,
        enabled_arg,
        target_arg,
    )
    body = builder.create_sequential()
    body.append_native(operator.graph_action(state_arg, candidate_arg))
    body.dispatch(
        commit_step,
        candidate_arg,
        state_arg,
        counter_arg,
        predicate_arg,
    )
    builder.while_loop(
        condition,
        body,
        predicate=predicate_arg,
        control_inputs=(counter_arg, enabled_arg, target_arg),
        carried_state=(state_arg, candidate_arg),
        counter=counter_arg,
        max_iterations=8,
        name="operator_iteration",
    )
    graph = builder.compile()

    state = ti.ndarray(ti.f32, shape=4)
    candidate = ti.ndarray(ti.f32, shape=4)
    predicate = ti.ndarray(ti.i32, shape=())
    counter = ti.ndarray(ti.i32, shape=())
    enabled = ti.ndarray(ti.i32, shape=())
    state.fill(0.0)
    candidate.fill(0.0)
    predicate.fill(0)
    counter.fill(0)
    enabled.fill(1)
    graph.run(
        {
            "state": state,
            "candidate": candidate,
            "predicate": predicate,
            "counter": counter,
            "enabled": enabled,
            "target": 3,
        }
    )
    np.testing.assert_allclose(state.to_numpy(), np.full(4, 3.0, np.float32))
    assert counter.to_numpy()[()] == 3
    report = graph.control_flow_stats()[0]
    assert report.logical_iterations == 3
    assert graph._debug_info["native_count"] == 1


@pytest.mark.parametrize("provider_mode", ("legacy_square", "generic_square"))
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_compiled_graph_operator_exports_ordered_multi_dispatch_action(
    monkeypatch,
    provider_mode,
):
    monkeypatch.setenv("TI_GRAPH_TWO_MAP_COMPOSER", "0")
    size = 4
    topology = ti.ndarray(ti.i32, shape=size)
    numeric = ti.ndarray(ti.f32, shape=size)
    workspace = ti.ndarray(ti.f32, shape=size)
    topology.from_numpy(np.arange(size, dtype=np.int32))
    numeric.from_numpy(np.asarray([2.0, 3.0, 5.0, 7.0], np.float32))

    @ti.kernel
    def stage(
        active_size: ti.i32,
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        numeric_data: ti.types.ndarray(dtype=ti.f32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        temporary: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(active_size):
            temporary[index] = (
                numeric_data[index] * x[topology_data[index]]
            )

    @ti.kernel
    def finish(
        active_size: ti.i32,
        temporary: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(active_size):
            y[index] = temporary[index] + 1.0

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
    workspace_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "workspace", ti.f32, ndim=1
    )
    provider_builder = ti.graph.GraphBuilder()
    provider_builder.dispatch(
        stage,
        active_arg,
        topology_arg,
        numeric_arg,
        input_arg,
        workspace_arg,
    )
    provider_builder.dispatch(
        finish, active_arg, workspace_arg, output_arg
    )
    operator = ti.linalg.LinearOperator.from_graph(
        provider_builder.compile(),
        size if provider_mode == "legacy_square" else (size, size),
        fixed_i32={"active_size": size},
        topology={"topology": topology},
        numeric={"numeric": numeric},
        workspace={"workspace": workspace},
    )

    action_node = operator.graph_action(input_arg, output_arg)
    action_info = action_node.compile().debug_info["recordable"]
    assert action_info["dispatch_count"] == 2
    assert action_info["fixed_ndarray_count"] == 3
    assert action_info["provider_fixed_snapshot_reserved_bytes"] == 48
    assert action_info["outer_graph_resource_snapshot_copies"] == 0
    assert action_info["outer_graph_resource_snapshot_reserved_bytes"] == 0
    assert action_info["state_snapshot_copies"] == 0

    outer_builder = ti.graph.GraphBuilder()
    outer_builder.append_native(action_node)
    graph = outer_builder.compile()
    values = np.asarray([0.5, -1.0, 2.0, 1.5], np.float32)
    input_array = ti.ndarray(ti.f32, shape=size)
    output_array = ti.ndarray(ti.f32, shape=size)
    input_array.from_numpy(values)
    graph.run({"input": input_array, "output": output_array})
    np.testing.assert_allclose(
        output_array.to_numpy(), numeric.to_numpy() * values + 1.0
    )

    replacement = ti.ndarray(ti.f32, shape=size)
    replacement.from_numpy(2.0 * numeric.to_numpy())
    operator.update_numeric(
        {"numeric": replacement},
        expected_topology_version=1,
        expected_numeric_version=1,
    )
    graph.run({"input": input_array, "output": output_array})
    np.testing.assert_allclose(
        output_array.to_numpy(), replacement.to_numpy() * values + 1.0
    )


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_compiled_graph_fixed_dense_field_state_replays_without_snapshot():
    size = 4
    fields = ti.FieldsBuilder()
    weights = ti.field(ti.f32)
    fields.dense(ti.i, size).place(weights)
    tree = fields.finalize()
    weights.from_numpy(
        np.asarray([2.0, 3.0, 5.0, 7.0], dtype=np.float32)
    )
    topology = ti.ndarray(ti.i32, shape=size)
    topology.from_numpy(np.arange(size, dtype=np.int32))

    @ti.kernel
    def apply_state(
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(size):
            y[index] = weights[index] * x[topology_data[index]]

    topology_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "topology", ti.i32, ndim=1
    )
    input_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "input", ti.f32, ndim=1
    )
    output_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1
    )
    provider_builder = ti.graph.GraphBuilder()
    provider_builder.dispatch(
        apply_state, topology_arg, input_arg, output_arg
    )
    provider_graph = provider_builder.compile()

    with pytest.raises(
        RuntimeError, match="explicitly declared"
    ):
        ti.linalg.LinearOperator.from_graph(
            provider_graph,
            size,
            topology={"topology": topology},
        )

    operator = ti.linalg.LinearOperator.from_graph(
        provider_graph,
        size,
        topology={"topology": topology},
        state={"weights": weights},
    )
    node = operator.graph_action(input_arg, output_arg)
    recordable = node.compile().debug_info["recordable"]
    assert recordable["state_tree_count"] == 1
    assert recordable["state_snapshot_copies"] == 0
    assert recordable["state_snapshot_reserved_bytes"] == 0
    assert recordable["outer_graph_resource_snapshot_copies"] == 0
    assert recordable["outer_graph_resource_snapshot_reserved_bytes"] == 0

    outer_builder = ti.graph.GraphBuilder()
    outer_builder.append_native(node)
    graph = outer_builder.compile()
    values = np.asarray([0.5, -1.0, 2.0, 1.5], np.float32)
    input_array = ti.ndarray(ti.f32, shape=size)
    output_array = ti.ndarray(ti.f32, shape=size)
    input_array.from_numpy(values)
    graph.run({"input": input_array, "output": output_array})
    np.testing.assert_allclose(
        output_array.to_numpy(), weights.to_numpy() * values
    )

    weights.fill(3.0)
    graph.run({"input": input_array, "output": output_array})
    np.testing.assert_allclose(output_array.to_numpy(), 3.0 * values)

    tree.destroy()
    with pytest.raises(
        ti.TaichiRuntimeError, match="destroyed SNodeTree|generation"
    ):
        graph.run({"input": input_array, "output": output_array})


@pytest.mark.parametrize("state_kind", ("vector", "matrix"))
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_graph_action_accepts_fixed_root_dense_packed_state(state_kind):
    size = 4
    fields = ti.FieldsBuilder()
    if state_kind == "vector":
        state = ti.Vector.field(2, dtype=ti.f32)
    else:
        state = ti.Matrix.field(2, 2, dtype=ti.f32)
    fields.dense(ti.i, size).place(state)
    tree = fields.finalize()
    state.fill(2.0)
    topology = ti.ndarray(ti.i32, shape=size)
    topology.from_numpy(np.arange(size, dtype=np.int32))

    if state_kind == "vector":

        @ti.kernel
        def apply_state(
            topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
            x: ti.types.ndarray(dtype=ti.f32, ndim=1),
            y: ti.types.ndarray(dtype=ti.f32, ndim=1),
        ):
            for index in range(size):
                y[index] = state[index][0] * x[topology_data[index]]

    else:

        @ti.kernel
        def apply_state(
            topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
            x: ti.types.ndarray(dtype=ti.f32, ndim=1),
            y: ti.types.ndarray(dtype=ti.f32, ndim=1),
        ):
            for index in range(size):
                y[index] = state[index][0, 0] * x[topology_data[index]]

    topology_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "topology", ti.i32, ndim=1
    )
    input_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "input", ti.f32, ndim=1
    )
    output_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1
    )
    provider_builder = ti.graph.GraphBuilder()
    provider_builder.dispatch(
        apply_state, topology_arg, input_arg, output_arg
    )
    operator = ti.linalg.LinearOperator.from_graph(
        provider_builder.compile(),
        (size, size),
        topology={"topology": topology},
        state={"state": state},
    )
    node = operator.graph_action(input_arg, output_arg)
    assert node.compile().debug_info["recordable"]["state_tree_count"] == 1
    outer_builder = ti.graph.GraphBuilder()
    outer_builder.append_native(node)
    graph = outer_builder.compile()

    values = np.asarray([0.5, -1.0, 2.0, 1.5], np.float32)
    input_array = ti.ndarray(ti.f32, shape=size)
    output_array = ti.ndarray(ti.f32, shape=size)
    input_array.from_numpy(values)
    graph.run({"input": input_array, "output": output_array})
    np.testing.assert_allclose(output_array.to_numpy(), 2.0 * values)

    state.fill(4.0)
    graph.run({"input": input_array, "output": output_array})
    np.testing.assert_allclose(output_array.to_numpy(), 4.0 * values)
    tree.destroy()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_compiled_graph_fixed_state_rejects_sparse_field():
    size = 4
    fields = ti.FieldsBuilder()
    values = ti.field(ti.f32)
    fields.pointer(ti.i, size).place(values)
    tree = fields.finalize()
    topology = ti.ndarray(ti.i32, shape=size)
    topology.from_numpy(np.arange(size, dtype=np.int32))

    @ti.kernel
    def apply_sparse_state(
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(size):
            y[index] = values[index] * x[topology_data[index]]

    topology_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "topology", ti.i32, ndim=1
    )
    input_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "input", ti.f32, ndim=1
    )
    output_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1
    )
    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        apply_sparse_state, topology_arg, input_arg, output_arg
    )
    with pytest.raises(RuntimeError, match="root-dense"):
        ti.linalg.LinearOperator.from_graph(
            builder.compile(),
            size,
            topology={"topology": topology},
            state={"values": values},
        )
    tree.destroy()


@pytest.mark.parametrize("provider_mode", ("legacy_square", "generic_square"))
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_compiled_graph_fixed_state_rejects_sparse_sibling_in_dense_tree(
    provider_mode,
):
    size = 4
    fields = ti.FieldsBuilder()
    dense_decoy = ti.field(ti.f32)
    sparse_actual = ti.field(ti.f32)
    fields.dense(ti.i, size).place(dense_decoy)
    fields.pointer(ti.i, size).place(sparse_actual)
    tree = fields.finalize()
    topology = ti.ndarray(ti.i32, shape=size)
    topology.from_numpy(np.arange(size, dtype=np.int32))

    @ti.kernel
    def apply_sparse_sibling(
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(size):
            y[index] = (
                sparse_actual[index] * x[topology_data[index]]
            )

    topology_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "topology", ti.i32, ndim=1
    )
    input_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "input", ti.f32, ndim=1
    )
    output_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1
    )
    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        apply_sparse_sibling, topology_arg, input_arg, output_arg
    )
    operator_size = (
        size if provider_mode == "legacy_square" else (size, size)
    )
    with pytest.raises(
        RuntimeError, match="sparse or dynamic SNodes"
    ):
        ti.linalg.LinearOperator.from_graph(
            builder.compile(),
            operator_size,
            topology={"topology": topology},
            # The dependency ABI is tree-granular, so this dense sibling used
            # to mask the sparse field actually accessed by the kernel.
            state={"dense_decoy": dense_decoy},
        )
    tree.destroy()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_compiled_kernel_action_generation_can_release_after_reset():
    operator = _diagonal_operator([1.0, 2.0, 3.0, 4.0])
    input_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "input", ti.f32, ndim=1
    )
    output_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1
    )
    action = operator.graph_action(input_arg, output_arg)
    escaped_executable = action.compile()
    escaped_record = escaped_executable._record

    ti.reset()
    del action, escaped_executable, escaped_record
    gc.collect()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_compiled_graph_action_fails_closed_after_reset():
    size = 4
    topology = ti.ndarray(ti.i32, shape=size)
    topology.from_numpy(np.arange(size, dtype=np.int32))

    @ti.kernel
    def apply(
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        source: ti.types.ndarray(dtype=ti.f32, ndim=1),
        destination: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(size):
            destination[index] = source[topology_data[index]]

    topology_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "topology", ti.i32, ndim=1
    )
    input_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "input", ti.f32, ndim=1
    )
    output_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1
    )
    provider_builder = ti.graph.GraphBuilder()
    provider_builder.dispatch(
        apply, topology_arg, input_arg, output_arg
    )
    operator = ti.linalg.LinearOperator.from_graph(
        provider_builder.compile(),
        (size, size),
        topology={"topology": topology},
    )
    action = operator.graph_action(input_arg, output_arg)
    escaped_executable = action.compile()
    escaped_record = escaped_executable._record
    outer_builder = ti.graph.GraphBuilder()
    outer_builder.append_native(action)
    graph = outer_builder.compile()
    input_array = ti.ndarray(ti.f32, shape=size)
    output_array = ti.ndarray(ti.f32, shape=size)

    ti.reset()
    with pytest.raises(ti.TaichiRuntimeError, match="after ti.reset"):
        operator.apply(input_array, out=output_array)
    with pytest.raises(
        ti.TaichiRuntimeError, match="compiled before ti.reset"
    ):
        graph.run({"input": input_array, "output": output_array})

    # A native action may be compiled before it is appended to an outer Graph.
    # Its provider generation can therefore outlive runtime invalidation. Late
    # release must defer to Program teardown instead of dereferencing the old
    # Program pointer.
    del graph, action, escaped_executable, escaped_record
    gc.collect()
