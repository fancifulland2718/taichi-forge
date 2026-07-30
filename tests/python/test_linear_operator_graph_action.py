import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


def _diagonal_operator(values):
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
        apply_diagonal, size, topology, numeric=numeric
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
    with pytest.raises(ti.TaichiRuntimeError, match="generation changed"):
        graph.run({"input": input_array, "output": output_array})

    rebuilt = _operator_graph(operator)
    rebuilt.run({"input": input_array, "output": output_array})
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
    for graph, input_value, output_value in (
        (forward, forward_input, forward_output),
        (adjoint, adjoint_input, adjoint_output),
    ):
        with pytest.raises(ti.TaichiRuntimeError, match="generation changed"):
            graph.run({"input": input_value, "output": output_value})

    rebuilt_adjoint = _operator_graph(operator, adjoint=True)
    rebuilt_adjoint.run(
        {"input": adjoint_input, "output": adjoint_output}
    )
    np.testing.assert_allclose(
        adjoint_output.to_numpy().reshape(-1),
        2.0 * matrix.T @ adjoint_values,
    )


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
