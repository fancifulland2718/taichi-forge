import gc

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
    with pytest.raises(ti.TaichiRuntimeError, match="generation changed"):
        graph.run({"input": input_array, "output": output_array})


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
