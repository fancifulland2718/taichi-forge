import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils


def _compiled_identity(size):
    topology = ti.ndarray(ti.i32, shape=size)
    topology.from_numpy(np.arange(size, dtype=np.int32))

    @ti.kernel
    def apply_identity(
        active_size: ti.i32,
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(active_size):
            y[index] = x[topology_data[index]]

    return ti.linalg.LinearOperator.from_kernel(
        apply_identity,
        size,
        topology,
        traits=ti.linalg.OperatorTraits.spd(),
    )


def _compiled_graph_identity(size, *, multi_dispatch=False):
    topology = ti.ndarray(ti.i32, shape=size)
    topology.from_numpy(np.arange(size, dtype=np.int32))

    @ti.kernel
    def apply_identity(
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(size):
            y[index] = x[topology_data[index]]

    topology_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "topology", ti.i32, ndim=1)
    input_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "input", ti.f32, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1)
    builder = ti.graph.GraphBuilder()
    workspace = None
    if multi_dispatch:
        temporary = ti.ndarray(ti.f32, shape=size)
        temporary_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "temporary", ti.f32, ndim=1)
        builder.dispatch(apply_identity, topology_arg, input_arg, temporary_arg)
        builder.dispatch(apply_identity, topology_arg, temporary_arg, output_arg)
        workspace = {"temporary": temporary}
    else:
        builder.dispatch(apply_identity, topology_arg, input_arg, output_arg)
    return ti.linalg.LinearOperator.from_graph(
        builder.compile(),
        size,
        topology={"topology": topology},
        workspace=workspace,
        traits=ti.linalg.OperatorTraits.spd(),
    )


def _compiled_graph_stencil_and_jacobi(size):
    topology = ti.ndarray(ti.i32, shape=size)
    topology.from_numpy(np.arange(size, dtype=np.int32))
    diagonal_values = 2.5 + 0.25 * np.sin(np.linspace(0.0, 5.0, size, dtype=np.float32))
    diagonal = ti.ndarray(ti.f32, shape=size)
    diagonal.from_numpy(diagonal_values.astype(np.float32))

    @ti.kernel
    def stencil_diagonal(
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        diagonal_data: ti.types.ndarray(dtype=ti.f32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        temporary: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for slot in range(size):
            index = topology_data[slot]
            temporary[index] = diagonal_data[index] * x[index]

    @ti.kernel
    def stencil_left(
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        temporary: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for slot in range(size):
            index = topology_data[slot]
            if index > 0:
                temporary[index] -= 0.9 * x[index - 1]

    @ti.kernel
    def stencil_right(
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        temporary: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for slot in range(size):
            index = topology_data[slot]
            y[index] = temporary[index]
            if index + 1 < size:
                y[index] -= 0.9 * x[index + 1]

    @ti.kernel
    def jacobi_apply(
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        diagonal_data: ti.types.ndarray(dtype=ti.f32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        temporary: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for slot in range(size):
            index = topology_data[slot]
            temporary[index] = x[index] / diagonal_data[index]

    @ti.kernel
    def copy_vector(
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        temporary: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for slot in range(size):
            index = topology_data[slot]
            y[index] = temporary[index]

    topology_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "topology", ti.i32, ndim=1)
    diagonal_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "diagonal", ti.f32, ndim=1)
    input_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "input", ti.f32, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1)
    temporary_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "temporary", ti.f32, ndim=1)

    operator_temporary = ti.ndarray(ti.f32, shape=size)
    operator_builder = ti.graph.GraphBuilder()
    operator_builder.dispatch(
        stencil_diagonal,
        topology_arg,
        diagonal_arg,
        input_arg,
        temporary_arg,
    )
    operator_builder.dispatch(stencil_left, topology_arg, input_arg, temporary_arg)
    operator_builder.dispatch(
        stencil_right,
        topology_arg,
        input_arg,
        temporary_arg,
        output_arg,
    )
    operator = ti.linalg.LinearOperator.from_graph(
        operator_builder.compile(),
        size,
        topology={"topology": topology},
        numeric={"diagonal": diagonal},
        workspace={"temporary": operator_temporary},
        traits=ti.linalg.OperatorTraits.spd(),
    )

    preconditioner_temporary = ti.ndarray(ti.f32, shape=size)
    preconditioner_builder = ti.graph.GraphBuilder()
    preconditioner_builder.dispatch(
        jacobi_apply,
        topology_arg,
        diagonal_arg,
        input_arg,
        temporary_arg,
    )
    preconditioner_builder.dispatch(copy_vector, topology_arg, temporary_arg, output_arg)
    preconditioner = ti.linalg.LinearOperator.from_graph(
        preconditioner_builder.compile(),
        size,
        topology={"topology": topology},
        numeric={"diagonal": diagonal},
        workspace={"temporary": preconditioner_temporary},
        traits=ti.linalg.OperatorTraits.spd(),
    )
    return operator, preconditioner, diagonal_values.astype(np.float32)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_dense_scalar_field_apply_solve_and_staging_reuse():
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    source = ti.field(ti.f32, shape=(2, 3))
    applied = ti.field(ti.f32, shape=(2, 3))
    solution = ti.field(ti.f32, shape=(2, 3))
    source.from_numpy(values)

    operator = _compiled_identity(values.size)
    assert operator.apply(source, out=applied) is applied
    assert operator.apply(source, out=applied) is applied
    np.testing.assert_array_equal(applied.to_numpy(), values)
    operator_stats = operator.statistics()["vector_io"]
    assert operator.capabilities.dense_storage_operands
    assert operator_stats["staging_buffer_builds"] == 0
    assert operator_stats["implicit_view_builds"] == 2
    assert operator_stats["implicit_view_reuses"] == 2
    assert operator_stats["transfer_plan_builds"] == 0
    assert operator_stats["transfer_native_submissions"] == 0
    assert operator_stats["transfer_graph_submissions"] == 0
    assert operator_stats["pack_calls"] == 0
    assert operator_stats["unpack_calls"] == 0
    assert operator_stats["direct_bindings"] == 4
    assert operator_stats["direct_dense_field_submissions"] == 2
    assert operator_stats["completion_syncs"] == 0
    assert operator_stats["coalesced_operator_syncs"] == 0
    assert operator_stats["last_input_execution_mode"] == "direct_contiguous"
    assert operator_stats["last_output_execution_mode"] == "direct_contiguous"

    plan = ti.linalg.experimental.SolvePlan(
        operator,
        method="cg",
        max_iterations=8,
        atol=1e-6,
    )
    first = plan.solve(source, out=solution)
    second = plan.solve(source, out=solution)
    assert first.converged and second.converged
    assert second.solution is solution
    np.testing.assert_allclose(solution.to_numpy(), values, rtol=1e-6)

    stats = plan.statistics()
    vector_stats = stats["vector_io"]
    direct_solve = stats["execution_capabilities"]["direct_dense_field_solve"]["selected"]
    if direct_solve:
        assert vector_stats["staging_buffer_builds"] == 0
        assert vector_stats["staging_buffer_reuses"] == 0
        assert vector_stats["transfer_plan_builds"] == 0
        assert vector_stats["transfer_plan_reuses"] == 0
        assert vector_stats["transfer_native_submissions"] == 0
        assert vector_stats["transfer_graph_submissions"] == 0
        assert vector_stats["pack_calls"] == 0
        assert vector_stats["unpack_calls"] == 0
        assert vector_stats["completion_syncs"] == 0
        assert vector_stats["direct_graph_solve_submissions"] == 2
        assert vector_stats["direct_graph_solve_full_boundary_submissions"] == 2
        assert vector_stats["direct_dense_field_submissions"] == 2
        assert vector_stats["direct_graph_solve_field_bindings"] == 4
    else:
        assert vector_stats["staging_buffer_builds"] == 2
        assert vector_stats["staging_buffer_reuses"] == 2
        assert vector_stats["transfer_plan_builds"] == 2
        assert vector_stats["transfer_plan_reuses"] == 2
        assert vector_stats["transfer_native_submissions"] == (0 if impl.current_cfg().arch == ti.cuda else 4)
        assert vector_stats["transfer_graph_submissions"] == (4 if impl.current_cfg().arch == ti.cuda else 0)
        assert vector_stats["pack_calls"] == 2
        assert vector_stats["unpack_calls"] == 2
        assert vector_stats["completion_syncs"] == 2
    assert vector_stats["implicit_view_builds"] == 2
    assert stats["execution_capabilities"]["vector_io"]["dense_field"]["execution_mode"] == "provider_qualified"
    capabilities = ti.linalg.vector_io_capabilities()
    assert capabilities["ndarray"]["zero_copy"] is True
    assert capabilities["dense_field"]["zero_copy"] is False
    assert capabilities["dense_field"]["zero_copy_condition"] == (
        "canonical full field or contiguous scalar-flat range and provider " "dense_storage_operands"
    )
    assert capabilities["dense_field"]["solve_direct_binding_semantics"] == (
        "Graph-fused boundary copy into plan-owned iterative storage"
    )
    assert capabilities["dense_field"]["value_host_transfer"] is False
    assert capabilities["dense_field"]["conversion_scope"] == ("apply_or_solve_boundary_only")
    assert capabilities["dense_field"]["conversion_submission"] == ("native_bulk_copy_or_compiled_graph_replay")

    device_capability = plan.execution_capabilities()["device_convergent"]
    if device_capability["supported"]:
        device_plan = ti.linalg.experimental.SolvePlan(
            operator,
            method="cg",
            max_iterations=8,
            atol=1e-6,
            execution_policy="device_convergent",
        )
        device_result = device_plan.solve(source, out=solution)
        assert device_result.converged
        np.testing.assert_allclose(solution.to_numpy(), values, rtol=1e-6)
        device_stats = device_plan.statistics()
        assert device_stats["identity"]["solver_control_path"] == ("generic_structured_graph")
        assert device_stats["identity"]["backend_family"] == (
            "vulkan" if impl.current_cfg().arch == ti.vulkan else "cuda"
        )
        device_vector_stats = device_stats["vector_io"]
        assert device_vector_stats["staging_buffer_builds"] == 0
        assert device_vector_stats["pack_calls"] == 0
        assert device_vector_stats["unpack_calls"] == 0
        assert device_vector_stats["transfer_graph_submissions"] == 0
        assert device_vector_stats["transfer_native_submissions"] == 0
        assert device_vector_stats["direct_graph_solve_submissions"] == 1
        assert device_vector_stats["direct_dense_field_submissions"] == 1
        device_operations = device_stats["operations"]
        assert device_operations["host_scalar_readbacks"] == 1
        assert device_operations["host_synchronizations"] == 1
        assert device_operations["terminal_submission_observations"] == 1
        assert device_operations["solver_chunk_submissions"] == 1
        assert device_operations["structured_control_observation_batches"] == 0
        if impl.current_cfg().arch == ti.vulkan:
            assert device_operations["last_structured_control_lowering"] == ("vulkan_compact_indirect")
            assert device_operations["last_logical_iterations"] == 1
            assert device_operations["last_executed_iterations"] == 8
        else:
            assert device_operations["last_structured_control_lowering"] == ("cuda_conditional_graph")
            assert device_operations["last_logical_iterations"] == 1
            assert device_operations["last_executed_iterations"] == 1
        with pytest.raises(
            ti.TaichiRuntimeError,
            match="unavailable after asynchronous submission",
        ):
            device_plan._solver._graph.control_flow_stats()
        graph_memory = device_plan._solver._graph.execution_stats().memory
        assert graph_memory.workspace_lanes_busy == 0
        assert graph_memory.workspace_lane_acquisitions == 1
        assert graph_memory.workspace_lane_waits == 0

    volume_values = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    volume_source = ti.field(ti.f32, shape=(2, 2, 2))
    volume_output = ti.field(ti.f32, shape=(2, 2, 2))
    volume_source.from_numpy(volume_values)
    _compiled_identity(volume_values.size).apply(volume_source, out=volume_output)
    np.testing.assert_array_equal(volume_output.to_numpy(), volume_values)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
@pytest.mark.parametrize("method", ["cg", "pcg"])
def test_solve_plan_complete_graph_action_terminal_and_workspace(method):
    size = 16
    operator = _compiled_identity(size)
    options = {"preconditioner": operator} if method == "pcg" else {}
    plan = ti.linalg.experimental.SolvePlan(
        operator,
        method=method,
        max_iterations=8,
        atol=1e-6,
        **options,
    )
    rhs_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "solve_rhs", ti.f32, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "solve_output", ti.f32, ndim=1)
    action = plan.graph_action(rhs_arg, output_arg, name=f"recorded_{method}")
    builder = ti.graph.GraphBuilder()
    builder.append_native(action)
    graph = builder.compile()
    initial_action_stats = action.statistics()
    assert initial_action_stats["compiled_executables"] == 1
    assert initial_action_stats["enclosing_graph_submissions"] == 0

    rhs = ti.ndarray(ti.f32, shape=size)
    output = ti.ndarray(ti.f32, shape=size)
    expected = np.linspace(-2.0, 3.0, size, dtype=np.float32)
    rhs.from_numpy(expected)
    packet = action.allocate_terminal()
    ticket = graph.submit(
        {
            "solve_rhs": rhs,
            "solve_output": output,
            **packet.arguments,
        },
        telemetry=True,
    )
    submitted_stats = action.statistics()
    assert submitted_stats["enclosing_graph_submissions"] == 1
    if impl.current_cfg().arch == ti.cpu:
        assert submitted_stats["observed_completions"] == 1
    else:
        assert submitted_stats["observed_completions"] in (0, 1)
    ticket.wait()
    ticket.wait()
    assert action.statistics()["observed_completions"] == 1

    snapshot = packet.snapshot()
    assert packet.snapshot() == snapshot
    assert snapshot.converged
    assert snapshot.iterations == 1
    assert snapshot.residual_norm == pytest.approx(0.0, abs=1e-7)
    np.testing.assert_allclose(output.to_numpy(), expected, rtol=1e-6)
    report = graph.execution_stats()
    assert report.memory.persistent_internal_storage_bytes > size * 4
    assert report.memory.internal_storage_exclusive
    telemetry = ticket.telemetry()
    assert tuple(region.path_id for region in telemetry.regions) == (f"recorded_{method}",)
    assert telemetry.regions[0].logical_invocations == 1
    assert telemetry.regions[0].logical_iterations == 1
    action_stats = action.statistics()
    assert action_stats["terminal_snapshots"] == 1
    assert action_stats["terminal_iteration_sum"] == 1
    plan_stats = plan.graph_action_statistics()
    assert plan_stats["actions_created"] == 1
    assert plan_stats["enclosing_graph_submissions"] == 1
    assert plan_stats["observed_completions"] == 1
    assert plan_stats["terminal_snapshots"] == 1
    assert plan.statistics()["graph_actions"] == plan_stats


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_solve_plan_graph_action_attributes_synchronous_graph_run():
    size = 8
    plan = ti.linalg.experimental.SolvePlan(_compiled_identity(size), method="cg", max_iterations=4, atol=1e-6)
    rhs_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "sync_rhs", ti.f32, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "sync_output", ti.f32, ndim=1)
    action = plan.graph_action(rhs_arg, output_arg, name="sync_recorded_cg")
    builder = ti.graph.GraphBuilder()
    builder.append_native(action)
    graph = builder.compile()
    rhs = ti.ndarray(ti.f32, shape=size)
    output = ti.ndarray(ti.f32, shape=size)
    rhs.from_numpy(np.arange(size, dtype=np.float32) + 1)
    packet = action.allocate_terminal()

    graph.run(
        {
            "sync_rhs": rhs,
            "sync_output": output,
            **packet.arguments,
        }
    )

    before_snapshot = action.statistics()
    assert before_snapshot["enclosing_graph_submissions"] == 1
    assert before_snapshot["observed_completions"] == 1
    assert before_snapshot["terminal_snapshots"] == 0
    assert packet.snapshot().converged
    after_snapshot = action.statistics()
    assert after_snapshot["terminal_snapshots"] == 1
    assert after_snapshot["terminal_iteration_sum"] == 1


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_solve_plan_graph_action_statistics_reject_stale_runtime():
    size = 4
    plan = ti.linalg.experimental.SolvePlan(_compiled_identity(size), method="cg", max_iterations=2, atol=1e-6)
    rhs_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "stale_rhs", ti.f32, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "stale_output", ti.f32, ndim=1)
    plan.graph_action(rhs_arg, output_arg)

    ti.reset()
    ti.init(arch=ti.cpu)

    with pytest.raises(ti.TaichiRuntimeError, match="after ti.reset"):
        plan.graph_action_statistics()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_solve_plan_submit_owns_terminal_ticket_and_workspace_lane():
    size = 64
    options = {}
    if impl.current_cfg().arch in (ti.cuda, ti.vulkan):
        options["execution_policy"] = "device_convergent"
    plan = ti.linalg.experimental.SolvePlan(
        _compiled_identity(size),
        method="cg",
        max_iterations=8,
        atol=1e-6,
        submission_workspace_lanes=2,
        submission_workspace_saturation="raise",
        **options,
    )
    expected = np.linspace(-2.0, 3.0, size, dtype=np.float32)
    rhs = ti.field(ti.f32, shape=size)
    output = ti.field(ti.f32, shape=size)
    rhs.from_numpy(expected)
    output.fill(0.0)

    requested_lane = 0 if impl.current_cfg().arch == ti.cpu else 1
    submission = plan.submit(
        rhs,
        out=output,
        telemetry=True,
        workspace_lane=requested_lane,
    )
    assert isinstance(submission, ti.linalg.experimental.SolvePlanSubmission)
    assert submission.workspace_lane == requested_lane
    before_result = plan.submission_statistics()
    assert before_result["submit_calls"] == 1
    assert before_result["submit_successes"] == 1
    assert before_result["terminal_materializations"] == 0
    assert before_result["configured_workspace_lane_capacity"] == 2
    if impl.current_cfg().arch == ti.cpu:
        assert before_result["graphs_materialized"] == 0
        assert before_result["workspace_lane_capacity"] == 1
        assert before_result["native_completed_results"] == 1
    else:
        assert before_result["graphs_materialized"] == 1
        assert before_result["workspace_lane_capacity"] == 2
        assert before_result["native_completed_results"] == 0

    submission.wait()
    assert plan.submission_statistics()["terminal_materializations"] == 0
    result = submission.result()
    assert submission.result() is result
    assert result.solution is output
    assert result.converged
    assert result.iterations == 1
    np.testing.assert_allclose(output.to_numpy(), expected, rtol=1e-6)
    telemetry = submission.telemetry()
    if impl.current_cfg().arch == ti.cpu:
        assert telemetry is None
    else:
        assert tuple(region.path_id for region in telemetry.regions) == ("cg_submit_zero",)

    initial = ti.ndarray(ti.f32, shape=size)
    initial.fill(0.25)
    second = plan.submit(rhs, initial_guess=initial, workspace_lane=0)
    second_result = second.result()
    assert second_result.converged
    np.testing.assert_allclose(second_result.solution.to_numpy(), expected, rtol=1e-6)
    final_stats = plan.statistics()["submission"]
    assert final_stats["submit_calls"] == 2
    assert final_stats["submit_successes"] == 2
    assert final_stats["submit_failures"] == 0
    assert final_stats["telemetry_requests"] == 1
    if impl.current_cfg().arch == ti.cpu:
        assert final_stats["execution_path"] == "native_cpu_completed"
        assert final_stats["terminal_materializations"] == 0
        assert final_stats["native_completed_results"] == 2
        assert final_stats["graphs_materialized"] == 0
        assert final_stats["variants"] == {}
        assert final_stats["persistent_internal_storage_bytes"] == 0
    else:
        assert final_stats["execution_path"] == "cached_graph_submission"
        assert final_stats["terminal_materializations"] == 2
        assert final_stats["native_completed_results"] == 0
        assert final_stats["graphs_materialized"] == 2
        assert set(final_stats["variants"]) == {
            "zero_initial_guess",
            "with_initial_guess",
        }
        assert final_stats["persistent_internal_storage_bytes"] > size * 4


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_solve_plan_submit_workspace_configuration_fails_closed():
    operator = _compiled_identity(4)
    with pytest.raises(RuntimeError, match="submission_workspace_lanes must be between"):
        ti.linalg.experimental.SolvePlan(operator, submission_workspace_lanes=0)
    with pytest.raises(RuntimeError, match="submission_workspace_saturation"):
        ti.linalg.experimental.SolvePlan(operator, submission_workspace_saturation="drop")


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_solve_plan_submit_requires_device_convergent_policy_on_gpu():
    plan = ti.linalg.experimental.SolvePlan(
        _compiled_identity(8),
        method="cg",
        max_iterations=8,
        atol=1e-6,
        execution_policy="host_check_every_k",
        check_interval=4,
    )
    submission = plan.submission_statistics()
    assert not submission["qualified"]
    assert not submission["asynchronous"]
    assert submission["unsupported_reason"] == "device_convergent_policy_required"
    rhs = ti.ndarray(ti.f32, shape=8)
    rhs.fill(1.0)
    with pytest.raises(RuntimeError, match="device_convergent_policy_required"):
        plan.submit(rhs)


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_compiled_graph_provider_pcg_uses_recordable_device_control():
    size = 32
    operator = _compiled_graph_identity(size, multi_dispatch=True)
    preconditioner = _compiled_graph_identity(size, multi_dispatch=True)
    plan = ti.linalg.experimental.SolvePlan(
        operator,
        method="pcg",
        preconditioner=preconditioner,
        max_iterations=8,
        atol=1e-6,
    )

    capabilities = plan.execution_capabilities()
    assert capabilities["default_execution_policy"] == "device_convergent"
    assert capabilities["device_convergent"]["supported"]
    assert capabilities["device_convergent"]["provider_qualified"]
    assert capabilities["device_convergent"]["automatic_selection_qualified"]

    expected = np.linspace(-2.0, 3.0, size, dtype=np.float32)
    rhs = ti.field(ti.f32, shape=size)
    output = ti.field(ti.f32, shape=size)
    rhs.from_numpy(expected)
    output.fill(0.0)
    result = plan.solve(rhs, out=output)

    assert result.converged
    assert result.iterations == 1
    np.testing.assert_allclose(output.to_numpy(), expected, rtol=1e-6)
    stats = plan.statistics()
    assert stats["identity"]["solver_execution_policy"] == "device_convergent"
    assert stats["identity"]["solver_control_path"] == "generic_structured_graph"
    assert stats["identity"]["preconditioner_method"] == "linear_operator"
    assert stats["operations"]["preconditioner_apply_calls"] == 2
    assert stats["operations"]["host_scalar_readbacks"] == 1
    assert stats["operations"]["terminal_submission_observations"] == (
        0 if impl.current_cfg().arch == ti.cpu else 1
    )
    assert stats["operations"]["structured_control_observation_batches"] == 0
    vector_stats = stats["vector_io"]
    assert vector_stats["pack_calls"] == 0
    assert vector_stats["unpack_calls"] == 0
    assert vector_stats["transfer_graph_submissions"] == 0
    assert vector_stats["transfer_native_submissions"] == 0
    assert vector_stats["direct_graph_solve_submissions"] == 1
    assert vector_stats["direct_graph_solve_full_boundary_submissions"] == 1
    assert vector_stats["direct_dense_field_submissions"] == 1
    if impl.current_cfg().arch == ti.vulkan:
        assert stats["operations"]["last_encoded_iterations"] == 8
        assert stats["operations"]["last_masked_iterations"] == 7
        assert stats["operations"]["last_window_sizes"] == [8]
    else:
        assert stats["operations"]["last_encoded_iterations"] == 1
        assert stats["operations"]["last_masked_iterations"] == 0


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_multidispatch_graph_pcg_submit_converges_and_reports_stop_position():
    size = 128
    operator, preconditioner, diagonal = _compiled_graph_stencil_and_jacobi(size)
    plan = ti.linalg.experimental.SolvePlan(
        operator,
        method="pcg",
        preconditioner=preconditioner,
        max_iterations=64,
        atol=1e-5,
        rtol=1e-5,
    )
    exact = np.sin(np.linspace(0.1, 3.0, size, dtype=np.float32))
    rhs_values = diagonal * exact
    rhs_values[1:] -= 0.9 * exact[:-1]
    rhs_values[:-1] -= 0.9 * exact[1:]
    rhs = ti.ndarray(ti.f32, shape=size)
    output = ti.ndarray(ti.f32, shape=size)
    rhs.from_numpy(rhs_values)

    submission = plan.submit(rhs, out=output, telemetry=True)
    result = submission.result()
    assert result.converged
    assert 2 < result.iterations < plan.max_iterations
    actual = output.to_numpy()
    np.testing.assert_allclose(actual, exact, rtol=4e-4, atol=4e-4)
    applied = diagonal * actual
    applied[1:] -= 0.9 * actual[:-1]
    applied[:-1] -= 0.9 * actual[1:]
    true_residual = float(np.linalg.norm(rhs_values - applied))
    assert true_residual <= max(plan.atol, plan.rtol * np.linalg.norm(rhs_values)) * 4

    telemetry = submission.telemetry()
    if impl.current_cfg().arch == ti.cpu:
        assert telemetry is None
    else:
        assert len(telemetry.regions) == 1
        region = telemetry.regions[0]
        assert region.path_id == "pcg_submit_zero"
        assert region.logical_iterations == result.iterations
        assert region.encoded_iterations >= region.logical_iterations
        assert region.masked_iterations == (region.encoded_iterations - region.logical_iterations)
    submission_stats = plan.submission_statistics()
    if impl.current_cfg().arch == ti.cpu:
        assert submission_stats["execution_path"] == "native_cpu_completed"
        assert submission_stats["terminal_materializations"] == 0
        assert submission_stats["native_completed_results"] == 1
        assert submission_stats["persistent_internal_storage_bytes"] == 0
    else:
        assert submission_stats["execution_path"] == "cached_graph_submission"
        assert submission_stats["terminal_materializations"] == 1
        assert submission_stats["native_completed_results"] == 0
        assert submission_stats["persistent_internal_storage_bytes"] > size * 4


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_solve_plan_graph_action_uses_independent_workspace_lanes():
    size = 4096
    plan = ti.linalg.experimental.SolvePlan(_compiled_identity(size), method="cg", max_iterations=8, atol=1e-6)
    rhs_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "rhs", ti.f32, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1)
    action = plan.graph_action(rhs_arg, output_arg, name="multi_lane_cg")
    builder = ti.graph.GraphBuilder()
    builder.append_native(action)
    graph = builder.compile(workspace_lanes=2, workspace_saturation="raise")

    first_rhs = ti.ndarray(ti.f32, shape=size)
    second_rhs = ti.ndarray(ti.f32, shape=size)
    first_output = ti.ndarray(ti.f32, shape=size)
    second_output = ti.ndarray(ti.f32, shape=size)
    first_expected = np.linspace(-1.0, 2.0, size, dtype=np.float32)
    second_expected = np.linspace(3.0, -2.0, size, dtype=np.float32)
    first_rhs.from_numpy(first_expected)
    second_rhs.from_numpy(second_expected)
    first_packet = action.allocate_terminal()
    second_packet = action.allocate_terminal()

    first_ticket = graph.submit(
        {
            "rhs": first_rhs,
            "output": first_output,
            **first_packet.arguments,
        }
    )
    one_lane_memory = graph.execution_stats().memory
    second_ticket = graph.submit(
        {
            "rhs": second_rhs,
            "output": second_output,
            **second_packet.arguments,
        }
    )
    assert first_ticket.workspace_lane == 0
    assert second_ticket.workspace_lane == 1
    first_ticket.wait()
    second_ticket.wait()

    assert first_packet.snapshot().converged
    assert second_packet.snapshot().converged
    np.testing.assert_allclose(first_output.to_numpy(), first_expected, rtol=1e-6)
    np.testing.assert_allclose(second_output.to_numpy(), second_expected, rtol=1e-6)
    two_lane_memory = graph.execution_stats().memory
    assert one_lane_memory.workspace_lane_capacity == 2
    assert one_lane_memory.workspace_lanes_materialized == 1
    assert two_lane_memory.workspace_lanes_materialized == 2
    assert two_lane_memory.persistent_internal_storage_bytes == (2 * one_lane_memory.persistent_internal_storage_bytes)
    assert two_lane_memory.workspace_lane_acquisitions == 2
    assert two_lane_memory.workspace_lane_waits == 0
    assert two_lane_memory.internal_storage_waits == 0
    assert two_lane_memory.workspace_lane_saturation_policy == "raise"

    class BusyCompletion:
        @staticmethod
        def done():
            return False

    lane_zero = graph._workspace_pool.primary
    saved_completion = lane_zero._exclusive_internal_completion
    lane_zero._exclusive_internal_completion = BusyCompletion()
    try:
        with pytest.raises(RuntimeError, match="workspace lanes are occupied"):
            graph.submit(
                {
                    "rhs": first_rhs,
                    "output": first_output,
                    **first_packet.arguments,
                },
                workspace_lane=0,
            )
    finally:
        lane_zero._exclusive_internal_completion = saved_completion

    pinned_packet = action.allocate_terminal()
    pinned_output = ti.ndarray(ti.f32, shape=size)
    pinned_ticket = graph.submit(
        {
            "rhs": first_rhs,
            "output": pinned_output,
            **pinned_packet.arguments,
        },
        workspace_lane=1,
    )
    assert pinned_ticket.workspace_lane == 1
    pinned_ticket.wait()
    assert pinned_packet.snapshot().converged
    np.testing.assert_allclose(pinned_output.to_numpy(), first_expected, rtol=1e-6)
    final_memory = graph.execution_stats().memory
    assert final_memory.workspace_lanes_materialized == 2
    assert final_memory.workspace_lane_saturation_errors == 1
    assert final_memory.persistent_internal_storage_bytes == (two_lane_memory.persistent_internal_storage_bytes)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_solve_plan_graph_action_runs_inside_nested_single_ticket_loop():
    size = 16
    plan = ti.linalg.experimental.SolvePlan(_compiled_identity(size), method="cg", max_iterations=8, atol=1e-6)
    rhs_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "nested_rhs", ti.f32, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "nested_output", ti.f32, ndim=1)
    action = plan.graph_action(rhs_arg, output_arg, name="nested_cg")
    terminal = action.terminal

    outer_predicate = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "outer_predicate", ti.i32, ndim=0)
    outer_counter = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "outer_counter", ti.i32, ndim=0)
    outer_target = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "outer_target", ti.i32)
    stop_trace = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "solve_stop_trace", ti.i32, ndim=1)

    @ti.kernel
    def evaluate_outer(
        counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        target: ti.i32,
    ):
        predicate[None] = int(counter[None] < target)

    @ti.kernel
    def consume_solve_terminal(
        counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        terminal_state: ti.types.ndarray(dtype=ti.i32, ndim=1),
        stops: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        if predicate[None] != 0 and terminal_state[3] == 1:
            stops[counter[None]] = terminal_state[1]
            counter[None] += int(terminal_state[0] == 2)

    builder = ti.graph.GraphBuilder()
    outer_condition = builder.create_sequential()
    outer_condition.dispatch(evaluate_outer, outer_counter, outer_predicate, outer_target)
    outer_body = builder.create_sequential()
    outer_body.append_native(action)
    outer_body.dispatch(
        consume_solve_terminal,
        outer_counter,
        outer_predicate,
        terminal.state,
        stop_trace,
    )
    builder.while_loop(
        outer_condition,
        outer_body,
        predicate=outer_predicate,
        control_inputs=(outer_counter, outer_target),
        carried_state=(
            rhs_arg,
            output_arg,
            outer_counter,
            terminal.state,
            terminal.metrics,
            stop_trace,
        ),
        counter=outer_counter,
        max_iterations=3,
        name="outer_newton",
    )
    graph = builder.compile()
    storage = ti.field(ti.f32, shape=48)
    expected = np.linspace(-1.0, 2.0, size, dtype=np.float32)
    host = np.zeros(48, dtype=np.float32)
    host[2 : 2 + size] = expected
    storage.from_numpy(host)
    rhs = ti.linalg.vector_view(storage, offset=2, length=size)
    output = ti.linalg.vector_view(storage, offset=28, length=size)
    predicate = ti.ndarray(ti.i32, shape=())
    counter = ti.ndarray(ti.i32, shape=())
    stops = ti.ndarray(ti.i32, shape=3)
    predicate.fill(0)
    counter.fill(0)
    stops.fill(0)
    packet = action.allocate_terminal()

    ticket = graph.submit(
        {
            "nested_rhs": rhs,
            "nested_output": output,
            "outer_predicate": predicate,
            "outer_counter": counter,
            "outer_target": 3,
            "solve_stop_trace": stops,
            **packet.arguments,
        },
        telemetry=True,
    )
    ticket.wait()

    assert counter.to_numpy()[()] == 3
    assert tuple(stops.to_numpy()) == (1, 1, 1)
    snapshot = packet.snapshot()
    assert snapshot.converged
    assert snapshot.iterations == 1
    telemetry = ticket.telemetry()
    assert tuple(region.path_id for region in telemetry.regions) == (
        "outer_newton",
        "outer_newton/body/nested_cg",
    )
    assert tuple(region.logical_invocations for region in telemetry.regions) == (
        1,
        3,
    )
    assert tuple(region.logical_iterations for region in telemetry.regions) == (
        3,
        1,
    )
    action_stats = action.statistics()
    assert action_stats["enclosing_graph_submissions"] == 1
    assert action_stats["observed_completions"] == 1
    assert action_stats["terminal_snapshots"] == 1
    assert action_stats["terminal_iteration_sum"] == 1
    assert action_stats["terminal_scope"] == "last_action_invocation_in_packet"
    np.testing.assert_allclose(storage.to_numpy()[28 : 28 + size], expected, rtol=1e-6)
    memory = graph.execution_stats().memory
    assert memory.internal_storage_exclusive
    assert memory.persistent_internal_storage_bytes > size * 4


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_solve_plan_graph_action_binds_disjoint_field_ranges_directly():
    size = 16
    storage = ti.field(ti.f32, shape=48)
    expected = np.linspace(0.5, 2.0, size, dtype=np.float32)
    host = np.zeros(48, dtype=np.float32)
    host[3 : 3 + size] = expected
    storage.from_numpy(host)
    rhs = ti.linalg.vector_view(storage, offset=3, length=size)
    output = ti.linalg.vector_view(storage, offset=27, length=size)

    plan = ti.linalg.experimental.SolvePlan(_compiled_identity(size), method="cg", max_iterations=8, atol=1e-6)
    rhs_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "rhs", ti.f32, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1)
    action = plan.graph_action(rhs_arg, output_arg)
    builder = ti.graph.GraphBuilder()
    builder.append_native(action)
    graph = builder.compile()
    packet = action.allocate_terminal()
    ticket = graph.submit({"rhs": rhs, "output": output, **packet.arguments})
    ticket.wait()

    assert packet.snapshot().converged
    np.testing.assert_allclose(storage.to_numpy()[27 : 27 + size], expected, rtol=1e-6)


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
@pytest.mark.parametrize("method", ["cg", "pcg"])
def test_graph_krylov_composition_binds_full_fields_and_initial_guess_directly(
    method,
):
    size = 16
    values = np.linspace(0.25, 4.0, size, dtype=np.float32)
    rhs = ti.field(ti.f32, shape=size)
    solution = ti.field(ti.f32, shape=size)
    initial = ti.field(ti.f32, shape=size)
    rhs.from_numpy(values)
    initial.from_numpy(values)

    base = _compiled_identity(size)
    operator = 0.5 * base + 0.5 * base
    options = {}
    if method == "pcg":
        options["preconditioner"] = base
    plan = ti.linalg.experimental.SolvePlan(
        operator,
        method=method,
        max_iterations=8,
        atol=1e-6,
        execution_policy="device_convergent",
        **options,
    )

    cold = plan.solve(rhs, out=solution)
    warm = plan.solve(rhs, initial_guess=initial, out=solution)
    solution.from_numpy(values)
    aliased_initial = plan.solve(rhs, initial_guess=solution, out=solution)

    assert cold.converged and cold.iterations == 1
    assert warm.converged and warm.iterations == 0
    assert aliased_initial.converged and aliased_initial.iterations == 0
    np.testing.assert_allclose(solution.to_numpy(), values, rtol=1e-6)

    stats = plan.statistics()["vector_io"]
    assert stats["staging_buffer_builds"] == 0
    assert stats["pack_calls"] == 0
    assert stats["unpack_calls"] == 0
    assert stats["transfer_graph_submissions"] == 0
    assert stats["transfer_native_submissions"] == 0
    assert stats["direct_graph_solve_submissions"] == 3
    assert stats["direct_graph_solve_full_boundary_submissions"] == 3
    assert stats["direct_graph_solve_field_bindings"] == 8
    assert stats["direct_graph_solve_initial_guess_bindings"] == 2
    assert stats["direct_dense_field_submissions"] == 3

    with pytest.raises(RuntimeError, match="RHS and output may not alias"):
        plan.solve(rhs, out=rhs)


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_graph_krylov_indexed_views_remain_explicitly_staged():
    source = ti.field(ti.f32, shape=6)
    output = ti.field(ti.f32, shape=6)
    source.from_numpy(np.arange(6, dtype=np.float32))
    output.fill(-1)
    indices = ti.ndarray(ti.i32, shape=3)
    indices.from_numpy(np.asarray([5, 1, 3], dtype=np.int32))
    source_view = ti.linalg.vector_view(source, indices=indices)
    output_view = ti.linalg.vector_view(output, indices=indices)

    plan = ti.linalg.experimental.SolvePlan(
        _compiled_identity(3),
        method="cg",
        max_iterations=8,
        atol=1e-6,
        execution_policy="device_convergent",
    )
    result = plan.solve(source_view, out=output_view)
    assert result.converged
    np.testing.assert_array_equal(
        output.to_numpy(),
        np.asarray([-1, 1, -1, 3, -1, 5], dtype=np.float32),
    )

    stats = plan.statistics()["vector_io"]
    assert stats["staging_buffer_builds"] == 2
    assert stats["pack_calls"] == 1
    assert stats["unpack_calls"] == 1
    assert stats["direct_graph_solve_submissions"] == 0
    assert stats["direct_graph_solve_full_boundary_submissions"] == 0
    assert stats["direct_dense_field_submissions"] == 0


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_graph_krylov_cached_direct_field_rejects_destroyed_tree():
    size = 4
    rhs = ti.field(ti.f32)
    solution = ti.field(ti.f32)
    builder = ti.FieldsBuilder()
    builder.dense(ti.i, size).place(rhs)
    builder.dense(ti.i, size).place(solution)
    tree = builder.finalize()
    rhs.from_numpy(np.arange(size, dtype=np.float32))

    plan = ti.linalg.experimental.SolvePlan(
        _compiled_identity(size),
        method="cg",
        max_iterations=8,
        atol=1e-6,
        execution_policy="device_convergent",
    )
    assert plan.solve(rhs, out=solution).converged
    tree.destroy()
    with pytest.raises(RuntimeError, match="(?:destroyed|retired) SNodeTree"):
        plan.solve(rhs, out=solution)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
@pytest.mark.parametrize("multi_dispatch", [False, True])
def test_compiled_graph_direct_scalar_field_operands(multi_dispatch):
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    source = ti.field(ti.f32, shape=values.shape)
    output = ti.field(ti.f32, shape=values.shape)
    source.from_numpy(values)

    operator = _compiled_graph_identity(values.size, multi_dispatch=multi_dispatch)
    assert operator.capabilities.dense_storage_operands
    assert operator.apply(source, out=output) is output
    assert operator.apply(source, out=output) is output
    np.testing.assert_array_equal(output.to_numpy(), values)

    stats = operator.statistics()["vector_io"]
    assert stats["staging_buffer_builds"] == 0
    assert stats["pack_calls"] == 0
    assert stats["unpack_calls"] == 0
    assert stats["direct_dense_field_submissions"] == 2
    assert stats["direct_bindings"] == 4
    assert stats["last_input_execution_mode"] == "direct_contiguous"
    assert stats["last_output_execution_mode"] == "direct_contiguous"


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_compiled_graph_direct_packed_fields_and_outer_graph_action():
    vector_values = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    vector_source = ti.Vector.field(3, ti.f32, shape=(2, 2))
    vector_output = ti.Vector.field(3, ti.f32, shape=(2, 2))
    vector_source.from_numpy(vector_values)
    vector_operator = _compiled_graph_identity(vector_values.size)
    vector_operator.apply(vector_source, out=vector_output)
    np.testing.assert_array_equal(vector_output.to_numpy(), vector_values)

    matrix_values = np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2)
    matrix_source = ti.Matrix.field(2, 2, ti.f32, shape=(2, 2))
    matrix_output = ti.Matrix.field(2, 2, ti.f32, shape=(2, 2))
    matrix_source.from_numpy(matrix_values)
    matrix_base = _compiled_graph_identity(matrix_values.size, multi_dispatch=True)
    matrix_operator = (2.0 * matrix_base + matrix_base).compose(matrix_base)

    input_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "input", ti.f32, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.append_native(matrix_operator.graph_action(input_arg, output_arg))
    graph = builder.compile()
    graph.run({"input": matrix_source, "output": matrix_output})
    graph.run({"input": matrix_source, "output": matrix_output})
    np.testing.assert_array_equal(matrix_output.to_numpy(), matrix_values * 3.0)

    assert vector_operator.statistics()["vector_io"]["direct_dense_field_submissions"] == 1
    graph_stats = graph.execution_stats()
    assert graph_stats.runtime_arg_count == 2
    assert graph_stats.memory.transient_temporary_bytes == matrix_values.size * 8
    assert graph_stats.memory.persistent_temporary_bytes == matrix_values.size * 8


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_graph_cached_direct_field_binding_rejects_destroyed_tree():
    size = 4
    source = ti.field(ti.f32)
    output = ti.field(ti.f32)
    source_builder = ti.FieldsBuilder()
    source_builder.dense(ti.i, size).place(source)
    source_builder.dense(ti.i, size).place(output)
    source_tree = source_builder.finalize()
    source.from_numpy(np.arange(size, dtype=np.float32))

    operator = _compiled_graph_identity(size)
    input_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "input", ti.f32, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.append_native(operator.graph_action(input_arg, output_arg))
    graph = builder.compile()
    graph.run({"input": source, "output": output})
    source_tree.destroy()
    with pytest.raises(RuntimeError, match="(?:destroyed|retired) SNodeTree"):
        graph.run({"input": source, "output": output})


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_dense_scalar_field_bicgstab_keeps_vector_values_device_resident():
    values = np.asarray([[1.0, -2.0, 3.0], [0.5, 4.0, -1.0]], dtype=np.float32)
    rhs = ti.field(ti.f32, shape=values.shape)
    solution = ti.field(ti.f32, shape=values.shape)
    rhs.from_numpy(values)
    operator = ti.linalg.aslinearoperator(_fixed_csr(np.eye(values.size, dtype=np.float32)))
    plan = ti.linalg.experimental.SolvePlan(
        operator,
        method="bicgstab",
        max_iterations=8,
        atol=1e-6,
    )

    result = plan.solve(rhs, out=solution)
    assert result.converged
    assert result.solution is solution
    np.testing.assert_allclose(solution.to_numpy(), values, rtol=1e-6)
    vector_stats = plan.statistics()["vector_io"]
    assert vector_stats["packed_logical_bytes"] == values.nbytes
    assert vector_stats["unpacked_logical_bytes"] == values.nbytes
    assert vector_stats["pack_calls"] == 1
    assert vector_stats["unpack_calls"] == 1
    assert ti.linalg.vector_io_capabilities()["dense_field"]["value_host_transfer"] is False


def _fixed_csr(dense):
    dense = np.asarray(dense, dtype=np.float32)
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
    numeric = ti.ndarray(ti.f32, shape=len(values))
    offsets.from_numpy(np.asarray(row_offsets, dtype=np.int32))
    indices.from_numpy(np.asarray(column_indices, dtype=np.int32))
    numeric.from_numpy(np.asarray(values, dtype=np.float32))
    return ti.linalg.SparsePattern.csr(rows, columns, offsets, indices).matrix(numeric)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_packed_vector_and_matrix_fields_use_scalar_flat_lane_order():
    vector_values = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    vector_source = ti.Vector.field(3, ti.f32, shape=(2, 2))
    vector_output = ti.Vector.field(3, ti.f32, shape=(2, 2))
    vector_source.from_numpy(vector_values)

    vector_view = ti.linalg.vector_view(vector_source)
    assert vector_view.element_shape == (3,)
    assert vector_view.scalar_extent == vector_values.size
    _compiled_identity(vector_values.size).apply(vector_source, out=vector_output)
    np.testing.assert_array_equal(vector_output.to_numpy(), vector_values)

    matrix_values = np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2)
    matrix_source = ti.Matrix.field(2, 2, ti.f32, shape=(2, 2))
    matrix_output = ti.Matrix.field(2, 2, ti.f32, shape=(2, 2))
    matrix_source.from_numpy(matrix_values)

    matrix_view = ti.linalg.vector_view(matrix_source)
    assert matrix_view.element_shape == (2, 2)
    assert matrix_view.scalar_extent == matrix_values.size
    _compiled_identity(matrix_values.size).apply(matrix_source, out=matrix_output)
    np.testing.assert_array_equal(matrix_output.to_numpy(), matrix_values)

    # Multiple selected lanes from one packed element must not race through
    # whole-element read/modify/write stores.
    indexed_output = ti.Vector.field(3, ti.f32, shape=(2, 2))
    indexed_output.fill(-1)
    indices = ti.ndarray(ti.i32, shape=4)
    indices.from_numpy(np.asarray([0, 1, 2, 11], dtype=np.int32))
    source_view = ti.linalg.vector_view(vector_source, indices=indices)
    output_view = ti.linalg.vector_view(indexed_output, indices=indices)
    _compiled_identity(4).apply(source_view, out=output_view)
    expected = np.full(vector_values.shape, -1, dtype=np.float32)
    expected.reshape(-1)[[0, 1, 2, 11]] = vector_values.reshape(-1)[[0, 1, 2, 11]]
    np.testing.assert_array_equal(indexed_output.to_numpy(), expected)

    for index_shape in ((4,), (2, 1, 2)):
        scalar_extent = int(np.prod(index_shape)) * 2
        values = np.arange(scalar_extent, dtype=np.float32).reshape(*index_shape, 2)
        shaped_source = ti.Vector.field(2, ti.f32, shape=index_shape)
        shaped_output = ti.Vector.field(2, ti.f32, shape=index_shape)
        shaped_source.from_numpy(values)
        shaped_output.fill(-1)
        shaped_indices = ti.ndarray(ti.i32, shape=3)
        selected = np.asarray([0, 1, scalar_extent - 1], dtype=np.int32)
        shaped_indices.from_numpy(selected)
        _compiled_identity(3).apply(
            ti.linalg.vector_view(shaped_source, indices=shaped_indices),
            out=ti.linalg.vector_view(shaped_output, indices=shaped_indices),
        )
        expected = np.full(values.shape, -1, dtype=np.float32)
        expected.reshape(-1)[selected] = values.reshape(-1)[selected]
        np.testing.assert_array_equal(shaped_output.to_numpy(), expected)


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
@pytest.mark.parametrize("field_kind", ("vector", "matrix"))
@pytest.mark.parametrize("method", ("cg", "pcg"))
def test_packed_field_solveplan_uses_direct_graph_boundary(field_kind, method):
    if field_kind == "vector":
        values = np.arange(12, dtype=np.float32).reshape(2, 2, 3) + 1.0
        rhs = ti.Vector.field(3, ti.f32, shape=(2, 2))
        solution = ti.Vector.field(3, ti.f32, shape=(2, 2))
    else:
        values = np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2) + 1.0
        rhs = ti.Matrix.field(2, 2, ti.f32, shape=(2, 2))
        solution = ti.Matrix.field(2, 2, ti.f32, shape=(2, 2))
    rhs.from_numpy(values)

    operator = _compiled_identity(values.size)
    plan = ti.linalg.experimental.SolvePlan(
        operator,
        method=method,
        preconditioner=operator if method == "pcg" else None,
        max_iterations=8,
        atol=1e-6,
        execution_policy="device_convergent",
    )

    first = plan.solve(rhs, out=solution)
    second = plan.solve(rhs, out=solution)
    assert first.converged and second.converged
    assert second.solution is solution
    np.testing.assert_allclose(solution.to_numpy(), values, rtol=1e-6, atol=1e-6)

    stats = plan.statistics()
    capability = stats["execution_capabilities"]["direct_dense_field_solve"]
    assert capability["selected"]
    assert "root_dense_packed_vector_matrix_contiguous" in capability["qualified_layouts"]
    vector_stats = stats["vector_io"]
    assert vector_stats["staging_buffer_builds"] == 0
    assert vector_stats["staging_buffer_reuses"] == 0
    assert vector_stats["pack_calls"] == 0
    assert vector_stats["unpack_calls"] == 0
    assert vector_stats["transfer_graph_submissions"] == 0
    assert vector_stats["transfer_native_submissions"] == 0
    assert vector_stats["direct_graph_solve_submissions"] == 2
    assert vector_stats["direct_graph_solve_full_boundary_submissions"] == 2
    assert vector_stats["direct_graph_solve_field_bindings"] == 4
    assert vector_stats["direct_dense_field_submissions"] == 2


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_scalar_flat_range_views_apply_and_preserve_disjoint_storage():
    source_values = np.arange(12, dtype=np.float32)
    storage = ti.field(ti.f32, shape=12)
    storage.from_numpy(source_values)

    source_view = ti.linalg.vector_view(storage, offset=1, length=4)
    output_view = ti.linalg.vector_view(storage, offset=7, length=4)
    assert source_view.metadata["layout_kind"] == "range_scalar_flat"
    assert source_view.metadata["range"] == (1, 4, 1)
    assert source_view.metadata["index_validation"] == ("host_once_immutable_bounds")
    assert source_view.scalar_extent == 4
    assert source_view.source_scalar_extent == 12

    operator = _compiled_identity(4)
    assert operator.apply(source_view, out=output_view) is output_view
    expected = source_values.copy()
    expected[7:11] = source_values[1:5]
    np.testing.assert_array_equal(storage.to_numpy(), expected)

    with pytest.raises(RuntimeError, match="input/output aliasing"):
        operator.apply(
            ti.linalg.vector_view(storage, offset=1, length=4),
            out=ti.linalg.vector_view(storage, offset=3, length=4),
        )


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_scalar_flat_range_views_cross_packed_lanes_and_direct_affine_stride():
    values = np.arange(18, dtype=np.float32).reshape(6, 3)
    source = ti.Vector.field(3, ti.f32, shape=6)
    output = ti.Vector.field(3, ti.f32, shape=6)
    source.from_numpy(values)
    output.fill(-1)

    contiguous_source = ti.linalg.vector_view(source, offset=2, length=7)
    contiguous_output = ti.linalg.vector_view(output, offset=5, length=7)
    _compiled_identity(7).apply(contiguous_source, out=contiguous_output)
    expected = np.full(values.size, -1, dtype=np.float32)
    expected[5:12] = values.reshape(-1)[2:9]
    np.testing.assert_array_equal(output.to_numpy().reshape(-1), expected)

    strided_output = ti.Vector.field(3, ti.f32, shape=6)
    strided_output.fill(-1)
    strided_source_view = ti.linalg.vector_view(source, offset=1, length=5, stride=3)
    strided_output_view = ti.linalg.vector_view(strided_output, offset=2, length=5, stride=3)
    operator = _compiled_identity(5)
    operator.apply(strided_source_view, out=strided_output_view)
    expected = np.full(values.size, -1, dtype=np.float32)
    expected[[2, 5, 8, 11, 14]] = values.reshape(-1)[[1, 4, 7, 10, 13]]
    np.testing.assert_array_equal(strided_output.to_numpy().reshape(-1), expected)
    vector_stats = operator.statistics()["vector_io"]
    assert vector_stats["last_input_execution_mode"] == "direct_affine"
    assert vector_stats["last_output_execution_mode"] == "direct_affine"
    assert vector_stats["pack_calls"] == 0
    assert vector_stats["unpack_calls"] == 0


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_solveplan_contiguous_range_direct_and_strided_range_staged():
    values = np.asarray([1.0, -2.0, 3.0, 4.0], dtype=np.float32)
    source = ti.field(ti.f32, shape=12)
    output = ti.field(ti.f32, shape=12)
    source.fill(0)
    output.fill(-1)
    source.from_numpy(np.asarray([0, 0, *values, 0, 0, 0, 0, 0, 0], dtype=np.float32))
    rhs = ti.linalg.vector_view(source, offset=2, length=4)
    solution = ti.linalg.vector_view(output, offset=5, length=4)
    plan = ti.linalg.experimental.SolvePlan(
        _compiled_identity(4),
        method="cg",
        max_iterations=8,
        atol=1e-6,
        execution_policy=("device_convergent" if impl.current_cfg().arch in (ti.cuda, ti.vulkan) else None),
    )

    result = plan.solve(rhs, out=solution)
    assert result.converged
    np.testing.assert_allclose(output.to_numpy()[5:9], values, rtol=1e-6, atol=1e-6)
    stats = plan.statistics()["vector_io"]
    if impl.current_cfg().arch in (ti.cuda, ti.vulkan):
        assert stats["direct_graph_solve_submissions"] == 1
        assert stats["direct_graph_solve_range_bindings"] == 2
        assert stats["range_gather_calls"] == 0
        assert stats["range_scatter_calls"] == 0
    else:
        assert stats["range_gather_calls"] == 1
        assert stats["range_scatter_calls"] == 1

    strided_source = ti.field(ti.f32, shape=12)
    strided_output = ti.field(ti.f32, shape=12)
    strided_source.from_numpy(np.asarray([1, 0, -2, 0, 3, 0, 4, 0, 0, 0, 0, 0], dtype=np.float32))
    strided_output.fill(-1)
    strided_plan = ti.linalg.experimental.SolvePlan(_compiled_identity(4), method="cg", max_iterations=8, atol=1e-6)
    strided_result = strided_plan.solve(
        ti.linalg.vector_view(strided_source, offset=0, length=4, stride=2),
        out=ti.linalg.vector_view(strided_output, offset=1, length=4, stride=2),
    )
    assert strided_result.converged
    np.testing.assert_allclose(
        strided_output.to_numpy()[[1, 3, 5, 7]],
        values,
        rtol=1e-6,
        atol=1e-6,
    )
    strided_stats = strided_plan.statistics()["vector_io"]
    assert strided_stats["range_gather_calls"] == 1
    assert strided_stats["range_scatter_calls"] == 1
    assert strided_stats["direct_graph_solve_range_bindings"] == 0


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_scalar_flat_range_view_validation():
    field = ti.field(ti.f32, shape=8)
    indices = ti.ndarray(ti.i32, shape=2)
    indices.from_numpy(np.asarray([0, 1], dtype=np.int32))

    invalid = (
        ({"offset": 1}, "both offset and length"),
        ({"length": 1}, "both offset and length"),
        ({"offset": -1, "length": 1}, "non-negative"),
        ({"offset": 0, "length": 0}, "positive"),
        ({"offset": 0, "length": 1, "stride": 0}, "positive"),
        ({"offset": 7, "length": 2}, "exceeds source extent"),
        ({"offset": True, "length": 1}, "must be an integer"),
    )
    for kwargs, message in invalid:
        with pytest.raises(RuntimeError, match=message):
            ti.linalg.vector_view(field, **kwargs)

    with pytest.raises(RuntimeError, match="mutually exclusive"):
        ti.linalg.vector_view(field, indices=indices, offset=0, length=2)
    with pytest.raises(RuntimeError, match="already a VectorView"):
        ti.linalg.vector_view(ti.linalg.vector_view(field), offset=0, length=2)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_stored_csr_dense_field_path_is_provider_qualified():
    dense = np.asarray(
        [[3.0, -1.0, 0.0], [0.5, 2.0, 1.0], [0.0, -2.0, 4.0]],
        dtype=np.float32,
    )
    values = np.asarray([2.0, -1.0, 0.5], dtype=np.float32)
    source = ti.field(ti.f32, shape=3)
    output = ti.field(ti.f32, shape=3)
    source.from_numpy(values)
    operator = ti.linalg.LinearOperator.from_sparse_matrix(
        _fixed_csr(dense),
        traits=ti.linalg.OperatorTraits(singular=False),
    )
    operator.apply(source, out=output)
    np.testing.assert_allclose(output.to_numpy(), dense @ values, rtol=1e-6)
    stats = operator.statistics()["vector_io"]
    if impl.current_cfg().arch == ti.vulkan:
        assert not operator.capabilities.dense_storage_operands
        assert stats["direct_dense_field_submissions"] == 0
        assert stats["pack_calls"] == 1
        assert stats["unpack_calls"] == 1
    else:
        assert operator.capabilities.dense_storage_operands
        assert stats["direct_dense_field_submissions"] == 1
        assert stats["pack_calls"] == 0
        assert stats["unpack_calls"] == 0


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_indexed_dense_views_snapshot_topology_and_scatter_selected_values():
    source = ti.field(ti.f32, shape=(2, 3))
    output = ti.field(ti.f32, shape=(2, 3))
    source.from_numpy(np.arange(6, dtype=np.float32).reshape(2, 3))
    output.fill(-1)

    indices = ti.field(ti.i32, shape=3)
    indices.from_numpy(np.asarray([5, 1, 3], dtype=np.int32))
    source_view = ti.linalg.vector_view(source, indices=indices)
    output_view = ti.linalg.vector_view(output, indices=indices)

    # VectorView owns an immutable validated topology snapshot.
    indices.from_numpy(np.asarray([0, 2, 4], dtype=np.int32))
    operator = _compiled_identity(3)
    assert operator.apply(source_view, out=output_view) is output_view
    np.testing.assert_array_equal(
        output.to_numpy().reshape(-1),
        np.asarray([-1, 1, -1, 3, -1, 5], dtype=np.float32),
    )
    stats = operator.statistics()["vector_io"]
    assert stats["indexed_gather_calls"] == 1
    assert stats["transfer_native_submissions"] == 0
    assert stats["transfer_graph_submissions"] == 2
    assert stats["indexed_scatter_calls"] == 1
    assert source_view.metadata["layout_kind"] == "indexed_scalar_flat"
    assert source_view.metadata["index_validation"] == ("host_once_immutable_snapshot")


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_dense_vector_view_validation_alias_and_tree_lifetime():
    source = ti.field(ti.f32, shape=4)
    output = ti.field(ti.f32, shape=4)
    source.from_numpy(np.arange(4, dtype=np.float32))
    operator = _compiled_identity(4)
    plan = ti.linalg.experimental.SolvePlan(operator, max_iterations=4)

    with pytest.raises(RuntimeError, match="input/output aliasing"):
        operator.apply(source, out=source)
    with pytest.raises(RuntimeError, match="RHS and output may not alias"):
        plan.solve(source, out=source)

    output.fill(7)
    result = plan.solve(source, initial_guess=output, out=output)
    assert result.converged
    np.testing.assert_array_equal(output.to_numpy(), np.arange(4, dtype=np.float32))

    rhs_array = ti.ndarray(ti.f32, shape=4)
    rhs_array.from_numpy(np.arange(4, dtype=np.float32))
    assert plan.solve(rhs_array, out=output).solution is output
    np.testing.assert_array_equal(output.to_numpy(), np.arange(4, dtype=np.float32))
    ndarray_output = ti.ndarray(ti.f32, shape=4)
    assert plan.solve(source, out=ndarray_output).solution is ndarray_output
    np.testing.assert_array_equal(ndarray_output.to_numpy(), np.arange(4, dtype=np.float32))

    permutation = ti.ndarray(ti.i32, shape=4)
    permutation.from_numpy(np.asarray([1, 0, 2, 3], dtype=np.int32))
    permuted_output = ti.linalg.vector_view(output, indices=permutation)
    with pytest.raises(RuntimeError, match="addend and output overlap"):
        operator.apply(source, out=output, beta=1, addend=permuted_output)

    output.fill(3)
    operator.apply(source, out=output, beta=2, addend=output)
    np.testing.assert_array_equal(output.to_numpy(), np.arange(4, dtype=np.float32) + 6)

    duplicate = ti.ndarray(ti.i32, shape=2)
    duplicate.from_numpy(np.asarray([1, 1], dtype=np.int32))
    with pytest.raises(RuntimeError, match="must be unique"):
        ti.linalg.vector_view(source, indices=duplicate)

    out_of_range = ti.ndarray(ti.i32, shape=2)
    out_of_range.from_numpy(np.asarray([0, 4], dtype=np.int32))
    with pytest.raises(RuntimeError, match="within the scalar-flat"):
        ti.linalg.vector_view(source, indices=out_of_range)

    field = ti.field(ti.f32)
    builder = ti.FieldsBuilder()
    builder.dense(ti.i, 4).place(field)
    tree = builder.finalize()
    retired_tree_id = int(tree.ptr.id())
    stale_view = ti.linalg.vector_view(field)
    tree.destroy()
    with pytest.raises(RuntimeError, match="destroyed SNodeTree"):
        operator.apply(stale_view)

    ranged_field = ti.field(ti.f32)
    ranged_builder = ti.FieldsBuilder()
    ranged_builder.dense(ti.i, 6).place(ranged_field)
    ranged_tree = ranged_builder.finalize()
    stale_range = ti.linalg.vector_view(ranged_field, offset=1, length=4)
    ranged_tree.destroy()
    with pytest.raises(RuntimeError, match="destroyed SNodeTree"):
        operator.apply(stale_range)

    replacement_field = ti.field(ti.f32)
    replacement_builder = ti.FieldsBuilder()
    replacement_builder.dense(ti.i, 4).place(replacement_field)
    replacement_tree = replacement_builder.finalize()
    assert int(replacement_tree.ptr.id()) == retired_tree_id
    replacement_view = ti.linalg.vector_view(replacement_field)

    dependency = (
        int(replacement_tree.ptr.id()),
        int(replacement_tree.ptr.generation()),
    )
    runtime = impl.get_runtime()
    notified = runtime.begin_snode_tree_destroy(dependency)
    runtime.cancel_snode_tree_destroy(dependency, notified)
    with pytest.raises(RuntimeError, match="destroyed SNodeTree"):
        operator.apply(stale_view)
    operator.apply(replacement_view, out=output)

    replacement_tree.destroy()
    with pytest.raises(RuntimeError, match="destroyed SNodeTree"):
        operator.apply(replacement_view)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_dense_field_f64_and_unsupported_sparse_layout():
    values = np.asarray([1.0, -2.0, 4.0, 0.5], dtype=np.float64)
    source = ti.field(ti.f64, shape=4)
    output = ti.field(ti.f64, shape=4)
    source.from_numpy(values)

    operator = ti.linalg.identity(4, dtype=ti.f64)
    result = ti.linalg.experimental.SolvePlan(operator, method="cg", max_iterations=4, atol=1e-12).solve(
        source, out=output
    )
    assert result.converged
    np.testing.assert_allclose(output.to_numpy(), values, rtol=1e-12)

    sparse = ti.field(ti.f32)
    builder = ti.FieldsBuilder()
    builder.pointer(ti.i, 4).place(sparse)
    tree = builder.finalize()
    try:
        with pytest.raises(RuntimeError, match="root-dense-place"):
            ti.linalg.vector_view(sparse)
    finally:
        tree.destroy()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_padded_dense_field_remains_explicitly_staged():
    source = ti.field(ti.f32)
    source_guard = ti.field(ti.f32)
    output = ti.field(ti.f32)
    output_guard = ti.field(ti.f32)
    source_builder = ti.FieldsBuilder()
    source_builder.dense(ti.i, 8).place(source, source_guard)
    source_tree = source_builder.finalize()
    output_builder = ti.FieldsBuilder()
    output_builder.dense(ti.i, 8).place(output, output_guard)
    output_tree = output_builder.finalize()
    try:
        values = np.arange(8, dtype=np.float32)
        source.from_numpy(values)
        output_guard.fill(17.0)
        operator = _compiled_identity(8)
        operator.apply(source, out=output)
        np.testing.assert_array_equal(output.to_numpy(), values)
        assert (output_guard.to_numpy() == 17.0).all()
        stats = operator.statistics()["vector_io"]
        assert stats["direct_dense_field_submissions"] == 0
        assert stats["pack_calls"] == 1
        assert stats["unpack_calls"] == 1
        assert stats["last_input_execution_mode"] == "device_staged"
        assert stats["last_output_execution_mode"] == "device_staged"
    finally:
        output_tree.destroy()
        source_tree.destroy()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_linear_operator_public_api_is_stable_and_unambiguous():
    assert not hasattr(ti.linalg.experimental, "LinearOperator")
    assert ti.linalg.FieldLinearOperator is not ti.linalg.LinearOperator


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    exclude=[(ti.vulkan, "Darwin")],
    offline_cache=False,
)
def test_runtime_dense_views_bind_directly_without_staging():
    compact_source = ti.ndarray(ti.f32, shape=(2, 3))
    compact_output = ti.ndarray(ti.f32, shape=(2, 3))
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    compact_source.from_numpy(values)

    compact_operator = _compiled_identity(6)
    compact_input = ti.experimental.ndarray_view(compact_source)
    compact_result = ti.experimental.ndarray_view(compact_output)
    assert compact_operator.apply(compact_input, out=compact_result) is compact_result
    assert compact_operator.apply(compact_input, out=compact_result) is compact_result
    np.testing.assert_array_equal(compact_output.to_numpy(), values)
    compact_stats = compact_operator.statistics()["vector_io"]
    assert compact_stats["staging_buffer_builds"] == 0
    assert compact_stats["direct_dense_view_submissions"] == 2
    assert compact_stats["direct_storage_operand_builds"] == 2
    assert compact_stats["direct_storage_operand_reuses"] == 2
    assert compact_stats["last_input_execution_mode"] == "direct_contiguous"
    assert compact_stats["last_output_execution_mode"] == "direct_contiguous"

    affine_source = ti.ndarray(ti.f32, shape=8)
    affine_output = ti.ndarray(ti.f32, shape=8)
    affine_values = np.arange(8, dtype=np.float32)
    affine_source.from_numpy(affine_values)
    affine_output.fill(-1.0)
    affine_input = ti.experimental.ndarray_view(affine_source, slices=slice(0, 8, 2))
    affine_result = ti.experimental.ndarray_view(affine_output, slices=slice(1, 8, 2))

    affine_operator = _compiled_identity(4)
    assert affine_operator.capabilities.dense_storage_affine_operands
    assert affine_operator.apply(affine_input, out=affine_result) is affine_result
    expected = np.full(8, -1.0, dtype=np.float32)
    expected[1::2] = affine_values[0::2]
    np.testing.assert_array_equal(affine_output.to_numpy(), expected)
    affine_stats = affine_operator.statistics()["vector_io"]
    assert affine_stats["staging_buffer_builds"] == 0
    assert affine_stats["direct_dense_view_submissions"] == 1
    assert affine_stats["last_input_execution_mode"] == "direct_affine"
    assert affine_stats["last_output_execution_mode"] == "direct_affine"


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    exclude=[(ti.vulkan, "Darwin")],
    offline_cache=False,
)
def test_compiled_graph_accepts_affine_runtime_storage_views():
    size = 4
    topology = ti.ndarray(ti.i32, shape=size)
    topology.from_numpy(np.arange(size, dtype=np.int32))

    @ti.kernel
    def copy_by_topology(
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in range(size):
            y[i] = x[topology_data[i]]

    topology_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "topology", ti.i32, ndim=1)
    input_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "input", ti.f32, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(copy_by_topology, topology_arg, input_arg, output_arg)
    operator = ti.linalg.LinearOperator.from_graph(
        builder.compile(),
        size,
        topology={"topology": topology},
        traits=ti.linalg.OperatorTraits.spd(),
    )

    source = ti.ndarray(ti.f32, shape=8)
    output = ti.ndarray(ti.f32, shape=8)
    values = np.arange(8, dtype=np.float32)
    source.from_numpy(values)
    output.fill(-1.0)
    input_view = ti.experimental.ndarray_view(source, slices=slice(0, 8, 2))
    output_view = ti.experimental.ndarray_view(output, slices=slice(1, 8, 2))

    assert operator.capabilities.dense_storage_operands
    assert operator.capabilities.dense_storage_affine_operands
    assert operator.apply(input_view, out=output_view) is output_view
    expected = np.full(8, -1.0, dtype=np.float32)
    expected[1::2] = values[0::2]
    np.testing.assert_array_equal(output.to_numpy(), expected)
    stats = operator.statistics()["vector_io"]
    assert stats["staging_buffer_builds"] == 0
    assert stats["direct_dense_view_submissions"] == 1
    assert stats["last_input_execution_mode"] == "direct_affine"
    assert stats["last_output_execution_mode"] == "direct_affine"


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_affine_runtime_view_fails_closed_for_native_sparse_provider():
    operator = ti.linalg.LinearOperator.from_sparse_matrix(
        _fixed_csr(np.eye(4, dtype=np.float32)),
        traits=ti.linalg.OperatorTraits.spd(),
    )
    source = ti.ndarray(ti.f32, shape=8)
    output = ti.ndarray(ti.f32, shape=8)
    input_view = ti.experimental.ndarray_view(source, slices=slice(0, 8, 2))
    output_view = ti.experimental.ndarray_view(output, slices=slice(1, 8, 2))

    assert not operator.capabilities.dense_storage_affine_operands
    with pytest.raises(RuntimeError, match="provider-qualified"):
        operator.apply(input_view, out=output_view)
    with pytest.raises(RuntimeError, match="explicit out"):
        ti.linalg.identity(4).apply(input_view)
