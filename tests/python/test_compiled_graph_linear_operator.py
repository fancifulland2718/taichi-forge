import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
)
def test_compiled_graph_operator_runs_data_dependent_dispatches():
    size = 6
    topology = ti.ndarray(ti.i32, shape=size)
    numeric = ti.ndarray(ti.f32, shape=size)
    workspace = ti.ndarray(ti.f32, shape=size)
    input_array = ti.ndarray(ti.f32, shape=size)
    output_array = ti.ndarray(ti.f32, shape=size)
    topology.from_numpy(np.arange(size, dtype=np.int32))
    diagonal = np.asarray([2.0, 3.0, 5.0, 7.0, 11.0, 13.0], dtype=np.float32)
    numeric.from_numpy(diagonal)
    workspace.fill(-37.0)

    @ti.kernel
    def stage_apply(
        active_size: ti.i32,
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        numeric_data: ti.types.ndarray(dtype=ti.f32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        temporary: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(active_size):
            temporary[index] = numeric_data[index] * x[topology_data[index]]

    @ti.kernel
    def finish_apply(
        active_size: ti.i32,
        temporary: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(active_size):
            y[index] = 2.0 * temporary[index]

    sym_active_size = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "active_size", ti.i32)
    sym_topology = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "topology", ti.i32, ndim=1)
    sym_numeric = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "numeric", ti.f32, ndim=1)
    sym_workspace = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "workspace", ti.f32, ndim=1)
    sym_input = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "input", ti.f32, ndim=1)
    sym_output = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        stage_apply,
        sym_active_size,
        sym_topology,
        sym_numeric,
        sym_input,
        sym_workspace,
    )
    builder.dispatch(
        finish_apply,
        sym_active_size,
        sym_workspace,
        sym_output,
    )
    graph = builder.compile()
    assert graph._debug_info["dispatch_count"] == 2

    program = impl.get_runtime().prog
    operator = program._create_compiled_graph_linear_operator(
        graph._compiled_graph,
        size,
        1,
        1,
        {"active_size": size},
        {"topology": topology.arr},
        {"numeric": numeric.arr},
        {"workspace": workspace.arr},
    )
    first_input = np.linspace(-1.5, 2.0, size, dtype=np.float32)
    input_array.from_numpy(first_input)
    operator.spmv(program, input_array.arr, output_array.arr)
    ti.sync()
    np.testing.assert_allclose(output_array.to_numpy(), 2.0 * diagonal * first_input)

    second_input = first_input[::-1].copy()
    input_array.from_numpy(second_input)
    operator.spmv(program, input_array.arr, output_array.arr)
    ti.sync()
    np.testing.assert_allclose(output_array.to_numpy(), 2.0 * diagonal * second_input)

    updated_numeric = ti.ndarray(ti.f32, shape=size)
    old_generation_output = ti.ndarray(ti.f32, shape=size)
    new_generation_output = ti.ndarray(ti.f32, shape=size)
    updated_diagonal = diagonal * 1.5
    updated_numeric.from_numpy(updated_diagonal)
    operator.spmv(program, input_array.arr, old_generation_output.arr)
    operator.update_numeric_data(
        program,
        {"numeric": updated_numeric.arr},
        1,
        1,
    )
    operator.spmv(program, input_array.arr, new_generation_output.arr)
    ti.sync()
    np.testing.assert_allclose(
        old_generation_output.to_numpy(), 2.0 * diagonal * second_input
    )
    np.testing.assert_allclose(
        new_generation_output.to_numpy(),
        2.0 * updated_diagonal * second_input,
    )

    if impl.current_cfg().arch == ti.cpu:
        plan = ti._lib.core._make_cpu_operator_cg_solver(
            program, operator, 16, 1e-6, 0.0
        )
        expected_execution = "explicit_sequence"
        expected_policy = "host_each_iteration"
        allowed_backend_paths = {"explicit_sequence"}
    elif impl.current_cfg().arch == ti.cuda:
        plan = ti._lib.core._make_cuda_compiled_graph_cg_solver(
            program, operator, 16, 1e-5, False, 0.0
        )
        expected_execution = "compiled_graph"
        expected_policy = "host_each_iteration"
        allowed_backend_paths = {
            "cuda_capture",
            "cuda_exact_replay",
            "cuda_patched_replay",
            "ordinary_graph_fallback",
        }
    else:
        plan = ti._lib.core._make_vulkan_compiled_graph_cg_convergence_plan(
            program, operator, 16, 1e-5
        )
        expected_execution = "compiled_graph"
        expected_policy = "fixed_budget_masked"
        allowed_backend_paths = {
            "vulkan_record",
            "vulkan_replay",
            "ordinary_graph_fallback",
        }

    exact = np.asarray([0.5, -1.0, 2.0, 1.5, -0.25, 0.75], dtype=np.float32)
    solution = ti.ndarray(ti.f32, shape=size)
    rhs = ti.ndarray(ti.f32, shape=size)
    solution.fill(0.0)
    rhs.from_numpy(2.0 * updated_diagonal * exact)
    plan.solve(program, solution.arr, rhs.arr)
    assert plan.is_success()
    np.testing.assert_allclose(solution.to_numpy(), exact, rtol=3e-4, atol=3e-4)

    # Numeric snapshots may be replaced between solves. The same operator
    # execution plan must observe the new generation and patch/rebind dynamic
    # ndarray addresses without replaying stale arguments.
    final_numeric = ti.ndarray(ti.f32, shape=size)
    final_diagonal = updated_diagonal * 0.75
    final_numeric.from_numpy(final_diagonal)
    operator.update_numeric_data(program, {"numeric": final_numeric.arr}, 1, 2)
    rebound_solution = ti.ndarray(ti.f32, shape=size)
    rebound_rhs = ti.ndarray(ti.f32, shape=size)
    rebound_solution.fill(0.0)
    rebound_rhs.from_numpy(2.0 * final_diagonal * exact)
    plan.solve(program, rebound_solution.arr, rebound_rhs.arr)
    assert plan.is_success()
    np.testing.assert_allclose(
        rebound_solution.to_numpy(), exact, rtol=3e-4, atol=3e-4
    )

    plan_stats = plan._debug_runtime_stats()
    assert plan_stats["identity"]["operator_action_provider"] == (
        "program_bound_multi_kernel"
    )
    assert plan_stats["identity"]["operator_execution_kind"] == expected_execution
    assert plan_stats["identity"]["operator_backend_execution_path"] in (
        allowed_backend_paths
    )
    assert plan_stats["identity"]["solver_execution_policy"] == expected_policy
    assert not plan_stats["identity"]["solver_graph_enabled"]
    assert plan_stats["operations"]["operator_execution_plan_builds"] == 1
    assert plan_stats["operations"]["operator_execution_plan_reuses"] > 0
    assert plan_stats["operations"]["operator_binding_rebinds"] > 0
    assert plan_stats["operations"]["operator_numeric_generation_changes"] == 1
    assert plan_stats["operations"]["workspace_builds"] == 1
    assert plan_stats["operations"]["workspace_reuses"] == 1
    if expected_execution == "explicit_sequence":
        assert plan_stats["operations"]["operator_sequence_submissions"] > 0
        assert plan_stats["operations"]["operator_compiled_graph_submissions"] == 0
    else:
        assert plan_stats["operations"]["operator_compiled_graph_submissions"] > 0
        assert (
            plan_stats["operations"]["operator_backend_captures"]
            + plan_stats["operations"]["operator_ordinary_fallbacks"]
            > 0
        )
        if plan_stats["operations"]["operator_backend_captures"] > 0:
            assert plan_stats["operations"]["operator_backend_replays"] > 0
    assert plan_stats["operations"]["operator_cache_invalidations"] == 0

    stats = operator._debug_runtime_stats()
    contract = ti.linalg.SparseMatrix(sm=operator)._get_format_contract()
    assert contract["constraints"]["matrix_free_provider_private"]
    assert not contract["constraints"]["silent_format_fallback"]
    expected_backend = {
        ti.cpu: "cpu",
        ti.cuda: "cuda",
        ti.vulkan: "vulkan",
    }[impl.current_cfg().arch]
    assert stats["identity"]["backend_family"] == expected_backend
    assert stats["identity"]["storage_format"] == "matrix_free_graph"
    assert stats["provider"]["name"] == "forge_compiled_graph"
    assert stats["identity"]["pattern_version"] == 1
    assert stats["identity"]["numeric_version"] == 3
    assert stats["operations"]["numeric_updates"] == 2
    assert stats["operations"]["numeric_update_bytes"] == size * 8
    assert stats["operations"]["spmv_calls"] > 4
    assert stats["operations"]["spmv_plan_builds"] == 3
    assert stats["operations"]["spmv_plan_reuses"] == stats["operations"]["spmv_calls"]
    assert stats["operations"]["spmv_workspace_allocations"] == 1
    assert stats["resources"]["spmv_workspace_reserved_bytes"] == size * 4
    assert stats["resources"]["operator_owned_reserved_bytes"] == size * 12
    assert stats["resources"]["numeric_update_peak_temporary_bytes"] == size * 4
    assert stats["transfers"]["device_to_host_bytes"] == 0
    assert stats["transfers"]["device_to_device_bytes"] == size * 20
