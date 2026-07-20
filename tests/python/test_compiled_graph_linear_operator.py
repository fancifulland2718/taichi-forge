import numpy as np

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

    stats = operator._debug_runtime_stats()
    expected_backend = {
        ti.cpu: "cpu",
        ti.cuda: "cuda",
        ti.vulkan: "vulkan",
    }[impl.current_cfg().arch]
    assert stats["identity"]["backend_family"] == expected_backend
    assert stats["identity"]["storage_format"] == "matrix_free_graph"
    assert stats["provider"]["name"] == "forge_compiled_graph"
    assert stats["operations"]["spmv_calls"] == 2
    assert stats["operations"]["spmv_plan_builds"] == 1
    assert stats["operations"]["spmv_plan_reuses"] == 2
    assert stats["operations"]["spmv_workspace_allocations"] == 1
    assert stats["resources"]["spmv_workspace_reserved_bytes"] == size * 4
    assert stats["resources"]["operator_owned_reserved_bytes"] == size * 12
    assert stats["transfers"]["device_to_host_bytes"] == 0
