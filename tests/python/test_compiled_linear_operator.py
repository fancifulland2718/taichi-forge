import numpy as np

import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils


def _make_compiled_diagonal_operator(diagonal):
    diagonal = np.asarray(diagonal, dtype=np.float32)
    size = diagonal.size
    topology = ti.ndarray(ti.i32, shape=size)
    numeric = ti.ndarray(ti.f32, shape=size)
    compile_input = ti.ndarray(ti.f32, shape=size)
    compile_output = ti.ndarray(ti.f32, shape=size)
    topology.from_numpy(np.arange(size, dtype=np.int32))
    numeric.from_numpy(diagonal)

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

    primal = apply_diagonal._primal
    key = primal.ensure_compiled(size, topology, numeric, compile_input, compile_output)
    kernel_cpp = primal.compiled_kernels[key]
    program = impl.get_runtime().prog
    operator = program._create_compiled_kernel_linear_operator_with_numeric_data(
        kernel_cpp,
        size,
        1,
        1,
        topology.arr,
        numeric.arr,
    )
    return program, operator


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_compiled_kernel_cg_applies_operator_end_to_end():
    diagonal = np.asarray([2.0, 3.0, 5.0, 7.0], dtype=np.float32)
    exact = np.asarray([0.5, -1.0, 2.0, 1.5], dtype=np.float32)
    program, operator = _make_compiled_diagonal_operator(diagonal)
    solution = ti.ndarray(ti.f32, shape=diagonal.size)
    rhs = ti.ndarray(ti.f32, shape=diagonal.size)
    solution.fill(0.0)
    rhs.from_numpy(diagonal * exact)

    plan = ti._lib.core._make_vulkan_compiled_kernel_cg_convergence_plan(
        program, operator, 16, 1e-5
    )
    plan.solve(program, solution.arr, rhs.arr)

    assert plan.is_success()
    np.testing.assert_allclose(solution.to_numpy(), exact, rtol=2e-4, atol=2e-4)
    plan_stats = plan._debug_runtime_stats()
    operator_stats = operator._debug_runtime_stats()
    assert plan_stats["identity"]["method"] == (
        "cg_compiled_kernel_bounded_masked_probe"
    )
    assert plan_stats["operations"]["operator_apply_calls"] == 17
    assert operator_stats["operations"]["spmv_calls"] == 17
    assert operator_stats["transfers"]["device_to_host_bytes"] == 0
