import numpy as np
import pytest

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


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
)
def test_compiled_kernel_numeric_update_preserves_inflight_generation():
    diagonal = np.asarray([2.0, 3.0, 5.0, 7.0], dtype=np.float32)
    updated_diagonal = diagonal * 1.5
    program, operator = _make_compiled_diagonal_operator(diagonal)
    input_array = ti.ndarray(ti.f32, shape=diagonal.size)
    first_output = ti.ndarray(ti.f32, shape=diagonal.size)
    second_output = ti.ndarray(ti.f32, shape=diagonal.size)
    updated_numeric = ti.ndarray(ti.f32, shape=diagonal.size)
    input_host = np.asarray([1.0, -2.0, 0.5, 3.0], dtype=np.float32)
    input_array.from_numpy(input_host)
    updated_numeric.from_numpy(updated_diagonal)

    operator.spmv(program, input_array.arr, first_output.arr)
    operator.update_numeric_data(program, updated_numeric.arr, 1, 1)
    operator.spmv(program, input_array.arr, second_output.arr)
    ti.sync()

    np.testing.assert_allclose(first_output.to_numpy(), diagonal * input_host)
    np.testing.assert_allclose(second_output.to_numpy(), updated_diagonal * input_host)
    stats = operator._debug_runtime_stats()
    contract = ti.linalg.SparseMatrix(sm=operator)._get_format_contract()
    assert contract["constraints"]["matrix_free_provider_private"]
    assert not contract["constraints"]["silent_format_fallback"]
    assert stats["identity"]["pattern_version"] == 1
    assert stats["identity"]["numeric_version"] == 2
    assert stats["operations"]["numeric_updates"] == 1
    assert stats["operations"]["numeric_update_bytes"] == diagonal.size * 4
    assert stats["operations"]["spmv_calls"] == 2
    assert stats["operations"]["spmv_plan_builds"] == 1
    assert stats["operations"]["spmv_plan_reuses"] == 2
    assert stats["resources"]["operator_owned_reserved_bytes"] == (diagonal.size * 8)
    assert stats["transfers"]["device_to_host_bytes"] == 0
    assert stats["transfers"]["device_to_device_bytes"] == diagonal.size * 12


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
)
def test_compiled_kernel_operator_rejects_reset_without_mutation():
    diagonal = np.asarray([2.0, 3.0, 5.0, 7.0], dtype=np.float32)
    program, operator = _make_compiled_diagonal_operator(diagonal)
    input_array = ti.ndarray(ti.f32, shape=diagonal.size)
    output_array = ti.ndarray(ti.f32, shape=diagonal.size)
    input_array.fill(1.0)
    output_array.fill(0.0)
    operator.spmv(program, input_array.arr, output_array.arr)
    ti.sync()
    np.testing.assert_allclose(output_array.to_numpy(), diagonal)

    arch = impl.current_cfg().arch
    ti.reset()
    ti.init(arch=arch, enable_fallback=False, offline_cache=False)
    replacement_input = ti.ndarray(ti.f32, shape=diagonal.size)
    replacement_output = ti.ndarray(ti.f32, shape=diagonal.size)
    replacement_input.fill(1.0)
    replacement_output.fill(23.0)
    with pytest.raises(RuntimeError, match="requires its owning Program"):
        operator.spmv(
            impl.get_runtime().prog,
            replacement_input.arr,
            replacement_output.arr,
        )
    ti.sync()
    np.testing.assert_array_equal(
        replacement_output.to_numpy(),
        np.full(diagonal.size, 23.0, dtype=np.float32),
    )


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
