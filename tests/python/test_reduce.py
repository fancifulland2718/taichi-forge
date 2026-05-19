import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils


@test_utils.test(arch=[ti.cuda])
def test_experimental_reduce_cuda_cub_ndarray_i32():
    n = 4096
    values = ti.ndarray(ti.i32, shape=n)
    output = ti.ndarray(ti.i32, shape=1)

    if not impl.get_runtime().prog.cuda_cub_reduce_available():
        pytest.skip("CUDA CUB reduce is unavailable in this build/runtime.")

    @ti.kernel
    def fill(values_arr: ti.types.ndarray(ti.i32, ndim=1)):
        for i in range(n):
            values_arr[i] = i % 97 - 48

    fill(values)
    values_np = np.array([i % 97 - 48 for i in range(n)], dtype=np.int32)
    workspace = ti.algorithms.ReduceWorkspace(max_items=n)
    for op, expected in (
        ("sum", np.sum(values_np, dtype=np.int64).astype(np.int32)),
        ("min", np.min(values_np)),
        ("max", np.max(values_np)),
    ):
        output.from_numpy(np.array([-777], dtype=np.int32))
        ti.algorithms.experimental_reduce(
            values, output, op=op, method="auto", workspace=workspace
        )
        assert output.to_numpy()[0] == expected
    assert workspace.workspace_bytes_peak > 0


@test_utils.test(arch=[ti.cuda])
def test_experimental_reduce_cuda_cub_ndarray_f32():
    n = 4096
    values = ti.ndarray(ti.f32, shape=n)
    output = ti.ndarray(ti.f32, shape=1)

    if not impl.get_runtime().prog.cuda_cub_reduce_available():
        pytest.skip("CUDA CUB reduce is unavailable in this build/runtime.")

    @ti.kernel
    def fill(values_arr: ti.types.ndarray(ti.f32, ndim=1)):
        for i in range(n):
            values_arr[i] = ti.cast(i % 41 - 20, ti.f32) * 0.25

    fill(values)
    values_np = (np.arange(n, dtype=np.int32) % 41 - 20).astype(np.float32) * 0.25
    workspace = ti.algorithms.ReduceWorkspace(max_items=n)
    ti.algorithms.experimental_reduce(
        values, output, op="sum", method="cuda_cub", workspace=workspace
    )
    assert output.to_numpy()[0] == pytest.approx(float(np.sum(values_np)), rel=1e-5)
    ti.algorithms.experimental_reduce(
        values, output, op="min", method="cuda_cub", workspace=workspace
    )
    assert output.to_numpy()[0] == pytest.approx(float(np.min(values_np)))
    ti.algorithms.experimental_reduce(
        values, output, op="max", method="cuda_cub", workspace=workspace
    )
    assert output.to_numpy()[0] == pytest.approx(float(np.max(values_np)))


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_reduce_vulkan_native_ndarray_i32():
    n = 8192
    values = ti.ndarray(ti.i32, shape=n)
    output = ti.ndarray(ti.i32, shape=1)

    prog = impl.get_runtime().prog
    if not hasattr(prog, "vulkan_reduce_available") or not prog.vulkan_reduce_available():
        pytest.skip("Vulkan native reduce is unavailable in this build/runtime.")

    @ti.kernel
    def fill(values_arr: ti.types.ndarray(ti.i32, ndim=1)):
        for i in range(n):
            values_arr[i] = i % 97 - 48

    fill(values)
    values_np = np.array([i % 97 - 48 for i in range(n)], dtype=np.int32)
    workspace = ti.algorithms.ReduceWorkspace(max_items=n)
    for op, expected in (
        ("sum", np.sum(values_np, dtype=np.int64).astype(np.int32)),
        ("min", np.min(values_np)),
        ("max", np.max(values_np)),
    ):
        output.from_numpy(np.array([-777], dtype=np.int32))
        ti.algorithms.experimental_reduce(
            values, output, op=op, method="auto", workspace=workspace
        )
        assert output.to_numpy()[0] == expected
    assert workspace.workspace_bytes_peak > 0


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_reduce_vulkan_native_reset_with_live_ndarray():
    n = 8192
    values = ti.ndarray(ti.i32, shape=n)
    output = ti.ndarray(ti.i32, shape=1)

    prog = impl.get_runtime().prog
    if not hasattr(prog, "vulkan_reduce_available") or not prog.vulkan_reduce_available():
        pytest.skip("Vulkan native reduce is unavailable in this build/runtime.")

    @ti.kernel
    def fill(values_arr: ti.types.ndarray(ti.i32, ndim=1)):
        for i in range(n):
            values_arr[i] = i % 17 - 8

    fill(values)
    ti.algorithms.experimental_reduce(values, output, op="sum", method="vulkan_native")
    expected = np.sum(np.array([i % 17 - 8 for i in range(n)], dtype=np.int32))
    assert output.to_numpy()[0] == expected.astype(np.int32)
    ti.reset()


@test_utils.test(arch=[ti.cpu])
def test_experimental_reduce_cpu_native_ndarray_i32():
    n = 131072
    values = ti.ndarray(ti.i32, shape=n)
    output = ti.ndarray(ti.i32, shape=1)

    prog = impl.get_runtime().prog
    if not hasattr(prog, "cpu_reduce_available") or not prog.cpu_reduce_available():
        pytest.skip("CPU native reduce is unavailable in this build/runtime.")

    @ti.kernel
    def fill(values_arr: ti.types.ndarray(ti.i32, ndim=1)):
        for i in range(n):
            values_arr[i] = i % 97 - 48

    fill(values)
    values_np = np.array([i % 97 - 48 for i in range(n)], dtype=np.int32)
    workspace = ti.algorithms.ReduceWorkspace(max_items=n)
    for op, expected in (
        ("sum", np.sum(values_np, dtype=np.int64).astype(np.int32)),
        ("min", np.min(values_np)),
        ("max", np.max(values_np)),
    ):
        output.from_numpy(np.array([-777], dtype=np.int32))
        ti.algorithms.experimental_reduce(
            values, output, op=op, method="auto", workspace=workspace
        )
        assert output.to_numpy()[0] == expected
    assert workspace.workspace_bytes_peak > 0
    assert impl.get_runtime().prog.cpu_reduce_workspace_bytes() == 0


@test_utils.test(arch=[ti.cpu])
def test_experimental_reduce_cpu_native_ndarray_f32():
    n = 8192
    values = ti.ndarray(ti.f32, shape=n)
    output = ti.ndarray(ti.f32, shape=1)

    prog = impl.get_runtime().prog
    if not hasattr(prog, "cpu_reduce_available") or not prog.cpu_reduce_available():
        pytest.skip("CPU native reduce is unavailable in this build/runtime.")

    @ti.kernel
    def fill(values_arr: ti.types.ndarray(ti.f32, ndim=1)):
        for i in range(n):
            values_arr[i] = ti.cast(i % 41 - 20, ti.f32) * 0.25

    fill(values)
    values_np = (np.arange(n, dtype=np.int32) % 41 - 20).astype(np.float32) * 0.25
    workspace = ti.algorithms.ReduceWorkspace(max_items=n)
    ti.algorithms.experimental_reduce(
        values, output, op="sum", method="cpu_native", workspace=workspace
    )
    assert output.to_numpy()[0] == pytest.approx(float(np.sum(values_np)), rel=1e-5)
    ti.algorithms.experimental_reduce(
        values, output, op="min", method="cpu_native", workspace=workspace
    )
    assert output.to_numpy()[0] == pytest.approx(float(np.min(values_np)))
    ti.algorithms.experimental_reduce(
        values, output, op="max", method="cpu_native", workspace=workspace
    )
    assert output.to_numpy()[0] == pytest.approx(float(np.max(values_np)))


@test_utils.test(arch=[ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_reduce_field_atomic_i32():
    n = 2048
    values = ti.field(ti.i32, shape=n)
    output = ti.field(ti.i32, shape=())

    @ti.kernel
    def fill():
        for i in range(n):
            values[i] = i % 53 - 26
        output[None] = -777

    fill()
    values_np = np.array([i % 53 - 26 for i in range(n)], dtype=np.int32)
    for op, expected in (
        ("sum", np.sum(values_np, dtype=np.int64).astype(np.int32)),
        ("min", np.min(values_np)),
        ("max", np.max(values_np)),
    ):
        ti.algorithms.experimental_reduce(values, output, op=op, method="field_atomic")
        assert output[None] == expected


@test_utils.test(arch=[ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_reduce_field_atomic_f32():
    n = 1024
    values = ti.field(ti.f32, shape=n)
    output = ti.field(ti.f32, shape=())

    @ti.kernel
    def fill():
        for i in range(n):
            values[i] = ti.cast(i % 37 - 18, ti.f32) * 0.5
        output[None] = -777.0

    fill()
    values_np = (np.arange(n, dtype=np.int32) % 37 - 18).astype(np.float32) * 0.5
    ti.algorithms.experimental_reduce(values, output, op="sum", method="field_atomic")
    assert output[None] == pytest.approx(float(np.sum(values_np)), rel=1e-5)
    ti.algorithms.experimental_reduce(values, output, op="min", method="field_atomic")
    assert output[None] == pytest.approx(float(np.min(values_np)))
    ti.algorithms.experimental_reduce(values, output, op="max", method="field_atomic")
    assert output[None] == pytest.approx(float(np.max(values_np)))
