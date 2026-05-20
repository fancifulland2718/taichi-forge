import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils


_REDUCE_DTYPE_CASES = (
    (ti.u32, np.uint32, 2),
    (ti.i32, np.int32, 0),
    (ti.f32, np.float32, 1),
    (ti.u64, np.uint64, 3),
    (ti.i64, np.int64, 4),
    (ti.f64, np.float64, 5),
)


def _values_np(n, np_dtype):
    index = np.arange(n, dtype=np.int64)
    if np.issubdtype(np_dtype, np.unsignedinteger):
        return ((index.astype(np.uint64) * 17 + 5) % 1021).astype(np_dtype)
    if np.issubdtype(np_dtype, np.floating):
        return (((index % 41) - 20).astype(np.float64) * 0.25).astype(np_dtype)
    return (index % 97 - 48).astype(np_dtype)


def _expected(values_np, op):
    if op == "min":
        return np.min(values_np)
    if op == "max":
        return np.max(values_np)
    return np.sum(values_np, dtype=values_np.dtype).astype(values_np.dtype)


def _assert_reduce_output(actual, expected, np_dtype):
    if np.issubdtype(np_dtype, np.floating):
        assert actual == pytest.approx(expected, rel=1e-5, abs=1e-6)
    else:
        assert actual == expected


def _run_ndarray_reduce_case(n, dtype, np_dtype, method, workspace):
    values = ti.ndarray(dtype, shape=n)
    output = ti.ndarray(dtype, shape=1)
    values_np = _values_np(n, np_dtype)
    values.from_numpy(values_np)
    for op in ("sum", "min", "max"):
        output.from_numpy(np.array([0], dtype=np_dtype))
        ti.algorithms.experimental_reduce(
            values, output, op=op, method=method, workspace=workspace
        )
        _assert_reduce_output(output.to_numpy()[0], _expected(values_np, op), np_dtype)


@test_utils.test(arch=[ti.cuda])
def test_experimental_reduce_cuda_cub_ndarray_dtypes():
    n = 4096
    prog = impl.get_runtime().prog
    if not prog.cuda_cub_reduce_available():
        pytest.skip("CUDA CUB reduce is unavailable in this build/runtime.")

    workspace = ti.algorithms.ReduceWorkspace(max_items=n)
    for dtype, np_dtype, _value_type in _REDUCE_DTYPE_CASES:
        _run_ndarray_reduce_case(n, dtype, np_dtype, "cuda_cub", workspace)
    assert workspace.workspace_bytes_peak > 0


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_reduce_vulkan_native_ndarray_dtypes():
    n = 8192
    prog = impl.get_runtime().prog
    if not hasattr(prog, "vulkan_reduce_available") or not prog.vulkan_reduce_available():
        pytest.skip("Vulkan native reduce is unavailable in this build/runtime.")

    workspace = ti.algorithms.ReduceWorkspace(max_items=n)
    tested = 0
    for dtype, np_dtype, value_type in _REDUCE_DTYPE_CASES:
        if hasattr(prog, "vulkan_reduce_value_type_available"):
            if not prog.vulkan_reduce_value_type_available(value_type):
                continue
        elif dtype != ti.i32:
            continue
        _run_ndarray_reduce_case(n, dtype, np_dtype, "vulkan_native", workspace)
        tested += 1
    assert tested >= 3
    assert workspace.workspace_bytes_peak > 0


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_reduce_vulkan_native_reset_with_live_ndarray():
    n = 8192
    values = ti.ndarray(ti.i32, shape=n)
    output = ti.ndarray(ti.i32, shape=1)

    prog = impl.get_runtime().prog
    if not hasattr(prog, "vulkan_reduce_available") or not prog.vulkan_reduce_available():
        pytest.skip("Vulkan native reduce is unavailable in this build/runtime.")

    values_np = (np.arange(n, dtype=np.int64) % 17 - 8).astype(np.int32)
    values.from_numpy(values_np)
    ti.algorithms.experimental_reduce(values, output, op="sum", method="vulkan_native")
    expected = np.sum(values_np, dtype=np.int32).astype(np.int32)
    assert output.to_numpy()[0] == expected
    ti.reset()


@test_utils.test(arch=[ti.cpu])
def test_experimental_reduce_cpu_native_ndarray_dtypes():
    n = 131072
    prog = impl.get_runtime().prog
    if not hasattr(prog, "cpu_reduce_available") or not prog.cpu_reduce_available():
        pytest.skip("CPU native reduce is unavailable in this build/runtime.")

    workspace = ti.algorithms.ReduceWorkspace(max_items=n)
    for dtype, np_dtype, _value_type in _REDUCE_DTYPE_CASES:
        _run_ndarray_reduce_case(n, dtype, np_dtype, "cpu_native", workspace)
    assert workspace.workspace_bytes_peak > 0
    assert impl.get_runtime().prog.cpu_reduce_workspace_bytes() == 0


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
