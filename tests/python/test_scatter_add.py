import gc

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils


def _scatter_add_input(n, buckets, np_dtype):
    indices = (np.arange(n, dtype=np.int32) * 37 + 11) % buckets
    if np.issubdtype(np_dtype, np.floating):
        values = np.full(n, np.float32(0.5), dtype=np_dtype)
        base = np.full(buckets, np.float32(1.25), dtype=np_dtype)
    else:
        values = (np.arange(n, dtype=np.int32) % 5 - 2).astype(np_dtype)
        base = np.full(buckets, np.int32(3), dtype=np_dtype)
    expected = base.copy()
    np.add.at(expected, indices, values)
    return values, indices, base, expected


def _assert_matches(actual, expected):
    if np.issubdtype(expected.dtype, np.floating):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    else:
        assert np.array_equal(actual, expected)


def _run_ndarray_scatter_add(dtype, np_dtype, method):
    n = 4096
    buckets = 257
    src = ti.ndarray(dtype, shape=n)
    indices = ti.ndarray(ti.i32, shape=n)
    dst = ti.ndarray(dtype, shape=buckets)
    values_np, indices_np, base_np, expected = _scatter_add_input(n, buckets, np_dtype)
    src.from_numpy(values_np)
    indices.from_numpy(indices_np)
    dst.from_numpy(base_np)
    workspace = ti.algorithms.ScatterAddWorkspace(max_items=n)
    ti.algorithms.experimental_scatter_add(
        src, indices, dst, method=method, workspace=workspace
    )
    _assert_matches(dst.to_numpy(), expected)
    assert workspace.workspace_bytes_peak == 0


@test_utils.test(arch=[ti.cuda])
def test_experimental_scatter_add_cuda_device_ndarray_i32_f32():
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "cuda_device_scatter_add_available")
        and prog.cuda_device_scatter_add_available()
    ):
        pytest.skip("CUDA toolkit scatter-add is unavailable in this runtime.")

    _run_ndarray_scatter_add(ti.i32, np.int32, "auto")
    _run_ndarray_scatter_add(ti.f32, np.float32, "cuda_device")


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_scatter_add_vulkan_native_ndarray_i32_and_f32_fallback():
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "vulkan_scatter_add_available")
        and prog.vulkan_scatter_add_available()
    ):
        pytest.skip("Vulkan native scatter-add is unavailable in this runtime.")

    _run_ndarray_scatter_add(ti.i32, np.int32, "auto")
    _run_ndarray_scatter_add(ti.f32, np.float32, "auto")
    with pytest.raises(RuntimeError, match="i32 values"):
        _run_ndarray_scatter_add(ti.f32, np.float32, "vulkan_native")


@pytest.mark.run_in_serial
@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_scatter_add_vulkan_reset_with_live_ndarray():
    n = 1024
    buckets = 64
    src = ti.ndarray(ti.i32, shape=n)
    indices = ti.ndarray(ti.i32, shape=n)
    dst = ti.ndarray(ti.i32, shape=buckets)
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "vulkan_scatter_add_available")
        and prog.vulkan_scatter_add_available()
    ):
        pytest.skip("Vulkan native scatter-add is unavailable in this runtime.")

    values_np, indices_np, base_np, expected = _scatter_add_input(
        n, buckets, np.int32
    )
    src.from_numpy(values_np)
    indices.from_numpy(indices_np)
    dst.from_numpy(base_np)
    ti.algorithms.experimental_scatter_add(src, indices, dst, method="vulkan_native")
    assert np.array_equal(dst.to_numpy(), expected)
    ti.reset()
    del src, indices, dst
    gc.collect()


@test_utils.test(arch=[ti.cpu])
def test_experimental_scatter_add_cpu_native_ndarray_i32_f32():
    _run_ndarray_scatter_add(ti.i32, np.int32, "auto")
    _run_ndarray_scatter_add(ti.f32, np.float32, "cpu_native")
    assert impl.get_runtime().prog.cpu_scatter_add_workspace_bytes() == 0


@test_utils.test(arch=[ti.cuda, ti.vulkan, ti.cpu], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_scatter_add_field_kernel_i32_f32():
    n = 2048
    buckets = 129
    indices_np = (np.arange(n, dtype=np.int32) * 13 + 5) % buckets

    for dtype, np_dtype in [(ti.i32, np.int32), (ti.f32, np.float32)]:
        src = ti.field(dtype, shape=n)
        indices = ti.field(ti.i32, shape=n)
        dst = ti.field(dtype, shape=buckets)
        values_np, _, base_np, expected = _scatter_add_input(n, buckets, np_dtype)
        expected = base_np.copy()
        np.add.at(expected, indices_np, values_np)
        src.from_numpy(values_np)
        indices.from_numpy(indices_np)
        dst.from_numpy(base_np)
        ti.algorithms.experimental_scatter_add(
            src, indices, dst, method="field_kernel"
        )
        _assert_matches(dst.to_numpy(), expected)


@test_utils.test(arch=[ti.cuda, ti.vulkan, ti.cpu], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_scatter_add_invalid_indices_are_ignored():
    src = ti.ndarray(ti.i32, shape=4)
    indices = ti.ndarray(ti.i32, shape=4)
    dst = ti.ndarray(ti.i32, shape=4)
    src.from_numpy(np.array([10, 20, 30, 40], dtype=np.int32))
    indices.from_numpy(np.array([0, -1, 99, 2], dtype=np.int32))
    dst.from_numpy(np.full(4, 5, dtype=np.int32))
    ti.algorithms.experimental_scatter_add(src, indices, dst, method="auto")
    assert np.array_equal(dst.to_numpy(), np.array([15, 5, 45, 5], dtype=np.int32))
