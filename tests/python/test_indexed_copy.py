import gc

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils


def _source_values(n, np_dtype):
    if np.issubdtype(np_dtype, np.floating):
        return (
            ((np.arange(n, dtype=np.float64) % 67) - 33) * np.float64(0.25)
        ).astype(np_dtype)
    if np.issubdtype(np_dtype, np.unsignedinteger):
        return ((np.arange(n, dtype=np.uint64) * 3 + 7) % 4294967291).astype(
            np_dtype
        )
    return (np.arange(n, dtype=np.int32) % 97 - 48).astype(np_dtype)


def _reverse_indices(n):
    return (n - 1 - np.arange(n, dtype=np.int32)).astype(np.int32)


def _assert_matches(actual, expected):
    if np.issubdtype(expected.dtype, np.floating):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    else:
        assert np.array_equal(actual, expected)


_INDEXED_COPY_DTYPES = [
    (ti.u32, np.uint32),
    (ti.i32, np.int32),
    (ti.f32, np.float32),
    (ti.u64, np.uint64),
    (ti.i64, np.int64),
    (ti.f64, np.float64),
]


def _run_ndarray_indexed_copy(dtype, np_dtype, method, scatter):
    n = 4096
    src = ti.ndarray(dtype, shape=n)
    indices = ti.ndarray(ti.i32, shape=n)
    dst = ti.ndarray(dtype, shape=n)
    data = _source_values(n, np_dtype).astype(np_dtype)
    index_data = _reverse_indices(n)
    src.from_numpy(data)
    indices.from_numpy(index_data)
    dst.fill(0)
    workspace = ti.algorithms.IndexedCopyWorkspace(max_items=n)

    if scatter:
        ti.algorithms.experimental_scatter(
            src, indices, dst, method=method, workspace=workspace
        )
    else:
        ti.algorithms.experimental_gather(
            src, indices, dst, method=method, workspace=workspace
        )

    _assert_matches(dst.to_numpy(), data[index_data])
    return workspace


def _run_invalid_index_ndarray(method, scatter):
    src = ti.ndarray(ti.i32, shape=4)
    indices = ti.ndarray(ti.i32, shape=4)
    dst = ti.ndarray(ti.i32, shape=4)
    data = np.array([10, 20, 30, 40], dtype=np.int32)
    index_data = np.array([3, -1, 99, 0], dtype=np.int32)
    src.from_numpy(data)
    indices.from_numpy(index_data)
    dst.from_numpy(np.full(4, -7, dtype=np.int32))

    if scatter:
        ti.algorithms.experimental_scatter(src, indices, dst, method=method)
        out = dst.to_numpy()
        assert out[0] == 40
        assert out[3] == 10
        return
    else:
        ti.algorithms.experimental_gather(src, indices, dst, method=method)
        expected = np.array([40, 0, 0, 10], dtype=np.int32)
    assert np.array_equal(dst.to_numpy(), expected)


@test_utils.test(arch=[ti.cuda])
def test_experimental_gather_scatter_cuda_device_ndarray_supported_dtypes():
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "cuda_device_indexed_copy_available")
        and prog.cuda_device_indexed_copy_available()
    ):
        pytest.skip("CUDA indexed-copy is unavailable in this runtime.")

    for dtype, np_dtype in _INDEXED_COPY_DTYPES:
        gather_ws = _run_ndarray_indexed_copy(dtype, np_dtype, "auto", scatter=False)
        scatter_ws = _run_ndarray_indexed_copy(
            dtype, np_dtype, "cuda_device", scatter=True
        )
        assert gather_ws.workspace_bytes_peak == 0
        assert scatter_ws.workspace_bytes_peak == 0


@test_utils.test(arch=[ti.cuda])
def test_experimental_indexed_copy_cuda_device_invalid_indices_are_ignored():
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "cuda_device_indexed_copy_available")
        and prog.cuda_device_indexed_copy_available()
    ):
        pytest.skip("CUDA driver indexed-copy is unavailable in this runtime.")

    _run_invalid_index_ndarray("cuda_device", scatter=False)
    _run_invalid_index_ndarray("cuda_device", scatter=True)


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_gather_scatter_vulkan_native_ndarray_supported_dtypes():
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "vulkan_indexed_copy_available")
        and prog.vulkan_indexed_copy_available()
    ):
        pytest.skip("Vulkan native indexed-copy is unavailable in this runtime.")

    for dtype, np_dtype in _INDEXED_COPY_DTYPES:
        gather_ws = _run_ndarray_indexed_copy(dtype, np_dtype, "auto", scatter=False)
        scatter_ws = _run_ndarray_indexed_copy(
            dtype, np_dtype, "vulkan_native", scatter=True
        )
        assert gather_ws.workspace_bytes_peak == 0
        assert scatter_ws.workspace_bytes_peak == 0
    assert prog.vulkan_indexed_copy_workspace_bytes() == 0


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_indexed_copy_vulkan_native_invalid_indices_are_ignored():
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "vulkan_indexed_copy_available")
        and prog.vulkan_indexed_copy_available()
    ):
        pytest.skip("Vulkan native indexed-copy is unavailable in this runtime.")

    _run_invalid_index_ndarray("vulkan_native", scatter=False)
    _run_invalid_index_ndarray("vulkan_native", scatter=True)


@pytest.mark.run_in_serial
@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_indexed_copy_vulkan_reset_with_live_ndarray():
    n = 1024
    src = ti.ndarray(ti.i32, shape=n)
    indices = ti.ndarray(ti.i32, shape=n)
    dst = ti.ndarray(ti.i32, shape=n)
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "vulkan_indexed_copy_available")
        and prog.vulkan_indexed_copy_available()
    ):
        pytest.skip("Vulkan native indexed-copy is unavailable in this runtime.")

    data = _source_values(n, np.int32)
    index_data = _reverse_indices(n)
    src.from_numpy(data)
    indices.from_numpy(index_data)
    ti.algorithms.experimental_gather(src, indices, dst, method="vulkan_native")
    assert dst.to_numpy()[0] == data[-1]
    ti.reset()
    del src, indices, dst
    gc.collect()


@test_utils.test(arch=[ti.cpu])
def test_experimental_gather_scatter_cpu_native_ndarray_supported_dtypes():
    for dtype, np_dtype in _INDEXED_COPY_DTYPES:
        gather_ws = _run_ndarray_indexed_copy(dtype, np_dtype, "auto", scatter=False)
        scatter_ws = _run_ndarray_indexed_copy(dtype, np_dtype, "cpu_native", scatter=True)
        assert gather_ws.workspace_bytes_peak == 0
        assert scatter_ws.workspace_bytes_peak == 0
    assert impl.get_runtime().prog.cpu_indexed_copy_workspace_bytes() == 0


@test_utils.test(arch=[ti.cpu])
def test_experimental_indexed_copy_cpu_native_invalid_indices_are_ignored():
    _run_invalid_index_ndarray("cpu_native", scatter=False)
    _run_invalid_index_ndarray("cpu_native", scatter=True)


@test_utils.test(arch=[ti.cuda, ti.vulkan, ti.cpu], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_gather_scatter_field_kernel_i32_f32():
    n = 2048
    indices_np = _reverse_indices(n)

    for dtype, np_dtype in [(ti.i32, np.int32), (ti.f32, np.float32)]:
        src = ti.field(dtype, shape=n)
        indices = ti.field(ti.i32, shape=n)
        dst = ti.field(dtype, shape=n)
        data = _source_values(n, np_dtype).astype(np_dtype)
        src.from_numpy(data)
        indices.from_numpy(indices_np)
        dst.fill(0)
        ti.algorithms.experimental_gather(src, indices, dst, method="field_kernel")
        _assert_matches(dst.to_numpy(), data[indices_np])

        dst.fill(0)
        ti.algorithms.experimental_scatter(src, indices, dst, method="field_kernel")
        _assert_matches(dst.to_numpy(), data[indices_np])


@test_utils.test(arch=[ti.cpu])
def test_experimental_indexed_copy_field_kernel_rejects_wide_dtype():
    n = 8
    src = ti.field(ti.i64, shape=n)
    indices = ti.field(ti.i32, shape=n)
    dst = ti.field(ti.i64, shape=n)

    with pytest.raises(RuntimeError, match="Wider scalar values require"):
        ti.algorithms.experimental_gather(src, indices, dst, method="field_kernel")
