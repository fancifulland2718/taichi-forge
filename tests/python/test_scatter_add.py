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
    elif np.issubdtype(np_dtype, np.unsignedinteger):
        values = (np.arange(n, dtype=np.uint64) % 5).astype(np_dtype)
        base = np.full(buckets, np.uint64(3), dtype=np_dtype)
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


def _run_struct_member_scatter_add(dtype, np_dtype, method):
    n = 2048
    buckets = 127
    payload = ti.types.struct(value=dtype, tag=ti.i32)
    src = ti.ndarray(payload, shape=n)
    indices = ti.ndarray(ti.i32, shape=n)
    dst = ti.ndarray(payload, shape=buckets)
    values_np, indices_np, base_np, expected = _scatter_add_input(n, buckets, np_dtype)
    host = np.zeros((n,), dtype=src.numpy_dtype)
    host["value"] = values_np
    host["tag"] = np.arange(n, dtype=np.int32) * 7 + 3
    dst_host = np.zeros((buckets,), dtype=dst.numpy_dtype)
    dst_host["value"] = base_np
    dst_host["tag"] = np.arange(buckets, dtype=np.int32) * 11 - 5
    src.from_numpy(host)
    indices.from_numpy(indices_np)
    dst.from_numpy(dst_host)
    workspace = ti.algorithms.ScatterAddWorkspace(max_items=n)
    ti.algorithms.experimental_scatter_add(
        src.field("value"), indices, dst.field("value"), method=method, workspace=workspace
    )
    result = dst.to_numpy()
    _assert_matches(result["value"], expected)
    np.testing.assert_array_equal(result["tag"], dst_host["tag"])
    np.testing.assert_array_equal(src.to_numpy()["tag"], host["tag"])


def _run_struct_tensor_member_scatter_add(method):
    n = 2048
    buckets = 127
    payload = ti.types.struct(
        vec=ti.types.vector(2, ti.i32),
        mat=ti.types.matrix(2, 2, ti.i32),
        tag=ti.i32,
    )
    src = ti.ndarray(payload, shape=n)
    indices = ti.ndarray(ti.i32, shape=n)
    dst = ti.ndarray(payload, shape=buckets)
    indices_np = (np.arange(n, dtype=np.int32) * 37 + 11) % buckets
    vec_np = (np.arange(n * 2, dtype=np.int32).reshape(n, 2) % 17) - 8
    mat_np = (np.arange(n * 4, dtype=np.int32).reshape(n, 2, 2) % 13) - 6
    base_vec = (np.arange(buckets * 2, dtype=np.int32).reshape(buckets, 2) % 5) - 2
    base_mat = (
        (np.arange(buckets * 4, dtype=np.int32).reshape(buckets, 2, 2) % 7) - 3
    )
    host = np.zeros((n,), dtype=src.numpy_dtype)
    host["vec"] = vec_np
    host["mat"] = mat_np
    host["tag"] = np.arange(n, dtype=np.int32) * 7 + 3
    dst_host = np.zeros((buckets,), dtype=dst.numpy_dtype)
    dst_host["vec"] = base_vec
    dst_host["mat"] = base_mat
    dst_host["tag"] = np.arange(buckets, dtype=np.int32) * 11 - 5
    expected_vec = base_vec.copy()
    expected_mat = base_mat.copy()
    np.add.at(expected_vec, indices_np, vec_np)
    np.add.at(expected_mat, indices_np, mat_np)

    src.from_numpy(host)
    indices.from_numpy(indices_np)
    dst.from_numpy(dst_host)
    ti.algorithms.experimental_scatter_add(
        src.field("vec"), indices, dst.field("vec"), method=method
    )
    ti.algorithms.experimental_scatter_add(
        src.field("mat"), indices, dst.field("mat"), method=method
    )

    result = dst.to_numpy()
    np.testing.assert_array_equal(result["vec"], expected_vec)
    np.testing.assert_array_equal(result["mat"], expected_mat)
    np.testing.assert_array_equal(result["tag"], dst_host["tag"])
    np.testing.assert_array_equal(src.to_numpy()["tag"], host["tag"])


@test_utils.test(arch=[ti.cuda])
def test_experimental_scatter_add_cuda_device_ndarray_wide_dtypes():
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "cuda_device_scatter_add_available")
        and prog.cuda_device_scatter_add_available()
    ):
        pytest.skip("CUDA toolkit scatter-add is unavailable in this runtime.")

    for dtype, np_dtype, method in [
        (ti.i32, np.int32, "auto"),
        (ti.f32, np.float32, "cuda_device"),
        (ti.u32, np.uint32, "cuda_device"),
        (ti.u64, np.uint64, "cuda_device"),
        (ti.i64, np.int64, "cuda_device"),
        (ti.f64, np.float64, "cuda_device"),
    ]:
        _run_ndarray_scatter_add(dtype, np_dtype, method)
        _run_struct_member_scatter_add(dtype, np_dtype, "cuda_device")
    _run_struct_tensor_member_scatter_add("cuda_device")


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_scatter_add_vulkan_native_ndarray_i32_u32_and_f32_atomic():
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "vulkan_scatter_add_available")
        and prog.vulkan_scatter_add_available()
    ):
        pytest.skip("Vulkan native scatter-add is unavailable in this runtime.")

    _run_ndarray_scatter_add(ti.i32, np.int32, "auto")
    _run_struct_member_scatter_add(ti.i32, np.int32, "vulkan_native")
    _run_struct_tensor_member_scatter_add("vulkan_native")
    if hasattr(prog, "vulkan_scatter_add_value_type_available"):
        assert prog.vulkan_scatter_add_value_type_available(2)
    _run_ndarray_scatter_add(ti.u32, np.uint32, "vulkan_native")
    _run_struct_member_scatter_add(ti.u32, np.uint32, "vulkan_native")
    _run_ndarray_scatter_add(ti.f32, np.float32, "auto")
    f32_native = (
        hasattr(prog, "vulkan_scatter_add_value_type_available")
        and prog.vulkan_scatter_add_value_type_available(1)
    )
    if f32_native:
        for _ in range(3):
            _run_ndarray_scatter_add(ti.f32, np.float32, "vulkan_native")
            _run_struct_member_scatter_add(ti.f32, np.float32, "vulkan_native")
    else:
        with pytest.raises(RuntimeError, match="value dtype"):
            _run_ndarray_scatter_add(ti.f32, np.float32, "vulkan_native")
        with pytest.raises(RuntimeError, match="scatter-add"):
            _run_struct_member_scatter_add(ti.f32, np.float32, "vulkan_native")
    for value_type, dtype, np_dtype in [
        (3, ti.u64, np.uint64),
        (4, ti.i64, np.int64),
        (5, ti.f64, np.float64),
    ]:
        if (
            hasattr(prog, "vulkan_scatter_add_value_type_available")
            and prog.vulkan_scatter_add_value_type_available(value_type)
        ):
            for _ in range(3):
                _run_ndarray_scatter_add(dtype, np_dtype, "vulkan_native")
                _run_struct_member_scatter_add(dtype, np_dtype, "vulkan_native")
        else:
            with pytest.raises(RuntimeError, match="value dtype"):
                _run_ndarray_scatter_add(dtype, np_dtype, "vulkan_native")
            with pytest.raises(RuntimeError, match="scatter-add"):
                _run_struct_member_scatter_add(dtype, np_dtype, "vulkan_native")


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
def test_experimental_scatter_add_cpu_native_ndarray_wide_dtypes():
    for dtype, np_dtype, method in [
        (ti.i32, np.int32, "auto"),
        (ti.f32, np.float32, "cpu_native"),
        (ti.u32, np.uint32, "cpu_native"),
        (ti.u64, np.uint64, "cpu_native"),
        (ti.i64, np.int64, "cpu_native"),
        (ti.f64, np.float64, "cpu_native"),
    ]:
        _run_ndarray_scatter_add(dtype, np_dtype, method)
        _run_struct_member_scatter_add(dtype, np_dtype, "cpu_native")
    _run_struct_tensor_member_scatter_add("cpu_native")
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


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_scatter_add_vulkan_native_i64_invalid_indices():
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "vulkan_scatter_add_value_type_available")
        and prog.vulkan_scatter_add_value_type_available(3)
        and prog.vulkan_scatter_add_value_type_available(4)
    ):
        pytest.skip("Vulkan native i64/u64 scatter-add atomics are unavailable.")

    for dtype, np_dtype in [(ti.u64, np.uint64), (ti.i64, np.int64)]:
        src = ti.ndarray(dtype, shape=4)
        indices = ti.ndarray(ti.i32, shape=4)
        dst = ti.ndarray(dtype, shape=4)
        src.from_numpy(np.array([10, 20, 30, 40], dtype=np_dtype))
        indices.from_numpy(np.array([0, -1, 99, 2], dtype=np.int32))
        dst.from_numpy(np.full(4, 5, dtype=np_dtype))
        ti.algorithms.experimental_scatter_add(
            src, indices, dst, method="vulkan_native"
        )
        assert np.array_equal(
            dst.to_numpy(), np.array([15, 5, 45, 5], dtype=np_dtype)
        )


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_scatter_add_vulkan_native_f64_invalid_and_special_values():
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "vulkan_scatter_add_value_type_available")
        and prog.vulkan_scatter_add_value_type_available(5)
    ):
        pytest.skip("Vulkan native f64 scatter-add atomics are unavailable.")

    src = ti.ndarray(ti.f64, shape=6)
    indices = ti.ndarray(ti.i32, shape=6)
    dst = ti.ndarray(ti.f64, shape=4)
    src.from_numpy(
        np.array([np.nan, np.inf, 1.25, 2.75, -np.inf, 99.0], dtype=np.float64)
    )
    indices.from_numpy(np.array([0, 1, 2, 2, 3, -1], dtype=np.int32))
    dst.from_numpy(np.zeros(4, dtype=np.float64))
    ti.algorithms.experimental_scatter_add(src, indices, dst, method="vulkan_native")
    out = dst.to_numpy()
    assert np.isnan(out[0])
    assert np.isposinf(out[1])
    assert np.isclose(out[2], 4.0)
    assert np.isneginf(out[3])
