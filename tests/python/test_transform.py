import gc

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.lang import impl
from taichi_forge.lang.misc import get_host_arch_list
from tests import test_utils


_TRANSFORM_DTYPES = (ti.u32, ti.i32, ti.f32, ti.u64, ti.i64, ti.f64)


def _transform_case(dtype, n):
    if dtype == ti.u32:
        data = (np.arange(n, dtype=np.uint32) % np.uint32(113)).astype(np.uint32)
        return data, 3, 7, (data * np.uint32(3) + np.uint32(7)).astype(np.uint32)
    if dtype == ti.i32:
        data = (np.arange(n, dtype=np.int32) % 97 - 48).astype(np.int32)
        return data, -2, 9, (data * np.int32(-2) + np.int32(9)).astype(np.int32)
    if dtype == ti.f32:
        data = (np.arange(n, dtype=np.float32) % 37 - 18) * np.float32(0.5)
        return data, 0.5, 3.25, data * np.float32(0.5) + np.float32(3.25)
    if dtype == ti.u64:
        data = (np.arange(n, dtype=np.uint64) % np.uint64(257)).astype(np.uint64)
        return data, 5, 11, (data * np.uint64(5) + np.uint64(11)).astype(np.uint64)
    if dtype == ti.i64:
        data = (np.arange(n, dtype=np.int64) % 211 - 105).astype(np.int64)
        return data, -3, 17, (data * np.int64(-3) + np.int64(17)).astype(np.int64)
    if dtype == ti.f64:
        data = (np.arange(n, dtype=np.float64) % 43 - 21) * np.float64(0.125)
        return data, -1.75, 0.5, data * np.float64(-1.75) + np.float64(0.5)
    raise AssertionError(dtype)


def _assert_transform_equal(dtype, actual, expected):
    if dtype in (ti.f32, ti.f64):
        np.testing.assert_allclose(actual, expected, rtol=1e-6)
    else:
        assert np.array_equal(actual, expected)


def _case_shape(shape):
    if isinstance(shape, int):
        return (shape,), shape
    shape = tuple(shape)
    return shape, int(np.prod(shape, dtype=np.int64))


def _run_struct_member_transform_case(dtype, shape, method, workspace=None):
    shape, n = _case_shape(shape)
    payload = ti.types.struct(value=dtype, tag=ti.i32)
    src = ti.ndarray(payload, shape=shape)
    dst = ti.ndarray(dtype, shape=shape)
    data, scale, bias, expected = _transform_case(dtype, n)
    data = data.reshape(shape)
    expected = expected.reshape(shape)
    host = np.zeros(shape, dtype=src.numpy_dtype)
    host["value"] = data
    host["tag"] = np.arange(n, dtype=np.int32).reshape(shape) * 3 + 1
    src.from_numpy(host)
    dst.fill(0)
    ti.algorithms.experimental_transform(
        src.field("value"),
        dst,
        scale=scale,
        bias=bias,
        method=method,
        workspace=workspace,
    )
    _assert_transform_equal(dtype, dst.to_numpy(), expected)
    # The strided transform must not touch unrelated struct fields.
    roundtrip = src.to_numpy()
    assert np.array_equal(roundtrip["tag"], host["tag"])


def _run_dense_field_transform_case(dtype, shape, method, workspace=None):
    shape, n = _case_shape(shape)
    src = ti.field(dtype, shape=shape)
    dst = ti.field(dtype, shape=shape)
    data, scale, bias, expected = _transform_case(dtype, n)
    src.from_numpy(data.reshape(shape))
    dst.fill(0)
    ti.algorithms.experimental_transform(
        src,
        dst,
        scale=scale,
        bias=bias,
        method=method,
        workspace=workspace,
    )
    _assert_transform_equal(dtype, dst.to_numpy(), expected.reshape(shape))


def _native_transform_method_for_current_arch():
    arch = impl.current_cfg().arch
    prog = impl.get_runtime().prog
    if arch == ti.cpu:
        if not (
            hasattr(prog, "cpu_transform_available")
            and prog.cpu_transform_available()
        ):
            pytest.skip("CPU native transform is unavailable.")
        return "cpu_native", "cpu_transform_affine_dense_field"
    if arch == ti.cuda:
        if not (
            hasattr(prog, "cuda_device_transform_available")
            and prog.cuda_device_transform_available()
        ):
            pytest.skip("CUDA device transform is unavailable.")
        return "cuda_device", "cuda_device_transform_affine_dense_field"
    if arch == ti.vulkan:
        if not (
            hasattr(prog, "vulkan_transform_available")
            and prog.vulkan_transform_available()
        ):
            pytest.skip("Vulkan native transform is unavailable.")
        return "vulkan_native", "vulkan_transform_affine_dense_field"
    pytest.skip("native transform is unavailable on this arch.")


def _run_dense_matrix_field_transform_case():
    n = 128
    method, _ = _native_transform_method_for_current_arch()
    src = ti.Vector.field(2, ti.i32, shape=n)
    dst = ti.Vector.field(2, ti.i32, shape=n)
    values = np.arange(n * 2, dtype=np.int32).reshape(n, 2) - 17
    src.from_numpy(values)
    dst.fill(0)
    workspace = ti.algorithms.TransformWorkspace(max_items=n)

    ti.algorithms.experimental_transform(
        src, dst, scale=3, bias=5, method=method, workspace=workspace
    )

    np.testing.assert_array_equal(dst.to_numpy(), values * 3 + 5)
    assert len(workspace._native_transform_plans) == 1
    assert (
        workspace._native_transform_plan.method_name
        == "transform_affine_dense_field_packed"
    )
    assert workspace.workspace_bytes_peak >= 0
    assert len(workspace._native_transform_plan_groups) == 0

    dst.fill(0)
    ti.algorithms.experimental_transform(
        src, dst, scale=3, bias=5, method=method, workspace=workspace
    )
    np.testing.assert_array_equal(dst.to_numpy(), values * 3 + 5)
    assert len(workspace._native_transform_plans) == 1


def _run_struct_tensor_member_transform_case(dtype, shape, method, workspace=None):
    shape, n = _case_shape(shape)
    payload = ti.types.struct(
        vec=ti.types.vector(2, dtype),
        mat=ti.types.matrix(2, 2, dtype),
        tag=ti.i32,
    )
    src = ti.ndarray(payload, shape=shape)
    dst = ti.ndarray(payload, shape=shape)
    vec_data, scale, bias, _ = _transform_case(dtype, n * 2)
    mat_data, _, _, _ = _transform_case(dtype, n * 4)
    vec_data = vec_data.reshape(shape + (2,))
    mat_data = mat_data.reshape(shape + (2, 2))
    host = np.zeros(shape, dtype=src.numpy_dtype)
    dst_host = np.zeros(shape, dtype=dst.numpy_dtype)
    host["vec"] = vec_data
    host["mat"] = mat_data
    host["tag"] = np.arange(n, dtype=np.int32).reshape(shape) * 3 + 1
    dst_host["tag"] = np.arange(n, dtype=np.int32).reshape(shape) * 11 + 5
    src.from_numpy(host)
    dst.from_numpy(dst_host)

    ti.algorithms.experimental_transform(
        src.field("vec"),
        dst.field("vec"),
        scale=scale,
        bias=bias,
        method=method,
        workspace=workspace,
    )
    ti.algorithms.experimental_transform(
        src.field("mat"),
        dst.field("mat"),
        scale=scale,
        bias=bias,
        method=method,
        workspace=workspace,
    )

    result = dst.to_numpy()
    expected_vec = (vec_data * scale + bias).astype(vec_data.dtype)
    expected_mat = (mat_data * scale + bias).astype(mat_data.dtype)
    _assert_transform_equal(dtype, result["vec"], expected_vec)
    _assert_transform_equal(dtype, result["mat"], expected_mat)
    assert np.array_equal(result["tag"], dst_host["tag"])
    assert np.array_equal(src.to_numpy()["tag"], host["tag"])


@test_utils.test(arch=[ti.cuda])
def test_experimental_transform_cuda_device_ndarray_i32():
    n = 4096
    src = ti.ndarray(ti.i32, shape=n)
    dst = ti.ndarray(ti.i32, shape=n)
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "cuda_device_transform_available")
        and prog.cuda_device_transform_available()
    ):
        pytest.skip("CUDA driver transform is unavailable in this runtime.")

    data = (np.arange(n, dtype=np.int32) % 97 - 48).astype(np.int32)
    src.from_numpy(data)
    dst.fill(0)
    workspace = ti.algorithms.TransformWorkspace(max_items=n)
    ti.algorithms.experimental_transform(
        src, dst, scale=3, bias=-7, method="auto", workspace=workspace
    )
    expected = (data * np.int32(3) + np.int32(-7)).astype(np.int32)
    assert np.array_equal(dst.to_numpy(), expected)
    assert workspace.workspace_bytes_peak == 0


@test_utils.test(arch=[ti.cuda])
def test_experimental_transform_cuda_device_ndarray_f32():
    n = 4096
    src = ti.ndarray(ti.f32, shape=n)
    dst = ti.ndarray(ti.f32, shape=n)
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "cuda_device_transform_available")
        and prog.cuda_device_transform_available()
    ):
        pytest.skip("CUDA driver transform is unavailable in this runtime.")

    data = (np.arange(n, dtype=np.float32) % 31 - 15) * np.float32(0.25)
    src.from_numpy(data)
    dst.fill(0.0)
    ti.algorithms.experimental_transform(
        src, dst, scale=1.5, bias=-0.25, method="cuda_device"
    )
    np.testing.assert_allclose(dst.to_numpy(), data * 1.5 - 0.25, rtol=1e-6)


@test_utils.test(arch=[ti.cuda])
def test_experimental_transform_cuda_device_ndarray_extended_dtypes():
    n = 4096
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "cuda_device_transform_available")
        and prog.cuda_device_transform_available()
    ):
        pytest.skip("CUDA device transform is unavailable in this runtime.")

    for dtype in _TRANSFORM_DTYPES:
        if dtype in (ti.u64, ti.i64, ti.f64) and not (
            hasattr(prog, "cuda_toolkit_transform_available")
            and prog.cuda_toolkit_transform_available()
        ):
            continue
        data, scale, bias, expected = _transform_case(dtype, n)
        src = ti.ndarray(dtype, shape=n)
        dst = ti.ndarray(dtype, shape=n)
        src.from_numpy(data)
        dst.fill(0)
        workspace = ti.algorithms.TransformWorkspace(max_items=n)
        ti.algorithms.experimental_transform(
            src, dst, scale=scale, bias=bias, method="cuda_device", workspace=workspace
        )
        _assert_transform_equal(dtype, dst.to_numpy(), expected)
        assert workspace.workspace_bytes_peak == 0


@test_utils.test(arch=[ti.cuda])
def test_experimental_transform_cuda_device_struct_member_view():
    n = 4096
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "cuda_toolkit_transform_available")
        and prog.cuda_toolkit_transform_available()
    ):
        pytest.skip("CUDA toolkit transform is unavailable in this runtime.")

    for dtype in _TRANSFORM_DTYPES:
        workspace = ti.algorithms.TransformWorkspace(max_items=n)
        _run_struct_member_transform_case(
            dtype, n, method="cuda_device", workspace=workspace
        )
        assert workspace.workspace_bytes_peak == 0


@test_utils.test(arch=[ti.cuda])
def test_experimental_transform_cuda_device_dense_field_dtypes():
    n = 4096
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "cuda_device_transform_available")
        and prog.cuda_device_transform_available()
    ):
        pytest.skip("CUDA device transform is unavailable in this runtime.")

    for dtype in _TRANSFORM_DTYPES:
        if dtype in (ti.u64, ti.i64, ti.f64) and not (
            hasattr(prog, "cuda_toolkit_transform_available")
            and prog.cuda_toolkit_transform_available()
        ):
            continue
        workspace = ti.algorithms.TransformWorkspace(max_items=n)
        _run_dense_field_transform_case(
            dtype, n, method="cuda_device", workspace=workspace
        )
        assert workspace.workspace_bytes_peak == 0


@test_utils.test(arch=[ti.cuda])
def test_experimental_transform_cuda_device_dense_field_workspace_replay():
    n = 128
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "cuda_device_transform_available")
        and prog.cuda_device_transform_available()
    ):
        pytest.skip("CUDA device transform is unavailable in this runtime.")

    src = ti.field(ti.i32, shape=n)
    dst = ti.field(ti.i32, shape=n)
    workspace = ti.algorithms.TransformWorkspace(max_items=n)
    for base in (0, 17):
        data = (np.arange(n, dtype=np.int32) + base).astype(np.int32)
        src.from_numpy(data)
        ti.algorithms.experimental_transform(
            src, dst, scale=3, bias=-2, method="cuda_device", workspace=workspace
        )
        np.testing.assert_array_equal(dst.to_numpy(), data * 3 - 2)
    assert workspace._native_transform_plan is not None
    assert workspace._native_transform_plan.backend == "cuda_device"


@test_utils.test(arch=[ti.cuda])
def test_experimental_transform_cuda_device_struct_tensor_member_view():
    n = 4096
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "cuda_toolkit_transform_available")
        and prog.cuda_toolkit_transform_available()
    ):
        pytest.skip("CUDA toolkit transform is unavailable in this runtime.")

    for dtype in (ti.i32, ti.f32):
        workspace = ti.algorithms.TransformWorkspace(max_items=n)
        _run_struct_tensor_member_transform_case(
            dtype, n, method="cuda_device", workspace=workspace
        )
        assert workspace.workspace_bytes_peak == 0


@test_utils.test(arch=[ti.cuda])
def test_experimental_transform_cuda_device_nd_shape():
    shape = (32, 17)
    n = int(np.prod(shape, dtype=np.int64))
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "cuda_device_transform_available")
        and prog.cuda_device_transform_available()
    ):
        pytest.skip("CUDA driver transform is unavailable in this runtime.")

    data, scale, bias, expected = _transform_case(ti.i32, n)
    src = ti.ndarray(ti.i32, shape=shape)
    dst = ti.ndarray(ti.i32, shape=shape)
    src.from_numpy(data.reshape(shape))
    dst.fill(0)
    workspace = ti.algorithms.TransformWorkspace(max_items=n)
    ti.algorithms.experimental_transform(
        src, dst, scale=scale, bias=bias, method="cuda_device", workspace=workspace
    )
    assert np.array_equal(dst.to_numpy(), expected.reshape(shape))
    assert workspace.workspace_bytes_peak == 0

    if (
        hasattr(prog, "cuda_toolkit_transform_available")
        and prog.cuda_toolkit_transform_available()
    ):
        _run_struct_tensor_member_transform_case(
            ti.i32, shape, method="cuda_device", workspace=workspace
        )


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_transform_vulkan_native_ndarray_i32_f32():
    n = 4096
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "vulkan_transform_available")
        and prog.vulkan_transform_available()
    ):
        pytest.skip("Vulkan native transform is unavailable in this runtime.")

    src_i = ti.ndarray(ti.i32, shape=n)
    dst_i = ti.ndarray(ti.i32, shape=n)
    data_i = (np.arange(n, dtype=np.int32) % 113 - 56).astype(np.int32)
    src_i.from_numpy(data_i)
    workspace = ti.algorithms.TransformWorkspace(max_items=n)
    ti.algorithms.experimental_transform(
        src_i, dst_i, scale=-2, bias=9, method="auto", workspace=workspace
    )
    expected_i = (data_i * np.int32(-2) + np.int32(9)).astype(np.int32)
    assert np.array_equal(dst_i.to_numpy(), expected_i)
    assert workspace.workspace_bytes_peak == 8

    src_f = ti.ndarray(ti.f32, shape=n)
    dst_f = ti.ndarray(ti.f32, shape=n)
    data_f = (np.arange(n, dtype=np.float32) % 37 - 18) * np.float32(0.5)
    src_f.from_numpy(data_f)
    ti.algorithms.experimental_transform(
        src_f, dst_f, scale=0.5, bias=3.25, method="vulkan_native"
    )
    np.testing.assert_allclose(dst_f.to_numpy(), data_f * 0.5 + 3.25, rtol=1e-6)

    prog.vulkan_transform_clear_workspace()
    dst_i.fill(0)
    ti.algorithms.experimental_transform(
        src_i, dst_i, scale=3, bias=-4, method="vulkan_native", workspace=workspace
    )
    expected_reuse = (data_i * np.int32(3) + np.int32(-4)).astype(np.int32)
    assert np.array_equal(dst_i.to_numpy(), expected_reuse)


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_transform_vulkan_native_ndarray_extended_dtypes():
    n = 4096
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "vulkan_transform_available")
        and prog.vulkan_transform_available()
    ):
        pytest.skip("Vulkan native transform is unavailable in this runtime.")

    for dtype, value_type in (
        (ti.u32, 2),
        (ti.i64, 4),
        (ti.u64, 3),
        (ti.f64, 5),
    ):
        if hasattr(prog, "vulkan_transform_value_type_available") and not (
            prog.vulkan_transform_value_type_available(value_type)
        ):
            continue
        data, scale, bias, expected = _transform_case(dtype, n)
        src = ti.ndarray(dtype, shape=n)
        dst = ti.ndarray(dtype, shape=n)
        src.from_numpy(data)
        dst.fill(0)
        workspace = ti.algorithms.TransformWorkspace(max_items=n)
        ti.algorithms.experimental_transform(
            src,
            dst,
            scale=scale,
            bias=bias,
            method="vulkan_native",
            workspace=workspace,
        )
        _assert_transform_equal(dtype, dst.to_numpy(), expected)
        assert workspace.workspace_bytes_peak >= 8


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_transform_vulkan_native_dense_field_dtypes():
    n = 4096
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "vulkan_transform_available")
        and prog.vulkan_transform_available()
    ):
        pytest.skip("Vulkan native transform is unavailable in this runtime.")
    prog.vulkan_transform_clear_workspace()

    tested = 0
    for dtype, value_type in (
        (ti.i32, 0),
        (ti.u32, 2),
        (ti.f32, 1),
        (ti.i64, 4),
        (ti.u64, 3),
        (ti.f64, 5),
    ):
        if hasattr(prog, "vulkan_transform_value_type_available") and not (
            prog.vulkan_transform_value_type_available(value_type)
        ):
            continue
        workspace = ti.algorithms.TransformWorkspace(max_items=n)
        _run_dense_field_transform_case(
            dtype, n, method="vulkan_native", workspace=workspace
        )
        if dtype in (ti.i32, ti.u32):
            assert workspace.workspace_bytes_peak <= 8
        else:
            assert workspace.workspace_bytes_peak >= 8
        tested += 1
    assert tested >= 3


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_transform_vulkan_native_dense_field_workspace_replay():
    n = 128
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "vulkan_transform_available")
        and prog.vulkan_transform_available()
    ):
        pytest.skip("Vulkan native transform is unavailable in this runtime.")

    src = ti.field(ti.i32, shape=n)
    dst = ti.field(ti.i32, shape=n)
    workspace = ti.algorithms.TransformWorkspace(max_items=n)
    for base in (0, 17):
        data = (np.arange(n, dtype=np.int32) + base).astype(np.int32)
        src.from_numpy(data)
        dst.fill(0)
        ti.algorithms.experimental_transform(
            src,
            dst,
            scale=3,
            bias=7,
            method="vulkan_native",
            workspace=workspace,
        )
        assert np.array_equal(dst.to_numpy(), data * np.int32(3) + np.int32(7))
    assert workspace._native_transform_plan is not None
    assert workspace._native_transform_plan.backend == "vulkan_native"


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_transform_vulkan_native_struct_member_view():
    n = 4096
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "vulkan_transform_available")
        and prog.vulkan_transform_available()
    ):
        pytest.skip("Vulkan native transform is unavailable in this runtime.")

    for dtype, value_type in (
        (ti.i32, 0),
        (ti.u32, 2),
        (ti.f32, 1),
        (ti.i64, 4),
        (ti.u64, 3),
        (ti.f64, 5),
    ):
        if hasattr(prog, "vulkan_transform_value_type_available") and not (
            prog.vulkan_transform_value_type_available(value_type)
        ):
            continue
        workspace = ti.algorithms.TransformWorkspace(max_items=n)
        _run_struct_member_transform_case(
            dtype, n, method="vulkan_native", workspace=workspace
        )
        assert workspace._native_transform_plan is not None
        assert workspace.workspace_bytes_peak < 28


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_transform_vulkan_native_struct_tensor_member_view():
    n = 4096
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "vulkan_transform_available")
        and prog.vulkan_transform_available()
    ):
        pytest.skip("Vulkan native transform is unavailable in this runtime.")

    for dtype, value_type in ((ti.i32, 0), (ti.f32, 1)):
        if hasattr(prog, "vulkan_transform_value_type_available") and not (
            prog.vulkan_transform_value_type_available(value_type)
        ):
            continue
        workspace = ti.algorithms.TransformWorkspace(max_items=n)
        _run_struct_tensor_member_transform_case(
            dtype, n, method="vulkan_native", workspace=workspace
        )
        assert workspace._native_transform_plan is not None
        assert (
            workspace._native_transform_plan.method_name
            == "vulkan_transform_affine_packed_strided_ndarray"
        )
        assert workspace.workspace_bytes_peak >= 8


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_transform_vulkan_native_nd_shape():
    shape = (32, 17)
    n = int(np.prod(shape, dtype=np.int64))
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "vulkan_transform_available")
        and prog.vulkan_transform_available()
    ):
        pytest.skip("Vulkan native transform is unavailable in this runtime.")

    data, scale, bias, expected = _transform_case(ti.i32, n)
    src = ti.ndarray(ti.i32, shape=shape)
    dst = ti.ndarray(ti.i32, shape=shape)
    src.from_numpy(data.reshape(shape))
    dst.fill(0)
    workspace = ti.algorithms.TransformWorkspace(max_items=n)
    ti.algorithms.experimental_transform(
        src, dst, scale=scale, bias=bias, method="vulkan_native", workspace=workspace
    )
    assert np.array_equal(dst.to_numpy(), expected.reshape(shape))
    assert workspace.workspace_bytes_peak >= 8

    _run_struct_tensor_member_transform_case(
        ti.i32, shape, method="vulkan_native", workspace=workspace
    )


@pytest.mark.run_in_serial
@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_transform_vulkan_native_reset_with_live_ndarray():
    n = 1024
    src = ti.ndarray(ti.i32, shape=n)
    dst = ti.ndarray(ti.i32, shape=n)
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "vulkan_transform_available")
        and prog.vulkan_transform_available()
    ):
        pytest.skip("Vulkan native transform is unavailable in this runtime.")

    src.from_numpy((np.arange(n, dtype=np.int32) - 13).astype(np.int32))
    ti.algorithms.experimental_transform(src, dst, scale=4, bias=1, method="vulkan_native")
    assert dst.to_numpy()[0] == -51
    ti.reset()
    del src, dst
    gc.collect()


@test_utils.test(arch=[ti.cpu])
def test_experimental_transform_cpu_native_ndarray_i32_f32():
    n = 131072
    src_i = ti.ndarray(ti.i32, shape=n)
    dst_i = ti.ndarray(ti.i32, shape=n)
    data_i = (np.arange(n, dtype=np.int32) % 101 - 50).astype(np.int32)
    src_i.from_numpy(data_i)
    workspace = ti.algorithms.TransformWorkspace(max_items=n)
    ti.algorithms.experimental_transform(
        src_i, dst_i, scale=5, bias=-11, method="auto", workspace=workspace
    )
    expected_i = (data_i * np.int32(5) + np.int32(-11)).astype(np.int32)
    assert np.array_equal(dst_i.to_numpy(), expected_i)
    assert workspace.workspace_bytes_peak == 0
    assert impl.get_runtime().prog.cpu_transform_workspace_bytes() == 0

    src_f = ti.ndarray(ti.f32, shape=n)
    dst_f = ti.ndarray(ti.f32, shape=n)
    data_f = (np.arange(n, dtype=np.float32) % 41 - 20) * np.float32(0.125)
    src_f.from_numpy(data_f)
    ti.algorithms.experimental_transform(
        src_f, dst_f, scale=-2.0, bias=0.75, method="cpu_native"
    )
    np.testing.assert_allclose(dst_f.to_numpy(), data_f * -2.0 + 0.75, rtol=1e-6)


@test_utils.test(arch=[ti.cpu])
def test_experimental_transform_cpu_native_dense_field_i32_f32():
    n = 4096
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "cpu_transform_available") and prog.cpu_transform_available()
    ):
        pytest.skip("CPU native transform is unavailable in this build/runtime.")

    src_i = ti.field(ti.i32, shape=n)
    dst_i = ti.field(ti.i32, shape=n)
    data_i = (np.arange(n, dtype=np.int32) % 13 - 6).astype(np.int32)
    src_i.from_numpy(data_i)
    workspace = ti.algorithms.experimental_transform(
        src_i, dst_i, scale=3, bias=-2, method="cpu_native"
    )
    np.testing.assert_array_equal(dst_i.to_numpy(), data_i * 3 - 2)
    assert workspace.workspace_bytes_peak == 0

    src_f = ti.field(ti.f32, shape=n)
    dst_f = ti.field(ti.f32, shape=n)
    data_f = ((np.arange(n, dtype=np.float32) % 11) - 5).astype(np.float32)
    src_f.from_numpy(data_f)
    workspace = ti.algorithms.experimental_transform(
        src_f, dst_f, scale=0.5, bias=-1.0, method="cpu_native"
    )
    np.testing.assert_allclose(dst_f.to_numpy(), data_f * 0.5 - 1.0, rtol=1e-6)
    assert workspace.workspace_bytes_peak == 0


@test_utils.test(arch=[ti.cpu])
def test_experimental_transform_cpu_native_dense_field_workspace_replay():
    n = 128
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "cpu_transform_available") and prog.cpu_transform_available()
    ):
        pytest.skip("CPU native transform is unavailable in this build/runtime.")

    src = ti.field(ti.i32, shape=n)
    dst = ti.field(ti.i32, shape=n)
    workspace = ti.algorithms.TransformWorkspace(max_items=n)
    for base in (0, 17):
        data = (np.arange(n, dtype=np.int32) + base).astype(np.int32)
        src.from_numpy(data)
        ti.algorithms.experimental_transform(
            src, dst, scale=3, bias=-2, method="cpu_native", workspace=workspace
        )
        np.testing.assert_array_equal(dst.to_numpy(), data * 3 - 2)
    assert workspace._native_transform_plan is not None
    assert workspace._native_transform_plan.backend == "cpu_native"


@test_utils.test(arch=[ti.cpu])
def test_experimental_transform_cpu_native_ndarray_workspace_replay():
    n = 128
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "cpu_transform_available") and prog.cpu_transform_available()
    ):
        pytest.skip("CPU native transform is unavailable in this build/runtime.")

    src = ti.ndarray(ti.i32, shape=n)
    dst = ti.ndarray(ti.i32, shape=n)
    workspace = ti.algorithms.TransformWorkspace(max_items=n)
    first_plan = None
    for base in (0, 17):
        data = (np.arange(n, dtype=np.int32) + base).astype(np.int32)
        src.from_numpy(data)
        ti.algorithms.experimental_transform(
            src, dst, scale=3, bias=-2, method="cpu_native", workspace=workspace
        )
        np.testing.assert_array_equal(dst.to_numpy(), data * 3 - 2)
        if first_plan is None:
            first_plan = workspace._native_transform_plan
        else:
            assert workspace._native_transform_plan is first_plan
    assert workspace._native_transform_plan.backend == "cpu_native"
    assert (
        workspace._native_transform_plan.method_name
        == "cpu_transform_affine_ndarray"
    )


@test_utils.test(arch=[ti.cpu])
def test_experimental_transform_cpu_native_struct_member_workspace_replay():
    n = 128
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "cpu_transform_available") and prog.cpu_transform_available()
    ):
        pytest.skip("CPU native transform is unavailable in this build/runtime.")

    payload = ti.types.struct(value=ti.i32, tag=ti.i32)
    src = ti.ndarray(payload, shape=n)
    dst = ti.ndarray(ti.i32, shape=n)
    workspace = ti.algorithms.TransformWorkspace(max_items=n)
    first_plan = None
    for base in (0, 17):
        data = (np.arange(n, dtype=np.int32) + base).astype(np.int32)
        host = np.zeros((n,), dtype=src.numpy_dtype)
        host["value"] = data
        host["tag"] = np.arange(n, dtype=np.int32) * 3 + 1
        src.from_numpy(host)
        ti.algorithms.experimental_transform(
            src.field("value"),
            dst,
            scale=3,
            bias=-2,
            method="cpu_native",
            workspace=workspace,
        )
        np.testing.assert_array_equal(dst.to_numpy(), data * 3 - 2)
        if first_plan is None:
            first_plan = workspace._native_transform_plan
        else:
            assert workspace._native_transform_plan is first_plan
    assert workspace._native_transform_plan.backend == "cpu_native"
    assert (
        workspace._native_transform_plan.method_name
        == "cpu_transform_affine_strided_ndarray"
    )


@test_utils.test(arch=[ti.cpu])
def test_experimental_transform_cpu_native_struct_member_multi_plan_cache():
    n = 128
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "cpu_transform_available") and prog.cpu_transform_available()
    ):
        pytest.skip("CPU native transform is unavailable in this build/runtime.")

    payload = ti.types.struct(value=ti.i32, tag=ti.i32)
    src = ti.ndarray(payload, shape=n)
    dst = ti.ndarray(ti.i32, shape=n)
    host = np.zeros((n,), dtype=src.numpy_dtype)
    host["value"] = np.arange(n, dtype=np.int32) * 2 - 9
    host["tag"] = np.arange(n, dtype=np.int32) * 3 + 1
    src.from_numpy(host)
    workspace = ti.algorithms.TransformWorkspace(max_items=n)

    ti.algorithms.experimental_transform(
        src.field("value"),
        dst,
        scale=2,
        bias=1,
        method="cpu_native",
        workspace=workspace,
    )
    first_plan = workspace._native_transform_plan

    ti.algorithms.experimental_transform(
        src.field("value"),
        dst,
        scale=5,
        bias=-3,
        method="cpu_native",
        workspace=workspace,
    )
    second_plan = workspace._native_transform_plan
    assert second_plan is not first_plan
    assert len(workspace._native_transform_plans) == 2

    ti.algorithms.experimental_transform(
        src.field("value"),
        dst,
        scale=2,
        bias=1,
        method="cpu_native",
        workspace=workspace,
    )
    assert workspace._native_transform_plan is first_plan
    np.testing.assert_array_equal(dst.to_numpy(), host["value"] * 2 + 1)


@test_utils.test(arch=[ti.cpu])
def test_experimental_transform_cpu_native_ndarray_extended_dtypes():
    n = 131072
    for dtype in _TRANSFORM_DTYPES:
        data, scale, bias, expected = _transform_case(dtype, n)
        src = ti.ndarray(dtype, shape=n)
        dst = ti.ndarray(dtype, shape=n)
        src.from_numpy(data)
        dst.fill(0)
        workspace = ti.algorithms.TransformWorkspace(max_items=n)
        ti.algorithms.experimental_transform(
            src, dst, scale=scale, bias=bias, method="cpu_native", workspace=workspace
        )
        _assert_transform_equal(dtype, dst.to_numpy(), expected)
        assert workspace.workspace_bytes_peak == 0


@test_utils.test(arch=[ti.cpu])
def test_experimental_transform_cpu_native_struct_member_view():
    n = 131072
    for dtype in _TRANSFORM_DTYPES:
        workspace = ti.algorithms.TransformWorkspace(max_items=n)
        _run_struct_member_transform_case(
            dtype, n, method="cpu_native", workspace=workspace
        )
        assert workspace.workspace_bytes_peak == 0


@test_utils.test(arch=[ti.cpu])
def test_experimental_transform_cpu_native_struct_tensor_member_view():
    n = 131072
    for dtype in (ti.i32, ti.f32):
        workspace = ti.algorithms.TransformWorkspace(max_items=n)
        _run_struct_tensor_member_transform_case(
            dtype, n, method="cpu_native", workspace=workspace
        )
        assert workspace.workspace_bytes_peak == 0


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_transform_native_dense_matrix_field_components():
    _run_dense_matrix_field_transform_case()


@test_utils.test(arch=[ti.cpu])
def test_experimental_transform_cpu_native_struct_tensor_member_workspace_replay():
    n = 128
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "cpu_transform_available")
        and prog.cpu_transform_available()
        and hasattr(prog, "cpu_transform_affine_packed_strided_ndarray")
    ):
        pytest.skip("CPU packed tensor member transform is unavailable.")

    payload = ti.types.struct(vec=ti.types.vector(2, ti.i32), tag=ti.i32)
    src = ti.ndarray(payload, shape=n)
    dst = ti.ndarray(payload, shape=n)
    workspace = ti.algorithms.TransformWorkspace(max_items=n)
    first_plan = None
    for base in (0, 17):
        values = (np.arange(n * 2, dtype=np.int32).reshape(n, 2) + base).astype(
            np.int32
        )
        src_host = np.zeros((n,), dtype=src.numpy_dtype)
        dst_host = np.zeros((n,), dtype=dst.numpy_dtype)
        src_host["vec"] = values
        src_host["tag"] = np.arange(n, dtype=np.int32) * 3 + 1
        dst_host["tag"] = np.arange(n, dtype=np.int32) * 11 - 3
        src.from_numpy(src_host)
        dst.from_numpy(dst_host)
        ti.algorithms.experimental_transform(
            src.field("vec"),
            dst.field("vec"),
            scale=3,
            bias=7,
            method="cpu_native",
            workspace=workspace,
        )
        result = dst.to_numpy()
        np.testing.assert_array_equal(result["vec"], values * 3 + 7)
        np.testing.assert_array_equal(result["tag"], dst_host["tag"])
        if first_plan is None:
            first_plan = workspace._native_transform_plan
        else:
            assert workspace._native_transform_plan is first_plan
    assert workspace._native_transform_plan.backend == "cpu_native"
    assert (
        workspace._native_transform_plan.method_name
        == "cpu_transform_affine_packed_strided_ndarray"
    )


@test_utils.test(arch=[ti.cpu])
def test_experimental_transform_cpu_native_nd_shape():
    shape = (32, 17)
    n = int(np.prod(shape, dtype=np.int64))
    data, scale, bias, expected = _transform_case(ti.f32, n)
    src = ti.ndarray(ti.f32, shape=shape)
    dst = ti.ndarray(ti.f32, shape=shape)
    src.from_numpy(data.reshape(shape))
    dst.fill(0)
    workspace = ti.algorithms.TransformWorkspace(max_items=n)
    ti.algorithms.experimental_transform(
        src, dst, scale=scale, bias=bias, method="cpu_native", workspace=workspace
    )
    np.testing.assert_allclose(dst.to_numpy(), expected.reshape(shape), rtol=1e-6)
    assert workspace.workspace_bytes_peak == 0

    _run_struct_tensor_member_transform_case(
        ti.f32, shape, method="cpu_native", workspace=workspace
    )
    assert workspace.workspace_bytes_peak == 0


@test_utils.test(arch=get_host_arch_list())
def test_experimental_transform_struct_member_view_rejections():
    payload = ti.types.struct(value=ti.f32, tag=ti.i32)
    src = ti.ndarray(payload, shape=8)
    view = src.field("value")
    scalar_dst = ti.ndarray(ti.f32, shape=8)
    struct_dst = ti.ndarray(payload, shape=8)

    with pytest.raises(TypeError, match="does not support StructNdarray"):
        ti.algorithms.experimental_transform(src, scalar_dst, method="auto")
    with pytest.raises(TypeError, match="does not support StructNdarray"):
        ti.algorithms.experimental_transform(view, struct_dst, method="auto")
    with pytest.raises(RuntimeError, match="member views require"):
        ti.algorithms.experimental_transform(
            view, scalar_dst, method="field_kernel"
        )


@test_utils.test(arch=[ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_transform_field_kernel_i32_f32():
    n = 2048
    src_i = ti.field(ti.i32, shape=n)
    dst_i = ti.field(ti.i32, shape=n)
    data_i = (np.arange(n, dtype=np.int32) % 29 - 14).astype(np.int32)
    src_i.from_numpy(data_i)
    ti.algorithms.experimental_transform(
        src_i, dst_i, scale=2, bias=5, method="field_kernel"
    )
    assert np.array_equal(dst_i.to_numpy(), data_i * 2 + 5)

    src_f = ti.field(ti.f32, shape=n)
    dst_f = ti.field(ti.f32, shape=n)
    data_f = (np.arange(n, dtype=np.float32) % 19 - 9) * np.float32(0.25)
    src_f.from_numpy(data_f)
    ti.algorithms.experimental_transform(
        src_f, dst_f, scale=1.25, bias=-0.5, method="field_kernel"
    )
    np.testing.assert_allclose(dst_f.to_numpy(), data_f * 1.25 - 0.5, rtol=1e-6)
