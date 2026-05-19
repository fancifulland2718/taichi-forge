import gc

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils


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
    assert workspace.workspace_bytes_peak >= 8

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
