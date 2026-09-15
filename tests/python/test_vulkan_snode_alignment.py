"""Mixed-width field layout must agree across shaders and native transfers."""

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge._kernels import ext_arr_to_tensor, tensor_to_ext_arr
from tests import test_utils


def _run_supported_dtype(operation, dtype):
    # The backend-wide data64 extension also requires atomics. These tests only
    # need scalar loads/stores; skip solely the device's exact unsupported type.
    try:
        operation()
    except (RuntimeError, ti.TaichiCompilationError) as exc:
        if f"Type {dtype} not supported." in str(exc):
            pytest.skip(f"Vulkan device does not support {dtype}")
        raise


@pytest.mark.parametrize("dtype,np_dtype", [(ti.i64, np.int64), (ti.u64, np.uint64), (ti.f64, np.float64)])
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_mixed_width_dense_shader_native_roundtrip(dtype, np_dtype):
    sentinel = ti.field(ti.i32, shape=())
    values = ti.field(dtype, shape=(3, 5))
    sentinel[None] = 12345
    if np_dtype == np.float64:
        expected = (np.arange(15, dtype=np.float64) / 4 + 0.125).reshape(3, 5)
    else:
        indices = np.arange(15, dtype=np_dtype)
        expected = (((indices + 0x12345678) << 32) | (indices + 17)).reshape(3, 5)
    exported = np.zeros_like(expected)

    # Kernel writes and native raw readback are deliberately independent paths.
    _run_supported_dtype(lambda: ext_arr_to_tensor(expected, values), dtype)
    ti.sync()
    np.testing.assert_array_equal(values.to_numpy(), expected)
    assert sentinel[None] == 12345

    values.from_numpy(expected + 3)
    tensor_to_ext_arr(values, exported)
    ti.sync()
    np.testing.assert_array_equal(exported, expected + 3)
    assert sentinel[None] == 12345

    values.fill(19)
    tensor_to_ext_arr(values, exported)
    ti.sync()
    np.testing.assert_array_equal(exported, np.full_like(expected, 19))
    assert sentinel[None] == 12345


@pytest.mark.parametrize("container", ["dense", "bitmasked", "dynamic", "pointer", "hash"])
@pytest.mark.parametrize("width", [1, 3])
@test_utils.test(arch=ti.vulkan, offline_cache=False, hash_snode_diagnostics=True)
def test_nested_mixed_width_cells_and_sparse_payloads(container, width):
    sentinel = ti.field(ti.i32, shape=())
    tag, value = ti.field(ti.i32), ti.field(ti.i64)
    outer = ti.root.dense(ti.i, 3)
    if container == "dynamic":
        inner = outer.dynamic(ti.j, width, chunk_size=width)
    elif container == "hash":
        inner = outer.hash(ti.j, width, capacity=4)
    else:
        inner = getattr(outer, container)(ti.j, width)
    inner.place(tag, value)
    sentinel[None] = 12345

    @ti.kernel
    def write():
        for i, j in ti.ndrange(3, width):
            n = i * width + j
            tag[i, j] = 700 + n
            value[i, j] = ((ti.i64(0x12345678) + n) << 32) | (17 + n)

    _run_supported_dtype(write, ti.i64)
    indices = np.arange(3 * width, dtype=np.int64)
    expected = ((indices + 0x12345678) << 32) | (indices + 17)
    np.testing.assert_array_equal(value.to_numpy().reshape(-1), expected)
    np.testing.assert_array_equal(tag.to_numpy().reshape(-1), indices + 700)
    assert sentinel[None] == 12345
    if container != "dense":
        ti.deactivate_all_snodes()
        write()
        np.testing.assert_array_equal(value.to_numpy().reshape(-1), expected)
        np.testing.assert_array_equal(tag.to_numpy().reshape(-1), indices + 700)
        assert sentinel[None] == 12345
