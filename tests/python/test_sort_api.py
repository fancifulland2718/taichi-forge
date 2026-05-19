import pytest
import numpy as np

import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu])
def test_sort_entrypoint_auto_uses_host_stable_fallback():
    keys = ti.field(ti.i32, 8)
    values = ti.field(ti.i32, 8)

    @ti.kernel
    def fill():
        for i in keys:
            keys[i] = 8 - i
            values[i] = i

    fill()
    ti.algorithms.sort(keys, values)

    assert keys.to_numpy().tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
    assert values.to_numpy().tolist() == [7, 6, 5, 4, 3, 2, 1, 0]


@test_utils.test(arch=[ti.cpu])
def test_sort_by_key_single_part_routes_to_sort():
    keys = ti.field(ti.i32, 4)
    values = ti.field(ti.i32, 4)

    @ti.kernel
    def fill():
        keys[0] = 3
        keys[1] = 1
        keys[2] = 2
        keys[3] = 0
        for i in values:
            values[i] = i

    fill()
    ti.algorithms.sort_by_key([keys], values)

    assert keys.to_numpy().tolist() == [0, 1, 2, 3]
    assert values.to_numpy().tolist() == [3, 1, 2, 0]


@test_utils.test(arch=[ti.cpu])
def test_sort_by_key_multi_part_is_stable():
    primary = ti.field(ti.i32, 6)
    secondary = ti.field(ti.i32, 6)
    values = ti.field(ti.i32, 6)

    @ti.kernel
    def fill():
        primary[0], secondary[0], values[0] = 1, 2, 0
        primary[1], secondary[1], values[1] = 0, 3, 1
        primary[2], secondary[2], values[2] = 1, 1, 2
        primary[3], secondary[3], values[3] = 0, 2, 3
        primary[4], secondary[4], values[4] = 1, 1, 4
        primary[5], secondary[5], values[5] = 0, 2, 5

    fill()
    ti.algorithms.sort_by_key([primary, secondary], values)

    assert primary.to_numpy().tolist() == [0, 0, 0, 1, 1, 1]
    assert secondary.to_numpy().tolist() == [2, 2, 3, 1, 1, 2]
    assert values.to_numpy().tolist() == [3, 5, 1, 2, 4, 0]


@test_utils.test(arch=[ti.cpu])
def test_sort_auto_is_stable_for_duplicate_keys():
    keys = ti.field(ti.i32, 8)
    values = ti.field(ti.i32, 8)

    @ti.kernel
    def fill():
        for i in keys:
            keys[i] = i % 2
            values[i] = i

    fill()
    ti.algorithms.sort(keys, values)

    assert keys.to_numpy().tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert values.to_numpy().tolist() == [0, 2, 4, 6, 1, 3, 5, 7]


@test_utils.test(arch=[ti.cpu])
def test_sort_host_stable_descending_preserves_duplicate_order():
    keys = ti.field(ti.i32, 8)
    values = ti.field(ti.i32, 8)

    @ti.kernel
    def fill():
        keys[0], values[0] = 1, 0
        keys[1], values[1] = 3, 1
        keys[2], values[2] = 2, 2
        keys[3], values[3] = 3, 3
        keys[4], values[4] = 1, 4
        keys[5], values[5] = 0, 5
        keys[6], values[6] = 2, 6
        keys[7], values[7] = 3, 7

    fill()
    ti.algorithms.sort(keys, values, descending=True)

    assert keys.to_numpy().tolist() == [3, 3, 3, 2, 2, 1, 1, 0]
    assert values.to_numpy().tolist() == [1, 3, 7, 2, 6, 0, 4, 5]


@test_utils.test(arch=[ti.cpu])
def test_sort_host_stable_descending_keeps_nan_last():
    keys = ti.field(ti.f32, 6)
    values = ti.field(ti.i32, 6)

    @ti.kernel
    def fill():
        keys[0], values[0] = 1.0, 0
        keys[1], values[1] = float("nan"), 1
        keys[2], values[2] = 3.0, 2
        keys[3], values[3] = 3.0, 3
        keys[4], values[4] = -2.0, 4
        keys[5], values[5] = float("nan"), 5

    fill()
    ti.algorithms.sort(keys, values, descending=True, nan_policy="last")

    sorted_keys = keys.to_numpy()
    assert sorted_keys[:4].tolist() == [3.0, 3.0, 1.0, -2.0]
    assert values.to_numpy().tolist() == [2, 3, 0, 4, 1, 5]
    assert sorted_keys[4] != sorted_keys[4]
    assert sorted_keys[5] != sorted_keys[5]


@test_utils.test(arch=[ti.cpu])
def test_sort_cpu_native_ndarray_descending():
    keys_np = np.array([2, 5, 2, -1, 5, 0], dtype=np.int32)
    values_np = np.arange(keys_np.shape[0], dtype=np.int32)
    keys = ti.ndarray(ti.i32, shape=keys_np.shape[0])
    values = ti.ndarray(ti.i32, shape=keys_np.shape[0])
    keys.from_numpy(keys_np)
    values.from_numpy(values_np)

    ti.algorithms.sort(keys, values, method="cpu_native", descending=True)

    assert keys.to_numpy().tolist() == [5, 5, 2, 2, 0, -1]
    assert values.to_numpy().tolist() == [1, 4, 0, 2, 5, 3]


@test_utils.test(arch=[ti.cpu])
def test_sort_auto_keys_only():
    keys = ti.field(ti.i32, 8)

    @ti.kernel
    def fill():
        keys[0] = 4
        keys[1] = 1
        keys[2] = 3
        keys[3] = 1
        keys[4] = 2
        keys[5] = 0
        keys[6] = 4
        keys[7] = 2

    fill()
    ti.algorithms.sort(keys)

    assert keys.to_numpy().tolist() == [0, 1, 1, 2, 2, 3, 4, 4]


@pytest.mark.parametrize("method", ["cuda_cub_native", "cuda_cub_split32"])
@test_utils.test(arch=[ti.cuda])
def test_sort_cuda_cub_wide_exact_methods(method):
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    if not impl.get_runtime().prog.cuda_cub_radix_sort_available():
        pytest.skip("CUDA CUB radix sort is not available.")

    keys_np = np.array([3.5, np.nan, -2.0, 1.0, 3.5, 0.0], dtype=np.float64)
    values_np = np.arange(keys_np.shape[0], dtype=np.int32)
    keys = ti.ndarray(ti.f64, shape=keys_np.shape[0])
    values = ti.ndarray(ti.i32, shape=keys_np.shape[0])
    keys.from_numpy(keys_np)
    values.from_numpy(values_np)

    ti.algorithms.sort(keys, values, method=method, nan_policy="last")

    actual_keys = keys.to_numpy()
    assert np.array_equal(
        actual_keys,
        np.array([-2.0, 0.0, 1.0, 3.5, 3.5, np.nan], dtype=np.float64),
        equal_nan=True,
    )
    assert values.to_numpy().tolist() == [2, 5, 3, 0, 4, 1]


@test_utils.test(arch=[ti.vulkan])
def test_sort_vulkan_native_radix_i32_ndarray():
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    if not impl.get_runtime().prog.vulkan_radix_sort_available():
        pytest.skip("Vulkan native radix sort is not available.")

    keys_np = np.array([4, -2, 4, 0, -7, 3, -2, 1], dtype=np.int32)
    values_np = np.arange(keys_np.shape[0], dtype=np.int32)
    keys = ti.ndarray(ti.i32, shape=keys_np.shape[0])
    values = ti.ndarray(ti.i32, shape=keys_np.shape[0])
    keys.from_numpy(keys_np)
    values.from_numpy(values_np)

    ti.algorithms.sort(keys, values, method="vulkan_native_radix_u32")

    assert keys.to_numpy().tolist() == [-7, -2, -2, 0, 1, 3, 4, 4]
    assert values.to_numpy().tolist() == [4, 1, 6, 3, 7, 5, 0, 2]


@pytest.mark.parametrize("dtype", [ti.i64, ti.u64, ti.f64])
@test_utils.test(arch=[ti.cpu])
def test_sort_auto_supports_wide_key_dtypes(dtype):
    keys = ti.field(dtype, 4)
    values = ti.field(ti.i32, 4)

    @ti.kernel
    def fill():
        keys[0] = 3
        keys[1] = 1
        keys[2] = 2
        keys[3] = 0
        for i in values:
            values[i] = i

    fill()
    ti.algorithms.sort(keys, values)

    assert keys.to_numpy().tolist() == [0, 1, 2, 3]
    assert values.to_numpy().tolist() == [3, 1, 2, 0]


def test_sort_workspace_metadata_only():
    workspace = ti.algorithms.SortWorkspace(max_items=16, device="test")
    assert workspace.max_items == 16
    assert workspace.device == "test"
    assert workspace.workspace_bytes_current == 0
    assert workspace.workspace_bytes_peak == 0

    assert workspace.reserve(dtype=ti.i32, value_dtype=ti.i32, n=8) is workspace
    workspace.clear()
    assert workspace.workspace_bytes_current == 0
    assert workspace.workspace_bytes_peak == 0

    with pytest.raises(ValueError):
        workspace.reserve(n=17)


@test_utils.test(arch=[ti.cpu])
def test_sort_rejects_unimplemented_modes():
    keys = ti.field(ti.i32, 4)

    with pytest.raises(NotImplementedError):
        ti.algorithms.sort(keys, stable=False)
    with pytest.raises(RuntimeError):
        ti.algorithms.sort(keys, method="cuda_cub_native")
    with pytest.raises(RuntimeError):
        ti.algorithms.sort(keys, method="cuda_cub_u32")
    with pytest.raises(NotImplementedError):
        ti.algorithms.sort(keys, precision="approx")
    with pytest.raises(ValueError):
        ti.algorithms.sort(keys, nan_policy="middle")
    with pytest.raises(RuntimeError):
        ti.algorithms.sort(keys, method="radix_u32")
    with pytest.raises(NotImplementedError):
        ti.algorithms.sort(keys, method="legacy", descending=True)
    with pytest.raises(NotImplementedError):
        ti.algorithms.sort_by_key([keys, keys], method="legacy")
