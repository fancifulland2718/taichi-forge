import pytest
import numpy as np

import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils

_SORT_VALUE_DTYPE_CASES = [
    (ti.i32, np.int32),
    (ti.u32, np.uint32),
    (ti.f32, np.float32),
    (ti.u64, np.uint64),
    (ti.i64, np.int64),
    (ti.f64, np.float64),
]

_SORT_KEY_DTYPE_CASES = [
    (ti.i32, np.int32),
    (ti.u32, np.uint32),
    (ti.f32, np.float32),
    (ti.i64, np.int64),
    (ti.u64, np.uint64),
    (ti.f64, np.float64),
]


def _sort_payload_values(np_dtype):
    if np.issubdtype(np_dtype, np.unsignedinteger):
        return np.array([10, 40, 20, 30, 60, 50], dtype=np_dtype)
    if np.issubdtype(np_dtype, np.floating):
        return np.array([1.5, -4.0, 2.25, 3.0, 6.5, -5.5], dtype=np_dtype)
    return np.array([10, -40, 20, 30, -60, 50], dtype=np_dtype)


def _sort_key_values(np_dtype):
    if np_dtype == np.uint32:
        return np.array([5, 1, 5, 0, 2, 1, 9, 2], dtype=np_dtype)
    if np_dtype == np.int32:
        return np.array([4, -2, 4, 0, -7, 3, -2, 1], dtype=np_dtype)
    if np_dtype == np.float32:
        return np.array([3.5, np.nan, -0.0, 1.0, 3.5, 0.0, -2.0, np.nan], dtype=np_dtype)
    if np_dtype == np.uint64:
        return np.array(
            [2**40 + 3, 7, 2**32 + 1, 0, 2**40 + 3, 2**32, 7, 5],
            dtype=np_dtype,
        )
    if np_dtype == np.int64:
        return np.array([2**40, -2**40, 7, -1, 0, 7, -2**40, 2], dtype=np_dtype)
    return np.array([3.5, np.nan, -0.0, 1.0, 3.5, 0.0, -2.0, np.nan], dtype=np_dtype)


def _expected_sort_order(keys_np):
    if keys_np.dtype == np.float32:
        bits = keys_np.view(np.uint32)
        sign = np.uint32(0x80000000)
        abs_mask = np.uint32(0x7FFFFFFF)
        sortable = np.where((bits & sign) != 0, ~bits, bits ^ sign).astype(np.uint32)
        sortable = np.where((bits & abs_mask) > np.uint32(0x7F800000), np.uint32(0xFFFFFFFF), sortable)
        sortable = np.where((bits & abs_mask) == 0, sign, sortable)
        return np.argsort(sortable, kind="stable")
    if keys_np.dtype == np.float64:
        bits = keys_np.view(np.uint64)
        sign = np.uint64(0x8000000000000000)
        abs_mask = np.uint64(0x7FFFFFFFFFFFFFFF)
        sortable = np.where((bits & sign) != 0, ~bits, bits ^ sign).astype(np.uint64)
        sortable = np.where(
            (bits & abs_mask) > np.uint64(0x7FF0000000000000),
            np.uint64(0xFFFFFFFFFFFFFFFF),
            sortable,
        )
        sortable = np.where((bits & abs_mask) == 0, sign, sortable)
        return np.argsort(sortable, kind="stable")
    return np.argsort(keys_np, kind="stable")


def _run_sort_payload_dtype_case(method, value_dtype, np_dtype):
    keys_np = np.array([3, -1, 3, 0, -7, 2], dtype=np.int32)
    values_np = _sort_payload_values(np_dtype)
    order = np.argsort(keys_np, kind="stable")
    keys = ti.ndarray(ti.i32, shape=keys_np.shape[0])
    values = ti.ndarray(value_dtype, shape=keys_np.shape[0])
    keys.from_numpy(keys_np)
    values.from_numpy(values_np)

    ti.algorithms.sort(keys, values, method=method)

    assert keys.to_numpy().tolist() == keys_np[order].tolist()
    assert np.array_equal(values.to_numpy(), values_np[order])


def _run_sort_vector_payload_case(method):
    keys_np = np.array([3, -1, 3, 0, -7, 2, -1, 3], dtype=np.int32)
    values_np = (
        ((np.arange(keys_np.shape[0] * 3, dtype=np.float32).reshape(-1, 3) % 47)
         - 21)
        * np.float32(0.5)
    )
    order = np.argsort(keys_np, kind="stable")
    keys = ti.ndarray(ti.i32, shape=keys_np.shape[0])
    values = ti.Vector.ndarray(3, ti.f32, shape=keys_np.shape[0])
    keys.from_numpy(keys_np)
    values.from_numpy(values_np)

    ti.algorithms.sort(keys, values, method=method)

    assert keys.to_numpy().tolist() == keys_np[order].tolist()
    np.testing.assert_allclose(values.to_numpy(), values_np[order], rtol=1e-6, atol=1e-6)


def _run_dense_field_sort_case(method):
    keys_np = np.array([3, -1, 3, 0, -7, 2, -1, 3], dtype=np.int32)
    values_np = (np.arange(keys_np.shape[0], dtype=np.int32) * 7) - 3
    order = np.argsort(keys_np, kind="stable")
    keys = ti.field(ti.i32, shape=keys_np.shape[0])
    values = ti.field(ti.i32, shape=keys_np.shape[0])
    keys.from_numpy(keys_np)
    values.from_numpy(values_np)
    workspace = ti.algorithms.SortWorkspace(max_items=keys_np.shape[0])

    ti.algorithms.sort(keys, values, method=method, workspace=workspace)

    assert keys.to_numpy().tolist() == keys_np[order].tolist()
    assert values.to_numpy().tolist() == values_np[order].tolist()
    assert workspace.workspace_bytes_peak >= 0
    return workspace


def _run_sort_by_key_native_ndarray_case(method):
    primary_np = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int32)
    secondary_np = np.array([2, 3, 1, 2, 1, 2, 0, 3], dtype=np.int32)
    values_np = (np.arange(primary_np.shape[0], dtype=np.int32) * 11) - 5
    order = np.lexsort((np.arange(primary_np.shape[0]), secondary_np, primary_np))
    primary = ti.ndarray(ti.i32, shape=primary_np.shape[0])
    secondary = ti.ndarray(ti.i32, shape=secondary_np.shape[0])
    values = ti.ndarray(ti.i32, shape=values_np.shape[0])
    primary.from_numpy(primary_np)
    secondary.from_numpy(secondary_np)
    values.from_numpy(values_np)
    workspace = ti.algorithms.SortWorkspace(max_items=primary_np.shape[0])

    ti.algorithms.sort_by_key(
        [primary, secondary], values, method=method, workspace=workspace
    )

    assert primary.to_numpy().tolist() == primary_np[order].tolist()
    assert secondary.to_numpy().tolist() == secondary_np[order].tolist()
    assert values.to_numpy().tolist() == values_np[order].tolist()
    assert workspace._order_apply_indexed_copy_workspace is not None
    assert workspace._order_apply_transform_workspace is not None


def _run_sort_struct_tensor_member_host_case(method, workspace=None):
    keys_np = np.array([3, -1, 3, 0, -7, 2, -1, 3], dtype=np.int32)
    order = np.argsort(keys_np, kind="stable")
    payload = ti.types.struct(
        vec=ti.types.vector(2, ti.i32),
        mat=ti.types.matrix(2, 2, ti.i32),
        tag=ti.i32,
    )
    keys = ti.ndarray(ti.i32, shape=keys_np.shape[0])
    values = ti.ndarray(payload, shape=keys_np.shape[0])
    values_np = np.zeros((keys_np.shape[0],), dtype=values.numpy_dtype)
    values_np["vec"] = (
        np.arange(keys_np.shape[0] * 2, dtype=np.int32).reshape(-1, 2) % 37
    ) - 18
    values_np["mat"] = (
        np.arange(keys_np.shape[0] * 4, dtype=np.int32).reshape(-1, 2, 2) % 41
    ) - 20
    values_np["tag"] = np.arange(keys_np.shape[0], dtype=np.int32) * 5 + 1
    keys.from_numpy(keys_np)
    values.from_numpy(values_np)

    ti.algorithms.sort(keys, values.field("vec"), method=method, workspace=workspace)

    result = values.to_numpy()
    assert keys.to_numpy().tolist() == keys_np[order].tolist()
    assert np.array_equal(result["vec"], values_np["vec"][order])
    assert np.array_equal(result["mat"], values_np["mat"])
    assert np.array_equal(result["tag"], values_np["tag"])

    keys.from_numpy(keys_np)
    values.from_numpy(values_np)
    ti.algorithms.sort(keys, values.field("mat"), method=method, workspace=workspace)
    result = values.to_numpy()
    assert keys.to_numpy().tolist() == keys_np[order].tolist()
    assert np.array_equal(result["vec"], values_np["vec"])
    assert np.array_equal(result["mat"], values_np["mat"][order])
    assert np.array_equal(result["tag"], values_np["tag"])
    return workspace


@test_utils.test(arch=[ti.cpu])
def test_sort_entrypoint_auto_sorts_dense_field():
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
def test_sort_cpu_native_dense_field_default_ready():
    _run_dense_field_sort_case("cpu_native")


@test_utils.test(arch=[ti.cpu])
def test_sort_auto_cpu_ndarray_uses_native_default():
    keys_np = np.array([3, -1, 3, 0, -7, 2], dtype=np.int32)
    values_np = np.arange(keys_np.shape[0], dtype=np.int32) * 3 - 1
    order = np.argsort(keys_np, kind="stable")
    keys = ti.ndarray(ti.i32, shape=keys_np.shape[0])
    values = ti.ndarray(ti.i32, shape=values_np.shape[0])
    keys.from_numpy(keys_np)
    values.from_numpy(values_np)
    workspace = ti.algorithms.SortWorkspace(max_items=keys_np.shape[0])

    ti.algorithms.sort(keys, values, workspace=workspace)

    assert keys.to_numpy().tolist() == keys_np[order].tolist()
    assert values.to_numpy().tolist() == values_np[order].tolist()
    assert workspace.workspace_bytes_peak > 0


@test_utils.test(arch=[ti.cpu])
def test_sort_auto_cpu_ndarray_descending_uses_native_default():
    keys_np = np.array([2, 5, 2, -1, 5, 0], dtype=np.int32)
    values_np = np.arange(keys_np.shape[0], dtype=np.int32)
    keys = ti.ndarray(ti.i32, shape=keys_np.shape[0])
    values = ti.ndarray(ti.i32, shape=values_np.shape[0])
    keys.from_numpy(keys_np)
    values.from_numpy(values_np)
    workspace = ti.algorithms.SortWorkspace(max_items=keys_np.shape[0])

    ti.algorithms.sort(keys, values, descending=True, workspace=workspace)

    assert keys.to_numpy().tolist() == [5, 5, 2, 2, 0, -1]
    assert values.to_numpy().tolist() == [1, 4, 0, 2, 5, 3]
    assert workspace.workspace_bytes_peak > 0


@test_utils.test(arch=[ti.cuda])
def test_sort_cuda_cub_dense_field_default_ready():
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "cuda_cub_radix_sort_available")
        and prog.cuda_cub_radix_sort_available()
        and hasattr(prog, "cuda_cub_radix_sort_dense_field")
    ):
        pytest.skip("CUDA CUB dense field sort is unavailable in this runtime.")
    _run_dense_field_sort_case("cuda_cub_native")


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_sort_vulkan_native_dense_field_default_ready():
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "vulkan_radix_sort_available")
        and prog.vulkan_radix_sort_available()
        and hasattr(prog, "vulkan_radix_sort_u32_dense_field")
    ):
        pytest.skip("Vulkan dense field radix sort is unavailable in this runtime.")
    workspace = _run_dense_field_sort_case("auto")
    assert workspace._vulkan_native_active
    workspace = _run_dense_field_sort_case("vulkan_native_radix_u32")
    assert workspace._vulkan_native_active


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_sort_vulkan_native_dense_field_nonzero_root_offset():
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "vulkan_radix_sort_available")
        and prog.vulkan_radix_sort_available()
        and hasattr(prog, "vulkan_radix_sort_u32_dense_field")
    ):
        pytest.skip("Vulkan dense field radix sort is unavailable in this runtime.")

    keys_np = np.array([4, -2, 4, 1, -9, 0, -2, 3], dtype=np.int32)
    values_np = (np.arange(keys_np.shape[0], dtype=np.int32) * 5) + 1
    order = np.argsort(keys_np, kind="stable")
    pad = ti.field(ti.i32)
    keys = ti.field(ti.i32)
    values = ti.field(ti.i32)
    ti.root.place(pad)
    ti.root.dense(ti.i, keys_np.shape[0]).place(keys)
    ti.root.dense(ti.i, keys_np.shape[0]).place(values)
    values.from_numpy(values_np)
    keys.from_numpy(keys_np)
    workspace = ti.algorithms.SortWorkspace(max_items=keys_np.shape[0])

    ti.algorithms.sort(
        keys, values, method="vulkan_native_radix_u32", workspace=workspace
    )

    assert keys.to_numpy().tolist() == keys_np[order].tolist()
    assert values.to_numpy().tolist() == values_np[order].tolist()
    assert workspace._vulkan_native_active


@test_utils.test(arch=[ti.cpu])
def test_sort_struct_tensor_member_values_host_stable():
    _run_sort_struct_tensor_member_host_case("host_stable")


@test_utils.test(arch=[ti.cpu])
def test_sort_struct_tensor_member_values_cpu_native():
    _run_sort_struct_tensor_member_host_case("cpu_native")


@test_utils.test(arch=[ti.cpu])
def test_sort_struct_tensor_member_values_reuses_order_apply_workspaces():
    workspace = ti.algorithms.SortWorkspace(max_items=8)
    _run_sort_struct_tensor_member_host_case("cpu_native", workspace=workspace)
    first_group_count = len(workspace._order_apply_inplace_plan_groups)
    _run_sort_struct_tensor_member_host_case("cpu_native", workspace=workspace)

    copy_workspace = workspace._order_apply_indexed_copy_workspace
    transform_workspace = workspace._order_apply_transform_workspace
    assert copy_workspace is not None
    assert transform_workspace is not None
    assert copy_workspace._native_indexed_copy_plan is not None
    assert transform_workspace._native_transform_plan is not None
    assert workspace._order_apply_inplace_plan_group is not None
    assert len(workspace._order_apply_inplace_plan_groups) >= first_group_count
    assert len(workspace._order_apply_inplace_plan_groups) >= 2
    assert len(copy_workspace._native_indexed_copy_plans) >= 2
    assert len(transform_workspace._native_transform_plans) >= 2


@test_utils.test(arch=[ti.cuda])
def test_sort_struct_tensor_member_values_cuda_native():
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "cuda_cub_radix_sort_available")
        and prog.cuda_cub_radix_sort_available()
    ):
        pytest.skip("CUDA CUB radix sort is unavailable in this runtime.")
    if not (
        hasattr(prog, "cuda_toolkit_transform_available")
        and prog.cuda_toolkit_transform_available()
    ):
        pytest.skip("CUDA toolkit strided member copy path is unavailable.")

    _run_sort_struct_tensor_member_host_case("cuda_cub_native")


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_sort_struct_tensor_member_values_vulkan_native():
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "vulkan_radix_sort_available")
        and prog.vulkan_radix_sort_available()
    ):
        pytest.skip("Vulkan native radix sort is unavailable in this runtime.")

    _run_sort_struct_tensor_member_host_case("vulkan_native_radix_u32")


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
def test_sort_by_key_multi_part_cpu_native_ndarray():
    _run_sort_by_key_native_ndarray_case("cpu_native")


@test_utils.test(arch=[ti.cuda])
def test_sort_by_key_multi_part_cuda_native_ndarray():
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "cuda_cub_radix_sort_available")
        and prog.cuda_cub_radix_sort_available()
    ):
        pytest.skip("CUDA CUB radix sort is not available.")
    _run_sort_by_key_native_ndarray_case("cuda_cub_native")


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_sort_by_key_multi_part_vulkan_native_ndarray():
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "vulkan_radix_sort_available")
        and prog.vulkan_radix_sort_available()
    ):
        pytest.skip("Vulkan native radix sort is not available.")
    _run_sort_by_key_native_ndarray_case("vulkan_native_radix_u32")


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


@pytest.mark.parametrize("value_dtype,np_dtype", _SORT_VALUE_DTYPE_CASES)
@test_utils.test(arch=[ti.cpu])
def test_sort_cpu_native_payload_dtypes(value_dtype, np_dtype):
    _run_sort_payload_dtype_case("cpu_native", value_dtype, np_dtype)


@test_utils.test(arch=[ti.cpu])
def test_sort_cpu_native_vector_payload():
    _run_sort_vector_payload_case("cpu_native")


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


@pytest.mark.parametrize("value_dtype,np_dtype", _SORT_VALUE_DTYPE_CASES)
@test_utils.test(arch=[ti.cuda])
def test_sort_cuda_cub_payload_dtypes(value_dtype, np_dtype):
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    if not impl.get_runtime().prog.cuda_cub_radix_sort_available():
        pytest.skip("CUDA CUB radix sort is not available.")

    _run_sort_payload_dtype_case("cuda_cub_native", value_dtype, np_dtype)


@test_utils.test(arch=[ti.cuda])
def test_sort_cuda_cub_vector_payload():
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    if not impl.get_runtime().prog.cuda_cub_radix_sort_available():
        pytest.skip("CUDA CUB radix sort is not available.")

    _run_sort_vector_payload_case("cuda_cub_native")


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


@pytest.mark.parametrize("key_dtype,np_dtype", _SORT_KEY_DTYPE_CASES)
@test_utils.test(arch=[ti.vulkan])
def test_sort_vulkan_native_key_dtypes(key_dtype, np_dtype):
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    if not impl.get_runtime().prog.vulkan_radix_sort_available():
        pytest.skip("Vulkan native radix sort is not available.")

    keys_np = _sort_key_values(np_dtype)
    values_np = np.arange(keys_np.shape[0], dtype=np.int64) * 10 - 3
    order = _expected_sort_order(keys_np)
    keys = ti.ndarray(key_dtype, shape=keys_np.shape[0])
    values = ti.ndarray(ti.i64, shape=keys_np.shape[0])
    keys.from_numpy(keys_np)
    values.from_numpy(values_np)

    ti.algorithms.sort(keys, values, method="vulkan_native_radix_u32")

    actual_keys = keys.to_numpy()
    if np.issubdtype(np_dtype, np.floating):
        bit_dtype = np.uint32 if np_dtype == np.float32 else np.uint64
        assert np.array_equal(
            actual_keys.view(bit_dtype),
            keys_np[order].view(bit_dtype),
        )
    else:
        assert np.array_equal(actual_keys, keys_np[order])
    assert np.array_equal(values.to_numpy(), values_np[order])


@test_utils.test(arch=[ti.vulkan])
def test_sort_vulkan_native_float_rejects_bitwise_nan_policy():
    keys = ti.ndarray(ti.f32, shape=4)
    keys.from_numpy(np.array([1.0, np.nan, -2.0, 0.0], dtype=np.float32))

    with pytest.raises(NotImplementedError):
        ti.algorithms.sort(
            keys, method="vulkan_native_radix_u32", nan_policy="bitwise"
        )


@pytest.mark.parametrize("value_dtype,np_dtype", _SORT_VALUE_DTYPE_CASES)
@test_utils.test(arch=[ti.vulkan])
def test_sort_vulkan_native_payload_dtypes(value_dtype, np_dtype):
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    if not impl.get_runtime().prog.vulkan_radix_sort_available():
        pytest.skip("Vulkan native radix sort is not available.")

    _run_sort_payload_dtype_case("vulkan_native_radix_u32", value_dtype, np_dtype)


@test_utils.test(arch=[ti.vulkan])
def test_sort_vulkan_native_vector_payload():
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    if not impl.get_runtime().prog.vulkan_radix_sort_available():
        pytest.skip("Vulkan native radix sort is not available.")

    _run_sort_vector_payload_case("vulkan_native_radix_u32")


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
