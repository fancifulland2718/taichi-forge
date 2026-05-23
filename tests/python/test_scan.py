import gc

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils

_SCAN_DTYPES = [
    (ti.i32, np.int32),
    (ti.u32, np.uint32),
    (ti.f32, np.float32),
    (ti.u64, np.uint64),
    (ti.i64, np.int64),
    (ti.f64, np.float64),
]
_VULKAN_SCAN_DTYPES = _SCAN_DTYPES
_SCAN_VALUE_TYPE = {
    ti.i32: 0,
    ti.f32: 1,
    ti.u32: 2,
    ti.u64: 3,
    ti.i64: 4,
    ti.f64: 5,
}


@test_utils.test(arch=[ti.cpu])
def test_dense_field_view_probe_accepts_only_root_dense_place():
    from taichi_forge.algorithms import _algorithms  # pylint: disable=import-outside-toplevel

    dense = ti.field(ti.i32, shape=8)
    sparse = ti.field(ti.i32)
    ti.root.pointer(ti.i, 4).dense(ti.i, 2).place(sparse)

    dense_view = _algorithms._primitive_view(dense)
    assert dense_view is not None
    assert dense_view.is_dense_field
    assert dense_view.shape == (8,)

    assert _algorithms._primitive_view(sparse) is None


def _scan_values(n, np_dtype):
    if np.issubdtype(np_dtype, np.unsignedinteger):
        return (np.arange(n, dtype=np.uint64) % 7).astype(np_dtype)
    if np.issubdtype(np_dtype, np.floating):
        return ((np.arange(n, dtype=np.float64) % 7) - 3).astype(np_dtype)
    return (np.arange(n, dtype=np.int64) % 7 - 3).astype(np_dtype)


def _assert_scan_equal(actual, expected):
    if np.issubdtype(expected.dtype, np.floating):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    else:
        np.testing.assert_array_equal(actual, expected)


def _run_struct_member_scan_case(n, dtype, np_dtype):
    payload = ti.types.struct(value=dtype, tag=ti.i32)
    arr = ti.ndarray(payload, shape=n)
    data = _scan_values(n, np_dtype)
    host = np.zeros((n,), dtype=arr.numpy_dtype)
    host["value"] = data
    host["tag"] = np.arange(n, dtype=np.int32) * 5 + 7
    arr.from_numpy(host)

    ti.algorithms.PrefixSumExecutor(n).run(arr.field("value"))

    result = arr.to_numpy()
    expected = np.cumsum(data, dtype=np_dtype).astype(np_dtype)
    _assert_scan_equal(result["value"], expected)
    np.testing.assert_array_equal(result["tag"], host["tag"])


def _run_struct_tensor_member_scan_case(n):
    payload = ti.types.struct(
        vec=ti.types.vector(2, ti.i32),
        mat=ti.types.matrix(2, 2, ti.i32),
        tag=ti.i32,
    )
    arr = ti.ndarray(payload, shape=n)
    host = np.zeros((n,), dtype=arr.numpy_dtype)
    host["vec"] = (np.arange(n * 2, dtype=np.int32).reshape(n, 2) % 7) - 3
    host["mat"] = (np.arange(n * 4, dtype=np.int32).reshape(n, 2, 2) % 5) - 2
    host["tag"] = np.arange(n, dtype=np.int32) * 5 + 7
    arr.from_numpy(host)

    vec = arr.field("vec")
    mat = arr.field("mat")
    vec_executor = ti.algorithms.PrefixSumExecutor(n)
    mat_executor = ti.algorithms.PrefixSumExecutor(n)
    vec_executor.run(vec)
    assert len(vec_executor._native_scan_plan_groups) == 1
    vec_executor.run(vec)
    assert len(vec_executor._native_scan_plan_groups) == 1
    mat_executor.run(mat)
    assert len(mat_executor._native_scan_plan_groups) == 1

    result = arr.to_numpy()
    expected_vec_once = np.cumsum(host["vec"], axis=0, dtype=np.int64).astype(np.int32)
    np.testing.assert_array_equal(
        result["vec"],
        np.cumsum(expected_vec_once, axis=0, dtype=np.int64).astype(np.int32),
    )
    np.testing.assert_array_equal(
        result["mat"], np.cumsum(host["mat"], axis=0, dtype=np.int64).astype(np.int32)
    )
    np.testing.assert_array_equal(result["tag"], host["tag"])


def _native_scan_method_for_current_arch():
    arch = impl.current_cfg().arch
    prog = impl.get_runtime().prog
    if arch == ti.cpu:
        if not (hasattr(prog, "cpu_scan_available") and prog.cpu_scan_available()):
            pytest.skip("CPU native scan is unavailable.")
        return "cpu_native", "cpu_inclusive_scan_dense_field"
    if arch == ti.cuda:
        if not (
            hasattr(prog, "cuda_cub_scan_available")
            and prog.cuda_cub_scan_available()
        ):
            pytest.skip("CUDA CUB scan is unavailable.")
        return "cuda_cub", "cuda_cub_inclusive_scan_dense_field"
    if arch == ti.vulkan:
        if not (
            hasattr(prog, "vulkan_scan_available")
            and prog.vulkan_scan_available()
        ):
            pytest.skip("Vulkan native scan is unavailable.")
        return "vulkan_native", "vulkan_inclusive_scan_dense_field"
    pytest.skip("native scan is unavailable on this arch.")


def _run_dense_matrix_field_scan_case():
    n = 128
    _backend, expected_method = _native_scan_method_for_current_arch()
    arr = ti.Vector.field(2, ti.i32, shape=n)
    values = (np.arange(n * 2, dtype=np.int32).reshape(n, 2) % 7) - 3
    arr.from_numpy(values)
    executor = ti.algorithms.PrefixSumExecutor(n)

    executor.run(arr)

    expected = np.cumsum(values, axis=0, dtype=np.int32)
    np.testing.assert_array_equal(arr.to_numpy(), expected)
    assert len(executor._native_scan_plans) == 2
    assert executor._native_scan_plan["method_name"] == expected_method
    assert len(executor._native_scan_plan_groups) == 1

    arr.from_numpy(values)
    executor.run(arr)
    np.testing.assert_array_equal(arr.to_numpy(), expected)
    assert len(executor._native_scan_plan_groups) == 1


@test_utils.test(arch=[ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_scan():
    def test_scan_for_dtype(dtype, N):
        arr = ti.field(dtype, N)
        arr_aux = ti.field(dtype, N)

        @ti.kernel
        def fill():
            for i in arr:
                arr[i] = ti.random() * N
                arr_aux[i] = arr[i]

        fill()

        # Performing an inclusive in-place's parallel prefix sum,
        # only one exectutor is needed for a specified sorting length.
        executor = ti.algorithms.PrefixSumExecutor(N)

        executor.run(arr)

        cur_sum = 0
        for i in range(N):
            cur_sum += arr_aux[i]
            assert arr[i] == cur_sum

    test_scan_for_dtype(ti.i32, 512)
    test_scan_for_dtype(ti.i32, 1024)
    test_scan_for_dtype(ti.i32, 4096)


@pytest.mark.parametrize("dtype", [ti.i32])
@pytest.mark.parametrize("N", [512, 1024, 4096])
@pytest.mark.parametrize("offset", [0, -1, 1, 256, -256, -23333, 23333])
@test_utils.test(arch=[ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_scan_with_offset(dtype, N, offset):
    arr = ti.field(dtype, N, offset=offset)
    arr_aux = ti.field(dtype, N, offset=offset)

    @ti.kernel
    def fill():
        for i in arr:
            arr[i] = ti.random() * N
            arr_aux[i] = arr[i]

    fill()

    # Performing an inclusive in-place's parallel prefix sum,
    # only one exectutor is needed for a specified sorting length.
    executor = ti.algorithms.PrefixSumExecutor(N)

    executor.run(arr)

    cur_sum = 0
    for i in range(N):
        cur_sum += arr_aux[i + offset]
        assert arr[i + offset] == cur_sum


@test_utils.test(arch=[ti.cuda])
def test_scan_ndarray_cuda_cub():
    N = 4096

    if not impl.get_runtime().prog.cuda_cub_scan_available():
        pytest.skip("CUDA CUB scan is unavailable in this build/runtime.")

    for dtype, np_dtype in _SCAN_DTYPES:
        arr = ti.ndarray(dtype, shape=N)
        data = _scan_values(N, np_dtype)
        arr.from_numpy(data)
        ti.algorithms.PrefixSumExecutor(N).run(arr)
        expected = np.cumsum(data, dtype=np_dtype).astype(np_dtype)
        _assert_scan_equal(arr.to_numpy(), expected)
    assert impl.get_runtime().prog.cuda_cub_scan_workspace_bytes() > 0


@test_utils.test(arch=[ti.cuda])
def test_scan_cuda_cub_struct_member_view():
    N = 4096

    if not impl.get_runtime().prog.cuda_cub_scan_available():
        pytest.skip("CUDA CUB scan is unavailable in this build/runtime.")

    for dtype, np_dtype in _SCAN_DTYPES:
        _run_struct_member_scan_case(N, dtype, np_dtype)
    assert impl.get_runtime().prog.cuda_cub_scan_workspace_bytes() > 0


@test_utils.test(arch=[ti.cuda])
def test_scan_cuda_cub_struct_tensor_member_view():
    N = 4096

    if not impl.get_runtime().prog.cuda_cub_scan_available():
        pytest.skip("CUDA CUB scan is unavailable in this build/runtime.")

    _run_struct_tensor_member_scan_case(N)
    assert impl.get_runtime().prog.cuda_cub_scan_workspace_bytes() > 0


@test_utils.test(arch=[ti.cpu])
def test_scan_ndarray_cpu_native():
    N = 4096

    if not impl.get_runtime().prog.cpu_scan_available():
        pytest.skip("CPU native scan is unavailable in this build/runtime.")

    for dtype, np_dtype in _SCAN_DTYPES:
        arr = ti.ndarray(dtype, shape=N)
        data = _scan_values(N, np_dtype)
        arr.from_numpy(data)
        ti.algorithms.PrefixSumExecutor(N).run(arr)
        expected = np.cumsum(data, dtype=np_dtype).astype(np_dtype)
        _assert_scan_equal(arr.to_numpy(), expected)
    assert impl.get_runtime().prog.cpu_scan_workspace_bytes() == 0


@test_utils.test(arch=[ti.cpu])
def test_scan_ndarray_cpu_native_executor_replay():
    N = 128

    if not impl.get_runtime().prog.cpu_scan_available():
        pytest.skip("CPU native scan is unavailable in this build/runtime.")

    arr = ti.ndarray(ti.i32, shape=N)
    executor = ti.algorithms.PrefixSumExecutor(N)
    first_plan = None
    for base in (0, 17):
        data = (_scan_values(N, np.int32) + base).astype(np.int32)
        arr.from_numpy(data)
        executor.run(arr)
        expected = np.cumsum(data, dtype=np.int32).astype(np.int32)
        _assert_scan_equal(arr.to_numpy(), expected)
        if first_plan is None:
            first_plan = executor._native_scan_plan
        else:
            assert executor._native_scan_plan is first_plan
    assert executor._native_scan_plan["backend"] == "cpu_native"
    assert executor._native_scan_plan["method_name"] == "cpu_inclusive_scan_ndarray"


@test_utils.test(arch=[ti.cpu])
def test_scan_dense_field_cpu_native():
    N = 4096

    if not impl.get_runtime().prog.cpu_scan_available():
        pytest.skip("CPU native scan is unavailable in this build/runtime.")

    for dtype, np_dtype in _SCAN_DTYPES:
        arr = ti.field(dtype, shape=N)
        data = _scan_values(N, np_dtype)
        arr.from_numpy(data)
        ti.algorithms.PrefixSumExecutor(N).run(arr)
        expected = np.cumsum(data, dtype=np_dtype).astype(np_dtype)
        _assert_scan_equal(arr.to_numpy(), expected)
    assert impl.get_runtime().prog.cpu_scan_workspace_bytes() == 0


@test_utils.test(arch=[ti.cpu])
def test_scan_dense_field_cpu_native_executor_replay():
    N = 128

    if not impl.get_runtime().prog.cpu_scan_available():
        pytest.skip("CPU native scan is unavailable in this build/runtime.")

    arr = ti.field(ti.i32, shape=N)
    executor = ti.algorithms.PrefixSumExecutor(N)
    for base in (0, 17):
        data = (_scan_values(N, np.int32) + base).astype(np.int32)
        arr.from_numpy(data)
        executor.run(arr)
        expected = np.cumsum(data, dtype=np.int32).astype(np.int32)
        _assert_scan_equal(arr.to_numpy(), expected)
    assert executor._native_scan_plan is not None
    assert executor._native_scan_plan["backend"] == "cpu_native"


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_scan_native_dense_matrix_field_components():
    _run_dense_matrix_field_scan_case()


@test_utils.test(arch=[ti.cuda])
def test_scan_dense_field_cuda_cub():
    N = 4096

    if not impl.get_runtime().prog.cuda_cub_scan_available():
        pytest.skip("CUDA CUB scan is unavailable in this build/runtime.")

    for dtype, np_dtype in _SCAN_DTYPES:
        arr = ti.field(dtype, shape=N)
        data = _scan_values(N, np_dtype)
        arr.from_numpy(data)
        ti.algorithms.PrefixSumExecutor(N).run(arr)
        expected = np.cumsum(data, dtype=np_dtype).astype(np_dtype)
        _assert_scan_equal(arr.to_numpy(), expected)
    assert impl.get_runtime().prog.cuda_cub_scan_workspace_bytes() > 0


@test_utils.test(arch=[ti.cuda])
def test_scan_dense_field_cuda_cub_executor_replay():
    N = 128

    if not impl.get_runtime().prog.cuda_cub_scan_available():
        pytest.skip("CUDA CUB scan is unavailable in this build/runtime.")

    arr = ti.field(ti.i32, shape=N)
    executor = ti.algorithms.PrefixSumExecutor(N)
    for base in (0, 17):
        data = (_scan_values(N, np.int32) + base).astype(np.int32)
        arr.from_numpy(data)
        executor.run(arr)
        expected = np.cumsum(data, dtype=np.int32).astype(np.int32)
        _assert_scan_equal(arr.to_numpy(), expected)
    assert executor._native_scan_plan is not None
    assert executor._native_scan_plan["backend"] == "cuda_cub"


@test_utils.test(arch=[ti.cpu])
def test_scan_cpu_native_struct_member_view():
    N = 4096

    if not impl.get_runtime().prog.cpu_scan_available():
        pytest.skip("CPU native scan is unavailable in this build/runtime.")

    for dtype, np_dtype in _SCAN_DTYPES:
        _run_struct_member_scan_case(N, dtype, np_dtype)
    assert impl.get_runtime().prog.cpu_scan_workspace_bytes() == 0


@test_utils.test(arch=[ti.cpu])
def test_scan_cpu_native_struct_member_executor_replay():
    N = 128

    if not impl.get_runtime().prog.cpu_scan_available():
        pytest.skip("CPU native scan is unavailable in this build/runtime.")

    payload = ti.types.struct(value=ti.i32, tag=ti.i32)
    arr = ti.ndarray(payload, shape=N)
    member = arr.field("value")
    executor = ti.algorithms.PrefixSumExecutor(N)
    first_plan = None
    for base in (0, 17):
        data = (_scan_values(N, np.int32) + base).astype(np.int32)
        host = np.zeros((N,), dtype=arr.numpy_dtype)
        host["value"] = data
        host["tag"] = np.arange(N, dtype=np.int32) * 5 + 7
        arr.from_numpy(host)
        executor.run(member)
        result = arr.to_numpy()
        expected = np.cumsum(data, dtype=np.int32).astype(np.int32)
        _assert_scan_equal(result["value"], expected)
        np.testing.assert_array_equal(result["tag"], host["tag"])
        if first_plan is None:
            first_plan = executor._native_scan_plan
        else:
            assert executor._native_scan_plan is first_plan
    assert executor._native_scan_plan["backend"] == "cpu_native"
    assert (
        executor._native_scan_plan["method_name"]
        == "cpu_inclusive_scan_member_ndarray"
    )


@test_utils.test(arch=[ti.cpu])
def test_scan_cpu_native_struct_tensor_member_view():
    N = 4096

    if not impl.get_runtime().prog.cpu_scan_available():
        pytest.skip("CPU native scan is unavailable in this build/runtime.")

    _run_struct_tensor_member_scan_case(N)
    assert impl.get_runtime().prog.cpu_scan_workspace_bytes() == 0


@test_utils.test(arch=[ti.vulkan])
def test_scan_ndarray_vulkan_native():
    N = 8192

    if not impl.get_runtime().prog.vulkan_scan_available():
        pytest.skip("Vulkan native scan is unavailable in this build/runtime.")

    prog = impl.get_runtime().prog
    for dtype, np_dtype in _VULKAN_SCAN_DTYPES:
        value_type = _SCAN_VALUE_TYPE[dtype]
        if hasattr(prog, "vulkan_scan_value_type_available") and not (
            prog.vulkan_scan_value_type_available(value_type)
        ):
            continue
        arr = ti.ndarray(dtype, shape=N)
        data = _scan_values(N, np_dtype)
        arr.from_numpy(data)
        ti.algorithms.PrefixSumExecutor(N).run(arr)
        expected = np.cumsum(data, dtype=np_dtype).astype(np_dtype)
        _assert_scan_equal(arr.to_numpy(), expected)
    assert impl.get_runtime().prog.vulkan_scan_workspace_bytes() > 0


@test_utils.test(arch=[ti.vulkan])
def test_scan_dense_field_vulkan_native():
    N = 8192

    if not impl.get_runtime().prog.vulkan_scan_available():
        pytest.skip("Vulkan native scan is unavailable in this build/runtime.")

    prog = impl.get_runtime().prog
    tested = 0
    for dtype, np_dtype in _VULKAN_SCAN_DTYPES:
        value_type = _SCAN_VALUE_TYPE[dtype]
        if hasattr(prog, "vulkan_scan_value_type_available") and not (
            prog.vulkan_scan_value_type_available(value_type)
        ):
            continue
        arr = ti.field(dtype, shape=N)
        data = _scan_values(N, np_dtype)
        arr.from_numpy(data)
        ti.algorithms.PrefixSumExecutor(N).run(arr)
        expected = np.cumsum(data, dtype=np_dtype).astype(np_dtype)
        _assert_scan_equal(arr.to_numpy(), expected)
        tested += 1
    assert tested >= 3
    assert impl.get_runtime().prog.vulkan_scan_workspace_bytes() > 0


@test_utils.test(arch=[ti.vulkan])
def test_scan_vulkan_native_struct_member_view():
    N = 8192

    if not impl.get_runtime().prog.vulkan_scan_available():
        pytest.skip("Vulkan native scan is unavailable in this build/runtime.")

    prog = impl.get_runtime().prog
    tested = 0
    for dtype, np_dtype in _VULKAN_SCAN_DTYPES:
        value_type = _SCAN_VALUE_TYPE[dtype]
        if hasattr(prog, "vulkan_scan_value_type_available") and not (
            prog.vulkan_scan_value_type_available(value_type)
        ):
            continue
        _run_struct_member_scan_case(N, dtype, np_dtype)
        tested += 1
    assert tested >= 3
    assert impl.get_runtime().prog.vulkan_scan_workspace_bytes() > 0


@test_utils.test(arch=[ti.vulkan])
def test_scan_vulkan_native_struct_tensor_member_view():
    N = 8192

    if not impl.get_runtime().prog.vulkan_scan_available():
        pytest.skip("Vulkan native scan is unavailable in this runtime.")

    _run_struct_tensor_member_scan_case(N)
    assert impl.get_runtime().prog.vulkan_scan_workspace_bytes() > 0


@test_utils.test(arch=[ti.vulkan])
def test_scan_ndarray_vulkan_native_respects_f64_capability_gate():
    N = 128
    arr = ti.ndarray(ti.f64, shape=N)
    arr.from_numpy(_scan_values(N, np.float64))

    if not impl.get_runtime().prog.vulkan_scan_available():
        pytest.skip("Vulkan native scan is unavailable in this build/runtime.")
    if impl.get_runtime().prog.vulkan_scan_value_type_available(5):
        ti.algorithms.PrefixSumExecutor(N).run(arr)
        expected = np.cumsum(_scan_values(N, np.float64), dtype=np.float64)
        _assert_scan_equal(arr.to_numpy(), expected)
        return
    if hasattr(impl.get_runtime().prog, "vulkan_scan_value_type_available"):
        assert not impl.get_runtime().prog.vulkan_scan_value_type_available(5)

    with pytest.raises(RuntimeError, match="native CPU/CUDA/Vulkan scan fast paths"):
        ti.algorithms.PrefixSumExecutor(N).run(arr)


@test_utils.test(arch=[ti.cpu])
def test_scan_struct_member_view_rejections():
    payload = ti.types.struct(value=ti.f32, tag=ti.i32)
    arr = ti.ndarray(payload, shape=8)

    with pytest.raises(TypeError, match="does not support StructNdarray"):
        ti.algorithms.PrefixSumExecutor(8).run(arr)


@pytest.mark.run_in_serial
@test_utils.test(arch=[ti.vulkan])
def test_scan_ndarray_vulkan_reset_with_live_ndarray():
    N = 4096
    arr = ti.ndarray(ti.i32, shape=N)

    if not impl.get_runtime().prog.vulkan_scan_available():
        pytest.skip("Vulkan native scan is unavailable in this build/runtime.")

    @ti.kernel
    def fill(data: ti.types.ndarray(ti.i32, ndim=1)):
        for i in range(N):
            data[i] = i % 9 - 4

    fill(arr)
    ti.algorithms.PrefixSumExecutor(N).run(arr)
    assert arr.to_numpy()[N - 1] == sum(i % 9 - 4 for i in range(N))

    ti.reset()
    del arr
    gc.collect()
