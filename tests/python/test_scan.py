import gc

import pytest
import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils


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
    arr = ti.ndarray(ti.i32, shape=N)

    if not impl.get_runtime().prog.cuda_cub_scan_available():
        pytest.skip("CUDA CUB scan is unavailable in this build/runtime.")

    @ti.kernel
    def fill(data: ti.types.ndarray(ti.i32, ndim=1)):
        for i in range(N):
            data[i] = i % 7 - 3

    fill(arr)

    executor = ti.algorithms.PrefixSumExecutor(N)
    executor.run(arr)

    host = arr.to_numpy()
    cur_sum = 0
    for i in range(N):
        cur_sum += i % 7 - 3
        assert host[i] == cur_sum
    assert impl.get_runtime().prog.cuda_cub_scan_workspace_bytes() > 0


@test_utils.test(arch=[ti.cpu])
def test_scan_ndarray_cpu_native():
    N = 4096
    arr = ti.ndarray(ti.i32, shape=N)

    if not impl.get_runtime().prog.cpu_scan_available():
        pytest.skip("CPU native scan is unavailable in this build/runtime.")

    @ti.kernel
    def fill(data: ti.types.ndarray(ti.i32, ndim=1)):
        for i in range(N):
            data[i] = i % 7 - 3

    fill(arr)

    executor = ti.algorithms.PrefixSumExecutor(N)
    executor.run(arr)

    host = arr.to_numpy()
    cur_sum = 0
    for i in range(N):
        cur_sum += i % 7 - 3
        assert host[i] == cur_sum
    assert impl.get_runtime().prog.cpu_scan_workspace_bytes() == 0


@test_utils.test(arch=[ti.vulkan])
def test_scan_ndarray_vulkan_native():
    N = 8192
    arr = ti.ndarray(ti.i32, shape=N)

    if not impl.get_runtime().prog.vulkan_scan_available():
        pytest.skip("Vulkan native scan is unavailable in this build/runtime.")

    @ti.kernel
    def fill(data: ti.types.ndarray(ti.i32, ndim=1)):
        for i in range(N):
            data[i] = i % 9 - 4

    fill(arr)

    executor = ti.algorithms.PrefixSumExecutor(N)
    executor.run(arr)

    host = arr.to_numpy()
    cur_sum = 0
    for i in range(N):
        cur_sum += i % 9 - 4
        assert host[i] == cur_sum
    assert impl.get_runtime().prog.vulkan_scan_workspace_bytes() > 0


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
