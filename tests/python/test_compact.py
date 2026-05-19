import gc

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils


@test_utils.test(arch=[ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_compact_field_scan():
    n = 2048
    values = ti.field(ti.i32, shape=n)
    flags = ti.field(ti.i32, shape=n)
    output = ti.field(ti.i32, shape=n)
    count = ti.field(ti.i32, shape=())

    @ti.kernel
    def fill():
        for i in range(n):
            values[i] = i * 3 - 17
            flags[i] = 1 if i % 5 == 0 or i % 7 == 0 else 0
            output[i] = -1
        count[None] = -1

    fill()
    workspace = ti.algorithms.CompactWorkspace(max_items=n)
    ti.algorithms.experimental_compact(
        values, flags, output, count, method="field_scan", workspace=workspace
    )

    values_np = np.arange(n, dtype=np.int32) * 3 - 17
    flags_np = ((np.arange(n) % 5 == 0) | (np.arange(n) % 7 == 0))
    expected = values_np[flags_np]
    assert count[None] == expected.shape[0]
    assert np.array_equal(output.to_numpy()[: expected.shape[0]], expected)
    assert workspace.workspace_bytes_peak >= n * 4


@test_utils.test(arch=[ti.cuda])
def test_experimental_compact_cuda_cub_ndarray():
    n = 4096
    values = ti.ndarray(ti.i32, shape=n)
    flags = ti.ndarray(ti.i32, shape=n)
    output = ti.ndarray(ti.i32, shape=n)
    count = ti.ndarray(ti.i32, shape=1)

    if not impl.get_runtime().prog.cuda_cub_select_available():
        pytest.skip("CUDA CUB select is unavailable in this build/runtime.")

    @ti.kernel
    def fill(
        values_arr: ti.types.ndarray(ti.i32, ndim=1),
        flags_arr: ti.types.ndarray(ti.i32, ndim=1),
        output_arr: ti.types.ndarray(ti.i32, ndim=1),
        count_arr: ti.types.ndarray(ti.i32, ndim=1),
    ):
        for i in range(n):
            values_arr[i] = i * 2 + 11
            flags_arr[i] = 1 if i % 3 == 0 else 0
            output_arr[i] = -1
        count_arr[0] = -1

    fill(values, flags, output, count)
    workspace = ti.algorithms.CompactWorkspace(max_items=n)
    ti.algorithms.experimental_compact(
        values, flags, output, count, method="auto", workspace=workspace
    )

    values_np = np.arange(n, dtype=np.int32) * 2 + 11
    expected = values_np[np.arange(n) % 3 == 0]
    assert count.to_numpy()[0] == expected.shape[0]
    assert np.array_equal(output.to_numpy()[: expected.shape[0]], expected)
    assert workspace.workspace_bytes_peak > 0


@test_utils.test(arch=[ti.cpu])
def test_experimental_compact_cpu_native_ndarray():
    n = 4096
    values = ti.ndarray(ti.i32, shape=n)
    flags = ti.ndarray(ti.i32, shape=n)
    output = ti.ndarray(ti.i32, shape=n)
    count = ti.ndarray(ti.i32, shape=1)

    if not impl.get_runtime().prog.cpu_compact_available():
        pytest.skip("CPU native compact is unavailable in this build/runtime.")

    @ti.kernel
    def fill(
        values_arr: ti.types.ndarray(ti.i32, ndim=1),
        flags_arr: ti.types.ndarray(ti.i32, ndim=1),
        output_arr: ti.types.ndarray(ti.i32, ndim=1),
        count_arr: ti.types.ndarray(ti.i32, ndim=1),
    ):
        for i in range(n):
            values_arr[i] = i * 7 - 13
            flags_arr[i] = 1 if i % 6 == 0 or i % 17 == 0 else 0
            output_arr[i] = -1
        count_arr[0] = -1

    fill(values, flags, output, count)
    workspace = ti.algorithms.CompactWorkspace(max_items=n)
    ti.algorithms.experimental_compact(
        values, flags, output, count, method="auto", workspace=workspace
    )

    values_np = np.arange(n, dtype=np.int32) * 7 - 13
    flags_np = ((np.arange(n) % 6 == 0) | (np.arange(n) % 17 == 0))
    expected = values_np[flags_np]
    assert count.to_numpy()[0] == expected.shape[0]
    assert np.array_equal(output.to_numpy()[: expected.shape[0]], expected)
    assert workspace.workspace_bytes_peak == 0
    assert impl.get_runtime().prog.cpu_compact_workspace_bytes() == 0


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_compact_vulkan_native_ndarray():
    n = 4096
    values = ti.ndarray(ti.i32, shape=n)
    flags = ti.ndarray(ti.i32, shape=n)
    output = ti.ndarray(ti.i32, shape=n)
    count = ti.ndarray(ti.i32, shape=1)

    if not impl.get_runtime().prog.vulkan_compact_available():
        pytest.skip("Vulkan native compact is unavailable in this build/runtime.")

    @ti.kernel
    def fill(
        values_arr: ti.types.ndarray(ti.i32, ndim=1),
        flags_arr: ti.types.ndarray(ti.i32, ndim=1),
        output_arr: ti.types.ndarray(ti.i32, ndim=1),
        count_arr: ti.types.ndarray(ti.i32, ndim=1),
        mode: ti.i32,
    ):
        for i in range(n):
            values_arr[i] = i * 5 - 23
            if mode == 1:
                flags_arr[i] = 0
            elif mode == 2:
                flags_arr[i] = 1
            else:
                flags_arr[i] = 1 if i % 4 == 0 or i % 11 == 0 else 0
            output_arr[i] = -1
        count_arr[0] = -1

    workspace = ti.algorithms.CompactWorkspace(max_items=n)
    values_np = np.arange(n, dtype=np.int32) * 5 - 23
    for mode in range(3):
        fill(values, flags, output, count, mode)
        ti.algorithms.experimental_compact(
            values, flags, output, count, method="auto", workspace=workspace
        )
        if mode == 1:
            flags_np = np.zeros(n, dtype=bool)
        elif mode == 2:
            flags_np = np.ones(n, dtype=bool)
        else:
            flags_np = ((np.arange(n) % 4 == 0) | (np.arange(n) % 11 == 0))
        expected = values_np[flags_np]
        assert count.to_numpy()[0] == expected.shape[0]
        assert np.array_equal(output.to_numpy()[: expected.shape[0]], expected)
    assert workspace.workspace_bytes_peak > 0
    assert impl.get_runtime().prog.vulkan_compact_workspace_bytes() > 0


@pytest.mark.run_in_serial
@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_compact_vulkan_native_reset_with_live_ndarray():
    n = 1024
    values = ti.ndarray(ti.i32, shape=n)
    flags = ti.ndarray(ti.i32, shape=n)
    output = ti.ndarray(ti.i32, shape=n)
    count = ti.ndarray(ti.i32, shape=1)

    if not impl.get_runtime().prog.vulkan_compact_available():
        pytest.skip("Vulkan native compact is unavailable in this build/runtime.")

    @ti.kernel
    def fill(
        values_arr: ti.types.ndarray(ti.i32, ndim=1),
        flags_arr: ti.types.ndarray(ti.i32, ndim=1),
        output_arr: ti.types.ndarray(ti.i32, ndim=1),
        count_arr: ti.types.ndarray(ti.i32, ndim=1),
    ):
        for i in range(n):
            values_arr[i] = i - 31
            flags_arr[i] = 1 if i % 9 == 0 else 0
            output_arr[i] = -1
        count_arr[0] = -1

    fill(values, flags, output, count)
    ti.algorithms.experimental_compact(values, flags, output, count, method="vulkan_native")
    expected = (np.arange(n, dtype=np.int32) - 31)[np.arange(n) % 9 == 0]
    assert count.to_numpy()[0] == expected.shape[0]
    assert np.array_equal(output.to_numpy()[: expected.shape[0]], expected)
    ti.reset()
    del values, flags, output, count
    gc.collect()
