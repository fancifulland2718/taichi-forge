import gc

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_histogram_field_atomic():
    n = 4096
    num_bins = 37
    values = ti.field(ti.i32, shape=n)
    bins = ti.field(ti.i32, shape=num_bins)

    @ti.kernel
    def fill(mode: ti.i32):
        for i in range(n):
            if mode == 2:
                values[i] = (i % (num_bins + 4)) - 2
            elif mode == 1:
                values[i] = 3
            else:
                values[i] = (i * 11 + 5) % num_bins
        for i in range(num_bins):
            bins[i] = -1

    for method in ("field_atomic", "field_direct", "field_private"):
        workspace = ti.algorithms.HistogramWorkspace(max_items=n, max_bins=num_bins)
        for mode in range(3):
            fill(mode)
            ti.algorithms.experimental_histogram(
                values, bins, method=method, workspace=workspace
            )
            if mode == 2:
                raw = (np.arange(n) % (num_bins + 4)) - 2
                expected_values = raw[(0 <= raw) & (raw < num_bins)].astype(np.int32)
            elif mode == 1:
                expected_values = np.full(n, 3, dtype=np.int32)
            else:
                expected_values = ((np.arange(n) * 11 + 5) % num_bins).astype(np.int32)
            expected = np.bincount(expected_values, minlength=num_bins).astype(np.int32)
            assert np.array_equal(bins.to_numpy(), expected)
        if method == "field_private":
            assert workspace.workspace_bytes_peak > 0
        else:
            assert workspace.workspace_bytes_peak == 0


@test_utils.test(arch=[ti.cuda])
def test_experimental_histogram_cuda_cub_ndarray():
    n = 8192
    num_bins = 64
    values = ti.ndarray(ti.i32, shape=n)
    bins = ti.ndarray(ti.i32, shape=num_bins)

    if not impl.get_runtime().prog.cuda_cub_histogram_available():
        pytest.skip("CUDA CUB histogram is unavailable in this build/runtime.")

    @ti.kernel
    def fill(
        values_arr: ti.types.ndarray(ti.i32, ndim=1),
        bins_arr: ti.types.ndarray(ti.i32, ndim=1),
        mode: ti.i32,
    ):
        for i in range(n):
            if mode == 1:
                values_arr[i] = 7
            else:
                values_arr[i] = (i * 13 + 9) % num_bins
        for i in range(num_bins):
            bins_arr[i] = -1

    workspace = ti.algorithms.HistogramWorkspace(max_items=n, max_bins=num_bins)
    for mode in range(2):
        fill(values, bins, mode)
        ti.algorithms.experimental_histogram(
            values, bins, method="auto", workspace=workspace
        )
        if mode == 1:
            expected_values = np.full(n, 7, dtype=np.int32)
        else:
            expected_values = ((np.arange(n) * 13 + 9) % num_bins).astype(np.int32)
        expected = np.bincount(expected_values, minlength=num_bins).astype(np.int32)
        assert np.array_equal(bins.to_numpy(), expected)
    assert workspace.workspace_bytes_peak > 0


@test_utils.test(arch=[ti.cpu])
def test_experimental_histogram_cpu_native_ndarray():
    n = 8192
    num_bins = 64
    values = ti.ndarray(ti.i32, shape=n)
    bins = ti.ndarray(ti.i32, shape=num_bins)

    if not impl.get_runtime().prog.cpu_histogram_available():
        pytest.skip("CPU native histogram is unavailable in this build/runtime.")

    @ti.kernel
    def fill(
        values_arr: ti.types.ndarray(ti.i32, ndim=1),
        bins_arr: ti.types.ndarray(ti.i32, ndim=1),
        mode: ti.i32,
    ):
        for i in range(n):
            if mode == 2:
                values_arr[i] = (i % (num_bins + 4)) - 2
            elif mode == 1:
                values_arr[i] = 7
            else:
                values_arr[i] = (i * 13 + 9) % num_bins
        for i in range(num_bins):
            bins_arr[i] = -1

    workspace = ti.algorithms.HistogramWorkspace(max_items=n, max_bins=num_bins)
    for mode in range(3):
        fill(values, bins, mode)
        ti.algorithms.experimental_histogram(
            values, bins, method="auto", workspace=workspace
        )
        if mode == 2:
            raw = (np.arange(n) % (num_bins + 4)) - 2
            expected_values = raw[(0 <= raw) & (raw < num_bins)].astype(np.int32)
        elif mode == 1:
            expected_values = np.full(n, 7, dtype=np.int32)
        else:
            expected_values = ((np.arange(n) * 13 + 9) % num_bins).astype(np.int32)
        expected = np.bincount(expected_values, minlength=num_bins).astype(np.int32)
        assert np.array_equal(bins.to_numpy(), expected)
    assert workspace.workspace_bytes_peak == 0
    assert impl.get_runtime().prog.cpu_histogram_workspace_bytes() == 0


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_histogram_vulkan_native_ndarray():
    n = 8192
    num_bins = 64
    values = ti.ndarray(ti.i32, shape=n)
    bins = ti.ndarray(ti.i32, shape=num_bins)

    if not impl.get_runtime().prog.vulkan_histogram_available():
        pytest.skip("Vulkan native histogram is unavailable in this build/runtime.")

    @ti.kernel
    def fill(
        values_arr: ti.types.ndarray(ti.i32, ndim=1),
        bins_arr: ti.types.ndarray(ti.i32, ndim=1),
        mode: ti.i32,
    ):
        for i in range(n):
            if mode == 2:
                values_arr[i] = (i % (num_bins + 4)) - 2
            elif mode == 1:
                values_arr[i] = 7
            else:
                values_arr[i] = (i * 13 + 9) % num_bins
        for i in range(num_bins):
            bins_arr[i] = -1

    workspace = ti.algorithms.HistogramWorkspace(max_items=n, max_bins=num_bins)
    for mode in range(3):
        fill(values, bins, mode)
        ti.algorithms.experimental_histogram(
            values, bins, method="auto", workspace=workspace
        )
        if mode == 2:
            raw = (np.arange(n) % (num_bins + 4)) - 2
            expected_values = raw[(0 <= raw) & (raw < num_bins)].astype(np.int32)
        elif mode == 1:
            expected_values = np.full(n, 7, dtype=np.int32)
        else:
            expected_values = ((np.arange(n) * 13 + 9) % num_bins).astype(np.int32)
        expected = np.bincount(expected_values, minlength=num_bins).astype(np.int32)
        assert np.array_equal(bins.to_numpy(), expected)
    assert impl.get_runtime().prog.vulkan_histogram_workspace_bytes() == 0


@pytest.mark.run_in_serial
@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_histogram_vulkan_native_reset_with_live_ndarray():
    n = 1024
    num_bins = 64
    values = ti.ndarray(ti.i32, shape=n)
    bins = ti.ndarray(ti.i32, shape=num_bins)

    if not impl.get_runtime().prog.vulkan_histogram_available():
        pytest.skip("Vulkan native histogram is unavailable in this build/runtime.")

    @ti.kernel
    def fill(
        values_arr: ti.types.ndarray(ti.i32, ndim=1),
        bins_arr: ti.types.ndarray(ti.i32, ndim=1),
    ):
        for i in range(n):
            values_arr[i] = (i * 7 + 1) % num_bins
        for i in range(num_bins):
            bins_arr[i] = -1

    fill(values, bins)
    ti.algorithms.experimental_histogram(values, bins, method="vulkan_native")
    expected_values = ((np.arange(n) * 7 + 1) % num_bins).astype(np.int32)
    expected = np.bincount(expected_values, minlength=num_bins).astype(np.int32)
    assert np.array_equal(bins.to_numpy(), expected)
    ti.reset()
    del values, bins
    gc.collect()
