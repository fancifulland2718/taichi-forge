import gc

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils


def _bucket_input(n, num_bins):
    keys = ((np.arange(n, dtype=np.int32) * 37 + 11) % num_bins).astype(np.int32)
    values = (np.arange(n, dtype=np.int32) * 3 - 17).astype(np.int32)
    if n >= 8:
        keys[1] = -1
        keys[5] = num_bins + 3
    counts = np.bincount(keys[(keys >= 0) & (keys < num_bins)], minlength=num_bins)
    offsets = np.zeros(num_bins + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(counts, dtype=np.int64).astype(np.int32)
    return keys, values, offsets


def _assert_bucket_matches(keys, values, offsets, output, expected_offsets):
    assert np.array_equal(offsets, expected_offsets)
    num_bins = expected_offsets.shape[0] - 1
    for bucket in range(num_bins):
        begin = expected_offsets[bucket]
        end = expected_offsets[bucket + 1]
        expected = values[keys == bucket]
        actual = output[begin:end]
        assert np.array_equal(np.sort(actual), np.sort(expected))


def _run_ndarray_bucket_builder(method):
    n = 4096
    num_bins = 257
    keys_np, values_np, expected_offsets = _bucket_input(n, num_bins)
    keys = ti.ndarray(ti.i32, shape=n)
    values = ti.ndarray(ti.i32, shape=n)
    offsets = ti.ndarray(ti.i32, shape=num_bins + 1)
    output = ti.ndarray(ti.i32, shape=n)
    keys.from_numpy(keys_np)
    values.from_numpy(values_np)
    offsets.fill(-7)
    output.fill(-99)
    workspace = ti.algorithms.BucketBuilderWorkspace(
        max_items=n, max_bins=num_bins
    )
    ti.algorithms.experimental_bucket_builder(
        keys, values, offsets, output, method=method, workspace=workspace
    )
    _assert_bucket_matches(
        keys_np, values_np, offsets.to_numpy(), output.to_numpy(), expected_offsets
    )
    assert workspace.workspace_bytes_peak >= 0


@test_utils.test(arch=[ti.cuda])
def test_experimental_bucket_builder_cuda_device_ndarray():
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "cuda_device_bucket_builder_available")
        and prog.cuda_device_bucket_builder_available()
    ):
        pytest.skip("CUDA driver bucket builder is unavailable in this runtime.")
    _run_ndarray_bucket_builder("auto")
    _run_ndarray_bucket_builder("cuda_device")


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_bucket_builder_vulkan_native_ndarray():
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "vulkan_bucket_builder_available")
        and prog.vulkan_bucket_builder_available()
    ):
        pytest.skip("Vulkan native bucket builder is unavailable in this runtime.")
    _run_ndarray_bucket_builder("auto")
    _run_ndarray_bucket_builder("vulkan_native")


@pytest.mark.run_in_serial
@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_bucket_builder_vulkan_reset_with_live_ndarray():
    n = 1024
    num_bins = 64
    keys_np, values_np, expected_offsets = _bucket_input(n, num_bins)
    keys = ti.ndarray(ti.i32, shape=n)
    values = ti.ndarray(ti.i32, shape=n)
    offsets = ti.ndarray(ti.i32, shape=num_bins + 1)
    output = ti.ndarray(ti.i32, shape=n)
    keys.from_numpy(keys_np)
    values.from_numpy(values_np)
    offsets.fill(0)
    output.fill(0)
    ti.algorithms.experimental_bucket_builder(
        keys, values, offsets, output, method="vulkan_native"
    )
    _assert_bucket_matches(
        keys_np, values_np, offsets.to_numpy(), output.to_numpy(), expected_offsets
    )
    ti.reset()
    del keys, values, offsets, output
    gc.collect()


@test_utils.test(arch=[ti.cpu])
def test_experimental_bucket_builder_cpu_native_ndarray():
    _run_ndarray_bucket_builder("auto")
    _run_ndarray_bucket_builder("cpu_native")


@test_utils.test(arch=[ti.cuda, ti.vulkan, ti.cpu], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_bucket_builder_field_kernel():
    n = 2048
    num_bins = 129
    keys_np, values_np, expected_offsets = _bucket_input(n, num_bins)
    keys = ti.field(ti.i32, shape=n)
    values = ti.field(ti.i32, shape=n)
    offsets = ti.field(ti.i32, shape=num_bins + 1)
    output = ti.field(ti.i32, shape=n)
    keys.from_numpy(keys_np)
    values.from_numpy(values_np)
    offsets.fill(0)
    output.fill(0)
    ti.algorithms.experimental_bucket_builder(
        keys, values, offsets, output, method="field_kernel"
    )
    _assert_bucket_matches(
        keys_np, values_np, offsets.to_numpy(), output.to_numpy(), expected_offsets
    )


@test_utils.test(arch=[ti.cuda, ti.vulkan, ti.cpu], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_bucket_builder_invalid_indices_ignored():
    keys_np = np.array([0, -1, 1, 9, 0], dtype=np.int32)
    values_np = np.array([10, 20, 30, 40, 50], dtype=np.int32)
    keys = ti.ndarray(ti.i32, shape=5)
    values = ti.ndarray(ti.i32, shape=5)
    offsets = ti.ndarray(ti.i32, shape=3)
    output = ti.ndarray(ti.i32, shape=5)
    keys.from_numpy(keys_np)
    values.from_numpy(values_np)
    offsets.fill(-1)
    output.fill(-1)
    ti.algorithms.experimental_bucket_builder(keys, values, offsets, output)
    expected_offsets = np.array([0, 2, 3], dtype=np.int32)
    _assert_bucket_matches(
        keys_np, values_np, offsets.to_numpy(), output.to_numpy(), expected_offsets
    )
