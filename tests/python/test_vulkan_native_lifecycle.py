import gc

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils


@pytest.mark.run_in_serial
@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_vulkan_native_primitive_cache_reset_after_mixed_use():
    prog = impl.get_runtime().prog
    live_arrays = []
    ran = 0

    if (
        hasattr(prog, "vulkan_scatter_add_available")
        and prog.vulkan_scatter_add_available()
    ):
        n = 1024
        buckets = 64
        src_np = (np.arange(n, dtype=np.int32) % 17) - 8
        indices_np = (np.arange(n, dtype=np.int32) * 7 + 3) % buckets
        base_np = np.full(buckets, 5, dtype=np.int32)
        expected = base_np.copy()
        np.add.at(expected, indices_np, src_np)
        src = ti.ndarray(ti.i32, shape=n)
        indices = ti.ndarray(ti.i32, shape=n)
        dst = ti.ndarray(ti.i32, shape=buckets)
        src.from_numpy(src_np)
        indices.from_numpy(indices_np)
        dst.from_numpy(base_np)
        ti.algorithms.experimental_scatter_add(
            src, indices, dst, method="vulkan_native"
        )
        assert np.array_equal(dst.to_numpy(), expected)
        live_arrays.extend([src, indices, dst])
        ran += 1

    if (
        hasattr(prog, "vulkan_histogram_available")
        and prog.vulkan_histogram_available()
        and prog.vulkan_histogram_value_type_available(0, 0)
    ):
        n = 1024
        num_bins = 64
        values_np = ((np.arange(n, dtype=np.int32) * 5 + 1) % num_bins).astype(
            np.int32
        )
        expected = np.bincount(values_np, minlength=num_bins).astype(np.int32)
        values = ti.ndarray(ti.i32, shape=n)
        bins = ti.ndarray(ti.i32, shape=num_bins)
        values.from_numpy(values_np)
        bins.fill(0)
        ti.algorithms.experimental_histogram(values, bins, method="vulkan_native")
        assert np.array_equal(bins.to_numpy(), expected)
        live_arrays.extend([values, bins])
        ran += 1

    if (
        hasattr(prog, "vulkan_bucket_builder_available")
        and prog.vulkan_bucket_builder_available()
    ):
        n = 1024
        num_bins = 64
        keys_np = (np.arange(n, dtype=np.int32) * 11 + 2) % num_bins
        values_np = (np.arange(n, dtype=np.int32) * 3) - 7
        counts = np.bincount(keys_np, minlength=num_bins).astype(np.int32)
        expected_offsets = np.zeros(num_bins + 1, dtype=np.int32)
        expected_offsets[1:] = np.cumsum(counts)
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
        assert np.array_equal(offsets.to_numpy(), expected_offsets)
        live_arrays.extend([keys, values, offsets, output])
        ran += 1

    if (
        hasattr(prog, "vulkan_grouped_reduce_available")
        and prog.vulkan_grouped_reduce_available()
    ):
        n = 1024
        groups = 64
        keys_np = (np.arange(n, dtype=np.int32) * 37 + 11) % groups
        values_np = (np.arange(n, dtype=np.int32) % 17) - 8
        expected = np.zeros(groups, dtype=np.int32)
        np.add.at(expected, keys_np, values_np)
        keys = ti.ndarray(ti.i32, shape=n)
        values = ti.ndarray(ti.i32, shape=n)
        output = ti.ndarray(ti.i32, shape=groups)
        keys.from_numpy(keys_np)
        values.from_numpy(values_np)
        output.fill(0)
        ti.algorithms.experimental_grouped_reduce(
            keys, values, output, method="vulkan_native"
        )
        assert np.array_equal(output.to_numpy(), expected)
        live_arrays.extend([keys, values, output])
        ran += 1

    if ran == 0:
        pytest.skip("No Vulkan native primitive was available in this runtime.")

    ti.reset()
    del live_arrays
    gc.collect()
