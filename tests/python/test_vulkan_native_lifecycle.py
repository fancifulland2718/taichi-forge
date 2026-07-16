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


@pytest.mark.run_in_serial
@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_vulkan_native_batch2_resource_replay_ring_wrap(monkeypatch):
    monkeypatch.setenv("TI_VULKAN_RESOURCE_REPLAY_RING_SIZE", "2")
    monkeypatch.setenv("TI_VULKAN_REDUCE_I32_SUM_ATOMIC", "0")
    monkeypatch.setenv("TI_VULKAN_REDUCE_SINGLE_SHARED_MAX_N", "0")
    monkeypatch.setenv("TI_VULKAN_COMPACT_FUSE_MAX_N", "0")
    monkeypatch.setenv("TI_VULKAN_BUCKET_BUILDER_PRIVATE", "1")
    monkeypatch.setenv("TI_VULKAN_BUCKET_BUILDER_PRIVATE_MIN_N", "1")
    monkeypatch.setenv("TI_VULKAN_BUCKET_BUILDER_PRIVATE_MAX_BINS", "128")
    monkeypatch.setenv("TI_VULKAN_BUCKET_BUILDER_PRIVATE_MAX_BYTES", "1048576")

    prog = impl.get_runtime().prog
    required = (
        ("vulkan_reduce_available", ()),
        ("vulkan_compact_available", ()),
        ("vulkan_bucket_builder_available", ()),
        ("vulkan_grouped_reduce_available", ()),
    )
    for name, args in required:
        if not hasattr(prog, name) or not getattr(prog, name)(*args):
            pytest.skip(f"{name} is unavailable in this runtime.")

    def run_reduce(values_np):
        values = ti.ndarray(ti.i32, shape=values_np.shape[0])
        output = ti.ndarray(ti.i32, shape=1)
        values.from_numpy(values_np)
        output.fill(0)
        ti.algorithms.experimental_reduce(values, output, method="vulkan_native")
        assert output.to_numpy()[0] == np.sum(values_np, dtype=np.int32)

    reduce_a = (np.arange(768, dtype=np.int32) % 37) - 18
    reduce_b = (np.arange(1024, dtype=np.int32) * 3 % 41) - 20
    for values_np in (reduce_a, reduce_a, reduce_b, reduce_a):
        run_reduce(values_np)

    def run_compact(values_np, flags_np):
        values = ti.ndarray(ti.i32, shape=values_np.shape[0])
        flags = ti.ndarray(ti.i32, shape=flags_np.shape[0])
        output = ti.ndarray(ti.i32, shape=values_np.shape[0])
        count = ti.ndarray(ti.i32, shape=1)
        values.from_numpy(values_np)
        flags.from_numpy(flags_np)
        output.fill(-1)
        count.fill(-1)
        ti.algorithms.experimental_compact(
            values, flags, output, count, method="vulkan_native"
        )
        expected = values_np[flags_np != 0]
        assert count.to_numpy()[0] == expected.shape[0]
        assert np.array_equal(output.to_numpy()[: expected.shape[0]], expected)

    compact_a = (np.arange(512, dtype=np.int32) * 5) - 11
    compact_b = (np.arange(768, dtype=np.int32) * 7) - 13
    flags_a = ((np.arange(512) % 3 == 0) | (np.arange(512) % 11 == 0)).astype(
        np.int32
    )
    flags_b = ((np.arange(768) % 5 == 0) | (np.arange(768) % 13 == 0)).astype(
        np.int32
    )
    for values_np, flags_np in (
        (compact_a, flags_a),
        (compact_a, flags_a),
        (compact_b, flags_b),
        (compact_a, flags_a),
    ):
        run_compact(values_np, flags_np)

    def bucket_expected(keys_np, values_np, num_bins):
        counts = np.bincount(
            keys_np[(keys_np >= 0) & (keys_np < num_bins)], minlength=num_bins
        )
        offsets = np.zeros(num_bins + 1, dtype=np.int32)
        offsets[1:] = np.cumsum(counts, dtype=np.int64).astype(np.int32)
        return offsets

    def run_bucket(keys_np, values_np, num_bins):
        keys = ti.ndarray(ti.i32, shape=keys_np.shape[0])
        values = ti.ndarray(ti.i32, shape=values_np.shape[0])
        offsets = ti.ndarray(ti.i32, shape=num_bins + 1)
        output = ti.ndarray(ti.i32, shape=values_np.shape[0])
        keys.from_numpy(keys_np)
        values.from_numpy(values_np)
        offsets.fill(-1)
        output.fill(0)
        ti.algorithms.experimental_bucket_builder(
            keys, values, offsets, output, method="vulkan_native"
        )
        expected_offsets = bucket_expected(keys_np, values_np, num_bins)
        assert np.array_equal(offsets.to_numpy(), expected_offsets)

    bucket_bins = 32
    bucket_a_keys = ((np.arange(2048, dtype=np.int32) * 17 + 3) % bucket_bins)
    bucket_b_keys = ((np.arange(3072, dtype=np.int32) * 19 + 5) % bucket_bins)
    bucket_a_values = (np.arange(2048, dtype=np.int32) * 2) - 7
    bucket_b_values = (np.arange(3072, dtype=np.int32) * 3) - 9
    for keys_np, values_np in (
        (bucket_a_keys, bucket_a_values),
        (bucket_a_keys, bucket_a_values),
        (bucket_b_keys, bucket_b_values),
        (bucket_a_keys, bucket_a_values),
    ):
        run_bucket(keys_np, values_np, bucket_bins)

    def run_grouped_reduce(keys_np, values_np, groups, method):
        keys = ti.ndarray(ti.i32, shape=keys_np.shape[0])
        values = ti.ndarray(ti.i32, shape=values_np.shape[0])
        output = ti.ndarray(ti.i32, shape=groups)
        keys.from_numpy(keys_np)
        values.from_numpy(values_np)
        output.fill(-777)
        ti.algorithms.experimental_grouped_reduce(
            keys, values, output, method=method
        )
        expected = np.zeros(groups, dtype=np.int32)
        np.add.at(expected, keys_np, values_np)
        assert np.array_equal(output.to_numpy(), expected)

    grouped_keys_a = (np.arange(1024, dtype=np.int32) * 7 + 1) % bucket_bins
    grouped_keys_b = (np.arange(1536, dtype=np.int32) * 11 + 2) % bucket_bins
    grouped_values_a = (np.arange(1024, dtype=np.int32) % 23) - 11
    grouped_values_b = (np.arange(1536, dtype=np.int32) % 29) - 14
    for method in ("vulkan_native", "vulkan_two_level"):
        for keys_np, values_np in (
            (grouped_keys_a, grouped_values_a),
            (grouped_keys_a, grouped_values_a),
            (grouped_keys_b, grouped_values_b),
            (grouped_keys_a, grouped_values_a),
        ):
            run_grouped_reduce(keys_np, values_np, bucket_bins, method)
