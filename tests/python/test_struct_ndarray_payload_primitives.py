import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils


def _payload_type():
    return ti.types.struct(
        depth=ti.f32,
        color=ti.types.vector(3, ti.f32),
        idx=ti.i32,
    )


def _payload_data(dtype, n):
    data = np.zeros((n,), dtype=dtype)
    data["depth"] = (np.arange(n, dtype=np.float32) * np.float32(0.25)) - 7.0
    data["color"] = (
        (np.arange(n * 3, dtype=np.float32).reshape(n, 3) % 37)
        * np.float32(0.5)
    )
    data["idx"] = np.arange(n, dtype=np.int32) * 5 - 13
    return data


def _record_words(arr):
    contiguous = np.ascontiguousarray(arr)
    return np.sort(contiguous.view(np.dtype((np.void, contiguous.dtype.itemsize))).reshape(-1))


def _assert_record_set_equal(actual, expected):
    np.testing.assert_array_equal(_record_words(actual), _record_words(expected))


def _run_struct_compact(method):
    n = 513
    payload = _payload_type()
    values = ti.ndarray(payload, shape=n)
    flags = ti.ndarray(ti.i32, shape=n)
    output = ti.ndarray(payload, shape=n)
    count = ti.ndarray(ti.i32, shape=1)

    values_np = _payload_data(values.numpy_dtype, n)
    flags_np = ((np.arange(n) % 3 == 0) | (np.arange(n) % 11 == 0)).astype(np.int32)
    values.from_numpy(values_np)
    flags.from_numpy(flags_np)
    output.fill(0)
    count.from_numpy(np.array([-1], dtype=np.int32))

    ti.algorithms.experimental_compact(values, flags, output, count, method=method)

    expected = values_np[flags_np != 0]
    assert count.to_numpy()[0] == expected.shape[0]
    np.testing.assert_array_equal(output.to_numpy()[: expected.shape[0]], expected)


def _run_struct_indexed_copy(method):
    n = 384
    payload = _payload_type()
    src = ti.ndarray(payload, shape=n)
    indices = ti.ndarray(ti.i32, shape=n)
    dst = ti.ndarray(payload, shape=n)

    values_np = _payload_data(src.numpy_dtype, n)
    indices_np = (n - 1 - np.arange(n, dtype=np.int32)).astype(np.int32)
    src.from_numpy(values_np)
    indices.from_numpy(indices_np)

    dst.fill(0)
    ti.algorithms.experimental_gather(src, indices, dst, method=method)
    np.testing.assert_array_equal(dst.to_numpy(), values_np[indices_np])

    dst.fill(0)
    ti.algorithms.experimental_scatter(src, indices, dst, method=method)
    np.testing.assert_array_equal(dst.to_numpy(), values_np[indices_np])


def _run_struct_sort(method):
    keys_np = np.array(
        [3, -1, 3, 0, -7, 2, -1, 5, 0, 3, -7, 8, 2, 2, -1, 4],
        dtype=np.int32,
    )
    n = keys_np.shape[0]
    payload = _payload_type()
    keys = ti.ndarray(ti.i32, shape=n)
    values = ti.ndarray(payload, shape=n)
    values_np = _payload_data(values.numpy_dtype, n)

    keys.from_numpy(keys_np)
    values.from_numpy(values_np)
    ti.algorithms.sort(keys, values, method=method)

    order = np.argsort(keys_np, kind="stable")
    np.testing.assert_array_equal(keys.to_numpy(), keys_np[order])
    np.testing.assert_array_equal(values.to_numpy(), values_np[order])


def _run_struct_sort_by_key():
    key0_np = np.array([2, 1, 2, 1, 3, 2, 1, 3], dtype=np.int32)
    key1_np = np.array([9, 4, 1, 7, 0, 1, 4, 2], dtype=np.int32)
    n = key0_np.shape[0]
    payload = _payload_type()
    key0 = ti.ndarray(ti.i32, shape=n)
    key1 = ti.ndarray(ti.i32, shape=n)
    values = ti.ndarray(payload, shape=n)
    values_np = _payload_data(values.numpy_dtype, n)

    key0.from_numpy(key0_np)
    key1.from_numpy(key1_np)
    values.from_numpy(values_np)
    ti.algorithms.sort_by_key([key0, key1], values, method="host_stable")

    tie = np.arange(n)
    order = np.lexsort((tie, key1_np, key0_np))
    np.testing.assert_array_equal(key0.to_numpy(), key0_np[order])
    np.testing.assert_array_equal(key1.to_numpy(), key1_np[order])
    np.testing.assert_array_equal(values.to_numpy(), values_np[order])


def _run_struct_bucket_builder(method):
    n = 512
    num_bins = 31
    payload = _payload_type()
    keys = ti.ndarray(ti.i32, shape=n)
    values = ti.ndarray(payload, shape=n)
    offsets = ti.ndarray(ti.i32, shape=num_bins + 1)
    output = ti.ndarray(payload, shape=n)

    keys_np = ((np.arange(n, dtype=np.int32) * 17 + 5) % num_bins).astype(np.int32)
    keys_np[3] = -1
    keys_np[29] = num_bins + 9
    values_np = _payload_data(values.numpy_dtype, n)
    counts = np.bincount(keys_np[(keys_np >= 0) & (keys_np < num_bins)], minlength=num_bins)
    expected_offsets = np.zeros(num_bins + 1, dtype=np.int32)
    expected_offsets[1:] = np.cumsum(counts, dtype=np.int64).astype(np.int32)

    keys.from_numpy(keys_np)
    values.from_numpy(values_np)
    offsets.fill(-1)
    output.fill(0)

    ti.algorithms.experimental_bucket_builder(
        keys, values, offsets, output, method=method
    )

    actual_offsets = offsets.to_numpy()
    actual_output = output.to_numpy()
    np.testing.assert_array_equal(actual_offsets, expected_offsets)
    for bucket in range(num_bins):
        begin = expected_offsets[bucket]
        end = expected_offsets[bucket + 1]
        _assert_record_set_equal(
            actual_output[begin:end], values_np[keys_np == bucket]
        )


@test_utils.test(arch=[ti.cpu])
def test_struct_ndarray_payload_cpu_native_primitives():
    _run_struct_compact("cpu_native")
    _run_struct_indexed_copy("cpu_native")
    _run_struct_sort("cpu_native")
    _run_struct_sort("host_stable")
    _run_struct_sort_by_key()
    _run_struct_bucket_builder("cpu_native")


@test_utils.test(arch=[ti.cuda])
def test_struct_ndarray_payload_cuda_native_primitives():
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "cuda_cub_select_available")
        and prog.cuda_cub_select_available()
        and hasattr(prog, "cuda_cub_radix_sort_available")
        and prog.cuda_cub_radix_sort_available()
        and hasattr(prog, "cuda_device_bucket_builder_available")
        and prog.cuda_device_bucket_builder_available()
        and hasattr(prog, "cuda_device_indexed_copy_payload_available")
        and prog.cuda_device_indexed_copy_payload_available(20)
    ):
        pytest.skip("CUDA native raw-payload primitive coverage is unavailable.")

    _run_struct_compact("cuda_cub")
    _run_struct_indexed_copy("cuda_device")
    _run_struct_sort("cuda_cub_native")
    _run_struct_bucket_builder("cuda_device")


@test_utils.test(arch=[ti.cuda])
def test_struct_ndarray_payload_cuda_driver_sort():
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "cuda_device_radix_sort_available")
        and prog.cuda_device_radix_sort_available()
    ):
        pytest.skip("CUDA Driver raw-payload sort is unavailable.")
    _run_struct_sort("cuda_device")


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_struct_ndarray_payload_vulkan_native_primitives():
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "vulkan_compact_available")
        and prog.vulkan_compact_available()
        and hasattr(prog, "vulkan_radix_sort_available")
        and prog.vulkan_radix_sort_available()
        and hasattr(prog, "vulkan_indexed_copy_available")
        and prog.vulkan_indexed_copy_available()
        and hasattr(prog, "vulkan_bucket_builder_available")
        and prog.vulkan_bucket_builder_available()
    ):
        pytest.skip("Vulkan native raw-payload primitive coverage is unavailable.")

    _run_struct_compact("vulkan_native")
    _run_struct_indexed_copy("vulkan_native")
    _run_struct_sort("vulkan_native_radix_u32")
    _run_struct_bucket_builder("vulkan_native")


@test_utils.test(arch=[ti.cpu])
def test_struct_ndarray_cannot_be_sort_key_or_legacy_payload():
    payload = _payload_type()
    keys = ti.ndarray(payload, shape=4)
    scalar_keys = ti.ndarray(ti.i32, shape=4)
    values = ti.ndarray(payload, shape=4)

    with pytest.raises(TypeError, match="keys must be scalar"):
        ti.algorithms.sort(keys)
    with pytest.raises(TypeError, match="legacy"):
        ti.algorithms.sort(scalar_keys, values, method="legacy")
    with pytest.raises(TypeError, match="key parts must be scalar"):
        ti.algorithms.sort_by_key([scalar_keys, keys], values, method="host_stable")


@test_utils.test(arch=[ti.cpu])
def test_struct_ndarray_rejects_numeric_primitives_directly():
    payload = _payload_type()
    n = 8
    values = ti.ndarray(payload, shape=n)
    dst = ti.ndarray(payload, shape=n)
    scalar = ti.ndarray(ti.i32, shape=n)
    indices = ti.ndarray(ti.i32, shape=n)
    output = ti.ndarray(payload, shape=4)
    bins = ti.ndarray(ti.i32, shape=4)

    with pytest.raises(TypeError, match="does not support StructNdarray"):
        ti.algorithms.experimental_reduce(values, output, method="cpu_native")
    with pytest.raises(TypeError, match="does not support StructNdarray"):
        ti.algorithms.experimental_histogram(values, bins, method="cpu_native")
    with pytest.raises(TypeError, match="does not support StructNdarray"):
        ti.algorithms.experimental_transform(values, dst, method="cpu_native")
    with pytest.raises(TypeError, match="does not support StructNdarray"):
        ti.algorithms.experimental_scatter_add(
            values, indices, dst, method="cpu_native"
        )
    with pytest.raises(TypeError, match="does not support StructNdarray"):
        ti.algorithms.experimental_grouped_reduce(
            scalar, values, output, method="cpu_native"
        )
    with pytest.raises(TypeError, match="does not support StructNdarray"):
        ti.algorithms.PrefixSumExecutor(n).run(values)


@test_utils.test(arch=[ti.cpu])
def test_struct_ndarray_grouped_reduce_rejects_non_i32_key_member_views():
    key_payload = ti.types.struct(key=ti.f32, tag=ti.i32)
    value_payload = ti.types.struct(value=ti.i32, tag=ti.i32)
    n = 8
    keys = ti.ndarray(key_payload, shape=n)
    values = ti.ndarray(value_payload, shape=n)
    scalar_output = ti.ndarray(ti.i32, shape=4)

    with pytest.raises(TypeError, match="expects ti.i32 keys"):
        ti.algorithms.experimental_grouped_reduce(
            keys.field("key"),
            values.field("value"),
            scalar_output,
            method="cpu_native",
        )
