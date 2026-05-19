import gc

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils


def _grouped_reduce_input(n, groups):
    keys = (np.arange(n, dtype=np.int32) * 37 + 11) % groups
    values = (np.arange(n, dtype=np.int32) % 17 - 8).astype(np.int32)
    expected = np.zeros(groups, dtype=np.int32)
    np.add.at(expected, keys, values)
    return keys, values, expected


def _run_ndarray_grouped_reduce(method):
    n = 4096
    groups = 257
    keys = ti.ndarray(ti.i32, shape=n)
    values = ti.ndarray(ti.i32, shape=n)
    output = ti.ndarray(ti.i32, shape=groups)
    keys_np, values_np, expected = _grouped_reduce_input(n, groups)
    keys.from_numpy(keys_np)
    values.from_numpy(values_np)
    output.fill(np.int32(-777))
    workspace = ti.algorithms.GroupedReduceWorkspace(max_items=n, max_groups=groups)
    ti.algorithms.experimental_grouped_reduce(
        keys, values, output, method=method, workspace=workspace
    )
    assert np.array_equal(output.to_numpy(), expected)
    assert workspace.workspace_bytes_peak >= 0


@test_utils.test(arch=[ti.cuda])
def test_experimental_grouped_reduce_cuda_device_ndarray_i32():
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "cuda_device_grouped_reduce_available")
        and prog.cuda_device_grouped_reduce_available()
    ):
        pytest.skip("CUDA toolkit grouped reduce is unavailable in this runtime.")

    _run_ndarray_grouped_reduce("auto")
    _run_ndarray_grouped_reduce("cuda_device")
    _run_ndarray_grouped_reduce("cuda_segmented")


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_grouped_reduce_vulkan_native_ndarray_i32():
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "vulkan_grouped_reduce_available")
        and prog.vulkan_grouped_reduce_available()
    ):
        pytest.skip("Vulkan native grouped reduce is unavailable in this runtime.")

    _run_ndarray_grouped_reduce("auto")
    _run_ndarray_grouped_reduce("vulkan_native")
    _run_ndarray_grouped_reduce("vulkan_segmented")


@pytest.mark.run_in_serial
@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_grouped_reduce_vulkan_reset_with_live_ndarray():
    n = 1024
    groups = 64
    keys = ti.ndarray(ti.i32, shape=n)
    values = ti.ndarray(ti.i32, shape=n)
    output = ti.ndarray(ti.i32, shape=groups)
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "vulkan_grouped_reduce_available")
        and prog.vulkan_grouped_reduce_available()
    ):
        pytest.skip("Vulkan native grouped reduce is unavailable in this runtime.")

    keys_np, values_np, expected = _grouped_reduce_input(n, groups)
    keys.from_numpy(keys_np)
    values.from_numpy(values_np)
    output.fill(np.int32(-777))
    ti.algorithms.experimental_grouped_reduce(keys, values, output, method="vulkan_native")
    assert np.array_equal(output.to_numpy(), expected)
    ti.reset()
    del keys, values, output
    gc.collect()


@test_utils.test(arch=[ti.cpu])
def test_experimental_grouped_reduce_cpu_native_ndarray_i32():
    _run_ndarray_grouped_reduce("auto")
    _run_ndarray_grouped_reduce("cpu_native")


@test_utils.test(arch=[ti.cuda, ti.vulkan, ti.cpu], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_grouped_reduce_field_kernel_i32():
    n = 2048
    groups = 129
    keys = ti.field(ti.i32, shape=n)
    values = ti.field(ti.i32, shape=n)
    output = ti.field(ti.i32, shape=groups)
    keys_np, values_np, expected = _grouped_reduce_input(n, groups)
    keys.from_numpy(keys_np)
    values.from_numpy(values_np)
    output.fill(np.int32(-777))
    ti.algorithms.experimental_grouped_reduce(keys, values, output, method="field_kernel")
    assert np.array_equal(output.to_numpy(), expected)


@test_utils.test(arch=[ti.cuda, ti.vulkan, ti.cpu], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_grouped_reduce_invalid_keys_are_ignored():
    keys = ti.ndarray(ti.i32, shape=5)
    values = ti.ndarray(ti.i32, shape=5)
    output = ti.ndarray(ti.i32, shape=4)
    keys.from_numpy(np.array([0, -1, 2, 99, 2], dtype=np.int32))
    values.from_numpy(np.array([10, 20, 30, 40, 5], dtype=np.int32))
    output.fill(np.int32(-777))
    ti.algorithms.experimental_grouped_reduce(keys, values, output, method="auto")
    assert np.array_equal(output.to_numpy(), np.array([10, 0, 35, 0], dtype=np.int32))


@test_utils.test(arch=[ti.cpu])
def test_experimental_grouped_reduce_rejects_non_sum_op():
    keys = ti.ndarray(ti.i32, shape=4)
    values = ti.ndarray(ti.i32, shape=4)
    output = ti.ndarray(ti.i32, shape=2)
    with pytest.raises(ValueError, match="grouped reduce op"):
        ti.algorithms.experimental_grouped_reduce(keys, values, output, op="max")
