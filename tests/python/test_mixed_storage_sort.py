"""Mixed scalar storage shares native sorting and cached binding lifetimes."""

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.algorithms import _algorithms as algorithms
from tests import test_utils


@pytest.mark.parametrize("field_keys", [False, True])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_mixed_sort_cache_stability_and_rebinding(field_keys, monkeypatch):
    n = 259
    fixed = ti.field(ti.f32 if field_keys else ti.i32, n)
    moving = ti.ndarray(ti.i32 if field_keys else ti.f32, n)
    keys, values = (fixed, moving) if field_keys else (moving, fixed)
    data = ((np.arange(n) * 17) % 31 - 15).astype(np.float32)
    data[:7] = [np.nan, -0.0, 0.0, np.inf, -np.inf, np.nan, -0.0]
    expected = np.argsort(data, kind="stable")
    workspace = ti.algorithms.SortWorkspace()

    def no_host(*args, **kwargs):
        raise AssertionError("Mixed native sorting downloaded its inputs")

    monkeypatch.setattr(algorithms, "_host_stable_sort", no_host)
    method = {ti.cpu: "cpu_native", ti.cuda: "cuda_device", ti.vulkan: "vulkan_native_radix_u32"}[
        ti.lang.impl.current_cfg().arch
    ]
    for selected in ("auto", method, "auto"):
        keys.from_numpy(data)
        values.from_numpy(np.arange(n, dtype=np.int32))
        ti.algorithms.sort(keys, values, workspace=workspace, method=selected)
        if selected == "auto":
            cached = workspace._mixed_storage_sort_plan
        else:
            assert workspace._mixed_storage_sort_plan is cached
        np.testing.assert_array_equal(values.to_numpy(), expected)
        # Includes signed zero and NaN payload preservation.
        np.testing.assert_array_equal(keys.to_numpy().view(np.uint32), data[expected].view(np.uint32))

    replacement = ti.ndarray(moving.dtype, n)
    keys, values = (fixed, replacement) if field_keys else (replacement, fixed)
    keys.from_numpy(data)
    values.from_numpy(np.arange(n, dtype=np.int32))
    ti.algorithms.sort_by_key([keys], values, workspace=workspace)
    assert workspace._mixed_storage_sort_plan is not cached
    assert cached.report()["closed"]
    workspace.clear()  # Release bindings while the ordered native work is in flight.
    assert workspace._mixed_storage_sort_plan is None
    np.testing.assert_array_equal(values.to_numpy(), expected)
    # No workspace is required; temporary bindings keep the same semantics.
    ti.algorithms.sort(keys, values)
    np.testing.assert_array_equal(values.to_numpy(), expected)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_mixed_sort_preserves_explicit_host_and_strided_fallback(monkeypatch):
    keys, sibling = ti.field(ti.i32), ti.field(ti.i32)
    ti.root.dense(ti.i, 7).place(keys, sibling)
    values = ti.ndarray(ti.i32, 7)
    data = np.array([3, 1, 3, -2, 0, 1, -2], np.int32)
    host = algorithms._host_stable_sort
    calls = []

    def observe(*args, **kwargs):
        calls.append(True)
        return host(*args, **kwargs)

    monkeypatch.setattr(algorithms, "_host_stable_sort", observe)
    for method, descending in (("auto", False), ("host_stable", False), ("auto", True)):
        keys.from_numpy(data)
        sibling.fill(123)
        values.from_numpy(np.arange(7, dtype=np.int32))
        ti.algorithms.sort(keys, values, method=method, descending=descending)
        expected = np.argsort(-data if descending else data, kind="stable")
        np.testing.assert_array_equal(values.to_numpy(), expected)
        np.testing.assert_array_equal(sibling.to_numpy(), [123] * 7)
    assert len(calls) == 3


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_mixed_sort_cache_preserves_nan_policy():
    keys, values = ti.field(ti.f32, 7), ti.ndarray(ti.i32, 7)
    data = np.array([np.nan, -3, 1, np.inf, -np.inf, 1, np.nan], np.float32)
    data.view(np.uint32)[0] = 0xFFC00001
    bits = data.view(np.uint32)
    sortable = np.where(bits >> 31, ~bits, bits ^ np.uint32(1 << 31))
    workspace = ti.algorithms.SortWorkspace()
    previous = None
    for policy in ("last", "bitwise", "last"):
        keys.from_numpy(data)
        values.from_numpy(np.arange(7, dtype=np.int32))
        ti.algorithms.sort(keys, values, workspace=workspace, nan_policy=policy)
        if previous is not None:
            assert previous.report()["closed"]
            assert previous is not workspace._mixed_storage_sort_plan
        previous = workspace._mixed_storage_sort_plan
        expected = np.argsort(data if policy == "last" else sortable, kind="stable")
        np.testing.assert_array_equal(values.to_numpy(), expected)
    workspace.clear()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_mixed_sort_workspace_rebinds_after_runtime_reset():
    arch = ti.lang.impl.current_cfg().arch
    workspace = ti.algorithms.SortWorkspace()
    keys, values = ti.field(ti.i32, 3), ti.ndarray(ti.i32, 3)
    keys.from_numpy(np.array([2, 1, 0], np.int32))
    values.from_numpy(np.array([0, 1, 2], np.int32))
    ti.algorithms.sort(keys, values, workspace=workspace)
    previous = workspace._mixed_storage_sort_plan
    ti.reset()
    ti.init(arch=arch, offline_cache=False)
    keys, values = ti.field(ti.i32, 3), ti.ndarray(ti.i32, 3)
    keys.from_numpy(np.array([1, 0, 2], np.int32))
    values.from_numpy(np.array([0, 1, 2], np.int32))
    ti.algorithms.sort(keys, values, workspace=workspace)
    assert previous.report()["closed"]
    np.testing.assert_array_equal(values.to_numpy(), [1, 0, 2])
    workspace.clear()
