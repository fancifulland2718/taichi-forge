"""Ordinary tensor Field sorting shares prepared bindings without downloading."""

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.algorithms import _algorithms as algorithms
from tests import test_utils


@pytest.mark.parametrize("n", [1, 17])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_tensor_field_sort_native_cache_rebind_and_clear(n, monkeypatch):
    pairs = [(ti.field(ti.f32, n), ti.Matrix.field(2, 3, ti.f32, n)) for _ in range(2)]
    data = ((np.arange(n) * 17) % 7 - 3).astype(np.float32)
    if n > 1:
        data[:7] = [np.nan, -0.0, 0.0, np.inf, -np.inf, np.nan, -0.0]
        data.view(np.uint32)[0] = 0xFFC00013
    payload = np.arange(n * 6, dtype=np.float32).reshape(n, 2, 3)
    if n > 1:
        payload.view(np.uint32).reshape(-1)[:4] = [0x7FC12345, 0xFFC54321, 0x80000000, 0]
    order = np.argsort(data, kind="stable")
    workspace = ti.algorithms.SortWorkspace()
    method = {ti.cpu: "cpu_native", ti.cuda: "cuda_device", ti.vulkan: "vulkan_native_radix_u32"}[ti.cfg.arch]

    def no_host(*args, **kwargs):
        raise AssertionError("Native tensor Field sort downloaded its inputs")

    monkeypatch.setattr(algorithms, "_host_stable_sort", no_host)
    previous = None
    for pair in (pairs[0], pairs[1], pairs[0]):
        keys, values = pair
        for selected in ("auto", method):
            keys.from_numpy(data)
            values.from_numpy(payload)
            with monkeypatch.context() as bounded:
                bounded.setattr(keys, "to_numpy", no_host)
                bounded.setattr(values, "to_numpy", no_host)
                ti.algorithms.sort_by_key([keys], values, workspace=workspace, method=selected)
            cached = workspace._prepared_storage_sort_plan
            if selected == method:
                assert cached is previous
            elif previous is not None:
                assert cached is not previous and previous.report()["closed"]
            previous = cached
            assert keys.to_numpy().tobytes() == data[order].tobytes()
            assert values.to_numpy().tobytes() == payload[order].tobytes()
    # Clearing the workspace can retire bindings while native work is queued.
    keys.from_numpy(data)
    values.from_numpy(payload)
    ti.algorithms.sort(keys, values, workspace=workspace)
    workspace.clear()
    assert previous.report()["closed"] and workspace._prepared_storage_sort_plan is None
    assert values.to_numpy().tobytes() == payload[order].tobytes()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_tensor_field_sort_preserves_host_choice_and_strided_fallback(monkeypatch):
    keys, sibling = ti.field(ti.i32), ti.field(ti.i32)
    ti.root.dense(ti.i, 7).place(keys, sibling)
    values = ti.Vector.field(3, ti.i32, 7)
    compact_keys = ti.field(ti.i32, 7)
    data = np.array([3, 1, 3, -2, 0, 1, -2], np.int32)
    payload = np.arange(21, dtype=np.int32).reshape(7, 3)
    host = algorithms._host_stable_sort
    calls = []

    def observe(*args, **kwargs):
        calls.append(True)
        return host(*args, **kwargs)

    monkeypatch.setattr(algorithms, "_host_stable_sort", observe)
    for target, method, descending in (
        (keys, "auto", False),
        (compact_keys, "host_stable", False),
        (compact_keys, "auto", True),
    ):
        target.from_numpy(data)
        sibling.fill(123)
        values.from_numpy(payload)
        ti.algorithms.sort(target, values, method=method, descending=descending)
        order = np.argsort(-data if descending else data, kind="stable")
        np.testing.assert_array_equal(values.to_numpy(), payload[order])
        np.testing.assert_array_equal(sibling.to_numpy(), [123] * 7)
    assert len(calls) == 3
