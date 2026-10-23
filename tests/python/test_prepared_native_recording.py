"""Private native workspaces must survive capture and retire with their owners."""

import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("storage", ("ndarray", "field"))
@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False, vulkan_sparse_experimental=False)
def test_persistent_sort_snapshot_and_shared_capture(storage, monkeypatch):
    n = 257
    trees = []
    if storage == "field":
        keys, values = ti.field(ti.u64), ti.field(ti.i64)
        for value in (keys, values):
            builder = ti.FieldsBuilder()
            builder.dense(ti.i, n).place(value)
            trees.append(builder.finalize())
    else:
        keys, values = ti.ndarray(ti.u64, n), ti.ndarray(ti.i64, n)
    source = np.arange(n, dtype=np.uint64) * 19 % 43 + np.uint64(1 << 48)
    payload = np.arange(n, dtype=np.int64) + (1 << 42)
    keys.from_numpy(source)
    values.from_numpy(payload)
    plan = ti.algorithms.prepare_sort(keys, values)
    assert plan.report()["persistent_recording_available"]
    recording = plan.record(persistent=True)
    assert recording.workspace_bytes > 0
    np.testing.assert_array_equal(keys.to_numpy(), source)
    plan.close()
    graphs = []
    for _ in range(2):
        builder = ti.graph.GraphBuilder()
        builder.append_native(recording)
        graphs.append(builder.compile())
    # Any ordinary provider call would invalidate the persistent replay claim.
    from taichi_forge.algorithms import _prepared_recording

    def unexpected(*args, **kwargs):
        raise AssertionError("persistent recording fell back to host provider execution")

    monkeypatch.setattr(_prepared_recording._Recording, "execute", unexpected)
    for index in range(3):
        data = np.roll(source, index)
        keys.from_numpy(data)
        values.from_numpy(payload)
        graphs[index % 2].run({})
        order = np.argsort(data, kind="stable")
        np.testing.assert_array_equal(keys.to_numpy(), data[order])
        np.testing.assert_array_equal(values.to_numpy(), payload[order])
    graphs[0].close()
    keys.from_numpy(source)
    values.from_numpy(payload)
    completion = graphs[1].submit({})
    graphs[1].close()
    completion.wait()
    np.testing.assert_array_equal(keys.to_numpy(), np.sort(source))
    if trees:
        builder = ti.graph.GraphBuilder()
        builder.append_native(recording)
        graph = builder.compile()
        graph.run({})
        trees[0].destroy()
        with pytest.raises(RuntimeError, match="retired|destroyed|generation|stale|inactive|SNode"):
            graph.run({})
        graph.close()


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False, vulkan_sparse_experimental=False)
def test_persistent_compact_columns_refresh_and_retirement(monkeypatch):
    n = 73
    values = ti.field(ti.i32, shape=n)
    wide = ti.field(ti.u64, shape=n)
    flags = ti.field(ti.i32, shape=n)
    output = ti.field(ti.i32, shape=n)
    wide_output = ti.field(ti.u64, shape=n)
    count = ti.field(ti.i32, shape=())
    plan = ti.algorithms.prepare_compact_columns(((values, output), (wide, wide_output)), flags, count)
    recording = plan.record(persistent=True)
    builder = ti.graph.GraphBuilder()
    builder.append_native(recording)
    graph = builder.compile()
    plan.close()
    from taichi_forge.algorithms import _prepared_recording

    def unexpected(*args, **kwargs):
        raise AssertionError("persistent compact invoked the ordinary provider")

    monkeypatch.setattr(_prepared_recording._Recording, "execute", unexpected)
    source = np.arange(n, dtype=np.int32) * 7
    wide_source = np.arange(n, dtype=np.uint64) + np.uint64(1 << 47)
    values.from_numpy(source)
    wide.from_numpy(wide_source)
    for mask in (np.arange(n) % 3, np.zeros(n), np.ones(n), np.arange(n) % 2):
        selected = mask != 0
        flags.from_numpy(mask.astype(np.int32))
        output.fill(-1)
        wide_output.fill(1)
        graph.run({})
        actual_count = int(count[None])
        assert actual_count == np.count_nonzero(selected)
        np.testing.assert_array_equal(output.to_numpy()[:actual_count], source[selected])
        np.testing.assert_array_equal(wide_output.to_numpy()[:actual_count], wide_source[selected])
    pending = graph.submit({})
    ti.reset()
    pending.wait()
    graph.close()
    assert recording._owner.workspace_bytes() == 0


@test_utils.test(arch=ti.cuda)
def test_persistent_sort_does_not_reinterpret_bitwise_nan_order():
    keys = ti.ndarray(ti.f32, 3)
    plan = ti.algorithms.prepare_sort(keys, nan_policy="bitwise")
    assert not plan.report()["persistent_recording_available"]
    with pytest.raises(RuntimeError, match="nan_policy"):
        plan.record(persistent=True)
    plan.run()
    plan.close()


@test_utils.test(arch=[ti.cuda, ti.vulkan], debug=True)
def test_persistent_recording_rejects_debug_during_cold_admission():
    plan = ti.algorithms.prepare_sort(ti.ndarray(ti.i32, 5))
    assert not plan.report()["persistent_recording_available"]
    with pytest.raises(RuntimeError, match="non-debug"):
        plan.record(persistent=True)
    plan.run()


@test_utils.test(arch=ti.vulkan, kernel_profiler=True)
def test_persistent_recording_rejects_vulkan_profiler_during_cold_admission():
    plan = ti.algorithms.prepare_sort(ti.ndarray(ti.i32, 5))
    assert not plan.report()["persistent_recording_available"]
    with pytest.raises(RuntimeError, match="kernel_profiler"):
        plan.record(persistent=True)
    plan.run()


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False, vulkan_sparse_experimental=False)
def test_persistent_primitives_between_kernels_and_automatic_ad(monkeypatch):
    n = 83
    keys, values, flags, selected, count, output = (
        ti.field(ti.i32, shape=shape) for shape in (n, n, n, n, (), n)
    )

    @ti.kernel
    def produce(seed: ti.i32):
        for i in keys:
            keys[i] = (i * 17 + seed) % 31
            values[i] = i
            flags[i] = (i + seed) % 3
            output[i] = -1

    @ti.kernel
    def consume():
        for i in range(count[None]):
            output[i] = selected[i] * 5

    sort = ti.algorithms.prepare_sort(keys, values)
    compact = ti.algorithms.prepare_compact(values, flags, selected, count)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(produce, ti.graph.Arg(ti.graph.ArgKind.SCALAR, "seed", ti.i32))
    for plan in (sort, compact):
        builder.append_native(plan.record(persistent=True))
    builder.dispatch(consume)
    graph = builder.compile()
    binding = graph.bind({"seed": 3})
    graph.run(binding)

    def unexpected(*args, **kwargs):
        raise AssertionError("warm replay prepared a new Vulkan recording")

    monkeypatch.setattr(ti._lib.core, "_prepare_vulkan_graph_recording", unexpected)
    for seed in (4, 5):
        binding.update(seed=seed)
        graph.run(binding)
        host = np.arange(n)
        order = np.argsort((host * 17 + seed) % 31, kind="stable")
        expected = order[(host + seed) % 3 != 0] * 5
        np.testing.assert_array_equal(output.to_numpy()[:len(expected)], expected)
        np.testing.assert_array_equal(output.to_numpy()[len(expected):], -1)

    loss = ti.field(ti.f32, shape=(), needs_grad=True, needs_dual=True)
    parameter = ti.field(ti.f32, shape=(), needs_grad=True, needs_dual=True)
    before = output.to_numpy()
    for context in (ti.ad.Tape(loss), ti.ad.FwdMode(loss=loss, param=parameter, seed=[1])):
        with context:
            with pytest.raises(RuntimeError, match="ti.ad.Tape|ti.ad.FwdMode"):
                graph.run(binding)
        np.testing.assert_array_equal(output.to_numpy(), before)
    graph.close()
