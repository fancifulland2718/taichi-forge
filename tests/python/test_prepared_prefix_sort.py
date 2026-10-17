import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize(
    "dtype,np_dtype",
    [
        (ti.i32, np.int32),
        (ti.u32, np.uint32),
        (ti.f32, np.float32),
        (ti.i64, np.int64),
        (ti.u64, np.uint64),
        (ti.f64, np.float64),
    ],
)
@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_prefix_sort_count_churn_stability_tail_and_overflow(dtype, np_dtype):
    capacity = 137
    keys, values = ti.ndarray(dtype, capacity), ti.ndarray(ti.i32, capacity)
    count = ti.ndarray(ti.i32, 1)
    initial = (np.arange(capacity) * 11 % 17).astype(np_dtype)
    if np.issubdtype(np_dtype, np.floating):
        initial[:8] = [np.nan, -np.inf, np.inf, -0.0, 0.0, np.nan, -1, 1]
        if np_dtype == np.float32:
            initial.view(np.uint32)[[0, 5]] = [0xFFC00001, 0x7FC0ABCD]
        else:
            initial.view(np.uint64)[[0, 5]] = [0xFFF8000000000001, 0x7FF800000000ABCD]
    elif np.issubdtype(np_dtype, np.signedinteger):
        initial -= 8
    else:
        initial += np_dtype(1 << (np.dtype(np_dtype).itemsize * 8 - 1))
    keys.from_numpy(initial)
    values.from_numpy(np.arange(capacity, dtype=np.int32))
    plan = ti.algorithms.prepare_sort(keys, values, count=count)
    np.testing.assert_array_equal(
        keys.to_numpy().view(np.uint8), initial.view(np.uint8)
    )
    assert plan.report()["device_capture"]
    assert plan.report()["count_host_readback"] is False
    workspace = plan.report()["workspace_bytes"]
    builder = ti.graph.GraphBuilder()
    builder.append_native(plan.record(), admission="auto")
    graph = builder.compile()
    for n in (0, 1, capacity, 13, 77, 2, -1, capacity + 1, 0, 33):
        keys.from_numpy(initial)
        values.from_numpy(np.arange(capacity, dtype=np.int32))
        count.from_numpy(np.array([n], dtype=np.int32))
        graph.run({})
        expected = np.arange(capacity)
        if 0 <= n <= capacity:
            expected[:n] = np.argsort(initial[:n], kind="stable")
            plan.check_status()
        else:
            with pytest.raises(ti.TaichiRuntimeError, match="invalid or overflowing"):
                plan.check_status()
        np.testing.assert_array_equal(values.to_numpy(), expected)
        # Preserve exact NaN payload and signed-zero bits, including the tail.
        np.testing.assert_array_equal(
            keys.to_numpy().view(np.uint8), initial[expected].view(np.uint8)
        )
        assert plan.report()["workspace_bytes"] == workspace
    graph.close()
    plan.close()
    with pytest.raises(ti.TaichiRuntimeError, match="closed"):
        plan.run()


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_prefix_sort_graph_producer_consumer_and_retained_close():
    capacity = 257
    keys, values, output = (ti.ndarray(ti.i32, capacity) for _ in range(3))
    extent = ti.DeviceExtent(capacity)

    @ti.kernel
    def produce(
        k: ti.types.ndarray(), v: ti.types.ndarray(), c: ti.types.ndarray(), n: ti.i32
    ):
        ti.device_extent_publish(c, capacity, n)
        for i in k:
            k[i] = (i * 17) % 31
            v[i] = i

    @ti.kernel
    def consume(k: ti.types.ndarray(), v: ti.types.ndarray(), out: ti.types.ndarray()):
        for i in out:
            out[i] = k[i] * 1000 + v[i]

    plan = ti.algorithms.prepare_sort(keys, values, count=extent)
    builder = ti.graph.GraphBuilder()
    args = {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.i32, ndim=1)
        for name in ("keys", "values", "count", "output")
    }
    n = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "n", ti.i32)
    builder.dispatch(produce, args["keys"], args["values"], args["count"], n)
    builder.append_native(plan.record(), admission="auto")
    builder.dispatch(consume, args["keys"], args["values"], args["output"])
    graph = builder.compile()
    bindings = graph.bind(
        dict(keys=keys, values=values, count=extent.state, output=output, n=19)
    )
    graph.run(bindings)
    ti.sync()
    before = ti.lang.impl.get_runtime().prog._runtime_statistics_snapshot()
    ticket = graph.submit(bindings)
    after = ti.lang.impl.get_runtime().prog._runtime_statistics_snapshot()
    assert (
        after["submission"]["backend_graph_launches"]
        - before["submission"]["backend_graph_launches"]
        == 1
    )
    # Existing recordings retain their immutable generation after handle close.
    plan.close()
    ticket.wait()
    graph.run(bindings)
    original = np.arange(capacity) * 17 % 31
    order = np.arange(capacity)
    order[:19] = np.argsort(original[:19], kind="stable")
    np.testing.assert_array_equal(output.to_numpy(), original[order] * 1000 + order)
    graph.close()


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_prefix_sort_rejects_count_alias_and_extent_overflow():
    keys = ti.ndarray(ti.i32, 1)
    with pytest.raises(ValueError, match="disjoint"):
        ti.algorithms.prepare_sort(keys, count=keys)
    extent = ti.DeviceExtent(1)
    extent.set(2)
    keys.from_numpy(np.array([19], dtype=np.int32))
    with ti.algorithms.prepare_sort(keys, count=extent) as plan:
        plan.run()
        with pytest.raises(ti.TaichiRuntimeError, match="invalid or overflowing"):
            plan.check_status()
        np.testing.assert_array_equal(keys.to_numpy(), [19])


@pytest.mark.parametrize("storage", ["field", "view"])
@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_prefix_sort_storage_and_no_replay_readback(storage, monkeypatch):
    capacity = 41
    if storage == "field":
        keys, values = (ti.field(ti.i32, shape=capacity) for _ in range(2))
        k, v = keys, values
        selection = slice(None)
    else:
        keys, values = (ti.ndarray(ti.i32, capacity + 2) for _ in range(2))
        selection = slice(1, capacity + 1)
        k, v = (
            ti.experimental.ndarray_view(x, slices=(selection,)) for x in (keys, values)
        )
    initial = (np.arange(keys.shape[0]) * 7 % 13).astype(np.int32)
    payload = np.arange(keys.shape[0], dtype=np.int32)
    keys.from_numpy(initial)
    values.from_numpy(payload)
    count = ti.ndarray(ti.i32, 1)
    count.from_numpy(np.array([23], dtype=np.int32))
    ti.sync()
    program = ti.lang.impl.get_runtime().prog
    before = program._runtime_statistics_snapshot()
    plan = ti.algorithms.prepare_sort(k, v, count=count)
    after = program._runtime_statistics_snapshot()
    assert after["submission"] == before["submission"]
    for name in ("program_syncs", "completion_waits", "backend_waits"):
        assert after["synchronization"][name] == before["synchronization"][name]

    def forbidden(*args, **kwargs):
        raise AssertionError("sort replay observed host count or allocated scratch")

    with monkeypatch.context() as patch:
        patch.setattr(count, "to_numpy", forbidden)
        from taichi_forge.lang._ndarray import ScalarNdarray

        patch.setattr(ScalarNdarray, "_private_scratch_storage", forbidden)
        plan.run()
        plan.run()
    expected = np.arange(capacity)
    expected[:23] = np.argsort(initial[selection][:23], kind="stable")
    expected_keys, expected_values = initial.copy(), payload.copy()
    expected_keys[selection] = initial[selection][expected]
    expected_values[selection] = payload[selection][expected]
    np.testing.assert_array_equal(keys.to_numpy(), expected_keys)
    np.testing.assert_array_equal(values.to_numpy(), expected_values)
    ti.reset()
    with pytest.raises(ti.TaichiRuntimeError, match="another runtime"):
        plan.run()
    plan.close()


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_prefix_sort_rejects_retired_field():
    builder = ti.FieldsBuilder()
    keys = ti.field(ti.i32)
    builder.dense(ti.i, 17).place(keys)
    tree = builder.finalize()
    count = ti.ndarray(ti.i32, 1)
    count.from_numpy(np.array([17], dtype=np.int32))
    plan = ti.algorithms.prepare_sort(keys, count=count)
    plan.run()
    ti.sync()
    tree.destroy()
    with pytest.raises(
        RuntimeError, match="retired|destroyed|generation|stale|inactive"
    ):
        plan.run()
    plan.close()
