from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge._lib import core as ti_core
from taichi_forge.lang import impl
from tests import test_utils


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_device_extent_value_identity_and_explicit_observation():
    with pytest.raises(TypeError, match="capacity.*integer"):
        ti.DeviceExtent(True)
    with pytest.raises(ValueError, match=r"\[0, 2\^31-1\]"):
        ti.DeviceExtent(-1)

    extent = ti.DeviceExtent(17)
    assert extent.capacity == 17
    assert extent.state is extent.count
    assert extent.state.dtype == ti.i32
    assert extent.state.shape == (2,)
    assert extent.binding.capacity == 17
    assert extent.binding.generation == extent.generation
    assert extent.binding.allocation_identity == extent.state._runtime_allocation_identity
    assert extent.runtime_arguments("extent") == {"extent": extent.state}
    with pytest.raises(ValueError, match="non-empty"):
        extent.runtime_arguments("")
    with pytest.raises(FrozenInstanceError):
        extent.binding.capacity = 18

    extent.set(9)
    snapshot = extent.snapshot()
    assert snapshot.raw_count == snapshot.count == 9
    assert snapshot.capacity == 17
    assert not snapshot.overflow
    assert extent.check() == 9

    extent.set(23)
    snapshot = extent.snapshot()
    assert snapshot.raw_count == snapshot.count == 17
    assert snapshot.overflow
    with pytest.raises(RuntimeError, match="overflow.*capacity=17"):
        extent.check()

    extent.reset()
    assert extent.snapshot().count == 0
    assert not extent.snapshot().overflow


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_device_extent_device_publish_zero_clamp_and_graph_sharing():
    capacity = 257
    extent = ti.DeviceExtent(capacity)
    output = ti.ndarray(ti.i32, shape=capacity)

    @ti.kernel
    def produce(
        requested: ti.i32,
        state: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        ti.device_extent_publish(state, capacity, requested)

    @ti.kernel
    def consume(
        state: ti.types.ndarray(dtype=ti.i32, ndim=1),
        out: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in range(capacity):
            if i < ti.device_extent_count(state):
                out[i] += i + 1

    state_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "extent", ti.i32, ndim=1
    )
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1)
    requested_arg = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "requested", ti.i32)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(produce, requested_arg, state_arg)
    builder.dispatch(consume, state_arg, output_arg)
    graph = builder.compile()

    for requested, expected_count, overflow in (
        (0, 0, False),
        (65, 65, False),
        (capacity + 11, capacity, True),
        (-3, 0, True),
    ):
        output.fill(0)
        args = {"requested": requested, "out": output}
        args.update(extent.runtime_arguments("extent"))
        graph.run(args)
        ti.sync()
        expected = np.zeros(capacity, dtype=np.int32)
        expected[:expected_count] = np.arange(1, expected_count + 1, dtype=np.int32)
        np.testing.assert_array_equal(output.to_numpy(), expected)
        snapshot = extent.snapshot()
        assert snapshot.count == expected_count
        assert snapshot.overflow == overflow


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_device_extent_normalizes_existing_count_producer_without_readback():
    capacity = 127
    extent = ti.DeviceExtent(capacity)

    @ti.kernel
    def write_raw(state: ti.types.ndarray(dtype=ti.i32, ndim=1), value: ti.i32):
        state[ti.DeviceExtent.count_index] = value

    program = impl.get_runtime().prog
    extent.reset()
    write_raw(extent.count, capacity + 9)
    before = program._runtime_statistics_snapshot()
    extent.normalize()
    after_enqueue = program._runtime_statistics_snapshot()

    # Normalization is device work, not a transfer/readback or implicit sync.
    assert after_enqueue["transfer"] == before["transfer"]
    assert after_enqueue["synchronization"] == before["synchronization"]
    snapshot = extent.snapshot()
    assert snapshot.count == capacity
    assert snapshot.overflow


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_device_extent_count_storage_is_primitive_compatible():
    capacity = 64
    values = ti.ndarray(ti.i32, shape=capacity)
    flags = ti.ndarray(ti.i32, shape=capacity)
    output = ti.ndarray(ti.i32, shape=capacity)
    extent = ti.DeviceExtent(capacity)
    values.from_numpy(np.arange(capacity, dtype=np.int32))
    flags.from_numpy((np.arange(capacity) % 3 == 1).astype(np.int32))

    extent.reset()
    ti.algorithms.experimental_compact(values, flags, output, extent.count)
    extent.normalize()
    ti.sync()

    expected = np.arange(capacity, dtype=np.int32)[np.arange(capacity) % 3 == 1]
    assert extent.check() == len(expected)
    np.testing.assert_array_equal(output.to_numpy()[: len(expected)], expected)


@pytest.mark.run_in_serial
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_device_extent_count_churn_keeps_binding_and_memory_stable():
    capacity = 1024
    extent = ti.DeviceExtent(capacity)

    @ti.kernel
    def produce(state: ti.types.ndarray(dtype=ti.i32, ndim=1), value: ti.i32):
        ti.device_extent_publish(state, capacity, value)

    state_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "extent", ti.i32, ndim=1
    )
    value_arg = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "value", ti.i32)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(produce, state_arg, value_arg)
    graph = builder.compile()

    graph.run({"extent": extent.state, "value": 0})
    ti.sync()
    # Enable backend counters before taking the stable baseline when available.
    graph.execution_stats()
    graph.run({"extent": extent.state, "value": 1})
    ti.sync()

    binding = extent.binding
    graph_identity = graph._instance_debug_info
    memory_before = impl.get_runtime().prog._runtime_statistics_snapshot()["memory"]
    host_before = dict(ti_core.get_host_memory_pool_stats())
    device_before = dict(ti_core.get_device_memory_pool_stats())

    for i in range(1000):
        graph.run({"extent": extent.state, "value": i % (capacity + 1)})
    ti.sync()

    assert extent.binding == binding
    assert extent.state._runtime_allocation_identity == binding.allocation_identity
    assert graph._instance_debug_info == graph_identity
    assert impl.get_runtime().prog._runtime_statistics_snapshot()["memory"] == memory_before
    assert dict(ti_core.get_host_memory_pool_stats()) == host_before
    assert dict(ti_core.get_device_memory_pool_stats()) == device_before
    assert extent.check() == 999 % (capacity + 1)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_device_extent_rejects_stale_runtime_binding():
    extent = ti.DeviceExtent(8)
    ti.reset()
    ti.init(arch=ti.cpu, offline_cache=False)
    with pytest.raises(RuntimeError, match="stale"):
        _ = extent.state
