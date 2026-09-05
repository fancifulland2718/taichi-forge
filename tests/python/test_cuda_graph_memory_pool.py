import gc
import threading

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge._lib import core as ti_core
from taichi_forge.lang import impl
from taichi_forge.lang._ndarray import ScalarNdarray
from tests import test_utils


def _pool(retained_bytes=32 << 20):
    impl.get_runtime().materialize()
    if not ti_core._CudaGraphMemoryPool.available():
        pytest.skip("CUDA driver/device does not support Graph-owned memory pools")
    return ti_core._CudaGraphMemoryPool(impl.get_runtime().prog, retained_bytes)


def _storage(pool, size):
    return ScalarNdarray._graph_pool_storage(ti.i32, (size,), pool)


def _graph():
    @ti.kernel
    def stage(value: ti.i32, scratch: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in scratch:
            scratch[i] = value + i * 3

    @ti.kernel
    def finish(
        scratch: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in output:
            output[i] = scratch[i] + 7

    value = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "value", ti.i32)
    scratch, output = (ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.i32, ndim=1) for name in ("scratch", "output"))
    builder = ti.graph.GraphBuilder()
    builder.dispatch(stage, value, scratch)
    builder.dispatch(finish, scratch, output)
    return builder.compile()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_pool_isolation_reuse_and_nonblocking_trim_preserve_live_storage():
    first, second = _pool(), _pool(0)
    allocations_before = ti_core.query_int64("cuda_async_allocation_calls")
    a, b = _storage(first, 65537), _storage(second, 4099)
    assert ti_core.query_int64("cuda_async_allocation_calls") == allocations_before + 2
    np.testing.assert_array_equal(a.to_numpy(), np.zeros(65537, np.int32))
    a.fill(123)
    b.fill(47)
    ti.sync()
    first_before, second_before = first.snapshot(), second.snapshot()
    assert first_before["release_threshold_bytes"] == 32 << 20
    assert second_before["release_threshold_bytes"] == 0
    assert first_before["used_current_bytes"] == 65537 * 4
    assert second_before["used_current_bytes"] == 4099 * 4
    first.trim()
    np.testing.assert_array_equal(a.to_numpy(), np.full(65537, 123, np.int32))
    del a
    gc.collect()
    ti.sync()
    assert first.snapshot()["used_current_bytes"] == 0
    for size in (1025, 8193, 257, 65537):
        a = _storage(first, size)
        np.testing.assert_array_equal(a.to_numpy(), np.zeros(size, np.int32))
        del a
        gc.collect()
        ti.sync()
    first.trim()
    assert first.snapshot()["used_current_bytes"] == 0
    assert first.snapshot()["reserved_current_bytes"] <= first_before["reserved_current_bytes"]
    assert second.snapshot()["used_current_bytes"] == second_before["used_current_bytes"]
    np.testing.assert_array_equal(b.to_numpy(), np.full(4099, 47, np.int32))
    first.close()
    second.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_pool_close_and_array_retirement_preserve_prepared_graph_allocations():
    pool = _pool()
    if not ti_core._CudaGraphBindingExecutor.available():
        pytest.skip("Graph executable update unavailable")
    size = 8193
    scratch, output = _storage(pool, size), ti.ndarray(ti.i32, size)
    graph = _graph()
    executor = ti_core._CudaGraphBindingExecutor(graph._compiled_graph, impl.current_cfg(), impl.get_runtime().prog)
    frames = [executor.prepare(dict(value=value, scratch=scratch.arr, output=output.arr)) for value in range(7)]
    for frame in frames:
        executor.run(frame)
    # Close the factory with queued work and allocations still pinned by
    # prepared graphs. The pool must outlive these allocation owners/frees.
    pool.close()
    pool.close()
    del scratch
    gc.collect()
    assert pool.snapshot() == {"closed": 1}
    with pytest.raises(RuntimeError, match="closed"):
        _storage(pool, size)
    for index in range(37):
        executor.run(frames[index % len(frames)])
    np.testing.assert_array_equal(output.to_numpy(), np.arange(size, dtype=np.int32) * 3 + 8)
    executor.close()
    frames.clear()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_reset_retires_pool_factory_before_native_device_and_python_owners():
    pool = _pool()
    scratch = _storage(pool, 4097)
    output = ti.ndarray(ti.i32, 4097)
    graph = _graph()
    binding = graph.bind(dict(value=91, scratch=scratch, output=output))
    graph.run(binding)
    ti.reset()
    assert pool.snapshot() == {"closed": 1}
    with pytest.raises(RuntimeError, match="closed"):
        pool.create_ndarray(ti.i32, [8])
    pool.close()
    del binding, graph, scratch, output, pool
    gc.collect()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_rejected_shapes_and_concurrent_factory_close_do_not_leak_or_deadlock():
    pool = _pool()
    for shape in ([0], [-1], [2147483647, 2147483647, 8]):
        with pytest.raises(RuntimeError, match="shape"):
            pool.create_ndarray(ti.i32, shape)
    assert pool.snapshot()["used_current_bytes"] == 0
    errors = []
    published = threading.Event()
    closed = threading.Event()

    def allocate():
        try:
            for index in range(24):
                try:
                    value = pool.create_ndarray(ti.i32, [8193])
                except RuntimeError as error:
                    assert "closed" in str(error)
                    return
                if index == 0:
                    published.set()
                    assert closed.wait(timeout=10)
                impl.get_runtime().prog.delete_ndarray(value)
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=allocate, daemon=True)
    worker.start()
    assert published.wait(timeout=10)
    pool.close()
    closed.set()
    worker.join(timeout=30)
    assert not worker.is_alive()
    assert not errors
    assert pool.snapshot() == {"closed": 1}
    ti.sync()
