import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_mixed_wait_modes_reuse_events_and_preserve_values():
    values = ti.ndarray(ti.i32, 32)
    values.fill(0)

    @ti.kernel
    def advance(dst: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in dst:
            dst[i] += 1

    builder = ti.graph.GraphBuilder()
    builder.dispatch(advance, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "dst", ti.i32, ndim=1))
    graph = builder.compile()
    bound = graph.bind({"dst": values})
    for mode in ("default", "blocking"):
        graph.submit(bound, wait_mode=mode).wait()
    program = impl.get_runtime().prog
    before = program._debug_runtime_completion_stats()
    for _ in range(32):
        for mode in ("blocking", "default"):
            ticket = graph.submit(bound, wait_mode=mode)
            ticket.wait()
            assert ticket.done()
    after = program._debug_runtime_completion_stats()
    if impl.current_cfg().arch == ti.cuda:
        assert after["cuda_completion_events_created"] == before["cuda_completion_events_created"]
        assert after["cuda_completion_events_reused"] - before["cuda_completion_events_reused"] == 64
        assert after["cuda_completion_events_cached"] == 2
    np.testing.assert_array_equal(values.to_numpy(), np.full(32, 66, dtype=np.int32))
    graph.close()
    ti.reset()
    ticket.wait()
    assert ticket.done()


@test_utils.test(arch=ti.cpu)
def test_invalid_wait_mode_does_not_submit_work(monkeypatch):
    counter = ti.field(ti.i32, shape=())

    @ti.kernel
    def increment():
        counter[None] += 1

    builder = ti.graph.GraphBuilder()
    builder.dispatch(increment)
    graph = builder.compile()
    for mode in (None, True, "adaptive", "spin", []):
        with pytest.raises(ValueError, match="wait_mode"):
            graph.submit({}, wait_mode=mode)
    with monkeypatch.context() as patch:
        patch.setattr(impl, "current_cfg", lambda: SimpleNamespace(arch=ti.opengl))
        with pytest.raises(ValueError, match="requires CPU, CUDA or Vulkan"):
            graph.submit({}, wait_mode="blocking")
    assert counter[None] == 0


@test_utils.test(arch=ti.cuda)
def test_reset_drains_a_pending_blocking_wait():
    values = ti.ndarray(ti.f32, 4096)
    values.fill(0)

    @ti.kernel
    def delay(dst: ti.types.ndarray(dtype=ti.f32, ndim=1), rounds: ti.i32):
        for i in dst:
            x = dst[i]
            for _ in range(rounds):
                x = (x + 0.000001) * 0.99999
            dst[i] = x

    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        delay,
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "dst", ti.f32, ndim=1),
        ti.graph.Arg(ti.graph.ArgKind.SCALAR, "rounds", ti.i32),
    )
    graph = builder.compile()
    args = {"dst": values, "rounds": 100000000}
    graph.submit(args, wait_mode="blocking").wait()
    ticket = graph.submit(args, wait_mode="blocking")
    entered = threading.Event()
    errors = []

    def wait():
        entered.set()
        try:
            ticket.wait()
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=wait)
    thread.start()
    assert entered.wait(2)
    time.sleep(0.001)
    try:
        assert not ticket.done()
        ti.reset()
    finally:
        thread.join(3)
    assert not thread.is_alive()
    assert not errors
    assert ticket.done()
