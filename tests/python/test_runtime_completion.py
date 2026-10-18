import gc
import threading
import time

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils


@pytest.mark.parametrize("wait_mode", ["default", "blocking"])
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_wait_query_and_paced_submit_share_a_pending_ticket(wait_mode):
    value = ti.ndarray(ti.i32, 32)
    scratch = ti.ndarray(ti.f32, 4096)
    scratch.fill(0)

    @ti.kernel
    def delay(a: ti.types.ndarray(dtype=ti.f32, ndim=1), rounds: ti.i32):
        for i in a:
            x = a[i]
            for j in range(rounds):
                x = (x + 0.000001) * 0.99999
            a[i] = x

    @ti.kernel
    def advance(dst: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in dst:
            dst[i] += 1

    builder = ti.graph.GraphBuilder()
    builder.dispatch(delay, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "scratch", ti.f32, ndim=1),
                     ti.graph.Arg(ti.graph.ArgKind.SCALAR, "rounds", ti.i32))
    builder.dispatch(advance, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "dst", ti.i32, ndim=1))
    graph = builder.compile()
    bound = graph.bind({"dst": value, "scratch": scratch, "rounds": 100000000})
    graph.submit(bound).wait()
    value.fill(0)
    ti.sync()
    pacer = ti.graph.SubmissionPacer(2, max_in_flight_per_lane=1, max_queued=2)
    first = graph.submit(bound, pacer=pacer, lane="first", wait_mode=wait_mode)
    started = threading.Event()
    failures = []

    def wait():
        started.set()
        try:
            first.wait()
        except BaseException as error:
            failures.append(error)

    waiter = threading.Thread(target=wait)
    waiter.start()
    assert started.wait(2)
    time.sleep(0.001)
    try:
        assert not first.done()
        second = graph.submit(bound, pacer=pacer, lane="second", wait_mode=wait_mode)
        assert not first.done(), "submission waited for the first ticket"
        second.wait()
    finally:
        waiter.join(3)
        graph.close()
    assert not waiter.is_alive()
    assert not failures
    np.testing.assert_array_equal(value.to_numpy(), np.full(32, 2, dtype=np.int32))
    ti.reset()
    assert first.done()


@pytest.mark.parametrize("wait_mode", ["default", "blocking"])
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_graph_ticket_reuses_batch_fence_and_orders_following_work(wait_mode):
    value = ti.ndarray(ti.i32, shape=4096)

    @ti.kernel
    def advance(dst: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in dst:
            dst[i] += 1

    builder = ti.graph.GraphBuilder()
    builder.dispatch(advance, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "dst", ti.i32, ndim=1))
    graph = builder.compile()
    bindings = graph.bind(dict(dst=value))
    value.fill(0)
    graph.submit(bindings).wait()
    program = impl.get_runtime().prog
    before = dict(program._debug_vulkan_queue_submission_stats())
    first = graph.submit(bindings, wait_mode=wait_mode)
    after = dict(program._debug_vulkan_queue_submission_stats())
    # One batched dispatch, no separate empty completion command/submit.
    assert after["queue_submit_calls"] - before["queue_submit_calls"] == 1
    assert after["submitted_command_buffers"] - before["submitted_command_buffers"] == 1
    # The next pending kernel must be flushed, not covered by the old fence.
    advance(value)
    second = program._record_runtime_completion()
    second.wait()
    first.wait()
    np.testing.assert_array_equal(value.to_numpy(), np.full(4096, 3, dtype=np.int32))
    graph.close()
    ti.reset()
    first.wait()
    assert first.done()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_internal_runtime_completion_contract_and_resource_retirement():
    prog = impl.get_runtime().prog
    empty = prog._record_runtime_completion()
    assert empty.done()
    assert not empty.has_backend_work
    assert empty.sequence == 0
    assert empty.program_domain != 0

    baseline = prog._debug_ndarray_resource_stats()
    arr = ti.ndarray(ti.i32, shape=4096)

    @ti.kernel
    def fill(value: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in value:
            value[i] = i * 3 + 1

    fill(arr)
    launched = prog._debug_ndarray_resource_stats()
    if impl.current_cfg().arch != ti.cpu:
        assert launched["inflight"] == baseline["inflight"] + 1

    ticket = prog._record_runtime_completion()
    assert ticket.program_domain == empty.program_domain
    assert ticket.sequence == 1
    assert ticket.backend in ("x64", "cuda", "vulkan")

    # Retirement may happen in record_runtime_completion() itself when a tiny
    # GPU dispatch has already completed. Both paths must converge without a
    # global ti.sync() and without keeping the Python view alive.
    del arr
    gc.collect()
    ticket.wait()
    assert ticket.done()

    no_new_work = prog._record_runtime_completion()
    assert no_new_work.done()
    assert not no_new_work.has_backend_work
    assert no_new_work.sequence == ticket.sequence

    completed = prog._debug_ndarray_resource_stats()
    for key in ("live", "retiring", "leases", "views", "inflight"):
        assert completed[key] == baseline[key], (key, baseline, completed)
    assert completed["released_total"] == baseline["released_total"] + 1
    completion_stats = prog._debug_runtime_completion_stats()
    assert completion_stats["active"] == 0
    assert completion_stats["retained_ndarrays"] == 0
    assert (
        completion_stats["submission_epoch"]
        == completion_stats["completed_submission_epoch"]
    )


@test_utils.test(arch=ti.cuda)
def test_cuda_runtime_completion_reuses_waited_events():
    value = ti.ndarray(ti.i32, shape=1)

    @ti.kernel
    def advance(dst: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        dst[0] += 1

    prog = impl.get_runtime().prog
    advance(value)
    ti.sync()
    baseline = prog._debug_runtime_completion_stats()

    iterations = 32
    for _ in range(iterations):
        advance(value)
        ticket = prog._record_runtime_completion()
        ticket.wait()

    # Collect the last completed state without publishing another event.
    assert prog._record_runtime_completion().done()
    final = prog._debug_runtime_completion_stats()
    created = (
        final["cuda_completion_events_created"]
        - baseline["cuda_completion_events_created"]
    )
    reused = (
        final["cuda_completion_events_reused"]
        - baseline["cuda_completion_events_reused"]
    )
    returned = (
        final["cuda_completion_events_returned"]
        - baseline["cuda_completion_events_returned"]
    )

    assert created == 1
    assert reused == iterations - 1
    assert returned == iterations
    assert final["cuda_completion_events_cached"] == 1
    assert final["cuda_completion_events_abandoned"] == 0
    assert final["active"] == 0
    assert value.to_numpy()[0] == iterations + 1


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_runtime_completion_survives_program_reset_as_completed_token():
    value = ti.field(ti.i32, shape=())

    @ti.kernel
    def update():
        for i in range(1 << 16):
            ti.atomic_add(value[None], i & 1)

    update()
    ticket = impl.get_runtime().prog._record_runtime_completion()
    domain = ticket.program_domain
    sequence = ticket.sequence
    ti.reset()

    assert ticket.program_domain == domain
    assert ticket.sequence == sequence
    ticket.wait()
    assert ticket.done()
    assert ticket.first_error == ""


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_runtime_completion_linearizes_concurrent_kernel_submission():
    value = ti.ndarray(ti.i32, shape=2048)

    @ti.kernel
    def advance(dst: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in dst:
            dst[i] += 1

    advance(value)
    ti.sync()

    start = threading.Event()
    failures = []

    def producer():
        try:
            start.wait()
            for _ in range(32):
                advance(value)
        except BaseException as exc:  # preserve the worker root cause
            failures.append(exc)

    worker = threading.Thread(target=producer)
    worker.start()
    start.set()
    tickets = []
    for _ in range(16):
        tickets.append(impl.get_runtime().prog._record_runtime_completion())
    worker.join()
    assert not failures

    final_ticket = impl.get_runtime().prog._record_runtime_completion()
    tickets.append(final_ticket)
    for ticket in tickets:
        ticket.wait()
    np.testing.assert_array_equal(
        value.to_numpy(), np.full(2048, 33, dtype=np.int32)
    )
    stats = impl.get_runtime().prog._debug_runtime_completion_stats()
    assert stats["active"] == 0
    assert stats["failed"] == 0
