import gc
import threading

import numpy as np

import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils


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
