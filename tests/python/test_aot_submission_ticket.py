"""AOT device completions share backend ownership without freezing results."""

import gc
import json
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import taichi_forge as ti
from taichi_forge_runtime.aot import ExecutionPlan, Runtime, SubmissionTicket
from tests import test_utils


def _archive(path):
    @ti.kernel
    def advance(
        state: ti.types.ndarray(ti.f32, ndim=1), x: ti.types.ndarray(ti.f32, ndim=1)
    ):
        for i in x:
            state[i] += 1
            x[i] = state[i] + i

    @ti.kernel
    def square(
        x: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True),
        y: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True),
    ):
        for i in x:
            y[i] = x[i] * x[i]

    module = ti.aot.Module()
    module.add_kernel(advance)
    module.add_kernel(square)
    module.add_kernel(square.grad, name="reverse")
    module.add_execution_plan(
        "step",
        ExecutionPlan(
            [
                dict(kind="kernel", entry="advance", arguments=["state", "x"]),
                dict(kind="fill32", destination="dx", value=0),
                dict(kind="fill32", destination="dy", value=0x3F800000),
                dict(kind="kernel", entry="square", arguments=["x", "y"]),
                dict(kind="kernel", entry="reverse", arguments=["x", "y"]),
            ],
            resources=[
                dict(
                    kind="ndarray",
                    name="state",
                    dtype="f32",
                    shape=[257],
                    zero_initialize=True,
                )
            ],
            gradients=dict(x="dx", y="dy"),
        ),
    )
    module.archive(str(path))


def _frame(runtime, module):
    arrays = {name: runtime.ndarray("f32", 257) for name in ("x", "y", "dx", "dy")}
    return module.prepare_execution("step", arrays), arrays


def _read(array):
    return list(struct.unpack("<257f", array.read()))


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_aot_submission_ticket_primal_gradients_and_retirement(tmp_path):
    artifact = tmp_path / "tickets.tcm"
    _archive(artifact)
    arch = str(ti.lang.impl.current_cfg().arch).split(".")[-1]
    if arch in ("x64", "arm64"):
        arch = "cpu"
    measurements = []
    for mode, factory in (
        ("standalone", lambda: Runtime(arch)),
        ("borrowed", ti.aot.borrow_runtime),
    ):
        runtime = factory()
        with runtime:
            module = runtime.load_module(artifact)
            frames = [_frame(runtime, module) for _ in range(2)]
            tickets = []
            for _ in range(3):
                for call, _ in frames:
                    tickets.append(call.submit())
            assert all(isinstance(ticket, SubmissionTicket) for ticket in tickets)
            assert isinstance(tickets[0].done(), bool)
            # A waiter must make progress while another host thread owns the
            # runtime's submission scope, including a borrowed Program guard.
            held, release = threading.Event(), threading.Event()

            def hold_submission_scope():
                with runtime._lock:
                    held.set()
                    release.wait()

            with ThreadPoolExecutor(max_workers=2) as workers:
                blocker = workers.submit(hold_submission_scope)
                try:
                    assert held.wait(2)
                    workers.submit(tickets[-1].wait).result(timeout=2)
                finally:
                    release.set()
                blocker.result()
            for ticket in reversed(tickets):
                ticket.wait()
                assert ticket.done()
            for call, arrays in frames:
                expected = [i + 3 for i in range(257)]
                assert _read(arrays["x"]) == expected
                assert _read(arrays["y"]) == [x * x for x in expected]
                assert _read(arrays["dx"]) == [2 * x for x in expected]
            stale = tickets[0]._handle
            for ticket in tickets:
                ticket.close()
                ticket.close()
                with pytest.raises(RuntimeError, match="closed"):
                    ticket.done()
            with pytest.raises(RuntimeError, match="closed or unknown"):
                runtime._api.call(runtime._api.ti_query_submission_ext, stale)
            with pytest.raises(RuntimeError):
                runtime._api.call(runtime._api.ti_query_submission_ext, None)
            call, arrays = frames[0]
            # Dropping a ticket must not release resources of queued work.
            call.submit().close()
            orphan = call.submit()
            del orphan
            gc.collect()
            completed = call.submit()
            completed.wait()
            assert _read(arrays["x"])[0] == 6
            # A completed ticket does not snapshot/freeze caller-owned output.
            call.run()
            assert completed.done()
            assert _read(arrays["x"])[0] == 7
            completed.close()
            samples = []
            for _ in range(3):
                began = time.perf_counter_ns()
                ticket = call.submit()
                submitted = time.perf_counter_ns()
                ticket.wait()
                samples.append(
                    dict(
                        host_us=(submitted - began) / 1e3,
                        complete_us=(time.perf_counter_ns() - began) / 1e3,
                    )
                )
                ticket.close()
            measurements.append(dict(mode=mode, samples=samples))
            retained = call.submit()
            module.close()
            with pytest.raises(RuntimeError):
                call.submit()
            runtime.close()
            # No runtime/device handle is needed to query the retired state.
            retained.wait()
            assert retained.done()
            retained.close()
            for frame, arrays in frames:
                frame.close()
                for array in arrays.values():
                    array.close()
    runtime = ti.aot.borrow_runtime()
    module = runtime.load_module(artifact)
    call, arrays = _frame(runtime, module)
    pending = call.submit()
    ti.reset()
    pending.wait()
    assert pending.done()
    pending.close()
    with pytest.raises(RuntimeError):
        call.submit()
    call.close()
    module.close()
    for array in arrays.values():
        array.close()
    runtime.close()
    (tmp_path / "measurements.json").write_text(
        json.dumps(dict(arch=arch, samples=measurements)), encoding="utf-8"
    )


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_aot_ticket_keeps_owner_fault_after_reset(tmp_path):
    artifact = tmp_path / "fault.tcm"
    _archive(artifact)
    runtime = ti.aot.borrow_runtime()
    module = runtime.load_module(artifact)
    call, arrays = _frame(runtime, module)
    pending = call.submit()
    # Quiesce the real device first without observing the pending completion.
    # The injected owner fault tests teardown bookkeeping, not a hardware fault.
    runtime.wait()
    ti.lang.impl.get_runtime().prog._debug_inject_runtime_fault(
        -1, "aot-ticket-test", "injected AOT owner fault"
    )
    ti.reset()
    for operation in (pending.done, pending.wait):
        with pytest.raises(RuntimeError, match="injected AOT owner fault"):
            operation()
    pending.close()
    call.close()
    module.close()
    for array in arrays.values():
        array.close()
    runtime.close()
