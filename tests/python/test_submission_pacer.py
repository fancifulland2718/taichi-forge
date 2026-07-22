import threading
import time

import pytest

from taichi_forge.graph._submission import (
    SubmissionPacer,
    _new_submission_lane,
)
from taichi_forge.lang.exception import TaichiRuntimeError


class _FakeRuntime:
    def __init__(self):
        self.registered = []

    def register_runtime_object(self, value):
        self.registered.append(value)


class _FakeCompletion:
    def __init__(self, *, backend_work=True):
        self.has_backend_work = backend_work
        self._ready = threading.Event()
        if not backend_work:
            self._ready.set()

    def done(self):
        return self._ready.is_set()

    def wait(self):
        if not self._ready.wait(2.0):
            raise RuntimeError("fake completion timed out")

    def complete(self):
        self._ready.set()


class _FailingCompletion(_FakeCompletion):
    def done(self):
        raise RuntimeError("backend completion failed")

    def wait(self):
        raise RuntimeError("backend completion failed")


def _wait_until(predicate):
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.001)
    raise AssertionError("condition did not become true")


def test_submission_pacer_round_robins_lanes_after_bounded_backlog():
    runtime = _FakeRuntime()
    pacer = SubmissionPacer(2, max_queued=4)
    default_lane = _new_submission_lane("test")

    first_completion = _FakeCompletion()
    first = pacer._reserve(
        runtime,
        default_lane,
        lane="fast",
        on_saturation="wait",
    )
    first._attach(first_completion)
    second_completion = _FakeCompletion()
    second = pacer._reserve(
        runtime,
        default_lane,
        lane="fast",
        on_saturation="wait",
    )
    second._attach(second_completion)

    order = []
    leases = {}
    errors = []

    def reserve(name, lane):
        try:
            lease = pacer._reserve(
                runtime,
                default_lane,
                lane=lane,
                on_saturation="wait",
            )
            leases[name] = lease
            order.append(name)
        except BaseException as exc:
            errors.append(exc)

    fast_thread = threading.Thread(target=reserve, args=("fast", "fast"))
    fast_thread.start()
    _wait_until(lambda: pacer.statistics()["queued"] == 1)
    slow_thread = threading.Thread(target=reserve, args=("slow", "slow"))
    slow_thread.start()
    _wait_until(lambda: pacer.statistics()["queued"] == 2)

    first_completion.complete()
    _wait_until(lambda: order == ["slow"])
    slow_completion = _FakeCompletion()
    leases["slow"]._attach(slow_completion)

    second_completion.complete()
    fast_thread.join(2.0)
    slow_thread.join(2.0)
    assert not fast_thread.is_alive()
    assert not slow_thread.is_alive()
    assert not errors
    assert order == ["slow", "fast"]

    fast_completion = _FakeCompletion()
    leases["fast"]._attach(fast_completion)
    slow_completion.complete()
    leases["slow"]._completion_wait(slow_completion)
    fast_completion.complete()
    leases["fast"]._completion_wait(fast_completion)

    stats = pacer.statistics()
    assert stats["peak_in_flight"] == 2
    assert stats["peak_queued"] == 2
    assert stats["completed"] == 4
    assert stats["waited_grants"] == 2
    assert stats["total_admission_wait_ns"] > 0
    assert stats["maximum_admission_wait_ns"] > 0
    assert stats["lanes"]["fast"]["grants"] == 3
    assert stats["lanes"]["slow"]["grants"] == 1


def test_submission_pacer_lane_cap_and_nonblocking_backpressure():
    runtime = _FakeRuntime()
    pacer = SubmissionPacer(2, max_in_flight_per_lane=1, max_queued=0)
    default_lane = _new_submission_lane("test")
    fast_completion = _FakeCompletion()
    fast = pacer._reserve(
        runtime,
        default_lane,
        lane="fast",
        on_saturation="wait",
    )
    fast._attach(fast_completion)
    slow_completion = _FakeCompletion()
    slow = pacer._reserve(
        runtime,
        default_lane,
        lane="slow",
        on_saturation="wait",
    )
    slow._attach(slow_completion)

    with pytest.raises(TaichiRuntimeError, match="saturated"):
        pacer._reserve(
            runtime,
            default_lane,
            lane="fast",
            on_saturation="raise",
        )
    with pytest.raises(TaichiRuntimeError, match="queue is full"):
        pacer._reserve(
            runtime,
            default_lane,
            lane="fast",
            on_saturation="wait",
        )

    fast_completion.complete()
    fast._completion_wait(fast_completion)
    slow_completion.complete()
    slow._completion_wait(slow_completion)
    stats = pacer.statistics()
    assert stats["saturation_rejections"] == 1
    assert stats["queue_full_rejections"] == 1
    assert stats["peak_queued"] == 0
    assert stats["lanes"]["fast"]["rejections"] == 2


def test_submission_pacer_serializes_complete_host_launch_turns():
    runtime = _FakeRuntime()
    pacer = SubmissionPacer(2)
    default_lane = _new_submission_lane("test")
    first = pacer._reserve(
        runtime,
        default_lane,
        lane="first",
        on_saturation="wait",
    )
    assert pacer.statistics()["launch_in_progress"]

    with pytest.raises(TaichiRuntimeError, match="saturated"):
        pacer._reserve(
            runtime,
            default_lane,
            lane="second",
            on_saturation="raise",
        )

    first._attach(_FakeCompletion(backend_work=False))
    stats = pacer.statistics()
    assert not stats["launch_in_progress"]
    assert stats["in_flight"] == 0
    assert stats["completed"] == 1


def test_submission_pacer_avoids_cross_lane_completion_head_of_line_blocking():
    runtime = _FakeRuntime()
    pacer = SubmissionPacer(2, max_in_flight_per_lane=1)
    default_lane = _new_submission_lane("test")

    slow_completion = _FakeCompletion()
    slow = pacer._reserve(
        runtime,
        default_lane,
        lane="slow",
        on_saturation="wait",
    )
    slow._attach(slow_completion)
    fast_completion = _FakeCompletion()
    fast = pacer._reserve(
        runtime,
        default_lane,
        lane="fast",
        on_saturation="wait",
    )
    fast._attach(fast_completion)

    granted = []

    def reserve_fast():
        granted.append(
            pacer._reserve(
                runtime,
                default_lane,
                lane="fast",
                on_saturation="wait",
            )
        )

    waiter = threading.Thread(target=reserve_fast)
    waiter.start()
    _wait_until(lambda: pacer.statistics()["queued"] == 1)
    fast_completion.complete()
    waiter.join(2.0)

    assert not waiter.is_alive()
    assert len(granted) == 1
    granted[0]._attach(_FakeCompletion(backend_work=False))
    slow_completion.complete()
    slow._completion_wait(slow_completion)
    assert pacer.statistics()["completed"] == 3


def test_submission_pacer_avoids_global_completion_head_of_line_blocking():
    runtime = _FakeRuntime()
    pacer = SubmissionPacer(2)
    default_lane = _new_submission_lane("test")

    slow_completion = _FakeCompletion()
    slow = pacer._reserve(
        runtime,
        default_lane,
        lane="slow",
        on_saturation="wait",
    )
    slow._attach(slow_completion)
    fast_completion = _FakeCompletion()
    fast = pacer._reserve(
        runtime,
        default_lane,
        lane="fast",
        on_saturation="wait",
    )
    fast._attach(fast_completion)

    granted = []
    errors = []

    def reserve_next():
        try:
            granted.append(
                pacer._reserve(
                    runtime,
                    default_lane,
                    lane="next",
                    on_saturation="wait",
                )
            )
        except BaseException as exc:
            errors.append(exc)

    waiter = threading.Thread(target=reserve_next)
    waiter.start()
    _wait_until(lambda: pacer.statistics()["queued"] == 1)
    fast_completion.complete()
    waiter.join(2.0)

    assert not waiter.is_alive()
    assert not errors
    assert len(granted) == 1
    granted[0]._attach(_FakeCompletion(backend_work=False))
    slow_completion.complete()
    slow._completion_wait(slow_completion)
    assert pacer.statistics()["completed"] == 3


def test_submission_pacer_fails_closed_after_backend_completion_failure():
    runtime = _FakeRuntime()
    pacer = SubmissionPacer(1)
    default_lane = _new_submission_lane("test")
    lease = pacer._reserve(
        runtime,
        default_lane,
        lane="faulting",
        on_saturation="wait",
    )
    lease._attach(_FailingCompletion())

    with pytest.raises(RuntimeError, match="backend completion failed"):
        pacer._reserve(
            runtime,
            default_lane,
            lane="waiting",
            on_saturation="wait",
        )
    stats = pacer.statistics()
    assert stats["faulted"]
    assert stats["backend_failures"] == 1
    assert stats["cancelled"] == 1
    with pytest.raises(TaichiRuntimeError, match="stopped"):
        pacer._reserve(
            runtime,
            default_lane,
            lane="new",
            on_saturation="raise",
        )


def test_submission_pacer_reset_unblocks_waiters_and_prevents_reuse():
    runtime = _FakeRuntime()
    pacer = SubmissionPacer(1)
    default_lane = _new_submission_lane("test")
    pacer._reserve(
        runtime,
        default_lane,
        lane="active",
        on_saturation="wait",
    )
    # Keep the first host launch turn reserved but not attached. Runtime reset
    # completes real backend work before invalidating registered Python
    # objects; this setup isolates the condition-variable wakeup contract.

    errors = []

    def wait_for_admission():
        try:
            pacer._reserve(
                runtime,
                default_lane,
                lane="waiting",
                on_saturation="wait",
            )
        except BaseException as exc:
            errors.append(exc)

    waiter = threading.Thread(target=wait_for_admission)
    waiter.start()
    _wait_until(lambda: pacer.statistics()["queued"] == 1)
    pacer._invalidate_runtime()
    waiter.join(2.0)

    assert not waiter.is_alive()
    assert len(errors) == 1
    assert "after ti.reset" in str(errors[0])
    with pytest.raises(TaichiRuntimeError, match="after ti.reset"):
        pacer._reserve(
            runtime,
            default_lane,
            lane="new",
            on_saturation="raise",
        )


def test_submission_pacer_validates_capacity_lane_and_runtime_generation():
    with pytest.raises(TaichiRuntimeError, match="max_in_flight"):
        SubmissionPacer(0)
    with pytest.raises(TaichiRuntimeError, match="must not exceed"):
        SubmissionPacer(2, max_in_flight_per_lane=3)

    runtime = _FakeRuntime()
    other_runtime = _FakeRuntime()
    pacer = SubmissionPacer(1)
    default_lane = _new_submission_lane("test")
    lease = pacer._reserve(
        runtime,
        default_lane,
        lane=None,
        on_saturation="wait",
    )
    lease._attach(_FakeCompletion(backend_work=False))

    with pytest.raises(TaichiRuntimeError, match="runtime generations"):
        pacer._reserve(
            other_runtime,
            default_lane,
            lane="other",
            on_saturation="raise",
        )
    with pytest.raises(TaichiRuntimeError, match="non-empty string"):
        pacer._reserve(
            runtime,
            default_lane,
            lane="",
            on_saturation="raise",
        )
