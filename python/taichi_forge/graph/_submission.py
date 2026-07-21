"""Cooperative pacing for asynchronous runtime submissions.

The pacer controls admission before a Graph or solver invocation starts
enqueueing backend work. It cannot reorder commands that have already reached
a CUDA stream or Vulkan queue.
"""

import itertools
import operator
import threading
import time
from collections import deque
from dataclasses import dataclass

from taichi_forge.lang.exception import TaichiRuntimeError


_lane_ids = itertools.count(1)


@dataclass(frozen=True)
class _DefaultSubmissionLane:
    identity: int
    label: str


def _new_submission_lane(kind):
    identity = next(_lane_ids)
    return _DefaultSubmissionLane(identity, f"{kind}:{identity}")


def _require_count(name, value, *, minimum):
    if isinstance(value, bool):
        raise TaichiRuntimeError(f"{name} must be an integer >= {minimum}")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TaichiRuntimeError(f"{name} must be an integer >= {minimum}") from exc
    if value < minimum:
        raise TaichiRuntimeError(f"{name} must be an integer >= {minimum}")
    return value


def _normalize_lane(default_lane, lane):
    if lane is None:
        return default_lane
    if not isinstance(lane, str) or not lane.strip():
        raise TaichiRuntimeError("submission lane must be a non-empty string")
    return lane


def _lane_label(lane):
    if isinstance(lane, _DefaultSubmissionLane):
        return lane.label
    return lane


class _AdmissionRequest:
    __slots__ = ("enqueued_ns", "lane", "lease", "waited")

    def __init__(self, lane):
        self.enqueued_ns = time.perf_counter_ns()
        self.lane = lane
        self.lease = None
        self.waited = False


class _SubmissionLease:
    __slots__ = (
        "_completion",
        "_lane",
        "_pacer",
        "_released",
        "_sequence",
    )

    def __init__(self, pacer, lane, sequence):
        self._pacer = pacer
        self._lane = lane
        self._sequence = sequence
        self._completion = None
        self._released = False

    def _attach(self, completion):
        self._pacer._attach_completion(self, completion)

    def _cancel(self):
        self._pacer._release_lease(self, completed=False)

    def _completion_done(self, completion):
        try:
            ready = completion.done()
        except Exception as exc:
            self._pacer._completion_failed(self, exc)
            raise
        if ready:
            self._pacer._release_lease(self, completed=True)
        return ready

    def _completion_wait(self, completion):
        try:
            completion.wait()
        except Exception as exc:
            self._pacer._completion_failed(self, exc)
            raise
        self._pacer._release_lease(self, completed=True)


class SubmissionPacer:
    """Bounds and fairly admits cooperating asynchronous submissions.

    One pacer may be shared by independent Graphs and batched solver workspace
    clones in the same runtime. Admission is work-conserving and round-robin
    across lanes while preserving FIFO order within each lane. Host launch of
    one complete invocation is serialized, but up to ``max_in_flight`` already
    launched invocations may remain in flight on the backend.

    This is a cooperative boundary: submissions that do not use the same pacer
    are not controlled by it.
    """

    def __init__(
        self,
        max_in_flight,
        *,
        max_in_flight_per_lane=None,
        max_queued=64,
    ):
        self.max_in_flight = _require_count("max_in_flight", max_in_flight, minimum=1)
        if max_in_flight_per_lane is None:
            self.max_in_flight_per_lane = None
        else:
            self.max_in_flight_per_lane = _require_count(
                "max_in_flight_per_lane",
                max_in_flight_per_lane,
                minimum=1,
            )
            if self.max_in_flight_per_lane > self.max_in_flight:
                raise TaichiRuntimeError(
                    "max_in_flight_per_lane must not exceed max_in_flight"
                )
        self.max_queued = _require_count("max_queued", max_queued, minimum=0)

        self._condition = threading.Condition()
        self._runtime = None
        self._valid = True
        self._invalid_reason = None
        self._failure = None
        self._active = {}
        self._active_by_lane = {}
        self._waiters = {}
        self._last_granted_lane = None
        self._launching_sequence = None
        self._progress_waiter = False
        self._next_sequence = 1

        self._admission_calls = 0
        self._grants = 0
        self._immediate_grants = 0
        self._waited_grants = 0
        self._rejections = 0
        self._queue_full_rejections = 0
        self._saturation_rejections = 0
        self._completed = 0
        self._cancelled = 0
        self._backend_failures = 0
        self._total_admission_wait_ns = 0
        self._maximum_admission_wait_ns = 0
        self._queued_count = 0
        self._peak_queued = 0
        self._peak_in_flight = 0
        self._lane_counters = {}

    @staticmethod
    def _normalize_saturation_policy(value):
        if not isinstance(value, str):
            raise TaichiRuntimeError("on_saturation must be 'wait' or 'raise'")
        value = value.casefold()
        if value not in ("wait", "raise"):
            raise TaichiRuntimeError("on_saturation must be 'wait' or 'raise'")
        return value

    def _lane_counter_locked(self, lane):
        counter = self._lane_counters.get(lane)
        if counter is None:
            counter = {
                "admission_calls": 0,
                "grants": 0,
                "waited_grants": 0,
                "rejections": 0,
                "completed": 0,
                "cancelled": 0,
                "total_admission_wait_ns": 0,
                "maximum_admission_wait_ns": 0,
            }
            self._lane_counters[lane] = counter
        return counter

    def _check_usable_locked(self, runtime):
        if not self._valid:
            raise TaichiRuntimeError(self._invalid_reason)
        if self._failure is not None:
            raise TaichiRuntimeError(
                "SubmissionPacer stopped after a backend completion failure; "
                "reset the runtime and create a new pacer"
            ) from self._failure
        if self._runtime is None:
            self._runtime = runtime
            runtime.register_runtime_object(self)
        elif self._runtime is not runtime:
            raise TaichiRuntimeError(
                "SubmissionPacer cannot coordinate submissions from different "
                "runtime generations"
            )

    def _lane_has_capacity_locked(self, lane):
        if self.max_in_flight_per_lane is None:
            return True
        return self._active_by_lane.get(lane, 0) < self.max_in_flight_per_lane

    def _pick_next_lane_locked(self):
        lanes = tuple(self._waiters)
        if not lanes:
            return None
        start = 0
        if self._last_granted_lane in self._waiters:
            start = (lanes.index(self._last_granted_lane) + 1) % len(lanes)
        for offset in range(len(lanes)):
            lane = lanes[(start + offset) % len(lanes)]
            if self._lane_has_capacity_locked(lane):
                return lane
        return None

    def _grant_next_locked(self):
        if self._failure is not None or not self._valid:
            return
        if self._launching_sequence is not None:
            return
        if len(self._active) >= self.max_in_flight:
            return
        lane = self._pick_next_lane_locked()
        if lane is None:
            return
        queue = self._waiters[lane]
        request = queue.popleft()
        if not queue:
            self._waiters.pop(lane)
        self._queued_count -= 1

        sequence = self._next_sequence
        self._next_sequence += 1
        lease = _SubmissionLease(self, lane, sequence)
        self._active[sequence] = lease
        self._active_by_lane[lane] = self._active_by_lane.get(lane, 0) + 1
        self._launching_sequence = sequence
        self._last_granted_lane = lane
        request.lease = lease

        self._grants += 1
        lane_counter = self._lane_counter_locked(lane)
        lane_counter["grants"] += 1
        if request.waited:
            wait_ns = time.perf_counter_ns() - request.enqueued_ns
            self._waited_grants += 1
            lane_counter["waited_grants"] += 1
            self._total_admission_wait_ns += wait_ns
            self._maximum_admission_wait_ns = max(
                self._maximum_admission_wait_ns, wait_ns
            )
            lane_counter["total_admission_wait_ns"] += wait_ns
            lane_counter["maximum_admission_wait_ns"] = max(
                lane_counter["maximum_admission_wait_ns"], wait_ns
            )
        else:
            self._immediate_grants += 1
        self._peak_in_flight = max(self._peak_in_flight, len(self._active))
        self._condition.notify_all()

    def _remove_waiter_locked(self, request):
        queue = self._waiters.get(request.lane)
        if queue is None:
            return False
        try:
            queue.remove(request)
        except ValueError:
            return False
        if not queue:
            self._waiters.pop(request.lane)
        self._queued_count -= 1
        return True

    def _oldest_completion_locked(self, preferred_lane=None):
        if preferred_lane is not None:
            for lease in self._active.values():
                if (
                    lease._lane == preferred_lane
                    and lease._completion is not None
                    and not lease._released
                ):
                    return lease, lease._completion
        for lease in self._active.values():
            if lease._completion is not None and not lease._released:
                return lease, lease._completion
        return None, None

    def _preferred_progress_lane_locked(self, lane):
        if self.max_in_flight_per_lane is None:
            return None
        if self._active_by_lane.get(lane, 0) >= self.max_in_flight_per_lane:
            return lane
        return None

    def _poll_ready_completions(self):
        with self._condition:
            snapshot = tuple(
                (lease, lease._completion)
                for lease in self._active.values()
                if lease._completion is not None and not lease._released
            )
        for lease, completion in snapshot:
            lease._completion_done(completion)

    def _reserve(self, runtime, default_lane, *, lane, on_saturation):
        lane = _normalize_lane(default_lane, lane)
        on_saturation = self._normalize_saturation_policy(on_saturation)
        request = _AdmissionRequest(lane)

        with self._condition:
            self._check_usable_locked(runtime)
        self._poll_ready_completions()

        with self._condition:
            self._check_usable_locked(runtime)
            self._admission_calls += 1
            lane_counter = self._lane_counter_locked(lane)
            lane_counter["admission_calls"] += 1
            queue = self._waiters.get(lane)
            if queue is None:
                queue = deque()
                self._waiters[lane] = queue
            queue.append(request)
            self._queued_count += 1
            self._grant_next_locked()
            if request.lease is not None:
                return request.lease

            if on_saturation == "raise":
                self._remove_waiter_locked(request)
                self._rejections += 1
                self._saturation_rejections += 1
                lane_counter["rejections"] += 1
                raise TaichiRuntimeError(
                    "SubmissionPacer is saturated; no backend work was " "submitted"
                )
            if self._queued_count > self.max_queued:
                self._remove_waiter_locked(request)
                self._rejections += 1
                self._queue_full_rejections += 1
                lane_counter["rejections"] += 1
                raise TaichiRuntimeError(
                    "SubmissionPacer wait queue is full; no backend work was "
                    "submitted"
                )
            self._peak_queued = max(self._peak_queued, self._queued_count)
            request.waited = True

        try:
            while True:
                completion_to_wait = None
                lease_to_wait = None
                with self._condition:
                    if request.lease is not None:
                        if self._failure is not None or not self._valid:
                            self._finish_lease_locked(request.lease, completed=False)
                        else:
                            return request.lease
                    if self._failure is not None:
                        self._remove_waiter_locked(request)
                        raise self._failure
                    if not self._valid:
                        self._remove_waiter_locked(request)
                        raise TaichiRuntimeError(self._invalid_reason)
                    if not self._progress_waiter:
                        preferred_lane = self._preferred_progress_lane_locked(
                            request.lane
                        )
                        lease_to_wait, completion_to_wait = (
                            self._oldest_completion_locked(preferred_lane)
                        )
                        if completion_to_wait is not None:
                            self._progress_waiter = True
                    if completion_to_wait is None:
                        self._condition.wait()
                        continue

                try:
                    completion_to_wait.wait()
                except Exception as exc:
                    self._completion_failed(lease_to_wait, exc)
                else:
                    self._release_lease(lease_to_wait, completed=True)
                finally:
                    with self._condition:
                        self._progress_waiter = False
                        self._condition.notify_all()
        except BaseException:
            with self._condition:
                if request.lease is not None:
                    self._finish_lease_locked(request.lease, completed=False)
                else:
                    self._remove_waiter_locked(request)
                self._condition.notify_all()
            raise

    def _attach_completion(self, lease, completion):
        with self._condition:
            current = self._active.get(lease._sequence)
            if current is not lease or lease._released:
                raise TaichiRuntimeError(
                    "SubmissionPacer lease no longer owns its launch turn"
                )
            if self._launching_sequence != lease._sequence:
                raise TaichiRuntimeError(
                    "SubmissionPacer launch completion was attached out of " "order"
                )
            lease._completion = completion
            self._launching_sequence = None
            if completion.has_backend_work:
                self._grant_next_locked()
            else:
                self._finish_lease_locked(lease, completed=True)
            self._condition.notify_all()

    def _finish_lease_locked(self, lease, *, completed):
        current = self._active.get(lease._sequence)
        if current is not lease:
            return False
        self._active.pop(lease._sequence)
        lane = lease._lane
        lane_active = self._active_by_lane[lane] - 1
        if lane_active:
            self._active_by_lane[lane] = lane_active
        else:
            self._active_by_lane.pop(lane)
        if self._launching_sequence == lease._sequence:
            self._launching_sequence = None
        lease._released = True
        lane_counter = self._lane_counter_locked(lane)
        if completed:
            self._completed += 1
            lane_counter["completed"] += 1
        else:
            self._cancelled += 1
            lane_counter["cancelled"] += 1
        self._grant_next_locked()
        self._condition.notify_all()
        return True

    def _release_lease(self, lease, *, completed):
        with self._condition:
            self._finish_lease_locked(lease, completed=completed)

    def _completion_failed(self, lease, exc):
        with self._condition:
            if self._active.get(lease._sequence) is not lease:
                return
            self._backend_failures += 1
            if self._failure is None:
                self._failure = exc
            self._finish_lease_locked(lease, completed=False)
            self._condition.notify_all()

    def _invalidate_runtime(self):
        with self._condition:
            self._valid = False
            self._invalid_reason = (
                "SubmissionPacer cannot be used after ti.reset(); create a "
                "new pacer for the new runtime"
            )
            for lease in self._active.values():
                lease._released = True
            self._active.clear()
            self._active_by_lane.clear()
            self._launching_sequence = None
            self._runtime = None
            self._condition.notify_all()

    def statistics(self):
        """Returns bounded-admission, fairness, and per-lane telemetry."""
        self._poll_ready_completions()
        with self._condition:
            lanes = {}
            for lane, counters in self._lane_counters.items():
                snapshot = dict(counters)
                snapshot["in_flight"] = self._active_by_lane.get(lane, 0)
                snapshot["queued"] = len(self._waiters.get(lane, ()))
                lanes[_lane_label(lane)] = snapshot
            return {
                "schema_version": 1,
                "valid": self._valid,
                "faulted": self._failure is not None,
                "max_in_flight": self.max_in_flight,
                "max_in_flight_per_lane": self.max_in_flight_per_lane,
                "max_queued": self.max_queued,
                "in_flight": len(self._active),
                "launch_in_progress": self._launching_sequence is not None,
                "queued": self._queued_count,
                "peak_in_flight": self._peak_in_flight,
                "peak_queued": self._peak_queued,
                "admission_calls": self._admission_calls,
                "grants": self._grants,
                "immediate_grants": self._immediate_grants,
                "waited_grants": self._waited_grants,
                "rejections": self._rejections,
                "queue_full_rejections": self._queue_full_rejections,
                "saturation_rejections": self._saturation_rejections,
                "completed": self._completed,
                "cancelled": self._cancelled,
                "backend_failures": self._backend_failures,
                "total_admission_wait_ns": self._total_admission_wait_ns,
                "maximum_admission_wait_ns": (self._maximum_admission_wait_ns),
                "lanes": lanes,
            }


def _reserve_paced_submission(
    pacer,
    runtime,
    default_lane,
    *,
    lane,
    on_saturation,
):
    if pacer is None:
        if lane is not None or on_saturation != "wait":
            raise TaichiRuntimeError("lane/on_saturation require a SubmissionPacer")
        return None
    if not isinstance(pacer, SubmissionPacer):
        raise TypeError("pacer must be a taichi_forge.graph.SubmissionPacer")
    return pacer._reserve(
        runtime,
        default_lane,
        lane=lane,
        on_saturation=on_saturation,
    )


__all__ = ["SubmissionPacer"]
