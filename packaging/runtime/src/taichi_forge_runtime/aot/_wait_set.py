"""Cancellable wait-any over frozen completion prefixes, without the DSL."""

from dataclasses import dataclass
import math
import threading

from . import _c_api as abi
from ._completion import SubmissionTicket
from ._domain import Completion


def _nanoseconds(value, name, *, optional=False, positive=False):
    if value is None and optional:
        return -1
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a duration in seconds")
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    if seconds >= (1 << 63) / 1_000_000_000:
        raise ValueError(f"{name} exceeds int64 nanoseconds")
    nanoseconds = math.ceil(seconds * 1_000_000_000)
    if not int(positive) <= nanoseconds <= (1 << 63) - 1:
        raise ValueError(
            f"{name} must fit {'positive ' if positive else ''}int64 nanoseconds"
        )
    return nanoseconds


@dataclass(frozen=True)
class CompletionWaitResult:
    """status is ready/timeout/cancelled; index is set only when ready."""

    status: str
    index: int | None = None


class CompletionSet:
    """Frozen exact prefixes; wait returns the first ready input index.

    Accepts JIT Graph tickets, AOT SubmissionTicket and independent Completion.
    Original host ticket handles may close after creation. Cancellation is
    terminal and never cancels GPU work. Polling uses the waiting caller, with
    no per-ticket worker or GPU callback; poll_interval is specified in seconds.
    """

    def __init__(self, completions, *, poll_interval=0.001):
        values = tuple(completions)
        if not values:
            raise ValueError("CompletionSet must not be empty")
        interval = _nanoseconds(poll_interval, "poll_interval", positive=True)
        self._lock = threading.RLock()
        self._handle = None
        self._hooks = {}
        self._api = getattr(values[0], "_api", None) or abi.Api()
        self._api.load_completion_sets()
        sources, loans = [], []
        try:
            for index, value in enumerate(values):
                if not isinstance(value, (Completion, SubmissionTicket)):
                    loan = getattr(value, "_loan_wait_completion", None)
                    if loan is None:
                        raise TypeError(
                            "CompletionSet requires Forge completion tickets"
                        )
                    self._hooks[index] = value.done
                    value = loan(self._api)
                    if not isinstance(value, SubmissionTicket):
                        raise TypeError(
                            "Completion loan did not return a submission ticket"
                        )
                    loans.append(value)
                if value._api.library._handle != self._api.library._handle:
                    raise ValueError(
                        "Completion belongs to another native runtime library"
                    )
                if isinstance(value, Completion):
                    sources.append(abi.CompletionSource(1, value._native_handle()))
                else:
                    with value._lock:
                        value._check()
                        sources.append(abi.CompletionSource(0, value._handle))
            native = (abi.CompletionSource * len(sources))(*sources)
            self._handle = self._api.call(
                self._api.ti_create_completion_set_ext,
                len(native),
                native,
                interval,
            )
        finally:
            for loan in loans:
                loan.close()

    def _native_handle(self):
        with self._lock:
            if self._handle is None:
                raise RuntimeError("CompletionSet is closed")
            return self._handle

    def wait(self, timeout=None):
        """Observe a ready prefix, timeout or cancellation; never consume output.

        None waits indefinitely; zero performs one nonblocking scan. Backend
        errors propagate. Readiness does not grant a workspace/result lease.
        """
        duration = _nanoseconds(timeout, "timeout", optional=True)
        result = abi.CompletionWaitResult()
        self._api.call(
            self._api.ti_wait_completion_set_ext,
            self._native_handle(),
            duration,
            abi.C.byref(result),
        )
        statuses = ("ready", "timeout", "cancelled")
        if result.status >= len(statuses):
            raise RuntimeError("Native completion set returned an invalid status")
        index = result.index if result.status == 0 else None
        if index is not None:
            hook = self._hooks.get(index)
            if hook is not None:
                hook()  # Preserve the JIT ticket's pacer/observation bookkeeping.
        return CompletionWaitResult(statuses[result.status], index)

    def cancel(self):
        with self._lock:
            self._api.call(
                self._api.ti_cancel_completion_set_ext, self._native_handle()
            )

    def close(self):
        with self._lock:
            handle, self._handle = self._handle, None
            if handle is not None:
                self._api.call(self._api.ti_release_completion_set_ext, handle)
            self._hooks = {}

    def __enter__(self):
        self._native_handle()
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
