"""Public runtime observability APIs for Taichi Forge."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Optional

from taichi_forge.lang import impl
from taichi_forge.lang.exception import TaichiRuntimeError


@dataclass(frozen=True)
class SubmissionStatistics:
    kernel_submissions: int
    graph_submissions: int
    graph_backend_submissions: int
    native_submissions: int
    failed_submissions: int


@dataclass(frozen=True)
class SynchronizationStatistics:
    program_syncs: int
    program_sync_wait_ns: int
    completion_polls: int
    completion_waits: int
    completion_wait_ns: int
    backend_waits: Optional[int]
    backend_wait_ns: Optional[int]
    backend_lock_samples: Optional[int]
    backend_lock_contentions: Optional[int]
    backend_lock_sampled_wait_ns: Optional[int]


@dataclass(frozen=True)
class MemoryStatistics:
    live_resources: int
    retiring_resources: int
    inflight_resources: int
    host_requested_live_bytes: Optional[int]
    host_raw_bytes: Optional[int]
    host_capacity_bytes: Optional[int]
    device_requested_live_bytes: Optional[int]
    device_raw_bytes: Optional[int]
    device_cached_bytes: Optional[int]
    cuda_mempool_reserved_bytes: Optional[int]
    cuda_mempool_used_bytes: Optional[int]


@dataclass(frozen=True)
class TransferStatistics:
    host_to_device_bytes: int
    device_to_host_bytes: int
    device_to_device_bytes: int
    cuda_vulkan_direct_bytes: int
    cuda_vulkan_fallback_bytes: int


@dataclass(frozen=True)
class GraphStatistics:
    captures: int
    recaptures: int
    replays: int
    ordinary_fallbacks: int
    replay_slot_saturation_fallbacks: int


@dataclass(frozen=True)
class DisplayStatistics:
    accepted_frames: int
    submitted_frames: int
    dropped_frames: int
    accepted_frame_bytes: int


@dataclass(frozen=True)
class RuntimeFault:
    backend: str
    backend_code: int
    sequence: int
    operation: str
    message: str


@dataclass(frozen=True)
class FaultStatistics:
    state: str
    first_fatal_faults: int
    rejected_submissions: int
    first_fault: Optional[RuntimeFault]


@dataclass(frozen=True)
class TraceStatistics:
    recorded_events: int
    dropped_events: int


@dataclass(frozen=True)
class RuntimeStatistics:
    schema_version: int
    backend: str
    program_domain: int
    submission: SubmissionStatistics
    synchronization: SynchronizationStatistics
    memory: MemoryStatistics
    transfer: TransferStatistics
    graph: GraphStatistics
    display: DisplayStatistics
    fault: FaultStatistics
    trace: TraceStatistics


@dataclass(frozen=True)
class RuntimeCapabilities:
    schema_version: int
    backend: str
    program_domain: Optional[int]
    statistics: bool
    statistics_schema_version: Optional[int]
    bounded_trace: bool
    trace_schema_version: Optional[int]
    chrome_trace_export: bool
    backend_wait_telemetry: bool
    backend_lock_telemetry: bool
    device_memory_telemetry: bool
    cuda_mempool_telemetry: bool


@dataclass(frozen=True)
class RuntimeTraceSession:
    program_domain: int
    session: int
    enabled: bool
    max_threads: int
    events_per_thread: int
    event_capacity: int
    allocated_bytes: int
    recorded_events: int
    dropped_events: int


def _require_program(operation: str):
    prog = impl.get_runtime().prog
    if prog is None:
        raise TaichiRuntimeError(f"ti.runtime.{operation}() requires ti.init()")
    return prog


def _require_program_method(prog, name: str, operation: str):
    method = getattr(prog, name, None)
    if method is None:
        raise TaichiRuntimeError(
            f"ti.runtime.{operation}() is unavailable in the installed native "
            "runtime; install matching taichi-forge and taichi-forge-runtime "
            "packages"
        )
    return method


def _runtime_fault(raw):
    if raw is None:
        return None
    return RuntimeFault(
        backend=raw["backend"],
        backend_code=raw["backend_code"],
        sequence=raw["sequence"],
        operation=raw["operation"],
        message=raw["message"],
    )


def _statistics_from_raw(raw) -> RuntimeStatistics:
    return RuntimeStatistics(
        schema_version=raw["schema_version"],
        backend=raw["backend"],
        program_domain=raw["program_domain"],
        submission=SubmissionStatistics(**raw["submission"]),
        synchronization=SynchronizationStatistics(**raw["synchronization"]),
        memory=MemoryStatistics(**raw["memory"]),
        transfer=TransferStatistics(**raw["transfer"]),
        graph=GraphStatistics(**raw["graph"]),
        display=DisplayStatistics(**raw["display"]),
        fault=FaultStatistics(
            state=raw["fault"]["state"],
            first_fatal_faults=raw["fault"]["first_fatal_faults"],
            rejected_submissions=raw["fault"]["rejected_submissions"],
            first_fault=_runtime_fault(raw["fault"]["first_fault"]),
        ),
        trace=TraceStatistics(**raw["trace"]),
    )


def _trace_session_from_raw(raw) -> RuntimeTraceSession:
    return RuntimeTraceSession(**raw)


def stats() -> RuntimeStatistics:
    """Return an immutable snapshot for the active Program generation."""
    prog = _require_program("stats")
    snapshot = _require_program_method(
        prog, "_runtime_statistics_snapshot", "stats"
    )()
    return _statistics_from_raw(snapshot)


def capabilities() -> RuntimeCapabilities:
    """Describe runtime observability supported by the active Program."""
    prog = _require_program("capabilities")
    snapshot_method = getattr(prog, "_runtime_statistics_snapshot", None)
    if snapshot_method is None:
        return RuntimeCapabilities(
            schema_version=1,
            backend=str(impl.current_cfg().arch),
            program_domain=None,
            statistics=False,
            statistics_schema_version=None,
            bounded_trace=False,
            trace_schema_version=None,
            chrome_trace_export=False,
            backend_wait_telemetry=False,
            backend_lock_telemetry=False,
            device_memory_telemetry=False,
            cuda_mempool_telemetry=False,
        )

    raw = snapshot_method()
    synchronization = raw["synchronization"]
    memory = raw["memory"]
    trace_methods = (
        "_runtime_trace_start",
        "_runtime_trace_stop",
        "_runtime_trace_snapshot",
        "_runtime_trace_export",
    )
    bounded_trace = all(getattr(prog, name, None) is not None for name in trace_methods)
    return RuntimeCapabilities(
        schema_version=1,
        backend=raw["backend"],
        program_domain=raw["program_domain"],
        statistics=True,
        statistics_schema_version=raw["schema_version"],
        bounded_trace=bounded_trace,
        trace_schema_version=1 if bounded_trace else None,
        chrome_trace_export=bounded_trace,
        backend_wait_telemetry=synchronization["backend_waits"] is not None,
        backend_lock_telemetry=(
            synchronization["backend_lock_samples"] is not None
        ),
        device_memory_telemetry=memory["device_raw_bytes"] is not None,
        cuda_mempool_telemetry=(
            memory["cuda_mempool_reserved_bytes"] is not None
        ),
    )


_active_trace_lock = threading.Lock()
_active_trace = None


class RuntimeTrace:
    """One-shot context manager for a bounded Program runtime trace."""

    def __init__(
        self,
        path,
        *,
        max_threads: int = 16,
        events_per_thread: int = 4096,
    ) -> None:
        self.path = os.fsdecode(os.fspath(path))
        self.max_threads = max_threads
        self.events_per_thread = events_per_thread
        self.started: Optional[RuntimeTraceSession] = None
        self.summary: Optional[RuntimeTraceSession] = None
        self.exported = False
        self._prog = None
        self._entered = False
        self._finished = False

    def __enter__(self) -> "RuntimeTrace":
        global _active_trace
        if self._entered or self._finished:
            raise TaichiRuntimeError("a RuntimeTrace context is one-shot")
        prog = _require_program("trace")
        start = _require_program_method(prog, "_runtime_trace_start", "trace")
        with _active_trace_lock:
            if _active_trace is not None:
                raise TaichiRuntimeError(
                    "nested or concurrent runtime traces are unsupported"
                )
            raw = start(self.max_threads, self.events_per_thread)
            try:
                started = _trace_session_from_raw(raw)
            except BaseException:
                stop = getattr(prog, "_runtime_trace_stop", None)
                if stop is not None:
                    try:
                        stop()
                    except BaseException:
                        pass
                raise
            _active_trace = self
        self._prog = prog
        self.started = started
        self._entered = True
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        global _active_trace
        cleanup_error = None
        try:
            stop = _require_program_method(
                self._prog, "_runtime_trace_stop", "trace"
            )
            export = _require_program_method(
                self._prog, "_runtime_trace_export", "trace"
            )
            self.summary = _trace_session_from_raw(stop())
            self.exported = bool(export(self.path))
            if not self.exported:
                raise TaichiRuntimeError(
                    f"unable to export runtime trace to {self.path!r}"
                )
        except BaseException as error:  # Preserve an active workload error.
            cleanup_error = error
        finally:
            with _active_trace_lock:
                if _active_trace is self:
                    _active_trace = None
            self._entered = False
            self._finished = True

        if cleanup_error is not None:
            if exc is not None:
                add_note = getattr(exc, "add_note", None)
                if add_note is not None:
                    add_note(f"runtime trace cleanup also failed: {cleanup_error}")
                return False
            raise cleanup_error
        return False


def trace(
    path,
    *,
    max_threads: int = 16,
    events_per_thread: int = 4096,
) -> RuntimeTrace:
    """Create a bounded runtime trace context for the active Program."""
    return RuntimeTrace(
        path,
        max_threads=max_threads,
        events_per_thread=events_per_thread,
    )


__all__ = [
    "DisplayStatistics",
    "FaultStatistics",
    "GraphStatistics",
    "MemoryStatistics",
    "RuntimeCapabilities",
    "RuntimeFault",
    "RuntimeStatistics",
    "RuntimeTrace",
    "RuntimeTraceSession",
    "SubmissionStatistics",
    "SynchronizationStatistics",
    "TraceStatistics",
    "TransferStatistics",
    "capabilities",
    "stats",
    "trace",
]
