"""Public runtime observability APIs for Taichi Forge."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Optional

from taichi_forge._lib.utils import (
    configure_startup_profile as _configure_startup_profile,
    startup_profile_raw_snapshot,
)
from taichi_forge.lang import impl
from taichi_forge.lang.exception import TaichiRuntimeError

_RUNTIME_STATISTICS_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class SubmissionStatistics:
    kernel_submissions: int
    graph_submissions: int
    graph_backend_submissions: int
    backend_graph_launches: int
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
class HostAllocatorStatistics:
    requested_live_bytes: Optional[int]
    peak_requested_live_bytes: Optional[int]
    reserved_bytes: Optional[int]
    committed_bytes: Optional[int]
    capacity_bytes: Optional[int]
    used_bytes: Optional[int]
    available_bytes: Optional[int]
    alignment_waste_bytes: Optional[int]
    unreclaimed_released_bytes: Optional[int]
    wasted_bytes: Optional[int]
    chunk_count: Optional[int]
    slab_chunk_count: Optional[int]
    large_chunk_count: Optional[int]
    exclusive_chunk_count: Optional[int]
    peak_reserved_bytes: Optional[int]
    peak_used_bytes: Optional[int]
    peak_wasted_bytes: Optional[int]
    peak_chunk_count: Optional[int]


@dataclass(frozen=True)
class MemoryStatistics:
    live_resources: int
    retiring_resources: int
    inflight_resources: int
    host_allocator: HostAllocatorStatistics
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
class StartupProfileEvent:
    name: str
    offset_ns: int


@dataclass(frozen=True)
class StartupProfilePhase:
    name: str
    begin_ns: int
    end_ns: int
    duration_ns: int


@dataclass(frozen=True)
class StartupProfile:
    schema_version: int
    enabled: bool
    elapsed_ns: int
    events: tuple
    phases: tuple


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
    schema_version = raw.get("schema_version")
    if schema_version != _RUNTIME_STATISTICS_SCHEMA_VERSION:
        raise TaichiRuntimeError(
            "unsupported runtime statistics schema "
            f"{schema_version!r}; expected "
            f"{_RUNTIME_STATISTICS_SCHEMA_VERSION}. Install matching "
            "taichi-forge and taichi-forge-runtime packages"
        )
    return RuntimeStatistics(
        schema_version=schema_version,
        backend=raw["backend"],
        program_domain=raw["program_domain"],
        submission=SubmissionStatistics(**raw["submission"]),
        synchronization=SynchronizationStatistics(**raw["synchronization"]),
        memory=MemoryStatistics(
            live_resources=raw["memory"]["live_resources"],
            retiring_resources=raw["memory"]["retiring_resources"],
            inflight_resources=raw["memory"]["inflight_resources"],
            host_allocator=HostAllocatorStatistics(
                **raw["memory"]["host_allocator"]
            ),
            host_requested_live_bytes=raw["memory"][
                "host_requested_live_bytes"
            ],
            host_raw_bytes=raw["memory"]["host_raw_bytes"],
            host_capacity_bytes=raw["memory"]["host_capacity_bytes"],
            device_requested_live_bytes=raw["memory"][
                "device_requested_live_bytes"
            ],
            device_raw_bytes=raw["memory"]["device_raw_bytes"],
            device_cached_bytes=raw["memory"]["device_cached_bytes"],
            cuda_mempool_reserved_bytes=raw["memory"][
                "cuda_mempool_reserved_bytes"
            ],
            cuda_mempool_used_bytes=raw["memory"][
                "cuda_mempool_used_bytes"
            ],
        ),
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
    statistics_schema_version = raw.get("schema_version")
    return RuntimeCapabilities(
        schema_version=1,
        backend=raw["backend"],
        program_domain=raw["program_domain"],
        statistics=(
            statistics_schema_version == _RUNTIME_STATISTICS_SCHEMA_VERSION
        ),
        statistics_schema_version=statistics_schema_version,
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


def configure_startup_profile(enabled=True, *, clear=False):
    """Enable or disable import/init profiling outside execution hot paths.

    Enabling here can profile subsequent ``ti.init()`` calls. Set
    ``TI_STARTUP_PROFILE=1`` before importing :mod:`taichi_forge` when native
    loader and Python import attribution is also required.
    """

    return _configure_startup_profile(enabled, clear=clear)


def startup_profile(*, clear=False) -> StartupProfile:
    """Return import and ``ti.init()`` checkpoints without requiring a Program."""

    raw = startup_profile_raw_snapshot(clear=clear)
    events = tuple(
        StartupProfileEvent(name=name, offset_ns=int(offset_ns))
        for name, offset_ns in raw["events"]
    )
    pending = {}
    phases = []
    for event in events:
        if event.name.endswith(".begin"):
            pending.setdefault(event.name[:-6], []).append(event.offset_ns)
        elif event.name.endswith(".end"):
            name = event.name[:-4]
            starts = pending.get(name)
            if starts:
                begin_ns = starts.pop()
                phases.append(
                    StartupProfilePhase(
                        name=name,
                        begin_ns=begin_ns,
                        end_ns=event.offset_ns,
                        duration_ns=max(0, event.offset_ns - begin_ns),
                    )
                )
    return StartupProfile(
        schema_version=1,
        enabled=bool(raw["enabled"]),
        elapsed_ns=int(raw["elapsed_ns"]),
        events=events,
        phases=tuple(phases),
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
    "HostAllocatorStatistics",
    "MemoryStatistics",
    "RuntimeCapabilities",
    "RuntimeFault",
    "RuntimeStatistics",
    "RuntimeTrace",
    "RuntimeTraceSession",
    "StartupProfile",
    "StartupProfileEvent",
    "StartupProfilePhase",
    "SubmissionStatistics",
    "SynchronizationStatistics",
    "TraceStatistics",
    "TransferStatistics",
    "capabilities",
    "configure_startup_profile",
    "stats",
    "startup_profile",
    "trace",
]
