import os
import threading
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from taichi_forge._lib import core as _ti_core
from taichi_forge.aot.utils import produce_injected_args_for_graph
from taichi_forge.lang import enums, impl, kernel_impl
from taichi_forge.lang._ndarray import Ndarray, ScalarNdarray
from taichi_forge.lang._storage_view import DenseNdarrayView, ndarray_view
from taichi_forge.lang._texture import Texture
from taichi_forge.lang.exception import TaichiCompilationError, TaichiRuntimeError
from taichi_forge.lang.field import ScalarField
from taichi_forge.lang.util import to_numpy_type
from taichi_forge.lang.matrix import Matrix, MatrixField, MatrixType
from taichi_forge.types._argument_descriptor import (
    describe_element_type,
)
from taichi_forge.types import ndarray_type
from taichi_forge.types.primitive_types import i32
from taichi_forge.types.texture_type import FORMAT2TY_CH, TY_CH2FORMAT
from taichi_forge.graph._native import (
    GraphTemporaryBuffer,
    ProviderOwnedNdarrayBinding,
    RecordableGraphAction,
    compile_native_graph_node,
)
from taichi_forge.graph._ir import (
    DispatchNode,
    IfRegion,
    GraphAccess,
    NativeCallNode,
    ObservationNode,
    ResourceEffect,
    RuntimeBinding,
    SequentialRegion,
    SwitchRegion,
    WhileRegion,
    analyze_elementwise_fusion,
    analyze_graph_ir,
    graph_ir_to_dict,
    plan_temporary_memory,
)
from taichi_forge.graph._submission import (
    SubmissionPacer,
    _new_submission_lane,
    _reserve_paced_submission,
)

ArgKind = _ti_core.ArgKind


def _new_runtime_graph_builder():
    builder = _ti_core.GraphBuilder()
    composer_setting = os.environ.get("TI_GRAPH_TWO_MAP_COMPOSER")
    composer_enabled = (
        composer_setting != "0"
        if composer_setting is not None
        else impl.current_cfg().arch != _ti_core.Arch.vulkan
    )
    if composer_enabled:
        builder._enable_two_map_composer()
    return builder


def _align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


class _GraphTemporaryArenaLease:
    def __init__(self, arena, slot):
        self._arena = arena
        self._slot = slot
        self.bindings = slot["bindings"]

    def attach(self, completion):
        if self._slot is None:
            return
        self._slot["completion"] = completion if completion.has_backend_work else None
        self._slot = None

    def cancel(self):
        if self._slot is not None:
            self._slot["completion"] = None
            self._slot = None


class _GraphTemporaryArena:
    """Runtime-owned byte arenas with a bounded async submission ring."""

    _BASE_ALIGNMENT = 16
    _WORD_BYTES = 4

    def __init__(self, plan, capacity=None):
        self.plan = plan
        self.capacity = int(
            capacity
            if capacity is not None
            else os.environ.get("TI_GRAPH_TEMPORARY_ARENA_SLOTS", "4")
        )
        if self.capacity < 1 or self.capacity > 64:
            raise TaichiRuntimeError(
                "TI_GRAPH_TEMPORARY_ARENA_SLOTS must be between 1 and 64"
            )
        self._slots = []
        self._allocations = 0
        self._reuses = 0
        self._waits = 0
        self._storage_bytes = _align_up(plan.planned_peak_bytes, self._WORD_BYTES)
        self._available = bool(plan.allocations) and not (
            plan.conflicting_requirements
            or any(
                allocation.alignment > self._BASE_ALIGNMENT
                for allocation in plan.allocations
            )
        )

    def _new_slot(self):
        storage = (
            None
            if self._storage_bytes == 0
            else ScalarNdarray(i32, (self._storage_bytes // self._WORD_BYTES,))
        )
        bindings = {
            allocation.name: GraphTemporaryBuffer(
                storage=storage,
                offset=allocation.offset,
                bytes=allocation.bytes,
                alignment=allocation.alignment,
                slot=allocation.slot,
            )
            for allocation in self.plan.allocations
        }
        slot = {"storage": storage, "bindings": bindings, "completion": None}
        self._slots.append(slot)
        self._allocations += 1
        return slot

    def _reclaim(self):
        for slot in self._slots:
            completion = slot["completion"]
            if completion is not None and completion.done():
                slot["completion"] = None

    def acquire(self):
        if not self.plan.allocations:
            return None
        if not self._available:
            raise TaichiRuntimeError(
                "Graph temporary requirements conflict or exceed the portable "
                "16-byte arena alignment contract"
            )
        self._reclaim()
        for slot in self._slots:
            if slot["completion"] is None:
                self._reuses += 1
                return _GraphTemporaryArenaLease(self, slot)
        if len(self._slots) < self.capacity:
            return _GraphTemporaryArenaLease(self, self._new_slot())
        slot = self._slots[0]
        slot["completion"].wait()
        slot["completion"] = None
        self._waits += 1
        self._reuses += 1
        return _GraphTemporaryArenaLease(self, slot)

    @property
    def stats(self):
        return {
            "materialized": bool(self._slots),
            "capacity": self.capacity if self.plan.allocations else 0,
            "slots": len(self._slots),
            "reserved_bytes": len(self._slots) * self._storage_bytes,
            "allocations": self._allocations,
            "reuses": self._reuses,
            "waits": self._waits,
        }


def _copy_observation_result(result):
    return {batch: dict(values) for batch, values in result.items()}


class _GraphObservationState:
    def __init__(self, arena, slot, sequence):
        self._arena = arena
        self._slot = slot
        self._sequence = sequence
        self._completion = None
        self._result = None
        self._discarded = False
        self._released = False
        self._lock = threading.Lock()

    @property
    def sequence(self):
        return self._sequence

    @property
    def bindings(self):
        return self._slot["bindings"]

    def attach(self, completion):
        with self._lock:
            if self._released:
                raise TaichiRuntimeError(
                    "Cannot attach a completion to a released observation slot"
                )
            self._completion = completion if completion.has_backend_work else None
        return self

    def _wait_locked(self):
        completion = self._completion
        if completion is None:
            return
        if not completion.done():
            self._arena._record_wait()
            completion.wait()

    def materialize(self):
        release = False
        with self._lock:
            if self._result is not None:
                return _copy_observation_result(self._result)
            if self._released:
                raise TaichiRuntimeError("Graph observation snapshot was discarded")
            self._wait_locked()
            result = self._arena._read_slot(self._slot)
            self._result = result
            self._released = True
            release = True
        if release:
            self._arena._release(self._slot, self)
        return _copy_observation_result(result)

    def discard(self):
        release = False
        with self._lock:
            if self._released:
                return
            completion = self._completion
            if completion is None or completion.done():
                self._released = True
                release = True
            else:
                self._discarded = True
        if release:
            self._arena._release(self._slot, self)

    def make_reusable(self):
        release = False
        with self._lock:
            if self._released:
                return
            self._wait_locked()
            if not self._discarded:
                self._result = self._arena._read_slot(self._slot)
            self._released = True
            release = True
        if release:
            self._arena._release(self._slot, self)


class _GraphObservationArenaLease:
    def __init__(self, state):
        self._state = state
        self.bindings = state.bindings

    def attach(self, completion):
        if self._state is None:
            return None
        state = self._state.attach(completion)
        self._state = None
        return state

    def materialize(self):
        if self._state is None:
            return {}
        state = self._state
        self._state = None
        return state.materialize()

    def cancel(self):
        if self._state is not None:
            self._state.discard()
            self._state = None


class _GraphObservationArena:
    """Bounded device snapshot slots with deferred packed host readback."""

    def __init__(self, nodes, capacity=None):
        self.nodes = tuple(nodes)
        if not self.nodes:
            self.capacity = 0
            self._slots = []
            self._next_sequence = 1
            self._allocations = 0
            self._reuses = 0
            self._waits = 0
            self._materializations = 0
            self._host_readback_bytes = 0
            self._lock = threading.Lock()
            return
        configured = (
            capacity
            if capacity is not None
            else os.environ.get("TI_GRAPH_OBSERVATION_SLOTS", "4")
        )
        self.capacity = int(configured)
        if self.capacity < 1 or self.capacity > 64:
            raise TaichiRuntimeError(
                "TI_GRAPH_OBSERVATION_SLOTS must be between 1 and 64"
            )
        self._slots = []
        self._next_sequence = 1
        self._allocations = 0
        self._reuses = 0
        self._waits = 0
        self._materializations = 0
        self._host_readback_bytes = 0
        self._lock = threading.Lock()

    def _new_slot(self):
        bindings = {}
        payload_bytes = 0
        for node in self.nodes:
            node_bindings, node_bytes = node.allocate_snapshot_buffers()
            bindings[node.name] = node_bindings
            payload_bytes += node_bytes
        slot = {
            "bindings": bindings,
            "payload_bytes": payload_bytes,
            "state": None,
        }
        self._slots.append(slot)
        self._allocations += 1
        return slot

    def acquire(self):
        if not self.nodes:
            return None
        while True:
            with self._lock:
                allocated = False
                slot = next(
                    (item for item in self._slots if item["state"] is None),
                    None,
                )
                if slot is None and len(self._slots) < self.capacity:
                    slot = self._new_slot()
                    allocated = True
                if slot is not None:
                    if not allocated:
                        self._reuses += 1
                    state = _GraphObservationState(self, slot, self._next_sequence)
                    self._next_sequence += 1
                    slot["state"] = state
                    return _GraphObservationArenaLease(state)
                oldest = min(
                    (item["state"] for item in self._slots),
                    key=lambda state: state.sequence,
                )
            oldest.make_reusable()

    def _read_slot(self, slot):
        sources = []
        hosts = []
        host_groups = {}
        for node in self.nodes:
            node_hosts = {}
            for key, storage in slot["bindings"][node.name].items():
                host = np.empty(
                    shape=storage.arr.total_shape(),
                    dtype=to_numpy_type(storage.dtype),
                )
                sources.append(storage.arr)
                hosts.append(host)
                node_hosts[key] = host
            host_groups[node.name] = node_hosts
        impl.get_runtime().prog.copy_graph_observations_to_host(sources, hosts)
        result = {
            node.name: node.decode_snapshot(host_groups[node.name])
            for node in self.nodes
        }
        byte_count = sum(host.nbytes for host in hosts)
        with self._lock:
            self._materializations += 1
            self._host_readback_bytes += byte_count
        return result

    def _record_wait(self):
        with self._lock:
            self._waits += 1

    def _release(self, slot, state):
        with self._lock:
            if slot["state"] is state:
                slot["state"] = None

    @property
    def stats(self):
        with self._lock:
            return {
                "materialized": bool(self._slots),
                "capacity": self.capacity if self.nodes else 0,
                "slots": len(self._slots),
                "reserved_bytes": sum(slot["payload_bytes"] for slot in self._slots),
                "allocations": self._allocations,
                "reuses": self._reuses,
                "waits": self._waits,
                "materializations": self._materializations,
                "host_readback_bytes": self._host_readback_bytes,
            }


@dataclass(frozen=True)
class GraphExecutionCounters:
    """Detailed backend counters captured after diagnostics are enabled."""

    attempts: int
    ordinary_fallbacks: int
    capture_attempts: int
    captures: int
    exact_replays: int
    patched_replays: int
    recaptures: int
    records: int
    replays: int
    structural_fallbacks: int
    transient_failures: int
    retry_backoff_fallbacks: int
    replay_slot_saturation_fallbacks: int
    capture_exceptions: int
    zero_arg_captures: int


@dataclass(frozen=True)
class GraphExecutionSegmentReport:
    """Read-only execution snapshot for one CGraph or native segment."""

    node_index: int
    kind: str
    dispatch_count: int
    compiled_task_count: Optional[int]
    runtime_arg_count: int
    static_dependency_count: int
    static_layout_fingerprint: str
    backend: str
    last_path: str
    fallback_reason: str
    backend_graph_path: bool
    backend_replay_path: bool
    zero_arg_eligible: bool
    persistent_argument_bytes: int
    last_driver_error: int
    retry_backoff_remaining: int
    consecutive_transient_failures: int
    counters_complete: bool
    counters: GraphExecutionCounters


@dataclass(frozen=True)
class GraphMemoryReport:
    """Known Graph-owned memory; driver-internal memory remains unknown."""

    persistent_argument_bytes: int
    persistent_observation_bytes: int
    persistent_temporary_bytes: int
    persistent_bytes: int
    transient_temporary_bytes: int
    planned_temporary_bytes: int
    temporary_reuse_bytes: int
    opaque_temporary_bytes: int
    temporary_plan_materialized: bool
    temporary_arena_capacity: int
    temporary_arena_slots: int
    temporary_arena_allocations: int
    temporary_arena_reuses: int
    temporary_arena_waits: int
    observation_arena_capacity: int
    observation_arena_slots: int
    observation_arena_allocations: int
    observation_arena_reuses: int
    observation_arena_waits: int
    observation_materializations: int
    observation_host_readback_bytes: int
    opaque_driver_bytes: Optional[int]


@dataclass(frozen=True)
class GraphExecutionReport:
    """Stable, immutable snapshot returned by Graph.execution_stats().

    Detailed backend counters are intentionally lazy. Calling
    execution_stats() enables them for later executions; the first report
    still exposes the latest path, fallback classification, task metadata and
    resource footprint. counters_complete is false only when GPU executions
    happened before this opt-in point.
    """

    schema_version: int
    arch: str
    lifecycle_state: str
    node_count: int
    cgraph_segment_count: int
    native_node_count: int
    observation_node_count: int
    dispatch_count: int
    compiled_task_count: Optional[int]
    runtime_arg_count: int
    static_dependency_count: int
    static_layout_fingerprint: str
    execution_path: str
    fallback_reason: str
    backend_graph_segments: int
    backend_replay_segments: int
    ordinary_fallback_segments: int
    counters_complete: bool
    segments: Tuple[GraphExecutionSegmentReport, ...]
    memory: GraphMemoryReport


@dataclass(frozen=True)
class GraphWhileReport:
    """Last execution of one structured Graph while region."""

    name: str
    backend: str
    lowering: str
    max_iterations: int
    logical_iterations: int
    executed_iterations: int
    overshoot_iterations: int
    observation_boundaries: Tuple[int, ...]
    predicate_values: Tuple[int, ...]
    counter_values: Tuple[int, ...]
    status_resource: Optional[str]
    status_values: Tuple[int, ...]
    chunk_sizes: Tuple[int, ...]
    observation_batches: int
    observation_scalar_count: int
    device_to_host_bytes: int
    initial_counter: Optional[int]
    final_counter: Optional[int]
    initial_status: Optional[int]
    final_status: Optional[int]
    native_upgrade_eligible: bool
    native_upgrade_reason: str
    persistent_staging_bytes: int
    staging_allocations: int
    staging_reuses: int
    packed_observation_batches: int
    direct_observation_batches: int
    staging_fallback_batches: int
    packed_observation_bytes: int
    condition_dispatch_count: int
    body_dispatch_count: int
    control_inputs: Tuple[str, ...]
    carried_state: Tuple[str, ...]


@dataclass(frozen=True)
class GraphBranchReport:
    """Last execution of one structured Graph if/switch region."""

    name: str
    backend: str
    kind: str
    lowering: str
    selector_resource: str
    selector_value: int
    selected_branch: str
    observation_scalar_count: int
    device_to_host_bytes: int
    condition_dispatch_count: int
    branch_dispatch_count: int
    control_inputs: Tuple[str, ...]


_COUNTER_FIELDS = (
    "attempts",
    "ordinary_fallbacks",
    "capture_attempts",
    "captures",
    "exact_replays",
    "patched_replays",
    "recaptures",
    "records",
    "replays",
    "structural_fallbacks",
    "transient_failures",
    "retry_backoff_fallbacks",
    "replay_slot_saturation_fallbacks",
    "capture_exceptions",
    "zero_arg_captures",
)
_BACKEND_GRAPH_PATHS = frozenset(
    (
        "cuda_capture",
        "cuda_exact_replay",
        "cuda_patched_replay",
        "vulkan_record",
        "vulkan_replay",
        "vulkan_patched_replay",
    )
)
_BACKEND_REPLAY_PATHS = frozenset(
    (
        "cuda_exact_replay",
        "cuda_patched_replay",
        "vulkan_replay",
        "vulkan_patched_replay",
    )
)


def _combine_layout_fingerprints(fingerprints):
    # FNV-1a over sorted fixed-width values. Tree identity and addresses are
    # deliberately absent, so equal layouts produce equal report values.
    value = 14695981039346656037
    prime = 1099511628211
    ordered = sorted(int(item) for item in fingerprints)
    for item in (len(ordered), *ordered):
        for shift in range(0, 64, 8):
            value ^= (item >> shift) & 0xFF
            value = (value * prime) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def _empty_backend_stats():
    stats = {name: 0 for name in _COUNTER_FIELDS}
    stats.update(
        {
            "backend": "none",
            "last_path": "none",
            "last_fallback_reason": "none",
            "zero_arg_eligible": False,
            "known_persistent_argument_bytes": 0,
            "known_compiled_tasks": 0,
            "known_compiled_dispatches": 0,
            "last_driver_error": 0,
            "retry_backoff_remaining": 0,
            "consecutive_transient_failures": 0,
            "diagnostics_previously_enabled": False,
            "diagnostics_counters_complete": True,
        }
    )
    return stats


def _backend_name(arch):
    if arch in ("x64", "arm64"):
        return "cpu"
    return arch


def _execution_report(
    definition,
    arch,
    lifecycle_state,
    instance_kind,
    backend_stats,
    observation_staging_bytes=0,
    temporary_memory_plan=None,
    temporary_arena_stats=None,
    observation_arena_stats=None,
):
    segments = []
    stats_cursor = 0
    for index, node in enumerate(definition["nodes"]):
        kind = node["kind"]
        if kind != "cgraph":
            path = (
                "unavailable"
                if lifecycle_state != "ready"
                else (
                    "asynchronous_snapshot"
                    if kind == "observation"
                    else (
                        "native_replay"
                        if instance_kind in ("cuda_native_replay", "cpu_native_replay")
                        else "native_dispatch"
                    )
                )
            )
            segments.append(
                GraphExecutionSegmentReport(
                    node_index=index,
                    kind=kind,
                    dispatch_count=node["dispatch_count"],
                    compiled_task_count=None,
                    runtime_arg_count=node["runtime_arg_count"],
                    static_dependency_count=0,
                    static_layout_fingerprint=_combine_layout_fingerprints(()),
                    backend=_backend_name(arch),
                    last_path=path,
                    fallback_reason=(
                        lifecycle_state if lifecycle_state != "ready" else "none"
                    ),
                    backend_graph_path=False,
                    backend_replay_path=False,
                    zero_arg_eligible=False,
                    persistent_argument_bytes=0,
                    last_driver_error=0,
                    retry_backoff_remaining=0,
                    consecutive_transient_failures=0,
                    counters_complete=True,
                    counters=GraphExecutionCounters(
                        **{name: 0 for name in _COUNTER_FIELDS}
                    ),
                )
            )
            continue

        stats = (
            backend_stats[stats_cursor]
            if stats_cursor < len(backend_stats)
            else _empty_backend_stats()
        )
        stats_cursor += 1
        known_dispatches = int(stats.get("known_compiled_dispatches", 0))
        tasks = (
            int(stats.get("known_compiled_tasks", 0))
            if known_dispatches
            == node.get("physical_dispatch_count", node["dispatch_count"])
            else None
        )
        backend = stats.get("backend", "none")
        if backend == "none":
            backend = _backend_name(arch)
        path = stats.get("last_path", "none")
        if lifecycle_state != "ready":
            path = "unavailable"
        elif path == "none" and tasks is not None:
            path = "ordinary"
        elif path == "none":
            path = "not_run"
        fallback_reason = stats.get("last_fallback_reason", "none")
        if lifecycle_state != "ready":
            fallback_reason = lifecycle_state
        gpu_backend = backend in ("cuda", "vulkan")
        counters_complete = not gpu_backend or bool(
            stats.get("diagnostics_counters_complete", True)
        )
        segments.append(
            GraphExecutionSegmentReport(
                node_index=index,
                kind=kind,
                dispatch_count=node["dispatch_count"],
                compiled_task_count=tasks,
                runtime_arg_count=node["runtime_arg_count"],
                static_dependency_count=len(node["dependency_info"]),
                static_layout_fingerprint=_combine_layout_fingerprints(
                    dependency[2] for dependency in node["dependency_info"]
                ),
                backend=backend,
                last_path=path,
                fallback_reason=fallback_reason,
                backend_graph_path=path in _BACKEND_GRAPH_PATHS,
                backend_replay_path=path in _BACKEND_REPLAY_PATHS,
                zero_arg_eligible=bool(stats.get("zero_arg_eligible", False)),
                persistent_argument_bytes=int(
                    stats.get("known_persistent_argument_bytes", 0)
                ),
                last_driver_error=int(stats.get("last_driver_error", 0)),
                retry_backoff_remaining=int(stats.get("retry_backoff_remaining", 0)),
                consecutive_transient_failures=int(
                    stats.get("consecutive_transient_failures", 0)
                ),
                counters_complete=counters_complete,
                counters=GraphExecutionCounters(
                    **{name: int(stats.get(name, 0)) for name in _COUNTER_FIELDS}
                ),
            )
        )

    cgraph_segments = tuple(s for s in segments if s.kind == "cgraph")
    task_counts = tuple(s.compiled_task_count for s in cgraph_segments)
    compiled_task_count = (
        sum(task_counts)
        if task_counts and all(value is not None for value in task_counts)
        else None
    )
    path_segments = cgraph_segments or tuple(segments)
    paths = {segment.last_path for segment in path_segments}
    execution_path = (
        lifecycle_state
        if lifecycle_state != "ready"
        else (next(iter(paths)) if len(paths) == 1 else "mixed") if paths else "not_run"
    )
    reasons = {
        segment.fallback_reason
        for segment in segments
        if segment.fallback_reason != "none"
    }
    fallback_reason = (
        "none" if not reasons else next(iter(reasons)) if len(reasons) == 1 else "mixed"
    )
    dependency_info = definition["dependency_info"]
    persistent_argument_bytes = sum(
        segment.persistent_argument_bytes for segment in cgraph_segments
    )
    temporary_memory_plan = temporary_memory_plan or {}
    temporary_arena_stats = temporary_arena_stats or {}
    observation_arena_stats = observation_arena_stats or {}
    temporary_plan_materialized = bool(temporary_arena_stats.get("materialized", False))
    planned_temporary_bytes = int(temporary_memory_plan.get("planned_peak_bytes", 0))
    persistent_temporary_bytes = int(temporary_arena_stats.get("reserved_bytes", 0))
    persistent_observation_bytes = int(observation_staging_bytes) + int(
        observation_arena_stats.get("reserved_bytes", 0)
    )
    memory = GraphMemoryReport(
        persistent_argument_bytes=persistent_argument_bytes,
        persistent_observation_bytes=persistent_observation_bytes,
        persistent_temporary_bytes=persistent_temporary_bytes,
        persistent_bytes=(
            persistent_argument_bytes
            + persistent_observation_bytes
            + persistent_temporary_bytes
        ),
        transient_temporary_bytes=(
            planned_temporary_bytes if temporary_plan_materialized else 0
        ),
        planned_temporary_bytes=planned_temporary_bytes,
        temporary_reuse_bytes=int(temporary_memory_plan.get("reused_bytes", 0)),
        opaque_temporary_bytes=int(temporary_memory_plan.get("opaque_bytes", 0)),
        temporary_plan_materialized=temporary_plan_materialized,
        temporary_arena_capacity=int(temporary_arena_stats.get("capacity", 0)),
        temporary_arena_slots=int(temporary_arena_stats.get("slots", 0)),
        temporary_arena_allocations=int(temporary_arena_stats.get("allocations", 0)),
        temporary_arena_reuses=int(temporary_arena_stats.get("reuses", 0)),
        temporary_arena_waits=int(temporary_arena_stats.get("waits", 0)),
        observation_arena_capacity=int(observation_arena_stats.get("capacity", 0)),
        observation_arena_slots=int(observation_arena_stats.get("slots", 0)),
        observation_arena_allocations=int(
            observation_arena_stats.get("allocations", 0)
        ),
        observation_arena_reuses=int(observation_arena_stats.get("reuses", 0)),
        observation_arena_waits=int(observation_arena_stats.get("waits", 0)),
        observation_materializations=int(
            observation_arena_stats.get("materializations", 0)
        ),
        observation_host_readback_bytes=int(
            observation_arena_stats.get("host_readback_bytes", 0)
        ),
        opaque_driver_bytes=None,
    )
    return GraphExecutionReport(
        schema_version=4,
        arch=arch,
        lifecycle_state=lifecycle_state,
        node_count=len(segments),
        cgraph_segment_count=len(cgraph_segments),
        native_node_count=definition["native_count"],
        observation_node_count=definition.get("observation_count", 0),
        dispatch_count=definition["dispatch_count"],
        compiled_task_count=compiled_task_count,
        runtime_arg_count=definition["runtime_arg_count"],
        static_dependency_count=len(dependency_info),
        static_layout_fingerprint=_combine_layout_fingerprints(
            dependency[2] for dependency in dependency_info
        ),
        execution_path=execution_path,
        fallback_reason=fallback_reason,
        backend_graph_segments=sum(
            segment.backend_graph_path for segment in cgraph_segments
        ),
        backend_replay_segments=sum(
            segment.backend_replay_path for segment in cgraph_segments
        ),
        ordinary_fallback_segments=sum(
            segment.last_path in ("ordinary", "ordinary_fallback")
            for segment in cgraph_segments
        ),
        counters_complete=all(segment.counters_complete for segment in segments),
        segments=tuple(segments),
        memory=memory,
    )


class _NativeReplayExecutable:
    def __init__(self, nodes):
        self._nodes = tuple(nodes)

    def prewarm(self):
        for node in self._nodes:
            node.executable.prewarm()
        return self

    def run(self, context, temporaries=None):
        for node in self._nodes:
            node.run(context, temporaries)


class _CGraphJITExecutable:
    def __init__(self, compiled_graph):
        self.compiled_graph = compiled_graph
        self._jit_cache = _ti_core.CompiledGraphJITCache()

    def prewarm(self):
        return self

    def run(self, context, temporaries=None):
        self.compiled_graph.jit_run_cached(
            context.compile_config(), context.flattened_args(), self._jit_cache
        )

    def invalidate_runtime(self):
        self._jit_cache.clear_runtime_state()

    @property
    def debug_graph_stats(self):
        return self._jit_cache._debug_graph_stats()


class _GraphRunContext:
    _empty_args = {}

    def __init__(self):
        self._args = None
        self._flattened_args = None
        self._compile_config = None
        self._last_arg_signature = None
        self._last_flattened = None

    def begin(self, args, fixed_args=None):
        if fixed_args:
            overlap = fixed_args.keys() & args.keys()
            if overlap:
                raise TaichiRuntimeError(
                    "Graph runtime arguments collide with provider-owned "
                    "fixed bindings: " + ", ".join(sorted(overlap))
                )
            merged = dict(fixed_args)
            merged.update(args)
            self._args = merged
        else:
            self._args = args
        self._flattened_args = None

    def end(self):
        # Runtime resource completion is owned by the native Program registry.
        # Keeping the Python argument dict here after submission only delays
        # wrapper retirement and can pin arbitrarily large user object graphs.
        # The generation-qualified flattened fast cache remains reusable.
        self._args = None
        self._flattened_args = None

    def runtime_args(self):
        return self._args

    def compile_config(self):
        if self._compile_config is None:
            self._compile_config = impl.get_runtime().prog.config()
        return self._compile_config

    def flattened_args(self, arg_names=None):
        if self._flattened_args is None:
            self._flattened_args = self._flatten_runtime_args(self._args)
        if arg_names is not None and not arg_names.issubset(self._flattened_args):
            raise TaichiRuntimeError(
                "CGraph segment requested undeclared runtime arguments"
            )
        # The CompiledGraph Python binding iterates the compiled segment's own
        # declarations and constructs a node-local C++ IValue map. Passing the
        # shared flattened dict here is therefore a zero-copy filtered view;
        # slicing it again in Python adds two dict copies to every mixed run.
        return self._flattened_args

    def _flatten_runtime_args(self, args):
        if not args:
            return self._empty_args

        signature = []
        dynamic_items = []
        arch = self.compile_config().arch
        runtime_storage_backend = arch in (
            _ti_core.Arch.x64,
            _ti_core.Arch.arm64,
            _ti_core.Arch.cuda,
            _ti_core.Arch.vulkan,
        )
        if arch == _ti_core.Arch.cuda:
            ndarray_consumer = "graph_capture"
            ndarray_mode = "capture"
        else:
            ndarray_consumer = "graph_replay"
            ndarray_mode = "replay"
        for k, v in args.items():
            if isinstance(v, (Ndarray, ProviderOwnedNdarrayBinding)):
                if v.arr is None:
                    raise TaichiRuntimeError(
                        "Cannot submit an Ndarray to Graph.run() after its Taichi runtime has been reset"
                    )
                signature.append((k, "ndarray", v._runtime_allocation_identity))
            elif isinstance(v, (DenseNdarrayView, ScalarField, MatrixField)):
                signature.append((k, "dense_storage", id(v)))
            elif isinstance(v, Texture):
                if v.tex is None:
                    raise TaichiRuntimeError(
                        "Cannot submit a Texture to Graph.run() after its Taichi runtime has been reset"
                    )
                signature.append((k, "texture", id(v), id(v.tex)))
            elif isinstance(v, Matrix):
                signature.append((k, "matrix"))
                dynamic_items.append((k, v.entries))
            elif isinstance(v, (int, float)):
                signature.append((k, "scalar", type(v)))
                dynamic_items.append((k, v))
            else:
                raise TaichiRuntimeError(
                    "Only Python scalars, ti.Matrix, ti.Ndarray, canonical "
                    "dense Field, and DenseNdarrayView are supported as "
                    f"runtime arguments but got {type(v)}"
                )

        signature = tuple(signature)
        if signature == self._last_arg_signature:
            flattened = self._last_flattened
        else:
            flattened = {}
            for k, v in args.items():
                if isinstance(v, (Ndarray, ProviderOwnedNdarrayBinding)):
                    if runtime_storage_backend:
                        flattened[k] = (
                            v.arr,
                            v._runtime_storage_argument(ndarray_consumer, ndarray_mode),
                        )
                    else:
                        flattened[k] = v.arr
                elif isinstance(v, (DenseNdarrayView, ScalarField, MatrixField)):
                    if not runtime_storage_backend:
                        raise TaichiRuntimeError(
                            "Dense storage Graph runtime arguments are supported "
                            "on CPU, CUDA, and Vulkan"
                        )
                    view = v if isinstance(v, DenseNdarrayView) else ndarray_view(v)
                    try:
                        runtime_argument = view._runtime_storage_argument(
                            ndarray_consumer, ndarray_mode
                        )
                    except ValueError:
                        if arch != _ti_core.Arch.cuda:
                            raise
                        runtime_argument = view._runtime_storage_argument(
                            "graph_replay", "replay"
                        )
                    flattened[k] = (view, runtime_argument)
                elif isinstance(v, Texture):
                    flattened[k] = v.tex
            self._last_arg_signature = signature
            self._last_flattened = flattened
        for k, v in dynamic_items:
            flattened[k] = v
        return flattened


class _CompiledCGraphNode:
    needs_runtime_args = True

    def __init__(
        self,
        compiled_graph,
        dispatch_count,
        runtime_arg_names=(),
        ir_node=None,
        recording_dispatches=(),
        lifetime_leases=(),
        source_native_count=0,
        region_kind="cgraph",
        fixed_runtime_args=None,
        temporary_actions=(),
    ):
        self.compiled_graph = compiled_graph
        self.dispatch_count = dispatch_count
        composer_stats = dict(getattr(compiled_graph, "_composer_stats", {}))
        self.physical_dispatch_count = int(
            composer_stats.get("physical_dispatches", dispatch_count)
        )
        self.composer_applied_groups = int(composer_stats.get("applied_groups", 0))
        self.composer_lowering_available = bool(
            composer_stats.get("lowering_available", False)
        )
        self.recording_runtime_arg_names = frozenset(runtime_arg_names)
        self.fixed_runtime_args = dict(
            {} if fixed_runtime_args is None else fixed_runtime_args
        )
        self.temporary_actions = tuple(temporary_actions)
        self.temporary_runtime_arg_names = frozenset().union(
            *(
                frozenset(action.temporary_bindings)
                for action in self.temporary_actions
            )
        )
        if not self.fixed_runtime_args.keys() <= self.recording_runtime_arg_names:
            raise TaichiRuntimeError(
                "CGraph fixed bindings must be declared runtime arguments"
            )
        if not self.temporary_runtime_arg_names <= self.recording_runtime_arg_names:
            raise TaichiRuntimeError(
                "CGraph temporary bindings must be declared runtime arguments"
            )
        if self.fixed_runtime_args.keys() & self.temporary_runtime_arg_names:
            raise TaichiRuntimeError(
                "CGraph fixed and temporary bindings must be disjoint"
            )
        self.runtime_arg_names = self.recording_runtime_arg_names.difference(
            (*self.fixed_runtime_args, *self.temporary_runtime_arg_names)
        )
        self.recording_dispatches = tuple(
            (kernel, tuple(args)) for kernel, args in recording_dispatches
        )
        self.lifetime_leases = tuple(lifetime_leases)
        self.source_native_count = int(source_native_count)
        self.region_kind = region_kind
        self.ir_node = ir_node or SequentialRegion(
            tuple(
                DispatchNode(name=f"dispatch_{index}")
                for index in range(dispatch_count)
            ),
            name="cgraph",
        )
        dependency_info = getattr(compiled_graph, "_snode_tree_dependency_info", None)
        if dependency_info is None:
            dependency_info = (
                (*dependency, 0)
                for dependency in getattr(
                    compiled_graph, "_snode_tree_dependencies", ()
                )
            )
        self.snode_tree_dependency_info = frozenset(
            tuple(dependency) for dependency in dependency_info
        )
        self.snode_tree_dependencies = frozenset(
            dependency[:2] for dependency in self.snode_tree_dependency_info
        )
        self._jit_cache = _ti_core.CompiledGraphJITCache()

    def run(self, context, temporaries=None):
        self.compiled_graph.jit_run_cached(
            context.compile_config(),
            context.flattened_args(self.recording_runtime_arg_names),
            self._jit_cache,
        )

    def invalidate_runtime(self):
        self._jit_cache.clear_runtime_state()

    @property
    def debug_graph_stats(self):
        return self._jit_cache._debug_graph_stats()

    @property
    def debug_info(self):
        info = {"kind": self.region_kind, "dispatch_count": self.dispatch_count}
        if self.physical_dispatch_count != self.dispatch_count:
            info["physical_dispatch_count"] = self.physical_dispatch_count
        if self.composer_applied_groups:
            info["composed_two_map_groups"] = self.composer_applied_groups
        if self.source_native_count:
            info["lowered_native_count"] = self.source_native_count
        return info


class _CompiledNativeGraphNode:
    snode_tree_dependencies = frozenset()
    snode_tree_dependency_info = frozenset()
    dispatch_count = 0
    source_native_count = 1
    region_kind = "native"

    def __init__(self, executable):
        self.executable = executable
        self.ir_node = getattr(
            executable,
            "graph_ir_node",
            NativeCallNode(type(executable).__name__),
        )
        schema = tuple(executable.runtime_arg_schema)
        if not all(isinstance(binding, RuntimeBinding) for binding in schema):
            raise TaichiRuntimeError(
                "Native Graph runtime_arg_schema must contain " "RuntimeBinding values"
            )
        if any(not binding.required for binding in schema):
            raise TaichiRuntimeError(
                "Optional native Graph runtime arguments are not supported"
            )
        public_runtime_arg_names = frozenset(binding.name for binding in schema)
        self.temporary_names = frozenset(
            requirement.name for requirement in executable.temporary_requirements
        )
        self.recordable_action = executable.recordable_action
        if self.recordable_action is not None and not isinstance(
            self.recordable_action, RecordableGraphAction
        ):
            raise TaichiRuntimeError(
                "Native Graph recordable_action must implement " "RecordableGraphAction"
            )
        self.fixed_runtime_args = (
            {}
            if self.recordable_action is None
            else dict(self.recordable_action.fixed_bindings)
        )
        temporary_binding_map = (
            {}
            if self.recordable_action is None
            else dict(self.recordable_action.temporary_bindings)
        )
        if any(
            not isinstance(symbol, str)
            or not symbol
            or not isinstance(requirement, str)
            or not requirement
            for symbol, requirement in temporary_binding_map.items()
        ):
            raise TaichiRuntimeError(
                "Recordable action temporary bindings must map nonempty "
                "symbol names to nonempty requirement names"
            )
        if (
            self.recordable_action is not None
            and set(temporary_binding_map.values()) != self.temporary_names
        ):
            raise TaichiRuntimeError(
                "Recordable action temporary bindings must cover exactly its "
                "Graph temporary requirements"
            )
        self.temporary_binding_map = temporary_binding_map
        self.temporary_runtime_arg_names = frozenset(temporary_binding_map)
        self.temporary_actions = (
            () if not temporary_binding_map else (self.recordable_action,)
        )
        overlap = public_runtime_arg_names & self.fixed_runtime_args.keys()
        if overlap:
            raise TaichiRuntimeError(
                "Recordable action fixed bindings overlap public runtime "
                "arguments: " + ", ".join(sorted(overlap))
            )
        private_overlap = self.temporary_runtime_arg_names & (
            public_runtime_arg_names | self.fixed_runtime_args.keys()
        )
        if private_overlap:
            raise TaichiRuntimeError(
                "Recordable action temporary symbols overlap public or fixed "
                "arguments: " + ", ".join(sorted(private_overlap))
            )
        self.recording_runtime_arg_names = frozenset(
            (
                *public_runtime_arg_names,
                *self.fixed_runtime_args.keys(),
                *self.temporary_runtime_arg_names,
            )
        )
        self.runtime_arg_names = public_runtime_arg_names
        self.needs_runtime_args = bool(self.recording_runtime_arg_names)
        self.lifetime_leases = (
            executable,
            *tuple(executable.lifetime_leases),
        )
        if self.recordable_action is not None:
            recorder_names = frozenset().union(
                *(
                    _runtime_arg_names(args)
                    for _, args in self.recordable_action.dispatches
                )
            )
            if recorder_names != self.recording_runtime_arg_names:
                raise TaichiRuntimeError(
                    "Recordable action dispatch arguments must match its "
                    "public and fixed bindings"
                )

    def run(self, context, temporaries=None):
        runtime_args = None
        if self.needs_runtime_args:
            all_args = context.runtime_args()
            runtime_args = {name: all_args[name] for name in self.runtime_arg_names}
        if not self.temporary_names:
            if runtime_args is None:
                return self.executable.run()
            return self.executable.run(runtime_args)
        if temporaries is None or not self.temporary_names.issubset(temporaries):
            raise TaichiRuntimeError(
                "Native Graph temporary requirements were not materialized"
            )
        bindings = {name: temporaries[name] for name in self.temporary_names}
        return self.executable.run_with_graph_temporaries(bindings, runtime_args)

    @property
    def debug_info(self):
        info = dict(self.executable.debug_info)
        if self.recordable_action is not None:
            info["recordable_action"] = self.recordable_action.capabilities.to_dict()
            info["fixed_binding_count"] = len(self.fixed_runtime_args)
        return info


_OBSERVATION_PACK_KERNELS = {}


def _observation_pack_kernel(dtype):
    key = str(dtype)
    kernel = _OBSERVATION_PACK_KERNELS.get(key)
    if kernel is not None:
        return kernel

    @kernel_impl.kernel
    def pack_scalar_snapshot(
        source: ndarray_type.ndarray(dtype=dtype, ndim=0),
        destination: ndarray_type.ndarray(dtype=dtype, ndim=1),
        index: i32,
    ):
        destination[index] = source[None]

    _OBSERVATION_PACK_KERNELS[key] = pack_scalar_snapshot
    return pack_scalar_snapshot


class _CompiledObservationGraphNode:
    needs_runtime_args = True
    snode_tree_dependencies = frozenset()
    snode_tree_dependency_info = frozenset()
    source_native_count = 0
    region_kind = "observation"

    def __init__(self, values, name):
        if not isinstance(name, str) or not name:
            raise TaichiRuntimeError(
                "Graph observation name must be a non-empty string"
            )
        values = tuple(values)
        if not values:
            raise TaichiRuntimeError("Graph observation requires at least one value")
        groups = {}
        entries = []
        for value in values:
            if getattr(value, "tag", None) != ArgKind.NDARRAY:
                raise TaichiRuntimeError(
                    "Graph observation values must be symbolic ndarray arguments"
                )
            descriptor = describe_element_type(value.dtype())
            if (
                value.field_dim != 0
                or value.element_shape
                or descriptor.category != "scalar"
            ):
                raise TaichiRuntimeError(
                    "Graph observation values must be scalar ndarrays with ndim=0"
                )
            dtype = value.dtype()
            key = str(dtype)
            group = groups.setdefault(key, {"dtype": dtype, "names": []})
            index = len(group["names"])
            group["names"].append(value.name)
            entries.append((value.name, key, index, dtype))
        names = tuple(entry[0] for entry in entries)
        if len(set(names)) != len(names):
            raise TaichiRuntimeError(
                "Graph observation values must have unique argument names"
            )
        self.name = name
        self._groups = tuple(
            (key, group["dtype"], tuple(group["names"]))
            for key, group in groups.items()
        )
        self._entries = tuple(entries)
        self._kernels = {
            key: _observation_pack_kernel(dtype) for key, dtype, _ in self._groups
        }
        self._active_buffers = None
        self.runtime_arg_names = frozenset(names)
        self.dispatch_count = len(entries)
        self.physical_dispatch_count = self.dispatch_count
        self.ir_node = ObservationNode(
            name=name,
            effects=tuple(
                ResourceEffect(arg_name, GraphAccess.READ) for arg_name in names
            ),
            bindings=tuple(RuntimeBinding(arg_name, "ndarray") for arg_name in names),
            batch=name,
            synchronization=False,
            opaque=False,
        )

    def allocate_snapshot_buffers(self):
        buffers = {}
        byte_count = 0
        for key, dtype, names in self._groups:
            buffers[key] = ScalarNdarray(dtype, (len(names),))
            byte_count += np.dtype(to_numpy_type(dtype)).itemsize * len(names)
        return buffers, byte_count

    def bind_snapshot_buffers(self, buffers):
        self._active_buffers = buffers

    def clear_snapshot_buffers(self):
        self._active_buffers = None

    def run(self, context, temporaries=None):
        if self._active_buffers is None:
            raise TaichiRuntimeError("Graph observation snapshot slot was not bound")
        runtime_args = context.runtime_args()
        for arg_name, key, index, dtype in self._entries:
            value = runtime_args[arg_name]
            if (
                not isinstance(value, Ndarray)
                or value.shape != ()
                or str(value.dtype) != str(dtype)
            ):
                raise TaichiRuntimeError(
                    f"Graph observation {arg_name} requires a scalar ndarray "
                    f"with dtype {dtype}"
                )
            self._kernels[key](value, self._active_buffers[key], index)

    def decode_snapshot(self, hosts):
        result = {}
        for key, _, names in self._groups:
            values = hosts[key].reshape(-1)
            result.update(
                (name, values[index].item()) for index, name in enumerate(names)
            )
        return result

    @property
    def debug_info(self):
        return {
            "kind": "observation",
            "name": self.name,
            "value_count": len(self._entries),
            "packed_group_count": len(self._groups),
            "asynchronous": True,
        }


def _control_scalar_values(values, names, *, use_transfer_planner):
    if len(values) != len(names):
        raise TaichiRuntimeError("Structured Graph control names do not match values")
    sources = []
    hosts = []
    for value, name in zip(values, names):
        ndarray = getattr(value, "arr", None)
        dtype = getattr(value, "dtype", None)
        if ndarray is None or dtype is None:
            raise TaichiRuntimeError(
                f"Structured Graph control {name} must be a device ndarray scalar"
            )
        host = np.empty(shape=ndarray.total_shape(), dtype=to_numpy_type(dtype))
        if host.size != 1:
            raise TaichiRuntimeError(
                f"Structured Graph control {name} must contain exactly one scalar"
            )
        sources.append(ndarray)
        hosts.append(host)
    program = impl.get_runtime().prog
    if use_transfer_planner:
        program.copy_graph_observations_to_host(sources, hosts)
    else:
        program.copy_ndarrays_to_host(sources, hosts)
    return (
        tuple(int(host.reshape(-1)[0]) for host in hosts),
        sum(host.nbytes for host in hosts),
    )


def _structured_chunk_limit(arch, requested, masked_execution):
    cpu_arches = (_ti_core.Arch.x64, _ti_core.Arch.arm64)
    if arch in cpu_arches or not masked_execution:
        return 1
    configured = (
        requested
        if requested is not None
        else os.environ.get("TI_GRAPH_WHILE_CHUNK_SIZE", "4")
    )
    if isinstance(configured, bool) or not isinstance(
        configured, (int, np.integer, str)
    ):
        raise TaichiRuntimeError("Graph while chunk_size must be an integer")
    try:
        configured = int(configured)
    except ValueError as error:
        raise TaichiRuntimeError("Graph while chunk_size must be an integer") from error
    if configured <= 0:
        raise TaichiRuntimeError("Graph while chunk_size must be positive")
    return min(configured, 64)


def _cuda_while_upgrade_status(arch, mode):
    if mode not in ("auto", "portable", "native_required"):
        raise TaichiRuntimeError(
            "Graph while lowering_mode must be auto, portable, or " "native_required"
        )
    if mode == "portable":
        return False, "forced_portable"
    if arch != _ti_core.Arch.cuda:
        if mode == "native_required":
            raise TaichiRuntimeError("Graph while native_required mode needs CUDA")
        return False, "not_cuda"
    capabilities = dict(_ti_core.cuda_conditional_graph_capabilities())
    if not capabilities["driver_version_eligible"]:
        reason = "cuda_driver_api_version_below_12_8"
    elif not capabilities["conditional_graph_symbols_loaded"]:
        reason = "cuda_conditional_graph_symbols_not_loaded"
    elif not capabilities.get("general_device_setter_lowering_compiled", False):
        reason = "cuda_conditional_setter_lowering_not_compiled"
    else:
        return True, "eligible"
    if mode == "native_required":
        raise TaichiRuntimeError(
            f"Graph while native CUDA lowering unavailable: {reason}"
        )
    return False, reason


def _cuda_branch_upgrade_status(arch, mode, kind):
    if mode not in ("auto", "portable", "native_required"):
        raise TaichiRuntimeError(
            f"Graph {kind} lowering_mode must be auto, portable, or "
            "native_required"
        )
    if mode == "portable":
        return False, "forced_portable"
    if arch != _ti_core.Arch.cuda:
        if mode == "native_required":
            raise TaichiRuntimeError(
                f"Graph {kind} native_required mode needs CUDA"
            )
        return False, "not_cuda"
    capabilities = dict(_ti_core.cuda_conditional_graph_capabilities())
    if not capabilities["driver_version_eligible"]:
        reason = "cuda_driver_api_version_below_12_8"
    elif not capabilities["conditional_graph_symbols_loaded"]:
        reason = "cuda_conditional_graph_symbols_not_loaded"
    elif not capabilities.get("general_device_setter_lowering_compiled", False):
        reason = "cuda_conditional_setter_lowering_not_compiled"
    else:
        return True, "eligible"
    if mode == "native_required":
        raise TaichiRuntimeError(
            f"Graph {kind} native CUDA lowering unavailable: {reason}"
        )
    return False, reason


def _compile_sequential_runtime_node(
    sequences,
    *,
    repetitions=1,
    name,
    region_kind="sequential",
    region_kinds=None,
):
    sequences = tuple(sequences)
    if not sequences or any(
        not isinstance(sequence, Sequential) for sequence in sequences
    ):
        raise TaichiRuntimeError("Structured Graph regions require Sequential values")
    if region_kinds is None:
        region_kinds = (region_kind,) * len(sequences)
    else:
        region_kinds = tuple(region_kinds)
        if len(region_kinds) != len(sequences):
            raise TaichiRuntimeError(
                "Structured Graph region kinds must match its Sequential values"
            )
    builder = _new_runtime_graph_builder()
    ir_nodes = []
    dispatch_count = 0
    runtime_arg_names = set()
    recording_dispatches = []
    fixed_runtime_args = {}
    lifetime_leases = []
    source_native_count = 0
    temporary_actions = []
    for _ in range(repetitions):
        for sequence, sequence_region_kind in zip(sequences, region_kinds):
            recording_dispatches.extend(
                sequence._dispatch_to(builder, region_kind=sequence_region_kind)
            )
            ir_nodes.extend(sequence._ir_nodes)
            dispatch_count += sequence._dispatch_count
            runtime_arg_names.update(sequence._recording_runtime_arg_names)
            for binding_name, value in sequence._fixed_runtime_args.items():
                existing = fixed_runtime_args.get(binding_name)
                if existing is not None and existing is not value:
                    if not (
                        isinstance(existing, (int, float))
                        and isinstance(value, (int, float))
                        and existing == value
                    ):
                        raise TaichiRuntimeError(
                            "Structured Graph regions provide conflicting "
                            f"fixed binding {binding_name!r}"
                        )
                fixed_runtime_args[binding_name] = value
            lifetime_leases.extend(sequence._lifetime_leases)
            source_native_count += sequence._source_native_count
            temporary_actions.extend(sequence._temporary_actions)
    return _CompiledCGraphNode(
        builder.compile(),
        dispatch_count,
        runtime_arg_names,
        SequentialRegion(tuple(ir_nodes), name=name),
        recording_dispatches=recording_dispatches,
        lifetime_leases=lifetime_leases,
        source_native_count=source_native_count,
        region_kind=region_kind,
        fixed_runtime_args=fixed_runtime_args,
        temporary_actions=_deduplicate_temporary_actions(temporary_actions),
    )


def _control_transfer_uses_planner(arch):
    return (
        arch == _ti_core.Arch.vulkan
        and os.environ.get("TI_GRAPH_OBSERVATION_TRANSFER_PLANNER", "1") != "0"
    )


def _structured_host_lowering(arch):
    if arch in (_ti_core.Arch.x64, _ti_core.Arch.arm64):
        return "cpu_host_control"
    return "portable_host_control"


class _CompiledWhileGraphNode:
    needs_runtime_args = True
    source_native_count = 0
    region_kind = "structured_while"

    def __init__(
        self,
        condition,
        body,
        *,
        predicate,
        control_inputs,
        carried_state,
        max_iterations,
        counter,
        status,
        chunk_size,
        masked_execution,
        lowering_mode,
        name,
    ):
        if not isinstance(condition, Sequential) or condition._dispatch_count == 0:
            raise TaichiRuntimeError(
                "Graph while condition must be a non-empty Sequential"
            )
        if not isinstance(body, Sequential) or body._dispatch_count == 0:
            raise TaichiRuntimeError("Graph while body must be a non-empty Sequential")
        if isinstance(max_iterations, bool) or not isinstance(
            max_iterations, (int, np.integer)
        ):
            raise TaichiRuntimeError("Graph while max_iterations must be an integer")
        if max_iterations < 0:
            raise TaichiRuntimeError("Graph while max_iterations must be non-negative")
        self.name = name
        self.predicate = predicate
        self.control_inputs = tuple(control_inputs)
        self.carried_state = tuple(carried_state)
        self.counter = counter
        self.status = status
        self.max_iterations = int(max_iterations)
        self.masked_execution = bool(masked_execution)
        self.lowering_mode = lowering_mode
        self.condition_dispatch_count = condition._dispatch_count
        self.body_dispatch_count = body._dispatch_count
        self.dispatch_count = self.condition_dispatch_count + self.body_dispatch_count
        required_condition = {predicate, *self.control_inputs}
        if status is not None:
            required_condition.add(status)
        missing_condition = sorted(
            required_condition.difference(condition._runtime_arg_names)
        )
        if missing_condition:
            raise TaichiRuntimeError(
                "Graph while condition does not declare control resources: "
                + ", ".join(missing_condition)
            )
        required_body = set(self.carried_state)
        if counter is not None:
            required_body.add(counter)
        missing_body = sorted(required_body.difference(body._runtime_arg_names))
        if missing_body:
            raise TaichiRuntimeError(
                "Graph while body does not declare carried resources: "
                + ", ".join(missing_body)
            )

        arch = impl.current_cfg().arch
        self.chunk_limit = min(
            self.max_iterations or 1,
            _structured_chunk_limit(arch, chunk_size, self.masked_execution),
        )
        if self.chunk_limit > 1 and self.counter is None:
            raise TaichiRuntimeError(
                "Chunked Graph while regions require a device counter"
            )
        self._condition = _compile_sequential_runtime_node(
            (condition,),
            name=f"{name}_condition",
            region_kind="while_condition",
        )
        self._chunks = {}
        chunk = 1
        while chunk <= self.chunk_limit and chunk <= self.max_iterations:
            self._chunks[chunk] = _compile_sequential_runtime_node(
                (body, condition),
                repetitions=chunk,
                name=f"{name}_body_condition_{chunk}",
                region_kinds=("while_body", "while_condition"),
            )
            chunk *= 2
        dependency_nodes = (self._condition, *self._chunks.values())
        self.temporary_actions = _merge_temporary_actions(dependency_nodes)
        self.temporary_runtime_arg_names = frozenset().union(
            *(
                node.temporary_runtime_arg_names
                for node in dependency_nodes
            )
        )
        self.fixed_runtime_args = _merge_fixed_runtime_args(dependency_nodes)
        self.recording_runtime_arg_names = frozenset().union(
            *(node.recording_runtime_arg_names for node in dependency_nodes)
        )
        self.runtime_arg_names = self.recording_runtime_arg_names.difference(
            (*self.fixed_runtime_args, *self.temporary_runtime_arg_names)
        )
        self.lifetime_leases = tuple(
            (*condition._lifetime_leases, *body._lifetime_leases)
        )
        self.source_native_count = (
            condition._source_native_count + body._source_native_count
        )
        self.snode_tree_dependencies = frozenset().union(
            *(node.snode_tree_dependencies for node in dependency_nodes)
        )
        self.snode_tree_dependency_info = frozenset().union(
            *(node.snode_tree_dependency_info for node in dependency_nodes)
        )
        self.ir_node = WhileRegion(
            predicate=predicate,
            max_iterations=self.max_iterations,
            condition=SequentialRegion(
                tuple(condition._ir_nodes), name=f"{name}_condition"
            ),
            body=SequentialRegion(tuple(body._ir_nodes), name=f"{name}_body"),
            control_inputs=self.control_inputs,
            carried_state=self.carried_state,
            counter=counter,
            status=status,
            chunk_size=self.chunk_limit,
            masked_execution=self.masked_execution,
            lowering_mode=lowering_mode,
            name=name,
        )
        self._last_report = None
        self._native_jit_cache = _ti_core.CompiledGraphJITCache()
        self._native_upgrade_eligible, self._native_upgrade_reason = (
            _cuda_while_upgrade_status(arch, lowering_mode)
        )
        if self._native_upgrade_eligible and self.counter is None:
            if lowering_mode == "native_required":
                raise TaichiRuntimeError(
                    "Native CUDA Graph while lowering requires an exact "
                    "iteration counter"
                )
            self._native_upgrade_eligible = False
            self._native_upgrade_reason = "exact_counter_required"

    def _select_chunk(self, remaining):
        return max(size for size in self._chunks if size <= remaining)

    @property
    def supports_native_submission(self):
        return (
            self.lowering_mode == "native_required"
            and self._native_upgrade_eligible
            and self.counter is not None
        )

    def run_for_submission(self, context, temporaries=None):
        if not self.supports_native_submission:
            raise TaichiRuntimeError(
                "Graph while submission requires native_required CUDA lowering"
            )
        self._last_report = None
        self._condition.run(context)
        if self.max_iterations == 0:
            return
        predicate_object = context.runtime_args()[self.predicate]
        predicate_ndarray = getattr(predicate_object, "arr", None)
        if predicate_ndarray is None:
            raise TaichiRuntimeError(
                "Native CUDA Graph while submission requires an ndarray predicate"
            )
        submitted = self._chunks[1].compiled_graph.jit_run_bounded_cuda_cached(
            context.compile_config(),
            context.flattened_args(self.recording_runtime_arg_names),
            self._native_jit_cache,
            predicate_ndarray,
            self.max_iterations,
            True,
        )
        if not submitted:
            raise TaichiRuntimeError(
                "Native CUDA Graph while submission became unavailable; "
                "synchronous fallback is disabled"
            )

    def run(self, context, temporaries=None):
        runtime_args = context.runtime_args()
        predicate_object = runtime_args[self.predicate]
        counter_object = (
            runtime_args[self.counter] if self.counter is not None else None
        )
        status_object = (
            runtime_args[self.status] if self.status is not None else None
        )
        observations = []
        predicate_values = []
        counter_values = []
        status_values = []
        chunks = []
        executed = 0
        observation_batches = 0
        observation_scalar_count = 0
        device_to_host_bytes = 0
        program = impl.get_runtime().prog
        arch = impl.current_cfg().arch
        use_transfer_planner = _control_transfer_uses_planner(arch)
        transfer_before = program._graph_observation_staging_stats()

        def observe_control(boundary):
            nonlocal observation_batches
            nonlocal observation_scalar_count
            nonlocal device_to_host_bytes
            values = [predicate_object]
            names = [self.predicate]
            if counter_object is not None:
                values.append(counter_object)
                names.append(self.counter)
            if status_object is not None:
                values.append(status_object)
                names.append(self.status)
            observed, byte_count = _control_scalar_values(
                values,
                names,
                use_transfer_planner=use_transfer_planner,
            )
            observation_batches += 1
            observation_scalar_count += len(observed)
            device_to_host_bytes += byte_count
            observations.append(boundary)
            predicate_values.append(observed[0])
            next_index = 1
            if counter_object is not None:
                counter_values.append(observed[next_index])
                next_index += 1
            if status_object is not None:
                status_values.append(observed[next_index])
            return observed[0]

        self._condition.run(context)
        predicate_value = observe_control(0)
        initial_counter = counter_values[-1] if counter_object is not None else None
        initial_status = status_values[-1] if status_object is not None else None
        active = predicate_value != 0
        native_selected = False
        native_reason = self._native_upgrade_reason
        if active and self.max_iterations > 0 and self._native_upgrade_eligible:
            predicate_ndarray = getattr(predicate_object, "arr", None)
            if predicate_ndarray is None:
                native_reason = "predicate_ndarray_required"
            else:
                native_selected = self._chunks[
                    1
                ].compiled_graph.jit_run_bounded_cuda_cached(
                    context.compile_config(),
                    context.flattened_args(self.recording_runtime_arg_names),
                    self._native_jit_cache,
                    predicate_ndarray,
                    self.max_iterations,
                    True,
                )
                native_reason = (
                    "selected" if native_selected else "conditional_capture_fallback"
                )
            if not native_selected and self.lowering_mode == "native_required":
                raise TaichiRuntimeError(
                    "Graph while native CUDA lowering failed: " f"{native_reason}"
                )
        if native_selected:
            predicate_value = observe_control(-1)
            logical_native = counter_values[-1] - initial_counter
            if logical_native < 0 or logical_native > self.max_iterations:
                raise TaichiRuntimeError(
                    "Graph while counter left its iteration budget"
                )
            active = predicate_value != 0
            executed = self.max_iterations if active else logical_native
            observations[-1] = executed
            if executed:
                chunks.append(executed)

        while not native_selected and active and executed < self.max_iterations:
            chunk = self._select_chunk(self.max_iterations - executed)
            self._chunks[chunk].run(context)
            executed += chunk
            chunks.append(chunk)
            predicate_value = observe_control(executed)
            active = predicate_value != 0

        if not observations or observations[-1] != executed:
            observe_control(executed)
        final_counter = counter_values[-1] if counter_object is not None else None
        final_status = status_values[-1] if status_object is not None else None
        if final_counter is not None:
            logical = final_counter - initial_counter
            if logical < 0 or logical > executed:
                raise TaichiRuntimeError(
                    "Graph while counter must increase by no more than the "
                    "executed iteration count"
                )
        else:
            logical = executed
        lowering = (
            "cuda_conditional_graph"
            if native_selected
            else (
                "cpu_host_loop"
                if arch in (_ti_core.Arch.x64, _ti_core.Arch.arm64)
                else (
                    "portable_chunk_replay"
                    if self.chunk_limit > 1
                    else "portable_exact_replay"
                )
            )
        )
        transfer_after = program._graph_observation_staging_stats()

        def transfer_delta(name):
            return int(transfer_after[name]) - int(transfer_before[name])

        self._last_report = GraphWhileReport(
            name=self.name,
            backend=_backend_name(_ti_core.arch_name(arch)),
            lowering=lowering,
            max_iterations=self.max_iterations,
            logical_iterations=logical,
            executed_iterations=executed,
            overshoot_iterations=executed - logical,
            observation_boundaries=tuple(observations),
            predicate_values=tuple(predicate_values),
            counter_values=tuple(counter_values),
            status_resource=self.status,
            status_values=tuple(status_values),
            chunk_sizes=tuple(chunks),
            observation_batches=observation_batches,
            observation_scalar_count=observation_scalar_count,
            device_to_host_bytes=device_to_host_bytes,
            initial_counter=initial_counter,
            final_counter=final_counter,
            initial_status=initial_status,
            final_status=final_status,
            native_upgrade_eligible=self._native_upgrade_eligible,
            native_upgrade_reason=native_reason,
            persistent_staging_bytes=int(transfer_after["persistent_bytes"]),
            staging_allocations=transfer_delta("allocations"),
            staging_reuses=transfer_delta("reuses"),
            packed_observation_batches=transfer_delta("packed_batches"),
            direct_observation_batches=(
                transfer_delta("direct_batches")
                if use_transfer_planner
                else observation_batches
            ),
            staging_fallback_batches=transfer_delta("fallback_batches"),
            packed_observation_bytes=transfer_delta("packed_payload_bytes"),
            condition_dispatch_count=self.condition_dispatch_count,
            body_dispatch_count=self.body_dispatch_count,
            control_inputs=self.control_inputs,
            carried_state=self.carried_state,
        )

    def invalidate_runtime(self):
        self._native_jit_cache.clear_runtime_state()
        self._condition.invalidate_runtime()
        for node in self._chunks.values():
            node.invalidate_runtime()

    @property
    def debug_graph_stats(self):
        chunk_stats = tuple(node.debug_graph_stats for node in self._chunks.values())
        condition_stats = (self._condition.debug_graph_stats,)
        if not self._native_upgrade_eligible:
            return (*condition_stats, *chunk_stats)
        return (
            self._native_jit_cache._debug_graph_stats(),
            *condition_stats,
            *chunk_stats,
        )

    @property
    def last_report(self):
        return self._last_report

    @property
    def debug_info(self):
        return {
            "kind": "structured_while",
            "name": self.name,
            "condition_dispatch_count": self.condition_dispatch_count,
            "body_dispatch_count": self.body_dispatch_count,
            "max_iterations": self.max_iterations,
            "chunk_limit": self.chunk_limit,
            "control_input_count": len(self.control_inputs),
            "carried_state_count": len(self.carried_state),
            "has_status": self.status is not None,
            "masked_execution": self.masked_execution,
            "lowering_mode": self.lowering_mode,
            "native_upgrade_eligible": self._native_upgrade_eligible,
        }


class _CompiledIfGraphNode:
    needs_runtime_args = True
    source_native_count = 0
    region_kind = "structured_if"

    def __init__(
        self,
        condition,
        then_region,
        else_region,
        *,
        predicate,
        control_inputs,
        lowering_mode,
        name,
    ):
        if not isinstance(condition, Sequential) or condition._dispatch_count == 0:
            raise TaichiRuntimeError(
                "Graph if condition must be a non-empty Sequential"
            )
        if not isinstance(then_region, Sequential) or then_region._dispatch_count == 0:
            raise TaichiRuntimeError(
                "Graph if then_region must be a non-empty Sequential"
            )
        if else_region is not None and (
            not isinstance(else_region, Sequential) or else_region._dispatch_count == 0
        ):
            raise TaichiRuntimeError(
                "Graph if else_region must be a non-empty Sequential"
            )
        self.name = name
        self.predicate = predicate
        self.control_inputs = tuple(control_inputs)
        self.lowering_mode = lowering_mode
        required = {predicate, *self.control_inputs}
        missing = sorted(required.difference(condition._runtime_arg_names))
        if missing:
            raise TaichiRuntimeError(
                "Graph if condition does not declare control resources: "
                + ", ".join(missing)
            )
        self._condition = _compile_sequential_runtime_node(
            (condition,),
            name=f"{name}_condition",
            region_kind="if_condition",
        )
        self._then = _compile_sequential_runtime_node(
            (then_region,), name=f"{name}_then", region_kind="if_branch"
        )
        self._else = (
            None
            if else_region is None
            else _compile_sequential_runtime_node(
                (else_region,), name=f"{name}_else", region_kind="if_branch"
            )
        )
        branch_sequences = (
            (then_region,)
            if else_region is None
            else (then_region, else_region)
        )
        self._native_branches = _compile_sequential_runtime_node(
            branch_sequences,
            name=f"{name}_native_branches",
            region_kind="if_branch",
            region_kinds=("if_branch",) * len(branch_sequences),
        )
        self._native_branch_dispatch_counts = tuple(
            region._dispatch_count for region in branch_sequences
        )
        arch = impl.current_cfg().arch
        self._native_upgrade_eligible, self._native_upgrade_reason = (
            _cuda_branch_upgrade_status(arch, lowering_mode, "if")
        )
        self._native_jit_cache = _ti_core.CompiledGraphJITCache()
        nodes = tuple(
            node for node in (self._condition, self._then, self._else) if node
        )
        self.temporary_actions = _merge_temporary_actions(nodes)
        self.temporary_runtime_arg_names = frozenset().union(
            *(node.temporary_runtime_arg_names for node in nodes)
        )
        self.fixed_runtime_args = _merge_fixed_runtime_args(nodes)
        self.recording_runtime_arg_names = frozenset().union(
            *(node.recording_runtime_arg_names for node in nodes)
        )
        self.runtime_arg_names = self.recording_runtime_arg_names.difference(
            (*self.fixed_runtime_args, *self.temporary_runtime_arg_names)
        )
        sequences = tuple(
            region
            for region in (condition, then_region, else_region)
            if region is not None
        )
        self.lifetime_leases = tuple(
            lease for region in sequences for lease in region._lifetime_leases
        )
        self.source_native_count = sum(
            region._source_native_count for region in sequences
        )
        self.snode_tree_dependencies = frozenset().union(
            *(node.snode_tree_dependencies for node in nodes)
        )
        self.snode_tree_dependency_info = frozenset().union(
            *(node.snode_tree_dependency_info for node in nodes)
        )
        self.condition_dispatch_count = condition._dispatch_count
        self.then_dispatch_count = then_region._dispatch_count
        self.else_dispatch_count = (
            0 if else_region is None else else_region._dispatch_count
        )
        self.dispatch_count = (
            self.condition_dispatch_count
            + self.then_dispatch_count
            + self.else_dispatch_count
        )
        self.ir_node = IfRegion(
            predicate=predicate,
            condition=SequentialRegion(
                tuple(condition._ir_nodes), name=f"{name}_condition"
            ),
            then_region=SequentialRegion(
                tuple(then_region._ir_nodes), name=f"{name}_then"
            ),
            else_region=(
                None
                if else_region is None
                else SequentialRegion(tuple(else_region._ir_nodes), name=f"{name}_else")
            ),
            control_inputs=self.control_inputs,
            name=name,
        )
        self._last_report = None
        self._pending_report = None

    @property
    def supports_native_submission(self):
        return (
            self.lowering_mode == "native_required"
            and self._native_upgrade_eligible
        )

    def _run_native_branch(self, context):
        predicate_object = context.runtime_args()[self.predicate]
        predicate_ndarray = getattr(predicate_object, "arr", None)
        if predicate_ndarray is None:
            return False
        return self._native_branches.compiled_graph.jit_run_conditional_cuda_cached(
            context.compile_config(),
            context.flattened_args(
                self._native_branches.recording_runtime_arg_names
            ),
            self._native_jit_cache,
            predicate_ndarray,
            self._native_branch_dispatch_counts,
            0,
            -1,
        )

    def run_for_submission(self, context, temporaries=None):
        if not self.supports_native_submission:
            raise TaichiRuntimeError(
                "Graph if submission requires native_required CUDA lowering"
            )
        self._last_report = None
        self._pending_report = None
        self._condition.run(context)
        if not self._run_native_branch(context):
            raise TaichiRuntimeError(
                "Native CUDA Graph if submission became unavailable; "
                "synchronous fallback is disabled"
            )

    def run(self, context, temporaries=None):
        self._condition.run(context)
        runtime_args = context.runtime_args()
        arch = impl.current_cfg().arch
        native_selected = False
        native_reason = self._native_upgrade_reason
        if self._native_upgrade_eligible:
            native_selected = self._run_native_branch(context)
            native_reason = (
                "selected" if native_selected else "conditional_capture_fallback"
            )
            if not native_selected and self.lowering_mode == "native_required":
                raise TaichiRuntimeError(
                    f"Graph if native CUDA lowering failed: {native_reason}"
                )
        if native_selected:
            self._last_report = None
            self._pending_report = (runtime_args[self.predicate], arch)
            return
        self._pending_report = None
        observed, byte_count = _control_scalar_values(
            [runtime_args[self.predicate]],
            [self.predicate],
            use_transfer_planner=_control_transfer_uses_planner(arch),
        )
        predicate_value = observed[0]
        if predicate_value != 0:
            selected = "then"
            selected_dispatches = self.then_dispatch_count
            if not native_selected:
                self._then.run(context)
        elif self._else is not None:
            selected = "else"
            selected_dispatches = self.else_dispatch_count
            if not native_selected:
                self._else.run(context)
        else:
            selected = "none"
            selected_dispatches = 0
        self._last_report = GraphBranchReport(
            name=self.name,
            backend=_backend_name(_ti_core.arch_name(arch)),
            kind="if",
            lowering=(
                "cuda_conditional_graph"
                if native_selected
                else _structured_host_lowering(arch)
            ),
            selector_resource=self.predicate,
            selector_value=predicate_value,
            selected_branch=selected,
            observation_scalar_count=1,
            device_to_host_bytes=byte_count,
            condition_dispatch_count=self.condition_dispatch_count,
            branch_dispatch_count=selected_dispatches,
            control_inputs=self.control_inputs,
        )

    def materialize_pending_report(self):
        if self._pending_report is None:
            return
        predicate_object, arch = self._pending_report
        observed, byte_count = _control_scalar_values(
            [predicate_object],
            [self.predicate],
            use_transfer_planner=_control_transfer_uses_planner(arch),
        )
        predicate_value = observed[0]
        if predicate_value != 0:
            selected = "then"
            selected_dispatches = self.then_dispatch_count
        elif self._else is not None:
            selected = "else"
            selected_dispatches = self.else_dispatch_count
        else:
            selected = "none"
            selected_dispatches = 0
        self._last_report = GraphBranchReport(
            name=self.name,
            backend=_backend_name(_ti_core.arch_name(arch)),
            kind="if",
            lowering="cuda_conditional_graph",
            selector_resource=self.predicate,
            selector_value=predicate_value,
            selected_branch=selected,
            observation_scalar_count=1,
            device_to_host_bytes=byte_count,
            condition_dispatch_count=self.condition_dispatch_count,
            branch_dispatch_count=selected_dispatches,
            control_inputs=self.control_inputs,
        )
        self._pending_report = None

    def invalidate_runtime(self):
        self._pending_report = None
        self._native_jit_cache.clear_runtime_state()
        self._condition.invalidate_runtime()
        self._then.invalidate_runtime()
        if self._else is not None:
            self._else.invalidate_runtime()

    @property
    def debug_graph_stats(self):
        nodes = tuple(
            node for node in (self._condition, self._then, self._else) if node
        )
        child_stats = tuple(node.debug_graph_stats for node in nodes)
        if not self._native_upgrade_eligible:
            return child_stats
        return (self._native_jit_cache._debug_graph_stats(), *child_stats)

    @property
    def last_report(self):
        return self._last_report

    @property
    def debug_info(self):
        return {
            "kind": "structured_if",
            "name": self.name,
            "condition_dispatch_count": self.condition_dispatch_count,
            "then_dispatch_count": self.then_dispatch_count,
            "else_dispatch_count": self.else_dispatch_count,
            "control_input_count": len(self.control_inputs),
            "lowering_mode": self.lowering_mode,
            "native_upgrade_eligible": self._native_upgrade_eligible,
        }


class _CompiledSwitchGraphNode:
    needs_runtime_args = True
    source_native_count = 0
    region_kind = "structured_switch"

    def __init__(
        self,
        condition,
        branches,
        default_region,
        *,
        selector,
        control_inputs,
        lowering_mode,
        name,
    ):
        if not isinstance(condition, Sequential) or condition._dispatch_count == 0:
            raise TaichiRuntimeError(
                "Graph switch condition must be a non-empty Sequential"
            )
        branches = tuple(branches)
        if not branches or any(
            not isinstance(branch, Sequential) or branch._dispatch_count == 0
            for branch in branches
        ):
            raise TaichiRuntimeError(
                "Graph switch requires non-empty Sequential branches"
            )
        if default_region is not None and (
            not isinstance(default_region, Sequential)
            or default_region._dispatch_count == 0
        ):
            raise TaichiRuntimeError(
                "Graph switch default_region must be a non-empty Sequential"
            )
        self.name = name
        self.selector = selector
        self.control_inputs = tuple(control_inputs)
        self.lowering_mode = lowering_mode
        required = {selector, *self.control_inputs}
        missing = sorted(required.difference(condition._runtime_arg_names))
        if missing:
            raise TaichiRuntimeError(
                "Graph switch condition does not declare control resources: "
                + ", ".join(missing)
            )
        self._condition = _compile_sequential_runtime_node(
            (condition,),
            name=f"{name}_condition",
            region_kind="switch_condition",
        )
        self._branches = tuple(
            _compile_sequential_runtime_node(
                (branch,),
                name=f"{name}_case_{index}",
                region_kind="switch_branch",
            )
            for index, branch in enumerate(branches)
        )
        self._default = (
            None
            if default_region is None
            else _compile_sequential_runtime_node(
                (default_region,),
                name=f"{name}_default",
                region_kind="switch_branch",
            )
        )
        branch_sequences = branches
        if default_region is not None:
            branch_sequences = (*branch_sequences, default_region)
        self._native_branches = _compile_sequential_runtime_node(
            branch_sequences,
            name=f"{name}_native_branches",
            region_kind="switch_branch",
            region_kinds=("switch_branch",) * len(branch_sequences),
        )
        self._native_branch_dispatch_counts = tuple(
            region._dispatch_count for region in branch_sequences
        )
        self._native_default_branch = (
            -1 if default_region is None else len(branches)
        )
        arch = impl.current_cfg().arch
        self._native_upgrade_eligible, self._native_upgrade_reason = (
            _cuda_branch_upgrade_status(arch, lowering_mode, "switch")
        )
        self._native_jit_cache = _ti_core.CompiledGraphJITCache()
        nodes = (self._condition, *self._branches)
        if self._default is not None:
            nodes = (*nodes, self._default)
        self.temporary_actions = _merge_temporary_actions(nodes)
        self.temporary_runtime_arg_names = frozenset().union(
            *(node.temporary_runtime_arg_names for node in nodes)
        )
        self.fixed_runtime_args = _merge_fixed_runtime_args(nodes)
        self.recording_runtime_arg_names = frozenset().union(
            *(node.recording_runtime_arg_names for node in nodes)
        )
        self.runtime_arg_names = self.recording_runtime_arg_names.difference(
            (*self.fixed_runtime_args, *self.temporary_runtime_arg_names)
        )
        sequences = (condition, *branches)
        if default_region is not None:
            sequences = (*sequences, default_region)
        self.lifetime_leases = tuple(
            lease for region in sequences for lease in region._lifetime_leases
        )
        self.source_native_count = sum(
            region._source_native_count for region in sequences
        )
        self.snode_tree_dependencies = frozenset().union(
            *(node.snode_tree_dependencies for node in nodes)
        )
        self.snode_tree_dependency_info = frozenset().union(
            *(node.snode_tree_dependency_info for node in nodes)
        )
        self.condition_dispatch_count = condition._dispatch_count
        self.branch_dispatch_counts = tuple(
            branch._dispatch_count for branch in branches
        )
        self.default_dispatch_count = (
            0 if default_region is None else default_region._dispatch_count
        )
        self.dispatch_count = (
            self.condition_dispatch_count
            + sum(self.branch_dispatch_counts)
            + self.default_dispatch_count
        )
        self.ir_node = SwitchRegion(
            selector=selector,
            condition=SequentialRegion(
                tuple(condition._ir_nodes), name=f"{name}_condition"
            ),
            branches=tuple(
                SequentialRegion(tuple(branch._ir_nodes), name=f"{name}_case_{index}")
                for index, branch in enumerate(branches)
            ),
            default_region=(
                None
                if default_region is None
                else SequentialRegion(
                    tuple(default_region._ir_nodes), name=f"{name}_default"
                )
            ),
            control_inputs=self.control_inputs,
            name=name,
        )
        self._last_report = None
        self._pending_report = None

    @property
    def supports_native_submission(self):
        return (
            self.lowering_mode == "native_required"
            and self._native_upgrade_eligible
        )

    def _run_native_branch(self, context):
        selector_object = context.runtime_args()[self.selector]
        selector_ndarray = getattr(selector_object, "arr", None)
        if selector_ndarray is None:
            return False
        return self._native_branches.compiled_graph.jit_run_conditional_cuda_cached(
            context.compile_config(),
            context.flattened_args(
                self._native_branches.recording_runtime_arg_names
            ),
            self._native_jit_cache,
            selector_ndarray,
            self._native_branch_dispatch_counts,
            2,
            self._native_default_branch,
        )

    def run_for_submission(self, context, temporaries=None):
        if not self.supports_native_submission:
            raise TaichiRuntimeError(
                "Graph switch submission requires native_required CUDA lowering"
            )
        self._last_report = None
        self._pending_report = None
        self._condition.run(context)
        if not self._run_native_branch(context):
            raise TaichiRuntimeError(
                "Native CUDA Graph switch submission became unavailable; "
                "synchronous fallback is disabled"
            )

    def run(self, context, temporaries=None):
        self._condition.run(context)
        runtime_args = context.runtime_args()
        arch = impl.current_cfg().arch
        native_selected = False
        native_reason = self._native_upgrade_reason
        if self._native_upgrade_eligible:
            native_selected = self._run_native_branch(context)
            native_reason = (
                "selected" if native_selected else "conditional_capture_fallback"
            )
            if not native_selected and self.lowering_mode == "native_required":
                raise TaichiRuntimeError(
                    f"Graph switch native CUDA lowering failed: {native_reason}"
                )
        if native_selected:
            self._last_report = None
            self._pending_report = (runtime_args[self.selector], arch)
            return
        self._pending_report = None
        observed, byte_count = _control_scalar_values(
            [runtime_args[self.selector]],
            [self.selector],
            use_transfer_planner=_control_transfer_uses_planner(arch),
        )
        selector_value = int(observed[0])
        if 0 <= selector_value < len(self._branches):
            selected = f"case_{selector_value}"
            selected_dispatches = self.branch_dispatch_counts[selector_value]
            if not native_selected:
                self._branches[selector_value].run(context)
        elif self._default is not None:
            selected = "default"
            selected_dispatches = self.default_dispatch_count
            if not native_selected:
                self._default.run(context)
        else:
            selected = "none"
            selected_dispatches = 0
        self._last_report = GraphBranchReport(
            name=self.name,
            backend=_backend_name(_ti_core.arch_name(arch)),
            kind="switch",
            lowering=(
                "cuda_conditional_graph"
                if native_selected
                else _structured_host_lowering(arch)
            ),
            selector_resource=self.selector,
            selector_value=selector_value,
            selected_branch=selected,
            observation_scalar_count=1,
            device_to_host_bytes=byte_count,
            condition_dispatch_count=self.condition_dispatch_count,
            branch_dispatch_count=selected_dispatches,
            control_inputs=self.control_inputs,
        )

    def materialize_pending_report(self):
        if self._pending_report is None:
            return
        selector_object, arch = self._pending_report
        observed, byte_count = _control_scalar_values(
            [selector_object],
            [self.selector],
            use_transfer_planner=_control_transfer_uses_planner(arch),
        )
        selector_value = int(observed[0])
        if 0 <= selector_value < len(self._branches):
            selected = f"case_{selector_value}"
            selected_dispatches = self.branch_dispatch_counts[selector_value]
        elif self._default is not None:
            selected = "default"
            selected_dispatches = self.default_dispatch_count
        else:
            selected = "none"
            selected_dispatches = 0
        self._last_report = GraphBranchReport(
            name=self.name,
            backend=_backend_name(_ti_core.arch_name(arch)),
            kind="switch",
            lowering="cuda_conditional_graph",
            selector_resource=self.selector,
            selector_value=selector_value,
            selected_branch=selected,
            observation_scalar_count=1,
            device_to_host_bytes=byte_count,
            condition_dispatch_count=self.condition_dispatch_count,
            branch_dispatch_count=selected_dispatches,
            control_inputs=self.control_inputs,
        )
        self._pending_report = None

    def invalidate_runtime(self):
        self._pending_report = None
        self._native_jit_cache.clear_runtime_state()
        self._condition.invalidate_runtime()
        for branch in self._branches:
            branch.invalidate_runtime()
        if self._default is not None:
            self._default.invalidate_runtime()

    @property
    def debug_graph_stats(self):
        nodes = (self._condition, *self._branches)
        if self._default is not None:
            nodes = (*nodes, self._default)
        child_stats = tuple(node.debug_graph_stats for node in nodes)
        if not self._native_upgrade_eligible:
            return child_stats
        return (self._native_jit_cache._debug_graph_stats(), *child_stats)

    @property
    def last_report(self):
        return self._last_report

    @property
    def debug_info(self):
        return {
            "kind": "structured_switch",
            "name": self.name,
            "condition_dispatch_count": self.condition_dispatch_count,
            "branch_dispatch_counts": self.branch_dispatch_counts,
            "default_dispatch_count": self.default_dispatch_count,
            "control_input_count": len(self.control_inputs),
            "lowering_mode": self.lowering_mode,
            "native_upgrade_eligible": self._native_upgrade_eligible,
        }


def _recordable_backend_dispatches(node, backend):
    if isinstance(node, _CompiledCGraphNode):
        if (
            node.dispatch_count == 0
            or len(node.recording_dispatches) != node.dispatch_count
        ):
            return None
        return node.recording_dispatches
    if not isinstance(node, _CompiledNativeGraphNode):
        return None
    recorder = node.recordable_action
    if recorder is None or not recorder.supports_backend(backend):
        return None
    dispatches = tuple(recorder.dispatches)
    return dispatches or None


def _merge_fixed_runtime_args(nodes):
    merged = {}
    for node in nodes:
        for name, value in getattr(node, "fixed_runtime_args", {}).items():
            existing = merged.get(name)
            if existing is not None and existing is not value:
                if not (
                    isinstance(existing, (int, float))
                    and isinstance(value, (int, float))
                    and existing == value
                ):
                    raise TaichiRuntimeError(
                        "Recordable actions provide conflicting fixed binding "
                        f"{name!r}"
                    )
            merged[name] = value
    return merged


def _deduplicate_temporary_actions(actions):
    merged = []
    seen = set()
    for action in actions:
        identity = id(action)
        if identity in seen:
            continue
        seen.add(identity)
        merged.append(action)
    return tuple(merged)


def _merge_temporary_actions(nodes):
    return _deduplicate_temporary_actions(
        action
        for node in nodes
        for action in getattr(node, "temporary_actions", ())
    )


def _lower_mixed_backend_regions(nodes):
    backend = _backend_name(_ti_core.arch_name(impl.current_cfg().arch))
    nodes = tuple(nodes)
    lowered = []
    mixed_region_count = 0
    lowered_native_count = 0
    cursor = 0
    while cursor < len(nodes):
        if _recordable_backend_dispatches(nodes[cursor], backend) is None:
            lowered.append(nodes[cursor])
            cursor += 1
            continue

        end = cursor
        region = []
        while end < len(nodes):
            dispatches = _recordable_backend_dispatches(nodes[end], backend)
            if dispatches is None:
                break
            region.append((nodes[end], dispatches))
            end += 1

        has_cgraph = any(isinstance(node, _CompiledCGraphNode) for node, _ in region)
        has_native = any(
            isinstance(node, _CompiledNativeGraphNode) for node, _ in region
        )
        if not has_native:
            lowered.extend(node for node, _ in region)
            cursor = end
            continue

        builder = _new_runtime_graph_builder()
        recording_dispatches = []
        runtime_arg_names = set()
        fixed_runtime_args = _merge_fixed_runtime_args(node for node, _ in region)
        ir_children = []
        lifetime_leases = []
        region_native_count = 0
        temporary_actions = []
        for node, dispatches in region:
            for kernel, args in dispatches:
                builder.dispatch(kernel, args)
                recording_dispatches.append((kernel, tuple(args)))
            runtime_arg_names.update(
                getattr(
                    node,
                    "recording_runtime_arg_names",
                    node.runtime_arg_names,
                )
            )
            if isinstance(node.ir_node, SequentialRegion):
                ir_children.extend(node.ir_node.children)
            else:
                ir_children.append(node.ir_node)
            lifetime_leases.extend(getattr(node, "lifetime_leases", ()))
            temporary_actions.extend(getattr(node, "temporary_actions", ()))
            region_native_count += getattr(node, "source_native_count", 0)

        lowered.append(
            _CompiledCGraphNode(
                builder.compile(),
                len(recording_dispatches),
                runtime_arg_names,
                SequentialRegion(
                    tuple(ir_children),
                    name=(
                        "mixed_backend_region"
                        if has_cgraph
                        else "recordable_provider_region"
                    ),
                ),
                recording_dispatches=recording_dispatches,
                lifetime_leases=lifetime_leases,
                source_native_count=region_native_count,
                region_kind=(
                    "mixed_cgraph_native" if has_cgraph else "recordable_provider"
                ),
                fixed_runtime_args=fixed_runtime_args,
                temporary_actions=_deduplicate_temporary_actions(
                    temporary_actions
                ),
            )
        )
        mixed_region_count += 1
        lowered_native_count += region_native_count
        cursor = end

    total_native_count = sum(getattr(node, "source_native_count", 0) for node in nodes)
    return tuple(lowered), {
        "backend": backend,
        "input_segments": len(nodes),
        "output_segments": len(lowered),
        "mixed_backend_regions": mixed_region_count,
        "lowered_native_nodes": lowered_native_count,
        "opaque_native_nodes": total_native_count - lowered_native_count,
    }


class _GraphSpec:
    def __init__(self, nodes, aot_graph_builder=None, aot_compiled_graph=None):
        source_nodes = tuple(nodes)
        self.pre_optimization_ir_root = SequentialRegion(
            tuple(node.ir_node for node in source_nodes), name="graph"
        )
        self.pre_optimization_ir_analysis = analyze_graph_ir(
            self.pre_optimization_ir_root
        )
        self.temporary_memory_plan = plan_temporary_memory(
            self.pre_optimization_ir_root
        )
        self.nodes, self.optimization = _lower_mixed_backend_regions(source_nodes)
        applied_groups = sum(
            getattr(node, "composer_applied_groups", 0) for node in self.nodes
        )
        lowering_available = any(
            getattr(node, "composer_lowering_available", False) for node in self.nodes
        )
        self.fusion_plan = analyze_elementwise_fusion(
            self.pre_optimization_ir_root,
            applied_groups=applied_groups,
            lowering_available=lowering_available,
        )
        self._aot_graph_builder = aot_graph_builder
        self._aot_compiled_graph = aot_compiled_graph
        self.needs_runtime_args = any(n.needs_runtime_args for n in self.nodes)
        self.dispatch_count = sum(getattr(n, "dispatch_count", 0) for n in self.nodes)
        self.native_count = sum(
            getattr(n, "source_native_count", 0) for n in self.nodes
        )
        self.structured_control_count = sum(
            isinstance(
                n,
                (
                    _CompiledWhileGraphNode,
                    _CompiledIfGraphNode,
                    _CompiledSwitchGraphNode,
                ),
            )
            for n in self.nodes
        )
        self.observation_count = sum(
            isinstance(n, _CompiledObservationGraphNode) for n in self.nodes
        )
        self.temporary_actions = _merge_temporary_actions(self.nodes)
        self._temporary_binding_cache = {}
        self.temporary_runtime_arg_names = frozenset().union(
            *(
                frozenset(action.temporary_bindings)
                for action in self.temporary_actions
            )
        )
        self.fixed_runtime_args = _merge_fixed_runtime_args(self.nodes)
        lifetime_leases = []
        seen_lifetime_leases = set()
        for node in self.nodes:
            for lease in getattr(node, "lifetime_leases", ()):
                identity = id(lease)
                if identity not in seen_lifetime_leases:
                    seen_lifetime_leases.add(identity)
                    lifetime_leases.append(lease)
        self.lifetime_leases = tuple(lifetime_leases)
        all_runtime_arg_names = frozenset().union(
            *(
                getattr(
                    node,
                    "recording_runtime_arg_names",
                    node.runtime_arg_names,
                )
                for node in self.nodes
            )
        )
        self.runtime_arg_names = all_runtime_arg_names.difference(
            (*self.fixed_runtime_args, *self.temporary_runtime_arg_names)
        )
        self.snode_tree_dependencies = frozenset().union(
            *(n.snode_tree_dependencies for n in self.nodes)
        )
        self.snode_tree_dependency_info = frozenset().union(
            *(n.snode_tree_dependency_info for n in self.nodes)
        )
        self.repeat_count = 0
        self.ir_root = SequentialRegion(
            tuple(node.ir_node for node in self.nodes), name="graph"
        )
        self.ir_analysis = analyze_graph_ir(self.ir_root)

    @property
    def supports_native_structured_submission(self):
        structured_nodes = [
            node
            for node in self.nodes
            if isinstance(
                node,
                (
                    _CompiledWhileGraphNode,
                    _CompiledIfGraphNode,
                    _CompiledSwitchGraphNode,
                ),
            )
        ]
        return bool(structured_nodes) and all(
            node.supports_native_submission
            for node in structured_nodes
        )

    def validate_lifetime_leases(self):
        for lease in self.lifetime_leases:
            validate = getattr(lease, "validate_graph_lifetime", None)
            if validate is not None:
                validate()

    def bind_runtime_args(self, args, temporaries=None):
        bound = args
        owners = {}
        for lease in self.lifetime_leases:
            bind = getattr(lease, "bind_graph_arguments", None)
            if bind is None:
                continue
            replacements = bind(args)
            if not replacements:
                continue
            if not isinstance(replacements, dict):
                raise TaichiRuntimeError(
                    "Graph provider argument bindings must be a dict"
                )
            if bound is args:
                bound = dict(args)
            for name, value in replacements.items():
                if name not in self.runtime_arg_names:
                    raise TaichiRuntimeError(
                        f"Graph provider attempted to bind unknown argument {name!r}"
                    )
                previous = owners.get(name)
                if previous is not None and bound[name] is not value:
                    raise TaichiRuntimeError(
                        "Graph providers produced conflicting argument "
                        f"bindings for {name!r}"
                    )
                bound[name] = value
                owners[name] = lease
        temporary_args = self.bind_temporary_args(temporaries)
        if temporary_args:
            if bound is args:
                bound = dict(args)
            for name, value in temporary_args.items():
                if name in bound:
                    raise TaichiRuntimeError(
                        "Graph temporary binding collides with runtime argument "
                        f"{name!r}"
                    )
                bound[name] = value
        return bound

    def bind_temporary_args(self, temporaries):
        if not self.temporary_actions:
            return {}
        if temporaries is None:
            raise TaichiRuntimeError(
                "Recordable Graph temporaries were not materialized"
            )
        cache_key = tuple(
            sorted(
                (
                    name,
                    id(binding.storage),
                    binding.offset,
                    binding.bytes,
                    binding.alignment,
                    binding.slot,
                )
                for name, binding in temporaries.items()
            )
        )
        cached = self._temporary_binding_cache.get(cache_key)
        if cached is not None:
            return cached
        resolved = {}
        owners = {}
        for action in self.temporary_actions:
            declarations = dict(action.temporary_bindings)
            required = set(declarations.values())
            missing = sorted(required.difference(temporaries))
            if missing:
                raise TaichiRuntimeError(
                    "Recordable action temporary requirements were not "
                    "materialized: " + ", ".join(missing)
                )
            provided = action.bind_graph_temporaries(
                {name: temporaries[name] for name in required}
            )
            if provided is None:
                raise TaichiRuntimeError(
                    "Recordable action rejected the active Graph temporary slot"
                )
            if not isinstance(provided, Mapping):
                raise TaichiRuntimeError(
                    "Recordable action temporary bindings must be a mapping"
                )
            provided = dict(provided)
            if provided.keys() != declarations.keys():
                raise TaichiRuntimeError(
                    "Recordable action returned unexpected temporary symbols"
                )
            for name, value in provided.items():
                if value is None:
                    raise TaichiRuntimeError(
                        f"Recordable action returned no storage for {name!r}"
                    )
                previous = owners.get(name)
                if previous is not None and resolved[name] is not value:
                    raise TaichiRuntimeError(
                        "Recordable actions produced conflicting temporary "
                        f"binding {name!r}"
                    )
                resolved[name] = value
                owners[name] = action
        if resolved.keys() != self.temporary_runtime_arg_names:
            raise TaichiRuntimeError(
                "Recordable action temporary binding coverage changed"
            )
        self._temporary_binding_cache[cache_key] = resolved
        return resolved

    def validate_runtime_args(self, args, entrypoint="Graph.run"):
        if not isinstance(args, dict):
            raise TaichiRuntimeError(
                f"{entrypoint}() expects a dict of runtime arguments, got {type(args)}"
            )
        if args.keys() == self.runtime_arg_names:
            for lease in self.lifetime_leases:
                validate = getattr(lease, "validate_graph_bindings", None)
                if validate is not None:
                    validate(args)
            return

        missing = sorted(self.runtime_arg_names.difference(args.keys()))
        unexpected = sorted(args.keys() - self.runtime_arg_names)
        details = []
        if missing:
            details.append(f"Missing graph runtime arguments: {', '.join(missing)}")
        if unexpected:
            details.append(
                f"Unexpected graph runtime arguments: {', '.join(unexpected)}"
            )
        raise TaichiRuntimeError("; ".join(details))

    def instantiate(self, key=None):
        if key is None:
            key = self.instance_key()
        return _GraphInstance(self, key)

    def invalidate_runtime(self):
        self._temporary_binding_cache.clear()
        for node in self.nodes:
            invalidate = getattr(node, "invalidate_runtime", None)
            if invalidate is not None:
                invalidate()

    def instance_key(self):
        runtime = impl.get_runtime()
        return (impl.runtime_generation(), impl.current_cfg().arch, id(runtime.prog))

    def compiled_graph(self):
        if self.native_count or self.structured_control_count or self.observation_count:
            raise TaichiRuntimeError(
                "Graphs containing native, observation, or structured-control nodes cannot "
                "be serialized as AOT CGraph yet"
            )
        if self._aot_compiled_graph is None:
            if self._aot_graph_builder is None:
                raise TaichiRuntimeError("This graph does not have an AOT CGraph")
            self._aot_compiled_graph = self._aot_graph_builder.compile()
        return self._aot_compiled_graph

    @property
    def debug_info(self):
        info = {
            "node_count": len(self.nodes),
            "dispatch_count": self.dispatch_count,
            "native_count": self.native_count,
            "observation_count": self.observation_count,
            "structured_control_count": self.structured_control_count,
            "repeat_count": self.repeat_count,
            "nodes": [n.debug_info for n in self.nodes],
            "optimization": dict(self.optimization),
        }
        if hasattr(self._aot_graph_builder, "item_count"):
            info["aot_item_count"] = self._aot_graph_builder.item_count
        return info

    @property
    def ir_debug_info(self):
        return {
            "metadata_version": 1,
            "analysis_only": not bool(self.optimization["mixed_backend_regions"]),
            "analysis": self.ir_analysis.to_dict(),
            "root": graph_ir_to_dict(self.ir_root),
            "pre_optimization_analysis": (self.pre_optimization_ir_analysis.to_dict()),
            "pre_optimization_root": graph_ir_to_dict(self.pre_optimization_ir_root),
            "optimization": dict(self.optimization),
            "fusion_plan": self.fusion_plan.to_dict(),
            "temporary_memory_plan": self.temporary_memory_plan.to_dict(),
        }

    @property
    def execution_definition(self):
        nodes = []
        for node in self.nodes:
            nodes.append(
                {
                    "kind": (
                        "cgraph"
                        if isinstance(node, _CompiledCGraphNode)
                        else (
                            "while"
                            if isinstance(node, _CompiledWhileGraphNode)
                            else (
                                "if"
                                if isinstance(node, _CompiledIfGraphNode)
                                else (
                                    "switch"
                                    if isinstance(node, _CompiledSwitchGraphNode)
                                    else (
                                        "observation"
                                        if isinstance(
                                            node, _CompiledObservationGraphNode
                                        )
                                        else "native"
                                    )
                                )
                            )
                        )
                    ),
                    "dispatch_count": getattr(node, "dispatch_count", 0),
                    "physical_dispatch_count": getattr(
                        node,
                        "physical_dispatch_count",
                        getattr(node, "dispatch_count", 0),
                    ),
                    "runtime_arg_count": len(node.runtime_arg_names),
                    "region_kind": getattr(node, "region_kind", "opaque"),
                    "source_native_count": getattr(node, "source_native_count", 0),
                    "dependency_info": tuple(sorted(node.snode_tree_dependency_info)),
                }
            )
        return {
            "nodes": tuple(nodes),
            "dispatch_count": self.dispatch_count,
            "native_count": self.native_count,
            "observation_count": self.observation_count,
            "structured_control_count": self.structured_control_count,
            "runtime_arg_count": len(self.runtime_arg_names),
            "fixed_runtime_arg_count": len(self.fixed_runtime_args),
            "dependency_info": tuple(sorted(self.snode_tree_dependency_info)),
            "temporary_memory_plan": self.temporary_memory_plan.to_dict(),
        }


class _GraphExecutable:
    def __init__(self, spec):
        self.spec = spec
        self._context = _GraphRunContext() if self.spec.needs_runtime_args else None

    def run(self, args, temporaries=None):
        # Graph.run() holds a per-Graph lock, so this context can safely reuse
        # flattened runtime arguments and resource signatures across invocations.
        context = self._context
        if context is not None:
            context.begin(
                self.spec.bind_runtime_args(args, temporaries),
                self.spec.fixed_runtime_args,
            )
        try:
            for node in self.spec.nodes:
                node.run(context, temporaries)
        finally:
            if context is not None:
                context.end()

    def run_for_submission(self, args, temporaries=None):
        context = self._context
        if context is not None:
            context.begin(
                self.spec.bind_runtime_args(args, temporaries),
                self.spec.fixed_runtime_args,
            )
        try:
            for node in self.spec.nodes:
                if isinstance(
                    node,
                    (
                        _CompiledWhileGraphNode,
                        _CompiledIfGraphNode,
                        _CompiledSwitchGraphNode,
                    ),
                ):
                    node.run_for_submission(context, temporaries)
                else:
                    node.run(context, temporaries)
        finally:
            if context is not None:
                context.end()


class _GraphInstance:
    def __init__(self, spec, key):
        self.spec = spec
        self.key = key
        self._executable = None
        self._native_nodes = None
        self._backend_executable = None
        self._run_context = None
        self._temporary_arena = _GraphTemporaryArena(spec.temporary_memory_plan)
        self._temporary_bindings = None
        self._observation_nodes = tuple(
            node
            for node in spec.nodes
            if isinstance(node, _CompiledObservationGraphNode)
        )
        self._observation_arena = _GraphObservationArena(self._observation_nodes)

        if len(spec.nodes) == 1 and isinstance(spec.nodes[0], _CompiledCGraphNode):
            node = spec.nodes[0]
            if spec.needs_runtime_args:
                self._run_context = _GraphRunContext()
            kind = (
                "mixed_backend_region" if node.source_native_count else "single_cgraph"
            )
            self._install_backend_executable(
                _CGraphJITExecutable(node.compiled_graph), kind
            )
        elif not spec.needs_runtime_args:
            self._native_nodes = spec.nodes
            self._kind = "native_only"
            self._set_run_impl(self._run_native_only)
        else:
            self._executable = _GraphExecutable(spec)
            self._kind = "dispatch_loop"
            self._set_run_impl(self._run_general)

        self._maybe_install_native_replay()

    @property
    def run_impl(self):
        return self.run

    def _set_run_impl(self, run_impl):
        # Store an unbound class function. Keeping a bound method here creates
        # a self-cycle (instance -> method -> instance), which can defer a JIT
        # cache and its backend leases until after Program teardown.
        self._run_impl = run_impl.__func__

    def run(self, args):
        self._run_impl(self, args, self._temporary_bindings)

    def run_for_submission(self, args):
        if not self.spec.supports_native_structured_submission:
            return self.run(args)
        if self._executable is None:
            self._executable = _GraphExecutable(self.spec)
        self._executable.run_for_submission(args, self._temporary_bindings)

    def bind_temporary_buffers(self, bindings):
        self._temporary_bindings = bindings

    def clear_temporary_buffers(self):
        self._temporary_bindings = None

    def acquire_temporary_lease(self):
        return self._temporary_arena.acquire()

    @property
    def temporary_arena_stats(self):
        return self._temporary_arena.stats

    def acquire_observation_lease(self):
        return self._observation_arena.acquire()

    def bind_observation_buffers(self, bindings):
        if bindings is None:
            return
        for node in self._observation_nodes:
            node.bind_snapshot_buffers(bindings[node.name])

    def clear_observation_buffers(self):
        for node in self._observation_nodes:
            node.clear_snapshot_buffers()

    @property
    def observation_arena_stats(self):
        return self._observation_arena.stats

    def _maybe_install_native_replay(self):
        arch = impl.current_cfg().arch
        if arch not in (_ti_core.Arch.cuda, _ti_core.Arch.x64, _ti_core.Arch.arm64):
            return
        if not all(
            isinstance(node, _CompiledNativeGraphNode) for node in self.spec.nodes
        ):
            return
        if self.spec.needs_runtime_args:
            return
        kind = (
            "cuda_native_replay" if arch == _ti_core.Arch.cuda else "cpu_native_replay"
        )
        self._install_backend_executable(
            _NativeReplayExecutable(self.spec.nodes),
            kind,
        )

    def _install_backend_executable(self, executable, kind):
        self._backend_executable = executable
        self._kind = kind
        self._set_run_impl(self._run_backend)
        return self

    def invalidate_runtime(self):
        if self._backend_executable is not None:
            invalidate = getattr(self._backend_executable, "invalidate_runtime", None)
            if invalidate is not None:
                invalidate()

    def prewarm(self):
        if self._backend_executable is not None:
            prewarm = getattr(self._backend_executable, "prewarm", None)
            if prewarm is not None:
                prewarm()
            return self

        for node in self.spec.nodes:
            if isinstance(node, _CompiledNativeGraphNode):
                node.executable.prewarm()
        return self

    def _run_backend(self, args, temporaries=None):
        if self._backend_executable is None:
            return self._run_general(args, temporaries)
        context = self._run_context
        if context is not None:
            context.begin(
                self.spec.bind_runtime_args(args, temporaries),
                self.spec.fixed_runtime_args,
            )
        try:
            self._backend_executable.run(context, temporaries)
        finally:
            if context is not None:
                context.end()

    def _run_native_only(self, args, temporaries=None):
        for node in self._native_nodes:
            node.run(None, temporaries)

    def _run_general(self, args, temporaries=None):
        self._executable.run(args, temporaries)

    @property
    def debug_info(self):
        return {"kind": self._kind}

    @property
    def debug_graph_stats(self):
        if isinstance(self._backend_executable, _CGraphJITExecutable):
            return [self._backend_executable.debug_graph_stats]
        result = []
        for node in self.spec.nodes:
            if isinstance(node, _CompiledCGraphNode):
                result.append(node.debug_graph_stats)
            elif isinstance(
                node,
                (
                    _CompiledWhileGraphNode,
                    _CompiledIfGraphNode,
                    _CompiledSwitchGraphNode,
                ),
            ):
                result.extend(node.debug_graph_stats)
        return result


class _AOTGraphBuilderPlan:
    def __init__(self):
        self._items = []
        self._runtime_arg_names = set()

    def dispatch(self, kernel_cpp, args):
        runtime_arg_names = frozenset(_runtime_arg_names(args))
        self._items.append(("dispatch", kernel_cpp, args, runtime_arg_names))
        self._runtime_arg_names.update(runtime_arg_names)

    @property
    def runtime_arg_names(self):
        """Return symbolic arguments recorded by the durable AOT plan.

        Low-level graph adapters historically dispatched a precompiled kernel
        directly to both ``_aot_graph_plan`` and the native graph builder.  In
        that path ``GraphBuilder.dispatch()`` cannot update its fast-path name
        cache, but the AOT plan still owns the complete symbolic argument list.
        Recovering names here keeps strict runtime validation compatible with
        those adapters without accepting genuinely unknown arguments.
        """
        return frozenset(self._runtime_arg_names)

    def runtime_arg_names_since(self, cursor):
        if cursor < 0 or cursor > len(self._items):
            raise TaichiRuntimeError(f"Invalid AOT graph plan cursor {cursor}")
        return frozenset().union(*(item[3] for item in self._items[cursor:]))

    def append(self, node):
        # Freeze each append at the point where the runtime builder consumes it.
        # Reusing and then mutating one Sequential between appends must not make
        # the lazily compiled AOT plan observe only its final definition.
        runtime_arg_names = frozenset(node._runtime_arg_names)
        self._items.append(
            (
                "append",
                _AOTSequentialSnapshot(node._dispatches),
                1,
                runtime_arg_names,
            )
        )
        self._runtime_arg_names.update(runtime_arg_names)

    def snapshot(self):
        items = []
        for item in self._items:
            if item[0] == "dispatch":
                _, kernel_cpp, args, runtime_arg_names = item
                items.append(
                    (
                        "dispatch",
                        kernel_cpp,
                        tuple(args),
                        runtime_arg_names,
                    )
                )
            elif item[0] == "append":
                _, node, count, runtime_arg_names = item
                items.append(
                    (
                        "append",
                        _AOTSequentialSnapshot(node._dispatches),
                        count,
                        runtime_arg_names,
                    )
                )
            else:
                raise TaichiRuntimeError(f"Unknown AOT graph item kind {item[0]}")

        snapshot = _AOTGraphBuilderPlan()
        snapshot._items = tuple(items)
        snapshot._runtime_arg_names = set(self._runtime_arg_names)
        return snapshot

    def compile(self):
        builder = _ti_core.GraphBuilder()
        for item in self._items:
            if item[0] == "dispatch":
                _, kernel_cpp, args, _ = item
                builder.dispatch(kernel_cpp, args)
            elif item[0] == "append":
                _, node, count, _ = item
                seq = builder.create_sequential()
                node._dispatch_to(seq)
                for _ in range(count):
                    builder.seq().append(seq)
            else:
                raise TaichiRuntimeError(f"Unknown AOT graph item kind {item[0]}")
        return builder.compile()

    @property
    def item_count(self):
        return len(self._items)


def gen_cpp_kernel(kernel_fn, args, *, template_args=None):
    kernel = (
        kernel_fn
        if isinstance(kernel_fn, kernel_impl.Kernel)
        else getattr(kernel_fn, "_primal", None)
    )
    if not isinstance(kernel, kernel_impl.Kernel):
        raise TaichiCompilationError(
            "Graph dispatch expects a decorated Taichi kernel or an explicit "
            "kernel.grad object. Python callables and ti.func objects cannot "
            "be submitted as Graph nodes."
        )
    injected_args = produce_injected_args_for_graph(
        kernel, symbolic_args=args, template_args=template_args
    )
    key = kernel.ensure_compiled(*injected_args)
    return kernel.compiled_kernels[key]


def flatten_args(args):
    """Normalize symbolic args while accepting the pre-native Matrix adapter."""

    normalized_args = []
    for arg in args:
        if isinstance(arg, list):
            if not all(isinstance(row, (list, tuple)) for row in arg):
                raise TaichiRuntimeError(
                    "Legacy Matrix Graph arguments must be a nested list of "
                    "symbolic scalar arguments"
                )
            for row in arg:
                normalized_args.extend(row)
        else:
            normalized_args.append(arg)
    return normalized_args


def _runtime_arg_names(args):
    return {arg.name for arg in args}


def _dispatch_ir_name(kernel_cpp):
    for attribute in ("name", "get_name"):
        value = getattr(kernel_cpp, attribute, None)
        if value is None:
            continue
        if callable(value):
            value = value()
        if value:
            return str(value)
    return type(kernel_cpp).__name__


def _dispatch_ir_node(kernel_cpp, args):
    effects = []
    bindings = []
    for arg in args:
        tag = getattr(arg, "tag", None)
        kind = str(tag)
        access = (
            GraphAccess.READ
            if tag in (ArgKind.SCALAR, ArgKind.MATRIX, ArgKind.TEXTURE)
            else GraphAccess.READ_WRITE
        )
        effects.append(ResourceEffect(arg.name, access))
        bindings.append(RuntimeBinding(arg.name, kind))
    # Until backend kernel access metadata is attached to this JIT record,
    # ndarray writes remain conservative and the node is not rewriteable.
    return DispatchNode(
        name=_dispatch_ir_name(kernel_cpp),
        effects=tuple(effects),
        bindings=tuple(bindings),
        opaque=True,
    )


def _metadata_symbolic_arg(record, arg_id):
    try:
        path = tuple(int(index) for index in arg_id)
        symbolic_args = tuple(record["symbolic_args"])
    except (KeyError, TypeError, ValueError):
        return None
    if not path or path[0] < 0 or path[0] >= len(symbolic_args):
        return None
    name = symbolic_args[path[0]].get("name")
    return str(name) if name else None


def _metadata_iteration_domain(record):
    domain = record.get("iteration_domain", {})
    kind = domain.get("kind")
    if kind == "constant_range":
        return f"range:{int(domain['begin'])}:{int(domain['end'])}"
    if kind in ("external_tensor", "scalar_argument"):
        name = _metadata_symbolic_arg(record, domain.get("arg_id", ()))
        if name is None:
            return None
        if kind == "external_tensor":
            axis = int(domain.get("axis", -1))
            return f"external_tensor:{name}:axis:{axis}"
        return f"scalar_argument:{name}"
    return None


def _metadata_resource_effect(effect, record):
    kind = effect.get("resource_kind")
    if kind == "argument":
        resource = _metadata_symbolic_arg(record, effect.get("arg_id", ()))
        if resource is None:
            return None
        runtime_bound = True
    elif kind == "snode":
        tree_id = int(effect.get("snode_tree_id", -1))
        snode_id = int(effect.get("snode_id", -1))
        if tree_id < 0 or snode_id < 0:
            return None
        grad = ":grad" if bool(effect.get("is_grad", False)) else ""
        resource = f"snode:{tree_id}:{snode_id}{grad}"
        runtime_bound = False
    else:
        return None
    try:
        access = GraphAccess(str(effect["access"]))
    except (KeyError, ValueError):
        return None
    return ResourceEffect(resource, access, runtime_bound=runtime_bound)


def _compiled_dispatch_ir_nodes(compiled_graph, fallback_nodes):
    fallback_nodes = tuple(fallback_nodes)
    records = getattr(compiled_graph, "_dispatch_metadata", None)
    if records is None or len(records) != len(fallback_nodes):
        return fallback_nodes
    result = []
    for record, fallback in zip(records, fallback_nodes):
        side_effects = tuple(str(item) for item in record.get("side_effects", ()))
        iteration_domain = _metadata_iteration_domain(record)
        effects = tuple(
            _metadata_resource_effect(effect, record)
            for effect in record.get("effects", ())
        )
        proven = (
            bool(record.get("available", False))
            and not bool(record.get("opaque", True))
            and iteration_domain is not None
            and all(effect is not None for effect in effects)
        )
        if not proven:
            result.append(
                DispatchNode(
                    name=fallback.name,
                    effects=fallback.effects,
                    bindings=fallback.bindings,
                    synchronization=bool(record.get("synchronization", False)),
                    opaque=True,
                    elementwise=False,
                    side_effects=side_effects,
                )
            )
            continue
        result.append(
            DispatchNode(
                name=fallback.name,
                effects=effects,
                bindings=fallback.bindings,
                iteration_domain=iteration_domain,
                synchronization=bool(record.get("synchronization", False)),
                opaque=False,
                elementwise=bool(record.get("elementwise", False)),
                side_effects=side_effects,
            )
        )
    return tuple(result)


class _AOTSequentialSnapshot:
    def __init__(self, dispatches):
        self._dispatches = tuple(
            (kernel_cpp, tuple(args)) for kernel_cpp, args in dispatches
        )

    def _dispatch_to(self, builder):
        for kernel_cpp, args in self._dispatches:
            builder.dispatch(kernel_cpp, args)


class Sequential:
    def __init__(self):
        self._dispatch_count = 0
        self._dispatches = []
        self._items = []
        self._ir_nodes = []
        self._runtime_arg_names = set()
        self._recording_runtime_arg_names = set()
        self._fixed_runtime_args = {}
        self._lifetime_leases = []
        self._source_native_count = 0
        self._temporary_actions = []

    def dispatch(self, kernel_fn, *args, template_args=None):
        kernel_cpp = gen_cpp_kernel(kernel_fn, args, template_args=template_args)
        unzipped_args = flatten_args(args)
        ir_node = _dispatch_ir_node(kernel_cpp, unzipped_args)
        self._dispatches.append((kernel_cpp, unzipped_args))
        self._items.append(("dispatch", kernel_cpp, unzipped_args))
        self._ir_nodes.append(ir_node)
        names = _runtime_arg_names(unzipped_args)
        self._runtime_arg_names.update(names)
        self._recording_runtime_arg_names.update(names)
        self._dispatch_count += 1
        return self

    def append_native(self, node, *, prewarm=False):
        executable = compile_native_graph_node(node)
        if prewarm:
            executable.prewarm()
        compiled = _CompiledNativeGraphNode(executable)
        action = compiled.recordable_action
        if action is None:
            raise TaichiRuntimeError(
                "Structured Graph Sequential.append_native requires a "
                "recordable provider action"
            )
        backend = _backend_name(_ti_core.arch_name(impl.current_cfg().arch))
        if not action.supports_backend(backend):
            raise TaichiRuntimeError(
                f"Recordable action does not support the active {backend} backend"
            )
        dispatches = tuple(action.dispatches)
        if not dispatches:
            raise TaichiRuntimeError(
                "Recordable action must provide at least one dispatch"
            )
        for name, value in compiled.fixed_runtime_args.items():
            existing = self._fixed_runtime_args.get(name)
            if existing is not None and existing is not value:
                if not (
                    isinstance(existing, (int, float))
                    and isinstance(value, (int, float))
                    and existing == value
                ):
                    raise TaichiRuntimeError(
                        "Sequential recordable actions provide conflicting "
                        f"fixed binding {name!r}"
                    )
            self._fixed_runtime_args[name] = value
        self._items.append(("native", compiled))
        self._ir_nodes.append(compiled.ir_node)
        self._runtime_arg_names.update(compiled.runtime_arg_names)
        self._recording_runtime_arg_names.update(compiled.recording_runtime_arg_names)
        self._lifetime_leases.extend(compiled.lifetime_leases)
        self._temporary_actions.extend(compiled.temporary_actions)
        self._source_native_count += compiled.source_native_count
        self._dispatch_count += len(dispatches)
        return self

    def _dispatch_to(self, builder, *, region_kind="sequential"):
        backend = _backend_name(_ti_core.arch_name(impl.current_cfg().arch))
        recording_dispatches = []
        for item in self._items:
            if item[0] == "dispatch":
                _, kernel_cpp, args = item
                dispatches = ((kernel_cpp, tuple(args)),)
            else:
                _, node = item
                action = node.recordable_action
                if not action.supports_backend(backend):
                    raise TaichiRuntimeError(
                        f"Recordable action does not support the active {backend} backend"
                    )
                if not action.supports_region(region_kind):
                    raise TaichiRuntimeError(
                        "Recordable action is not qualified for Graph region "
                        f"{region_kind!r}"
                    )
                dispatches = tuple(action.dispatches)
            for kernel_cpp, args in dispatches:
                builder.dispatch(kernel_cpp, args)
                recording_dispatches.append((kernel_cpp, tuple(args)))
        return tuple(recording_dispatches)


class GraphBuilder:
    def __init__(self):
        self._aot_graph_plan = _AOTGraphBuilderPlan()
        self._aot_plan_cursor = 0
        self._runtime_graph_builder = _new_runtime_graph_builder()
        self._dispatch_count = 0
        self._runtime_graph_arg_names = set()
        self._runtime_graph_dispatches = []
        self._nodes = []
        self._pending_ir_nodes = []
        self._observation_names = set()

    def dispatch(self, kernel_fn, *args, template_args=None):
        kernel_cpp = gen_cpp_kernel(kernel_fn, args, template_args=template_args)
        unzipped_args = flatten_args(args)
        self._record_dispatch(kernel_cpp, unzipped_args)

    def _record_dispatch(self, kernel_cpp, unzipped_args):
        self._aot_graph_plan.dispatch(kernel_cpp, unzipped_args)
        self._ensure_runtime_graph_builder().dispatch(kernel_cpp, unzipped_args)
        self._runtime_graph_dispatches.append((kernel_cpp, tuple(unzipped_args)))
        self._runtime_graph_arg_names.update(_runtime_arg_names(unzipped_args))
        self._pending_ir_nodes.append(_dispatch_ir_node(kernel_cpp, unzipped_args))
        self._dispatch_count += 1

    def create_sequential(self):
        return Sequential()

    def append(self, node):
        # TODO: support appending dispatch node as well.
        assert isinstance(node, Sequential)
        if node._source_native_count:
            self._flush_graph_builder()
            self._nodes.append(
                _compile_sequential_runtime_node(
                    (node,), name="sequential", region_kind="sequential"
                )
            )
            return self
        self._aot_graph_plan.append(node)
        node._dispatch_to(self._runtime_graph_builder)
        self._runtime_graph_dispatches.extend(
            (kernel, tuple(args)) for kernel, args in node._dispatches
        )
        self._runtime_graph_arg_names.update(node._runtime_arg_names)
        self._dispatch_count += node._dispatch_count
        self._pending_ir_nodes.extend(node._ir_nodes)
        return self

    def _ensure_runtime_graph_builder(self):
        return self._runtime_graph_builder

    def _flush_graph_builder(self):
        if self._dispatch_count == 0:
            return
        # ``_aot_graph_plan`` is the durable source of truth for dispatches.
        # Only recover items added since the previous flush: legacy low-level
        # adapters can bypass ``_record_dispatch()``, but names from an older
        # segment must not leak across a native node boundary.
        self._runtime_graph_arg_names.update(
            self._aot_graph_plan.runtime_arg_names_since(self._aot_plan_cursor)
        )
        self._aot_plan_cursor = self._aot_graph_plan.item_count
        compiled_graph = self._runtime_graph_builder.compile()
        ir_nodes = _compiled_dispatch_ir_nodes(compiled_graph, self._pending_ir_nodes)
        self._nodes.append(
            _CompiledCGraphNode(
                compiled_graph,
                self._dispatch_count,
                self._runtime_graph_arg_names,
                SequentialRegion(ir_nodes, name="cgraph"),
                recording_dispatches=self._runtime_graph_dispatches,
            )
        )
        self._runtime_graph_builder = _new_runtime_graph_builder()
        self._dispatch_count = 0
        self._runtime_graph_arg_names = set()
        self._runtime_graph_dispatches = []
        self._pending_ir_nodes = []

    def _append_native(self, node, *, prewarm=False):
        self._flush_graph_builder()
        executable = compile_native_graph_node(node)
        if prewarm:
            executable.prewarm()
        self._nodes.append(_CompiledNativeGraphNode(executable))
        return self

    @staticmethod
    def _control_name(value, role):
        name = value if isinstance(value, str) else getattr(value, "name", None)
        if not isinstance(name, str) or not name:
            raise TaichiRuntimeError(
                f"Graph {role} must be a symbolic argument or resource name"
            )
        return name

    @classmethod
    def _control_names(cls, values, role):
        names = tuple(cls._control_name(value, role) for value in values)
        if len(names) != len(set(names)):
            raise TaichiRuntimeError(
                f"Graph {role} must not contain duplicate resources"
            )
        return names

    def while_loop(
        self,
        condition,
        body,
        *,
        predicate,
        max_iterations,
        control_inputs=(),
        carried_state=(),
        counter=None,
        status=None,
        chunk_size=None,
        masked_execution=False,
        lowering_mode="auto",
        name="while",
    ):
        """Append a structured, capped while region.

        ``condition`` computes ``predicate`` from any number of declared
        ``control_inputs``. Nonzero means continue. ``body`` updates the fixed
        ``carried_state`` and optional exact iteration ``counter``. An optional
        user-defined ``status`` resource records why iteration terminated; it
        is observed with the predicate but never interpreted by Graph. Backend
        lowering is selected automatically unless ``lowering_mode`` requests
        a portable or required-native path.
        """
        self._flush_graph_builder()
        predicate_name = self._control_name(predicate, "while predicate")
        counter_name = (
            None if counter is None else self._control_name(counter, "while counter")
        )
        status_name = (
            None if status is None else self._control_name(status, "while status")
        )
        self._nodes.append(
            _CompiledWhileGraphNode(
                condition,
                body,
                predicate=predicate_name,
                control_inputs=self._control_names(
                    control_inputs, "while control_inputs"
                ),
                carried_state=self._control_names(carried_state, "while carried_state"),
                max_iterations=max_iterations,
                counter=counter_name,
                status=status_name,
                chunk_size=chunk_size,
                masked_execution=masked_execution,
                lowering_mode=lowering_mode,
                name=name,
            )
        )
        return self

    def if_then_else(
        self,
        condition,
        then_region,
        *,
        predicate,
        control_inputs=(),
        else_region=None,
        lowering_mode="auto",
        name="if",
    ):
        """Append a structured conditional region."""
        self._flush_graph_builder()
        self._nodes.append(
            _CompiledIfGraphNode(
                condition,
                then_region,
                else_region,
                predicate=self._control_name(predicate, "if predicate"),
                control_inputs=self._control_names(control_inputs, "if control_inputs"),
                lowering_mode=lowering_mode,
                name=name,
            )
        )
        return self

    def switch(
        self,
        condition,
        branches,
        *,
        selector,
        control_inputs=(),
        default_region=None,
        lowering_mode="auto",
        name="switch",
    ):
        """Append a zero-based structured switch region."""
        self._flush_graph_builder()
        self._nodes.append(
            _CompiledSwitchGraphNode(
                condition,
                tuple(branches),
                default_region,
                selector=self._control_name(selector, "switch selector"),
                control_inputs=self._control_names(
                    control_inputs, "switch control_inputs"
                ),
                lowering_mode=lowering_mode,
                name=name,
            )
        )
        return self

    def observe(self, *values, name="observation"):
        """Append a deferred packed snapshot of scalar ndarray arguments.

        ``Graph.submit()`` captures values on device and returns before host
        readback. Consume the immutable snapshot through
        ``SubmissionTicket.observations()``.
        """
        if name in self._observation_names:
            raise TaichiRuntimeError(
                f"Graph observation name {name!r} is already defined"
            )
        node = _CompiledObservationGraphNode(values, name)
        self._flush_graph_builder()
        self._nodes.append(node)
        self._observation_names.add(name)
        return self

    def append_native(self, node, *, prewarm=False):
        return self._append_native(node, prewarm=prewarm)

    def compile(self):
        self._flush_graph_builder()
        if not self._nodes:
            return Graph(
                _CompiledCGraphNode(
                    self._ensure_runtime_graph_builder().compile(),
                    0,
                    (),
                    SequentialRegion((), name="cgraph"),
                )
            )
        return Graph(
            _GraphSpec(
                self._nodes,
                aot_graph_builder=self._aot_graph_plan.snapshot(),
            )
        )


class SubmissionTicket:
    """Completion for one opt-in ``Graph.submit()`` invocation.

    Construct tickets through ``Graph.submit()``. The runtime keeps native
    Graph ownership valid until backend completion, even when the user drops
    the ticket without waiting.
    """

    __slots__ = ("_admission", "_completion", "_observation", "_runtime")

    def __init__(self, completion, runtime, admission=None, observation=None):
        self._admission = admission
        self._completion = completion
        self._observation = observation
        self._runtime = runtime

    def done(self):
        if self._admission is None:
            ready = self._completion.done()
        else:
            ready = self._admission._completion_done(self._completion)
        if ready:
            self._runtime.release_runtime_submission_owner(self._completion)
        return ready

    def wait(self):
        if self._admission is None:
            self._completion.wait()
        else:
            self._admission._completion_wait(self._completion)
        self._runtime.release_runtime_submission_owner(self._completion)

    def observations(self):
        """Wait if needed, then materialize this submission's snapshot."""
        if self._observation is None:
            return {}
        self.wait()
        return self._observation.materialize()

    @property
    def backend(self):
        return self._completion.backend

    @property
    def sequence(self):
        return self._completion.sequence

    @property
    def _has_backend_work(self):
        return self._completion.has_backend_work

    def __del__(self):
        observation = getattr(self, "_observation", None)
        if observation is not None:
            try:
                observation.discard()
            except Exception:
                pass


class Graph:
    def __init__(self, compiled_graph) -> None:
        self._lifecycle_lock = threading.Lock()
        self._stale_snode_tree_dependencies = set()
        if isinstance(compiled_graph, _GraphSpec):
            self._spec = compiled_graph
        elif isinstance(compiled_graph, _CompiledCGraphNode):
            self._spec = _GraphSpec(
                [compiled_graph], aot_compiled_graph=compiled_graph.compiled_graph
            )
        else:
            node = _CompiledCGraphNode(compiled_graph, 0, ())
            self._spec = _GraphSpec([node], aot_compiled_graph=compiled_graph)
        self._contains_native_nodes_value = self._spec.native_count > 0
        self._contains_structured_control_value = (
            self._spec.structured_control_count > 0
        )
        self._contains_observations_value = self._spec.observation_count > 0
        self._last_observations = {}
        self._latest_control_flow_was_async = False
        self._submission_lane = _new_submission_lane("graph")
        self._execution_definition = self._spec.execution_definition
        self._execution_arch = _ti_core.arch_name(impl.current_cfg().arch)
        self._instances = {}
        self._instance = self._instance_for_current_runtime()
        self._runtime_valid = True
        self._run_impl = self._instance.run_impl
        impl.get_runtime().register_runtime_object(self)

    def run(self, args):
        # A graph invocation is one host-side transaction, including mixed
        # CGraph/native sequences. The lock is per Graph and does not wait for
        # GPU completion, so independent graphs remain independently submitable.
        with self._lifecycle_lock:
            self._check_runtime_valid()
            runtime = impl.pytaichi
            self._spec.validate_runtime_args(args)
            submission_state = runtime._active_graph_submissions
            if submission_state < 0:
                raise TaichiRuntimeError(
                    "Graph.run() is primal-only and cannot start while another "
                    "Python thread is entering ti.ad.Tape() or ti.ad.FwdMode(). "
                    "Wait for automatic AD setup to finish."
                )
            if runtime.target_tape is not None:
                raise TaichiRuntimeError(
                    "Graph.run() is primal-only and cannot execute inside an "
                    "active ti.ad.Tape(). Graph dispatches are opaque to Tape "
                    "and would omit gradients. Run the Graph outside automatic "
                    "AD, or manually run an explicit grad-kernel Graph outside "
                    "the Tape context."
                )
            if runtime.fwd_mode_manager is not None:
                raise TaichiRuntimeError(
                    "Graph.run() is primal-only and cannot execute inside an "
                    "active ti.ad.FwdMode(). Graph dispatches are opaque to "
                    "forward AD and would omit dual propagation. Run the Graph "
                    "outside automatic AD."
                )
            # Runtime AD state is process-global rather than thread-local. The
            # signed state closes the window where a native call releases the
            # GIL and another Python thread enters Tape/FwdMode. Publish the
            # increment before the first native call: otherwise two independent
            # Graphs can both snapshot zero, overwrite the count with one, and
            # drive it negative when their paired finally blocks run.
            temporary_lease = self._instance.acquire_temporary_lease()
            observation_lease = None
            try:
                observation_lease = self._instance.acquire_observation_lease()
                temporary_bindings = (
                    temporary_lease.bindings if temporary_lease is not None else None
                )
                runtime._active_graph_submissions = submission_state + 1
                try:
                    self._instance.bind_temporary_buffers(temporary_bindings)
                    self._instance.bind_observation_buffers(
                        observation_lease.bindings
                        if observation_lease is not None
                        else None
                    )
                    runtime.prog._record_runtime_graph_submission()
                    self._run_impl(args)
                    self._latest_control_flow_was_async = False
                    if observation_lease is not None:
                        self._last_observations = observation_lease.materialize()
                finally:
                    self._instance.clear_observation_buffers()
                    self._instance.clear_temporary_buffers()
                    runtime._active_graph_submissions -= 1
            finally:
                if observation_lease is not None:
                    observation_lease.cancel()
                if temporary_lease is not None:
                    temporary_lease.cancel()

    def submit(
        self,
        args,
        *,
        pacer=None,
        lane=None,
        on_saturation="wait",
    ):
        """Submit one Graph invocation and return a ``SubmissionTicket``.

        Submission is asynchronous on CUDA/Vulkan when backend work remains;
        CPU tickets are already complete. The runtime argument, lifecycle,
        concurrency, and automatic-differentiation rules are identical to
        ``run()``. A shared ``SubmissionPacer`` can bound backend backlog and
        fairly arbitrate complete host submissions before they enqueue work.
        """
        if (
            self._contains_structured_control_value
            and not self._spec.supports_native_structured_submission
        ):
            raise TaichiRuntimeError(
                "Graph.submit() supports structured control only when every "
                "region has native_required asynchronous backend lowering"
            )
        runtime = impl.pytaichi
        with self._lifecycle_lock:
            self._check_runtime_valid()
            self._spec.validate_runtime_args(args, "Graph.submit")
        admission = _reserve_paced_submission(
            pacer,
            runtime,
            self._submission_lane,
            lane=lane,
            on_saturation=on_saturation,
        )
        temporary_lease = None
        observation_lease = None
        observation_state = None
        try:
            with self._lifecycle_lock:
                self._check_runtime_valid()
                if runtime is not impl.pytaichi:
                    raise TaichiRuntimeError(
                        "This graph was compiled before ti.reset() or a "
                        "runtime reinitialization. Please rebuild the graph "
                        "after ti.init()."
                    )
                self._spec.validate_runtime_args(args, "Graph.submit")
                submission_state = runtime._active_graph_submissions
                if submission_state < 0:
                    raise TaichiRuntimeError(
                        "Graph submission cannot start while another Python "
                        "thread is entering ti.ad.Tape() or ti.ad.FwdMode()."
                    )
                if runtime.target_tape is not None:
                    raise TaichiRuntimeError(
                        "Graph submission is primal-only and cannot execute "
                        "inside an active ti.ad.Tape()."
                    )
                if runtime.fwd_mode_manager is not None:
                    raise TaichiRuntimeError(
                        "Graph submission is primal-only and cannot execute "
                        "inside an active ti.ad.FwdMode()."
                    )

                # Publish the AD exclusion count before transaction creation:
                # the pybind call may release the GIL while waiting for another
                # runtime submission reader/writer boundary.
                temporary_lease = self._instance.acquire_temporary_lease()
                observation_lease = self._instance.acquire_observation_lease()
                temporary_bindings = (
                    temporary_lease.bindings if temporary_lease is not None else None
                )
                runtime._active_graph_submissions = submission_state + 1
                try:
                    self._instance.bind_temporary_buffers(temporary_bindings)
                    self._instance.bind_observation_buffers(
                        observation_lease.bindings
                        if observation_lease is not None
                        else None
                    )
                    transaction = runtime.prog._begin_runtime_submission_transaction()
                    runtime.prog._record_runtime_graph_submission()
                    if self._contains_structured_control_value:
                        self._instance.run_for_submission(args)
                        self._latest_control_flow_was_async = True
                    else:
                        self._run_impl(args)
                    # CGraph/kernel paths publish work themselves. Native plans
                    # use Program methods outside that launch path, so publish
                    # once for the whole native portion without changing run().
                    if self._contains_native_nodes_value:
                        transaction._mark_submission()
                    completion = transaction._finish()
                    if temporary_lease is not None:
                        temporary_lease.attach(completion)
                        temporary_lease = None
                    if observation_lease is not None:
                        observation_state = observation_lease.attach(completion)
                        observation_lease = None
                finally:
                    self._instance.clear_observation_buffers()
                    self._instance.clear_temporary_buffers()
                    runtime._active_graph_submissions -= 1

                if self._contains_native_nodes_value and completion.has_backend_work:
                    runtime.retain_runtime_submission_owner(completion, self)
        except BaseException:
            if observation_lease is not None:
                observation_lease.cancel()
            if temporary_lease is not None:
                temporary_lease.cancel()
            if admission is not None:
                admission._cancel()
            raise
        if admission is not None:
            admission._attach(completion)
        return SubmissionTicket(
            completion,
            runtime,
            admission,
            observation=observation_state,
        )

    def _instance_for_current_runtime(self):
        key = self._spec.instance_key()
        instance = self._instances.get(key)
        if instance is None:
            instance = self._spec.instantiate(key)
            self._instances[key] = instance
        return instance

    def _prewarm(self):
        with self._lifecycle_lock:
            self._check_runtime_valid()
            self._instance.prewarm()
        return self

    def _check_runtime_valid(self):
        if not self._runtime_valid:
            raise TaichiRuntimeError(
                "This graph was compiled before ti.reset() or a runtime "
                "reinitialization. Please rebuild the graph after ti.init()."
            )
        self._spec.validate_lifetime_leases()
        if self._stale_snode_tree_dependencies:
            dependencies = ", ".join(
                f"id={tree_id} generation={generation}"
                for tree_id, generation in sorted(self._stale_snode_tree_dependencies)
            )
            raise TaichiRuntimeError(
                "This graph references a destroyed SNodeTree "
                f"({dependencies}); rebuild the Graph."
            )

    def _retire_snode_tree(self, dependency):
        dependency = tuple(dependency)
        with self._lifecycle_lock:
            if (
                self._spec is None
                or dependency not in self._spec.snode_tree_dependencies
            ):
                return False
            if dependency in self._stale_snode_tree_dependencies:
                return True
            self._stale_snode_tree_dependencies.add(dependency)
            for instance in self._instances.values():
                instance.invalidate_runtime()
            self._spec.invalidate_runtime()
            return True

    def _cancel_snode_tree_retirement(self, dependency):
        with self._lifecycle_lock:
            self._stale_snode_tree_dependencies.discard(tuple(dependency))

    def _invalidate_runtime(self):
        with self._lifecycle_lock:
            self._runtime_valid = False
            self._run_impl = None
            for instance in self._instances.values():
                instance.invalidate_runtime()
            if self._spec is not None:
                self._spec.invalidate_runtime()
            self._instance = None
            self._instances.clear()
            # Definition nodes currently own mixed-graph JIT caches and native
            # executables. Release them before Program/backend teardown so
            # backend allocation leases cannot outlive their Device registry.
            self._spec = None

    @property
    def _debug_info(self):
        with self._lifecycle_lock:
            self._check_runtime_valid()
            return self._spec.debug_info

    @property
    def _ir_debug_info(self):
        with self._lifecycle_lock:
            self._check_runtime_valid()
            return self._spec.ir_debug_info

    @property
    def _instance_debug_info(self):
        with self._lifecycle_lock:
            self._check_runtime_valid()
            return self._instance.debug_info

    @property
    def _graph_stats(self):
        with self._lifecycle_lock:
            self._check_runtime_valid()
            return self._instance.debug_graph_stats

    def control_flow_stats(self):
        """Return immutable reports for the latest structured-control run."""
        with self._lifecycle_lock:
            self._check_runtime_valid()
            if self._latest_control_flow_was_async:
                raise TaichiRuntimeError(
                    "Control-flow reports are unavailable after asynchronous "
                    "structured submission. Use SubmissionTicket.observations() "
                    "for explicit terminal state, or Graph.run() for a "
                    "synchronous control-flow report."
                )
            for node in self._spec.nodes:
                materialize = getattr(node, "materialize_pending_report", None)
                if materialize is not None:
                    materialize()
            return tuple(
                node.last_report
                for node in self._spec.nodes
                if isinstance(
                    node,
                    (
                        _CompiledWhileGraphNode,
                        _CompiledIfGraphNode,
                        _CompiledSwitchGraphNode,
                    ),
                )
            )

    def latest_observations(self):
        """Return the most recent synchronous ``run()`` snapshot."""
        with self._lifecycle_lock:
            self._check_runtime_valid()
            return _copy_observation_result(self._last_observations)

    def execution_stats(self):
        """Return an immutable execution-path and static-Field report.

        The first call enables detailed backend counters for subsequent runs.
        No per-run report objects or strings are created while diagnostics are
        disabled.
        """
        with self._lifecycle_lock:
            if not self._runtime_valid:
                lifecycle_state = "runtime_invalid"
                instance_kind = "unavailable"
                backend_stats = ()
            elif self._stale_snode_tree_dependencies:
                lifecycle_state = "stale_field_dependency"
                instance_kind = self._instance.debug_info["kind"]
                backend_stats = ()
            else:
                lifecycle_state = "ready"
                instance_kind = self._instance.debug_info["kind"]
                backend_stats = self._instance.debug_graph_stats
            temporary_arena_stats = (
                self._instance.temporary_arena_stats
                if lifecycle_state == "ready"
                else {}
            )
            observation_arena_stats = (
                self._instance.observation_arena_stats
                if lifecycle_state == "ready"
                else {}
            )
            observation_staging_bytes = 0
            if lifecycle_state == "ready" and (
                self._contains_structured_control_value
                or self._contains_observations_value
            ):
                observation_staging_bytes = int(
                    impl.get_runtime().prog._graph_observation_staging_stats()[
                        "persistent_bytes"
                    ]
                )
            return _execution_report(
                self._execution_definition,
                self._execution_arch,
                lifecycle_state,
                instance_kind,
                backend_stats,
                observation_staging_bytes=observation_staging_bytes,
                temporary_memory_plan=self._execution_definition[
                    "temporary_memory_plan"
                ],
                temporary_arena_stats=temporary_arena_stats,
                observation_arena_stats=observation_arena_stats,
            )

    @property
    def _compiled_graph(self):
        with self._lifecycle_lock:
            self._check_runtime_valid()
            return self._spec.compiled_graph()

    @property
    def _contains_native_nodes(self):
        return self._contains_native_nodes_value


def _deprecate_arg_args(kwargs: Dict[str, Any]):
    if "field_dim" in kwargs:
        warnings.warn(
            "The field_dim argument for ndarray will be deprecated in v1.6.0, use ndim instead.",
            DeprecationWarning,
        )
        if "ndim" in kwargs:
            raise TaichiRuntimeError(
                "field_dim is deprecated, please do not specify field_dim and ndim at the same time."
            )
        kwargs["ndim"] = kwargs["field_dim"]
        del kwargs["field_dim"]
    tag = kwargs["tag"]

    if tag == ArgKind.SCALAR:
        if "element_shape" in kwargs:
            raise TaichiRuntimeError(
                "The element_shape argument for scalar is deprecated in v1.6.0, and is removed in v1.7.0. "
                "Please remove them."
            )

    if tag == ArgKind.NDARRAY:
        if "element_shape" in kwargs:
            raise TaichiRuntimeError(
                "The element_shape argument for ndarray is deprecated in v1.6.0, and it is removed in v1.7.0. "
                "Please use vector or matrix data type instead."
            )

    if tag == ArgKind.RWTEXTURE or tag == ArgKind.TEXTURE:
        if "dtype" in kwargs:
            warnings.warn(
                "The dtype argument for texture will be deprecated in v1.6.0, use format instead.",
                DeprecationWarning,
            )
            del kwargs["dtype"]

        if "shape" in kwargs:
            raise TaichiRuntimeError(
                "The shape argument for texture is deprecated in v1.6.0, and it is removed in v1.7.0. "
                "Please use ndim instead. (Note that you no longer need the exact texture size.)"
            )

        if "channel_format" in kwargs or "num_channels" in kwargs:
            if "fmt" in kwargs:
                raise TaichiRuntimeError(
                    "channel_format and num_channels are deprecated, please do not specify channel_format/num_channels and fmt at the same time."
                )
            if tag == ArgKind.RWTEXTURE:
                fmt = TY_CH2FORMAT[(kwargs["channel_format"], kwargs["num_channels"])]
                kwargs["fmt"] = fmt
                raise TaichiRuntimeError(
                    "The channel_format and num_channels arguments for texture are deprecated in v1.6.0, "
                    "and they are removed in v1.7.0. Please use fmt instead."
                )
            else:
                raise TaichiRuntimeError(
                    "The channel_format and num_channels arguments are no longer required for non-RW textures "
                    "since v1.6.0, and they are removed in v1.7.0. Please remove them."
                )


def _check_args(kwargs: Dict[str, Any], allowed_kwargs: List[str]):
    for k, v in kwargs.items():
        if k not in allowed_kwargs:
            raise TaichiRuntimeError(
                f"Invalid argument: {k}, you can only create a graph argument with: {allowed_kwargs}"
            )
        if k == "tag":
            if not isinstance(v, ArgKind):
                raise TaichiRuntimeError(
                    f"tag must be a ArgKind variant, but found {type(v)}."
                )
        if k == "name":
            if not isinstance(v, str):
                raise TaichiRuntimeError(f"name must be a string, but found {type(v)}.")


def _make_arg_scalar(kwargs: Dict[str, Any]):
    allowed_kwargs = [
        "tag",
        "name",
        "dtype",
    ]
    _check_args(kwargs, allowed_kwargs)
    name = kwargs["name"]
    dtype = kwargs["dtype"]
    descriptor = describe_element_type(dtype)
    if descriptor.category != "scalar":
        raise TaichiRuntimeError(
            f"Tag ArgKind.SCALAR must specify a scalar type, but found {type(dtype)}."
        )
    return _ti_core.Arg(ArgKind.SCALAR, name, descriptor.logical_type, 0, [])


def _make_arg_ndarray(kwargs: Dict[str, Any]):
    allowed_kwargs = [
        "tag",
        "name",
        "dtype",
        "ndim",
    ]
    _check_args(kwargs, allowed_kwargs)
    name = kwargs["name"]
    ndim = kwargs["ndim"]
    dtype = kwargs["dtype"]
    descriptor = describe_element_type(dtype)
    if not descriptor.is_complete:
        raise TaichiRuntimeError(
            f"Tag ArgKind.NDARRAY requires a concrete scalar, vector, matrix, "
            f"or struct element type, but found {dtype}."
        )
    if descriptor.category == "tensor" and len(descriptor.shape) not in (1, 2):
        raise TaichiRuntimeError(
            "Graph ndarray tensor elements support vector (rank 1) and "
            f"matrix (rank 2) types only, but got shape {descriptor.shape}."
        )
    if descriptor.category == "struct":
        raise TaichiRuntimeError(
            "Graph StructNdarray arguments are not supported by the current "
            "serialized Graph schema."
        )
    return _ti_core.Arg(ArgKind.NDARRAY, name, descriptor.logical_type, ndim, [])


def _make_arg_matrix(kwargs: Dict[str, Any]):
    allowed_kwargs = [
        "tag",
        "name",
        "dtype",
    ]
    _check_args(kwargs, allowed_kwargs)
    name = kwargs["name"]
    dtype = kwargs["dtype"]
    descriptor = describe_element_type(dtype)
    if not isinstance(dtype, MatrixType) or descriptor.category != "tensor":
        raise TaichiRuntimeError(
            f"Tag ArgKind.MATRIX must specify matrix type, but got {dtype}."
        )
    return _ti_core.Arg(ArgKind.MATRIX, f"{name}", descriptor.logical_type, 0, [])


def _make_arg_texture(kwargs: Dict[str, Any]):
    allowed_kwargs = [
        "tag",
        "name",
        "ndim",
    ]
    _check_args(kwargs, allowed_kwargs)
    name = kwargs["name"]
    ndim = kwargs["ndim"]
    return _ti_core.Arg(ArgKind.TEXTURE, name, impl.f32, 4, [2] * ndim)


def _make_arg_rwtexture(kwargs: Dict[str, Any]):
    allowed_kwargs = [
        "tag",
        "name",
        "ndim",
        "fmt",
    ]
    _check_args(kwargs, allowed_kwargs)
    name = kwargs["name"]
    ndim = kwargs["ndim"]
    fmt = kwargs["fmt"]
    if fmt == enums.Format.unknown:
        raise TaichiRuntimeError(
            f"Tag ArgKind.RWTEXTURE must specify a valid color format, but found {fmt}."
        )
    channel_format, num_channels = FORMAT2TY_CH[fmt]
    return _ti_core.Arg(
        ArgKind.RWTEXTURE, name, channel_format, num_channels, [2] * ndim
    )


def _make_arg(kwargs: Dict[str, Any]):
    assert "tag" in kwargs
    _deprecate_arg_args(kwargs)
    proc = {
        ArgKind.SCALAR: _make_arg_scalar,
        ArgKind.NDARRAY: _make_arg_ndarray,
        ArgKind.MATRIX: _make_arg_matrix,
        ArgKind.TEXTURE: _make_arg_texture,
        ArgKind.RWTEXTURE: _make_arg_rwtexture,
    }
    tag = kwargs["tag"]
    return proc[tag](kwargs)


def _kwarg_rewriter(args, kwargs):
    for i, arg in enumerate(args):
        rewrite_map = {
            0: "tag",
            1: "name",
            2: "dtype",
            3: "ndim",
            4: "field_dim",
            5: "element_shape",
            6: "channel_format",
            7: "shape",
            8: "num_channels",
        }
        if i in rewrite_map:
            kwargs[rewrite_map[i]] = arg
        else:
            raise TaichiRuntimeError(f"Unexpected {i}th positional argument")


def Arg(*args, **kwargs):
    _kwarg_rewriter(args, kwargs)
    return _make_arg(kwargs)


__all__ = [
    "GraphBuilder",
    "Graph",
    "SubmissionTicket",
    "SubmissionPacer",
    "GraphExecutionCounters",
    "GraphExecutionSegmentReport",
    "GraphExecutionReport",
    "GraphWhileReport",
    "GraphBranchReport",
    "Arg",
    "ArgKind",
]
