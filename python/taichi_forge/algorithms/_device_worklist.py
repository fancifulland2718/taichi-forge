"""Fixed-capacity device worklists and deterministic keyed arbitration.

The worklist owns stable front/back storage, :class:`DeviceExtent` state, and
small device counters.  Producers may append atomically without reading the
count on host.  Stable selection and keyed claim paths reuse Forge's existing
device prefix, compact, and native stable-sort providers.
"""

from dataclasses import dataclass

import numpy as np

from taichi_forge._lib import core as _ti_core
from taichi_forge._kernels import (
    device_prefix_copy_masked_ndarray,
    device_prefix_fill_tail_ndarray,
)
from taichi_forge.algorithms import _algorithms as _alg
from taichi_forge.algorithms._device_prefix import (
    DevicePrefix,
    DevicePrefixWorkspace,
    _dtype_bytes,
    _sort_tail,
)
from taichi_forge.graph._ir import GraphAccess, ResourceEffect, RuntimeBinding
from taichi_forge.graph._native import (
    DispatchGraphAction,
    NativeGraphExecutable,
    NativeGraphNode,
)
from taichi_forge.lang import impl, ops
from taichi_forge.lang._ndarray import ScalarNdarray
from taichi_forge.lang.device_extent import (
    DeviceDispatchState,
    DeviceExtent,
    device_dispatch_state_publish,
)
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.lang.impl import ndarray as ti_ndarray
from taichi_forge.lang.kernel_impl import func, kernel
from taichi_forge.types import ndarray_type
from taichi_forge.types.annotations import template
from taichi_forge.types.primitive_types import f32, f64, i32, i64, u32, u64


_WORKLIST_DTYPES = (i32, u32, i64, u64, f32, f64)
_CONFLICT_KEY_DTYPES = (i32, u32, i64, u64)
_STAT_NAMES = (
    "generated",
    "accepted",
    "rejected",
    "conflicts",
    "winners",
    "overflow",
)


def _current_backend_name():
    arch = impl.current_cfg().arch
    if arch in (_ti_core.Arch.x64, _ti_core.Arch.arm64):
        return "cpu"
    return _ti_core.arch_name(arch)


def _require_capacity(value, role="capacity"):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"DeviceWorklist {role} must be a Python integer")
    if not 1 <= value <= 0x7FFFFFFF:
        raise ValueError(f"DeviceWorklist {role} must be in [1, 2^31-1]")
    return value


def _require_worklist_array(value, role, capacity, dtype=None):
    if not isinstance(value, ScalarNdarray):
        raise TypeError(f"DeviceWorklist {role} must be a scalar ti.ndarray")
    if tuple(value.shape) != (capacity,):
        raise ValueError(f"DeviceWorklist {role} must have fixed shape ({capacity},)")
    if value.dtype not in _WORKLIST_DTYPES:
        raise TypeError(f"DeviceWorklist {role} supports ti.i32/u32/i64/u64/f32/f64")
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"DeviceWorklist {role} dtype must be {dtype}")
    if value.arr is None:
        raise TaichiRuntimeError(f"DeviceWorklist {role} belongs to a stale runtime")
    return value


def _require_stat_array(value, role):
    if (
        not isinstance(value, ScalarNdarray)
        or value.dtype != i32
        or tuple(value.shape) != ()
    ):
        raise TypeError(f"DeviceWorklist {role} must be a scalar i32 ndarray")
    return value


@func
def device_worklist_append(
    values: template(),
    extent_state: template(),
    generated: template(),
    overflow: template(),
    capacity: i32,
    value: template(),
):
    """Atomically append one value and return its slot, or ``-1`` on overflow.

    Atomic append order is deliberately unspecified.  Use stable selection or
    deterministic keyed claim when output order is part of the contract.
    """

    result = -1
    if capacity != values.shape[0]:
        # The scalar capacity is part of the Graph ABI.  Reject a forged or
        # accidentally mismatched runtime binding before reserving or writing.
        ops.atomic_or(overflow[None], 1)
        ops.atomic_or(extent_state[1], 1)
    else:
        slot = ops.atomic_add(generated[None], 1)
        if slot < capacity:
            values[slot] = value
            result = slot
        else:
            ops.atomic_or(overflow[None], 1)
            ops.atomic_or(extent_state[1], 1)
    return result


@kernel
def _reset_worklist_target(
    extent_state: ndarray_type.ndarray(dtype=i32, ndim=1),
    generated: ndarray_type.ndarray(dtype=i32, ndim=0),
    accepted: ndarray_type.ndarray(dtype=i32, ndim=0),
    rejected: ndarray_type.ndarray(dtype=i32, ndim=0),
    conflicts: ndarray_type.ndarray(dtype=i32, ndim=0),
    winners: ndarray_type.ndarray(dtype=i32, ndim=0),
    overflow: ndarray_type.ndarray(dtype=i32, ndim=0),
):
    extent_state[0] = 0
    extent_state[1] = 0
    generated[None] = 0
    accepted[None] = 0
    rejected[None] = 0
    conflicts[None] = 0
    winners[None] = 0
    overflow[None] = 0


@kernel
def _finalize_atomic_worklist(
    extent_state: ndarray_type.ndarray(dtype=i32, ndim=1),
    generated: ndarray_type.ndarray(dtype=i32, ndim=0),
    accepted: ndarray_type.ndarray(dtype=i32, ndim=0),
    rejected: ndarray_type.ndarray(dtype=i32, ndim=0),
    conflicts: ndarray_type.ndarray(dtype=i32, ndim=0),
    winners: ndarray_type.ndarray(dtype=i32, ndim=0),
    overflow: ndarray_type.ndarray(dtype=i32, ndim=0),
    capacity: i32,
):
    raw = generated[None]
    bounded = ops.min(ops.max(raw, 0), capacity)
    rejected_count = ops.max(0, raw - capacity)
    accepted[None] = bounded
    rejected[None] = rejected_count
    conflicts[None] = 0
    winners[None] = bounded
    extent_state[0] = bounded
    status = 1 if overflow[None] != 0 or rejected_count != 0 else 0
    overflow[None] = status
    extent_state[1] = status


@kernel
def _finalize_atomic_worklist_with_dispatch(
    extent_state: ndarray_type.ndarray(dtype=i32, ndim=1),
    generated: ndarray_type.ndarray(dtype=i32, ndim=0),
    accepted: ndarray_type.ndarray(dtype=i32, ndim=0),
    rejected: ndarray_type.ndarray(dtype=i32, ndim=0),
    conflicts: ndarray_type.ndarray(dtype=i32, ndim=0),
    winners: ndarray_type.ndarray(dtype=i32, ndim=0),
    overflow: ndarray_type.ndarray(dtype=i32, ndim=0),
    dispatch_packet: ndarray_type.ndarray(dtype=u32, ndim=1),
    capacity: i32,
):
    raw = generated[None]
    bounded = ops.min(ops.max(raw, 0), capacity)
    rejected_count = ops.max(0, raw - capacity)
    accepted[None] = bounded
    rejected[None] = rejected_count
    conflicts[None] = 0
    winners[None] = bounded
    status = 1 if overflow[None] != 0 or rejected_count != 0 else 0
    overflow[None] = status
    device_dispatch_state_publish(extent_state, dispatch_packet, capacity, bounded)
    extent_state[1] = status


@func
def _cuda_graph_apply_bounded_group(group_control: template()):
    if group_control[8] != 0:
        nodes_address = ops.cast(group_control[2], u64) | (
            ops.cast(group_control[3], u64) << 32
        )
        driver_status = impl.call_internal(
            "cuda_graph_update_bounded_group",
            nodes_address,
            group_control[4],
            group_control[5],
            group_control[6],
            with_runtime_context=False,
        )
        group_control[7] = ops.cast(driver_status, u32)


@kernel
def _finalize_atomic_worklist_with_cuda_group(
    extent_state: ndarray_type.ndarray(dtype=i32, ndim=1),
    generated: ndarray_type.ndarray(dtype=i32, ndim=0),
    accepted: ndarray_type.ndarray(dtype=i32, ndim=0),
    rejected: ndarray_type.ndarray(dtype=i32, ndim=0),
    conflicts: ndarray_type.ndarray(dtype=i32, ndim=0),
    winners: ndarray_type.ndarray(dtype=i32, ndim=0),
    overflow: ndarray_type.ndarray(dtype=i32, ndim=0),
    dispatch_packet: ndarray_type.ndarray(dtype=u32, ndim=1),
    group_control: ndarray_type.ndarray(dtype=u32, ndim=1),
    capacity: i32,
):
    raw = generated[None]
    bounded = ops.min(ops.max(raw, 0), capacity)
    rejected_count = ops.max(0, raw - capacity)
    accepted[None] = bounded
    rejected[None] = rejected_count
    conflicts[None] = 0
    winners[None] = bounded
    status = 1 if overflow[None] != 0 or rejected_count != 0 else 0
    overflow[None] = status
    device_dispatch_state_publish(extent_state, dispatch_packet, capacity, bounded)
    extent_state[1] = status
    if group_control[8] != 0 and group_control[9] != 0:
        block_dim = ops.cast(group_control[9], i32)
        group_control[5] = ops.cast(
            (bounded + block_dim - 1) // block_dim, u32
        )
        group_control[6] = 1 if bounded != 0 else 0
        _cuda_graph_apply_bounded_group(group_control)


@kernel
def _publish_worklist_transition(
    input_extent: ndarray_type.ndarray(dtype=i32, ndim=1),
    output_extent: ndarray_type.ndarray(dtype=i32, ndim=1),
    generated: ndarray_type.ndarray(dtype=i32, ndim=0),
    accepted: ndarray_type.ndarray(dtype=i32, ndim=0),
    rejected: ndarray_type.ndarray(dtype=i32, ndim=0),
    conflicts: ndarray_type.ndarray(dtype=i32, ndim=0),
    winners: ndarray_type.ndarray(dtype=i32, ndim=0),
    overflow: ndarray_type.ndarray(dtype=i32, ndim=0),
    resolution: i32,
):
    source_count = input_extent[0]
    output_count = output_extent[0]
    removed = ops.max(0, source_count - output_count)
    generated[None] = source_count
    accepted[None] = output_count
    rejected[None] = removed
    conflicts[None] = removed if resolution != 0 else 0
    winners[None] = output_count
    status = 1 if input_extent[1] != 0 or output_extent[1] != 0 else 0
    overflow[None] = status
    if status != 0:
        output_extent[1] = 1


@kernel
def _stage_conflict_source_indices(
    staged: ndarray_type.ndarray(dtype=i32, ndim=1),
    extent_state: ndarray_type.ndarray(dtype=i32, ndim=1),
):
    for i in staged:
        staged[i] = i if i < extent_state[0] else 0


@kernel
def _select_conflict_winners(
    sorted_keys: ndarray_type.ndarray(),
    sorted_sources: ndarray_type.ndarray(dtype=i32, ndim=1),
    priorities: ndarray_type.ndarray(dtype=i32, ndim=1),
    ordinals: ndarray_type.ndarray(dtype=i32, ndim=1),
    flags: ndarray_type.ndarray(dtype=i32, ndim=1),
    winner_sources: ndarray_type.ndarray(dtype=i32, ndim=1),
    extent_state: ndarray_type.ndarray(dtype=i32, ndim=1),
    policy: i32,
    has_priorities: i32,
    has_ordinals: i32,
):
    for i in flags:
        count = extent_state[0]
        boundary = 0
        if i < count:
            boundary = i == 0
            if i > 0:
                boundary = sorted_keys[i] != sorted_keys[i - 1]
        flags[i] = boundary
        winner_sources[i] = 0
        if boundary != 0:
            best_source = sorted_sources[i]
            best_priority = 0
            if has_priorities != 0:
                best_priority = priorities[best_source]
            best_ordinal = best_source
            if has_ordinals != 0:
                best_ordinal = ordinals[best_source]
            cursor = i + 1
            while cursor < count and sorted_keys[cursor] == sorted_keys[i]:
                source = sorted_sources[cursor]
                priority = 0
                if has_priorities != 0:
                    priority = priorities[source]
                ordinal = source
                if has_ordinals != 0:
                    ordinal = ordinals[source]
                better = False
                if policy == 1:
                    better = priority < best_priority
                elif policy == 2:
                    better = priority > best_priority
                if policy == 0 or priority == best_priority:
                    if ordinal < best_ordinal:
                        better = True
                    elif ordinal == best_ordinal and source < best_source:
                        better = True
                if better:
                    best_source = source
                    best_priority = priority
                    best_ordinal = ordinal
                cursor += 1
            winner_sources[i] = best_source


@kernel
def _materialize_conflict_winners(
    source_values: ndarray_type.ndarray(),
    priorities: ndarray_type.ndarray(dtype=i32, ndim=1),
    ordinals: ndarray_type.ndarray(dtype=i32, ndim=1),
    winner_sources: ndarray_type.ndarray(dtype=i32, ndim=1),
    output_values: ndarray_type.ndarray(),
    output_priorities: ndarray_type.ndarray(dtype=i32, ndim=1),
    output_ordinals: ndarray_type.ndarray(dtype=i32, ndim=1),
    output_extent_state: ndarray_type.ndarray(dtype=i32, ndim=1),
    has_priorities: i32,
    has_ordinals: i32,
):
    for i in output_values:
        if i < output_extent_state[0]:
            source = winner_sources[i]
            output_values[i] = source_values[source]
            output_priorities[i] = priorities[source] if has_priorities != 0 else 0
            output_ordinals[i] = ordinals[source] if has_ordinals != 0 else source
        else:
            output_values[i] = 0
            output_priorities[i] = 0
            output_ordinals[i] = 0


@dataclass(frozen=True)
class DeviceWorklistBinding:
    """Immutable allocation identity of one worklist generation."""

    capacity: int
    dtype: object
    generation: int
    value_allocation_identities: tuple
    extent_allocation_identities: tuple
    stat_allocation_identities: tuple


@dataclass(frozen=True)
class DeviceWorklistStatistics:
    """Explicit host-visible accounting for the latest transition."""

    schema_version: int
    generated: int
    accepted: int
    rejected: int
    conflicts: int
    winners: int
    overflow: bool

    @property
    def useful_count(self):
        return self.accepted


@dataclass(frozen=True)
class DeviceWorklistSnapshot:
    """Explicit synchronized snapshot of one worklist front."""

    values: np.ndarray
    extent: object
    statistics: DeviceWorklistStatistics


@dataclass(frozen=True)
class DeviceWorklistExecutionReport:
    """Unified useful/executed/overflow accounting at an explicit boundary."""

    schema_version: int
    backend: str
    route: str
    useful_count: int
    capacity: int
    executed_count: object
    skipped_count: object
    encoded_lanes: object
    generated: int
    accepted: int
    rejected: int
    conflicts: int
    winners: int
    overflow: bool
    exact_physical_grid: bool


@dataclass(frozen=True)
class DeviceConflictResult:
    """Device-owned winner arrays from deterministic keyed arbitration."""

    keys: ScalarNdarray
    values: ScalarNdarray
    priorities: ScalarNdarray
    ordinals: ScalarNdarray
    extent: DeviceExtent
    statistics: tuple
    policy: str


@dataclass(frozen=True)
class DeviceWorklistGraphArgs:
    """Symbolic Graph arguments for one fixed-capacity worklist."""

    name: str
    capacity_value: int
    dtype: object
    current_values: object
    current_extent: object
    next_values: object
    next_extent: object
    generated: object
    accepted: object
    rejected: object
    conflicts: object
    winners: object
    overflow: object
    capacity: object

    @property
    def stat_args(self):
        return (
            self.generated,
            self.accepted,
            self.rejected,
            self.conflicts,
            self.winners,
            self.overflow,
        )

    def append_arguments(self, *, target="next"):
        if target == "next":
            values, extent = self.next_values, self.next_extent
        elif target == "current":
            values, extent = self.current_values, self.current_extent
        else:
            raise ValueError("DeviceWorklist append target must be current or next")
        return (
            values,
            extent,
            self.generated,
            self.overflow,
            self.capacity,
        )

    def observe(self, builder, *, name=None):
        """Append completion-attached observation of all worklist counters."""

        builder.observe(*self.stat_args, name=name or f"{self.name}_worklist")
        return builder

    def decode_observation(self, values):
        """Decode one ticket observation into ``DeviceWorklistStatistics``."""

        if not isinstance(values, dict):
            raise TypeError("DeviceWorklist observation must be a mapping")
        decoded = {}
        for stat in _STAT_NAMES:
            key = getattr(self, stat).name
            if key not in values:
                raise ValueError(
                    f"DeviceWorklist observation is missing counter {key!r}"
                )
            decoded[stat] = int(values[key])
        return DeviceWorklistStatistics(
            schema_version=1,
            generated=decoded["generated"],
            accepted=decoded["accepted"],
            rejected=decoded["rejected"],
            conflicts=decoded["conflicts"],
            winners=decoded["winners"],
            overflow=bool(decoded["overflow"]),
        )


def device_worklist_graph_args(name, capacity, dtype=i32):
    """Create the symbolic argument bundle paired with ``runtime_arguments``."""

    if not isinstance(name, str) or not name:
        raise ValueError("DeviceWorklist Graph name must be non-empty")
    capacity = _require_capacity(capacity)
    if dtype not in _WORKLIST_DTYPES:
        raise TypeError("DeviceWorklist Graph dtype is not supported")
    from taichi_forge import graph  # pylint: disable=import-outside-toplevel

    ndarray = graph.ArgKind.NDARRAY
    scalar = graph.ArgKind.SCALAR
    return DeviceWorklistGraphArgs(
        name=name,
        capacity_value=capacity,
        dtype=dtype,
        current_values=graph.Arg(ndarray, f"{name}_current_values", dtype, ndim=1),
        current_extent=graph.Arg(ndarray, f"{name}_current_extent", i32, ndim=1),
        next_values=graph.Arg(ndarray, f"{name}_next_values", dtype, ndim=1),
        next_extent=graph.Arg(ndarray, f"{name}_next_extent", i32, ndim=1),
        generated=graph.Arg(ndarray, f"{name}_generated", i32, ndim=0),
        accepted=graph.Arg(ndarray, f"{name}_accepted", i32, ndim=0),
        rejected=graph.Arg(ndarray, f"{name}_rejected", i32, ndim=0),
        conflicts=graph.Arg(ndarray, f"{name}_conflicts", i32, ndim=0),
        winners=graph.Arg(ndarray, f"{name}_winners", i32, ndim=0),
        overflow=graph.Arg(ndarray, f"{name}_overflow", i32, ndim=0),
        capacity=graph.Arg(scalar, f"{name}_capacity", i32),
    )


def _native_key_sort_method(method):
    if method != "auto":
        return method
    arch = impl.current_cfg().arch
    if arch in (_ti_core.Arch.x64, _ti_core.Arch.arm64):
        return "cpu_native"
    if arch == _ti_core.Arch.cuda:
        return "cuda_device"
    if arch == _ti_core.Arch.vulkan:
        return "vulkan_native_radix_u32"
    raise TaichiRuntimeError(
        "DeviceWorklist deterministic claim requires CPU/CUDA/Vulkan native sort"
    )


def _stats_tuple(stats):
    return tuple(stats[name] for name in _STAT_NAMES)


def _reset_target(extent, stats):
    _reset_worklist_target(extent.state, *_stats_tuple(stats))


def _finalize_atomic_target(extent, stats, capacity, dispatch_state=None):
    if dispatch_state is None or impl.current_cfg().arch not in (
        _ti_core.Arch.cuda,
        _ti_core.Arch.vulkan,
    ):
        _finalize_atomic_worklist(extent.state, *_stats_tuple(stats), capacity)
        return
    dispatch_state.validate_extent(extent, require_identity=True)
    _finalize_atomic_worklist_with_dispatch(
        extent.state,
        *_stats_tuple(stats),
        dispatch_state.packet,
        capacity,
    )


def _publish_transition(input_extent, output_extent, stats, resolution):
    _publish_worklist_transition(
        input_extent.state,
        output_extent.state,
        *_stats_tuple(stats),
        int(bool(resolution)),
    )


def _select_impl(
    source_values,
    source_extent,
    flags,
    output_values,
    output_extent,
    stats,
    workspace,
    *,
    method,
    dispatch_state,
):
    source = DevicePrefix(source_values, source_extent, workspace=workspace)
    result = source.compact(
        flags,
        output_values,
        output_extent,
        method=method,
        dispatch_state=dispatch_state,
    )
    _publish_transition(source_extent, output_extent, stats, False)
    return result


def _resolve_impl(
    source_values,
    source_extent,
    keys,
    output_values,
    output_extent,
    stats,
    workspace,
    *,
    priorities,
    ordinals,
    output_keys,
    output_priorities,
    output_ordinals,
    policy,
    method,
    dispatch_state,
):
    capacity = source_extent.capacity
    _require_worklist_array(source_values, "conflict values", capacity)
    _require_worklist_array(keys, "conflict keys", capacity)
    if keys.dtype not in _CONFLICT_KEY_DTYPES:
        raise TypeError("deterministic conflict keys must use an integer dtype")
    if priorities is not None:
        _require_worklist_array(priorities, "conflict priorities", capacity, i32)
    if ordinals is not None:
        _require_worklist_array(ordinals, "conflict ordinals", capacity, i32)
    if policy not in ("first", "claim", "min_priority", "max_priority"):
        raise ValueError(
            "conflict policy must be first, claim, min_priority, or max_priority"
        )
    if policy in ("min_priority", "max_priority") and priorities is None:
        raise ValueError(f"conflict policy {policy!r} requires priorities")

    stage_keys = workspace._buffer("conflict_keys", keys.dtype, capacity)
    stage_sources = workspace._buffer("conflict_sources", i32, capacity)
    flags = workspace._buffer("conflict_flags", i32, capacity)
    boundary_sources = workspace._buffer("conflict_boundary_sources", i32, capacity)
    winner_sources = workspace._buffer("conflict_winner_sources", i32, capacity)
    device_prefix_copy_masked_ndarray(
        keys, stage_keys, source_extent.state, _sort_tail(keys.dtype, False, "last")
    )
    _stage_conflict_source_indices(stage_sources, source_extent.state)
    _alg.sort(
        stage_keys,
        stage_sources,
        method=_native_key_sort_method(method),
        workspace=workspace._sort,
    )
    priority_values = priorities if priorities is not None else stage_sources
    ordinal_values = ordinals if ordinals is not None else stage_sources
    effective_policy = 0
    if policy in ("claim", "min_priority") and priorities is not None:
        effective_policy = 1
    elif policy == "max_priority":
        effective_policy = 2
    _select_conflict_winners(
        stage_keys,
        stage_sources,
        priority_values,
        ordinal_values,
        flags,
        boundary_sources,
        source_extent.state,
        effective_policy,
        int(priorities is not None),
        int(ordinals is not None),
    )
    DevicePrefix(stage_keys, source_extent, workspace=workspace).compact(
        flags,
        output_keys,
        output_extent,
        method="auto",
        dispatch_state=dispatch_state,
    )
    DevicePrefix(boundary_sources, source_extent, workspace=workspace).compact(
        flags, winner_sources, output_extent, method="auto"
    )
    device_prefix_fill_tail_ndarray(winner_sources, output_extent.state, 0)
    _materialize_conflict_winners(
        source_values,
        priority_values,
        ordinal_values,
        winner_sources,
        output_values,
        output_priorities,
        output_ordinals,
        output_extent.state,
        int(priorities is not None),
        int(ordinals is not None),
    )
    _publish_transition(source_extent, output_extent, stats, True)
    workspace._refresh_usage()
    return DeviceConflictResult(
        keys=output_keys,
        values=output_values,
        priorities=output_priorities,
        ordinals=output_ordinals,
        extent=output_extent,
        statistics=_stats_tuple(stats),
        policy=policy,
    )


class DeviceWorklist:
    """Stable front/back storage for a device-driven fixed-capacity worklist."""

    def __init__(self, capacity, dtype=i32, *, workspace=None):
        capacity = _require_capacity(capacity)
        if dtype not in _WORKLIST_DTYPES:
            raise TypeError("DeviceWorklist dtype is not supported")
        if impl.get_runtime().prog is None:
            raise TaichiRuntimeError("DeviceWorklist requires an initialized runtime")
        if workspace is None:
            workspace = DevicePrefixWorkspace(capacity)
        if not isinstance(workspace, DevicePrefixWorkspace):
            raise TypeError("DeviceWorklist workspace must be DevicePrefixWorkspace")
        workspace._check_capacity(capacity)
        self._capacity = capacity
        self._dtype = dtype
        self._workspace = workspace
        self._values = (
            ti_ndarray(dtype, shape=capacity),
            ti_ndarray(dtype, shape=capacity),
        )
        self._extents = (DeviceExtent(capacity), DeviceExtent(capacity))
        self._stats = {name: ti_ndarray(i32, shape=()) for name in _STAT_NAMES}
        self._front = 0
        self._next_requires_finalize = False
        self._generation = int(impl.runtime_generation())
        self._program = impl.get_runtime().prog
        self._binding = DeviceWorklistBinding(
            capacity=capacity,
            dtype=dtype,
            generation=self._generation,
            value_allocation_identities=tuple(
                int(value._runtime_allocation_identity) for value in self._values
            ),
            extent_allocation_identities=tuple(
                extent.binding.allocation_identity for extent in self._extents
            ),
            stat_allocation_identities=tuple(
                int(self._stats[name]._runtime_allocation_identity)
                for name in _STAT_NAMES
            ),
        )
        self.clear()

    @property
    def capacity(self):
        return self._capacity

    @property
    def dtype(self):
        return self._dtype

    @property
    def binding(self):
        return self._binding

    @property
    def values(self):
        self._validate_current()
        return self._values[self._front]

    @property
    def extent(self):
        self._validate_current()
        return self._extents[self._front]

    @property
    def next_values(self):
        self._validate_current()
        return self._values[1 - self._front]

    @property
    def next_extent(self):
        self._validate_current()
        return self._extents[1 - self._front]

    @property
    def workspace(self):
        return self._workspace

    @property
    def stats(self):
        self._validate_current()
        return dict(self._stats)

    @property
    def workspace_bytes_current(self):
        self._workspace._refresh_usage()
        return self._workspace.workspace_bytes_current

    @property
    def workspace_bytes_peak(self):
        self._workspace._refresh_usage()
        return self._workspace.workspace_bytes_peak

    def _validate_current(self):
        runtime = impl.get_runtime()
        if (
            impl.runtime_generation() != self._generation
            or runtime.prog is None
            or runtime.prog is not self._program
        ):
            raise TaichiRuntimeError("DeviceWorklist is stale after runtime reset")
        for value, identity in zip(
            self._values, self._binding.value_allocation_identities
        ):
            if value.arr is None or int(value._runtime_allocation_identity) != identity:
                raise TaichiRuntimeError("DeviceWorklist storage is no longer valid")
        for extent in self._extents:
            extent._validate_current()

    def clear(self):
        """Clear both fronts and all counters without reallocating storage."""

        self._validate_current()
        _reset_target(self._extents[0], self._stats)
        self._extents[1].reset()
        self._front = 0
        self._next_requires_finalize = False
        return self

    def prepare_next(self):
        """Reset the back extent and counters before atomic production."""

        self._validate_current()
        _reset_target(self.next_extent, self._stats)
        self._next_requires_finalize = True
        return self

    def commit_next(self, *, dispatch_state=None):
        """Swap front/back ownership; this operation does not synchronize."""

        self._validate_current()
        if self._next_requires_finalize:
            _finalize_atomic_target(
                self.next_extent,
                self._stats,
                self._capacity,
                dispatch_state,
            )
            self._next_requires_finalize = False
        self._front = 1 - self._front
        return self

    def append_arguments(self, *, target="next"):
        """Return arguments consumed by :func:`device_worklist_append`."""

        self._validate_current()
        if target == "next":
            values, extent = self.next_values, self.next_extent
        elif target == "current":
            values, extent = self.values, self.extent
        else:
            raise ValueError("DeviceWorklist append target must be current or next")
        return (
            values,
            extent.state,
            self._stats["generated"],
            self._stats["overflow"],
            self._capacity,
        )

    def prefix(self):
        return DevicePrefix(self.values, self.extent, workspace=self._workspace)

    def select(self, flags, *, method="auto", dispatch_state=None):
        """Stable-select the current front into the back and commit it."""

        self._validate_current()
        _require_worklist_array(flags, "selection flags", self._capacity, i32)
        source_extent = self.extent
        _select_impl(
            self.values,
            source_extent,
            flags,
            self.next_values,
            self.next_extent,
            self._stats,
            self._workspace,
            method=method,
            dispatch_state=dispatch_state,
        )
        self._next_requires_finalize = False
        self.commit_next()
        return self

    def resolve_conflicts(
        self,
        keys,
        *,
        priorities=None,
        ordinals=None,
        policy="first",
        method="auto",
        dispatch_state=None,
    ):
        """Select one deterministic winner for every active integer key."""

        self._validate_current()
        source_extent = self.extent
        output_keys = self._workspace._buffer(
            "conflict_output_keys", keys.dtype, self._capacity
        )
        output_priorities = self._workspace._buffer(
            "conflict_output_priorities", i32, self._capacity
        )
        output_ordinals = self._workspace._buffer(
            "conflict_output_ordinals", i32, self._capacity
        )
        result = _resolve_impl(
            self.values,
            source_extent,
            keys,
            self.next_values,
            self.next_extent,
            self._stats,
            self._workspace,
            priorities=priorities,
            ordinals=ordinals,
            output_keys=output_keys,
            output_priorities=output_priorities,
            output_ordinals=output_ordinals,
            policy=policy,
            method=method,
            dispatch_state=dispatch_state,
        )
        self._next_requires_finalize = False
        self.commit_next()
        return DeviceConflictResult(
            keys=result.keys,
            values=self.values,
            priorities=result.priorities,
            ordinals=result.ordinals,
            extent=self.extent,
            statistics=result.statistics,
            policy=result.policy,
        )

    def statistics(self):
        """Synchronize and materialize the latest transition counters."""

        self._validate_current()
        values = {
            name: int(self._stats[name].to_numpy().item()) for name in _STAT_NAMES
        }
        return DeviceWorklistStatistics(
            schema_version=1,
            generated=values["generated"],
            accepted=values["accepted"],
            rejected=values["rejected"],
            conflicts=values["conflicts"],
            winners=values["winners"],
            overflow=bool(values["overflow"]),
        )

    def snapshot(self):
        """Synchronize and copy only the active front to host."""

        self._validate_current()
        extent = self.extent.snapshot()
        values = self.values.to_numpy()[: extent.count].copy()
        return DeviceWorklistSnapshot(
            values=values,
            extent=extent,
            statistics=self.statistics(),
        )

    def execution_report(self, dispatch=None, *, target="current"):
        """Synchronize and join worklist counters with bounded launch work."""

        self._validate_current()
        if target == "current":
            extent = self.extent
        elif target == "next":
            extent = self.next_extent
        else:
            raise ValueError("DeviceWorklist report target must be current or next")
        statistics = self.statistics()
        extent_snapshot = extent.snapshot()
        backend = _current_backend_name()
        route = "not_attached"
        executed = None
        skipped = None
        encoded = None
        exact = False
        overflow = statistics.overflow or extent_snapshot.overflow
        if dispatch is not None:
            snapshot = dispatch.snapshot(extent)
            route = snapshot.capabilities.execution_semantics
            executed = snapshot.executed_count
            skipped = snapshot.skipped_count
            encoded = snapshot.encoded_lanes
            exact = snapshot.capabilities.exact_grid
            overflow = overflow or snapshot.overflow
        return DeviceWorklistExecutionReport(
            schema_version=1,
            backend=backend,
            route=route,
            useful_count=extent_snapshot.count,
            capacity=self._capacity,
            executed_count=executed,
            skipped_count=skipped,
            encoded_lanes=encoded,
            generated=statistics.generated,
            accepted=statistics.accepted,
            rejected=statistics.rejected,
            conflicts=statistics.conflicts,
            winners=statistics.winners,
            overflow=overflow,
            exact_physical_grid=exact,
        )

    def graph_args(self, name):
        return device_worklist_graph_args(name, self._capacity, self._dtype)

    def runtime_arguments(self, name, *, include_capacity=False):
        """Bind this worklist to :func:`device_worklist_graph_args`."""

        self._validate_current()
        result = {
            f"{name}_current_values": self.values,
            f"{name}_current_extent": self.extent,
            f"{name}_next_values": self.next_values,
            f"{name}_next_extent": self.next_extent,
        }
        if include_capacity:
            result[f"{name}_capacity"] = self._capacity
        result.update((f"{name}_{key}", value) for key, value in self._stats.items())
        return result

    def memory_report(self):
        self._validate_current()
        front_back = 2 * self._capacity * _dtype_bytes(self._dtype)
        owned = front_back + 16 + 24
        return {
            "schema_version": 1,
            "capacity": self._capacity,
            "front_back_value_bytes": front_back,
            "extent_bytes": 16,
            "counter_bytes": 24,
            "workspace_bytes_current": self.workspace_bytes_current,
            "workspace_bytes_peak": self.workspace_bytes_peak,
            "total_bytes_current": owned + self.workspace_bytes_current,
            "total_bytes_peak": owned + self.workspace_bytes_peak,
            "fixed_capacity": True,
            "replay_allocation_count": 0,
        }


def _symbolic_arg(value, role, *, dtype=None, ndim=None):
    if getattr(value, "tag", None) != _ti_core.ArgKind.NDARRAY:
        raise TypeError(f"DeviceWorklistSequence {role} must be a Graph ndarray Arg")
    if dtype is not None and value.dtype() != dtype:
        raise TypeError(f"DeviceWorklistSequence {role} must use {dtype}")
    if ndim is not None and int(value.field_dim) != ndim:
        raise TypeError(f"DeviceWorklistSequence {role} must use ndim={ndim}")
    if getattr(value, "element_shape", ()):
        raise TypeError(f"DeviceWorklistSequence {role} must contain scalars")
    return value


class DeviceWorklistSequence:
    """Record one worklist transition as a reusable native Graph node."""

    def __init__(self, args, *, workspace=None):
        if not isinstance(args, DeviceWorklistGraphArgs):
            raise TypeError("DeviceWorklistSequence requires DeviceWorklistGraphArgs")
        self.args = args
        self.capacity = args.capacity_value
        self.workspace = (
            DevicePrefixWorkspace(self.capacity) if workspace is None else workspace
        )
        if not isinstance(self.workspace, DevicePrefixWorkspace):
            raise TypeError(
                "DeviceWorklistSequence workspace must be DevicePrefixWorkspace"
            )
        self.workspace._check_capacity(self.capacity)
        self._operation = None
        self._compiled = False
        self._arg_descriptors = {}
        self._leases = []
        for value, role, dtype, ndim in (
            (args.current_values, "current values", args.dtype, 1),
            (args.current_extent, "current extent", i32, 1),
            (args.next_values, "next values", args.dtype, 1),
            (args.next_extent, "next extent", i32, 1),
            *(
                (value, name, i32, 0)
                for value, name in zip(args.stat_args, _STAT_NAMES)
            ),
        ):
            self._register(value, role, dtype=dtype, ndim=ndim)

    @property
    def workspace_bytes_current(self):
        self.workspace._refresh_usage()
        return self.workspace.workspace_bytes_current

    @property
    def workspace_bytes_peak(self):
        self.workspace._refresh_usage()
        return self.workspace.workspace_bytes_peak

    def memory_report(self):
        return {
            "schema_version": 1,
            "capacity": self.capacity,
            "workspace_bytes_current": self.workspace_bytes_current,
            "workspace_bytes_peak": self.workspace_bytes_peak,
            "workspace_allocation_count": self.workspace.allocation_count,
            "replay_allocation_count": 0,
        }

    def _ensure_mutable(self):
        if self._compiled:
            raise TaichiRuntimeError(
                "DeviceWorklistSequence cannot change after Graph compilation"
            )

    def _register(self, value, role, *, dtype=None, ndim=None):
        value = _symbolic_arg(value, role, dtype=dtype, ndim=ndim)
        descriptor = (
            value.tag,
            str(value.dtype()),
            int(value.field_dim),
            tuple(value.element_shape),
        )
        previous = self._arg_descriptors.get(value.name)
        if previous is not None and previous != descriptor:
            raise ValueError(
                f"DeviceWorklistSequence argument {value.name!r} changes descriptor"
            )
        self._arg_descriptors[value.name] = descriptor
        return value

    def _set_operation(self, kind, values, options):
        self._ensure_mutable()
        if self._operation is not None:
            raise TaichiRuntimeError(
                "DeviceWorklistSequence records one transition per native node"
            )
        self._operation = (kind, tuple(values), dict(options))
        return self

    def prepare_next(self):
        """Record a reset before a user-defined atomic producer dispatch."""

        return self._set_operation("reset", (), {})

    def finalize_next(self, *, dispatch_state=None):
        """Record counter/extent publication after an atomic producer."""

        self._ensure_mutable()
        if dispatch_state is not None:
            if not isinstance(dispatch_state, DeviceDispatchState):
                raise TypeError("worklist dispatch_state must be DeviceDispatchState")
            dispatch_state._validate_current()
            if dispatch_state.capacity != self.capacity:
                raise ValueError("worklist dispatch_state capacity mismatch")
            self._leases.append(dispatch_state)
        return self._set_operation("finalize", (), {"dispatch_state": dispatch_state})

    def select(self, flags, *, method="auto", dispatch_state=None):
        self._ensure_mutable()
        flags = self._register(flags, "flags", dtype=i32, ndim=1)
        # Graph native actions execute inside one backend submission batch.
        # Allocate Python-owned staging now: creating an ndarray from run()
        # would force a Vulkan stream synchronization inside that batch.
        self.workspace._buffer("compact_flags", i32, self.capacity)
        if dispatch_state is not None:
            if not isinstance(dispatch_state, DeviceDispatchState):
                raise TypeError("worklist dispatch_state must be DeviceDispatchState")
            dispatch_state._validate_current()
            if dispatch_state.capacity != self.capacity:
                raise ValueError("worklist dispatch_state capacity mismatch")
            self._leases.append(dispatch_state)
        return self._set_operation(
            "select",
            (flags.name,),
            {"method": method, "dispatch_state": dispatch_state},
        )

    def resolve_conflicts(
        self,
        keys,
        output_keys,
        output_priorities,
        output_ordinals,
        *,
        priorities=None,
        ordinals=None,
        policy="first",
        method="auto",
        dispatch_state=None,
    ):
        self._ensure_mutable()
        keys = self._register(keys, "keys", ndim=1)
        output_keys = self._register(
            output_keys, "output keys", dtype=keys.dtype(), ndim=1
        )
        output_priorities = self._register(
            output_priorities, "output priorities", dtype=i32, ndim=1
        )
        output_ordinals = self._register(
            output_ordinals, "output ordinals", dtype=i32, ndim=1
        )
        for role, dtype in (
            ("conflict_keys", keys.dtype()),
            ("conflict_sources", i32),
            ("conflict_flags", i32),
            ("conflict_boundary_sources", i32),
            ("conflict_winner_sources", i32),
            ("compact_flags", i32),
        ):
            self.workspace._buffer(role, dtype, self.capacity)
        if priorities is not None:
            priorities = self._register(priorities, "priorities", dtype=i32, ndim=1)
        if ordinals is not None:
            ordinals = self._register(ordinals, "ordinals", dtype=i32, ndim=1)
        if dispatch_state is not None:
            if not isinstance(dispatch_state, DeviceDispatchState):
                raise TypeError("worklist dispatch_state must be DeviceDispatchState")
            dispatch_state._validate_current()
            if dispatch_state.capacity != self.capacity:
                raise ValueError("worklist dispatch_state capacity mismatch")
            self._leases.append(dispatch_state)
        return self._set_operation(
            "resolve",
            (
                keys.name,
                output_keys.name,
                output_priorities.name,
                output_ordinals.name,
                None if priorities is None else priorities.name,
                None if ordinals is None else ordinals.name,
            ),
            {
                "policy": policy,
                "method": method,
                "dispatch_state": dispatch_state,
            },
        )

    def _as_graph_native_node(self):
        self._ensure_mutable()
        if self._operation is None:
            raise TaichiRuntimeError("DeviceWorklistSequence has no transition")
        self._compiled = True
        return _DeviceWorklistSequenceNode(self)


class _DeviceWorklistSequenceExecutable(NativeGraphExecutable):
    def __init__(self, sequence):
        self._args = sequence.args
        self._workspace = sequence.workspace
        self._operation = sequence._operation
        self._arg_names = tuple(sequence._arg_descriptors)
        self._leases = tuple(dict.fromkeys(sequence._leases))

    @property
    def runtime_arg_schema(self):
        return tuple(RuntimeBinding(name, "ndarray") for name in self._arg_names)

    @property
    def resource_effects(self):
        return tuple(
            ResourceEffect(name, GraphAccess.READ_WRITE) for name in self._arg_names
        )

    @property
    def lifetime_leases(self):
        return self._leases

    @property
    def _cuda_bounded_fusion_state(self):
        kind, _, options = self._operation
        if kind != "finalize" or impl.current_cfg().arch != _ti_core.Arch.cuda:
            return None
        return options.get("dispatch_state")

    def _cuda_bounded_fusion_action(
        self,
        group_control_arg,
        group_control,
        dispatch_packet_arg,
        capacity_arg,
    ):
        kind, _, options = self._operation
        dispatch_state = options.get("dispatch_state")
        if (
            kind != "finalize"
            or dispatch_state is None
            or impl.current_cfg().arch != _ti_core.Arch.cuda
        ):
            return None
        dispatch_state._validate_current()
        from taichi_forge.graph._graph import gen_cpp_kernel

        if group_control is None:
            symbolic_args = (
                self._args.next_extent,
                *self._args.stat_args,
                dispatch_packet_arg,
                capacity_arg,
            )
            kernel_cpp = gen_cpp_kernel(
                _finalize_atomic_worklist_with_dispatch, symbolic_args
            )
            fixed_bindings = {
                dispatch_packet_arg.name: dispatch_state.packet,
                capacity_arg.name: self._args.capacity_value,
            }
        else:
            symbolic_args = (
                self._args.next_extent,
                *self._args.stat_args,
                dispatch_packet_arg,
                group_control_arg,
                capacity_arg,
            )
            kernel_cpp = gen_cpp_kernel(
                _finalize_atomic_worklist_with_cuda_group, symbolic_args
            )
            fixed_bindings = {
                dispatch_packet_arg.name: dispatch_state.packet,
                group_control_arg.name: group_control,
                capacity_arg.name: self._args.capacity_value,
            }
        action = DispatchGraphAction(
            ((kernel_cpp, symbolic_args),),
            backends=("cuda",),
            conditional_body_safe=False,
            fixed_bindings=fixed_bindings,
            update_policy="immutable",
        )
        return action, dispatch_state

    def _stats(self, runtime_args):
        return {
            name: runtime_args[getattr(self._args, name).name] for name in _STAT_NAMES
        }

    def run(self, runtime_args=None):
        if runtime_args is None:
            raise TaichiRuntimeError(
                "DeviceWorklistSequence requires Graph runtime arguments"
            )
        args = self._args
        current_values = runtime_args[args.current_values.name]
        current_extent = runtime_args[args.current_extent.name]
        next_values = runtime_args[args.next_values.name]
        next_extent = runtime_args[args.next_extent.name]
        stats = self._stats(runtime_args)
        kind, values, options = self._operation
        if kind == "reset":
            _reset_target(next_extent, stats)
        elif kind == "finalize":
            _finalize_atomic_target(
                next_extent,
                stats,
                self._args.capacity_value,
                options["dispatch_state"],
            )
        elif kind == "select":
            _select_impl(
                current_values,
                current_extent,
                runtime_args[values[0]],
                next_values,
                next_extent,
                stats,
                self._workspace,
                method=options["method"],
                dispatch_state=options["dispatch_state"],
            )
        elif kind == "resolve":
            (
                keys,
                output_keys,
                output_priorities,
                output_ordinals,
                priorities,
                ordinals,
            ) = values
            _resolve_impl(
                current_values,
                current_extent,
                runtime_args[keys],
                next_values,
                next_extent,
                stats,
                self._workspace,
                priorities=(None if priorities is None else runtime_args[priorities]),
                ordinals=None if ordinals is None else runtime_args[ordinals],
                output_keys=runtime_args[output_keys],
                output_priorities=runtime_args[output_priorities],
                output_ordinals=runtime_args[output_ordinals],
                policy=options["policy"],
                method=options["method"],
                dispatch_state=options["dispatch_state"],
            )
        else:
            raise TaichiRuntimeError(f"Unsupported worklist operation {kind!r}")

    @property
    def debug_info(self):
        return {
            "kind": "device_worklist_sequence",
            "operation": self._operation[0],
            "capacity": self._args.capacity_value,
            "counter_count": len(_STAT_NAMES),
            "workspace_bytes_peak": self._workspace.workspace_bytes_peak,
        }


class _DeviceWorklistSequenceNode(NativeGraphNode):
    def __init__(self, sequence):
        self._sequence = sequence

    def compile(self):
        return _DeviceWorklistSequenceExecutable(self._sequence)


__all__ = [
    "DeviceConflictResult",
    "DeviceWorklist",
    "DeviceWorklistBinding",
    "DeviceWorklistGraphArgs",
    "DeviceWorklistExecutionReport",
    "DeviceWorklistSequence",
    "DeviceWorklistSnapshot",
    "DeviceWorklistStatistics",
    "device_worklist_append",
    "device_worklist_graph_args",
]
