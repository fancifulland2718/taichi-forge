"""Fixed-capacity device worklists and deterministic keyed arbitration.

The worklist owns stable front/back storage, :class:`DeviceExtent` state, and
small device counters.  Producers may append atomically without reading the
count on host.  Stable selection and keyed claim paths reuse Forge's existing
device prefix, compact, and native stable-sort providers.
"""

import itertools
import os
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
_OPTIONAL_STAT_NAMES = (
    "accepted",
    "rejected",
    "conflicts",
    "winners",
)
_STAT_NAMES = (
    "generated",
    *_OPTIONAL_STAT_NAMES,
    "overflow",
)
_STATE_NAMES = (*_STAT_NAMES, "generation")
_WORKLIST_RECORDING_IDS = itertools.count(1)


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


def _require_transition_mode(value):
    if value not in ("staged", "direct"):
        raise ValueError("DeviceWorklist transition_mode must be staged or direct")
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


def _require_dense_winner_table(value, key_capacity):
    if (
        not isinstance(value, ScalarNdarray)
        or value.dtype != i32
        or tuple(value.shape) != (key_capacity,)
    ):
        raise TypeError(
            "dense winner table must be an i32 ndarray with shape "
            f"({key_capacity},)"
        )
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


@func
def device_worklist_append_direct(
    values: template(),
    extent_state: template(),
    overflow: template(),
    capacity: i32,
    value: template(),
):
    """Append directly into a bounded extent without a finalize dispatch.

    This lower-overhead contract deliberately omits an exact generated-count
    statistic.  Overflow is sticky and the published extent remains clamped.
    """

    result = -1
    if capacity != values.shape[0]:
        ops.atomic_or(overflow[None], 1)
        ops.atomic_or(extent_state[1], 1)
    else:
        slot = ops.atomic_add(extent_state[0], 1)
        if slot < capacity:
            values[slot] = value
            result = slot
        else:
            ops.atomic_min(extent_state[0], capacity)
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
def _reset_worklist_target_minimal(
    extent_state: ndarray_type.ndarray(dtype=i32, ndim=1),
    generated: ndarray_type.ndarray(dtype=i32, ndim=0),
    overflow: ndarray_type.ndarray(dtype=i32, ndim=0),
):
    extent_state[0] = 0
    extent_state[1] = 0
    generated[None] = 0
    overflow[None] = 0


@kernel
def _begin_direct_worklist_transition(
    extent_state: ndarray_type.ndarray(dtype=i32, ndim=1),
    overflow: ndarray_type.ndarray(dtype=i32, ndim=0),
    generation: ndarray_type.ndarray(dtype=i32, ndim=0),
    advance_generation: i32,
):
    extent_state[0] = 0
    extent_state[1] = 0
    overflow[None] = 0
    if advance_generation != 0:
        generation[None] += 1


@kernel
def _finalize_atomic_worklist(
    extent_state: ndarray_type.ndarray(dtype=i32, ndim=1),
    generated: ndarray_type.ndarray(dtype=i32, ndim=0),
    accepted: ndarray_type.ndarray(dtype=i32, ndim=0),
    rejected: ndarray_type.ndarray(dtype=i32, ndim=0),
    conflicts: ndarray_type.ndarray(dtype=i32, ndim=0),
    winners: ndarray_type.ndarray(dtype=i32, ndim=0),
    overflow: ndarray_type.ndarray(dtype=i32, ndim=0),
    generation: ndarray_type.ndarray(dtype=i32, ndim=0),
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
    generation[None] += 1


@kernel
def _finalize_atomic_worklist_minimal(
    extent_state: ndarray_type.ndarray(dtype=i32, ndim=1),
    generated: ndarray_type.ndarray(dtype=i32, ndim=0),
    overflow: ndarray_type.ndarray(dtype=i32, ndim=0),
    generation: ndarray_type.ndarray(dtype=i32, ndim=0),
    capacity: i32,
):
    raw = generated[None]
    bounded = ops.min(ops.max(raw, 0), capacity)
    status = 1 if overflow[None] != 0 or raw != bounded else 0
    extent_state[0] = bounded
    extent_state[1] = status
    overflow[None] = status
    generation[None] += 1


@kernel
def _finalize_atomic_worklist_with_dispatch(
    extent_state: ndarray_type.ndarray(dtype=i32, ndim=1),
    generated: ndarray_type.ndarray(dtype=i32, ndim=0),
    accepted: ndarray_type.ndarray(dtype=i32, ndim=0),
    rejected: ndarray_type.ndarray(dtype=i32, ndim=0),
    conflicts: ndarray_type.ndarray(dtype=i32, ndim=0),
    winners: ndarray_type.ndarray(dtype=i32, ndim=0),
    overflow: ndarray_type.ndarray(dtype=i32, ndim=0),
    generation: ndarray_type.ndarray(dtype=i32, ndim=0),
    dispatch_packet: ndarray_type.ndarray(dtype=u32, ndim=1),
    capacity: i32,
    block_dim: i32,
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
    dispatch_packet[3] = ops.cast(block_dim, u32)
    device_dispatch_state_publish(extent_state, dispatch_packet, capacity, bounded)
    extent_state[1] = status
    generation[None] += 1


@kernel
def _finalize_atomic_worklist_minimal_with_dispatch(
    extent_state: ndarray_type.ndarray(dtype=i32, ndim=1),
    generated: ndarray_type.ndarray(dtype=i32, ndim=0),
    overflow: ndarray_type.ndarray(dtype=i32, ndim=0),
    generation: ndarray_type.ndarray(dtype=i32, ndim=0),
    dispatch_packet: ndarray_type.ndarray(dtype=u32, ndim=1),
    capacity: i32,
    block_dim: i32,
):
    raw = generated[None]
    bounded = ops.min(ops.max(raw, 0), capacity)
    status = 1 if overflow[None] != 0 or raw != bounded else 0
    overflow[None] = status
    dispatch_packet[3] = ops.cast(block_dim, u32)
    device_dispatch_state_publish(extent_state, dispatch_packet, capacity, bounded)
    extent_state[1] = status
    generation[None] += 1


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
    generation: ndarray_type.ndarray(dtype=i32, ndim=0),
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
    generation[None] += 1


@kernel
def _publish_worklist_transition_minimal(
    input_extent: ndarray_type.ndarray(dtype=i32, ndim=1),
    output_extent: ndarray_type.ndarray(dtype=i32, ndim=1),
    generated: ndarray_type.ndarray(dtype=i32, ndim=0),
    overflow: ndarray_type.ndarray(dtype=i32, ndim=0),
    generation: ndarray_type.ndarray(dtype=i32, ndim=0),
):
    status = 1 if input_extent[1] != 0 or output_extent[1] != 0 else 0
    generated[None] = input_extent[0]
    overflow[None] = status
    if status != 0:
        output_extent[1] = 1
    generation[None] += 1


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


@kernel
def _reset_dense_conflict_workspace(
    best_priorities: ndarray_type.ndarray(dtype=i32, ndim=1),
    best_ordinals: ndarray_type.ndarray(dtype=i32, ndim=1),
    best_sources: ndarray_type.ndarray(dtype=i32, ndim=1),
    flags: ndarray_type.ndarray(dtype=i32, ndim=1),
    invalid: ndarray_type.ndarray(dtype=i32, ndim=1),
    policy: i32,
):
    for i in flags:
        best_priorities[i] = (
            -0x80000000 if policy == 2 else 0x7FFFFFFF
        )
        best_ordinals[i] = 0x7FFFFFFF
        best_sources[i] = 0x7FFFFFFF
        flags[i] = 0
    invalid[0] = 0


@kernel
def _reset_dense_conflict_table(
    best_priorities: ndarray_type.ndarray(dtype=i32, ndim=1),
    best_ordinals: ndarray_type.ndarray(dtype=i32, ndim=1),
    best_sources: ndarray_type.ndarray(dtype=i32, ndim=1),
    invalid: ndarray_type.ndarray(dtype=i32, ndim=1),
    policy: i32,
):
    for key in best_sources:
        best_priorities[key] = (
            -0x80000000 if policy == 2 else 0x7FFFFFFF
        )
        best_ordinals[key] = 0x7FFFFFFF
        best_sources[key] = 0x7FFFFFFF
    invalid[0] = 0


@kernel
def _publish_dense_conflict_table(
    input_extent: ndarray_type.ndarray(dtype=i32, ndim=1),
    invalid: ndarray_type.ndarray(dtype=i32, ndim=1),
    generated: ndarray_type.ndarray(dtype=i32, ndim=0),
    overflow: ndarray_type.ndarray(dtype=i32, ndim=0),
    generation: ndarray_type.ndarray(dtype=i32, ndim=0),
):
    generated[None] = input_extent[0]
    overflow[None] = 1 if input_extent[1] != 0 or invalid[0] != 0 else 0
    generation[None] += 1


@kernel
def _select_dense_conflict_priorities(
    keys: ndarray_type.ndarray(),
    priorities: ndarray_type.ndarray(dtype=i32, ndim=1),
    best_priorities: ndarray_type.ndarray(dtype=i32, ndim=1),
    invalid: ndarray_type.ndarray(dtype=i32, ndim=1),
    extent_state: ndarray_type.ndarray(dtype=i32, ndim=1),
    key_capacity: i32,
    policy: i32,
):
    for source in keys:
        if source < extent_state[0]:
            raw_key = keys[source]
            if raw_key >= 0 and raw_key < key_capacity:
                key = ops.cast(raw_key, i32)
                if policy == 1:
                    ops.atomic_min(best_priorities[key], priorities[source])
                elif policy == 2:
                    ops.atomic_max(best_priorities[key], priorities[source])
            else:
                ops.atomic_or(invalid[0], 1)


@kernel
def _select_dense_conflict_ordinals(
    keys: ndarray_type.ndarray(),
    priorities: ndarray_type.ndarray(dtype=i32, ndim=1),
    ordinals: ndarray_type.ndarray(dtype=i32, ndim=1),
    best_priorities: ndarray_type.ndarray(dtype=i32, ndim=1),
    best_ordinals: ndarray_type.ndarray(dtype=i32, ndim=1),
    extent_state: ndarray_type.ndarray(dtype=i32, ndim=1),
    key_capacity: i32,
    policy: i32,
):
    for source in keys:
        if source < extent_state[0]:
            raw_key = keys[source]
            if raw_key >= 0 and raw_key < key_capacity:
                key = ops.cast(raw_key, i32)
                if policy == 0 or priorities[source] == best_priorities[key]:
                    ops.atomic_min(best_ordinals[key], ordinals[source])


@kernel
def _select_dense_conflict_sources(
    keys: ndarray_type.ndarray(),
    priorities: ndarray_type.ndarray(dtype=i32, ndim=1),
    ordinals: ndarray_type.ndarray(dtype=i32, ndim=1),
    best_priorities: ndarray_type.ndarray(dtype=i32, ndim=1),
    best_ordinals: ndarray_type.ndarray(dtype=i32, ndim=1),
    best_sources: ndarray_type.ndarray(dtype=i32, ndim=1),
    extent_state: ndarray_type.ndarray(dtype=i32, ndim=1),
    key_capacity: i32,
    policy: i32,
):
    for source in keys:
        if source < extent_state[0]:
            raw_key = keys[source]
            if raw_key >= 0 and raw_key < key_capacity:
                key = ops.cast(raw_key, i32)
                priority_matches = (
                    policy == 0 or priorities[source] == best_priorities[key]
                )
                if priority_matches and ordinals[source] == best_ordinals[key]:
                    ops.atomic_min(best_sources[key], source)


@kernel
def _mark_dense_conflict_winners(
    best_sources: ndarray_type.ndarray(dtype=i32, ndim=1),
    flags: ndarray_type.ndarray(dtype=i32, ndim=1),
    key_capacity: i32,
):
    for key in flags:
        flags[key] = (
            1
            if key < key_capacity and best_sources[key] != 0x7FFFFFFF
            else 0
        )


@kernel
def _emit_dense_conflict_winners(
    flags: ndarray_type.ndarray(dtype=i32, ndim=1),
    best_sources: ndarray_type.ndarray(dtype=i32, ndim=1),
    invalid: ndarray_type.ndarray(dtype=i32, ndim=1),
    output_keys: ndarray_type.ndarray(),
    winner_sources: ndarray_type.ndarray(dtype=i32, ndim=1),
    output_extent_state: ndarray_type.ndarray(dtype=i32, ndim=1),
    key_capacity: i32,
):
    for key in flags:
        if key < key_capacity:
            previous = flags[key - 1] if key > 0 else 0
            if flags[key] != previous:
                output = flags[key] - 1
                output_keys[output] = key
                winner_sources[output] = best_sources[key]
        if key == flags.shape[0] - 1:
            output_extent_state[0] = flags[key]
            output_extent_state[1] = invalid[0]


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
    accepted: object
    rejected: object
    conflicts: object
    winners: object
    overflow: bool
    generation: int = 0
    telemetry_available: bool = True

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
    generation: int = 0
    telemetry_available: bool = True


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
    strategy: str
    sort_method: object
    key_capacity: object
    output_shape: str = "compact_winner_list"
    dense_winner_sources: object = None


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
    generation: object
    capacity: object
    telemetry: bool = True
    transition_mode: str = "staged"

    @property
    def stat_args(self):
        return tuple(
            value
            for value in (
                self.generated,
                self.accepted,
                self.rejected,
                self.conflicts,
                self.winners,
                self.overflow,
            )
            if value is not None
        )

    @property
    def state_args(self):
        return (*self.stat_args, self.generation)

    def append_arguments(self, *, target="next"):
        if target == "next":
            values, extent = self.next_values, self.next_extent
        elif target == "current":
            values, extent = self.current_values, self.current_extent
        else:
            raise ValueError("DeviceWorklist append target must be current or next")
        if self.transition_mode == "direct":
            return values, extent, self.overflow, self.capacity
        return values, extent, self.generated, self.overflow, self.capacity

    def observe(self, builder, *, name=None):
        """Append completion-attached observation of all worklist counters."""

        if not self.telemetry:
            raise TaichiRuntimeError(
                "DeviceWorklist optional telemetry was disabled for this Graph ABI"
            )
        builder.observe(
            *self.state_args, name=name or f"{self.name}_worklist"
        )
        return builder

    def decode_observation(self, values):
        """Decode one ticket observation into ``DeviceWorklistStatistics``."""

        if not isinstance(values, dict):
            raise TypeError("DeviceWorklist observation must be a mapping")
        decoded = {}
        if not self.telemetry:
            raise TaichiRuntimeError(
                "DeviceWorklist optional telemetry was disabled for this Graph ABI"
            )
        for stat in _STAT_NAMES:
            key = getattr(self, stat).name
            if key not in values:
                raise ValueError(
                    f"DeviceWorklist observation is missing counter {key!r}"
                )
            decoded[stat] = int(values[key])
        return DeviceWorklistStatistics(
            schema_version=2,
            generated=decoded["generated"],
            accepted=decoded["accepted"],
            rejected=decoded["rejected"],
            conflicts=decoded["conflicts"],
            winners=decoded["winners"],
            overflow=bool(decoded["overflow"]),
            generation=int(values[self.generation.name]),
            telemetry_available=True,
        )


def device_worklist_graph_args(
    name,
    capacity,
    dtype=i32,
    *,
    telemetry=True,
    transition_mode="staged",
):
    """Create the symbolic argument bundle paired with ``runtime_arguments``."""

    if not isinstance(name, str) or not name:
        raise ValueError("DeviceWorklist Graph name must be non-empty")
    capacity = _require_capacity(capacity)
    if dtype not in _WORKLIST_DTYPES:
        raise TypeError("DeviceWorklist Graph dtype is not supported")
    if not isinstance(telemetry, bool):
        raise TypeError("DeviceWorklist Graph telemetry must be a bool")
    transition_mode = _require_transition_mode(transition_mode)
    if transition_mode == "direct" and telemetry:
        raise ValueError(
            "direct DeviceWorklist transitions require telemetry=False"
        )
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
        generated=(
            graph.Arg(ndarray, f"{name}_generated", i32, ndim=0)
            if transition_mode == "staged"
            else None
        ),
        accepted=(
            graph.Arg(ndarray, f"{name}_accepted", i32, ndim=0)
            if telemetry
            else None
        ),
        rejected=(
            graph.Arg(ndarray, f"{name}_rejected", i32, ndim=0)
            if telemetry
            else None
        ),
        conflicts=(
            graph.Arg(ndarray, f"{name}_conflicts", i32, ndim=0)
            if telemetry
            else None
        ),
        winners=(
            graph.Arg(ndarray, f"{name}_winners", i32, ndim=0)
            if telemetry
            else None
        ),
        overflow=graph.Arg(ndarray, f"{name}_overflow", i32, ndim=0),
        generation=graph.Arg(ndarray, f"{name}_generation", i32, ndim=0),
        capacity=graph.Arg(scalar, f"{name}_capacity", i32),
        telemetry=telemetry,
        transition_mode=transition_mode,
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


def _normalize_conflict_sort_method(method, sort_method):
    if sort_method is None:
        sort_method = method
    elif method != "auto":
        raise ValueError(
            "DeviceWorklist conflict method= is the backward-compatible sort "
            "provider alias and cannot be combined with sort_method="
        )
    if not isinstance(sort_method, str) or not sort_method:
        raise TypeError("DeviceWorklist conflict sort_method must be a string")
    return sort_method


def _normalize_conflict_output_shape(output_shape):
    aliases = {
        "compact": "compact_winner_list",
        "compact_winner_list": "compact_winner_list",
        "dense": "dense_winner_table",
        "dense_winner_table": "dense_winner_table",
    }
    try:
        return aliases[output_shape]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "conflict output_shape must be compact_winner_list or "
            "dense_winner_table"
        ) from exc


def _choose_conflict_strategy(strategy, key_capacity, capacity):
    if strategy not in ("auto", "dense_atomic", "radix_grouped"):
        raise ValueError(
            "conflict strategy must be auto, dense_atomic, or radix_grouped"
        )
    if key_capacity is not None:
        key_capacity = _require_capacity(key_capacity, "key_capacity")
        if key_capacity > capacity:
            raise ValueError(
                "DeviceWorklist dense key_capacity cannot exceed worklist capacity"
            )
    if strategy == "dense_atomic":
        if key_capacity is None:
            raise ValueError("dense_atomic conflict strategy requires key_capacity")
        return strategy, key_capacity, "explicit_dense_domain"
    if strategy == "radix_grouped":
        return strategy, key_capacity, "explicit_radix"
    # The dense path wins decisively once the problem is large enough to
    # amortize its fixed reset/scan passes.  Small CPU worklists are the one
    # measured exception: radix grouping is slightly cheaper there even for a
    # compact key domain.  Keep explicit ``dense_atomic`` available for users
    # who know their workload, but make ``auto`` conservative at that crossover.
    cpu_small_problem = (
        impl.current_cfg().arch == _ti_core.Arch.x64 and capacity < 4096
    )
    if (
        key_capacity is not None
        and key_capacity * 4 <= capacity
        and not cpu_small_problem
    ):
        return "dense_atomic", key_capacity, "bounded_dense_domain"
    if key_capacity is not None and cpu_small_problem:
        return "radix_grouped", key_capacity, "small_cpu_radix_fallback"
    return "radix_grouped", key_capacity, "conservative_radix_fallback"


def _stats_tuple(stats):
    return tuple(stats[name] for name in _STAT_NAMES)


def _telemetry_enabled(stats):
    return all(name in stats for name in _OPTIONAL_STAT_NAMES)


def _statistics_tuple(stats):
    if not _telemetry_enabled(stats):
        return ()
    return _stats_tuple(stats)


def _reset_target(extent, stats):
    if _telemetry_enabled(stats):
        _reset_worklist_target(extent.state, *_stats_tuple(stats))
    else:
        _reset_worklist_target_minimal(
            extent.state, stats["generated"], stats["overflow"]
        )


def _finalize_atomic_target(extent, stats, capacity, dispatch_state=None):
    if dispatch_state is not None and impl.current_cfg().arch == _ti_core.Arch.cuda:
        raise TaichiRuntimeError(
            "CUDA worklist publication does not produce a consumer-owned "
            "dispatch packet; pass the DeviceExtent directly to bounded consumers"
        )
    if dispatch_state is None or impl.current_cfg().arch != _ti_core.Arch.vulkan:
        if _telemetry_enabled(stats):
            _finalize_atomic_worklist(
                extent.state,
                *_stats_tuple(stats),
                stats["generation"],
                capacity,
            )
        else:
            _finalize_atomic_worklist_minimal(
                extent.state,
                stats["generated"],
                stats["overflow"],
                stats["generation"],
                capacity,
            )
        return
    dispatch_state.validate_extent(extent, require_identity=True)
    if _telemetry_enabled(stats):
        _finalize_atomic_worklist_with_dispatch(
            extent.state,
            *_stats_tuple(stats),
            stats["generation"],
            dispatch_state.packet,
            capacity,
            dispatch_state.block_dim,
        )
    else:
        _finalize_atomic_worklist_minimal_with_dispatch(
            extent.state,
            stats["generated"],
            stats["overflow"],
            stats["generation"],
            dispatch_state.packet,
            capacity,
            dispatch_state.block_dim,
        )


def _publish_transition(input_extent, output_extent, stats, resolution):
    if _telemetry_enabled(stats):
        _publish_worklist_transition(
            input_extent.state,
            output_extent.state,
            *_stats_tuple(stats),
            stats["generation"],
            int(bool(resolution)),
        )
    else:
        _publish_worklist_transition_minimal(
            input_extent.state,
            output_extent.state,
            stats["generated"],
            stats["overflow"],
            stats["generation"],
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


def _resolve_dense_table_impl(
    source_extent,
    keys,
    dense_winner_sources,
    stats,
    workspace,
    *,
    priorities,
    ordinals,
    policy,
    key_capacity,
):
    capacity = source_extent.capacity
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
    _require_dense_winner_table(dense_winner_sources, key_capacity)
    if _telemetry_enabled(stats):
        raise ValueError(
            "dense winner-only conflict resolution requires telemetry=False; "
            "winner-count telemetry would require the compact scan being skipped"
        )

    stage_sources = workspace._buffer("conflict_sources", i32, capacity)
    best_priorities = workspace._buffer(
        "dense_conflict_priorities", i32, key_capacity
    )
    best_ordinals = workspace._buffer(
        "dense_conflict_ordinals", i32, key_capacity
    )
    invalid = workspace._buffer("dense_conflict_invalid", i32, 1)
    _stage_conflict_source_indices(stage_sources, source_extent.state)
    priority_values = priorities if priorities is not None else stage_sources
    ordinal_values = ordinals if ordinals is not None else stage_sources
    effective_policy = 0
    if policy in ("claim", "min_priority") and priorities is not None:
        effective_policy = 1
    elif policy == "max_priority":
        effective_policy = 2
    _reset_dense_conflict_table(
        best_priorities,
        best_ordinals,
        dense_winner_sources,
        invalid,
        effective_policy,
    )
    _select_dense_conflict_priorities(
        keys,
        priority_values,
        best_priorities,
        invalid,
        source_extent.state,
        key_capacity,
        effective_policy,
    )
    _select_dense_conflict_ordinals(
        keys,
        priority_values,
        ordinal_values,
        best_priorities,
        best_ordinals,
        source_extent.state,
        key_capacity,
        effective_policy,
    )
    _select_dense_conflict_sources(
        keys,
        priority_values,
        ordinal_values,
        best_priorities,
        best_ordinals,
        dense_winner_sources,
        source_extent.state,
        key_capacity,
        effective_policy,
    )
    _publish_dense_conflict_table(
        source_extent.state,
        invalid,
        stats["generated"],
        stats["overflow"],
        stats["generation"],
    )
    workspace._refresh_usage()
    return DeviceConflictResult(
        keys=None,
        values=None,
        priorities=None,
        ordinals=None,
        extent=None,
        statistics=(),
        policy=policy,
        strategy="dense_atomic",
        sort_method=None,
        key_capacity=key_capacity,
        output_shape="dense_winner_table",
        dense_winner_sources=dense_winner_sources,
    )


def _resolve_dense_impl(
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
    key_capacity,
):
    capacity = source_extent.capacity
    stage_sources = workspace._buffer("conflict_sources", i32, capacity)
    best_priorities = workspace._buffer(
        "dense_conflict_priorities", i32, key_capacity
    )
    best_ordinals = workspace._buffer(
        "dense_conflict_ordinals", i32, key_capacity
    )
    best_sources = workspace._buffer(
        "dense_conflict_sources", i32, key_capacity
    )
    flags = workspace._buffer("dense_conflict_flags", i32, key_capacity)
    winner_sources = workspace._buffer("conflict_winner_sources", i32, capacity)
    invalid = workspace._buffer("dense_conflict_invalid", i32, 1)
    _stage_conflict_source_indices(stage_sources, source_extent.state)
    priority_values = priorities if priorities is not None else stage_sources
    ordinal_values = ordinals if ordinals is not None else stage_sources
    effective_policy = 0
    if policy in ("claim", "min_priority") and priorities is not None:
        effective_policy = 1
    elif policy == "max_priority":
        effective_policy = 2
    _reset_dense_conflict_workspace(
        best_priorities,
        best_ordinals,
        best_sources,
        flags,
        invalid,
        effective_policy,
    )
    _select_dense_conflict_priorities(
        keys,
        priority_values,
        best_priorities,
        invalid,
        source_extent.state,
        key_capacity,
        effective_policy,
    )
    _select_dense_conflict_ordinals(
        keys,
        priority_values,
        ordinal_values,
        best_priorities,
        best_ordinals,
        source_extent.state,
        key_capacity,
        effective_policy,
    )
    _select_dense_conflict_sources(
        keys,
        priority_values,
        ordinal_values,
        best_priorities,
        best_ordinals,
        best_sources,
        source_extent.state,
        key_capacity,
        effective_policy,
    )
    _mark_dense_conflict_winners(best_sources, flags, key_capacity)
    workspace._scanner(key_capacity).run(flags)
    _emit_dense_conflict_winners(
        flags,
        best_sources,
        invalid,
        output_keys,
        winner_sources,
        output_extent.state,
        key_capacity,
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
        statistics=_statistics_tuple(stats),
        policy=policy,
        strategy="dense_atomic",
        sort_method=None,
        key_capacity=key_capacity,
    )


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
    strategy,
    sort_method,
    key_capacity,
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

    strategy, key_capacity, _ = _choose_conflict_strategy(
        strategy, key_capacity, capacity
    )
    if strategy == "dense_atomic":
        if dispatch_state is not None:
            raise ValueError(
                "dense_atomic conflict resolution publishes its own extent; "
                "dispatch_state is currently supported only by radix_grouped"
            )
        if sort_method != "auto":
            raise ValueError(
                "sort_method applies only to radix_grouped conflict resolution"
            )
        return _resolve_dense_impl(
            source_values,
            source_extent,
            keys,
            output_values,
            output_extent,
            stats,
            workspace,
            priorities=priorities,
            ordinals=ordinals,
            output_keys=output_keys,
            output_priorities=output_priorities,
            output_ordinals=output_ordinals,
            policy=policy,
            key_capacity=key_capacity,
        )

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
        method=_native_key_sort_method(sort_method),
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
        statistics=_statistics_tuple(stats),
        policy=policy,
        strategy="radix_grouped",
        sort_method=_native_key_sort_method(sort_method),
        key_capacity=key_capacity,
    )


class DeviceWorklist:
    """Stable front/back storage for a device-driven fixed-capacity worklist."""

    def __init__(
        self,
        capacity,
        dtype=i32,
        *,
        workspace=None,
        telemetry=True,
        transition_mode="staged",
    ):
        capacity = _require_capacity(capacity)
        if dtype not in _WORKLIST_DTYPES:
            raise TypeError("DeviceWorklist dtype is not supported")
        if impl.get_runtime().prog is None:
            raise TaichiRuntimeError("DeviceWorklist requires an initialized runtime")
        if not isinstance(telemetry, bool):
            raise TypeError("DeviceWorklist telemetry must be a bool")
        transition_mode = _require_transition_mode(transition_mode)
        if transition_mode == "direct" and telemetry:
            raise ValueError(
                "direct DeviceWorklist transitions require telemetry=False"
            )
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
        self._telemetry = telemetry
        self._transition_mode = transition_mode
        state_names = (
            _STATE_NAMES
            if telemetry
            else (
                ("generated", "overflow", "generation")
                if transition_mode == "staged"
                else ("overflow", "generation")
            )
        )
        self._stats = {name: ti_ndarray(i32, shape=()) for name in state_names}
        self._stats["generation"].fill(0)
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
                for name in state_names
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
    def telemetry_enabled(self):
        return self._telemetry

    @property
    def transition_mode(self):
        return self._transition_mode

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
        if self._transition_mode == "direct":
            _begin_direct_worklist_transition(
                self._extents[0].state,
                self._stats["overflow"],
                self._stats["generation"],
                0,
            )
        else:
            _reset_target(self._extents[0], self._stats)
        self._extents[1].reset()
        self._front = 0
        self._next_requires_finalize = False
        return self

    def prepare_next(self):
        """Reset the back extent and counters before atomic production."""

        self._validate_current()
        if self._transition_mode == "direct":
            _begin_direct_worklist_transition(
                self.next_extent.state,
                self._stats["overflow"],
                self._stats["generation"],
                1,
            )
            self._next_requires_finalize = False
        else:
            _reset_target(self.next_extent, self._stats)
            self._next_requires_finalize = True
        return self

    def commit_next(self, *, dispatch_state=None):
        """Swap front/back ownership; this operation does not synchronize."""

        self._validate_current()
        if dispatch_state is not None and impl.current_cfg().arch == _ti_core.Arch.cuda:
            raise TaichiRuntimeError(
                "CUDA worklist publication does not produce a consumer-owned "
                "dispatch packet; use the next DeviceExtent directly"
            )
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
        if self._transition_mode == "direct":
            return values, extent.state, self._stats["overflow"], self._capacity
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
        if self._transition_mode == "direct":
            raise TaichiRuntimeError(
                "direct DeviceWorklist mode is limited to atomic append transitions"
            )
        if dispatch_state is not None and impl.current_cfg().arch == _ti_core.Arch.cuda:
            raise TaichiRuntimeError(
                "CUDA worklist selection does not produce a consumer-owned "
                "dispatch packet; use the next DeviceExtent directly"
            )
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
        strategy="auto",
        sort_method=None,
        key_capacity=None,
        dispatch_state=None,
        output_shape="compact_winner_list",
    ):
        """Select one deterministic winner for every active integer key.

        ``strategy`` chooses the arbitration algorithm. ``sort_method`` chooses
        only the sort provider used by ``radix_grouped``; the legacy
        ``method`` spelling remains a backward-compatible alias. Supplying a
        bounded ``key_capacity`` lets ``auto`` admit ``dense_atomic`` when the
        key domain is at most one quarter of the candidate capacity; small CPU
        worklists conservatively retain radix unless dense is explicit.
        """

        self._validate_current()
        if self._transition_mode == "direct":
            raise TaichiRuntimeError(
                "direct DeviceWorklist mode is limited to atomic append transitions"
            )
        if dispatch_state is not None and impl.current_cfg().arch == _ti_core.Arch.cuda:
            raise TaichiRuntimeError(
                "CUDA conflict resolution does not produce a consumer-owned "
                "dispatch packet; use the next DeviceExtent directly"
            )
        sort_method = _normalize_conflict_sort_method(method, sort_method)
        output_shape = _normalize_conflict_output_shape(output_shape)
        source_extent = self.extent
        if output_shape == "dense_winner_table":
            selected_strategy, key_capacity, _ = _choose_conflict_strategy(
                strategy, key_capacity, self._capacity
            )
            if selected_strategy != "dense_atomic":
                raise ValueError(
                    "dense_winner_table output requires the dense_atomic strategy"
                )
            if sort_method != "auto":
                raise ValueError(
                    "sort_method applies only to compact radix_grouped output"
                )
            if dispatch_state is not None:
                raise ValueError(
                    "dense_winner_table output does not publish a compact extent"
                )
            table = self._workspace._buffer(
                "dense_conflict_winner_table", i32, key_capacity
            )
            return _resolve_dense_table_impl(
                source_extent,
                keys,
                table,
                self._stats,
                self._workspace,
                priorities=priorities,
                ordinals=ordinals,
                policy=policy,
                key_capacity=key_capacity,
            )
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
            strategy=strategy,
            sort_method=sort_method,
            key_capacity=key_capacity,
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
            strategy=result.strategy,
            sort_method=result.sort_method,
            key_capacity=result.key_capacity,
        )

    def statistics(self):
        """Synchronize and materialize the latest transition counters."""

        self._validate_current()
        values = {
            name: int(value.to_numpy().item())
            for name, value in self._stats.items()
        }
        return DeviceWorklistStatistics(
            schema_version=2,
            generated=values.get("generated"),
            accepted=values.get("accepted"),
            rejected=values.get("rejected"),
            conflicts=values.get("conflicts"),
            winners=values.get("winners"),
            overflow=bool(values["overflow"]),
            generation=values["generation"],
            telemetry_available=self._telemetry,
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
            schema_version=2,
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
            generation=statistics.generation,
            telemetry_available=statistics.telemetry_available,
        )

    def graph_args(self, name):
        return device_worklist_graph_args(
            name,
            self._capacity,
            self._dtype,
            telemetry=self._telemetry,
            transition_mode=self._transition_mode,
        )

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
        counter_bytes = len(self._stats) * 4
        mandatory_counter_bytes = (
            8 if self._transition_mode == "direct" else 12
        )
        owned = front_back + 16 + counter_bytes
        return {
            "schema_version": 1,
            "capacity": self._capacity,
            "front_back_value_bytes": front_back,
            "extent_bytes": 16,
            "counter_bytes": counter_bytes,
            "mandatory_counter_bytes": mandatory_counter_bytes,
            "optional_telemetry_bytes": (
                counter_bytes - mandatory_counter_bytes
            ),
            "telemetry_enabled": self._telemetry,
            "transition_mode": self._transition_mode,
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
                for name in _STATE_NAMES
                if (value := getattr(args, name)) is not None
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
        if self.args.transition_mode == "direct":
            raise TaichiRuntimeError(
                "direct DeviceWorklist transitions publish during append and "
                "do not use finalize_next()"
            )
        if dispatch_state is not None:
            if not isinstance(dispatch_state, DeviceDispatchState):
                raise TypeError("worklist dispatch_state must be DeviceDispatchState")
            dispatch_state._validate_current()
            if dispatch_state.capacity != self.capacity:
                raise ValueError("worklist dispatch_state capacity mismatch")
            if impl.current_cfg().arch == _ti_core.Arch.cuda:
                raise TaichiRuntimeError(
                    "CUDA worklist publication does not produce a consumer-owned "
                    "dispatch packet; use the next DeviceExtent directly"
                )
            self._leases.append(dispatch_state)
        return self._set_operation("finalize", (), {"dispatch_state": dispatch_state})

    def select(self, flags, *, method="auto", dispatch_state=None):
        self._ensure_mutable()
        if self.args.transition_mode == "direct":
            raise TaichiRuntimeError(
                "direct DeviceWorklist mode is limited to atomic append transitions"
            )
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
            if impl.current_cfg().arch == _ti_core.Arch.cuda:
                raise TaichiRuntimeError(
                    "CUDA worklist selection does not produce a consumer-owned "
                    "dispatch packet; use the next DeviceExtent directly"
                )
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
        strategy="auto",
        sort_method=None,
        key_capacity=None,
        dispatch_state=None,
    ):
        self._ensure_mutable()
        if self.args.transition_mode == "direct":
            raise TaichiRuntimeError(
                "direct DeviceWorklist mode is limited to atomic append transitions"
            )
        sort_method = _normalize_conflict_sort_method(method, sort_method)
        selected_strategy, key_capacity, strategy_reason = (
            _choose_conflict_strategy(strategy, key_capacity, self.capacity)
        )
        selected_sort_method = (
            _native_key_sort_method(sort_method)
            if selected_strategy == "radix_grouped"
            else sort_method
        )
        if selected_strategy == "dense_atomic" and selected_sort_method != "auto":
            raise ValueError(
                "sort_method applies only to radix_grouped conflict resolution"
            )
        if selected_strategy == "dense_atomic" and dispatch_state is not None:
            raise ValueError(
                "dense_atomic conflict resolution currently owns extent "
                "publication and cannot use dispatch_state"
            )
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
        workspace_buffers = [
            ("conflict_sources", i32),
            ("conflict_winner_sources", i32),
        ]
        if selected_strategy == "radix_grouped":
            workspace_buffers.extend(
                (
                    ("conflict_keys", keys.dtype()),
                    ("conflict_flags", i32),
                    ("conflict_boundary_sources", i32),
                    ("compact_flags", i32),
                )
            )
        else:
            for role in (
                "dense_conflict_priorities",
                "dense_conflict_ordinals",
                "dense_conflict_sources",
                "dense_conflict_flags",
            ):
                self.workspace._buffer(role, i32, key_capacity)
            self.workspace._buffer("dense_conflict_invalid", i32, 1)
            self.workspace._scanner(key_capacity)
        for role, dtype in workspace_buffers:
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
            if impl.current_cfg().arch == _ti_core.Arch.cuda:
                raise TaichiRuntimeError(
                    "CUDA conflict resolution does not produce a consumer-owned "
                    "dispatch packet; use the next DeviceExtent directly"
                )
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
                "strategy": selected_strategy,
                "strategy_reason": strategy_reason,
                "sort_method": selected_sort_method,
                "key_capacity": key_capacity,
                "dispatch_state": dispatch_state,
            },
        )

    def resolve_conflict_winner_table(
        self,
        keys,
        winner_sources,
        *,
        priorities=None,
        ordinals=None,
        policy="first",
        key_capacity,
    ):
        """Record dense winner-source arbitration without list materialization.

        Empty keys contain ``0x7fffffff``.  The operation intentionally
        requires telemetry-free worklist arguments because counting winners
        would reintroduce the scan that this output contract removes.
        """

        self._ensure_mutable()
        if self.args.transition_mode == "direct":
            raise TaichiRuntimeError(
                "direct DeviceWorklist mode is limited to atomic append transitions"
            )
        if self.args.telemetry:
            raise ValueError(
                "dense winner-only conflict resolution requires telemetry=False"
            )
        key_capacity = _require_capacity(key_capacity, "key_capacity")
        if key_capacity > self.capacity:
            raise ValueError(
                "DeviceWorklist dense key_capacity cannot exceed worklist capacity"
            )
        keys = self._register(keys, "keys", ndim=1)
        winner_sources = self._register(
            winner_sources, "dense winner sources", dtype=i32, ndim=1
        )
        if priorities is not None:
            priorities = self._register(
                priorities, "priorities", dtype=i32, ndim=1
            )
        if ordinals is not None:
            ordinals = self._register(ordinals, "ordinals", dtype=i32, ndim=1)
        if policy not in ("first", "claim", "min_priority", "max_priority"):
            raise ValueError(
                "conflict policy must be first, claim, min_priority, or max_priority"
            )
        if policy in ("min_priority", "max_priority") and priorities is None:
            raise ValueError(f"conflict policy {policy!r} requires priorities")
        self.workspace._buffer("conflict_sources", i32, self.capacity)
        self.workspace._buffer(
            "dense_conflict_priorities", i32, key_capacity
        )
        self.workspace._buffer(
            "dense_conflict_ordinals", i32, key_capacity
        )
        self.workspace._buffer("dense_conflict_invalid", i32, 1)
        return self._set_operation(
            "resolve_dense_table",
            (
                keys.name,
                winner_sources.name,
                None if priorities is None else priorities.name,
                None if ordinals is None else ordinals.name,
            ),
            {
                "policy": policy,
                "strategy": "dense_atomic",
                "strategy_reason": "dense_winner_table_contract",
                "sort_method": None,
                "key_capacity": key_capacity,
                "dispatch_state": None,
                "output_shape": "dense_winner_table",
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
        self._recording_id = next(_WORKLIST_RECORDING_IDS)
        self._recordable_action_cache = None
        self._recordable_action_initialized = False
        self._runner = self._compile_runner()
        legacy = os.environ.get("TI_DEBUG_NATIVE_SEQUENCE_LEGACY_REPLAY", "")
        self._legacy_replay = legacy.strip().lower() in ("1", "true", "on", "yes")
        self._run_impl = self._run_legacy if self._legacy_replay else self._runner

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

    def _build_recordable_transition_action(self):
        kind, _, options = self._operation
        if kind not in ("reset", "finalize"):
            return None
        from taichi_forge.graph._graph import Arg, ArgKind, gen_cpp_kernel

        if kind == "reset":
            if self._args.transition_mode == "direct":
                prefix = f"__ti_worklist_transition_{self._recording_id}"
                advance_arg = Arg(
                    ArgKind.SCALAR, f"{prefix}_advance_generation", i32
                )
                symbolic_args = (
                    self._args.next_extent,
                    self._args.overflow,
                    self._args.generation,
                    advance_arg,
                )
                reset_kernel = _begin_direct_worklist_transition
                fixed_bindings = {advance_arg.name: 1}
            elif self._args.telemetry:
                symbolic_args = (
                    self._args.next_extent,
                    *self._args.stat_args,
                )
                reset_kernel = _reset_worklist_target
                fixed_bindings = {}
            else:
                symbolic_args = (
                    self._args.next_extent,
                    self._args.generated,
                    self._args.overflow,
                )
                reset_kernel = _reset_worklist_target_minimal
                fixed_bindings = {}
            kernel_cpp = gen_cpp_kernel(reset_kernel, symbolic_args)
        else:
            prefix = f"__ti_worklist_transition_{self._recording_id}"
            capacity_arg = Arg(ArgKind.SCALAR, f"{prefix}_capacity", i32)
            dispatch_state = options.get("dispatch_state")
            if dispatch_state is not None and impl.current_cfg().arch in (
                _ti_core.Arch.cuda,
                _ti_core.Arch.vulkan,
            ):
                dispatch_state._validate_current()
                packet_arg = Arg(
                    ArgKind.NDARRAY,
                    f"{prefix}_dispatch_packet",
                    u32,
                    ndim=1,
                )
                block_arg = Arg(
                    ArgKind.SCALAR,
                    f"{prefix}_dispatch_block",
                    i32,
                )
                if self._args.telemetry:
                    symbolic_args = (
                        self._args.next_extent,
                        *self._args.stat_args,
                        self._args.generation,
                        packet_arg,
                        capacity_arg,
                        block_arg,
                    )
                    finalize_kernel = _finalize_atomic_worklist_with_dispatch
                else:
                    symbolic_args = (
                        self._args.next_extent,
                        self._args.generated,
                        self._args.overflow,
                        self._args.generation,
                        packet_arg,
                        capacity_arg,
                        block_arg,
                    )
                    finalize_kernel = (
                        _finalize_atomic_worklist_minimal_with_dispatch
                    )
                kernel_cpp = gen_cpp_kernel(finalize_kernel, symbolic_args)
                fixed_bindings = {
                    packet_arg.name: dispatch_state.packet,
                    capacity_arg.name: self._args.capacity_value,
                    block_arg.name: dispatch_state.block_dim,
                }
            else:
                if self._args.telemetry:
                    symbolic_args = (
                        self._args.next_extent,
                        *self._args.stat_args,
                        self._args.generation,
                        capacity_arg,
                    )
                    finalize_kernel = _finalize_atomic_worklist
                else:
                    symbolic_args = (
                        self._args.next_extent,
                        self._args.generated,
                        self._args.overflow,
                        self._args.generation,
                        capacity_arg,
                    )
                    finalize_kernel = _finalize_atomic_worklist_minimal
                kernel_cpp = gen_cpp_kernel(finalize_kernel, symbolic_args)
                fixed_bindings = {
                    capacity_arg.name: self._args.capacity_value,
                }
        return DispatchGraphAction(
            ((kernel_cpp, symbolic_args),),
            backends=(_current_backend_name(),),
            conditional_body_safe=True,
            fixed_bindings=fixed_bindings,
            allow_unused_public_bindings=True,
            update_policy="immutable",
            synchronization_domain="runtime_ordered",
        )

    @property
    def recordable_action(self):
        if not self._recordable_action_initialized:
            self._recordable_action_cache = (
                self._build_recordable_transition_action()
            )
            self._recordable_action_initialized = True
        return self._recordable_action_cache

    def recordable_bounded_publication(self, target):
        kind, _, options = self._operation
        if (
            kind != "finalize"
            or options.get("dispatch_state") is not None
            or target.backend != "vulkan"
            or target.packet_layout != "dispatch_indirect_u32x4"
            or target.extent_name != self._args.next_extent.name
            or int(target.capacity) != self._args.capacity_value
        ):
            return None
        from taichi_forge.graph._graph import Arg, ArgKind, gen_cpp_kernel

        prefix = f"__ti_worklist_transition_{self._recording_id}_publication"
        capacity_arg = Arg(ArgKind.SCALAR, f"{prefix}_capacity", i32)
        block_arg = Arg(ArgKind.SCALAR, f"{prefix}_block", i32)
        if self._args.telemetry:
            symbolic_args = (
                self._args.next_extent,
                *self._args.stat_args,
                self._args.generation,
                target.packet_binding,
                capacity_arg,
                block_arg,
            )
            finalize_kernel = _finalize_atomic_worklist_with_dispatch
        else:
            symbolic_args = (
                self._args.next_extent,
                self._args.generated,
                self._args.overflow,
                self._args.generation,
                target.packet_binding,
                capacity_arg,
                block_arg,
            )
            finalize_kernel = _finalize_atomic_worklist_minimal_with_dispatch
        kernel_cpp = gen_cpp_kernel(finalize_kernel, symbolic_args)
        return DispatchGraphAction(
            ((kernel_cpp, symbolic_args),),
            backends=(target.backend,),
            conditional_body_safe=True,
            fixed_bindings={
                target.packet_binding.name: target.packet_storage,
                capacity_arg.name: int(target.capacity),
                block_arg.name: int(target.block_dim),
            },
            allow_unused_public_bindings=True,
            update_policy="immutable",
            synchronization_domain="runtime_ordered",
        )

    def _stats(self, runtime_args):
        return {
            name: runtime_args[value.name]
            for name in _STATE_NAMES
            if (value := getattr(self._args, name)) is not None
        }

    def _compile_runner(self):
        kind, values, options = self._operation
        if kind == "reset":

            def reset(
                _current_values,
                _current_extent,
                _next_values,
                next_extent,
                stats,
                _runtime_args,
            ):
                if self._args.transition_mode == "direct":
                    _begin_direct_worklist_transition(
                        next_extent.state,
                        stats["overflow"],
                        stats["generation"],
                        1,
                    )
                else:
                    _reset_target(next_extent, stats)

            return reset
        if kind == "finalize":
            dispatch_state = options["dispatch_state"]

            def finalize(
                _current_values,
                _current_extent,
                _next_values,
                next_extent,
                stats,
                _runtime_args,
            ):
                _finalize_atomic_target(
                    next_extent,
                    stats,
                    self._args.capacity_value,
                    dispatch_state,
                )

            return finalize
        if kind == "select":
            flags = values[0]
            method = options["method"]
            dispatch_state = options["dispatch_state"]

            def select(
                current_values,
                current_extent,
                next_values,
                next_extent,
                stats,
                runtime_args,
            ):
                _select_impl(
                    current_values,
                    current_extent,
                    runtime_args[flags],
                    next_values,
                    next_extent,
                    stats,
                    self._workspace,
                    method=method,
                    dispatch_state=dispatch_state,
                )

            return select
        if kind == "resolve":
            (
                keys,
                output_keys,
                output_priorities,
                output_ordinals,
                priorities,
                ordinals,
            ) = values
            policy = options["policy"]
            strategy = options["strategy"]
            sort_method = options["sort_method"]
            key_capacity = options["key_capacity"]
            dispatch_state = options["dispatch_state"]

            def resolve(
                current_values,
                current_extent,
                next_values,
                next_extent,
                stats,
                runtime_args,
            ):
                _resolve_impl(
                    current_values,
                    current_extent,
                    runtime_args[keys],
                    next_values,
                    next_extent,
                    stats,
                    self._workspace,
                    priorities=(
                        None if priorities is None else runtime_args[priorities]
                    ),
                    ordinals=None if ordinals is None else runtime_args[ordinals],
                    output_keys=runtime_args[output_keys],
                    output_priorities=runtime_args[output_priorities],
                    output_ordinals=runtime_args[output_ordinals],
                    policy=policy,
                    strategy=strategy,
                    sort_method=sort_method,
                    key_capacity=key_capacity,
                    dispatch_state=dispatch_state,
                )

            return resolve
        if kind == "resolve_dense_table":
            keys, winner_sources, priorities, ordinals = values
            policy = options["policy"]
            key_capacity = options["key_capacity"]

            def resolve_dense_table(
                _current_values,
                current_extent,
                _next_values,
                _next_extent,
                stats,
                runtime_args,
            ):
                _resolve_dense_table_impl(
                    current_extent,
                    runtime_args[keys],
                    runtime_args[winner_sources],
                    stats,
                    self._workspace,
                    priorities=(
                        None if priorities is None else runtime_args[priorities]
                    ),
                    ordinals=None if ordinals is None else runtime_args[ordinals],
                    policy=policy,
                    key_capacity=key_capacity,
                )

            return resolve_dense_table
        raise TaichiRuntimeError(f"Unsupported worklist operation {kind!r}")

    def _run_legacy(
        self,
        current_values,
        current_extent,
        next_values,
        next_extent,
        stats,
        runtime_args,
    ):
        kind, values, options = self._operation
        if kind == "reset":
            if self._args.transition_mode == "direct":
                _begin_direct_worklist_transition(
                    next_extent.state,
                    stats["overflow"],
                    stats["generation"],
                    1,
                )
            else:
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
                priorities=(
                    None if priorities is None else runtime_args[priorities]
                ),
                ordinals=None if ordinals is None else runtime_args[ordinals],
                output_keys=runtime_args[output_keys],
                output_priorities=runtime_args[output_priorities],
                output_ordinals=runtime_args[output_ordinals],
                policy=options["policy"],
                strategy=options["strategy"],
                sort_method=options["sort_method"],
                key_capacity=options["key_capacity"],
                dispatch_state=options["dispatch_state"],
            )
        elif kind == "resolve_dense_table":
            _resolve_dense_table_impl(
                current_extent,
                runtime_args[values[0]],
                runtime_args[values[1]],
                stats,
                self._workspace,
                priorities=(
                    None if values[2] is None else runtime_args[values[2]]
                ),
                ordinals=(
                    None if values[3] is None else runtime_args[values[3]]
                ),
                policy=options["policy"],
                key_capacity=options["key_capacity"],
            )
        else:
            raise TaichiRuntimeError(f"Unsupported worklist operation {kind!r}")

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
        self._run_impl(
            current_values,
            current_extent,
            next_values,
            next_extent,
            stats,
            runtime_args,
        )

    @property
    def debug_info(self):
        result = {
            "kind": "device_worklist_sequence",
            "operation": self._operation[0],
            "capacity": self._args.capacity_value,
            "workspace_bytes_peak": self._workspace.workspace_bytes_peak,
            "provider_selection": "materialization_time",
            "replay_operation_branch": self._legacy_replay,
            "legacy_replay_forced": self._legacy_replay,
            "backend_native_recording": self._operation[0] in ("reset", "finalize"),
            "telemetry_enabled": self._args.telemetry,
            "transition_mode": self._args.transition_mode,
            "counter_count": len(self._args.state_args),
        }
        if self._operation[0] in ("resolve", "resolve_dense_table"):
            options = self._operation[2]
            result.update(
                conflict_strategy=options["strategy"],
                conflict_strategy_reason=options["strategy_reason"],
                sort_provider=(
                    options["sort_method"]
                    if options["strategy"] == "radix_grouped"
                    else None
                ),
                key_capacity=options["key_capacity"],
                conflict_output_shape=options.get(
                    "output_shape", "compact_winner_list"
                ),
            )
        return result


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
    "device_worklist_append_direct",
    "device_worklist_graph_args",
]
