"""Internal structured Graph metadata.

This module is deliberately separate from the serialized AOT CGraph v1
schema. It gives JIT/runtime optimization passes a versioned, conservative
description without changing the stable public execution-report contract.
"""

from dataclasses import dataclass
from enum import Enum
from math import gcd
from typing import Optional, Tuple


class GraphAccess(str, Enum):
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"
    ATOMIC = "atomic"
    OPAQUE = "opaque"


@dataclass(frozen=True)
class ResourceEffect:
    resource: str
    access: GraphAccess
    runtime_bound: bool = True

    def to_dict(self):
        return {
            "resource": self.resource,
            "access": self.access.value,
            "runtime_bound": self.runtime_bound,
        }


@dataclass(frozen=True)
class RuntimeBinding:
    name: str
    kind: str
    required: bool = True

    def to_dict(self):
        return {
            "name": self.name,
            "kind": self.kind,
            "required": self.required,
        }


@dataclass(frozen=True)
class TemporaryRequirement:
    name: str
    bytes: int
    alignment: int = 1

    def __post_init__(self):
        if not self.name:
            raise ValueError("Graph temporary name must not be empty")
        if self.bytes < 0:
            raise ValueError("Graph temporary bytes must be non-negative")
        if self.alignment <= 0:
            raise ValueError("Graph temporary alignment must be positive")

    def to_dict(self):
        return {
            "name": self.name,
            "bytes": self.bytes,
            "alignment": self.alignment,
        }


@dataclass(frozen=True)
class DispatchNode:
    name: str
    effects: Tuple[ResourceEffect, ...] = ()
    bindings: Tuple[RuntimeBinding, ...] = ()
    temporaries: Tuple[TemporaryRequirement, ...] = ()
    iteration_domain: Optional[str] = None
    synchronization: bool = False
    opaque: bool = True
    elementwise: bool = False
    side_effects: Tuple[str, ...] = ()

    @property
    def kind(self):
        return "dispatch"

    @property
    def children(self):
        return ()


@dataclass(frozen=True)
class NativeCallNode:
    name: str
    effects: Tuple[ResourceEffect, ...] = ()
    bindings: Tuple[RuntimeBinding, ...] = ()
    temporaries: Tuple[TemporaryRequirement, ...] = ()
    iteration_domain: Optional[str] = None
    synchronization: bool = False
    opaque: bool = True

    @property
    def kind(self):
        return "native_call"

    @property
    def children(self):
        return ()


@dataclass(frozen=True)
class ObservationNode:
    name: str
    effects: Tuple[ResourceEffect, ...]
    batch: str = "default"
    synchronization: bool = True
    opaque: bool = False

    @property
    def kind(self):
        return "observation"

    @property
    def children(self):
        return ()

    @property
    def bindings(self):
        return ()

    @property
    def temporaries(self):
        return ()

    @property
    def iteration_domain(self):
        return None


@dataclass(frozen=True)
class SequentialRegion:
    children: Tuple[object, ...]
    name: str = "sequential"
    synchronization: bool = False
    opaque: bool = False

    @property
    def kind(self):
        return "sequential_region"

    @property
    def effects(self):
        return ()

    @property
    def bindings(self):
        return ()

    @property
    def temporaries(self):
        return ()

    @property
    def iteration_domain(self):
        return None


@dataclass(frozen=True)
class BoundedLoopRegion:
    predicate: str
    max_iterations: int
    body: SequentialRegion
    counter: Optional[str] = None
    predicate_convention: str = "continue_while_nonzero"
    initial_observation: bool = True
    terminal_observation: bool = True
    masked_execution: bool = False
    cuda_native_mode: str = "auto"
    name: str = "bounded_loop"
    synchronization: bool = True
    opaque: bool = False

    def __post_init__(self):
        if not self.predicate:
            raise ValueError("BoundedLoopRegion requires a predicate")
        if self.max_iterations < 0:
            raise ValueError(
                "BoundedLoopRegion max_iterations must be non-negative"
            )
        if self.predicate_convention not in (
            "continue_while_nonzero",
            "stop_when_nonzero",
        ):
            raise ValueError("Unsupported bounded-loop predicate convention")
        if self.cuda_native_mode not in (
            "auto",
            "portable",
            "native_required",
        ):
            raise ValueError("Unsupported bounded-loop CUDA native mode")

    @property
    def kind(self):
        return "bounded_loop_region"

    @property
    def children(self):
        return (self.body,)

    @property
    def effects(self):
        return (
            ResourceEffect(self.predicate, GraphAccess.READ),
            *(
                (ResourceEffect(self.counter, GraphAccess.WRITE),)
                if self.counter is not None
                else ()
            ),
        )

    @property
    def bindings(self):
        return ()

    @property
    def temporaries(self):
        return ()

    @property
    def iteration_domain(self):
        return f"bounded:{self.max_iterations}"


@dataclass(frozen=True)
class GraphIRAnalysis:
    node_count: int
    dispatch_nodes: int
    native_call_nodes: int
    sequential_regions: int
    bounded_loop_regions: int
    observation_nodes: int
    effect_reads: int
    effect_writes: int
    effect_atomics: int
    opaque_nodes: int
    runtime_bindings: int
    temporary_bytes: int

    def to_dict(self):
        return self.__dict__.copy()


@dataclass(frozen=True)
class ElementwiseFusionPlan:
    candidate_groups: Tuple[Tuple[str, ...], ...]
    eligible_dispatches: int
    blocked_dispatches: int
    blocker_counts: Tuple[Tuple[str, int], ...]
    applied_groups: int = 0
    lowering_available: bool = False

    def to_dict(self):
        return {
            "candidate_groups": len(self.candidate_groups),
            "candidate_dispatches": sum(
                len(group) for group in self.candidate_groups
            ),
            "eligible_dispatches": self.eligible_dispatches,
            "blocked_dispatches": self.blocked_dispatches,
            "blockers": dict(self.blocker_counts),
            "applied_groups": self.applied_groups,
            "lowering_available": self.lowering_available,
            "decision": (
                "applied"
                if self.applied_groups
                else "qualified_not_applied"
                if self.candidate_groups and self.lowering_available
                else "cross_kernel_ir_composer_unavailable"
                if self.candidate_groups
                else "no_safe_candidates"
            ),
        }


@dataclass(frozen=True)
class TemporaryAllocation:
    name: str
    offset: int
    bytes: int
    alignment: int
    slot: int


@dataclass(frozen=True)
class TemporaryMemoryPlan:
    declared_bytes: int
    logical_bytes: int
    planned_peak_bytes: int
    reused_bytes: int
    alignment_padding_bytes: int
    slot_count: int
    conflicting_requirements: int
    opaque_bytes: int
    materialized: bool = False
    allocations: Tuple[TemporaryAllocation, ...] = ()

    def to_dict(self):
        return {
            "declared_bytes": self.declared_bytes,
            "logical_bytes": self.logical_bytes,
            "planned_peak_bytes": self.planned_peak_bytes,
            "reused_bytes": self.reused_bytes,
            "alignment_padding_bytes": self.alignment_padding_bytes,
            "slot_count": self.slot_count,
            "conflicting_requirements": self.conflicting_requirements,
            "opaque_bytes": self.opaque_bytes,
            "materialized": self.materialized,
        }


def _fusion_blocker(node):
    if not isinstance(node, DispatchNode):
        return "not_dispatch"
    if node.opaque:
        return "opaque_dispatch"
    if node.synchronization:
        return "synchronization_boundary"
    if not node.elementwise:
        return "not_elementwise"
    if node.iteration_domain is None:
        return "missing_iteration_domain"
    if node.side_effects:
        return "side_effect"
    for effect in node.effects:
        if effect.access == GraphAccess.ATOMIC:
            return "atomic_effect"
        if effect.access == GraphAccess.OPAQUE:
            return "opaque_effect"
    return None


def analyze_elementwise_fusion(
    root, *, applied_groups=0, lowering_available=False
):
    """Find safe pointwise fusion groups without pretending to lower them."""

    groups = []
    blockers = {}
    eligible_dispatches = 0
    blocked_dispatches = 0

    def add_blocker(reason):
        blockers[reason] = blockers.get(reason, 0) + 1

    def scan(region):
        nonlocal eligible_dispatches
        nonlocal blocked_dispatches
        pending = []
        domain = None

        def flush():
            nonlocal pending
            nonlocal domain
            if len(pending) >= 2:
                groups.append(tuple(node.name for node in pending))
            pending = []
            domain = None

        for node in region.children:
            if isinstance(node, DispatchNode):
                blocker = _fusion_blocker(node)
                if blocker is not None:
                    blocked_dispatches += 1
                    add_blocker(blocker)
                    flush()
                else:
                    eligible_dispatches += 1
                    if pending and node.iteration_domain != domain:
                        add_blocker("iteration_domain_mismatch")
                        flush()
                    if not pending:
                        domain = node.iteration_domain
                    pending.append(node)
            else:
                flush()
            if isinstance(node, SequentialRegion):
                scan(node)
            else:
                for child in node.children:
                    if isinstance(child, SequentialRegion):
                        scan(child)
        flush()

    if isinstance(root, SequentialRegion):
        scan(root)
    return ElementwiseFusionPlan(
        candidate_groups=tuple(groups),
        eligible_dispatches=eligible_dispatches,
        blocked_dispatches=blocked_dispatches,
        blocker_counts=tuple(sorted(blockers.items())),
        applied_groups=int(applied_groups),
        lowering_available=bool(lowering_available),
    )


def plan_temporary_memory(root):
    """Plan reusable abstract slots from declared temporary live intervals."""

    declarations = {}
    declared_bytes = 0
    position = 0

    def visit(node):
        nonlocal declared_bytes
        nonlocal position
        current_position = position
        position += 1
        for requirement in node.temporaries:
            declared_bytes += requirement.bytes
            entry = declarations.get(requirement.name)
            if entry is None:
                declarations[requirement.name] = {
                    "first": current_position,
                    "last": current_position,
                    "bytes": requirement.bytes,
                    "alignment": requirement.alignment,
                    "conflict": False,
                }
            else:
                entry["last"] = current_position
                if (
                    entry["bytes"] != requirement.bytes
                    or entry["alignment"] != requirement.alignment
                ):
                    entry["conflict"] = True
                    entry["bytes"] = max(entry["bytes"], requirement.bytes)
                    entry["alignment"] = (
                        entry["alignment"]
                        * requirement.alignment
                        // gcd(entry["alignment"], requirement.alignment)
                    )
        for child in node.children:
            visit(child)

    visit(root)
    logical_bytes = sum(entry["bytes"] for entry in declarations.values())
    opaque_entries = [
        entry for entry in declarations.values() if entry["conflict"]
    ]
    opaque_bytes = sum(entry["bytes"] for entry in opaque_entries)
    intervals = sorted(
        (
            entry["first"],
            entry["last"],
            name,
            entry["bytes"],
            entry["alignment"],
        )
        for name, entry in declarations.items()
        if not entry["conflict"]
    )
    slots = []
    allocation_slots = {}
    for first, last, name, byte_count, alignment in intervals:
        available = [
            (index, slot)
            for index, slot in enumerate(slots)
            if slot["last"] < first
        ]
        if available:
            slot_index, slot = min(
                available,
                key=lambda value: (
                    max(value[1]["bytes"], byte_count),
                    value[1]["alignment"],
                ),
            )
            slot["last"] = last
            slot["bytes"] = max(slot["bytes"], byte_count)
            slot["alignment"] = (
                slot["alignment"]
                * alignment
                // gcd(slot["alignment"], alignment)
            )
        else:
            slot_index = len(slots)
            slots.append(
                {
                    "last": last,
                    "bytes": byte_count,
                    "alignment": alignment,
                }
            )
        allocation_slots[name] = slot_index

    peak_bytes = 0
    slot_payload_bytes = 0
    slot_offsets = []
    for slot in slots:
        alignment = slot["alignment"]
        peak_bytes = (peak_bytes + alignment - 1) // alignment * alignment
        slot_offsets.append(peak_bytes)
        peak_bytes += slot["bytes"]
        slot_payload_bytes += slot["bytes"]
    planned_logical_bytes = logical_bytes - opaque_bytes
    return TemporaryMemoryPlan(
        declared_bytes=declared_bytes,
        logical_bytes=logical_bytes,
        planned_peak_bytes=peak_bytes,
        reused_bytes=max(0, planned_logical_bytes - slot_payload_bytes),
        alignment_padding_bytes=peak_bytes - slot_payload_bytes,
        slot_count=len(slots),
        conflicting_requirements=len(opaque_entries),
        opaque_bytes=opaque_bytes,
        allocations=tuple(
            TemporaryAllocation(
                name=name,
                offset=slot_offsets[allocation_slots[name]],
                bytes=entry["bytes"],
                alignment=entry["alignment"],
                slot=allocation_slots[name],
            )
            for name, entry in sorted(declarations.items())
            if not entry["conflict"]
        ),
    )


def analyze_graph_ir(root):
    counters = {
        "node_count": 0,
        "dispatch_nodes": 0,
        "native_call_nodes": 0,
        "sequential_regions": 0,
        "bounded_loop_regions": 0,
        "observation_nodes": 0,
        "effect_reads": 0,
        "effect_writes": 0,
        "effect_atomics": 0,
        "opaque_nodes": 0,
        "runtime_bindings": 0,
        "temporary_bytes": 0,
    }

    def visit(node):
        counters["node_count"] += 1
        counter_name = {
            "dispatch": "dispatch_nodes",
            "native_call": "native_call_nodes",
            "sequential_region": "sequential_regions",
            "bounded_loop_region": "bounded_loop_regions",
            "observation": "observation_nodes",
        }[node.kind]
        counters[counter_name] += 1
        counters["opaque_nodes"] += int(node.opaque)
        counters["runtime_bindings"] += len(node.bindings)
        counters["temporary_bytes"] += sum(
            requirement.bytes for requirement in node.temporaries
        )
        for effect in node.effects:
            if effect.access in (GraphAccess.READ, GraphAccess.READ_WRITE):
                counters["effect_reads"] += 1
            if effect.access in (GraphAccess.WRITE, GraphAccess.READ_WRITE):
                counters["effect_writes"] += 1
            if effect.access == GraphAccess.ATOMIC:
                counters["effect_reads"] += 1
                counters["effect_writes"] += 1
                counters["effect_atomics"] += 1
        for child in node.children:
            visit(child)

    visit(root)
    return GraphIRAnalysis(**counters)


def graph_ir_to_dict(node):
    result = {
        "kind": node.kind,
        "name": node.name,
        "effects": tuple(effect.to_dict() for effect in node.effects),
        "bindings": tuple(binding.to_dict() for binding in node.bindings),
        "temporaries": tuple(
            requirement.to_dict() for requirement in node.temporaries
        ),
        "iteration_domain": node.iteration_domain,
        "synchronization": node.synchronization,
        "opaque": node.opaque,
        "children": tuple(graph_ir_to_dict(child) for child in node.children),
    }
    if isinstance(node, BoundedLoopRegion):
        result.update(
            {
                "predicate": node.predicate,
                "counter": node.counter,
                "max_iterations": node.max_iterations,
                "predicate_convention": node.predicate_convention,
                "initial_observation": node.initial_observation,
                "terminal_observation": node.terminal_observation,
                "masked_execution": node.masked_execution,
                "cuda_native_mode": node.cuda_native_mode,
            }
        )
    if isinstance(node, DispatchNode):
        result["elementwise"] = node.elementwise
        result["side_effects"] = node.side_effects
    if isinstance(node, ObservationNode):
        result["batch"] = node.batch
    return result
