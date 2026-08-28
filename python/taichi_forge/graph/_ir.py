"""Internal structured Graph metadata.

This module is deliberately separate from the serialized AOT CGraph v1
schema. It gives JIT/runtime optimization passes a versioned, conservative
description without changing the stable public execution-report contract.
"""

import hashlib
import json
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
    subresource: object = None

    def __post_init__(self):
        if self.subresource is not None:
            try:
                hash(self.subresource)
            except TypeError as exc:
                raise TypeError("Graph effect subresource must be hashable") from exc

    def to_dict(self):
        result = {
            "resource": self.resource,
            "access": self.access.value,
            "runtime_bound": self.runtime_bound,
        }
        if self.subresource is not None:
            result["subresource"] = self.subresource
        return result


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
    storage_kind: str = "raw_i32"

    def __post_init__(self):
        if not self.name:
            raise ValueError("Graph temporary name must not be empty")
        if self.bytes < 0:
            raise ValueError("Graph temporary bytes must be non-negative")
        if self.alignment <= 0:
            raise ValueError("Graph temporary alignment must be positive")
        if self.storage_kind not in ("raw_i32", "f32"):
            raise ValueError("Unsupported Graph temporary storage kind")
        if self.storage_kind == "f32" and (self.bytes % 4 != 0 or self.alignment % 4 != 0):
            raise ValueError("f32 Graph temporaries require four-byte size and alignment")

    def to_dict(self):
        result = {
            "name": self.name,
            "bytes": self.bytes,
            "alignment": self.alignment,
        }
        if self.storage_kind != "raw_i32":
            result["storage_kind"] = self.storage_kind
        return result


@dataclass(frozen=True)
class InternalNdarrayRequirement:
    """Graph-instance-owned ndarray storage required by a recorded action.

    Unlike a temporary, this storage remains address-stable for the lifetime
    of one compiled Graph instance. ``exclusive_submission`` declares that a
    later asynchronous invocation must not reuse it until the preceding
    completion fence has retired.
    """

    dtype: object
    shape: tuple
    element_bytes: int
    exclusive_submission: bool = False

    def __post_init__(self):
        shape = tuple(int(value) for value in self.shape)
        if any(value <= 0 for value in shape):
            raise ValueError("Graph internal ndarray shape must be positive")
        if self.element_bytes <= 0:
            raise ValueError("Graph internal ndarray element size must be positive")
        object.__setattr__(self, "shape", shape)

    @property
    def storage_bytes(self):
        elements = 1
        for value in self.shape:
            elements *= value
        return elements * self.element_bytes


@dataclass(frozen=True)
class BoundedDomain:
    """Backend-neutral iteration domain driven by a device extent.

    Backend packets, updater controls, and physical node identities are
    deliberately excluded. ``publication_epoch`` remains unknown until a
    producer/effect analysis can prove which extent write reaches the
    consumer.
    """

    extent: str
    capacity: int
    block_dim: Optional[int] = None
    block_mode: str = "auto"
    physical_grid_requirement: str = "auto"
    publication_epoch: Optional[int] = None
    count_source: str = "device_extent"
    ordered: bool = False
    segment_index: Optional[int] = None
    segment_count: int = 0

    def __post_init__(self):
        if not isinstance(self.extent, str) or not self.extent:
            raise ValueError("Bounded domain extent must be a non-empty resource")
        if isinstance(self.capacity, bool) or not isinstance(self.capacity, int):
            raise TypeError("Bounded domain capacity must be an integer")
        if self.capacity <= 0:
            raise ValueError("Bounded domain capacity must be positive")
        if self.block_dim is not None and (
            isinstance(self.block_dim, bool)
            or not isinstance(self.block_dim, int)
            or not 1 <= self.block_dim <= 1024
        ):
            raise ValueError("Bounded domain block_dim must be in [1, 1024]")
        if self.block_mode not in ("auto", "hint", "require"):
            raise ValueError("Bounded domain block_mode is invalid")
        if self.physical_grid_requirement not in (
            "auto",
            "fixed_capacity",
            "logical_exact",
            "adaptive_grid",
            "require_exact",
        ):
            raise ValueError("Bounded domain physical grid requirement is invalid")
        if self.publication_epoch is not None and self.publication_epoch < 0:
            raise ValueError("Bounded domain publication epoch must be non-negative")
        if self.count_source not in ("device_extent", "host_scalar"):
            raise ValueError("Bounded domain count source is invalid")
        if self.ordered:
            if self.count_source != "device_extent":
                raise ValueError("Ordered bounded domains require a device extent")
            if self.segment_index is None or self.segment_count <= 0:
                raise ValueError(
                    "Ordered bounded domains require a segment index and count"
                )
            if not 0 <= self.segment_index < self.segment_count:
                raise ValueError("Bounded domain segment index is out of range")
        elif self.segment_index is not None or self.segment_count != 0:
            raise ValueError(
                "Non-ordered bounded domains cannot carry segment metadata"
            )

    def to_dict(self):
        return {
            "extent": self.extent,
            "capacity": self.capacity,
            "block_dim": self.block_dim,
            "block_mode": self.block_mode,
            "physical_grid_requirement": self.physical_grid_requirement,
            "publication_epoch": self.publication_epoch,
            "count_source": self.count_source,
            "ordered": self.ordered,
            "segment_index": self.segment_index,
            "segment_count": self.segment_count,
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
    bounded_domain: Optional[BoundedDomain] = None
    dispatch_label: str = ""
    logical_dispatch_id: str = ""
    fusion_blocker: str = ""

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
    bindings: Tuple[RuntimeBinding, ...] = ()
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


def _validate_control_name(value, role):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{role} must be a non-empty resource name")


def _validate_unique_control_names(values, role):
    values = tuple(values)
    for value in values:
        _validate_control_name(value, role)
    if len(values) != len(set(values)):
        raise ValueError(f"{role} must not contain duplicate resources")
    return values


@dataclass(frozen=True)
class WhileRegion:
    predicate: str
    max_iterations: int
    condition: SequentialRegion
    body: SequentialRegion
    control_inputs: Tuple[str, ...] = ()
    carried_state: Tuple[str, ...] = ()
    counter: Optional[str] = None
    status: Optional[str] = None
    chunk_size: int = 1
    compound_chunk_size: int = 1
    vulkan_first_chunk_strategy: str = "auto"
    masked_execution: bool = False
    lowering_mode: str = "auto"
    name: str = "while"
    synchronization: bool = True
    opaque: bool = False

    def __post_init__(self):
        _validate_control_name(self.predicate, "While predicate")
        if self.max_iterations < 0:
            raise ValueError("While max_iterations must be non-negative")
        if not isinstance(self.condition, SequentialRegion):
            raise ValueError("While condition must be a SequentialRegion")
        if not isinstance(self.body, SequentialRegion):
            raise ValueError("While body must be a SequentialRegion")
        object.__setattr__(
            self,
            "control_inputs",
            _validate_unique_control_names(self.control_inputs, "While control_inputs"),
        )
        object.__setattr__(
            self,
            "carried_state",
            _validate_unique_control_names(self.carried_state, "While carried_state"),
        )
        if self.predicate in self.control_inputs:
            raise ValueError(
                "While predicate is an output and cannot be a control input"
            )
        if self.counter is not None:
            _validate_control_name(self.counter, "While counter")
        if self.status is not None:
            _validate_control_name(self.status, "While status")
            reserved = {
                self.predicate,
                *self.control_inputs,
                *self.carried_state,
            }
            if self.counter is not None:
                reserved.add(self.counter)
            if self.status in reserved:
                raise ValueError(
                    "While status must be a distinct control resource"
                )
        if self.chunk_size <= 0:
            raise ValueError("While chunk_size must be positive")
        if self.compound_chunk_size <= 0:
            raise ValueError("While compound_chunk_size must be positive")
        if self.vulkan_first_chunk_strategy not in (
            "auto",
            "compact",
            "coarse_conditional",
        ):
            raise ValueError("Unsupported Vulkan first-chunk strategy")
        if self.lowering_mode not in (
            "auto",
            "portable",
            "native_required",
        ):
            raise ValueError("Unsupported while lowering mode")

    @property
    def kind(self):
        return "while_region"

    @property
    def children(self):
        return (self.condition, self.body)

    @property
    def effects(self):
        return (
            ResourceEffect(self.predicate, GraphAccess.READ_WRITE),
            *(ResourceEffect(name, GraphAccess.READ) for name in self.control_inputs),
            *(
                ResourceEffect(name, GraphAccess.READ_WRITE)
                for name in self.carried_state
            ),
            *(
                (ResourceEffect(self.counter, GraphAccess.READ_WRITE),)
                if self.counter is not None
                else ()
            ),
            *(
                (ResourceEffect(self.status, GraphAccess.READ_WRITE),)
                if self.status is not None
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
        return f"while:{self.max_iterations}"


@dataclass(frozen=True)
class IfRegion:
    predicate: str
    condition: SequentialRegion
    then_region: SequentialRegion
    else_region: Optional[SequentialRegion] = None
    control_inputs: Tuple[str, ...] = ()
    name: str = "if"
    synchronization: bool = True
    opaque: bool = False

    def __post_init__(self):
        _validate_control_name(self.predicate, "If predicate")
        for region, role in (
            (self.condition, "If condition"),
            (self.then_region, "If then_region"),
        ):
            if not isinstance(region, SequentialRegion):
                raise ValueError(f"{role} must be a SequentialRegion")
        if self.else_region is not None and not isinstance(
            self.else_region, SequentialRegion
        ):
            raise ValueError("If else_region must be a SequentialRegion")
        object.__setattr__(
            self,
            "control_inputs",
            _validate_unique_control_names(self.control_inputs, "If control_inputs"),
        )
        if self.predicate in self.control_inputs:
            raise ValueError("If predicate is an output and cannot be a control input")

    @property
    def kind(self):
        return "if_region"

    @property
    def children(self):
        regions = (self.condition, self.then_region)
        return regions if self.else_region is None else (*regions, self.else_region)

    @property
    def effects(self):
        return (
            ResourceEffect(self.predicate, GraphAccess.READ_WRITE),
            *(ResourceEffect(name, GraphAccess.READ) for name in self.control_inputs),
        )

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
class SwitchRegion:
    selector: str
    condition: SequentialRegion
    branches: Tuple[SequentialRegion, ...]
    default_region: Optional[SequentialRegion] = None
    control_inputs: Tuple[str, ...] = ()
    name: str = "switch"
    synchronization: bool = True
    opaque: bool = False

    def __post_init__(self):
        _validate_control_name(self.selector, "Switch selector")
        if not isinstance(self.condition, SequentialRegion):
            raise ValueError("Switch condition must be a SequentialRegion")
        branches = tuple(self.branches)
        if not branches:
            raise ValueError("Switch requires at least one branch")
        if not all(isinstance(branch, SequentialRegion) for branch in branches):
            raise ValueError("Switch branches must contain SequentialRegion values")
        object.__setattr__(self, "branches", branches)
        if self.default_region is not None and not isinstance(
            self.default_region, SequentialRegion
        ):
            raise ValueError("Switch default_region must be a SequentialRegion")
        object.__setattr__(
            self,
            "control_inputs",
            _validate_unique_control_names(
                self.control_inputs, "Switch control_inputs"
            ),
        )
        if self.selector in self.control_inputs:
            raise ValueError(
                "Switch selector is an output and cannot be a control input"
            )

    @property
    def kind(self):
        return "switch_region"

    @property
    def children(self):
        regions = (self.condition, *self.branches)
        return (
            regions if self.default_region is None else (*regions, self.default_region)
        )

    @property
    def effects(self):
        return (
            ResourceEffect(self.selector, GraphAccess.READ_WRITE),
            *(ResourceEffect(name, GraphAccess.READ) for name in self.control_inputs),
        )

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
class GraphIRAnalysis:
    node_count: int
    dispatch_nodes: int
    native_call_nodes: int
    sequential_regions: int
    while_regions: int
    if_regions: int
    switch_regions: int
    max_structured_depth: int
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
    candidate_recipes: tuple = ()
    applied_groups: int = 0
    lowering_available: bool = False

    def to_dict(self):
        return {
            "candidate_groups": len(self.candidate_groups),
            "candidate_dispatches": sum(len(group) for group in self.candidate_groups),
            "eligible_dispatches": self.eligible_dispatches,
            "blocked_dispatches": self.blocked_dispatches,
            "blockers": dict(self.blocker_counts),
            "recipes": tuple(
                recipe.to_dict() for recipe in self.candidate_recipes
            ),
            "applied_groups": self.applied_groups,
            "lowering_available": self.lowering_available,
            "decision": (
                "applied"
                if self.applied_groups
                else (
                    "qualified_not_applied"
                    if self.candidate_groups and self.lowering_available
                    else (
                        "cross_kernel_ir_composer_unavailable"
                        if self.candidate_groups
                        else "no_safe_candidates"
                    )
                )
            ),
        }


@dataclass(frozen=True)
class _KernelFusionRecipe:
    recipe_id: str
    region_path: str
    source_dispatch_ids: Tuple[str, ...]
    source_names: Tuple[str, ...]
    iteration_domain: str
    lowering_kind: str = "preoffload_range_map"
    expected_physical_dispatches: int = 1

    def __post_init__(self):
        if not self.recipe_id.startswith("fusion:map2:"):
            raise ValueError("pair fusion recipe ID is invalid")
        if len(self.source_dispatch_ids) != 2:
            raise ValueError("pair fusion recipe requires two source dispatches")
        if len(set(self.source_dispatch_ids)) != 2:
            raise ValueError("pair fusion source dispatches must be unique")
        if len(self.source_names) != 2:
            raise ValueError("pair fusion source names must match dispatches")
        if not self.region_path or not self.iteration_domain:
            raise ValueError("pair fusion recipe requires region and domain")
        if self.expected_physical_dispatches != 1:
            raise ValueError("pair fusion must materialize one dispatch")

    def to_dict(self):
        return {
            "schema_version": 1,
            "recipe_id": self.recipe_id,
            "region_path": self.region_path,
            "source_dispatch_ids": self.source_dispatch_ids,
            "source_names": self.source_names,
            "iteration_domain": self.iteration_domain,
            "lowering_kind": self.lowering_kind,
            "expected_physical_dispatches": self.expected_physical_dispatches,
        }


@dataclass(frozen=True)
class TemporaryAllocation:
    name: str
    offset: int
    bytes: int
    alignment: int
    slot: int
    storage_kind: str = "raw_i32"


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


@dataclass(frozen=True)
class ParallelEffectDependency:
    left_branch: int
    right_branch: int
    left_resource: str
    right_resource: str
    left_access: GraphAccess
    right_access: GraphAccess
    alias: str
    dependencies: Tuple[str, ...]

    def to_dict(self):
        return {
            "left_branch": self.left_branch,
            "right_branch": self.right_branch,
            "left_resource": self.left_resource,
            "right_resource": self.right_resource,
            "left_access": self.left_access.value,
            "right_access": self.right_access.value,
            "alias": self.alias,
            "dependencies": self.dependencies,
        }


@dataclass(frozen=True)
class ParallelBranchSummary:
    index: int
    node_names: Tuple[str, ...]
    effects: Tuple[ResourceEffect, ...]
    temporary_peak_bytes: int
    temporary_opaque_bytes: int

    def to_dict(self):
        return {
            "index": self.index,
            "node_names": self.node_names,
            "effects": tuple(effect.to_dict() for effect in self.effects),
            "temporary_peak_bytes": self.temporary_peak_bytes,
            "temporary_opaque_bytes": self.temporary_opaque_bytes,
        }


@dataclass(frozen=True)
class ParallelCandidatePlan:
    branches: Tuple[ParallelBranchSummary, ...]
    conflicts: Tuple[ParallelEffectDependency, ...]
    unresolved_aliases: Tuple[ParallelEffectDependency, ...]
    blockers: Tuple[str, ...]
    sequential_fallback_peak_bytes: int
    parallel_branch_temporary_bytes: int
    parallel_peak_bytes: int
    memory_overhead_vs_sequential: int

    @property
    def decision(self):
        if self.blockers or self.conflicts:
            return "rejected"
        if self.unresolved_aliases:
            return "runtime_binding_required"
        return "safe"

    @property
    def safe(self):
        if self.decision == "runtime_binding_required":
            return None
        return self.decision == "safe"

    def to_dict(self):
        return {
            "schema_version": 1,
            "analysis_only": True,
            "execution_changed": False,
            "decision": self.decision,
            "safe": self.safe,
            "branches": tuple(branch.to_dict() for branch in self.branches),
            "conflicts": tuple(conflict.to_dict() for conflict in self.conflicts),
            "unresolved_aliases": tuple(
                dependency.to_dict() for dependency in self.unresolved_aliases
            ),
            "blockers": self.blockers,
            "sequential_fallback_peak_bytes": (
                self.sequential_fallback_peak_bytes
            ),
            "parallel_branch_temporary_bytes": (
                self.parallel_branch_temporary_bytes
            ),
            "parallel_peak_bytes": self.parallel_peak_bytes,
            "memory_overhead_vs_sequential": (
                self.memory_overhead_vs_sequential
            ),
            "partial_output_bytes": 0,
        }


def _parallel_access_properties(access):
    reads = access in (GraphAccess.READ, GraphAccess.READ_WRITE, GraphAccess.ATOMIC)
    writes = access in (
        GraphAccess.WRITE,
        GraphAccess.READ_WRITE,
        GraphAccess.ATOMIC,
        GraphAccess.OPAQUE,
    )
    return reads, writes


def _merge_parallel_access(lhs, rhs):
    if GraphAccess.OPAQUE in (lhs, rhs):
        return GraphAccess.OPAQUE
    if GraphAccess.ATOMIC in (lhs, rhs):
        return GraphAccess.ATOMIC
    lhs_reads, lhs_writes = _parallel_access_properties(lhs)
    rhs_reads, rhs_writes = _parallel_access_properties(rhs)
    reads = lhs_reads or rhs_reads
    writes = lhs_writes or rhs_writes
    if reads and writes:
        return GraphAccess.READ_WRITE
    return GraphAccess.WRITE if writes else GraphAccess.READ


def _parallel_dependency_kinds(lhs, rhs):
    if GraphAccess.OPAQUE in (lhs, rhs):
        return ("opaque",)
    if GraphAccess.ATOMIC in (lhs, rhs):
        return ("atomic_overlap",)
    lhs_reads, lhs_writes = _parallel_access_properties(lhs)
    rhs_reads, rhs_writes = _parallel_access_properties(rhs)
    dependencies = []
    if lhs_writes and rhs_reads:
        dependencies.append("raw")
    if lhs_reads and rhs_writes:
        dependencies.append("war")
    if lhs_writes and rhs_writes:
        dependencies.append("waw")
    return tuple(dependencies)


def _parallel_branch_metadata(region, branch_index):
    effects = {}
    node_names = []
    blockers = []
    temporary_names = set()

    def visit(node):
        if not isinstance(node, SequentialRegion):
            node_names.append(node.name)
        if node.opaque:
            blockers.append(f"branch_{branch_index}:opaque_node:{node.name}")
        if node.synchronization:
            blockers.append(
                f"branch_{branch_index}:synchronization_boundary:{node.name}"
            )
        if isinstance(node, ObservationNode):
            blockers.append(f"branch_{branch_index}:observation:{node.name}")
        if node.kind in ("while_region", "if_region", "switch_region"):
            blockers.append(
                f"branch_{branch_index}:structured_control:{node.name}"
            )
        for side_effect in getattr(node, "side_effects", ()):
            blockers.append(
                f"branch_{branch_index}:side_effect:{node.name}:{side_effect}"
            )
        for effect in node.effects:
            if effect.access == GraphAccess.OPAQUE:
                blockers.append(
                    f"branch_{branch_index}:opaque_effect:{effect.resource}"
                )
            key = (effect.resource, effect.runtime_bound, effect.subresource)
            previous = effects.get(key)
            effects[key] = (
                effect.access
                if previous is None
                else _merge_parallel_access(previous, effect.access)
            )
        for requirement in node.temporaries:
            temporary_names.add(requirement.name)
        for child in node.children:
            visit(child)

    visit(region)
    temporary_plan = plan_temporary_memory(region)
    if temporary_plan.conflicting_requirements:
        blockers.append(
            f"branch_{branch_index}:conflicting_temporary_requirement"
        )
    summary = ParallelBranchSummary(
        index=branch_index,
        node_names=tuple(node_names),
        effects=tuple(
            ResourceEffect(resource, access, runtime_bound, subresource)
            for (resource, runtime_bound, subresource), access in sorted(
                effects.items(), key=lambda item: repr(item[0])
            )
        ),
        temporary_peak_bytes=temporary_plan.planned_peak_bytes,
        temporary_opaque_bytes=temporary_plan.opaque_bytes,
    )
    return summary, tuple(blockers), frozenset(temporary_names)


def analyze_parallel_candidate(branches):
    """Conservatively analyze sibling branches without changing execution."""

    branches = tuple(branches)
    if not 2 <= len(branches) <= 4:
        raise ValueError("parallel candidate requires between 2 and 4 branches")
    if not all(isinstance(branch, SequentialRegion) for branch in branches):
        raise TypeError("parallel candidate branches must be SequentialRegion values")

    summaries = []
    blockers = []
    temporary_owners = {}
    for branch_index, branch in enumerate(branches):
        summary, branch_blockers, temporary_names = _parallel_branch_metadata(
            branch, branch_index
        )
        summaries.append(summary)
        blockers.extend(branch_blockers)
        for name in temporary_names:
            previous = temporary_owners.get(name)
            if previous is not None:
                blockers.append(
                    "temporary_shared_across_branches:"
                    f"{name}:branch_{previous}:branch_{branch_index}"
                )
            else:
                temporary_owners[name] = branch_index

    conflicts = []
    unresolved = []
    for left_index, left in enumerate(summaries):
        for right in summaries[left_index + 1 :]:
            for left_effect in left.effects:
                for right_effect in right.effects:
                    dependencies = _parallel_dependency_kinds(
                        left_effect.access, right_effect.access
                    )
                    if not dependencies:
                        continue
                    same_resource = left_effect.resource == right_effect.resource
                    if same_resource:
                        conflicts.append(
                            ParallelEffectDependency(
                                left.index,
                                right.index,
                                left_effect.resource,
                                right_effect.resource,
                                left_effect.access,
                                right_effect.access,
                                "proven_overlap",
                                dependencies,
                            )
                        )
                    elif left_effect.runtime_bound or right_effect.runtime_bound:
                        unresolved.append(
                            ParallelEffectDependency(
                                left.index,
                                right.index,
                                left_effect.resource,
                                right_effect.resource,
                                left_effect.access,
                                right_effect.access,
                                "runtime_required",
                                dependencies,
                            )
                        )

    sequential_region = SequentialRegion(
        tuple(child for branch in branches for child in branch.children),
        name="parallel_candidate_sequential_fallback",
    )
    sequential_plan = plan_temporary_memory(sequential_region)
    parallel_temporary_bytes = sum(
        branch.temporary_peak_bytes + branch.temporary_opaque_bytes
        for branch in summaries
    )
    parallel_peak_bytes = parallel_temporary_bytes
    return ParallelCandidatePlan(
        branches=tuple(summaries),
        conflicts=tuple(conflicts),
        unresolved_aliases=tuple(unresolved),
        blockers=tuple(sorted(set(blockers))),
        sequential_fallback_peak_bytes=(
            sequential_plan.planned_peak_bytes + sequential_plan.opaque_bytes
        ),
        parallel_branch_temporary_bytes=parallel_temporary_bytes,
        parallel_peak_bytes=parallel_peak_bytes,
        memory_overhead_vs_sequential=max(
            0,
            parallel_peak_bytes
            - sequential_plan.planned_peak_bytes
            - sequential_plan.opaque_bytes,
        ),
    )


def _fusion_blocker(node):
    if not isinstance(node, DispatchNode):
        return "not_dispatch"
    if node.opaque:
        return node.fusion_blocker or "opaque_dispatch"
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


def _pair_fusion_recipe(region_path, indexed_nodes, iteration_domain):
    source_ids = tuple(
        f"{region_path}/{node.logical_dispatch_id or f'node:{index}'}"
        for index, node in indexed_nodes
    )
    source_names = tuple(node.name for _, node in indexed_nodes)
    effect_signatures = tuple(
        tuple(
            (
                effect.resource,
                effect.access.value,
                effect.runtime_bound,
                repr(effect.subresource),
            )
            for effect in node.effects
        )
        for _, node in indexed_nodes
    )
    binding_signatures = tuple(
        tuple(
            (binding.name, binding.kind, binding.required)
            for binding in node.bindings
        )
        for _, node in indexed_nodes
    )
    payload = json.dumps(
        {
            "region_path": region_path,
            "source_dispatch_ids": source_ids,
            "iteration_domain": iteration_domain,
            "effects": effect_signatures,
            "bindings": binding_signatures,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return _KernelFusionRecipe(
        recipe_id=f"fusion:map2:{digest}",
        region_path=region_path,
        source_dispatch_ids=source_ids,
        source_names=source_names,
        iteration_domain=iteration_domain,
    )


def analyze_elementwise_fusion(root, *, applied_groups=0, lowering_available=False):
    """Find safe pointwise fusion groups without pretending to lower them."""

    groups = []
    recipes = []
    blockers = {}
    eligible_dispatches = 0
    blocked_dispatches = 0

    def add_blocker(reason):
        blockers[reason] = blockers.get(reason, 0) + 1

    def scan(region, path):
        nonlocal eligible_dispatches
        nonlocal blocked_dispatches
        pending = []
        domain = None

        def flush():
            nonlocal pending
            nonlocal domain
            if len(pending) >= 2:
                groups.append(tuple(node.name for _, node in pending))
                for offset in range(0, len(pending) - 1, 2):
                    recipes.append(
                        _pair_fusion_recipe(
                            path,
                            tuple(pending[offset : offset + 2]),
                            domain,
                        )
                    )
            pending = []
            domain = None

        for child_index, node in enumerate(region.children):
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
                    pending.append((child_index, node))
            else:
                flush()
            if isinstance(node, SequentialRegion):
                scan(node, f"{path}/{child_index}:{node.name}")
            else:
                for nested_index, child in enumerate(node.children):
                    if isinstance(child, SequentialRegion):
                        scan(
                            child,
                            f"{path}/{child_index}:{node.name}/"
                            f"{nested_index}:{child.name}",
                        )
        flush()

    if isinstance(root, SequentialRegion):
        scan(root, root.name or "root")
    return ElementwiseFusionPlan(
        candidate_groups=tuple(groups),
        eligible_dispatches=eligible_dispatches,
        blocked_dispatches=blocked_dispatches,
        blocker_counts=tuple(sorted(blockers.items())),
        candidate_recipes=tuple(recipes),
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
                    "storage_kind": requirement.storage_kind,
                    "conflict": False,
                }
            else:
                entry["last"] = current_position
                if (
                    entry["bytes"] != requirement.bytes
                    or entry["alignment"] != requirement.alignment
                    or entry["storage_kind"] != requirement.storage_kind
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
    opaque_entries = [entry for entry in declarations.values() if entry["conflict"]]
    opaque_bytes = sum(entry["bytes"] for entry in opaque_entries)
    intervals = sorted(
        (
            entry["first"],
            entry["last"],
            name,
            entry["bytes"],
            entry["alignment"],
            entry["storage_kind"],
        )
        for name, entry in declarations.items()
        if not entry["conflict"]
    )
    slots = []
    allocation_slots = {}
    for first, last, name, byte_count, alignment, storage_kind in intervals:
        available = [
            (index, slot)
            for index, slot in enumerate(slots)
            if slot["last"] < first and slot["storage_kind"] == storage_kind
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
                slot["alignment"] * alignment // gcd(slot["alignment"], alignment)
            )
        else:
            slot_index = len(slots)
            slots.append(
                {
                    "last": last,
                    "bytes": byte_count,
                    "alignment": alignment,
                    "storage_kind": storage_kind,
                }
            )
        allocation_slots[name] = slot_index

    peak_by_kind = {}
    slot_payload_bytes = 0
    slot_offsets = {}
    for slot_index, slot in enumerate(slots):
        storage_kind = slot["storage_kind"]
        alignment = slot["alignment"]
        kind_peak = peak_by_kind.get(storage_kind, 0)
        kind_peak = (kind_peak + alignment - 1) // alignment * alignment
        slot_offsets[slot_index] = kind_peak
        peak_by_kind[storage_kind] = kind_peak + slot["bytes"]
        slot_payload_bytes += slot["bytes"]
    peak_bytes = sum(peak_by_kind.values())
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
                storage_kind=entry["storage_kind"],
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
        "while_regions": 0,
        "if_regions": 0,
        "switch_regions": 0,
        "max_structured_depth": 0,
        "observation_nodes": 0,
        "effect_reads": 0,
        "effect_writes": 0,
        "effect_atomics": 0,
        "opaque_nodes": 0,
        "runtime_bindings": 0,
        "temporary_bytes": 0,
    }

    def visit(node, structured_depth=0):
        counters["node_count"] += 1
        if node.kind in ("while_region", "if_region", "switch_region"):
            structured_depth += 1
            counters["max_structured_depth"] = max(
                counters["max_structured_depth"], structured_depth
            )
        counter_name = {
            "dispatch": "dispatch_nodes",
            "native_call": "native_call_nodes",
            "sequential_region": "sequential_regions",
            "while_region": "while_regions",
            "if_region": "if_regions",
            "switch_region": "switch_regions",
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
            visit(child, structured_depth)

    visit(root)
    return GraphIRAnalysis(**counters)


def graph_ir_to_dict(node, _structured_depth=0):
    if node.kind in ("while_region", "if_region", "switch_region"):
        _structured_depth += 1
    result = {
        "kind": node.kind,
        "name": node.name,
        "effects": tuple(effect.to_dict() for effect in node.effects),
        "bindings": tuple(binding.to_dict() for binding in node.bindings),
        "temporaries": tuple(requirement.to_dict() for requirement in node.temporaries),
        "iteration_domain": node.iteration_domain,
        "synchronization": node.synchronization,
        "opaque": node.opaque,
        "children": tuple(
            graph_ir_to_dict(child, _structured_depth) for child in node.children
        ),
    }
    if node.kind in ("while_region", "if_region", "switch_region"):
        result["structured_depth"] = _structured_depth
    if isinstance(node, WhileRegion):
        result.update(
            {
                "predicate": node.predicate,
                "counter": node.counter,
                "status": node.status,
                "max_iterations": node.max_iterations,
                "control_inputs": node.control_inputs,
                "carried_state": node.carried_state,
                "chunk_size": node.chunk_size,
                "compound_chunk_size": node.compound_chunk_size,
                "vulkan_first_chunk_strategy": node.vulkan_first_chunk_strategy,
                "masked_execution": node.masked_execution,
                "lowering_mode": node.lowering_mode,
            }
        )
    if isinstance(node, IfRegion):
        result.update(
            {
                "predicate": node.predicate,
                "control_inputs": node.control_inputs,
                "has_else": node.else_region is not None,
            }
        )
    if isinstance(node, SwitchRegion):
        result.update(
            {
                "selector": node.selector,
                "control_inputs": node.control_inputs,
                "branch_count": len(node.branches),
                "has_default": node.default_region is not None,
            }
        )
    if isinstance(node, DispatchNode):
        result["elementwise"] = node.elementwise
        result["side_effects"] = node.side_effects
        result["dispatch_label"] = node.dispatch_label
        result["logical_dispatch_id"] = node.logical_dispatch_id
        result["fusion_blocker"] = node.fusion_blocker
        result["bounded_domain"] = (
            None
            if node.bounded_domain is None
            else node.bounded_domain.to_dict()
        )
    if isinstance(node, ObservationNode):
        result["batch"] = node.batch
    return result
