"""Internal structured Graph metadata.

This module is deliberately separate from the serialized AOT CGraph v1
schema. It gives JIT/runtime optimization passes a versioned, conservative
description without changing the stable public execution-report contract.
"""

from dataclasses import dataclass
from enum import Enum
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
            }
        )
    if isinstance(node, ObservationNode):
        result["batch"] = node.batch
    return result
