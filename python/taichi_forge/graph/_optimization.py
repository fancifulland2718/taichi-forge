"""Private executable-plan optimization identities for Forge Graphs."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from taichi_forge.graph._ir import graph_ir_to_dict


_GRAPH_FUSION_QUALIFICATION_SCHEMA = "taichi_forge.graph_fusion_qualification.v1"
_GRAPH_FUSION_QUALIFICATION_MAX_BYTES = 4 * 1024 * 1024
_INTERNAL_STRUCTURED_CONTROL_ENV = "TAICHI_FORGE_INTERNAL_STRUCTURED_CONTROL_RECIPE"
_CUDA_CONDITIONAL_CONTROL_RECIPE_ID = "control:cuda_conditional_graph:v1"
_CUDA_MASKED_CONTROL_RECIPE_ID = "control:cuda_masked_bounded_graph:v1"
_CUDA_CONTROL_RECIPE_IDS = (
    _CUDA_CONDITIONAL_CONTROL_RECIPE_ID,
    _CUDA_MASKED_CONTROL_RECIPE_ID,
)
_CUDA_NESTED_DEVICE_UPDATE_CONTROL_RECIPE_ID = "control:cuda_nested_device_update:v1"
_CUDA_NESTED_MASKED_CONTROL_RECIPE_ID = "control:cuda_nested_masked_bounded:v1"
_CUDA_NESTED_CONTROL_RECIPE_IDS = (
    _CUDA_NESTED_DEVICE_UPDATE_CONTROL_RECIPE_ID,
    _CUDA_NESTED_MASKED_CONTROL_RECIPE_ID,
)
_CUDA_STRUCTURED_CONTROL_RECIPE_DOMAINS = (
    _CUDA_CONTROL_RECIPE_IDS,
    _CUDA_NESTED_CONTROL_RECIPE_IDS,
)
_CUDA_STRUCTURED_CONTROL_RECIPE_IDS = tuple(
    recipe_id
    for domain in _CUDA_STRUCTURED_CONTROL_RECIPE_DOMAINS
    for recipe_id in domain
)

_GRAPH_MEMORY_RECIPE_SCHEMA_VERSION = 4
_GRAPH_MEMORY_RECIPE_PREFIXES = {
    "direct": "graph-memory:direct:",
    "shared_staged_1d": "graph-memory:shared-staged-1d:",
}

_GRAPH_BOUNDED_RECIPE_SCHEMA_VERSION = 1
_GRAPH_BOUNDED_RECIPE_PREFIXES = {
    "logical_exact": "graph-bounded:logical-exact:",
    "adaptive_per_node": "graph-bounded:adaptive-per-node:",
    "adaptive_grouped": "graph-bounded:adaptive-grouped:",
    "masked_capacity": "graph-bounded:masked-capacity:",
}

_GRAPH_REDUCTION_RECIPE_SCHEMA_VERSION = 2
_GRAPH_REDUCTION_RECIPE_PREFIXES = {
    "direct_atomic_tls": "graph-reduction:direct-atomic-tls:",
    "block_partial_finalize": "graph-reduction:block-partial-finalize:",
    "hierarchical_partial_finalize": ("graph-reduction:hierarchical-partial-finalize:"),
}

_GRAPH_NATIVE_ALGORITHM_RECIPE_SCHEMA_VERSION = 2
_GRAPH_NATIVE_ALGORITHM_RECIPE_PREFIXES = {
    ("segmented_scan", "segment_local_serial"): (
        "graph-native-algorithm:segmented-scan:serial:"
    ),
    ("segmented_scan", "global_scan_segment_correction"): (
        "graph-native-algorithm:segmented-scan:global-scan:"
    ),
    ("segmented_scan", "warp_chunked_carry"): (
        "graph-native-algorithm:segmented-scan:warp-chunked:"
    ),
    ("segmented_scan", "block_chunked_carry"): (
        "graph-native-algorithm:segmented-scan:block-chunked:"
    ),
    ("segmented_scan", "length_bucket_hybrid"): (
        "graph-native-algorithm:segmented-scan:length-bucket:"
    ),
}


def _canonical_hash(value):
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


@dataclass(frozen=True)
class _GraphMemoryRecipeManifest:
    """Immutable complete physical recipe retained and verified by Forge."""

    recipe_id: str
    payload_json: str

    def __post_init__(self):
        if not isinstance(self.recipe_id, str) or not self.recipe_id:
            raise ValueError("GraphMemory recipe ID must be a nonempty string")
        if not isinstance(self.payload_json, str) or not self.payload_json:
            raise ValueError("GraphMemory recipe payload must be canonical JSON")
        try:
            payload = json.loads(self.payload_json)
        except json.JSONDecodeError as error:
            raise ValueError("GraphMemory recipe payload is invalid") from error
        if (
            not isinstance(payload, dict)
            or _canonical_json(payload) != self.payload_json
        ):
            raise ValueError("GraphMemory recipe payload is not canonical")
        if payload.get("schema_version") != _GRAPH_MEMORY_RECIPE_SCHEMA_VERSION:
            raise ValueError("GraphMemory recipe schema is unsupported")
        if frozenset(payload) != {
            "schema_version",
            "strategy",
            "dispatch_label",
            "symbolic_abi",
            "semantic_kernel_identity",
            "offload_plan_identity",
            "offload_compilation_identity",
            "offload_plan",
            "materialized_tasks",
            "memory_disjoint_pairs",
            "memory_layout_requirements",
            "staged_sources",
        }:
            raise ValueError("GraphMemory recipe payload fields are invalid")
        strategy = payload.get("strategy")
        try:
            prefix = _GRAPH_MEMORY_RECIPE_PREFIXES[strategy]
        except (KeyError, TypeError) as error:
            raise ValueError("GraphMemory recipe strategy is unsupported") from error
        sources = payload.get("staged_sources")
        pairs = payload.get("memory_disjoint_pairs")
        layouts = payload.get("memory_layout_requirements")
        if (
            not isinstance(sources, list)
            or not isinstance(pairs, list)
            or not isinstance(layouts, list)
        ):
            raise ValueError("GraphMemory physical memory contract is invalid")
        if strategy == "direct":
            if sources or pairs or layouts:
                raise ValueError("direct GraphMemory recipe must not own staged layout")
        else:
            expected_source_fields = {
                "arg_index",
                "arg_name",
                "halo_low",
                "halo_high",
                "element_bytes",
                "scalar_bytes",
                "element_shape",
                "lane_count",
                "alignment",
                "byte_offset",
                "tile_elements",
                "tile_bytes",
                "access_offsets",
                "logical_output_count",
                "direct_input_records",
                "staged_input_records",
                "direct_input_bytes",
                "staged_input_bytes",
            }
            if not 1 <= len(sources) <= 2 or any(
                not isinstance(source, dict)
                or frozenset(source) != expected_source_fields
                for source in sources
            ):
                raise ValueError("staged GraphMemory recipe has invalid sources")
            indices = tuple(source["arg_index"] for source in sources)
            if indices != tuple(sorted(set(indices))):
                raise ValueError("staged GraphMemory source order is not canonical")
            tile_end = 0
            source_names = set()
            for source in sources:
                integer_fields = (
                    "arg_index",
                    "halo_low",
                    "halo_high",
                    "element_bytes",
                    "scalar_bytes",
                    "lane_count",
                    "alignment",
                    "byte_offset",
                    "tile_elements",
                    "tile_bytes",
                    "logical_output_count",
                    "direct_input_records",
                    "staged_input_records",
                    "direct_input_bytes",
                    "staged_input_bytes",
                )
                if (
                    not isinstance(source["arg_name"], str)
                    or not source["arg_name"]
                    or source["arg_name"] in source_names
                    or any(
                        isinstance(source[name], bool)
                        or not isinstance(source[name], int)
                        for name in integer_fields
                    )
                    or source["arg_index"] < 0
                    or source["halo_low"] >= source["halo_high"]
                    or source["scalar_bytes"] not in (2, 4, 8)
                    or not isinstance(source["element_shape"], list)
                    or len(source["element_shape"]) > 2
                    or any(
                        isinstance(extent, bool)
                        or not isinstance(extent, int)
                        or extent <= 0
                        for extent in source["element_shape"]
                    )
                    or not 1 <= source["lane_count"] <= 16
                    or source["lane_count"]
                    != math.prod(source["element_shape"] or (1,))
                    or source["element_bytes"]
                    != source["scalar_bytes"] * source["lane_count"]
                    or source["alignment"] not in (2, 4, 8)
                    or source["byte_offset"] < tile_end
                    or source["byte_offset"] % source["alignment"]
                    or source["tile_elements"] <= 0
                    or source["tile_bytes"]
                    != source["tile_elements"] * source["element_bytes"]
                    or source["tile_elements"]
                    <= source["halo_high"] - source["halo_low"]
                    or not isinstance(source["access_offsets"], list)
                    or len(source["access_offsets"]) < 2
                    or any(
                        isinstance(offset, bool) or not isinstance(offset, int)
                        for offset in source["access_offsets"]
                    )
                    or source["access_offsets"] != sorted(set(source["access_offsets"]))
                    or source["access_offsets"][0] != source["halo_low"]
                    or source["access_offsets"][-1] != source["halo_high"]
                    or source["logical_output_count"] <= 0
                    or source["direct_input_records"]
                    != source["logical_output_count"] * len(source["access_offsets"])
                    or source["staged_input_records"]
                    != source["logical_output_count"]
                    + math.ceil(
                        source["logical_output_count"]
                        / (
                            source["tile_elements"]
                            - source["halo_high"]
                            + source["halo_low"]
                        )
                    )
                    * (source["halo_high"] - source["halo_low"])
                    or source["direct_input_bytes"]
                    != source["direct_input_records"] * source["element_bytes"]
                    or source["staged_input_bytes"]
                    != source["staged_input_records"] * source["element_bytes"]
                ):
                    raise ValueError("staged GraphMemory tile layout is invalid")
                source_names.add(source["arg_name"])
                tile_end = source["byte_offset"] + source["tile_bytes"]
            if tile_end > 32 * 1024 or not pairs or not layouts:
                raise ValueError("staged GraphMemory memory contract is incomplete")
            if any(
                not isinstance(layout, list)
                or len(layout) not in (4, 6)
                or not isinstance(layout[0], str)
                or not layout[0]
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or value <= 0
                    for value in layout[1:4]
                )
                or (
                    len(layout) == 6
                    and (
                        not isinstance(layout[4], list)
                        or not 1 <= len(layout[4]) <= 2
                        or any(
                            isinstance(extent, bool)
                            or not isinstance(extent, int)
                            or extent <= 0
                            for extent in layout[4]
                        )
                        or layout[5] != "aos"
                    )
                )
                for layout in layouts
            ):
                raise ValueError("staged GraphMemory binding layout is invalid")
        expected = prefix + _canonical_hash(payload)[:24]
        if self.recipe_id != expected:
            raise ValueError("GraphMemory recipe ID does not match its payload")

    @classmethod
    def from_payload(cls, payload):
        if not isinstance(payload, dict):
            raise TypeError("GraphMemory recipe payload must be a dictionary")
        payload = dict(payload)
        payload["schema_version"] = _GRAPH_MEMORY_RECIPE_SCHEMA_VERSION
        strategy = payload.get("strategy")
        try:
            prefix = _GRAPH_MEMORY_RECIPE_PREFIXES[strategy]
        except (KeyError, TypeError) as error:
            raise ValueError("GraphMemory recipe strategy is unsupported") from error
        payload_json = _canonical_json(payload)
        return cls(
            recipe_id=prefix + _canonical_hash(payload)[:24],
            payload_json=payload_json,
        )

    @property
    def strategy(self):
        return json.loads(self.payload_json)["strategy"]

    def to_dict(self):
        return {"recipe_id": self.recipe_id, **json.loads(self.payload_json)}


@dataclass(frozen=True)
class _GraphBoundedExecutionRecipeManifest:
    """One complete CUDA bounded-execution strategy over an immutable scope."""

    recipe_id: str
    payload_json: str

    def __post_init__(self):
        if not isinstance(self.recipe_id, str) or not self.recipe_id:
            raise ValueError("GraphBounded recipe ID must be a nonempty string")
        if not isinstance(self.payload_json, str) or not self.payload_json:
            raise ValueError("GraphBounded recipe payload must be canonical JSON")
        try:
            payload = json.loads(self.payload_json)
        except json.JSONDecodeError as error:
            raise ValueError("GraphBounded recipe payload is invalid") from error
        if (
            not isinstance(payload, dict)
            or _canonical_json(payload) != self.payload_json
        ):
            raise ValueError("GraphBounded recipe payload is not canonical")
        if payload.get("schema_version") != _GRAPH_BOUNDED_RECIPE_SCHEMA_VERSION:
            raise ValueError("GraphBounded recipe schema is unsupported")
        if frozenset(payload) != {
            "schema_version",
            "strategy",
            "source_physical_grid_policy",
            "bounded_dispatch_count",
            "publication_groups",
        }:
            raise ValueError("GraphBounded recipe payload fields are invalid")
        if payload.get("source_physical_grid_policy") != "auto":
            raise ValueError("GraphBounded recipe source policy must be auto")
        strategy = payload.get("strategy")
        try:
            prefix = _GRAPH_BOUNDED_RECIPE_PREFIXES[strategy]
        except (KeyError, TypeError) as error:
            raise ValueError("GraphBounded recipe strategy is unsupported") from error
        groups = payload.get("publication_groups")
        if not isinstance(groups, list) or not groups:
            raise ValueError("GraphBounded recipe requires publication groups")
        if any(
            not isinstance(group, dict)
            or not isinstance(group.get("count_name"), str)
            or not group["count_name"]
            or isinstance(group.get("capacity"), bool)
            or not isinstance(group.get("capacity"), int)
            or group["capacity"] <= 0
            or (
                group.get("block_dim") is not None
                and (
                    isinstance(group["block_dim"], bool)
                    or not isinstance(group["block_dim"], int)
                    or not 1 <= group["block_dim"] <= 1024
                )
            )
            or isinstance(group.get("publication_epoch"), bool)
            or not isinstance(group.get("publication_epoch"), int)
            or group["publication_epoch"] < 0
            or isinstance(group.get("consumer_count"), bool)
            or not isinstance(group.get("consumer_count"), int)
            or group["consumer_count"] <= 0
            for group in groups
        ):
            raise ValueError("GraphBounded publication group is invalid")
        dispatch_count = payload.get("bounded_dispatch_count")
        if (
            isinstance(dispatch_count, bool)
            or not isinstance(dispatch_count, int)
            or dispatch_count <= 0
            or dispatch_count != sum(group["consumer_count"] for group in groups)
        ):
            raise ValueError("GraphBounded dispatch count does not match its groups")
        if strategy == "adaptive_grouped" and not any(
            group["consumer_count"] >= 2 for group in groups
        ):
            raise ValueError(
                "GraphBounded grouped recipe requires a shared publication"
            )
        expected = prefix + _canonical_hash(payload)[:24]
        if self.recipe_id != expected:
            raise ValueError("GraphBounded recipe ID does not match its payload")

    @classmethod
    def from_payload(cls, payload):
        if not isinstance(payload, dict):
            raise TypeError("GraphBounded recipe payload must be a dictionary")
        payload = dict(payload)
        payload["schema_version"] = _GRAPH_BOUNDED_RECIPE_SCHEMA_VERSION
        strategy = payload.get("strategy")
        try:
            prefix = _GRAPH_BOUNDED_RECIPE_PREFIXES[strategy]
        except (KeyError, TypeError) as error:
            raise ValueError("GraphBounded recipe strategy is unsupported") from error
        payload_json = _canonical_json(payload)
        return cls(
            recipe_id=prefix + _canonical_hash(payload)[:24],
            payload_json=payload_json,
        )

    @property
    def strategy(self):
        return json.loads(self.payload_json)["strategy"]

    def to_dict(self):
        return {"recipe_id": self.recipe_id, **json.loads(self.payload_json)}


@dataclass(frozen=True)
class _GraphReductionRecipeManifest:
    """One complete typed reduction phase over an immutable Graph scope."""

    recipe_id: str
    payload_json: str

    def __post_init__(self):
        if not isinstance(self.recipe_id, str) or not self.recipe_id:
            raise ValueError("Graph reduction recipe ID must be a nonempty string")
        if not isinstance(self.payload_json, str) or not self.payload_json:
            raise ValueError("Graph reduction recipe payload must be canonical JSON")
        try:
            payload = json.loads(self.payload_json)
        except json.JSONDecodeError as error:
            raise ValueError("Graph reduction recipe payload is invalid") from error
        if (
            not isinstance(payload, dict)
            or _canonical_json(payload) != self.payload_json
        ):
            raise ValueError("Graph reduction recipe payload is not canonical")
        if payload.get("schema_version") != _GRAPH_REDUCTION_RECIPE_SCHEMA_VERSION:
            raise ValueError("Graph reduction recipe schema is unsupported")
        if frozenset(payload) != {
            "schema_version",
            "strategy",
            "semantics",
            "symbolic_abi",
            "topology",
            "physical_stages",
            "workspace",
        }:
            raise ValueError("Graph reduction recipe payload fields are invalid")
        strategy = payload.get("strategy")
        try:
            prefix = _GRAPH_REDUCTION_RECIPE_PREFIXES[strategy]
        except (KeyError, TypeError) as error:
            raise ValueError(
                "Graph reduction recipe strategy is unsupported"
            ) from error
        semantics = payload.get("semantics")
        if not isinstance(semantics, dict) or frozenset(semantics) != {
            "operation",
            "dtype",
            "count",
            "identity",
            "associativity",
            "reduction_order",
            "determinism",
            "absolute_tolerance",
            "relative_tolerance",
            "input",
            "output",
        }:
            raise ValueError("Graph reduction typed semantics are incomplete")
        if semantics.get("operation") != "sum":
            raise ValueError("Graph reduction currently supports only sum")
        if semantics.get("dtype") not in ("f32", "i32"):
            raise ValueError("Graph reduction dtype is unsupported")
        count = semantics.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("Graph reduction count must be positive")
        for name in ("input", "output"):
            if not isinstance(semantics.get(name), str) or not semantics[name]:
                raise ValueError("Graph reduction resource names must be nonempty")
        if semantics["input"] == semantics["output"]:
            raise ValueError("Graph reduction input and output must be distinct")
        if semantics["dtype"] == "f32":
            if (
                semantics.get("determinism") != "within_tolerance"
                or semantics.get("reduction_order") != "relaxed"
                or semantics.get("associativity") != "floating_point_declared_tolerance"
            ):
                raise ValueError("f32 Graph reduction requires relaxed typed semantics")
            tolerances = (
                semantics.get("absolute_tolerance"),
                semantics.get("relative_tolerance"),
            )
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
                for value in tolerances
            ) or not any(float(value) > 0.0 for value in tolerances):
                raise ValueError("f32 Graph reduction requires a positive tolerance")
        elif (
            semantics.get("determinism") != "exact"
            or semantics.get("reduction_order") != "unspecified_integer"
            or semantics.get("associativity") != "modular_integer_sum"
            or semantics.get("absolute_tolerance") != 0.0
            or semantics.get("relative_tolerance") != 0.0
        ):
            raise ValueError("i32 Graph reduction requires exact modular semantics")
        abi = payload.get("symbolic_abi")
        if not isinstance(abi, list) or len(abi) != 2:
            raise ValueError("Graph reduction requires input/output symbolic ABI")
        stages = payload.get("physical_stages")
        if not isinstance(stages, list) or not stages:
            raise ValueError("Graph reduction recipe requires physical stages")
        if any(
            not isinstance(stage, dict)
            or not isinstance(stage.get("name"), str)
            or not stage["name"]
            or isinstance(stage.get("dispatch_count"), bool)
            or not isinstance(stage.get("dispatch_count"), int)
            or stage["dispatch_count"] <= 0
            or not isinstance(stage.get("tasks"), list)
            for stage in stages
        ):
            raise ValueError("Graph reduction physical stage is invalid")
        topology = payload.get("topology")
        if (
            not isinstance(topology, dict)
            or frozenset(topology)
            != {
                "kind",
                "block_dim",
                "items_per_thread",
                "levels",
                "load",
                "in_block_reduction",
            }
            or topology.get("kind") != strategy
            or topology.get("load") != "scalar_coalesced"
            or topology.get("in_block_reduction")
            not in (
                "tls_atomic",
                "shared_tree",
                "warp_shuffle_shared_finalize",
            )
            or isinstance(topology.get("block_dim"), bool)
            or not isinstance(topology.get("block_dim"), int)
            or isinstance(topology.get("items_per_thread"), bool)
            or not isinstance(topology.get("items_per_thread"), int)
            or isinstance(topology.get("levels"), bool)
            or not isinstance(topology.get("levels"), int)
        ):
            raise ValueError("Graph reduction generated topology is invalid")
        workspace = payload.get("workspace")
        if not isinstance(workspace, dict) or frozenset(workspace) != {
            "ownership",
            "exclusive_submission",
            "elements",
            "bytes",
        }:
            raise ValueError("Graph reduction workspace contract is incomplete")
        if workspace["ownership"] not in ("none", "graph_instance"):
            raise ValueError("Graph reduction workspace ownership is invalid")
        for name in ("elements", "bytes"):
            value = workspace[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("Graph reduction workspace size is invalid")
        if strategy == "direct_atomic_tls":
            if (
                topology
                != {
                    "kind": "direct_atomic_tls",
                    "block_dim": 0,
                    "items_per_thread": 1,
                    "levels": 1,
                    "load": "scalar_coalesced",
                    "in_block_reduction": "tls_atomic",
                }
                or workspace
                != {
                    "ownership": "none",
                    "exclusive_submission": False,
                    "elements": 0,
                    "bytes": 0,
                }
                or len(stages) != 1
            ):
                raise ValueError("direct Graph reduction must not own a workspace")
        elif (
            topology["block_dim"] not in (64, 128, 256)
            or topology["items_per_thread"] not in (1, 2, 4)
            or topology["levels"]
            != (3 if strategy == "hierarchical_partial_finalize" else 2)
            or topology["in_block_reduction"]
            not in ("shared_tree", "warp_shuffle_shared_finalize")
            or workspace["ownership"] != "graph_instance"
            or not workspace["exclusive_submission"]
            or workspace["elements"] <= 0
            or workspace["bytes"] <= 0
            or len(stages) != topology["levels"]
        ):
            raise ValueError("partial Graph reduction requires an exclusive workspace")
        expected = prefix + _canonical_hash(payload)[:24]
        if self.recipe_id != expected:
            raise ValueError("Graph reduction recipe ID does not match its payload")

    @classmethod
    def from_payload(cls, payload):
        if not isinstance(payload, dict):
            raise TypeError("Graph reduction recipe payload must be a dictionary")
        payload = dict(payload)
        payload["schema_version"] = _GRAPH_REDUCTION_RECIPE_SCHEMA_VERSION
        strategy = payload.get("strategy")
        try:
            prefix = _GRAPH_REDUCTION_RECIPE_PREFIXES[strategy]
        except (KeyError, TypeError) as error:
            raise ValueError(
                "Graph reduction recipe strategy is unsupported"
            ) from error
        payload_json = _canonical_json(payload)
        return cls(
            recipe_id=prefix + _canonical_hash(payload)[:24],
            payload_json=payload_json,
        )

    @property
    def strategy(self):
        return json.loads(self.payload_json)["strategy"]

    @property
    def semantics(self):
        return json.loads(self.payload_json)["semantics"]

    def to_dict(self):
        return {"recipe_id": self.recipe_id, **json.loads(self.payload_json)}


@dataclass(frozen=True)
class _GraphNativeAlgorithmRecipeManifest:
    """One complete internal strategy for a typed Graph-native algorithm."""

    recipe_id: str
    payload_json: str

    def __post_init__(self):
        if not isinstance(self.recipe_id, str) or not self.recipe_id:
            raise ValueError("Graph native-algorithm recipe ID must be nonempty")
        if not isinstance(self.payload_json, str) or not self.payload_json:
            raise ValueError("Graph native-algorithm payload must be canonical JSON")
        try:
            payload = json.loads(self.payload_json)
        except json.JSONDecodeError as error:
            raise ValueError("Graph native-algorithm payload is invalid") from error
        if (
            not isinstance(payload, dict)
            or _canonical_json(payload) != self.payload_json
        ):
            raise ValueError("Graph native-algorithm payload is not canonical")
        if (
            payload.get("schema_version")
            != _GRAPH_NATIVE_ALGORITHM_RECIPE_SCHEMA_VERSION
        ):
            raise ValueError("Graph native-algorithm schema is unsupported")
        if frozenset(payload) != {
            "schema_version",
            "algorithm",
            "strategy",
            "semantics",
            "physical_stages",
            "topology",
            "workspace",
            "submission",
            "source_lock",
        }:
            raise ValueError("Graph native-algorithm payload fields are invalid")
        key = (payload.get("algorithm"), payload.get("strategy"))
        try:
            prefix = _GRAPH_NATIVE_ALGORITHM_RECIPE_PREFIXES[key]
        except (KeyError, TypeError) as error:
            raise ValueError(
                "Graph native-algorithm strategy is unsupported"
            ) from error
        semantics = payload.get("semantics")
        if not isinstance(semantics, dict) or frozenset(semantics) != {
            "operation",
            "dtype",
            "inclusive",
            "capacity",
            "num_items",
            "num_segments",
            "max_segment_length",
            "topology_fingerprint",
            "input",
            "output",
        }:
            raise ValueError("Graph native-algorithm semantics are incomplete")
        if semantics.get("operation") != "sum" or semantics.get("dtype") not in (
            "i32",
            "u32",
        ):
            raise ValueError("Graph segmented scan typed semantics are unsupported")
        if not isinstance(semantics.get("inclusive"), bool):
            raise ValueError("Graph segmented scan inclusive mode must be bool")
        for name in ("capacity", "num_items", "num_segments", "max_segment_length"):
            value = semantics.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("Graph segmented scan extent is invalid")
        if (
            semantics["capacity"] <= 0
            or semantics["num_segments"] <= 0
            or semantics["num_items"] > semantics["capacity"]
            or not isinstance(semantics.get("topology_fingerprint"), str)
            or not semantics["topology_fingerprint"].startswith("segmented-layout:")
        ):
            raise ValueError("Graph segmented scan topology is invalid")
        for role in ("input", "output"):
            resource = semantics.get(role)
            if (
                not isinstance(resource, dict)
                or frozenset(resource) != {"storage", "shape", "fixed_resource"}
                or resource.get("storage") != "plain_ndarray"
                or resource.get("shape") != [semantics["capacity"]]
                or resource.get("fixed_resource") is not True
            ):
                raise ValueError("Graph segmented scan resource ABI is invalid")
        stages = payload.get("physical_stages")
        if (
            not isinstance(stages, list)
            or not stages
            or any(
                not isinstance(stage, dict)
                or frozenset(stage) != {"name", "execution_kind", "call_count"}
                or not isinstance(stage.get("name"), str)
                or not stage["name"]
                or stage.get("execution_kind")
                not in (
                    "taichi_dispatch",
                    "cuda_program_call",
                )
                or isinstance(stage.get("call_count"), bool)
                or not isinstance(stage.get("call_count"), int)
                or stage["call_count"] <= 0
                for stage in stages
            )
        ):
            raise ValueError("Graph native-algorithm physical stages are invalid")
        workspace = payload.get("workspace")
        if not isinstance(workspace, dict) or frozenset(workspace) != {
            "ownership",
            "action_owned_bytes",
            "provider_shared_scope",
        }:
            raise ValueError("Graph native-algorithm workspace is incomplete")
        if (
            workspace.get("ownership") not in ("none", "graph_native_action")
            or isinstance(workspace.get("action_owned_bytes"), bool)
            or not isinstance(workspace.get("action_owned_bytes"), int)
            or workspace["action_owned_bytes"] < 0
            or workspace.get("provider_shared_scope")
            not in (
                "none",
                "program_scan_arena",
            )
        ):
            raise ValueError("Graph native-algorithm workspace is invalid")
        expected_stages = {
            "segment_local_serial": [
                {
                    "name": "segment_local_serial",
                    "execution_kind": "taichi_dispatch",
                    "call_count": 1,
                }
            ],
            "global_scan_segment_correction": [
                {
                    "name": "copy_input",
                    "execution_kind": "cuda_program_call",
                    "call_count": 1,
                },
                {
                    "name": "global_inclusive_scan",
                    "execution_kind": "cuda_program_call",
                    "call_count": 1,
                },
                {
                    "name": "gather_segment_bases",
                    "execution_kind": "taichi_dispatch",
                    "call_count": 1,
                },
                {
                    "name": "apply_segment_correction",
                    "execution_kind": "taichi_dispatch",
                    "call_count": 1,
                },
            ],
            "warp_chunked_carry": [
                {
                    "name": "warp_sized_segment_chunks",
                    "execution_kind": "taichi_dispatch",
                    "call_count": 1,
                }
            ],
            "block_chunked_carry": [
                {
                    "name": "block_segment_chunks",
                    "execution_kind": "taichi_dispatch",
                    "call_count": 1,
                }
            ],
        }.get(key[1])
        topology = payload.get("topology")
        if not isinstance(topology, dict) or topology.get("kind") != key[1]:
            raise ValueError("Graph native-algorithm topology is incomplete")
        expected_topology = {
            "segment_local_serial": {
                "kind": "segment_local_serial",
                "block_dim": 0,
                "chunk_items": 0,
            },
            "global_scan_segment_correction": {
                "kind": "global_scan_segment_correction",
                "correction_block_dim": 128,
                "correction_graph_nodes": 2,
            },
            "warp_chunked_carry": {
                "kind": "warp_chunked_carry",
                "block_dim": 32,
                "chunk_items": 32,
            },
            "block_chunked_carry": {
                "kind": "block_chunked_carry",
                "block_dim": 128,
                "chunk_items": 128,
            },
        }.get(key[1])
        if key[1] == "length_bucket_hybrid":
            expected_keys = {
                "kind",
                "short_max_items",
                "short_segment_count",
                "long_segment_count",
                "short_block_dim",
                "long_block_dim",
            }
            if (
                frozenset(topology) != expected_keys
                or topology.get("short_max_items") != 32
                or topology.get("short_block_dim") != 32
                or topology.get("long_block_dim") != 128
                or not isinstance(topology.get("short_segment_count"), int)
                or not isinstance(topology.get("long_segment_count"), int)
                or topology["short_segment_count"] <= 0
                or topology["long_segment_count"] <= 0
                or topology["short_segment_count"] + topology["long_segment_count"]
                != semantics["num_segments"]
            ):
                raise ValueError("Graph segmented scan bucket topology is invalid")
            expected_stages = [
                {
                    "name": "short_segment_warp_chunks",
                    "execution_kind": "taichi_dispatch",
                    "call_count": 1,
                },
                {
                    "name": "long_segment_block_chunks",
                    "execution_kind": "taichi_dispatch",
                    "call_count": 1,
                },
            ]
        elif topology != expected_topology:
            raise ValueError("Graph segmented scan topology is invalid")
        if stages != expected_stages:
            raise ValueError(
                "Graph native-algorithm physical topology does not match strategy"
            )
        submission = payload.get("submission")
        if submission != {
            "resource_binding": "fixed_graph_action",
            "exclusive": True,
        }:
            raise ValueError("Graph native-algorithm submission contract is invalid")
        source_lock = payload.get("source_lock")
        if (
            not isinstance(source_lock, str)
            or not source_lock.startswith("sha256:")
            or len(source_lock) != 71
        ):
            raise ValueError("Graph native-algorithm source lock is invalid")
        if key[1] in (
            "segment_local_serial",
            "warp_chunked_carry",
            "block_chunked_carry",
        ):
            if len(stages) != 1 or workspace != {
                "ownership": "none",
                "action_owned_bytes": 0,
                "provider_shared_scope": "none",
            }:
                raise ValueError("workspace-free segmented scan recipe is incomplete")
        elif key[1] == "global_scan_segment_correction" and (
            workspace["ownership"] != "graph_native_action"
            or workspace["action_owned_bytes"] != semantics["num_segments"] * 4
            or workspace["provider_shared_scope"] != "program_scan_arena"
        ):
            raise ValueError("global segmented scan recipe is not complete")
        elif key[1] == "length_bucket_hybrid" and (
            workspace["ownership"] != "graph_native_action"
            or workspace["action_owned_bytes"] != semantics["num_segments"] * 4
            or workspace["provider_shared_scope"] != "none"
        ):
            raise ValueError("hybrid segmented scan recipe is not complete")
        expected = prefix + _canonical_hash(payload)[:24]
        if self.recipe_id != expected:
            raise ValueError(
                "Graph native-algorithm recipe ID does not match its payload"
            )

    @classmethod
    def from_payload(cls, payload):
        if not isinstance(payload, dict):
            raise TypeError("Graph native-algorithm payload must be a dictionary")
        payload = dict(payload)
        payload["schema_version"] = _GRAPH_NATIVE_ALGORITHM_RECIPE_SCHEMA_VERSION
        key = (payload.get("algorithm"), payload.get("strategy"))
        try:
            prefix = _GRAPH_NATIVE_ALGORITHM_RECIPE_PREFIXES[key]
        except (KeyError, TypeError) as error:
            raise ValueError(
                "Graph native-algorithm strategy is unsupported"
            ) from error
        payload_json = _canonical_json(payload)
        return cls(
            recipe_id=prefix + _canonical_hash(payload)[:24],
            payload_json=payload_json,
        )

    @property
    def algorithm(self):
        return json.loads(self.payload_json)["algorithm"]

    @property
    def strategy(self):
        return json.loads(self.payload_json)["strategy"]

    @property
    def semantics(self):
        return json.loads(self.payload_json)["semantics"]

    def to_dict(self):
        return {"recipe_id": self.recipe_id, **json.loads(self.payload_json)}


@dataclass(frozen=True)
class _ExecutableOptimizationSpec:
    spec_id: str
    semantic_plan_id: str
    backend: str
    fusion_recipe_ids: tuple
    compilation_identity: str
    execution_identity: str
    control_recipe_id: str = ""
    fusion_source_groups: tuple = ()
    memory_recipe_id: str = ""
    memory_recipe_manifest: object = None
    bounded_recipe_id: str = ""
    bounded_recipe_manifest: object = None
    reduction_recipe_id: str = ""
    reduction_recipe_manifest: object = None
    native_algorithm_recipe_id: str = ""
    native_algorithm_recipe_manifest: object = None

    def __post_init__(self):
        if not self.spec_id.startswith("executable:"):
            raise ValueError("executable optimization spec ID is invalid")
        if not self.semantic_plan_id.startswith("semantic-plan:"):
            raise ValueError("semantic plan ID is invalid")
        if not isinstance(self.backend, str) or not self.backend:
            raise ValueError("executable optimization backend is invalid")
        if len(set(self.fusion_recipe_ids)) != len(self.fusion_recipe_ids):
            raise ValueError("fusion recipe IDs must be unique")
        if self.fusion_source_groups and len(self.fusion_source_groups) != len(
            self.fusion_recipe_ids
        ):
            raise ValueError(
                "fusion source groups must correspond to every fusion recipe"
            )
        claimed_dispatches = set()
        for group in self.fusion_source_groups:
            if (
                not isinstance(group, tuple)
                or len(group) < 2
                or len(group) > 4
                or any(
                    isinstance(item, bool) or not isinstance(item, int) or item < 0
                    for item in group
                )
                or tuple(range(group[0], group[0] + len(group))) != group
            ):
                raise ValueError(
                    "fusion source groups must be contiguous logical dispatch IDs"
                )
            if claimed_dispatches.intersection(group):
                raise ValueError("fusion source groups must be disjoint")
            claimed_dispatches.update(group)
        if not isinstance(self.control_recipe_id, str):
            raise ValueError("control recipe ID must be a string")
        if (
            self.control_recipe_id
            and self.control_recipe_id not in _CUDA_STRUCTURED_CONTROL_RECIPE_IDS
        ):
            raise ValueError("control recipe ID is unsupported")
        if self.control_recipe_id and self.fusion_recipe_ids:
            raise ValueError(
                "executable specs cannot combine control and fusion recipes"
            )
        if self.control_recipe_id and self.fusion_source_groups:
            raise ValueError(
                "structured-control specs cannot contain fusion source groups"
            )
        if bool(self.memory_recipe_id) != bool(self.memory_recipe_manifest):
            raise ValueError(
                "memory recipe ID and complete manifest must be provided together"
            )
        if self.memory_recipe_manifest is not None:
            if not isinstance(self.memory_recipe_manifest, _GraphMemoryRecipeManifest):
                raise TypeError(
                    "memory recipe manifest must be a _GraphMemoryRecipeManifest"
                )
            if self.memory_recipe_id != self.memory_recipe_manifest.recipe_id:
                raise ValueError("memory recipe ID does not match its manifest")
            if self.control_recipe_id or self.fusion_recipe_ids:
                raise ValueError(
                    "executable specs cannot combine memory with control or fusion"
                )
            if self.fusion_source_groups:
                raise ValueError(
                    "GraphMemory specs cannot contain fusion source groups"
                )
        if bool(self.bounded_recipe_id) != bool(self.bounded_recipe_manifest):
            raise ValueError(
                "bounded recipe ID and complete manifest must be provided together"
            )
        if self.bounded_recipe_manifest is not None:
            if not isinstance(
                self.bounded_recipe_manifest,
                _GraphBoundedExecutionRecipeManifest,
            ):
                raise TypeError(
                    "bounded recipe manifest must be a "
                    "_GraphBoundedExecutionRecipeManifest"
                )
            if self.bounded_recipe_id != self.bounded_recipe_manifest.recipe_id:
                raise ValueError("bounded recipe ID does not match its manifest")
            if (
                self.control_recipe_id
                or self.fusion_recipe_ids
                or self.memory_recipe_id
                or self.fusion_source_groups
            ):
                raise ValueError(
                    "executable specs cannot combine bounded execution with "
                    "control, fusion, or GraphMemory"
                )
        if bool(self.reduction_recipe_id) != bool(self.reduction_recipe_manifest):
            raise ValueError(
                "reduction recipe ID and complete manifest must be provided together"
            )
        if self.reduction_recipe_manifest is not None:
            if not isinstance(
                self.reduction_recipe_manifest,
                _GraphReductionRecipeManifest,
            ):
                raise TypeError(
                    "reduction recipe manifest must be a "
                    "_GraphReductionRecipeManifest"
                )
            if self.reduction_recipe_id != self.reduction_recipe_manifest.recipe_id:
                raise ValueError("reduction recipe ID does not match its manifest")
            if (
                self.control_recipe_id
                or self.fusion_recipe_ids
                or self.memory_recipe_id
                or self.bounded_recipe_id
                or self.fusion_source_groups
            ):
                raise ValueError(
                    "executable specs cannot combine Graph reduction with control, "
                    "fusion, GraphMemory, or bounded execution"
                )
        if bool(self.native_algorithm_recipe_id) != bool(
            self.native_algorithm_recipe_manifest
        ):
            raise ValueError(
                "native-algorithm recipe ID and manifest must be provided together"
            )
        if self.native_algorithm_recipe_manifest is not None:
            if not isinstance(
                self.native_algorithm_recipe_manifest,
                _GraphNativeAlgorithmRecipeManifest,
            ):
                raise TypeError(
                    "native-algorithm recipe manifest must be a "
                    "_GraphNativeAlgorithmRecipeManifest"
                )
            if (
                self.native_algorithm_recipe_id
                != self.native_algorithm_recipe_manifest.recipe_id
            ):
                raise ValueError(
                    "native-algorithm recipe ID does not match its manifest"
                )
            if (
                self.control_recipe_id
                or self.fusion_recipe_ids
                or self.memory_recipe_id
                or self.bounded_recipe_id
                or self.reduction_recipe_id
                or self.fusion_source_groups
            ):
                raise ValueError(
                    "executable specs cannot combine a Graph native algorithm "
                    "with another recipe axis"
                )
        if not self.compilation_identity or not self.execution_identity:
            raise ValueError("executable optimization identities are required")

    def to_dict(self):
        value = {
            "schema_version": (
                6
                if self.native_algorithm_recipe_id
                else (
                    5
                    if self.reduction_recipe_id
                    else (
                        4
                        if self.bounded_recipe_id
                        else (3 if self.memory_recipe_id else 2)
                    )
                )
            ),
            "spec_id": self.spec_id,
            "semantic_plan_id": self.semantic_plan_id,
            "backend": self.backend,
            "fusion_recipe_ids": self.fusion_recipe_ids,
            "compilation_identity": self.compilation_identity,
            "execution_identity": self.execution_identity,
            "fusion_source_groups": self.fusion_source_groups,
        }
        # Preserve the exact v1 map-fusion manifest and identities when this
        # optional physical axis is absent.
        if self.control_recipe_id:
            value["control_recipe_id"] = self.control_recipe_id
        if self.memory_recipe_id:
            value["memory_recipe_id"] = self.memory_recipe_id
            value["memory_recipe_manifest"] = self.memory_recipe_manifest.to_dict()
        if self.bounded_recipe_id:
            value["bounded_recipe_id"] = self.bounded_recipe_id
            value["bounded_recipe_manifest"] = self.bounded_recipe_manifest.to_dict()
        if self.reduction_recipe_id:
            value["reduction_recipe_id"] = self.reduction_recipe_id
            value["reduction_recipe_manifest"] = (
                self.reduction_recipe_manifest.to_dict()
            )
        if self.native_algorithm_recipe_id:
            value["native_algorithm_recipe_id"] = self.native_algorithm_recipe_id
            value["native_algorithm_recipe_manifest"] = (
                self.native_algorithm_recipe_manifest.to_dict()
            )
        return value


@dataclass(frozen=True)
class _ExecutableOptimizationSpace:
    semantic_plan_id: str
    baseline: _ExecutableOptimizationSpec
    candidates: tuple
    selected_spec_id: object
    selection_status: str
    partition_stage: str = "exact_contiguous_v1"
    partitions_complete: bool = True
    partition_combination_count: int = 1
    partition_candidate_limit: int = 4095
    partition_parent_domain_fingerprint: str = ""
    partition_frontier_spec_ids: tuple = ()

    @property
    def selected(self):
        for spec in (self.baseline, *self.candidates):
            if spec.spec_id == self.selected_spec_id:
                return spec
        return None

    def to_dict(self):
        return {
            "schema_version": 2,
            "semantic_plan_id": self.semantic_plan_id,
            "baseline": self.baseline.to_dict(),
            "candidates": tuple(spec.to_dict() for spec in self.candidates),
            "selected_spec_id": self.selected_spec_id,
            "selected": None if self.selected is None else self.selected.to_dict(),
            "selection_status": self.selection_status,
            "partition_stage": self.partition_stage,
            "partitions_complete": self.partitions_complete,
            "partition_combination_count": self.partition_combination_count,
            "partition_candidate_limit": self.partition_candidate_limit,
            "partition_parent_domain_fingerprint": (
                self.partition_parent_domain_fingerprint or None
            ),
            "partition_frontier_spec_ids": self.partition_frontier_spec_ids,
        }


def _required_string(value, role):
    if not isinstance(value, str) or not value or "\n" in value:
        raise ValueError(f"{role} must be a nonempty single-line string")
    return value


def _optional_finite_number(value, role):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{role} must be a finite number or null")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{role} must be finite")
    return value


def _positive_shape(value, role):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{role} must be a nonempty shape")
    shape = tuple(int(extent) for extent in value)
    if any(
        isinstance(extent, bool) or not isinstance(extent, int) for extent in value
    ) or any(extent <= 0 for extent in shape):
        raise ValueError(f"{role} extents must be positive integers")
    return shape


@dataclass(frozen=True)
class _GraphFusionBindingScope:
    name: str
    kind: str
    dtype: str = ""
    rank: int = 0
    element_shape: tuple = ()
    shape_min: tuple = ()
    shape_max: tuple = ()
    scalar_min: object = None
    scalar_max: object = None

    def __post_init__(self):
        _required_string(self.name, "fusion binding name")
        if self.kind not in ("ndarray", "scalar"):
            raise ValueError("fusion binding kind must be ndarray or scalar")
        if self.kind == "ndarray":
            _required_string(self.dtype, "fusion ndarray dtype")
            if (
                isinstance(self.rank, bool)
                or not isinstance(self.rank, int)
                or self.rank < 1
                or len(self.shape_min) != self.rank
                or len(self.shape_max) != self.rank
            ):
                raise ValueError("fusion ndarray rank and shape bounds disagree")
            if any(
                minimum <= 0 or maximum < minimum
                for minimum, maximum in zip(self.shape_min, self.shape_max)
            ):
                raise ValueError("fusion ndarray shape bounds are invalid")
            if self.scalar_min is not None or self.scalar_max is not None:
                raise ValueError("fusion ndarray scope cannot contain scalar bounds")
        elif (
            self.dtype
            or self.rank
            or self.element_shape
            or self.shape_min
            or self.shape_max
        ):
            raise ValueError("fusion scalar scope cannot contain ndarray metadata")
        if (
            self.scalar_min is not None
            and self.scalar_max is not None
            and self.scalar_max < self.scalar_min
        ):
            raise ValueError("fusion scalar bounds are invalid")

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("fusion binding scope must be an object")
        kind = value.get("kind")
        if kind == "ndarray":
            shape_min = _positive_shape(
                value.get("shape_min"), "fusion ndarray shape_min"
            )
            shape_max = _positive_shape(
                value.get("shape_max"), "fusion ndarray shape_max"
            )
            rank = value.get("rank")
            if isinstance(rank, bool) or not isinstance(rank, int):
                raise ValueError("fusion ndarray rank must be an integer")
            element_shape = value.get("element_shape", ())
            if not isinstance(element_shape, (list, tuple)):
                raise ValueError("fusion ndarray element_shape must be a shape")
            element_shape = tuple(int(extent) for extent in element_shape)
            if any(extent <= 0 for extent in element_shape):
                raise ValueError(
                    "fusion ndarray element_shape extents must be positive"
                )
            return cls(
                name=_required_string(value.get("name"), "fusion binding name"),
                kind=kind,
                dtype=_required_string(value.get("dtype"), "fusion ndarray dtype"),
                rank=rank,
                element_shape=element_shape,
                shape_min=shape_min,
                shape_max=shape_max,
            )
        if kind == "scalar":
            return cls(
                name=_required_string(value.get("name"), "fusion binding name"),
                kind=kind,
                scalar_min=_optional_finite_number(
                    value.get("minimum"), "fusion scalar minimum"
                ),
                scalar_max=_optional_finite_number(
                    value.get("maximum"), "fusion scalar maximum"
                ),
            )
        raise ValueError("fusion binding kind must be ndarray or scalar")

    def matches(self, descriptor):
        if not isinstance(descriptor, dict) or descriptor.get("kind") != self.kind:
            return False
        if self.kind == "ndarray":
            shape = tuple(descriptor.get("shape", ()))
            return bool(
                descriptor.get("dtype") == self.dtype
                and int(descriptor.get("rank", -1)) == self.rank
                and tuple(descriptor.get("element_shape", ())) == self.element_shape
                and len(shape) == self.rank
                and all(
                    minimum <= extent <= maximum
                    for extent, minimum, maximum in zip(
                        shape, self.shape_min, self.shape_max
                    )
                )
            )
        value = descriptor.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        value = float(value)
        return bool(
            math.isfinite(value)
            and (self.scalar_min is None or value >= self.scalar_min)
            and (self.scalar_max is None or value <= self.scalar_max)
        )

    def to_dict(self):
        if self.kind == "ndarray":
            return {
                "name": self.name,
                "kind": self.kind,
                "dtype": self.dtype,
                "rank": self.rank,
                "element_shape": self.element_shape,
                "shape_min": self.shape_min,
                "shape_max": self.shape_max,
            }
        return {
            "name": self.name,
            "kind": self.kind,
            "minimum": self.scalar_min,
            "maximum": self.scalar_max,
        }


@dataclass(frozen=True)
class _GraphFusionQualificationEntry:
    semantic_plan_id: str
    backend: str
    baseline_execution_identity: str
    selected_spec_id: str
    execution_identity: str
    source_commit: str
    runtime_scope: tuple
    binding_scopes: tuple
    minimum_expected_replays: int
    evidence_id: str

    def __post_init__(self):
        if not self.semantic_plan_id.startswith("semantic-plan:"):
            raise ValueError("fusion qualification semantic plan ID is invalid")
        _required_string(self.backend, "fusion qualification backend")
        _required_string(
            self.baseline_execution_identity,
            "fusion qualification baseline execution identity",
        )
        if not self.selected_spec_id.startswith("executable:"):
            raise ValueError("fusion qualification executable spec ID is invalid")
        _required_string(
            self.execution_identity, "fusion qualification execution identity"
        )
        if (
            not isinstance(self.source_commit, str)
            or len(self.source_commit) != 40
            or any(
                character not in "0123456789abcdef" for character in self.source_commit
            )
        ):
            raise ValueError("fusion qualification source commit is invalid")
        if not self.runtime_scope:
            raise ValueError("fusion qualification runtime scope is required")
        if len({scope.name for scope in self.binding_scopes}) != len(
            self.binding_scopes
        ):
            raise ValueError("fusion qualification binding names must be unique")
        if (
            isinstance(self.minimum_expected_replays, bool)
            or not isinstance(self.minimum_expected_replays, int)
            or self.minimum_expected_replays < 1
        ):
            raise ValueError(
                "fusion qualification minimum_expected_replays must be positive"
            )
        _required_string(self.evidence_id, "fusion qualification evidence ID")
        # Qualification entries are immutable. Their canonical identity is an
        # admission-time fact, not work to repeat on every Graph replay.
        object.__setattr__(self, "_identity", _canonical_hash(self.to_dict()))

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("fusion qualification entry must be an object")
        qualification = value.get("qualification")
        if not isinstance(qualification, dict) or any(
            qualification.get(name) is not True
            for name in ("correctness", "memory_stable", "worst_positive")
        ):
            raise ValueError(
                "fusion qualification entry did not pass every admission gate"
            )
        runtime_scope = value.get("runtime_scope")
        if not isinstance(runtime_scope, dict) or not runtime_scope:
            raise ValueError("fusion qualification runtime_scope is required")
        normalized_scope = []
        for key, item in sorted(runtime_scope.items()):
            _required_string(key, "fusion runtime scope key")
            if isinstance(item, bool) or not isinstance(item, (str, int)):
                raise ValueError(
                    "fusion runtime scope values must be strings or integers"
                )
            normalized_scope.append((key, item))
        raw_bindings = value.get("binding_scope", ())
        if not isinstance(raw_bindings, (list, tuple)):
            raise ValueError("fusion qualification binding_scope must be a list")
        minimum_expected_replays = value.get("minimum_expected_replays")
        if isinstance(minimum_expected_replays, bool) or not isinstance(
            minimum_expected_replays, int
        ):
            raise ValueError(
                "fusion qualification minimum_expected_replays must be an integer"
            )
        return cls(
            semantic_plan_id=_required_string(
                value.get("semantic_plan_id"),
                "fusion qualification semantic plan ID",
            ),
            backend=_required_string(
                value.get("backend"), "fusion qualification backend"
            ),
            baseline_execution_identity=_required_string(
                value.get("baseline_execution_identity"),
                "fusion qualification baseline execution identity",
            ),
            selected_spec_id=_required_string(
                value.get("selected_spec_id"),
                "fusion qualification executable spec ID",
            ),
            execution_identity=_required_string(
                value.get("execution_identity"),
                "fusion qualification execution identity",
            ),
            source_commit=_required_string(
                value.get("source_commit"), "fusion qualification source commit"
            ).lower(),
            runtime_scope=tuple(normalized_scope),
            binding_scopes=tuple(
                _GraphFusionBindingScope.from_dict(item) for item in raw_bindings
            ),
            minimum_expected_replays=minimum_expected_replays,
            evidence_id=_required_string(
                value.get("evidence_id"), "fusion qualification evidence ID"
            ),
        )

    @property
    def identity(self):
        return self._identity

    def matches(
        self,
        *,
        semantic_plan_id,
        backend,
        source_commit,
        runtime_scope,
        bindings,
        expected_replays,
    ):
        if (
            semantic_plan_id != self.semantic_plan_id
            or backend != self.backend
            or source_commit != self.source_commit
            or expected_replays < self.minimum_expected_replays
            or tuple(sorted(runtime_scope.items())) != self.runtime_scope
        ):
            return False
        return len(self.binding_scopes) == len(bindings) and all(
            scope.name in bindings and scope.matches(bindings[scope.name])
            for scope in self.binding_scopes
        )

    def to_dict(self):
        return {
            "semantic_plan_id": self.semantic_plan_id,
            "backend": self.backend,
            "baseline_execution_identity": self.baseline_execution_identity,
            "selected_spec_id": self.selected_spec_id,
            "execution_identity": self.execution_identity,
            "source_commit": self.source_commit,
            "runtime_scope": dict(self.runtime_scope),
            "binding_scope": tuple(scope.to_dict() for scope in self.binding_scopes),
            "minimum_expected_replays": self.minimum_expected_replays,
            "evidence_id": self.evidence_id,
            "qualification": {
                "correctness": True,
                "memory_stable": True,
                "worst_positive": True,
            },
        }


@dataclass(frozen=True)
class _GraphFusionQualificationCache:
    entries: tuple
    source_path: str = ""

    @classmethod
    def from_dict(cls, value, *, source_path=""):
        if not isinstance(value, dict):
            raise ValueError("graph fusion qualification cache must be an object")
        if value.get("schema") != _GRAPH_FUSION_QUALIFICATION_SCHEMA:
            raise ValueError("graph fusion qualification cache schema is invalid")
        raw_entries = value.get("entries")
        if not isinstance(raw_entries, list):
            raise ValueError("graph fusion qualification entries must be a list")
        entries = tuple(
            _GraphFusionQualificationEntry.from_dict(item) for item in raw_entries
        )
        identities = tuple(entry.identity for entry in entries)
        if len(set(identities)) != len(identities):
            raise ValueError("graph fusion qualification entries must be unique")
        return cls(entries=entries, source_path=str(source_path))

    @classmethod
    def load(cls, path):
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"graph fusion qualification cache does not exist: {path}")
        if path.stat().st_size > _GRAPH_FUSION_QUALIFICATION_MAX_BYTES:
            raise ValueError("graph fusion qualification cache is too large")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("graph fusion qualification cache is invalid") from exc
        return cls.from_dict(value, source_path=str(path))

    def select(self, **scope):
        matches = tuple(entry for entry in self.entries if entry.matches(**scope))
        if not matches:
            return None, "no_exact_qualification"
        if len(matches) != 1:
            return None, "ambiguous_qualification"
        return matches[0], "qualified"

    def to_dict(self):
        return {
            "schema": _GRAPH_FUSION_QUALIFICATION_SCHEMA,
            "entries": tuple(entry.to_dict() for entry in self.entries),
        }


def _make_spec(
    semantic_plan_id,
    backend,
    fusion_recipe_ids,
    control_recipe_id="",
    *,
    fusion_source_groups=(),
    memory_recipe_manifest=None,
    bounded_recipe_manifest=None,
    reduction_recipe_manifest=None,
    native_algorithm_recipe_manifest=None,
):
    fusion_recipe_ids = tuple(fusion_recipe_ids)
    fusion_source_groups = tuple(tuple(group) for group in fusion_source_groups)
    dispatch_reduction = 0
    for recipe_id in fusion_recipe_ids:
        fields = recipe_id.split(":")
        source_count = 2
        if not (
            len(fields) != 3 or fields[0] != "fusion" or not fields[1].startswith("map")
        ):
            try:
                parsed_count = int(fields[1][3:])
            except ValueError:
                parsed_count = 2
            if 2 <= parsed_count <= 4:
                source_count = parsed_count
        dispatch_reduction += source_count - 1
    compilation_payload = {
        "semantic_plan_id": semantic_plan_id,
        "backend": backend,
        "fusion_recipe_ids": fusion_recipe_ids,
        "fusion_source_groups": fusion_source_groups,
    }
    if control_recipe_id:
        compilation_payload["control_recipe_id"] = control_recipe_id
    memory_recipe_id = ""
    if memory_recipe_manifest is not None:
        if not isinstance(memory_recipe_manifest, _GraphMemoryRecipeManifest):
            raise TypeError(
                "memory_recipe_manifest must be a _GraphMemoryRecipeManifest"
            )
        memory_recipe_id = memory_recipe_manifest.recipe_id
        compilation_payload["memory_recipe"] = memory_recipe_manifest.to_dict()
    bounded_recipe_id = ""
    if bounded_recipe_manifest is not None:
        if not isinstance(
            bounded_recipe_manifest,
            _GraphBoundedExecutionRecipeManifest,
        ):
            raise TypeError(
                "bounded_recipe_manifest must be a "
                "_GraphBoundedExecutionRecipeManifest"
            )
        bounded_recipe_id = bounded_recipe_manifest.recipe_id
        compilation_payload["bounded_recipe"] = bounded_recipe_manifest.to_dict()
    reduction_recipe_id = ""
    if reduction_recipe_manifest is not None:
        if not isinstance(
            reduction_recipe_manifest,
            _GraphReductionRecipeManifest,
        ):
            raise TypeError(
                "reduction_recipe_manifest must be a " "_GraphReductionRecipeManifest"
            )
        reduction_recipe_id = reduction_recipe_manifest.recipe_id
        compilation_payload["reduction_recipe"] = reduction_recipe_manifest.to_dict()
    native_algorithm_recipe_id = ""
    if native_algorithm_recipe_manifest is not None:
        if not isinstance(
            native_algorithm_recipe_manifest,
            _GraphNativeAlgorithmRecipeManifest,
        ):
            raise TypeError(
                "native_algorithm_recipe_manifest must be a "
                "_GraphNativeAlgorithmRecipeManifest"
            )
        native_algorithm_recipe_id = native_algorithm_recipe_manifest.recipe_id
        compilation_payload["native_algorithm_recipe"] = (
            native_algorithm_recipe_manifest.to_dict()
        )
    compilation_identity = _canonical_hash(compilation_payload)
    execution_payload = {
        "compilation_identity": compilation_identity,
        "physical_dispatch_delta": -dispatch_reduction,
        "fusion_source_groups": fusion_source_groups,
        "memory_recipe_id": memory_recipe_id,
        "bounded_recipe_id": bounded_recipe_id,
        "reduction_recipe_id": reduction_recipe_id,
    }
    # Keep every established Graph recipe identity byte-for-byte stable when
    # the optional native-algorithm axis is absent.
    if native_algorithm_recipe_id:
        execution_payload["native_algorithm_recipe_id"] = native_algorithm_recipe_id
    execution_identity = _canonical_hash(execution_payload)
    return _ExecutableOptimizationSpec(
        spec_id=f"executable:{compilation_identity[:24]}",
        semantic_plan_id=semantic_plan_id,
        backend=backend,
        fusion_recipe_ids=fusion_recipe_ids,
        compilation_identity=compilation_identity,
        execution_identity=execution_identity,
        control_recipe_id=control_recipe_id,
        fusion_source_groups=fusion_source_groups,
        memory_recipe_id=memory_recipe_id,
        memory_recipe_manifest=memory_recipe_manifest,
        bounded_recipe_id=bounded_recipe_id,
        bounded_recipe_manifest=bounded_recipe_manifest,
        reduction_recipe_id=reduction_recipe_id,
        reduction_recipe_manifest=reduction_recipe_manifest,
        native_algorithm_recipe_id=native_algorithm_recipe_id,
        native_algorithm_recipe_manifest=native_algorithm_recipe_manifest,
    )


def _fusion_source_groups(fusion_plan, fusion_recipe_ids):
    recipes = {recipe.recipe_id: recipe for recipe in fusion_plan.candidate_recipes}
    groups = []
    claimed = set()
    for recipe_id in fusion_recipe_ids:
        try:
            recipe = recipes[recipe_id]
        except KeyError as error:
            raise ValueError(
                f"fusion recipe {recipe_id!r} is absent from the semantic plan"
            ) from error
        group = []
        for source_id in recipe.source_dispatch_ids:
            marker = source_id.rsplit("/dispatch:", 1)
            if len(marker) != 2 or not marker[1].isdigit():
                raise ValueError("fusion recipe has no exact logical dispatch lineage")
            logical_id = int(marker[1])
            if logical_id in claimed:
                raise ValueError("fusion partition recipes overlap a dispatch")
            claimed.add(logical_id)
            group.append(logical_id)
        groups.append(tuple(group))
    return tuple(groups)


def _build_executable_optimization_space(
    root,
    fusion_plan,
    backend,
    *,
    control_recipe_ids=(),
    selected_control_recipe_id="",
    memory_recipe_manifests=(),
    selected_memory_recipe_id="",
    bounded_recipe_manifests=(),
    selected_bounded_recipe_id="",
    reduction_recipe_manifests=(),
    selected_reduction_recipe_id="",
    native_algorithm_recipe_manifests=(),
    selected_native_algorithm_recipe_id="",
    semantic_root=None,
):
    semantic_digest = _canonical_hash(
        graph_ir_to_dict(root if semantic_root is None else semantic_root)
    )
    semantic_plan_id = f"semantic-plan:{semantic_digest[:24]}"
    control_recipe_ids = tuple(control_recipe_ids)
    memory_recipe_manifests = tuple(memory_recipe_manifests)
    bounded_recipe_manifests = tuple(bounded_recipe_manifests)
    reduction_recipe_manifests = tuple(reduction_recipe_manifests)
    native_algorithm_recipe_manifests = tuple(native_algorithm_recipe_manifests)
    if native_algorithm_recipe_manifests:
        if (
            control_recipe_ids
            or memory_recipe_manifests
            or bounded_recipe_manifests
            or reduction_recipe_manifests
        ):
            raise ValueError(
                "Graph native-algorithm recipes cannot combine with another axis"
            )
        if fusion_plan.applied_groups or any(fusion_plan.candidate_partitions):
            raise ValueError(
                "Graph native-algorithm recipes cannot combine with fusion"
            )
        if len(native_algorithm_recipe_manifests) != 2:
            raise ValueError(
                "the initial Graph native-algorithm domain requires two recipes"
            )
        algorithms = {
            manifest.algorithm for manifest in native_algorithm_recipe_manifests
        }
        strategies = {
            manifest.strategy for manifest in native_algorithm_recipe_manifests
        }
        scopes = {
            _canonical_json(manifest.semantics)
            for manifest in native_algorithm_recipe_manifests
        }
        if (
            algorithms != {"segmented_scan"}
            or strategies != {"segment_local_serial", "global_scan_segment_correction"}
            or len(scopes) != 1
        ):
            raise ValueError(
                "Graph native-algorithm domain is incomplete or semantically mixed"
            )
        specs = tuple(
            _make_spec(
                semantic_plan_id,
                backend,
                (),
                native_algorithm_recipe_manifest=manifest,
            )
            for manifest in native_algorithm_recipe_manifests
        )
        selected = next(
            (
                spec
                for spec in specs
                if spec.native_algorithm_recipe_id
                == selected_native_algorithm_recipe_id
            ),
            None,
        )
        return _ExecutableOptimizationSpace(
            semantic_plan_id=semantic_plan_id,
            baseline=specs[0],
            candidates=specs[1:],
            selected_spec_id=None if selected is None else selected.spec_id,
            selection_status=(
                "native_algorithm_recipe_not_materialized"
                if selected is None
                else (
                    "selected_native_algorithm_baseline"
                    if selected is specs[0]
                    else "selected_native_algorithm_recipe"
                )
            ),
            partition_stage="graph_native_algorithm_complete_recipe",
            partitions_complete=True,
            partition_combination_count=len(specs),
        )
    if reduction_recipe_manifests:
        if (
            control_recipe_ids
            or memory_recipe_manifests
            or bounded_recipe_manifests
            or native_algorithm_recipe_manifests
        ):
            raise ValueError(
                "Graph reduction recipes cannot combine with control, GraphMemory, "
                "or bounded execution"
            )
        if fusion_plan.applied_groups or any(fusion_plan.candidate_partitions):
            raise ValueError("Graph reduction recipes cannot combine with fusion")
        if len(reduction_recipe_manifests) < 2:
            raise ValueError(
                "Graph reduction domain requires direct and generated candidates"
            )
        if reduction_recipe_manifests[0].strategy != "direct_atomic_tls" or any(
            manifest.strategy
            not in (
                "block_partial_finalize",
                "hierarchical_partial_finalize",
            )
            for manifest in reduction_recipe_manifests[1:]
        ):
            raise ValueError(
                "Graph reduction domain must order direct before generated topologies"
            )
        scopes = {
            _canonical_json(manifest.semantics)
            for manifest in reduction_recipe_manifests
        }
        if len(scopes) != 1:
            raise ValueError("Graph reduction recipes must share typed semantics")
        specs = tuple(
            _make_spec(
                semantic_plan_id,
                backend,
                (),
                reduction_recipe_manifest=manifest,
            )
            for manifest in reduction_recipe_manifests
        )
        selected = next(
            (
                spec
                for spec in specs
                if spec.reduction_recipe_id == selected_reduction_recipe_id
            ),
            None,
        )
        return _ExecutableOptimizationSpace(
            semantic_plan_id=semantic_plan_id,
            baseline=specs[0],
            candidates=specs[1:],
            selected_spec_id=None if selected is None else selected.spec_id,
            selection_status=(
                "reduction_recipe_not_materialized"
                if selected is None
                else (
                    "selected_reduction_baseline"
                    if selected is specs[0]
                    else "selected_reduction_recipe"
                )
            ),
            partition_stage="graph_reduction_complete_recipe",
            partitions_complete=True,
            partition_combination_count=len(specs),
        )
    if bounded_recipe_manifests:
        if control_recipe_ids or memory_recipe_manifests or reduction_recipe_manifests:
            raise ValueError(
                "GraphBounded recipes cannot combine with control or GraphMemory"
            )
        if fusion_plan.applied_groups or any(fusion_plan.candidate_partitions):
            raise ValueError("GraphBounded recipes cannot combine with fusion")
        strategies = tuple(manifest.strategy for manifest in bounded_recipe_manifests)
        scopes = {
            _canonical_json(
                {
                    key: value
                    for key, value in manifest.to_dict().items()
                    if key not in ("recipe_id", "strategy")
                }
            )
            for manifest in bounded_recipe_manifests
        }
        if len(scopes) != 1:
            raise ValueError("GraphBounded recipes must share one exact scope")
        allowed_domains = (
            ("logical_exact", "masked_capacity"),
            ("logical_exact", "adaptive_per_node", "masked_capacity"),
            (
                "logical_exact",
                "adaptive_per_node",
                "adaptive_grouped",
                "masked_capacity",
            ),
        )
        if strategies not in allowed_domains:
            raise ValueError("GraphBounded recipe domain is incomplete or unordered")
        specs = tuple(
            _make_spec(
                semantic_plan_id,
                backend,
                (),
                bounded_recipe_manifest=manifest,
            )
            for manifest in bounded_recipe_manifests
        )
        selected = next(
            (
                spec
                for spec in specs
                if spec.bounded_recipe_id == selected_bounded_recipe_id
            ),
            None,
        )
        return _ExecutableOptimizationSpace(
            semantic_plan_id=semantic_plan_id,
            baseline=specs[0],
            candidates=specs[1:],
            selected_spec_id=None if selected is None else selected.spec_id,
            selection_status=(
                "bounded_recipe_not_materialized"
                if selected is None
                else (
                    "selected_bounded_baseline"
                    if selected is specs[0]
                    else "selected_bounded_recipe"
                )
            ),
            partition_stage="graph_bounded_complete_recipe",
            partitions_complete=True,
            partition_combination_count=len(specs),
        )
    if memory_recipe_manifests:
        if control_recipe_ids:
            raise ValueError("GraphMemory recipes cannot combine with control")
        if fusion_plan.applied_groups or any(fusion_plan.candidate_partitions):
            raise ValueError("GraphMemory recipes cannot combine with fusion")
        if len(memory_recipe_manifests) < 2:
            raise ValueError(
                "GraphMemory domain requires direct and generated candidates"
            )
        if memory_recipe_manifests[0].strategy != "direct" or any(
            manifest.strategy != "shared_staged_1d"
            for manifest in memory_recipe_manifests[1:]
        ):
            raise ValueError("GraphMemory domain must order direct before shared-stage")
        specs = tuple(
            _make_spec(
                semantic_plan_id,
                backend,
                (),
                memory_recipe_manifest=manifest,
            )
            for manifest in memory_recipe_manifests
        )
        selected = next(
            (
                spec
                for spec in specs
                if spec.memory_recipe_id == selected_memory_recipe_id
            ),
            None,
        )
        return _ExecutableOptimizationSpace(
            semantic_plan_id=semantic_plan_id,
            baseline=specs[0],
            candidates=specs[1:],
            selected_spec_id=None if selected is None else selected.spec_id,
            selection_status=(
                "memory_recipe_not_materialized"
                if selected is None
                else (
                    "selected_memory_baseline"
                    if selected is specs[0]
                    else "selected_memory_recipe"
                )
            ),
            partition_stage="graph_memory_complete_recipe",
            partitions_complete=True,
            partition_combination_count=len(specs),
        )
    if control_recipe_ids:
        if control_recipe_ids not in _CUDA_STRUCTURED_CONTROL_RECIPE_DOMAINS:
            raise ValueError("structured-control recipe domain is unsupported")
        baseline = _make_spec(
            semantic_plan_id,
            backend,
            (),
            control_recipe_ids[0],
        )
        candidates = tuple(
            _make_spec(semantic_plan_id, backend, (), recipe_id)
            for recipe_id in control_recipe_ids[1:]
        )
        specs = (baseline, *candidates)
        if fusion_plan.applied_groups:
            selected_spec_id = None
            selection_status = "control_recipe_requires_unfused_source"
        else:
            selected = next(
                (
                    spec
                    for spec in specs
                    if spec.control_recipe_id == selected_control_recipe_id
                ),
                None,
            )
            selected_spec_id = None if selected is None else selected.spec_id
            selection_status = (
                "control_recipe_not_materialized"
                if selected is None
                else (
                    "selected_control_baseline"
                    if selected is baseline
                    else "selected_control_recipe"
                )
            )
        return _ExecutableOptimizationSpace(
            semantic_plan_id=semantic_plan_id,
            baseline=baseline,
            candidates=candidates,
            selected_spec_id=selected_spec_id,
            selection_status=selection_status,
            partition_stage="structured_control",
            partitions_complete=True,
            partition_combination_count=len(specs),
        )

    baseline = _make_spec(semantic_plan_id, backend, ())
    candidate_recipe_sets = []
    for partition in fusion_plan.candidate_partitions:
        partition = tuple(partition)
        if partition and partition not in candidate_recipe_sets:
            candidate_recipe_sets.append(partition)
    applied_recipe_ids = tuple(fusion_plan.applied_recipe_ids)
    if applied_recipe_ids and applied_recipe_ids not in candidate_recipe_sets:
        candidate_recipe_sets.append(applied_recipe_ids)
    candidates = tuple(
        _make_spec(
            semantic_plan_id,
            backend,
            candidate,
            fusion_source_groups=_fusion_source_groups(fusion_plan, candidate),
        )
        for candidate in candidate_recipe_sets
    )
    if fusion_plan.applied_groups == 0:
        selected_spec_id = baseline.spec_id
        selection_status = "selected_baseline"
    elif (
        applied_recipe_ids
        and fusion_plan.applied_groups == len(applied_recipe_ids)
        and fusion_plan.unmatched_applied_groups == 0
    ):
        selected_spec_id = next(
            spec.spec_id
            for spec in candidates
            if spec.fusion_recipe_ids == applied_recipe_ids
        )
        selection_status = "selected_map_recipe"
    else:
        selected_spec_id = None
        selection_status = "applied_group_count_mismatch"
    return _ExecutableOptimizationSpace(
        semantic_plan_id=semantic_plan_id,
        baseline=baseline,
        candidates=candidates,
        selected_spec_id=selected_spec_id,
        selection_status=selection_status,
        partition_stage=fusion_plan.partition_stage,
        partitions_complete=fusion_plan.partitions_complete,
        partition_combination_count=(fusion_plan.partition_combination_count),
        partition_candidate_limit=fusion_plan.partition_candidate_limit,
    )


__all__ = [
    "_CUDA_CONDITIONAL_CONTROL_RECIPE_ID",
    "_CUDA_CONTROL_RECIPE_IDS",
    "_CUDA_MASKED_CONTROL_RECIPE_ID",
    "_CUDA_NESTED_CONTROL_RECIPE_IDS",
    "_CUDA_NESTED_DEVICE_UPDATE_CONTROL_RECIPE_ID",
    "_CUDA_NESTED_MASKED_CONTROL_RECIPE_ID",
    "_CUDA_STRUCTURED_CONTROL_RECIPE_DOMAINS",
    "_CUDA_STRUCTURED_CONTROL_RECIPE_IDS",
    "_GRAPH_FUSION_QUALIFICATION_SCHEMA",
    "_GraphBoundedExecutionRecipeManifest",
    "_GraphMemoryRecipeManifest",
    "_GraphReductionRecipeManifest",
    "_INTERNAL_STRUCTURED_CONTROL_ENV",
    "_ExecutableOptimizationSpace",
    "_ExecutableOptimizationSpec",
    "_GraphFusionBindingScope",
    "_GraphFusionQualificationCache",
    "_GraphFusionQualificationEntry",
    "_build_executable_optimization_space",
]
