"""Immutable semantic input for Forge-owned complete Graph recipes.

This module deliberately owns only the frozen definition and its baseline
recipe. Fragment composition and transactional materialization are layered on
top in later stages; the existing Graph runtime remains the baseline executor.
"""

import hashlib
import json
from dataclasses import dataclass, field

from taichi_forge.graph._ir import graph_ir_to_dict

_GRAPH_DEFINITION_SCHEMA = "taichi_forge.graph_definition.v1"
_BASELINE_RECIPE_SCHEMA = "taichi_forge.graph_baseline_recipe.v1"
_PROVENANCE_SCHEMA = "taichi_forge.graph_compile_provenance.v1"


def _canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _digest(value):
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stable_execution_definition(spec):
    """Keep definition-time execution facts and omit runtime generations."""

    execution = spec.execution_definition
    return {
        "nodes": tuple(
            {
                "kind": node["kind"],
                "dispatch_count": int(node["dispatch_count"]),
                "physical_dispatch_count": int(node["physical_dispatch_count"]),
                "runtime_arg_count": int(node["runtime_arg_count"]),
                "region_kind": node["region_kind"],
                "source_native_count": int(node["source_native_count"]),
            }
            for node in execution["nodes"]
        ),
        "dispatch_count": int(execution["dispatch_count"]),
        "native_count": int(execution["native_count"]),
        "observation_count": int(execution["observation_count"]),
        "structured_control_count": int(execution["structured_control_count"]),
        "max_structured_depth": int(execution["max_structured_depth"]),
        "runtime_arg_count": int(execution["runtime_arg_count"]),
        "fixed_runtime_arg_count": int(execution["fixed_runtime_arg_count"]),
        "internal_storage_bytes": int(execution["internal_storage_bytes"]),
        "temporary_memory_plan": execution["temporary_memory_plan"],
    }


@dataclass(frozen=True)
class GraphSemanticRegion:
    """One stable, address-independent node in a frozen semantic Graph."""

    region_id: str
    path: str
    parent_region_id: str | None
    child_index: int
    kind: str
    name: str
    semantic_digest: str

    def to_dict(self):
        return {
            "region_id": self.region_id,
            "path": self.path,
            "parent_region_id": self.parent_region_id,
            "child_index": self.child_index,
            "kind": self.kind,
            "name": self.name,
            "semantic_digest": self.semantic_digest,
        }


@dataclass(frozen=True)
class GraphDefinitionSource:
    """Stable kernel/native/observation source registered by semantic region."""

    source_id: str
    region_id: str
    path: str
    kind: str
    name: str
    semantic_identity: str

    def to_dict(self):
        return {
            "source_id": self.source_id,
            "region_id": self.region_id,
            "path": self.path,
            "kind": self.kind,
            "name": self.name,
            "semantic_identity": self.semantic_identity,
        }


@dataclass(frozen=True)
class GraphBindingABIEntry:
    """Canonical symbolic binding facts visible to recipe providers."""

    name: str
    kinds: tuple[str, ...]
    required: bool
    scope: str
    region_ids: tuple[str, ...]

    def to_dict(self):
        return {
            "name": self.name,
            "kinds": self.kinds,
            "required": self.required,
            "scope": self.scope,
            "region_ids": self.region_ids,
        }


@dataclass(frozen=True)
class GraphCompileProvenance:
    """Build provenance excluded from cross-build semantic identity."""

    core_commit: str

    def to_dict(self):
        return {
            "schema": _PROVENANCE_SCHEMA,
            "core_commit": self.core_commit,
        }


@dataclass(frozen=True)
class GraphBaselineRecipe:
    """Complete baseline coverage for one immutable GraphDefinition."""

    recipe_id: str
    semantic_graph_id: str
    planned_physical_id: str
    coverage_region_ids: tuple[str, ...]

    def to_dict(self):
        return {
            "schema": _BASELINE_RECIPE_SCHEMA,
            "recipe_id": self.recipe_id,
            "semantic_graph_id": self.semantic_graph_id,
            "planned_physical_id": self.planned_physical_id,
            "kind": "baseline",
            "coverage_region_ids": self.coverage_region_ids,
            "fragments": (),
        }


def _semantic_inventory(root):
    regions = []
    sources = []
    region_by_path = {}

    def visit(node, path, parent_region_id, child_index):
        semantic = graph_ir_to_dict(node)
        semantic_digest = _digest(semantic)
        identity_payload = {
            "path": path,
            "kind": node.kind,
            "semantic_digest": semantic_digest,
        }
        region_id = f"graph-region:{_digest(identity_payload)[:32]}"
        regions.append(
            GraphSemanticRegion(
                region_id=region_id,
                path=path,
                parent_region_id=parent_region_id,
                child_index=child_index,
                kind=node.kind,
                name=node.name,
                semantic_digest=semantic_digest,
            )
        )
        region_by_path[path] = region_id
        if node.kind in ("dispatch", "native_call", "observation"):
            semantic_identity = getattr(node, "logical_kernel_identity", "")
            if not semantic_identity:
                semantic_identity = semantic_digest
            source_payload = {
                "region_id": region_id,
                "path": path,
                "kind": node.kind,
                "semantic_identity": semantic_identity,
            }
            sources.append(
                GraphDefinitionSource(
                    source_id=f"graph-source:{_digest(source_payload)[:32]}",
                    region_id=region_id,
                    path=path,
                    kind=node.kind,
                    name=node.name,
                    semantic_identity=semantic_identity,
                )
            )
        for index, child in enumerate(node.children):
            visit(
                child,
                f"{path}/{index}:{child.kind}",
                region_id,
                index,
            )

    visit(root, "graph", None, 0)
    return tuple(regions), tuple(sources), region_by_path


def _binding_abi(root, spec, region_by_path):
    bindings = {}
    public = frozenset(spec.runtime_arg_names)

    def record(name, kind, required, region_id):
        if name not in public:
            return
        item = bindings.setdefault(
            name,
            {"kinds": set(), "required": False, "region_ids": set()},
        )
        item["kinds"].add(kind)
        item["required"] = item["required"] or bool(required)
        item["region_ids"].add(region_id)

    def visit(node, path):
        region_id = region_by_path[path]
        for binding in node.bindings:
            record(binding.name, binding.kind, binding.required, region_id)
        for index, child in enumerate(node.children):
            visit(child, f"{path}/{index}:{child.kind}")

    visit(root, "graph")
    for name in sorted(public):
        bindings.setdefault(
            name,
            {"kinds": {"opaque"}, "required": True, "region_ids": set()},
        )

    result = []
    for name, item in sorted(bindings.items()):
        result.append(
            GraphBindingABIEntry(
                name=name,
                kinds=tuple(sorted(item["kinds"])),
                required=bool(item["required"]),
                scope="public",
                region_ids=tuple(sorted(item["region_ids"])),
            )
        )
    return tuple(result)


@dataclass(frozen=True)
class GraphDefinition:
    """Frozen semantic Graph and the complete baseline recipe.

    ``_runtime_spec`` is an opaque baseline materialization owner. It is
    intentionally excluded from equality, hashing, and serialization: stable
    identities come only from canonical Graph facts, never Python addresses.
    """

    semantic_graph_id: str
    backend: str
    regions: tuple[GraphSemanticRegion, ...]
    sources: tuple[GraphDefinitionSource, ...]
    binding_abi: tuple[GraphBindingABIEntry, ...]
    compile_provenance: GraphCompileProvenance
    baseline_recipe: GraphBaselineRecipe
    _semantic_payload_json: str = field(repr=False)
    _planned_payload_json: str = field(repr=False)
    _runtime_spec: object = field(repr=False, compare=False, hash=False)

    @classmethod
    def _from_graph_spec(cls, spec, backend, *, core_commit=""):
        semantic_root = getattr(
            spec,
            "definition_semantic_root",
            spec.pre_optimization_ir_root,
        )
        regions, sources, region_by_path = _semantic_inventory(semantic_root)
        binding_abi = _binding_abi(semantic_root, spec, region_by_path)
        provider_sources = getattr(spec, "definition_semantic_sources", ())
        semantic_payload = {
            "schema": _GRAPH_DEFINITION_SCHEMA,
            "root": graph_ir_to_dict(semantic_root),
            "binding_abi": tuple(item.to_dict() for item in binding_abi),
            "provider_sources": provider_sources,
        }
        semantic_payload_json = _canonical_json(semantic_payload)
        semantic_digest = hashlib.sha256(
            semantic_payload_json.encode("utf-8")
        ).hexdigest()
        semantic_graph_id = f"semantic-graph:{semantic_digest}"

        planned_payload = {
            "schema": "taichi_forge.graph_planned_physical.v1",
            "semantic_graph_id": semantic_graph_id,
            "backend": backend,
            "root": graph_ir_to_dict(spec.ir_root),
            "execution": _stable_execution_definition(spec),
        }
        planned_payload_json = _canonical_json(planned_payload)
        planned_digest = hashlib.sha256(
            planned_payload_json.encode("utf-8")
        ).hexdigest()
        planned_physical_id = f"planned-physical:{planned_digest}"
        baseline_recipe_payload = {
            "semantic_graph_id": semantic_graph_id,
            "planned_physical_id": planned_physical_id,
            "coverage_region_ids": tuple(region.region_id for region in regions),
            "kind": "baseline",
        }
        baseline_recipe = GraphBaselineRecipe(
            recipe_id=f"graph-recipe:{_digest(baseline_recipe_payload)}",
            semantic_graph_id=semantic_graph_id,
            planned_physical_id=planned_physical_id,
            coverage_region_ids=baseline_recipe_payload["coverage_region_ids"],
        )
        return cls(
            semantic_graph_id=semantic_graph_id,
            backend=backend,
            regions=regions,
            sources=sources,
            binding_abi=binding_abi,
            compile_provenance=GraphCompileProvenance(core_commit=core_commit),
            baseline_recipe=baseline_recipe,
            _semantic_payload_json=semantic_payload_json,
            _planned_payload_json=planned_payload_json,
            _runtime_spec=spec,
        )

    @property
    def semantic_root(self):
        return json.loads(self._semantic_payload_json)["root"]

    @property
    def planned_physical_manifest(self):
        return json.loads(self._planned_payload_json)

    def region(self, region_id):
        for region in self.regions:
            if region.region_id == region_id:
                return region
        raise KeyError(f"unknown Graph semantic region {region_id!r}")

    def compile(self, *, workspace_lanes=1, workspace_saturation="wait"):
        """Materialize the frozen baseline through the existing Graph runtime."""

        from taichi_forge.graph._graph import Graph

        return Graph(
            self._runtime_spec,
            workspace_lanes=workspace_lanes,
            workspace_saturation=workspace_saturation,
            definition=self,
        )

    def materialization_context(self, **options):
        """Create an explicit owner for transactional recipe materialization."""

        from taichi_forge.graph._recipes.families import (
            assemble_existing_family_recipe,
            materialize_existing_family_baseline,
        )

        from taichi_forge.graph._recipes.materialize import (
            GraphMaterializationContext,
        )

        options.setdefault("assembler", assemble_existing_family_recipe)
        options.setdefault(
            "baseline_materializer",
            materialize_existing_family_baseline,
        )

        return GraphMaterializationContext(self, **options)

    def recipe_catalog(self, *, available_capabilities=(), providers=None):
        """Discover established optimization families in one staged catalog."""

        from taichi_forge.graph._recipes.catalog import GraphRecipeCatalog
        from taichi_forge.graph._recipes.families import GraphExistingFamilyProvider

        if providers is None:
            providers = (GraphExistingFamilyProvider(), )
        catalog = GraphRecipeCatalog(
            self,
            available_capabilities=available_capabilities,
        )
        catalog.discover(tuple(providers))
        catalog.build_single_region_stage()
        return catalog

    def search_recipes(self, *, engine="compileiq", target=None, budget):
        """Create a complete-Graph optimization session.

        Candidate construction and materialization remain Forge-owned.  The
        engine schedules opaque recipe identities and returns measurements.
        """

        from taichi_forge.graph._optimization_api import _GraphRecipeSearchSession

        return _GraphRecipeSearchSession(
            self,
            engine=engine,
            target=target,
            budget=budget,
        )

    def materialize(self, recipe=None, *, context=None, **context_options):
        """Transactionally materialize a complete recipe.

        Omitting ``recipe`` preserves the baseline sentinel. Supplying a
        context makes ownership and physical de-duplication explicit across a
        search batch; a one-shot call keeps its private context alive through
        the returned handle.
        """

        from taichi_forge.graph._optimization_api import GraphRecipeHandle

        if isinstance(recipe, GraphRecipeHandle):
            if recipe.semantic_graph_id != self.semantic_graph_id:
                raise ValueError(
                    "Graph recipe handle belongs to a different GraphDefinition"
                )
            recipe = recipe._recipe
        if context is not None and context_options:
            raise TypeError(
                "GraphDefinition.materialize context options require a new context"
            )
        if context is not None and context.definition is not self:
            raise ValueError(
                "GraphDefinition.materialize context belongs to another definition"
            )
        owns_context = context is None
        context = context or self.materialization_context(**context_options)
        try:
            result = context.materialize(recipe)
        except BaseException:
            if owns_context:
                context.close()
            raise
        if owns_context:
            result._close_context_on_release = True
        return result

    def to_dict(self):
        return {
            "schema": _GRAPH_DEFINITION_SCHEMA,
            "semantic_graph_id": self.semantic_graph_id,
            "backend": self.backend,
            "compile_provenance": self.compile_provenance.to_dict(),
            "semantic_root": self.semantic_root,
            "regions": tuple(region.to_dict() for region in self.regions),
            "sources": tuple(source.to_dict() for source in self.sources),
            "binding_abi": tuple(item.to_dict() for item in self.binding_abi),
            "provider_sources": json.loads(self._semantic_payload_json)[
                "provider_sources"
            ],
            "baseline_recipe": self.baseline_recipe.to_dict(),
            "planned_physical_manifest": self.planned_physical_manifest,
        }
