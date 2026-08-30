"""Graph recipe transport for the reviewed modified CompileIQ fork."""

from __future__ import annotations

from importlib import import_module
from types import MappingProxyType

from taichi_forge._compileiq_opaque import (
    CompileIQOpaqueUnavailableError,
    _CompileIQOpaqueRecipeTransport,
    _EXPECTED_CAPABILITY_ID as _EXPECTED_CAPABILITY_ID,
    _EXPECTED_CORE_COMMIT as _EXPECTED_CORE_COMMIT,
    _EXPECTED_CORE_LOCK as _EXPECTED_CORE_LOCK,
    _EXPECTED_PYTHON_SOURCE_LOCK as _EXPECTED_PYTHON_SOURCE_LOCK,
    _identity,
    _validated_compileiq_capability as _validate_shared_compileiq_capability,
)
from taichi_forge.graph._compileiq_adapter import _CompileIQExecutableAdapter


class CompileIQGraphUnavailableError(CompileIQOpaqueUnavailableError):
    """The installed CompileIQ is not the reviewed Graph recipe fork."""


def _validated_compileiq_capability():
    """Keep Graph's optional import boundary independently monkeypatchable."""

    return _validate_shared_compileiq_capability(
        importer=import_module,
        error_type=CompileIQGraphUnavailableError,
    )


class CompileIQGraphRecipeSearch:
    """One frozen Graph executable-recipe domain for modified CompileIQ.

    Forge owns the recipe table, materialization checks, qualification, and
    admission boundary. CompileIQ sees only fixed ordinal tokens and returns a
    decoded recipe ID that Forge validates again through :meth:`select`.
    """

    __slots__ = (
        "_adapter",
        "_semantic_fingerprint",
        "_transport",
    )

    def __init__(self, graph):
        capability_components = _validated_compileiq_capability()
        adapter = _CompileIQExecutableAdapter.from_graph(graph)
        recipe_ids = adapter.spec_ids()
        if not recipe_ids or adapter.baseline_spec_id not in recipe_ids:
            raise ValueError("Graph recipe domain must contain its executable baseline")

        adapter_manifest = adapter.manifest()
        semantic_payload = {
            "schema": (
                "taichi_forge.graph.compileiq-structured-control-semantics.v1"
                if adapter.recipe_kind == "structured_control"
                else "taichi_forge.graph.compileiq-semantics.v1"
            ),
            "semantic_plan_id": adapter.semantic_plan_id,
            "backend": adapter.backend,
            "baseline_spec_id": adapter.baseline_spec_id,
            "specs": adapter_manifest["specs"],
        }
        if adapter.recipe_kind == "structured_control":
            semantic_payload["recipe_kind"] = adapter.recipe_kind
        semantic_fingerprint = _identity("forge-graph-semantics-v1:", semantic_payload)
        transport = _CompileIQOpaqueRecipeTransport(
            provider_namespace=(
                "taichi_forge.graph.structured_control"
                if adapter.recipe_kind == "structured_control"
                else "taichi_forge.graph.map_fusion"
            ),
            domain_version=(
                "structured-control-executable-spec.v1"
                if adapter.recipe_kind == "structured_control"
                else "executable-spec.v1"
            ),
            provider_semantic_fingerprint=semantic_fingerprint,
            recipe_ids=recipe_ids,
            baseline_recipe_id=adapter.baseline_spec_id,
            capability_components=capability_components,
            domain_owner="Graph",
            recipe_description="Graph recipe",
        )

        self._adapter = adapter
        self._semantic_fingerprint = semantic_fingerprint
        self._transport = transport

    @property
    def capability(self):
        return self._transport.capability

    @property
    def search_space(self):
        """Return the exact modified-fork ``OpaqueRecipeDomainV1`` object."""

        return self._transport.search_space

    @property
    def worker_type(self):
        return self._transport.worker_type

    @property
    def python_source_lock(self):
        return self._transport.python_source_lock

    @property
    def domain_fingerprint(self):
        return self._transport.domain_fingerprint

    @property
    def recipe_ids(self):
        return self._transport.recipe_ids

    @property
    def baseline_recipe_id(self):
        return self._adapter.baseline_spec_id

    @property
    def semantic_plan_id(self):
        return self._adapter.semantic_plan_id

    @property
    def backend(self):
        return self._adapter.backend

    def _decoded_recipe_id(self, parameters):
        return self._transport.decode(parameters)

    def select(self, parameters):
        recipe_id = self._decoded_recipe_id(parameters)
        return self._adapter.select({self._adapter.parameter: recipe_id})

    def search_coverage(self, compileiq_search):
        """Validate exact-fork audit records and report recipe coverage."""

        return self._transport.search_coverage(compileiq_search)

    def require_complete_search(self, compileiq_search):
        return self._transport.require_complete_search(compileiq_search)

    def select_best_result(self, compileiq_search, result):
        """Select a winner only after complete exact-fork coverage is proven."""

        recipe_id = self._transport.select_best_recipe_id(compileiq_search, result)
        return self._adapter.select({self._adapter.parameter: recipe_id})

    def recipe_manifest(self, recipe_id):
        selection = self._adapter.select({self._adapter.parameter: recipe_id})
        return MappingProxyType(
            {
                **selection.to_dict(),
                "is_baseline": recipe_id == self.baseline_recipe_id,
            }
        )

    def worker_environment(self, parameters):
        """Return the explicit environment overlay for Graph reconstruction."""

        return self.select(parameters).worker_environment

    def verify_materialized_graph(self, parameters, graph):
        recipe_id = self._decoded_recipe_id(parameters)
        return self._adapter.verify_materialized_graph(
            {self._adapter.parameter: recipe_id}, graph
        )

    def paired_schedule(self, *, blocks=2):
        return self._adapter.paired_schedule(blocks=blocks)

    def rank_paired(self, measurements, *, blocks=2):
        return self._adapter.rank_paired(measurements, blocks=blocks)

    def final_candidate(self, recipe_id, provider_candidate_id="baseline"):
        return self._adapter.final_candidate(recipe_id, provider_candidate_id)

    def qualification_stage(self, finalists, *, blocks=10):
        return self._adapter.qualification_stage(finalists, blocks=blocks)

    def qualify(self, measurements, finalists, **scope):
        return self._adapter.qualify(measurements, finalists, **scope)

    def qualification_cache(self, decision, **scope):
        return self._adapter.qualification_cache(decision, **scope)

    def qualification_cache_json(self, decision, **scope):
        return self._adapter.qualification_cache_json(decision, **scope)

    def manifest(self):
        value = {
            "schema": "taichi_forge.graph.compileiq-recipe-search.v1",
            **self._transport.manifest(),
            "semantic_plan_id": self.semantic_plan_id,
            "backend": self.backend,
            "recipes": tuple(
                dict(self.recipe_manifest(recipe_id)) for recipe_id in self.recipe_ids
            ),
            "qualification": "independent_forge_worst_positive_v1",
            "runtime_admission": (
                "offline_explicit_reconstruction_only"
                if self._adapter.recipe_kind == "structured_control"
                else "explicit_qualified_cache_only"
            ),
        }
        if self._adapter.recipe_kind == "structured_control":
            value["recipe_kind"] = self._adapter.recipe_kind
        return value


def compileiq_recipe_search(graph):
    """Build a baseline-inclusive Graph domain for the exact modified fork."""

    return CompileIQGraphRecipeSearch(graph)


__all__ = [
    "CompileIQGraphRecipeSearch",
    "CompileIQGraphUnavailableError",
    "compileiq_recipe_search",
]
