"""Graph recipe transport for the reviewed modified CompileIQ fork."""

from __future__ import annotations

from importlib import import_module
import math
from types import MappingProxyType

from typing import ClassVar

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
from taichi_forge.graph._optimization import (
    _ExecutableOptimizationSpace,
    _make_spec,
)


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
        "_capability_components",
        "_semantic_fingerprint",
        "_transport",
    )

    def __init__(self, graph):
        capability_components = _validated_compileiq_capability()
        adapter = _CompileIQExecutableAdapter.from_graph(graph)
        self._initialize(adapter, capability_components)

    def _initialize(self, adapter, capability_components):
        recipe_ids = adapter.spec_ids()
        if not recipe_ids or adapter.baseline_spec_id not in recipe_ids:
            raise ValueError("Graph recipe domain must contain its executable baseline")

        adapter_manifest = adapter.manifest()
        nested_structured_control = (
            adapter.structured_control_domain == "cuda_nested_while_while"
        )
        if nested_structured_control:
            semantic_schema = (
                "taichi_forge.graph.compileiq-nested-structured-control-semantics.v1"
            )
            provider_namespace = "taichi_forge.graph.nested_structured_control"
            domain_version = "nested-structured-control-executable-spec.v1"
        elif adapter.recipe_kind == "graph_memory":
            semantic_schema = "taichi_forge.graph.compileiq-graph-memory-semantics.v1"
            provider_namespace = "taichi_forge.graph.memory"
            domain_version = "graph-memory-complete-recipe"
        elif adapter.recipe_kind == "graph_bounded_execution":
            semantic_schema = (
                "taichi_forge.graph.compileiq-bounded-execution-semantics.v1"
            )
            provider_namespace = "taichi_forge.graph.bounded_execution"
            domain_version = "graph-bounded-complete-recipe.v1"
        elif adapter.recipe_kind == "graph_reduction":
            semantic_schema = "taichi_forge.graph.compileiq-reduction-semantics.v1"
            provider_namespace = "taichi_forge.graph.reduction"
            domain_version = "graph-reduction-complete-recipe.v1"
        elif adapter.recipe_kind == "graph_native_algorithm":
            semantic_schema = (
                "taichi_forge.graph.compileiq-native-algorithm-semantics.v1"
            )
            provider_namespace = "taichi_forge.graph.native_algorithm"
            domain_version = "graph-native-algorithm-complete-recipe.v1"
        elif adapter.recipe_kind == "structured_control":
            semantic_schema = (
                "taichi_forge.graph.compileiq-structured-control-semantics.v1"
            )
            provider_namespace = "taichi_forge.graph.structured_control"
            domain_version = "structured-control-executable-spec.v1"
        else:
            semantic_schema = "taichi_forge.graph.compileiq-partition-semantics.v2"
            provider_namespace = "taichi_forge.graph.map_fusion"
            domain_version = "graph-partition-plan.v2"
        semantic_payload = {
            "schema": semantic_schema,
            "semantic_plan_id": adapter.semantic_plan_id,
            "backend": adapter.backend,
            "baseline_spec_id": adapter.baseline_spec_id,
            "specs": adapter_manifest["specs"],
        }
        if adapter.recipe_kind == "map_fusion":
            semantic_payload.update(
                {
                    "partition_stage": adapter_manifest["partition_stage"],
                    "partitions_complete": adapter_manifest["partitions_complete"],
                    "partition_combination_count": adapter_manifest[
                        "partition_combination_count"
                    ],
                    "partition_candidate_limit": adapter_manifest[
                        "partition_candidate_limit"
                    ],
                    "partition_parent_domain_fingerprint": adapter_manifest[
                        "partition_parent_domain_fingerprint"
                    ],
                    "partition_frontier_spec_ids": adapter_manifest[
                        "partition_frontier_spec_ids"
                    ],
                }
            )
        if adapter.recipe_kind in (
            "structured_control",
            "graph_memory",
            "graph_bounded_execution",
            "graph_reduction",
            "graph_native_algorithm",
        ):
            semantic_payload["recipe_kind"] = adapter.recipe_kind
        if nested_structured_control:
            semantic_payload["structured_control_domain"] = (
                adapter.structured_control_domain
            )
        semantic_fingerprint = _identity("forge-graph-semantics-v1:", semantic_payload)
        transport = _CompileIQOpaqueRecipeTransport(
            provider_namespace=provider_namespace,
            domain_version=domain_version,
            provider_semantic_fingerprint=semantic_fingerprint,
            recipe_ids=recipe_ids,
            baseline_recipe_id=adapter.baseline_spec_id,
            capability_components=capability_components,
            domain_owner="Graph",
            recipe_description="Graph recipe",
        )

        self._adapter = adapter
        self._capability_components = capability_components
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

    def compileiq_search(
        self,
        objective_function,
        *,
        problem_type="min",
        target_contract=None,
    ):
        """Create the reviewed fork's complete bounded opaque search."""

        return self._transport.exhaustive_search(
            objective_function,
            problem_type=problem_type,
            target_contract=target_contract,
        )

    def refine(self, compileiq_search, frontier_recipe_ids):
        """Build the next exact-partition stage from an observed frontier."""

        self.require_complete_search(compileiq_search)
        manifest = self._adapter.manifest()
        if self._adapter.recipe_kind != "map_fusion":
            raise RuntimeError("complete Graph recipe domains cannot be refined")
        if manifest["partitions_complete"]:
            raise RuntimeError("the Graph partition domain is already complete")
        frontier_recipe_ids = tuple(dict.fromkeys(frontier_recipe_ids))
        frontier_recipe_ids = tuple(
            recipe_id
            for recipe_id in frontier_recipe_ids
            if recipe_id != self.baseline_recipe_id
        )
        if not frontier_recipe_ids:
            raise ValueError("partition refinement requires a nonbaseline frontier")
        if len(frontier_recipe_ids) > 32:
            raise ValueError("partition refinement frontier exceeds 32 recipes")
        known = set(self.recipe_ids)
        unknown = tuple(
            recipe_id for recipe_id in frontier_recipe_ids if recipe_id not in known
        )
        if unknown:
            raise KeyError(f"partition refinement contains unknown recipes {unknown}")

        source_space = self._adapter._space
        source_specs = {
            spec.spec_id: spec
            for spec in (source_space.baseline, *source_space.candidates)
        }
        frontier = tuple(source_specs[recipe_id] for recipe_id in frontier_recipe_ids)
        generated = {spec.spec_id: spec for spec in frontier}
        for left_index, left in enumerate(frontier):
            for right in frontier[left_index + 1 :]:
                left_dispatches = {
                    item for group in left.fusion_source_groups for item in group
                }
                right_dispatches = {
                    item for group in right.fusion_source_groups for item in group
                }
                if left_dispatches.intersection(right_dispatches):
                    continue
                paired = sorted(
                    (
                        *zip(left.fusion_recipe_ids, left.fusion_source_groups),
                        *zip(right.fusion_recipe_ids, right.fusion_source_groups),
                    ),
                    key=lambda item: item[1][0],
                )
                spec = _make_spec(
                    source_space.semantic_plan_id,
                    source_space.baseline.backend,
                    tuple(item[0] for item in paired),
                    fusion_source_groups=tuple(item[1] for item in paired),
                )
                generated.setdefault(spec.spec_id, spec)
        if len(generated) + 1 > source_space.partition_candidate_limit + 1:
            raise ValueError(
                "observed-frontier partition domain exceeds its explicit limit"
            )
        refined_space = _ExecutableOptimizationSpace(
            semantic_plan_id=source_space.semantic_plan_id,
            baseline=source_space.baseline,
            candidates=tuple(generated[spec_id] for spec_id in sorted(generated)),
            selected_spec_id=source_space.baseline.spec_id,
            selection_status="selected_baseline",
            partition_stage="observed_frontier_pairwise_v1",
            partitions_complete=False,
            partition_combination_count=(source_space.partition_combination_count),
            partition_candidate_limit=source_space.partition_candidate_limit,
            partition_parent_domain_fingerprint=self.domain_fingerprint,
            partition_frontier_spec_ids=tuple(sorted(frontier_recipe_ids)),
        )
        refined = object.__new__(type(self))
        refined._initialize(
            _CompileIQExecutableAdapter(refined_space),
            self._capability_components,
        )
        return refined

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
            "schema": "taichi_forge.graph.compileiq-recipe-search.v2",
            **self._transport.manifest(),
            "semantic_plan_id": self.semantic_plan_id,
            "backend": self.backend,
            "recipes": tuple(
                dict(self.recipe_manifest(recipe_id)) for recipe_id in self.recipe_ids
            ),
            "qualification": "independent_forge_worst_positive_v1",
            "runtime_admission": (
                "offline_explicit_reconstruction_only"
                if self._adapter.recipe_kind
                in (
                    "structured_control",
                    "graph_memory",
                    "graph_bounded_execution",
                    "graph_reduction",
                    "graph_native_algorithm",
                )
                else "explicit_qualified_cache_only"
            ),
        }
        if self._adapter.recipe_kind == "map_fusion":
            adapter_manifest = self._adapter.manifest()
            value.update(
                {
                    "partition_stage": adapter_manifest["partition_stage"],
                    "partitions_complete": adapter_manifest["partitions_complete"],
                    "partition_combination_count": adapter_manifest[
                        "partition_combination_count"
                    ],
                    "partition_candidate_limit": adapter_manifest[
                        "partition_candidate_limit"
                    ],
                    "partition_parent_domain_fingerprint": adapter_manifest[
                        "partition_parent_domain_fingerprint"
                    ],
                    "partition_frontier_spec_ids": adapter_manifest[
                        "partition_frontier_spec_ids"
                    ],
                }
            )
        if self._adapter.recipe_kind in (
            "structured_control",
            "graph_memory",
            "graph_bounded_execution",
            "graph_reduction",
            "graph_native_algorithm",
        ):
            value["recipe_kind"] = self._adapter.recipe_kind
        if self._adapter.structured_control_domain == "cuda_nested_while_while":
            value["structured_control_domain"] = self._adapter.structured_control_domain
        return value


class CompileIQCompleteGraphRecipeSearch:
    """V2 search façade over the Forge-owned complete recipe catalog.

    The transport remains compatible with the reviewed modified CompileIQ fork,
    but it neither rebuilds Graphs from environment variables nor owns a second
    family materializer. All selected recipes cross GraphMaterializationContext.
    """

    _FAMILY_CONTRACTS: ClassVar[dict[str, tuple[str, str]]] = {
        "map_fusion": (
            "taichi_forge.graph.map_fusion",
            "graph-partition-plan.v3",
        ),
        "graph_memory": (
            "taichi_forge.graph.memory",
            "graph-memory-complete-recipe.v2",
        ),
        "bounded_execution": (
            "taichi_forge.graph.bounded_execution",
            "graph-bounded-complete-recipe.v2",
        ),
        "structured_control": (
            "taichi_forge.graph.structured_control",
            "structured-control-complete-recipe.v2",
        ),
        "graph_reduction": (
            "taichi_forge.graph.reduction",
            "graph-reduction-complete-recipe.v2",
        ),
        "native_algorithm": (
            "taichi_forge.graph.native_algorithm",
            "graph-native-algorithm-complete-recipe.v2",
        ),
        "branch_join_schedule": (
            "taichi_forge.graph.branch_join_schedule",
            "cuda-branch-join-complete-recipe.v1",
        ),
    }

    def __init__(self, graph, *, catalog=None):
        capability_components = _validated_compileiq_capability()
        definition = getattr(graph, "definition", None)
        if definition is None:
            raise TypeError("complete Graph recipe search requires a GraphDefinition")
        if catalog is None:
            catalog = definition.recipe_catalog()
        elif catalog.definition is not definition:
            raise ValueError(
                "complete Graph recipe catalog belongs to another definition"
            )
        entries = catalog.entries()
        families = tuple(
            dict.fromkeys(
                fragment.materializer.selection.family for fragment in catalog.fragments
            )
        )
        family = families[0] if len(families) == 1 else ""
        provider_namespace, domain_version = self._FAMILY_CONTRACTS.get(
            family,
            (
                "taichi_forge.graph.complete_recipe",
                "complete-graph-recipe.v1",
            ),
        )
        semantic_payload = {
            "schema": "taichi_forge.graph.complete-recipe-semantics.v1",
            "definition": definition.to_dict(),
            "recipes": tuple(entry.recipe.to_dict() for entry in entries),
        }
        semantic_fingerprint = _identity(
            "forge-complete-graph-semantics-v1:",
            semantic_payload,
        )
        self._transport = _CompileIQOpaqueRecipeTransport(
            provider_namespace=provider_namespace,
            domain_version=domain_version,
            provider_semantic_fingerprint=semantic_fingerprint,
            recipe_ids=tuple(entry.recipe.recipe_id for entry in entries),
            baseline_recipe_id=catalog.baseline.recipe.recipe_id,
            capability_components=capability_components,
            domain_owner="complete Graph",
            recipe_description="complete Graph recipe",
        )
        self._graph = graph
        self._definition = definition
        self._catalog = catalog
        self._family = family
        self._families = families
        self._workspace_lanes = graph._workspace_lane_capacity
        self._workspace_saturation = graph._workspace_saturation

    @property
    def capability(self):
        return self._transport.capability

    @property
    def search_space(self):
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
        return self._catalog.baseline.recipe.recipe_id

    @property
    def semantic_plan_id(self):
        return self._definition.semantic_graph_id

    @property
    def backend(self):
        return self._definition.backend

    def _decoded_recipe_id(self, parameters):
        return self._transport.decode(parameters)

    def _source_manifest(self, family, choice_id):
        spec = self._definition._runtime_spec
        if family == "graph_memory":
            sources = spec._graph_memory_sources
        elif family == "graph_reduction":
            sources = spec._graph_reduction_sources
        elif family == "native_algorithm":
            sources = spec._graph_native_algorithm_sources
        elif family == "bounded_execution":
            from taichi_forge.graph._graph import _graph_bounded_recipe_scope

            sources, _, _ = _graph_bounded_recipe_scope(spec.pipeline_definition)
            for manifest in sources:
                if manifest.recipe_id == choice_id:
                    return manifest
            return None
        elif family == "structured_control":
            return type(
                "_ControlRecipeIdentity",
                (),
                {"recipe_id": choice_id},
            )()
        else:
            return None
        for source in sources:
            for manifest in source.manifests():
                if manifest.recipe_id == choice_id:
                    return manifest
        return None

    def _baseline_family_manifest(self):
        spec = self._definition._runtime_spec
        family = self._family
        if family == "graph_memory" and len(spec._graph_memory_sources) == 1:
            source = spec._graph_memory_sources[0]
            return self._source_manifest(family, source.selected_recipe_id)
        if family == "graph_reduction" and len(spec._graph_reduction_sources) == 1:
            source = spec._graph_reduction_sources[0]
            return self._source_manifest(family, source.selected_recipe_id)
        if (
            family == "native_algorithm"
            and len(spec._graph_native_algorithm_sources) == 1
        ):
            source = spec._graph_native_algorithm_sources[0]
            return self._source_manifest(family, source.selected_recipe_id)
        if family == "bounded_execution":
            from taichi_forge.graph._graph import _graph_bounded_recipe_scope

            manifests, selected, _ = _graph_bounded_recipe_scope(
                spec.pipeline_definition
            )
            return next(
                (item for item in manifests if item.recipe_id == selected),
                None,
            )
        if family == "structured_control" and spec.selected_control_recipe_id:
            return type(
                "_ControlRecipeIdentity",
                (),
                {"recipe_id": spec.selected_control_recipe_id},
            )()
        return None

    def _selection(self, recipe_id):
        entry = self._catalog.entry(recipe_id)
        recipe = entry.recipe
        if not recipe.fragments:
            return _CompleteGraphRecipeSelection(
                recipe,
                family=self._family,
                source_manifest=self._baseline_family_manifest(),
            )
        if len(recipe.fragments) != 1:
            return _CompleteGraphRecipeSelection(recipe)
        family_selection = recipe.fragments[0].materializer.selection
        manifest = self._source_manifest(
            family_selection.family,
            family_selection.choice_id,
        )
        return _CompleteGraphRecipeSelection(
            recipe,
            family=family_selection.family,
            source_manifest=manifest,
            selection=family_selection,
        )

    def select(self, parameters):
        return self._selection(self._decoded_recipe_id(parameters))

    def materialize(self, parameters, *, context=None):
        """Materialize the selected complete recipe without worker overlays."""

        selection = self.select(parameters)
        if context is not None:
            return self._definition.materialize(selection.recipe, context=context)
        return self._definition.materialize(
            selection.recipe,
            workspace_lanes=self._workspace_lanes,
            workspace_saturation=self._workspace_saturation,
        )

    def batch(
        self,
        *,
        recipe_ids=None,
        stage_index=0,
        parent_batch=None,
        parent_recipe_ids=None,
        fidelity_name="full",
        fidelity_ordinal=0,
        repeat_count=1,
        work_scale=1.0,
    ):
        """Build one CompileIQ V2 batch from frozen complete recipes."""

        recipe_ids = self.recipe_ids if recipe_ids is None else tuple(recipe_ids)
        recipe_ids = tuple(sorted(recipe_ids, key=lambda item: item.encode("utf-8")))
        if parent_recipe_ids is None and parent_batch is not None:
            parent_recipe_ids = {
                recipe_id: (recipe_id,)
                for recipe_id in recipe_ids
                if recipe_id in parent_batch.recipe_ids
            }
        parent_recipe_ids = {
            recipe_id: tuple(sorted(parent_ids, key=lambda item: item.encode("utf-8")))
            for recipe_id, parent_ids in (parent_recipe_ids or {}).items()
        }
        estimates = {
            recipe_id: (
                self._catalog.entry(recipe_id).recipe.declared_persistent_resource_bytes
                + self._catalog.entry(
                    recipe_id
                ).recipe.declared_transient_resource_bytes
            )
            for recipe_id in recipe_ids
        }
        stage_fingerprint = _identity(
            "forge-complete-recipe-stage-v2:",
            {
                "semantic_graph_id": self.semantic_plan_id,
                "stage_index": stage_index,
                "parent_batch_fingerprint": (
                    None if parent_batch is None else parent_batch.batch_fingerprint
                ),
                "recipe_ids": recipe_ids,
                "parent_recipe_ids": parent_recipe_ids,
                "fidelity": {
                    "name": fidelity_name,
                    "ordinal": fidelity_ordinal,
                    "repeat_count": repeat_count,
                    "work_scale": work_scale,
                },
            },
        )
        return self._transport.batch_v2(
            recipe_ids=recipe_ids,
            stage_index=stage_index,
            stage_fingerprint=stage_fingerprint,
            parent_batch=parent_batch,
            parent_recipe_ids=parent_recipe_ids,
            fidelity_name=fidelity_name,
            fidelity_ordinal=fidelity_ordinal,
            repeat_count=repeat_count,
            work_scale=work_scale,
            estimated_materialized_bytes=estimates,
        )

    def compileiq_search(
        self,
        objective_function,
        *,
        budget,
        problem_type="min",
        target_contract=None,
        deterministic_seed=0,
        halving_factor=2,
        minimum_survivors=1,
        repeat_count=1,
        fidelity_name="full",
        checkpoint=None,
        context=None,
    ):
        """Create the Forge-owned materializing CompileIQ V2 search session."""

        return _CompleteGraphRecipeSearchSessionV2(
            self,
            objective_function,
            budget=budget,
            problem_type=problem_type,
            target_contract=target_contract,
            deterministic_seed=deterministic_seed,
            halving_factor=halving_factor,
            minimum_survivors=minimum_survivors,
            repeat_count=repeat_count,
            fidelity_name=fidelity_name,
            checkpoint=checkpoint,
            context=context,
        )

    def compileiq_search_v1(
        self,
        objective_function,
        *,
        problem_type="min",
        target_contract=None,
    ):
        """Retained exhaustive compatibility route for external V1 callers."""

        return self._transport.exhaustive_search(
            objective_function,
            problem_type=problem_type,
            target_contract=target_contract,
        )

    def search_coverage(self, compileiq_search):
        if getattr(compileiq_search, "PROTOCOL", "") == (
            "budgeted_staged_pareto_racing_main_thread_v2"
        ):
            checkpoint = compileiq_search.checkpoint()
            if not checkpoint.stages:
                return MappingProxyType(
                    {
                        "complete": False,
                        "baseline_observed": False,
                        "evaluation_count": checkpoint.evaluation_count,
                        "observed_recipe_ids": (),
                        "missing_recipe_ids": self.recipe_ids,
                        "verified_core": True,
                        "termination_reason": compileiq_search.termination_reason,
                    }
                )
            stage = checkpoint.stages[-1]
            observed = tuple(stage.evaluated_recipe_ids)
            missing = tuple(
                recipe_id
                for recipe_id in checkpoint.batches[-1].recipe_ids
                if recipe_id not in observed
            )
            return MappingProxyType(
                {
                    "complete": stage.complete,
                    "baseline_observed": self.baseline_recipe_id in observed,
                    "evaluation_count": checkpoint.evaluation_count,
                    "observed_recipe_ids": observed,
                    "missing_recipe_ids": missing,
                    "verified_core": True,
                    "termination_reason": compileiq_search.termination_reason,
                }
            )
        return self._transport.search_coverage(compileiq_search)

    def require_complete_search(self, compileiq_search):
        if getattr(compileiq_search, "PROTOCOL", "") == (
            "budgeted_staged_pareto_racing_main_thread_v2"
        ):
            coverage = self.search_coverage(compileiq_search)
            if not coverage["complete"]:
                raise RuntimeError(
                    "CompileIQ V2 stopped with a partial current-stage frontier; "
                    f"missing={coverage['missing_recipe_ids']!r}"
                )
            return coverage
        return self._transport.require_complete_search(compileiq_search)

    def select_best_result(self, compileiq_search, result):
        if getattr(compileiq_search, "PROTOCOL", "") == (
            "budgeted_staged_pareto_racing_main_thread_v2"
        ):
            best = result.get_best_result()
            if not isinstance(best, dict):
                raise TypeError("CompileIQ V2 best result is not a dictionary")
            recipe_id = best.get("recipe_id")
            if recipe_id not in self.recipe_ids:
                raise ValueError("CompileIQ V2 selected an unknown complete recipe")
            return self._selection(recipe_id)
        recipe_id = self._transport.select_best_recipe_id(compileiq_search, result)
        return self._selection(recipe_id)

    def recipe_manifest(self, recipe_id):
        if recipe_id not in self.recipe_ids:
            raise KeyError(f"unknown complete Graph recipe {recipe_id!r}")
        return MappingProxyType(
            {
                **self._selection(recipe_id).to_dict(),
                "is_baseline": recipe_id == self.baseline_recipe_id,
            }
        )

    def worker_environment(self, parameters):
        self._decoded_recipe_id(parameters)
        return MappingProxyType({})

    def verify_materialized_graph(self, parameters, graph):
        recipe_id = self._decoded_recipe_id(parameters)
        manifest = getattr(graph, "manifest", None)
        if manifest is not None and hasattr(graph, "executor"):
            if manifest.recipe_id != recipe_id:
                raise ValueError(
                    "materialized Graph selected a different complete recipe"
                )
            graph = graph.executor
        if getattr(graph, "definition", None) is not self._definition:
            raise ValueError("materialized Graph belongs to a different definition")
        actual_recipe_id = getattr(
            getattr(graph, "_spec", None),
            "_complete_recipe_id",
            self.baseline_recipe_id,
        )
        if actual_recipe_id != recipe_id:
            raise ValueError("materialized Graph selected a different complete recipe")
        return self._selection(recipe_id)

    def manifest(self):
        value = {
            "schema": "taichi_forge.graph.compileiq-complete-recipe-search.v2",
            **self._transport.manifest(),
            "semantic_plan_id": self.semantic_plan_id,
            "backend": self.backend,
            "families": self._families,
            "recipes": tuple(
                dict(self.recipe_manifest(recipe_id)) for recipe_id in self.recipe_ids
            ),
            "runtime_admission": "explicit_materialization_context_only",
        }
        if self._family:
            value["recipe_kind"] = {
                "bounded_execution": "graph_bounded_execution",
                "native_algorithm": "graph_native_algorithm",
            }.get(self._family, self._family)
        return value


class _CompleteGraphRecipeSearchSessionV2:
    """Bind CompileIQ racing to Forge's sole Graph materialization context."""

    PROTOCOL = "budgeted_staged_pareto_racing_main_thread_v2"

    def __init__(
        self,
        plans,
        objective_function,
        *,
        budget,
        problem_type,
        target_contract,
        deterministic_seed,
        halving_factor,
        minimum_survivors,
        repeat_count,
        fidelity_name,
        checkpoint,
        context,
    ):
        if not callable(objective_function):
            raise TypeError("objective_function must be callable")
        if problem_type not in ("min", "max"):
            raise ValueError("problem_type must be 'min' or 'max'")
        try:
            support = import_module("compileiq.forge_support")
            budget_type = getattr(support, "ForgeOpaqueSearchBudgetV2")
            target_type = getattr(support, "ForgeOpaqueTargetContractV1")
            objective_type = getattr(support, "ForgeOpaqueObjectiveV1")
            outcome_type = getattr(support, "TrialOutcomeV2")
            cleanup_type = getattr(support, "TrialCleanupV2")
            failure_type = getattr(support, "TrialFailureV2")
        except (ImportError, AttributeError) as error:
            raise CompileIQGraphUnavailableError(
                "modified CompileIQ does not expose the V2 Forge search contract"
            ) from error
        if not isinstance(budget, budget_type):
            if not isinstance(budget, dict):
                raise TypeError("budget must be a ForgeOpaqueSearchBudgetV2")
            budget = budget_type(**budget)
        scalar_metric = target_contract is None
        if target_contract is None:
            target_contract = target_type(
                objectives=(objective_type(name="score", direction=problem_type),)
            )
        elif not isinstance(target_contract, target_type):
            raise TypeError("target_contract must be a ForgeOpaqueTargetContractV1")
        elif problem_type != "min":
            raise ValueError(
                "problem_type is unavailable with an explicit target contract"
            )

        self._plans = plans
        self._objective_function = objective_function
        self._target_contract = target_contract
        self._scalar_metric = scalar_metric
        self._outcome_type = outcome_type
        self._cleanup_type = cleanup_type
        self._failure_type = failure_type
        self._owns_context = context is None
        self._context = context or plans._definition.materialization_context(
            workspace_lanes=plans._workspace_lanes,
            workspace_saturation=plans._workspace_saturation,
        )
        self._closed = False
        self._default_batch = plans.batch(
            fidelity_name=fidelity_name,
            repeat_count=repeat_count,
        )
        self._session = plans._transport.search_session_v2(
            self._evaluate,
            target_contract=target_contract,
            budget=budget,
            deterministic_seed=deterministic_seed,
            halving_factor=halving_factor,
            minimum_survivors=minimum_survivors,
            checkpoint=checkpoint,
        )

    @property
    def opaque_recipe_capability(self):
        return self._session.opaque_recipe_capability

    @property
    def opaque_recipe_core_provenance(self):
        return self._session.opaque_recipe_core_provenance

    @property
    def observations(self):
        return self._session.observations

    @property
    def evaluation_count(self):
        return self._session.evaluation_count

    @property
    def termination_reason(self):
        return self._session.termination_reason

    def _cleanup(self, status, released, detail):
        return self._cleanup_type(
            status=status,
            released_resources=released,
            detail_code=detail,
        )

    def _failure(
        self,
        request,
        recipe,
        *,
        category,
        code,
        message,
        cleanup,
        manifest=None,
    ):
        return self._outcome_type(
            metrics={},
            planned_physical_id=recipe.planned_physical_id,
            materialized_physical_id=(
                None if manifest is None else manifest.materialized_physical_id
            ),
            materialized_memory_bytes=(
                0
                if manifest is None
                else (
                    manifest.persistent_allocated_bytes
                    + manifest.transient_allocated_bytes
                )
            ),
            provenance={
                "backend": self._plans.backend,
                "batch_fingerprint": request.batch_fingerprint,
                "compileiq_python_source_lock": self._plans.python_source_lock,
                "recipe_id": request.recipe_id,
                "semantic_graph_id": self._plans.semantic_plan_id,
            },
            cleanup=cleanup,
            failure=self._failure_type(
                category=category,
                code=code,
                message=message,
                retryable=False,
            ),
        )

    def _evaluate(self, request):
        recipe = self._plans._catalog.entry(request.recipe_id).recipe
        try:
            materialized = self._plans._definition.materialize(
                recipe,
                context=self._context,
            )
        except Exception as error:
            cleanup_complete = bool(getattr(error, "cleanup_complete", False))
            return self._failure(
                request,
                recipe,
                category="materialization",
                code=type(error).__name__,
                message=str(error).strip() or type(error).__name__,
                cleanup=(
                    self._cleanup(
                        "not_required",
                        False,
                        "materialization_rolled_back",
                    )
                    if cleanup_complete
                    else self._cleanup(
                        "incomplete",
                        False,
                        "materialization_cleanup_incomplete",
                    )
                ),
            )

        manifest = materialized.manifest
        objective_error = None
        observation_error = None
        raw_metrics = None
        try:
            raw_metrics = self._objective_function(materialized.executor, request)
        except Exception as error:
            objective_error = error
        if objective_error is None:
            try:
                # Some Graph-owned resources are intentionally allocated only
                # after the evaluator publishes concrete bindings.  Refresh
                # the observation here, outside the replay hot path, so the
                # V2 memory budget sees the evaluated physical instance rather
                # than the empty pre-binding shell.
                from taichi_forge.graph._recipes.physical import (
                    observe_graph_physical_manifest,
                )

                manifest = observe_graph_physical_manifest(
                    self._plans._definition,
                    recipe,
                    materialized.executor,
                )
            except Exception as error:
                observation_error = error
        try:
            materialized.close()
        except Exception as error:
            return self._failure(
                request,
                recipe,
                category="cleanup",
                code=type(error).__name__,
                message=str(error).strip() or type(error).__name__,
                cleanup=self._cleanup(
                    "incomplete",
                    False,
                    "materialized_graph_release_failed",
                ),
                manifest=manifest,
            )
        cleanup = self._cleanup(
            "complete",
            True,
            "materialized_graph_release_complete",
        )
        if objective_error is not None:
            return self._failure(
                request,
                recipe,
                category="objective",
                code=type(objective_error).__name__,
                message=(
                    str(objective_error).strip() or type(objective_error).__name__
                ),
                cleanup=cleanup,
                manifest=manifest,
            )
        if observation_error is not None:
            return self._failure(
                request,
                recipe,
                category="materialization",
                code=type(observation_error).__name__,
                message=(
                    str(observation_error).strip() or type(observation_error).__name__
                ),
                cleanup=cleanup,
                manifest=manifest,
            )
        if self._scalar_metric:
            if isinstance(raw_metrics, bool) or not isinstance(
                raw_metrics, (int, float)
            ):
                return self._failure(
                    request,
                    recipe,
                    category="protocol",
                    code="scalar_objective_required",
                    message="scalar objective must return one finite numeric score",
                    cleanup=cleanup,
                    manifest=manifest,
                )
            raw_metrics = {"score": float(raw_metrics)}
        elif not isinstance(raw_metrics, dict):
            return self._failure(
                request,
                recipe,
                category="protocol",
                code="named_metrics_required",
                message="explicit target objective must return a metric dictionary",
                cleanup=cleanup,
                manifest=manifest,
            )
        try:
            if any(
                not isinstance(name, str)
                or not name
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for name, value in raw_metrics.items()
            ):
                raise ValueError(
                    "metric names must be nonempty strings and values finite numbers"
                )
            metrics = {name: float(value) for name, value in raw_metrics.items()}
        except (AttributeError, TypeError, ValueError) as error:
            return self._failure(
                request,
                recipe,
                category="protocol",
                code="invalid_named_metrics",
                message=str(error).strip() or type(error).__name__,
                cleanup=cleanup,
                manifest=manifest,
            )
        if "materialized_memory_bytes" in self._target_contract.metric_names:
            metrics.setdefault(
                "materialized_memory_bytes",
                float(
                    manifest.persistent_allocated_bytes
                    + manifest.transient_allocated_bytes
                ),
            )
        return self._outcome_type(
            metrics=metrics,
            planned_physical_id=recipe.planned_physical_id,
            materialized_physical_id=manifest.materialized_physical_id,
            materialized_memory_bytes=(
                manifest.persistent_allocated_bytes + manifest.transient_allocated_bytes
            ),
            provenance={
                "backend": self._plans.backend,
                "batch_fingerprint": request.batch_fingerprint,
                "compileiq_python_source_lock": self._plans.python_source_lock,
                "recipe_id": request.recipe_id,
                "semantic_graph_id": self._plans.semantic_plan_id,
            },
            cleanup=cleanup,
        )

    def start(self):
        return self.submit_batch(self._default_batch)

    def submit_batch(self, batch):
        if self._closed:
            raise RuntimeError("CompileIQ V2 Graph search session is closed")
        return self._session.submit_batch(batch)

    def result(self):
        return self._session.result()

    def checkpoint(self):
        return self._session.checkpoint()

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._owns_context:
            self._context.close()

    def __enter__(self):
        if self._closed:
            raise RuntimeError("CompileIQ V2 Graph search session is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del exc_type, exc_value, traceback
        self.close()


class _CompleteGraphRecipeSelection:
    """Compatibility view over one Forge-owned complete Graph recipe."""

    def __init__(self, recipe, *, family="", source_manifest=None, selection=None):
        self.recipe = recipe
        self.spec_id = recipe.recipe_id
        self.recipe_id = recipe.recipe_id
        self.compilation_identity = recipe.planned_physical_id
        self.execution_identity = recipe.planned_physical_id
        self.family = family
        self._source_manifest = source_manifest
        self._selection = selection
        self.materialization_recipe = (
            "baseline" if selection is None else selection.materialization_choice
        )
        self.worker_environment = MappingProxyType({})
        self.memory_recipe_manifest = (
            source_manifest if family == "graph_memory" else None
        )
        self.reduction_recipe_manifest = (
            source_manifest if family == "graph_reduction" else None
        )
        self.native_algorithm_recipe_manifest = (
            source_manifest if family == "native_algorithm" else None
        )
        self.bounded_recipe_manifest = (
            source_manifest if family == "bounded_execution" else None
        )
        self.control_recipe_id = (
            ""
            if family != "structured_control"
            else (
                selection.choice_id
                if selection is not None
                else getattr(source_manifest, "recipe_id", "")
            )
        )
        if family == "map_fusion" and selection is not None:
            marker = selection.source_key.removeprefix("dispatches:")
            self.fusion_source_groups = (
                tuple(int(value) for value in marker.split(",")),
            )
            self.fusion_recipe_ids = (selection.choice_id,)
        else:
            self.fusion_source_groups = ()
            self.fusion_recipe_ids = ()

    def to_dict(self):
        value = {
            "spec_id": self.spec_id,
            "recipe_id": self.recipe_id,
            "materialization_recipe": self.materialization_recipe,
            "compilation_identity": self.compilation_identity,
            "execution_identity": self.execution_identity,
            "fragment_ids": tuple(
                fragment.fragment_id for fragment in self.recipe.fragments
            ),
            "family": self.family or "baseline",
            "fusion_recipe_ids": self.fusion_recipe_ids,
            "fusion_source_groups": self.fusion_source_groups,
        }
        manifest = self._source_manifest
        if manifest is not None and hasattr(manifest, "to_dict"):
            payload = manifest.to_dict()
            if self.family == "graph_memory":
                value.update(
                    memory_recipe_id=manifest.recipe_id,
                    memory_recipe_manifest=payload,
                )
            elif self.family == "graph_reduction":
                value.update(
                    reduction_recipe_id=manifest.recipe_id,
                    reduction_recipe_manifest=payload,
                )
            elif self.family == "native_algorithm":
                value.update(
                    native_algorithm_recipe_id=manifest.recipe_id,
                    native_algorithm_recipe_manifest=payload,
                )
            elif self.family == "bounded_execution":
                value.update(
                    bounded_recipe_id=manifest.recipe_id,
                    bounded_recipe_manifest=payload,
                )
        if self.control_recipe_id:
            value["control_recipe_id"] = self.control_recipe_id
        return value


def compileiq_recipe_search(graph):
    """Build a baseline-inclusive domain from the complete recipe catalog."""

    if getattr(graph, "definition", None) is not None:
        return CompileIQCompleteGraphRecipeSearch(graph)

    return CompileIQGraphRecipeSearch(graph)


__all__ = [
    "CompileIQCompleteGraphRecipeSearch",
    "CompileIQGraphRecipeSearch",
    "CompileIQGraphUnavailableError",
    "compileiq_recipe_search",
]
