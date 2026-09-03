"""Public complete-Graph optimization workflow.

The public surface deals in whole Graph recipes and measured objectives.  Raw
provider knobs remain private to Forge's fragment/materializer layer.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field


def _nonempty_text(value, role):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{role} must be non-empty text")
    return value


def _finite_number(value, role):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{role} must be a finite number")
    return float(value)


def _canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


@dataclass(frozen=True)
class GraphOptimizationTarget:
    """Ordered whole-workload objectives and optional explicit constraints.

    Objectives are ``(metric_name, direction)`` pairs.  Their order is used
    only to choose one deterministic recipe from the measured Pareto frontier;
    CompileIQ itself retains the unsquashed multi-objective frontier.
    """

    objectives: tuple[tuple[str, str], ...] = (("device_time_ns", "min"),)
    constraints: tuple[tuple[str, str, float], ...] = ()

    def __post_init__(self):
        objectives = tuple(tuple(item) for item in self.objectives)
        constraints = tuple(tuple(item) for item in self.constraints)
        if not objectives:
            raise ValueError("Graph optimization target requires an objective")
        names = []
        for item in objectives:
            if len(item) != 2:
                raise ValueError("Graph objective must be a (name, direction) pair")
            name, direction = item
            _nonempty_text(name, "Graph objective name")
            if direction not in ("min", "max"):
                raise ValueError("Graph objective direction must be min or max")
            names.append(name)
        if len(names) != len(set(names)):
            raise ValueError("Graph objective names must be unique")
        normalized_constraints = []
        for item in constraints:
            if len(item) != 3:
                raise ValueError(
                    "Graph constraint must be a (metric, relation, bound) triple"
                )
            metric, relation, bound = item
            _nonempty_text(metric, "Graph constraint metric")
            if relation not in ("<=", ">="):
                raise ValueError("Graph constraint relation must be <= or >=")
            normalized_constraints.append(
                (metric, relation, _finite_number(bound, "Graph constraint bound"))
            )
        object.__setattr__(self, "objectives", objectives)
        object.__setattr__(self, "constraints", tuple(normalized_constraints))

    def to_dict(self):
        return {
            "objectives": tuple(
                {"name": name, "direction": direction}
                for name, direction in self.objectives
            ),
            "constraints": tuple(
                {"metric": metric, "relation": relation, "bound": bound}
                for metric, relation, bound in self.constraints
            ),
        }

    def _compileiq_contract(self):
        from compileiq.forge_support import (
            ForgeOpaqueConstraintV1,
            ForgeOpaqueObjectiveV1,
            ForgeOpaqueTargetContractV1,
        )

        return ForgeOpaqueTargetContractV1(
            objectives=tuple(
                ForgeOpaqueObjectiveV1(name=name, direction=direction)
                for name, direction in self.objectives
            ),
            constraints=tuple(
                ForgeOpaqueConstraintV1(
                    metric=metric,
                    relation=relation,
                    bound=bound,
                )
                for metric, relation, bound in self.constraints
            ),
        )


@dataclass(frozen=True)
class GraphSearchBudget:
    """Work and materialization limits for one complete-recipe search."""

    evaluation_limit: int
    time_limit_seconds: float = 300.0
    materialized_memory_limit_bytes: int | None = None
    repeat_count: int = 1
    deterministic_seed: int = 0
    halving_factor: int = 2
    minimum_survivors: int = 1

    def __post_init__(self):
        integer_fields = (
            ("evaluation_limit", self.evaluation_limit, 1),
            ("repeat_count", self.repeat_count, 1),
            ("halving_factor", self.halving_factor, 2),
            ("minimum_survivors", self.minimum_survivors, 1),
        )
        for name, value, minimum in integer_fields:
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"Graph search {name} must be at least {minimum}")
        if self.evaluation_limit < self.repeat_count:
            raise ValueError(
                "Graph search evaluation_limit must cover one complete recipe"
            )
        if isinstance(self.deterministic_seed, bool) or not isinstance(
            self.deterministic_seed, int
        ):
            raise TypeError("Graph search deterministic_seed must be an integer")
        if _finite_number(
            self.time_limit_seconds,
            "Graph search time_limit_seconds",
        ) <= 0:
            raise ValueError("Graph search time_limit_seconds must be positive")
        memory = self.materialized_memory_limit_bytes
        if memory is not None and (
            isinstance(memory, bool) or not isinstance(memory, int) or memory < 0
        ):
            raise ValueError(
                "Graph search materialized_memory_limit_bytes must be non-negative"
            )

    @property
    def recipe_capacity(self):
        return max(1, self.evaluation_limit // self.repeat_count)

    def to_dict(self):
        return {
            "evaluation_limit": self.evaluation_limit,
            "time_limit_seconds": float(self.time_limit_seconds),
            "materialized_memory_limit_bytes": self.materialized_memory_limit_bytes,
            "repeat_count": self.repeat_count,
            "deterministic_seed": self.deterministic_seed,
            "halving_factor": self.halving_factor,
            "minimum_survivors": self.minimum_survivors,
        }

    def _compileiq_budget(self):
        from compileiq.forge_support import ForgeOpaqueSearchBudgetV2

        return ForgeOpaqueSearchBudgetV2(
            evaluation_limit=self.evaluation_limit,
            time_limit_seconds=float(self.time_limit_seconds),
            materialized_memory_limit_bytes=(
                sys.maxsize
                if self.materialized_memory_limit_bytes is None
                else self.materialized_memory_limit_bytes
            ),
        )


@dataclass(frozen=True)
class GraphRecipeManifest:
    """Public summary of one complete recipe without provider-local knobs."""

    semantic_graph_id: str
    recipe_id: str
    planned_physical_id: str
    provider_registry_id: str
    generation_domain_id: str
    assembly_protocol: str
    assembly_provider_namespace: str
    families: tuple[str, ...]
    semantic_region_count: int
    selected_fragment_count: int
    execution_step_count: int
    declared_persistent_resource_bytes: int
    declared_transient_resource_bytes: int
    exclusive_submission: bool
    queue_count: int
    barrier_count: int
    is_baseline: bool

    @classmethod
    def _from_recipe(cls, definition, recipe, provider_set=None):
        families = []
        for fragment in recipe.fragments:
            selection = fragment.provider_metadata.get("family_selection", {})
            family = str(selection.get("family", ""))
            if not family:
                family = fragment.provider_namespace.rsplit(".", 1)[-1]
            families.append(family)
        if not families:
            families.append("baseline")
        return cls(
            semantic_graph_id=definition.semantic_graph_id,
            recipe_id=recipe.recipe_id,
            planned_physical_id=recipe.planned_physical_id,
            provider_registry_id=(
                "" if provider_set is None else provider_set.provider_registry_id
            ),
            generation_domain_id=(
                "" if provider_set is None else provider_set.generation_domain_id
            ),
            assembly_protocol=recipe.assembly_protocol,
            assembly_provider_namespace=recipe.assembly_provider_namespace,
            families=tuple(dict.fromkeys(families)),
            semantic_region_count=len(recipe.region_selections),
            selected_fragment_count=len(recipe.fragments),
            execution_step_count=len(recipe.execution_steps),
            declared_persistent_resource_bytes=(
                recipe.declared_persistent_resource_bytes
            ),
            declared_transient_resource_bytes=(
                recipe.declared_transient_resource_bytes
            ),
            exclusive_submission=recipe.exclusive_submission,
            queue_count=len(recipe.queues),
            barrier_count=recipe.barrier_count,
            is_baseline=(recipe.recipe_id == definition.baseline_recipe.recipe_id),
        )

    @property
    def declared_materialized_resource_bytes(self):
        return (
            self.declared_persistent_resource_bytes
            + self.declared_transient_resource_bytes
        )

    def to_dict(self):
        return {
            "semantic_graph_id": self.semantic_graph_id,
            "recipe_id": self.recipe_id,
            "planned_physical_id": self.planned_physical_id,
            "provider_registry_id": self.provider_registry_id,
            "generation_domain_id": self.generation_domain_id,
            "assembly": {
                "protocol": self.assembly_protocol,
                "provider_namespace": self.assembly_provider_namespace,
            },
            "families": self.families,
            "semantic_region_count": self.semantic_region_count,
            "selected_fragment_count": self.selected_fragment_count,
            "execution_step_count": self.execution_step_count,
            "resources": {
                "declared_persistent_bytes": (
                    self.declared_persistent_resource_bytes
                ),
                "declared_transient_bytes": self.declared_transient_resource_bytes,
            },
            "submission": {
                "exclusive": self.exclusive_submission,
                "queue_count": self.queue_count,
                "barrier_count": self.barrier_count,
            },
            "is_baseline": self.is_baseline,
        }


@dataclass(frozen=True)
class GraphRecipeHandle:
    """Opaque, definition-bound selection returned by Graph recipe search."""

    manifest: GraphRecipeManifest
    _recipe: object = field(repr=False, compare=False, hash=False)
    _provider_set: object = field(
        default=None,
        repr=False,
        compare=False,
        hash=False,
    )

    @classmethod
    def _from_recipe(cls, definition, recipe, provider_set=None):
        return cls(
            GraphRecipeManifest._from_recipe(
                definition,
                recipe,
                provider_set,
            ),
            recipe,
            provider_set,
        )

    @property
    def semantic_graph_id(self):
        return self.manifest.semantic_graph_id

    @property
    def recipe_id(self):
        return self.manifest.recipe_id

    @property
    def planned_physical_id(self):
        return self.manifest.planned_physical_id

    def to_dict(self):
        return self.manifest.to_dict()


@dataclass(frozen=True)
class GraphOptimizationReport:
    """Detached, serializable evidence from one modified-CompileIQ search."""

    semantic_graph_id: str
    target: GraphOptimizationTarget
    budget: GraphSearchBudget
    selected_recipe_id: str
    pareto_recipe_ids: tuple[str, ...]
    search_complete: bool
    termination_reason: str
    evaluation_count: int
    measured_recipe_ids: tuple[str, ...]
    missing_recipe_ids: tuple[str, ...]
    _results_json: str = field(repr=False)
    _capability_json: str = field(repr=False)
    _provenance_json: str = field(repr=False)
    _checkpoint_json: str = field(repr=False)

    @property
    def results(self):
        return tuple(json.loads(self._results_json))

    @property
    def compileiq_capability(self):
        return json.loads(self._capability_json)

    @property
    def compileiq_provenance(self):
        return json.loads(self._provenance_json)

    @property
    def checkpoint(self):
        return json.loads(self._checkpoint_json)

    def to_dict(self):
        return {
            "schema": "taichi_forge.graph_optimization_report.v1",
            "semantic_graph_id": self.semantic_graph_id,
            "target": self.target.to_dict(),
            "budget": self.budget.to_dict(),
            "selected_recipe_id": self.selected_recipe_id,
            "pareto_recipe_ids": self.pareto_recipe_ids,
            "search": {
                "complete": self.search_complete,
                "termination_reason": self.termination_reason,
                "evaluation_count": self.evaluation_count,
                "measured_recipe_ids": self.measured_recipe_ids,
                "missing_recipe_ids": self.missing_recipe_ids,
            },
            "results": self.results,
            "compileiq_capability": self.compileiq_capability,
            "compileiq_provenance": self.compileiq_provenance,
            "checkpoint": self.checkpoint,
        }


@dataclass(frozen=True)
class GraphOptimizationDecision:
    """One selected complete recipe plus its full measured frontier."""

    selection: GraphRecipeHandle
    pareto_frontier: tuple[GraphRecipeHandle, ...]
    report: GraphOptimizationReport

    def to_dict(self):
        return {
            "selection": self.selection.to_dict(),
            "pareto_frontier": tuple(
                item.to_dict() for item in self.pareto_frontier
            ),
            "report": self.report.to_dict(),
        }


class _GraphRecipeSearchSession:
    """Single-use public façade over Forge-owned recipe construction and V2."""

    def __init__(
        self,
        definition,
        *,
        engine,
        target,
        budget,
        providers,
        available_capabilities,
    ):
        if engine != "compileiq":
            raise ValueError("Graph recipe search engine must be compileiq")
        if target is None:
            target = GraphOptimizationTarget()
        if not isinstance(target, GraphOptimizationTarget):
            raise TypeError(
                "Graph recipe search target must be GraphOptimizationTarget"
            )
        if not isinstance(budget, GraphSearchBudget):
            raise TypeError("Graph recipe search budget must be GraphSearchBudget")

        from taichi_forge.graph._compileiq_opaque import (
            CompileIQCompleteGraphRecipeSearch,
        )

        catalog = definition.recipe_catalog(
            providers=providers,
            available_capabilities=available_capabilities,
        )
        remaining_capacity = max(
            0,
            budget.recipe_capacity - len(catalog.entries()),
        )
        catalog.build_compatible_stage(candidate_limit=remaining_capacity)
        graph = definition.compile()
        self._definition = definition
        self._plans = CompileIQCompleteGraphRecipeSearch(graph, catalog=catalog)
        self._target = target
        self._budget = budget
        self._handles = {
            entry.recipe.recipe_id: GraphRecipeHandle._from_recipe(
                definition,
                entry.recipe,
                catalog.provider_set,
            )
            for entry in catalog.entries()
        }
        self._used = False

    @property
    def recipes(self):
        return tuple(
            self._handles[recipe_id] for recipe_id in self._plans.recipe_ids
        )

    @property
    def baseline(self):
        return self._handles[self._plans.baseline_recipe_id]

    def _selection_key(self, result):
        values = []
        for name, direction in self._target.objectives:
            value = result["metrics"][name]
            values.append(value if direction == "min" else -value)
        values.append(result["recipe_id"])
        return tuple(values)

    def _dominates(self, left, right):
        no_worse = True
        strictly_better = False
        for name, direction in self._target.objectives:
            left_value = left["metrics"][name]
            right_value = right["metrics"][name]
            if direction == "min":
                no_worse = no_worse and left_value <= right_value
                strictly_better = strictly_better or left_value < right_value
            else:
                no_worse = no_worse and left_value >= right_value
                strictly_better = strictly_better or left_value > right_value
        return no_worse and strictly_better

    def run(self, evaluator):
        """Evaluate complete recipes and return a reproducible decision.

        ``evaluator`` receives ``(materialized_graph, recipe_handle)`` and must
        return a dictionary containing the target metrics.  Forge injects the
        exact ``materialized_memory_bytes`` metric when it is requested.
        """

        if self._used:
            raise RuntimeError("Graph recipe search sessions are single-use")
        if not callable(evaluator):
            raise TypeError("Graph recipe evaluator must be callable")
        self._used = True

        def objective(graph, request):
            return evaluator(graph, self._handles[request.recipe_id])

        with self._plans.compileiq_search(
            objective,
            budget=self._budget._compileiq_budget(),
            target_contract=self._target._compileiq_contract(),
            deterministic_seed=self._budget.deterministic_seed,
            halving_factor=self._budget.halving_factor,
            minimum_survivors=self._budget.minimum_survivors,
            repeat_count=self._budget.repeat_count,
        ) as search:
            result = search.start()
            coverage = dict(self._plans.search_coverage(search))
            checkpoint = search.checkpoint().as_dict()
            capability = search.opaque_recipe_capability
            provenance = search.opaque_recipe_core_provenance

        results = result.get_results()
        complete_feasible = tuple(
            item for item in results if item["complete"] and item["feasible"]
        )
        frontier = tuple(
            item
            for item in complete_feasible
            if not any(
                self._dominates(other, item)
                for other in complete_feasible
                if other["recipe_id"] != item["recipe_id"]
            )
        )
        if not frontier:
            raise RuntimeError(
                "Graph recipe search produced no complete feasible measured recipe"
            )
        selected_result = min(frontier, key=self._selection_key)
        selected = self._handles[selected_result["recipe_id"]]
        frontier_handles = tuple(
            self._handles[item["recipe_id"]]
            for item in sorted(frontier, key=self._selection_key)
        )
        report = GraphOptimizationReport(
            semantic_graph_id=self._definition.semantic_graph_id,
            target=self._target,
            budget=self._budget,
            selected_recipe_id=selected.recipe_id,
            pareto_recipe_ids=tuple(item.recipe_id for item in frontier_handles),
            search_complete=bool(coverage["complete"]),
            termination_reason=result.termination_reason,
            evaluation_count=int(coverage["evaluation_count"]),
            measured_recipe_ids=tuple(coverage["observed_recipe_ids"]),
            missing_recipe_ids=tuple(coverage["missing_recipe_ids"]),
            _results_json=_canonical_json(results),
            _capability_json=_canonical_json(capability),
            _provenance_json=_canonical_json(provenance),
            _checkpoint_json=_canonical_json(checkpoint),
        )
        return GraphOptimizationDecision(
            selection=selected,
            pareto_frontier=frontier_handles,
            report=report,
        )


__all__ = [
    "GraphOptimizationDecision",
    "GraphOptimizationReport",
    "GraphOptimizationTarget",
    "GraphRecipeHandle",
    "GraphRecipeManifest",
    "GraphSearchBudget",
]
