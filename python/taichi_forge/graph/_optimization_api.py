"""Public complete-Graph optimization workflow.

The public surface deals in whole Graph recipes and measured objectives.  Raw
provider knobs remain private to Forge's fragment/materializer layer.
"""

from __future__ import annotations

import hashlib
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
class GraphRecipeSearchStrategy:
    """Bounded complete-recipe generation policy owned by Forge."""

    mode: str = "exact_if_bounded"
    exact_composition_limit: int = 256
    max_generation_rounds: int = 3
    max_generated_recipes: int = 1024

    def __post_init__(self):
        if self.mode not in ("exact_if_bounded", "staged"):
            raise ValueError(
                "Graph recipe search strategy mode must be exact_if_bounded or staged"
            )
        for name, value, minimum in (
            ("exact_composition_limit", self.exact_composition_limit, 1),
            ("max_generation_rounds", self.max_generation_rounds, 0),
            ("max_generated_recipes", self.max_generated_recipes, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(
                    f"Graph recipe search strategy {name} must be at least {minimum}"
                )
        if (
            self.mode == "exact_if_bounded"
            and self.exact_composition_limit > self.max_generated_recipes
        ):
            raise ValueError(
                "Graph exact composition limit cannot exceed max generated recipes"
            )

    @property
    def strategy_id(self):
        payload = _canonical_json(self.to_dict()).encode("ascii")
        return "graph-recipe-search-strategy-v1:" + hashlib.sha256(payload).hexdigest()

    def to_dict(self):
        return {
            "schema": "taichi_forge.graph_recipe_search_strategy.v1",
            "mode": self.mode,
            "exact_composition_limit": self.exact_composition_limit,
            "max_generation_rounds": self.max_generation_rounds,
            "max_generated_recipes": self.max_generated_recipes,
        }


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
    strategy: GraphRecipeSearchStrategy
    selected_recipe_id: str | None
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
    _status_json: str = field(repr=False)

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

    @property
    def status(self):
        return json.loads(self._status_json)

    def to_dict(self):
        return {
            "schema": "taichi_forge.graph_optimization_report.v1",
            "semantic_graph_id": self.semantic_graph_id,
            "target": self.target.to_dict(),
            "budget": self.budget.to_dict(),
            "strategy": self.strategy.to_dict(),
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
            "status": self.status,
            "checkpoint": self.checkpoint,
        }


@dataclass(frozen=True)
class GraphOptimizationDecision:
    """One selected complete recipe plus its full measured frontier."""

    selection: GraphRecipeHandle | None
    pareto_frontier: tuple[GraphRecipeHandle, ...]
    report: GraphOptimizationReport

    def to_dict(self):
        return {
            "selection": (
                None if self.selection is None else self.selection.to_dict()
            ),
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
        strategy,
        checkpoint,
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
        if strategy is None:
            strategy = GraphRecipeSearchStrategy()
        if not isinstance(strategy, GraphRecipeSearchStrategy):
            raise TypeError(
                "Graph recipe search strategy must be GraphRecipeSearchStrategy"
            )

        from taichi_forge.graph._compileiq_opaque import (
            CompileIQCompleteGraphRecipeSearch,
        )

        catalog = definition.recipe_catalog(
            providers=providers,
            available_capabilities=available_capabilities,
        )
        self._definition = definition
        self._catalog = catalog
        self._target = target
        self._budget = budget
        self._strategy = strategy
        self._seed_fragment_ids = tuple(
            fragment.fragment_id for fragment in catalog.fragments
        )
        exact = None
        if strategy.mode == "exact_if_bounded":
            exact = catalog.build_exact_stage(
                candidate_limit=strategy.exact_composition_limit,
            )
        self._execution_mode = (
            "exact" if exact is not None and exact.exhaustive else "staged"
        )
        self._exact_enumeration = exact
        if len(catalog.entries()) > strategy.max_generated_recipes:
            raise ValueError(
                "initial Graph recipe domain exceeds max_generated_recipes"
            )
        self._checkpoint = self._normalize_checkpoint(checkpoint)
        if self._checkpoint is not None:
            self._rebuild_checkpoint_catalog(self._checkpoint)
        graph = definition.compile()
        self._plans = CompileIQCompleteGraphRecipeSearch(
            graph,
            catalog=catalog,
            search_strategy_id=strategy.strategy_id,
        )
        self._handles = {}
        self._refresh_handles()
        self._used = False

    @staticmethod
    def _normalize_checkpoint(checkpoint):
        if checkpoint is None:
            return None
        if hasattr(checkpoint, "as_dict"):
            checkpoint = checkpoint.as_dict()
        if not isinstance(checkpoint, dict):
            raise TypeError("Graph recipe checkpoint must be a dictionary")
        return json.loads(_canonical_json(checkpoint))

    def _refresh_handles(self):
        for entry in self._catalog.entries():
            self._handles.setdefault(
                entry.recipe.recipe_id,
                GraphRecipeHandle._from_recipe(
                    self._definition,
                    entry.recipe,
                    self._catalog.provider_set,
                ),
            )

    @staticmethod
    def _batch_recipe_ids(batch):
        return tuple(item["recipe_id"] for item in batch.get("recipes", ()))

    def _remaining_generation_capacity(self):
        return max(
            0,
            self._strategy.max_generated_recipes - len(self._catalog.entries()),
        )

    def _generate(self, survivor_recipe_ids):
        return self._catalog.build_survivor_stage(
            survivor_recipe_ids,
            seed_fragment_ids=self._seed_fragment_ids,
            candidate_limit=self._remaining_generation_capacity(),
        )

    def _stage_members(self, survivor_recipe_ids, new_entries=()):
        baseline_id = self._catalog.baseline.recipe.recipe_id
        survivor_ids = tuple(
            sorted(
                dict.fromkeys(survivor_recipe_ids),
                key=lambda item: item.encode("utf-8"),
            )
        )
        recipe_ids = tuple(
            sorted(
                dict.fromkeys(
                    (baseline_id, *survivor_ids)
                    + tuple(entry.recipe.recipe_id for entry in new_entries)
                ),
                key=lambda item: item.encode("utf-8"),
            )
        )
        parents = {}
        survivor_set = set(survivor_ids)
        for recipe_id in recipe_ids:
            if recipe_id in survivor_set or recipe_id == baseline_id:
                parents[recipe_id] = (recipe_id,)
                continue
            entry = self._catalog.entry(recipe_id)
            if not set(entry.parent_recipe_ids).issubset(survivor_set):
                raise RuntimeError(
                    "generated Graph recipe lineage escaped the measured frontier"
                )
            parents[recipe_id] = entry.parent_recipe_ids
        return recipe_ids, parents

    def _validate_checkpoint_batch(
        self,
        batch,
        expected_recipe_ids,
        expected_parents=None,
    ):
        actual_ids = self._batch_recipe_ids(batch)
        if actual_ids != tuple(
            sorted(expected_recipe_ids, key=lambda item: item.encode("utf-8"))
        ):
            raise ValueError(
                "Graph recipe checkpoint generation differs from this strategy"
            )
        expected_parents = expected_parents or {
            recipe_id: () for recipe_id in actual_ids
        }
        actual_parents = {
            item["recipe_id"]: tuple(item.get("parent_recipe_ids", ()))
            for item in batch.get("recipes", ())
        }
        normalized_expected_parents = {
            recipe_id: tuple(
                sorted(parent_ids, key=lambda item: item.encode("utf-8"))
            )
            for recipe_id, parent_ids in expected_parents.items()
        }
        if actual_parents != normalized_expected_parents:
            raise ValueError(
                "Graph recipe checkpoint lineage differs from this strategy"
            )
        planned = {
            item["recipe_id"]: item["planned_physical_id"]
            for item in batch.get("recipes", ())
        }
        if any(
            planned.get(recipe_id)
            != self._catalog.entry(recipe_id).recipe.planned_physical_id
            for recipe_id in actual_ids
        ):
            raise ValueError(
                "Graph recipe checkpoint planned physical identity drifted"
            )

    def _rebuild_checkpoint_catalog(self, checkpoint):
        batches = tuple(checkpoint.get("batches", ()))
        stages = tuple(checkpoint.get("stages", ()))
        if len(stages) != len(batches):
            raise ValueError("Graph recipe checkpoint has inconsistent stage history")
        if not batches:
            return
        initial_ids = tuple(entry.recipe.recipe_id for entry in self._catalog.entries())
        self._validate_checkpoint_batch(batches[0], initial_ids)
        if self._execution_mode == "exact" and len(batches) != 1:
            raise ValueError("exact Graph recipe checkpoint contains extra stages")
        for index in range(1, len(batches)):
            previous_survivors = tuple(stages[index - 1].get("survivor_recipe_ids", ()))
            fidelity = batches[index].get("fidelity", {})
            if fidelity.get("terminal", False):
                expected_ids, parents = self._stage_members(previous_survivors)
            else:
                generated = self._generate(previous_survivors)
                expected_ids, parents = self._stage_members(
                    previous_survivors,
                    generated,
                )
            self._validate_checkpoint_batch(
                batches[index],
                expected_ids,
                parents,
            )

    @property
    def recipes(self):
        self._refresh_handles()
        return tuple(self._handles[recipe_id] for recipe_id in self._plans.recipe_ids)

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

    @staticmethod
    def _finalization_type():
        from compileiq.forge_support import ForgeOpaqueSearchFinalizationV1

        return ForgeOpaqueSearchFinalizationV1

    def _finalize(self, search, *, generation_status, reason):
        checkpoint = search.checkpoint()
        if checkpoint.finalization is not None:
            return search.result()
        if search.termination_reason == "poisoned":
            return search.result()
        terminal_stage = None
        if checkpoint.batches and checkpoint.batches[-1].fidelity.terminal:
            terminal_stage = checkpoint.stages[-1]
        terminal_status = (
            "not_reached"
            if terminal_stage is None
            else ("complete" if terminal_stage.complete else "partial")
        )
        return search.finalize(
            self._finalization_type()(
                generation_status=generation_status,
                terminal_fidelity_status=terminal_status,
                reason=reason,
            )
        )

    def _run_exact(self, search):
        checkpoint = search.checkpoint()
        if checkpoint.finalization is not None:
            return search.result()
        if not checkpoint.batches:
            batch = self._plans.batch(
                recipe_ids=self._plans.recipe_ids,
                stage_index=0,
                fidelity_name="terminal-exact",
                fidelity_ordinal=0,
                repeat_count=self._budget.repeat_count,
                terminal=True,
            )
            result = search.submit_batch(batch)
        else:
            if len(checkpoint.batches) != 1:
                raise ValueError("exact Graph recipe search has multiple batches")
            result = search.result()
            if not checkpoint.stages[-1].complete:
                result = search.submit_batch(checkpoint.batches[-1])
        terminal_complete = search.checkpoint().stages[-1].complete
        if not terminal_complete:
            return result
        return self._finalize(
            search,
            generation_status="exhaustive",
            reason="exact_domain_complete",
        )

    @staticmethod
    def _terminal_reason_from_fidelity(name):
        return {
            "terminal-no-new-physical-identity": "no_new_physical_identity",
            "terminal-generation-round-limit": "generation_round_limit",
            "terminal-generated-recipe-limit": "generated_recipe_limit",
        }.get(name, "strategy_complete")

    def _run_staged(self, search):
        checkpoint = search.checkpoint()
        if checkpoint.finalization is not None:
            return search.result()
        if not checkpoint.batches:
            batch = self._plans.batch(
                recipe_ids=self._plans.recipe_ids,
                stage_index=0,
                fidelity_name="screen",
                fidelity_ordinal=0,
                repeat_count=self._budget.repeat_count,
                terminal=False,
            )
            result = search.submit_batch(batch)
        else:
            result = search.result()
            if not checkpoint.stages[-1].complete:
                result = search.submit_batch(checkpoint.batches[-1])

        while True:
            checkpoint = search.checkpoint()
            batch = checkpoint.batches[-1]
            stage = checkpoint.stages[-1]
            if not stage.complete:
                # Keep budget/time interruptions resumable.  Finalizing here
                # would correctly describe a partial run but would also freeze
                # CompileIQ's checkpoint against its missing measurement keys.
                return result
            if batch.fidelity.terminal:
                return self._finalize(
                    search,
                    generation_status="strategy_complete",
                    reason=self._terminal_reason_from_fidelity(batch.fidelity.name),
                )

            generated_rounds = sum(
                previous.stage_index > 0 and not previous.fidelity.terminal
                for previous in checkpoint.batches
            )
            if generated_rounds >= self._strategy.max_generation_rounds:
                terminal_name = "terminal-generation-round-limit"
            elif self._remaining_generation_capacity() == 0:
                terminal_name = "terminal-generated-recipe-limit"
            else:
                generated = self._generate(stage.survivor_recipe_ids)
                self._refresh_handles()
                if generated:
                    recipe_ids, parents = self._stage_members(
                        stage.survivor_recipe_ids,
                        generated,
                    )
                    next_batch = self._plans.batch(
                        recipe_ids=recipe_ids,
                        stage_index=batch.stage_index + 1,
                        parent_batch=batch,
                        parent_recipe_ids=parents,
                        fidelity_name="screen",
                        fidelity_ordinal=batch.fidelity.ordinal + 1,
                        repeat_count=self._budget.repeat_count,
                        terminal=False,
                    )
                    result = search.submit_batch(next_batch)
                    continue
                terminal_name = "terminal-no-new-physical-identity"

            recipe_ids, parents = self._stage_members(stage.survivor_recipe_ids)
            terminal_batch = self._plans.batch(
                recipe_ids=recipe_ids,
                stage_index=batch.stage_index + 1,
                parent_batch=batch,
                parent_recipe_ids=parents,
                fidelity_name=terminal_name,
                fidelity_ordinal=batch.fidelity.ordinal + 1,
                repeat_count=self._budget.repeat_count,
                terminal=True,
            )
            result = search.submit_batch(terminal_batch)

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
            checkpoint=self._checkpoint,
        ) as search:
            result = (
                self._run_exact(search)
                if self._execution_mode == "exact"
                else self._run_staged(search)
            )
            coverage = dict(self._plans.search_coverage(search))
            checkpoint = search.checkpoint().as_dict()
            capability = search.opaque_recipe_capability
            provenance = search.opaque_recipe_core_provenance
            status = result.status.model_dump(by_alias=True)

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
        self._refresh_handles()
        selected = None
        if coverage["complete"] and frontier:
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
            strategy=self._strategy,
            selected_recipe_id=(None if selected is None else selected.recipe_id),
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
            _status_json=_canonical_json(status),
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
    "GraphRecipeSearchStrategy",
    "GraphSearchBudget",
]
