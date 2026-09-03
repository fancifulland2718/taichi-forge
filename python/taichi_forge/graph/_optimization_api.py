"""Public complete-Graph optimization workflow.

The public surface deals in whole Graph recipes and measured objectives.  Raw
provider knobs remain private to Forge's fragment/materializer layer.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import uuid
from dataclasses import dataclass, field

from taichi_forge.graph._reuse import (
    GraphBackendEnvironment,
    GraphEvaluationContract,
    GraphRecipeSearchCheckpointV1,
    GraphRecipeSelectionArtifact,
    GraphWorkloadContext,
)


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


def _compileiq_identity(prefix, value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return prefix + hashlib.sha256(encoded).hexdigest()


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

    @property
    def target_contract_id(self):
        payload = _canonical_json(self.to_dict()).encode("ascii")
        return "graph-optimization-target-v1:" + hashlib.sha256(payload).hexdigest()

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
        if (
            _finite_number(
                self.time_limit_seconds,
                "Graph search time_limit_seconds",
            )
            <= 0
        ):
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
                "declared_persistent_bytes": (self.declared_persistent_resource_bytes),
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
class GraphOptimizationReportV2:
    """Forge explanation around one unmodified CompileIQ fact report."""

    semantic_graph_id: str
    target: GraphOptimizationTarget
    budget: GraphSearchBudget
    strategy: GraphRecipeSearchStrategy
    outcome_status: str
    next_action: str
    selected_recipe_id: str | None
    pareto_recipe_ids: tuple[str, ...]
    selection_artifact_id: str | None
    search_complete: bool
    termination_reason: str
    evaluation_count: int
    measured_recipe_ids: tuple[str, ...]
    missing_recipe_ids: tuple[str, ...]
    _selection_reason_json: str = field(repr=False)
    _tradeoffs_json: str = field(repr=False)
    _recipe_annotations_json: str = field(repr=False)
    _reuse_json: str = field(repr=False)
    _compileiq_report_json: str = field(repr=False)
    _provenance_json: str = field(repr=False)
    _checkpoint_json: str = field(repr=False)
    report_id: str = field(init=False)

    def __post_init__(self):
        allowed = {
            "selected": "apply_selection",
            "resumable": "resume_search",
            "no_feasible_candidate": "review_evidence",
            "failed": "review_failures",
        }
        if allowed.get(self.outcome_status) != self.next_action:
            raise ValueError("Graph optimization outcome and next action disagree")
        if self.selected_recipe_id is not None:
            if self.outcome_status != "selected":
                raise ValueError("Graph report selection requires selected outcome")
            if self.selected_recipe_id not in self.pareto_recipe_ids:
                raise ValueError(
                    "Graph report selection must belong to the Pareto frontier"
                )
        elif self.outcome_status == "selected":
            raise ValueError("selected Graph outcome requires a recipe")
        for payload in (
            self._selection_reason_json,
            self._tradeoffs_json,
            self._recipe_annotations_json,
            self._reuse_json,
            self._provenance_json,
        ):
            json.loads(payload)
        compileiq_report = self.compileiq_report
        if compileiq_report.detail != "summary":
            raise ValueError("Forge report must embed a summary CompileIQ report")
        checkpoint = self.checkpoint
        checkpoint_digest = _compileiq_identity(
            "ciq-checkpoint-facts-v1:",
            checkpoint.compileiq_checkpoint,
        )
        if checkpoint_digest != compileiq_report.checkpoint.digest:
            raise ValueError("Forge checkpoint does not match the CompileIQ report")
        object.__setattr__(
            self,
            "report_id",
            _compileiq_identity("graph-optimization-report-v2:", self._payload()),
        )

    @property
    def selection_reason(self):
        return json.loads(self._selection_reason_json)

    @property
    def pareto_tradeoffs(self):
        return tuple(json.loads(self._tradeoffs_json))

    @property
    def recipe_annotations(self):
        return tuple(json.loads(self._recipe_annotations_json))

    @property
    def reuse(self):
        return json.loads(self._reuse_json)

    @property
    def compileiq_report(self):
        from compileiq.forge_support import OpaqueOptimizationReportV1

        return OpaqueOptimizationReportV1.from_json(self._compileiq_report_json)

    @property
    def compileiq_capability(self):
        return dict(self.compileiq_report.session.capability)

    @property
    def compileiq_provenance(self):
        return json.loads(self._provenance_json)

    @property
    def checkpoint(self):
        return GraphRecipeSearchCheckpointV1.from_dict(
            json.loads(self._checkpoint_json)
        )

    @property
    def status(self):
        return self.compileiq_report.status.model_dump(by_alias=True)

    @property
    def results(self):
        """Compatibility projection of the terminal candidate aggregates."""

        report = self.compileiq_report
        if not report.stages:
            return ()
        terminal_batch = report.stages[-1].batch_fingerprint
        results = []
        for candidate in report.candidates:
            if candidate.batch_fingerprint != terminal_batch:
                continue
            metrics = {
                name: summary.median for name, summary in candidate.metrics.items()
            }
            results.append(
                {
                    "recipe_id": candidate.recipe_id,
                    "params": {"recipe_id": candidate.recipe_id},
                    "stage_index": candidate.stage_index,
                    "fidelity": candidate.fidelity_name,
                    "observation_count": candidate.successful_observation_count,
                    "required_observation_count": (
                        candidate.required_observation_count
                    ),
                    "complete": candidate.complete,
                    "metrics": metrics,
                    "metric_bounds": {
                        name: {
                            "lower": summary.observed_min,
                            "median": summary.median,
                            "upper": summary.observed_max,
                        }
                        for name, summary in candidate.metrics.items()
                    },
                    "feasible": candidate.feasible,
                    "constraint_violations": tuple(
                        item.model_dump(by_alias=True)
                        for item in candidate.constraint_violations
                    ),
                    "failures": tuple(
                        item.model_dump(by_alias=True) for item in candidate.failures
                    ),
                    "planned_physical_id": (
                        candidate.planned_physical_ids[0]
                        if candidate.planned_physical_ids
                        else None
                    ),
                    "materialized_physical_id": (
                        candidate.materialized_physical_ids[0]
                        if candidate.materialized_physical_ids
                        else None
                    ),
                    "materialized_memory_bytes": (
                        candidate.materialized_memory_peak_bytes
                    ),
                }
            )
        return tuple(results)

    def _payload(self):
        return {
            "schema": "taichi_forge.graph_optimization_report.v2",
            "outcome": {
                "status": self.outcome_status,
                "next_action": self.next_action,
            },
            "semantic_graph_id": self.semantic_graph_id,
            "target": self.target.to_dict(),
            "budget": self.budget.to_dict(),
            "strategy": self.strategy.to_dict(),
            "selection": {
                "selected_recipe_id": self.selected_recipe_id,
                "selection_artifact_id": self.selection_artifact_id,
                "reason": self.selection_reason,
            },
            "pareto": {
                "recipe_ids": self.pareto_recipe_ids,
                "tradeoffs": self.pareto_tradeoffs,
            },
            "search": {
                "complete": self.search_complete,
                "termination_reason": self.termination_reason,
                "evaluation_count": self.evaluation_count,
                "measured_recipe_ids": self.measured_recipe_ids,
                "missing_recipe_ids": self.missing_recipe_ids,
            },
            "recipe_annotations": self.recipe_annotations,
            "reuse": self.reuse,
            "compileiq_report": self.compileiq_report.to_dict(),
            "compileiq_provenance": self.compileiq_provenance,
            "checkpoint": self.checkpoint.to_dict(),
        }

    def to_dict(self):
        return {"report_id": self.report_id, **self._payload()}

    def to_json(self):
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise TypeError("Graph optimization report must be a dictionary")
        expected_fields = {
            "report_id",
            "schema",
            "outcome",
            "semantic_graph_id",
            "target",
            "budget",
            "strategy",
            "selection",
            "pareto",
            "search",
            "recipe_annotations",
            "reuse",
            "compileiq_report",
            "compileiq_provenance",
            "checkpoint",
        }
        if set(value) != expected_fields:
            raise ValueError(
                "Graph optimization report has missing or unexpected fields"
            )
        if value.get("schema") != "taichi_forge.graph_optimization_report.v2":
            raise ValueError("Graph optimization report schema is unsupported")
        target_payload = value["target"]
        budget_payload = value["budget"]
        strategy_payload = dict(value["strategy"])
        strategy_payload.pop("schema", None)
        selection = value["selection"]
        pareto = value["pareto"]
        search = value["search"]
        outcome = value["outcome"]
        report = cls(
            semantic_graph_id=value["semantic_graph_id"],
            target=GraphOptimizationTarget(
                objectives=tuple(
                    (item["name"], item["direction"])
                    for item in target_payload["objectives"]
                ),
                constraints=tuple(
                    (item["metric"], item["relation"], item["bound"])
                    for item in target_payload["constraints"]
                ),
            ),
            budget=GraphSearchBudget(**budget_payload),
            strategy=GraphRecipeSearchStrategy(**strategy_payload),
            outcome_status=outcome["status"],
            next_action=outcome["next_action"],
            selected_recipe_id=selection["selected_recipe_id"],
            pareto_recipe_ids=tuple(pareto["recipe_ids"]),
            selection_artifact_id=selection["selection_artifact_id"],
            search_complete=bool(search["complete"]),
            termination_reason=search["termination_reason"],
            evaluation_count=int(search["evaluation_count"]),
            measured_recipe_ids=tuple(search["measured_recipe_ids"]),
            missing_recipe_ids=tuple(search["missing_recipe_ids"]),
            _selection_reason_json=_canonical_json(selection["reason"]),
            _tradeoffs_json=_canonical_json(pareto["tradeoffs"]),
            _recipe_annotations_json=_canonical_json(value["recipe_annotations"]),
            _reuse_json=_canonical_json(value["reuse"]),
            _compileiq_report_json=_canonical_json(value["compileiq_report"]),
            _provenance_json=_canonical_json(value["compileiq_provenance"]),
            _checkpoint_json=_canonical_json(value["checkpoint"]),
        )
        if report.report_id != value.get("report_id"):
            raise ValueError("Graph optimization report identity mismatch")
        return report

    @classmethod
    def from_json(cls, value):
        return cls.from_dict(json.loads(value))

    def to_markdown(self):
        lines = [
            "# Taichi Forge Graph Optimization Report",
            "",
            f"- Report ID: `{self.report_id}`",
            f"- Outcome: `{self.outcome_status}`",
            f"- Next action: `{self.next_action}`",
            f"- Search complete: `{str(self.search_complete).lower()}`",
            f"- Termination: `{self.termination_reason}`",
            "",
            "## Selection",
            "",
            self.selection_reason["summary"],
            "",
            "## Recipe effects",
            "",
            "| Recipe | Families | Regions | Steps | Queues | Barriers | Peak bytes |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for annotation in self.recipe_annotations:
            manifest = annotation["manifest"]
            measurement = annotation["measurement"]
            lines.append(
                f"| `{annotation['recipe_id']}` | "
                f"{', '.join(manifest['families'])} | "
                f"{len(annotation['optimized_semantic_region_ids'])} | "
                f"{manifest['execution_step_count']} | "
                f"{manifest['submission']['queue_count']} | "
                f"{manifest['submission']['barrier_count']} | "
                f"{measurement['materialized_memory_peak_bytes']} |"
            )
        lines.extend(
            ["", "Provider notes above are declarations, not measured claims."]
        )
        if self.pareto_tradeoffs:
            lines.extend(["", "## Pareto trade-offs", ""])
            for tradeoff in self.pareto_tradeoffs:
                comparisons = ", ".join(
                    f"{item['metric']}={item['relation']}"
                    for item in tradeoff["relative_to_selected"]
                )
                lines.append(f"- `{tradeoff['recipe_id']}`: {comparisons}")
        lines.extend(
            [
                "",
                "## Reuse",
                "",
                f"- Scope: `{self.reuse['scope']}`",
                f"- Checkpoint: `{self.reuse['checkpoint_id']}`",
                "- Selection artifact: "
                f"`{self.selection_artifact_id or 'unavailable'}`",
                "",
                "## CompileIQ measurement facts",
                "",
            ]
        )
        compileiq_lines = self.compileiq_report.to_markdown().splitlines()
        if compileiq_lines and compileiq_lines[0].startswith("# "):
            compileiq_lines = compileiq_lines[2:]
        lines.extend(compileiq_lines)
        return "\n".join(lines).rstrip() + "\n"


# Public current name plus an explicit schema-versioned name.
GraphOptimizationReport = GraphOptimizationReportV2


@dataclass(frozen=True)
class GraphOptimizationOutcome:
    """Outcome-first view of selection, recovery and measured evidence."""

    selection: GraphRecipeHandle | None
    pareto_frontier: tuple[GraphRecipeHandle, ...]
    selection_artifact: GraphRecipeSelectionArtifact | None
    report: GraphOptimizationReportV2

    @property
    def status(self):
        return self.report.outcome_status

    @property
    def next_action(self):
        return self.report.next_action

    @property
    def checkpoint(self):
        return self.report.checkpoint

    def to_dict(self):
        return {
            "schema": "taichi_forge.graph_optimization_outcome.v1",
            "status": self.status,
            "next_action": self.next_action,
            "selection": (None if self.selection is None else self.selection.to_dict()),
            "pareto_frontier": tuple(item.to_dict() for item in self.pareto_frontier),
            "selection_artifact": (
                None
                if self.selection_artifact is None
                else self.selection_artifact.to_dict()
            ),
            "report": self.report.to_dict(),
        }


@dataclass(frozen=True)
class GraphOptimizationDecision(GraphOptimizationOutcome):
    """Compatibility name for the outcome returned by ``session.run()``."""


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
        workload_context,
        evaluation_contract,
        backend_environment,
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
        for name, value, expected_type in (
            ("workload_context", workload_context, GraphWorkloadContext),
            ("evaluation_contract", evaluation_contract, GraphEvaluationContract),
            ("backend_environment", backend_environment, GraphBackendEnvironment),
        ):
            if value is not None and not isinstance(value, expected_type):
                raise TypeError(
                    f"Graph recipe search {name} must be {expected_type.__name__}"
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
        self._workload_context = workload_context
        self._evaluation_contract = evaluation_contract
        self._backend_environment = backend_environment
        self._session_nonce = uuid.uuid4().hex
        self._reuse_scope = (
            "portable"
            if all(
                item is not None
                for item in (
                    workload_context,
                    evaluation_contract,
                    backend_environment,
                )
            )
            else "session_only"
        )
        self._workload_context_id = (
            workload_context.workload_context_id
            if workload_context is not None
            else f"session-only-workload:{self._session_nonce}"
        )
        self._evaluation_contract_id = (
            evaluation_contract.evaluation_contract_id
            if evaluation_contract is not None
            else f"session-only-evaluation:{self._session_nonce}"
        )
        self._backend_environment_id = (
            backend_environment.backend_environment_id
            if backend_environment is not None
            else f"session-only-backend:{self._session_nonce}"
        )
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
        self._joint_checkpoint = self._normalize_checkpoint(checkpoint)
        self._checkpoint = None
        if self._joint_checkpoint is not None:
            self._validate_checkpoint_contract(self._joint_checkpoint)
            self._checkpoint = self._joint_checkpoint.compileiq_checkpoint
            self._rebuild_checkpoint_catalog(self._checkpoint)
            self._validate_checkpoint_generation(self._joint_checkpoint)
        graph = definition.compile()
        self._plans = CompileIQCompleteGraphRecipeSearch(
            graph,
            catalog=catalog,
            search_strategy_id=strategy.strategy_id,
        )
        if self._joint_checkpoint is not None and (
            self._joint_checkpoint.contract.get("compileiq_python_source_lock")
            != self._plans.python_source_lock
        ):
            raise ValueError(
                "Graph recipe checkpoint CompileIQ Python source lock drifted"
            )
        self._handles = {}
        self._refresh_handles()
        self._used = False

    @staticmethod
    def _normalize_checkpoint(checkpoint):
        if checkpoint is None:
            return None
        return GraphRecipeSearchCheckpointV1.from_dict(checkpoint)

    def _checkpoint_contract(self, *, capability=None, provenance=None):
        contract = {
            "schema": "taichi_forge.graph_recipe_search_contract.v1",
            "semantic_graph_id": self._definition.semantic_graph_id,
            "backend": self._definition.backend,
            "provider_registry_id": self._catalog.provider_registry_id,
            "generation_domain_id": self._catalog.generation_domain_id,
            "search_strategy_id": self._strategy.strategy_id,
            "target_contract_id": self._target.target_contract_id,
            "workload_context_id": self._workload_context_id,
            "evaluation_contract_id": self._evaluation_contract_id,
            "backend_environment_id": self._backend_environment_id,
            "reuse_scope": self._reuse_scope,
            "forge_compile_provenance": (self._definition.compile_provenance.to_dict()),
        }
        if capability is not None:
            contract["compileiq_capability"] = capability
        if provenance is not None:
            contract["compileiq_provenance"] = provenance
        if hasattr(self, "_plans"):
            contract["compileiq_python_source_lock"] = self._plans.python_source_lock
        return contract

    def _generation_checkpoint(self):
        return {
            "schema": "taichi_forge.graph_recipe_generation_checkpoint.v1",
            "execution_mode": self._execution_mode,
            "seed_fragment_ids": self._seed_fragment_ids,
            "fragments": tuple(
                fragment.to_dict() for fragment in self._catalog.fragments
            ),
            "recipes": tuple(entry.to_dict() for entry in self._catalog.entries()),
            "planned_physical_duplicates": self._catalog.physical_duplicates,
        }

    def _validate_checkpoint_contract(self, checkpoint):
        contract = checkpoint.contract
        if contract.get("reuse_scope") != "portable":
            raise ValueError(
                "session-only Graph recipe checkpoints cannot resume in a new session"
            )
        expected = self._checkpoint_contract()
        fields = (
            "semantic_graph_id",
            "backend",
            "provider_registry_id",
            "generation_domain_id",
            "search_strategy_id",
            "target_contract_id",
            "workload_context_id",
            "evaluation_contract_id",
            "backend_environment_id",
            "reuse_scope",
            "forge_compile_provenance",
        )
        drift = tuple(
            name for name in fields if contract.get(name) != expected.get(name)
        )
        if drift:
            raise ValueError(
                "Graph recipe checkpoint contract drift: " + ", ".join(drift)
            )

    def _validate_checkpoint_generation(self, checkpoint):
        if _canonical_json(checkpoint.generation) != _canonical_json(
            self._generation_checkpoint()
        ):
            raise ValueError(
                "Graph recipe checkpoint provider generation state drifted"
            )

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
            recipe_id: tuple(sorted(parent_ids, key=lambda item: item.encode("utf-8")))
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

    def _selection_artifact(
        self,
        selection,
        selected_result,
        checkpoint,
        *,
        capability,
        provenance,
        status,
    ):
        recipe_manifest = selection._recipe.to_dict()
        recipe_manifest_digest = (
            "graph-recipe-manifest-v1:"
            + hashlib.sha256(
                _canonical_json(recipe_manifest).encode("ascii")
            ).hexdigest()
        )
        compileiq_checkpoint = checkpoint.compileiq_checkpoint
        terminal_fidelity = None
        if compileiq_checkpoint.get("batches"):
            terminal_fidelity = {
                **compileiq_checkpoint["batches"][-1]["fidelity"],
                "fidelity_fingerprint": compileiq_checkpoint["stages"][-1][
                    "fidelity_fingerprint"
                ],
            }
        return GraphRecipeSelectionArtifact.create(
            structure={
                "semantic_graph_id": self._definition.semantic_graph_id,
                "backend": self._definition.backend,
                "provider_registry_id": self._catalog.provider_registry_id,
                "generation_domain_id": self._catalog.generation_domain_id,
                "provider_registry": self._catalog.provider_set.to_dict(),
                "recipe_id": selection.recipe_id,
                "recipe_manifest_digest": recipe_manifest_digest,
                "planned_physical_id": selection.planned_physical_id,
                "materialized_physical_id": selected_result.get(
                    "materialized_physical_id"
                ),
            },
            recipe_manifest=recipe_manifest,
            evidence={
                "reuse_scope": self._reuse_scope,
                "workload_context": (
                    None
                    if self._workload_context is None
                    else self._workload_context.to_dict()
                ),
                "evaluation_contract": (
                    None
                    if self._evaluation_contract is None
                    else self._evaluation_contract.to_dict()
                ),
                "backend_environment": (
                    None
                    if self._backend_environment is None
                    else self._backend_environment.to_dict()
                ),
                "workload_context_id": self._workload_context_id,
                "evaluation_contract_id": self._evaluation_contract_id,
                "backend_environment_id": self._backend_environment_id,
                "target": self._target.to_dict(),
                "target_contract_id": self._target.target_contract_id,
                "terminal_fidelity": terminal_fidelity,
                "search_status": status,
                "compileiq_capability": capability,
                "compileiq_provenance": provenance,
                "compileiq_python_source_lock": self._plans.python_source_lock,
                "forge_compile_provenance": (
                    self._definition.compile_provenance.to_dict()
                ),
                "checkpoint_id": checkpoint.checkpoint_id,
            },
        )

    @staticmethod
    def _outcome_state(selected, status):
        if selected is not None:
            return "selected", "apply_selection"
        if status["terminal_state"] in ("active", "budget_exhausted"):
            return "resumable", "resume_search"
        if status["terminal_state"] in (
            "all_failed",
            "poisoned",
            "provider_failed",
        ):
            return "failed", "review_failures"
        return "no_feasible_candidate", "review_evidence"

    def _selection_reason(self, selected_result, status, *, search_complete):
        if selected_result is not None:
            values = tuple(
                {
                    "metric": name,
                    "direction": direction,
                    "observed_median": selected_result["metrics"][name],
                }
                for name, direction in self._target.objectives
            )
            return {
                "status": "selected",
                "rule": "forge_ordered_objective_lexicographic_v1",
                "ordered_objectives": values,
                "provider_claims_used": False,
                "summary": (
                    "Forge selected the lexicographically best recipe from the "
                    "complete measured Pareto frontier using the caller's objective "
                    "order; recipe ID is the final deterministic tie-breaker."
                ),
            }
        if not search_complete and status["terminal_state"] in (
            "active",
            "budget_exhausted",
        ):
            return {
                "status": "not_selected",
                "rule": "incomplete_evidence_no_selection",
                "ordered_objectives": (),
                "provider_claims_used": False,
                "summary": (
                    "No recipe was selected because terminal evidence is incomplete; "
                    "the checkpoint can be resumed under the same contract."
                ),
            }
        return {
            "status": "not_selected",
            "rule": "no_complete_feasible_frontier",
            "ordered_objectives": (),
            "provider_claims_used": False,
            "summary": (
                "No recipe was selected because the completed evidence contains no "
                "eligible feasible frontier."
            ),
        }

    def _pareto_tradeoffs(self, frontier, selected_result):
        if selected_result is None:
            return ()
        tradeoffs = []
        for candidate in sorted(frontier, key=self._selection_key):
            if candidate["recipe_id"] == selected_result["recipe_id"]:
                continue
            comparisons = []
            for name, direction in self._target.objectives:
                candidate_value = candidate["metrics"][name]
                selected_value = selected_result["metrics"][name]
                if candidate_value == selected_value:
                    relation = "equal"
                elif (direction == "min" and candidate_value < selected_value) or (
                    direction == "max" and candidate_value > selected_value
                ):
                    relation = "better"
                else:
                    relation = "worse"
                comparisons.append(
                    {
                        "metric": name,
                        "direction": direction,
                        "candidate_value": candidate_value,
                        "selected_value": selected_value,
                        "delta": candidate_value - selected_value,
                        "relation": relation,
                    }
                )
            tradeoffs.append(
                {
                    "recipe_id": candidate["recipe_id"],
                    "relative_to_selected": tuple(comparisons),
                }
            )
        return tuple(tradeoffs)

    def _recipe_annotations(self, compileiq_report):
        latest_candidates = {}
        for candidate in sorted(
            compileiq_report.candidates,
            key=lambda item: (item.stage_index, item.candidate_key),
        ):
            latest_candidates[candidate.recipe_id] = candidate
        baseline_manifest = self.baseline.manifest
        annotations = []
        for recipe_id in sorted(latest_candidates):
            handle = self._handles[recipe_id]
            recipe = handle._recipe
            manifest = handle.manifest
            candidate = latest_candidates[recipe_id]
            provider_claims = []
            declared_applicability = []
            declared_limitations = []
            for fragment in recipe.fragments:
                claims = self._catalog.provider_set.describe(fragment)
                if isinstance(claims, dict):
                    applicability = claims.get("applicability")
                    limitations = claims.get("limitations")
                    if applicability is not None:
                        declared_applicability.append(applicability)
                    if limitations is not None:
                        declared_limitations.append(limitations)
                provider_claims.append(
                    {
                        "source": "provider_declared_not_measured",
                        "provider_namespace": fragment.provider_namespace,
                        "fragment_key": fragment.fragment_key,
                        "claims": claims,
                    }
                )
            physical_changes = {
                "planned_identity_changed": (
                    manifest.planned_physical_id
                    != baseline_manifest.planned_physical_id
                ),
                "selected_fragment_delta": (
                    manifest.selected_fragment_count
                    - baseline_manifest.selected_fragment_count
                ),
                "execution_step_delta": (
                    manifest.execution_step_count
                    - baseline_manifest.execution_step_count
                ),
                "queue_count_delta": (
                    manifest.queue_count - baseline_manifest.queue_count
                ),
                "barrier_count_delta": (
                    manifest.barrier_count - baseline_manifest.barrier_count
                ),
                "declared_persistent_bytes_delta": (
                    manifest.declared_persistent_resource_bytes
                    - baseline_manifest.declared_persistent_resource_bytes
                ),
                "declared_transient_bytes_delta": (
                    manifest.declared_transient_resource_bytes
                    - baseline_manifest.declared_transient_resource_bytes
                ),
            }
            annotations.append(
                {
                    "recipe_id": recipe_id,
                    "display_name": (
                        "Baseline"
                        if manifest.is_baseline
                        else " + ".join(manifest.families)
                    ),
                    "manifest": manifest.to_dict(),
                    "semantic_region_ids": tuple(
                        selection.region_id for selection in recipe.region_selections
                    ),
                    "optimized_semantic_region_ids": tuple(
                        dict.fromkeys(
                            region_id
                            for fragment in recipe.fragments
                            for region_id in fragment.coverage_region_ids
                        )
                    ),
                    "physical_changes_from_baseline": physical_changes,
                    "measurement": {
                        "stage_index": candidate.stage_index,
                        "fidelity_name": candidate.fidelity_name,
                        "complete": candidate.complete,
                        "feasible": candidate.feasible,
                        "metrics": {
                            name: summary.model_dump(by_alias=True)
                            for name, summary in candidate.metrics.items()
                        },
                        "planned_physical_ids": candidate.planned_physical_ids,
                        "materialized_physical_ids": (
                            candidate.materialized_physical_ids
                        ),
                        "materialized_memory_peak_bytes": (
                            candidate.materialized_memory_peak_bytes
                        ),
                        "failure_count": len(candidate.failures),
                    },
                    "provider_claims": tuple(provider_claims),
                    "provider_declared_applicability": tuple(
                        declared_applicability
                    ),
                    "provider_declared_limitations": tuple(declared_limitations),
                }
            )
        return tuple(annotations)

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

        from compileiq.forge_support import ForgeOpaqueEvaluationContextV1

        evaluation_context = ForgeOpaqueEvaluationContextV1(
            reuse_scope=self._reuse_scope,
            workload_context_id=self._workload_context_id,
            evaluation_contract_id=self._evaluation_contract_id,
            backend_environment_id=self._backend_environment_id,
        )

        with self._plans.compileiq_search(
            objective,
            budget=self._budget._compileiq_budget(),
            target_contract=self._target._compileiq_contract(),
            deterministic_seed=self._budget.deterministic_seed,
            halving_factor=self._budget.halving_factor,
            minimum_survivors=self._budget.minimum_survivors,
            repeat_count=self._budget.repeat_count,
            evaluation_context=evaluation_context,
            checkpoint=self._checkpoint,
        ) as search:
            result = (
                self._run_exact(search)
                if self._execution_mode == "exact"
                else self._run_staged(search)
            )
            coverage = dict(self._plans.search_coverage(search))
            compileiq_checkpoint = search.checkpoint().as_dict()
            capability = search.opaque_recipe_capability
            provenance = search.opaque_recipe_core_provenance
            status = result.status.model_dump(by_alias=True)
            compileiq_report = result.report(detail="full")

        checkpoint = GraphRecipeSearchCheckpointV1.create(
            contract=self._checkpoint_contract(
                capability=capability,
                provenance=provenance,
            ),
            generation=self._generation_checkpoint(),
            compileiq_checkpoint=compileiq_checkpoint,
        )

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
        selected_result = None
        if coverage["complete"] and frontier:
            selected_result = min(frontier, key=self._selection_key)
            selected = self._handles[selected_result["recipe_id"]]
        frontier_handles = tuple(
            self._handles[item["recipe_id"]]
            for item in sorted(frontier, key=self._selection_key)
        )
        selection_artifact = (
            None
            if selected is None
            else self._selection_artifact(
                selected,
                selected_result,
                checkpoint,
                capability=capability,
                provenance=provenance,
                status=status,
            )
        )
        outcome_status, next_action = self._outcome_state(selected, status)
        report = GraphOptimizationReportV2(
            semantic_graph_id=self._definition.semantic_graph_id,
            target=self._target,
            budget=self._budget,
            strategy=self._strategy,
            outcome_status=outcome_status,
            next_action=next_action,
            selected_recipe_id=(None if selected is None else selected.recipe_id),
            pareto_recipe_ids=tuple(item.recipe_id for item in frontier_handles),
            selection_artifact_id=(
                None
                if selection_artifact is None
                else selection_artifact.artifact_id
            ),
            search_complete=bool(coverage["complete"]),
            termination_reason=result.termination_reason,
            evaluation_count=int(coverage["evaluation_count"]),
            measured_recipe_ids=tuple(coverage["observed_recipe_ids"]),
            missing_recipe_ids=tuple(coverage["missing_recipe_ids"]),
            _selection_reason_json=_canonical_json(
                self._selection_reason(
                    selected_result,
                    status,
                    search_complete=bool(coverage["complete"]),
                )
            ),
            _tradeoffs_json=_canonical_json(
                self._pareto_tradeoffs(frontier, selected_result)
            ),
            _recipe_annotations_json=_canonical_json(
                self._recipe_annotations(compileiq_report)
            ),
            _reuse_json=_canonical_json(
                {
                    "scope": self._reuse_scope,
                    "checkpoint_id": checkpoint.checkpoint_id,
                    "selection_artifact_id": (
                        None
                        if selection_artifact is None
                        else selection_artifact.artifact_id
                    ),
                    "workload_context_id": self._workload_context_id,
                    "evaluation_contract_id": self._evaluation_contract_id,
                    "backend_environment_id": self._backend_environment_id,
                    "resolution": "rebuild_provider_catalog_by_stable_recipe_id",
                }
            ),
            _compileiq_report_json=compileiq_report.summary().to_json(),
            _provenance_json=_canonical_json(provenance),
            _checkpoint_json=_canonical_json(checkpoint.to_dict()),
        )
        return GraphOptimizationDecision(
            selection=selected,
            pareto_frontier=frontier_handles,
            selection_artifact=selection_artifact,
            report=report,
        )


__all__ = [
    "GraphOptimizationDecision",
    "GraphOptimizationOutcome",
    "GraphOptimizationReport",
    "GraphOptimizationReportV2",
    "GraphOptimizationTarget",
    "GraphRecipeHandle",
    "GraphRecipeManifest",
    "GraphRecipeSearchStrategy",
    "GraphSearchBudget",
]
