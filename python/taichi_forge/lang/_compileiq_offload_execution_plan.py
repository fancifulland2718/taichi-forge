"""Opaque CompileIQ search domains for complete CUDA offload plans."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from importlib import import_module
import json
import math
import statistics
from types import MappingProxyType

from taichi_forge._compileiq_opaque import (
    CompileIQOpaqueUnavailableError,
    _CompileIQOpaqueRecipeTransport,
    _validated_compileiq_capability as _validate_shared_compileiq_capability,
)
from taichi_forge.lang._offload_execution_plan import (
    _OffloadExecutionPlan,
    _bind_offload_execution_plan,
)


_PROVIDER_NAMESPACE = "taichi_forge.lang.offload_execution_plan"
_FIRST_STAGE_VERSION = "single-task-perturbation.v1"
_REFINED_STAGE_VERSION = "observed-frontier-pairwise.v1"
_WORKGROUP_SIZES = (64, 128, 256, 512)
_THREAD_LOCAL_MODES = ("on", "off")
_CUDA_MIN_BLOCKS_PER_SM = (1, 4)
_CUDA_MAX_REGISTERS = (24, 48)
_RANGE_WORK_PER_THREAD_TARGETS = (2, 4, 8)
_CUDA_GRID_RESIDENCY_WAVES = (1, 2, 4)
_MAX_FRONTIER_RECIPES = 32
_MIN_QUALIFICATION_BLOCKS = 10


class CompileIQOffloadPlanUnavailableError(CompileIQOpaqueUnavailableError):
    """The installed CompileIQ is not the reviewed opaque-recipe fork."""


def _validated_compileiq_capability():
    return _validate_shared_compileiq_capability(
        importer=import_module,
        error_type=CompileIQOffloadPlanUnavailableError,
    )


def _canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _identity(prefix, value):
    return prefix + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _PlanEdit:
    task_index: int
    field: str
    value: object

    @property
    def sort_key(self):
        return (self.task_index, self.field, _canonical_json(self.value))


@dataclass(frozen=True)
class _PlanRecipe:
    plan: _OffloadExecutionPlan
    edits: tuple[_PlanEdit, ...] = ()
    parent_recipe_ids: tuple[str, ...] = ()

    @property
    def recipe_id(self):
        return self.plan.recipe_id


@dataclass(frozen=True)
class OffloadExecutionPlanRecipeSelection:
    """One Forge-owned complete plan selected through an opaque token."""

    recipe_id: str
    execution_plan_identity: str
    compilation_identity: str
    semantic_kernel_identity: str
    task_count: int
    stage: str
    edits: tuple
    parent_recipe_ids: tuple[str, ...]

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class OffloadExecutionPlanCandidateEvidence:
    recipe_id: str
    ratios: tuple[float, ...]
    median_ratio: float
    worst_ratio: float
    best_ratio: float
    ratio_cv: float

    @property
    def worst_positive(self):
        return self.worst_ratio < 1.0


@dataclass(frozen=True)
class OffloadExecutionPlanQualificationDecision:
    status: str
    reason: str
    selected_recipe_id: str | None
    scope_id: str
    evidence: tuple[OffloadExecutionPlanCandidateEvidence, ...]

    @property
    def admitted(self):
        return self.status == "qualified"


def _paired_schedule(recipe_ids, *, blocks):
    if isinstance(blocks, bool) or not isinstance(blocks, int):
        raise TypeError("blocks must be an integer")
    if blocks < 2 or blocks % 2:
        raise ValueError("blocks must be an even integer >= 2")
    return tuple(
        {
            "recipe_id": recipe_id,
            "block": block,
            "order": (("baseline", "candidate") if block % 2 == 0 else ("candidate", "baseline")),
        }
        for recipe_id in recipe_ids
        for block in range(blocks)
    )


def _rank_paired(measurements, recipe_ids, *, blocks):
    _paired_schedule((), blocks=blocks)
    recipe_ids = tuple(recipe_ids)
    if not isinstance(measurements, dict):
        raise TypeError("measurements must map recipe IDs to paired ratios")
    if set(measurements) != set(recipe_ids):
        missing = tuple(item for item in recipe_ids if item not in measurements)
        extra = tuple(item for item in measurements if item not in recipe_ids)
        raise ValueError("paired measurements do not match the finalist domain; " f"missing={missing}, extra={extra}")
    evidence = []
    for recipe_id in recipe_ids:
        ratios = tuple(measurements[recipe_id])
        if len(ratios) != blocks:
            raise ValueError(f"recipe {recipe_id!r} requires exactly {blocks} paired ratios")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in ratios
        ):
            raise ValueError("paired ratios must be finite positive numbers")
        ratios = tuple(float(value) for value in ratios)
        mean = statistics.fmean(ratios)
        ratio_cv = 0.0 if mean == 0.0 else statistics.pstdev(ratios) / mean
        evidence.append(
            OffloadExecutionPlanCandidateEvidence(
                recipe_id=recipe_id,
                ratios=ratios,
                median_ratio=statistics.median(ratios),
                worst_ratio=max(ratios),
                best_ratio=min(ratios),
                ratio_cv=ratio_cv,
            )
        )
    return tuple(
        sorted(
            evidence,
            key=lambda item: (
                item.worst_ratio,
                item.median_ratio,
                item.ratio_cv,
                item.recipe_id,
            ),
        )
    )


class CompileIQOffloadExecutionPlanSearch:
    """A frozen domain of complete task-indexed plans for one kernel.

    CompileIQ sees only opaque recipe tokens. Forge owns candidate generation,
    task legality, exact materialization, staged refinement, and qualification.
    """

    __slots__ = (
        "_baseline_plan",
        "_capability_components",
        "_eligibility",
        "_kernel",
        "_recipes",
        "_sample_args",
        "_semantic_fingerprint",
        "_stage",
        "_stage_manifest",
        "_transport",
    )

    def __init__(self, kernel, *sample_args):
        capability_components = _validated_compileiq_capability()
        manifests = tuple(kernel.task_manifest(*sample_args))
        self._validate_baseline_manifests(manifests)
        baseline = _OffloadExecutionPlan.from_task_manifests(manifests)
        recipes, eligibility = self._first_stage_recipes(baseline, manifests)
        stage_manifest = {
            "generation": "baseline_plus_all_legal_single_task_perturbations",
            "candidate_limit": 4096,
            "axes": {
                "workgroup_size": _WORKGROUP_SIZES,
                "thread_local": ("auto", *_THREAD_LOCAL_MODES),
                "cuda_min_blocks_per_sm": (2, *_CUDA_MIN_BLOCKS_PER_SM),
                "cuda_max_registers": (None, *_CUDA_MAX_REGISTERS),
                "range_work_per_thread_target": (
                    1,
                    *_RANGE_WORK_PER_THREAD_TARGETS,
                ),
                "grid_residency_waves": (
                    None,
                    *_CUDA_GRID_RESIDENCY_WAVES,
                ),
            },
        }
        self._initialize(
            kernel=kernel,
            sample_args=sample_args,
            baseline_plan=baseline,
            recipes=recipes,
            eligibility=eligibility,
            stage="single_task_perturbation",
            stage_manifest=stage_manifest,
            capability_components=capability_components,
        )

    @staticmethod
    def _validate_baseline_manifests(manifests):
        if not manifests:
            raise ValueError("offload plan search requires at least one physical task")
        if any(task.backend != "cuda" for task in manifests):
            raise ValueError("offload plan search requires a native CUDA kernel")
        if tuple(task.task_index for task in manifests) != tuple(range(len(manifests))):
            raise ValueError("baseline task manifests are not in physical order")

    @classmethod
    def _first_stage_recipes(cls, baseline, manifests):
        recipes = {baseline.recipe_id: _PlanRecipe(baseline)}
        eligibility = []
        for spec, manifest in zip(baseline.tasks, manifests):
            axis_status = {}
            if spec.task_kind != "range_for":
                axis_status["all"] = "non-range physical task retained at baseline"
                eligibility.append((spec.logical_task_id, axis_status))
                continue

            dense_grid_stride = manifest.range_mapping == "grid_stride"
            no_shared = manifest.static_shared_bytes == 0 and manifest.dynamic_shared_bytes == 0
            if manifest.source_block_size_explicit:
                axis_status["workgroup_size"] = "source-owned block contract"
            elif not dense_grid_stride:
                axis_status["workgroup_size"] = "range mapping is not grid-stride"
            elif not no_shared:
                axis_status["workgroup_size"] = "shared-memory task retains its materialized block contract"
            else:
                legal = tuple(value for value in _WORKGROUP_SIZES if value != manifest.selected_block_size)
                cls._add_single_axis(recipes, baseline, spec.task_index, "workgroup_size", legal)
                axis_status["workgroup_size"] = f"{len(legal)} perturbations"

            if manifest.thread_local_bytes > 0:
                cls._add_single_axis(
                    recipes,
                    baseline,
                    spec.task_index,
                    "thread_local",
                    _THREAD_LOCAL_MODES,
                )
                axis_status["thread_local"] = "manifest-proven TLS reduction"
            else:
                axis_status["thread_local"] = "no manifest-proven TLS reduction"

            cls._add_single_axis(
                recipes,
                baseline,
                spec.task_index,
                "cuda_min_blocks_per_sm",
                _CUDA_MIN_BLOCKS_PER_SM,
            )
            cls._add_single_axis(
                recipes,
                baseline,
                spec.task_index,
                "cuda_max_registers",
                _CUDA_MAX_REGISTERS,
            )
            axis_status["cuda_min_blocks_per_sm"] = "eligible CUDA range task"
            axis_status["cuda_max_registers"] = "eligible CUDA range task"

            if dense_grid_stride and manifest.constant_range_size is not None:
                cls._add_single_axis(
                    recipes,
                    baseline,
                    spec.task_index,
                    "range_work_per_thread_target",
                    _RANGE_WORK_PER_THREAD_TARGETS,
                )
                axis_status["range_work_per_thread_target"] = "constant grid-stride range"
            else:
                axis_status["range_work_per_thread_target"] = "requires a constant grid-stride range"
            if dense_grid_stride:
                cls._add_single_axis(
                    recipes,
                    baseline,
                    spec.task_index,
                    "grid_residency_waves",
                    _CUDA_GRID_RESIDENCY_WAVES,
                )
                axis_status["grid_residency_waves"] = "dense grid-stride range"
            else:
                axis_status["grid_residency_waves"] = "requires a dense grid-stride range"
            eligibility.append((spec.logical_task_id, axis_status))

        if len(recipes) > 4096:
            raise ValueError(
                "single-task plan domain exceeds 4096 recipes; staged generation " "must be narrowed explicitly"
            )
        return recipes, tuple(eligibility)

    @staticmethod
    def _add_single_axis(recipes, baseline, task_index, field, values):
        for value in values:
            plan = baseline.replace_task(task_index, **{field: value})
            edit = _PlanEdit(task_index, field, value)
            recipes.setdefault(plan.recipe_id, _PlanRecipe(plan, (edit,)))

    def _initialize(
        self,
        *,
        kernel,
        sample_args,
        baseline_plan,
        recipes,
        eligibility,
        stage,
        stage_manifest,
        capability_components,
    ):
        if baseline_plan.recipe_id not in recipes:
            raise ValueError("complete plan domain must retain its baseline sentinel")
        if len(recipes) > 4096:
            raise ValueError("complete plan domain exceeds the 4096-recipe limit")
        semantic_payload = {
            "schema": "taichi_forge.compileiq-offload-plan-semantics.v1",
            "semantic_kernel_identity": baseline_plan.semantic_kernel_identity,
            "stage": stage,
            "stage_manifest": stage_manifest,
            "recipes": tuple(
                {
                    "recipe_id": recipe.recipe_id,
                    "plan": recipe.plan.stable_payload,
                    "edits": tuple(asdict(edit) for edit in recipe.edits),
                    "parent_recipe_ids": recipe.parent_recipe_ids,
                }
                for recipe in sorted(recipes.values(), key=lambda item: item.recipe_id)
            ),
        }
        semantic_fingerprint = _identity("forge-offload-plan-semantics-v1:", semantic_payload)
        domain_version = _FIRST_STAGE_VERSION if stage == "single_task_perturbation" else _REFINED_STAGE_VERSION
        transport = _CompileIQOpaqueRecipeTransport(
            provider_namespace=_PROVIDER_NAMESPACE,
            domain_version=domain_version,
            provider_semantic_fingerprint=semantic_fingerprint,
            recipe_ids=tuple(recipes),
            baseline_recipe_id=baseline_plan.recipe_id,
            capability_components=capability_components,
            domain_owner="offload execution-plan",
            recipe_description="complete offload execution plan",
        )
        self._kernel = kernel
        self._sample_args = tuple(sample_args)
        self._baseline_plan = baseline_plan
        self._recipes = MappingProxyType(dict(recipes))
        self._eligibility = tuple(eligibility)
        self._stage = stage
        self._stage_manifest = MappingProxyType(dict(stage_manifest))
        self._capability_components = capability_components
        self._semantic_fingerprint = semantic_fingerprint
        self._transport = transport

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
    def recipe_ids(self):
        return self._transport.recipe_ids

    @property
    def baseline_recipe_id(self):
        return self._baseline_plan.recipe_id

    @property
    def domain_fingerprint(self):
        return self._transport.domain_fingerprint

    @property
    def semantic_fingerprint(self):
        return self._semantic_fingerprint

    @property
    def stage(self):
        return self._stage

    def _recipe(self, recipe_id):
        try:
            return self._recipes[recipe_id]
        except KeyError as error:
            raise KeyError(f"unknown complete offload execution plan {recipe_id!r}") from error

    def _decoded_recipe_id(self, parameters):
        return self._transport.decode(parameters)

    def select(self, parameters):
        recipe = self._recipe(self._decoded_recipe_id(parameters))
        return OffloadExecutionPlanRecipeSelection(
            recipe_id=recipe.recipe_id,
            execution_plan_identity=recipe.plan.identity,
            compilation_identity=recipe.plan.compilation_identity,
            semantic_kernel_identity=recipe.plan.semantic_kernel_identity,
            task_count=len(recipe.plan.tasks),
            stage=self.stage,
            edits=tuple(asdict(edit) for edit in recipe.edits),
            parent_recipe_ids=recipe.parent_recipe_ids,
        )

    def bind(self, parameters):
        recipe = self._recipe(self._decoded_recipe_id(parameters))
        return _bind_offload_execution_plan(self._kernel, recipe.plan)

    def materialize(self, parameters, *args):
        bound = self.bind(parameters)
        return bound.report(*(args or self._sample_args))

    def objective(self, parameters, *, reset_workload, measure):
        if not callable(reset_workload) or not callable(measure):
            raise TypeError("reset_workload and measure must be callable")
        reset_workload()
        return measure(self.bind(parameters))

    def compileiq_search(self, objective_function, *, problem_type="min"):
        """Create a complete, deterministic search in the reviewed fork."""

        return self._transport.exhaustive_search(objective_function, problem_type=problem_type)

    def search_coverage(self, compileiq_search):
        return self._transport.search_coverage(compileiq_search)

    def require_complete_search(self, compileiq_search):
        return self._transport.require_complete_search(compileiq_search)

    def select_best_result(self, compileiq_search, result):
        recipe_id = self._transport.select_best_recipe_id(compileiq_search, result)
        return self.select(
            {
                "domain_fingerprint": self.domain_fingerprint,
                "recipe_id": recipe_id,
            }
        )

    def recipe_manifest(self, recipe_id):
        recipe = self._recipe(recipe_id)
        return MappingProxyType(
            {
                "recipe_id": recipe.recipe_id,
                "is_baseline": recipe.recipe_id == self.baseline_recipe_id,
                "execution_plan_identity": recipe.plan.identity,
                "compilation_identity": recipe.plan.compilation_identity,
                "semantic_kernel_identity": recipe.plan.semantic_kernel_identity,
                "tasks": tuple(asdict(task) for task in recipe.plan.tasks),
                "edits": tuple(asdict(edit) for edit in recipe.edits),
                "parent_recipe_ids": recipe.parent_recipe_ids,
            }
        )

    def refine(self, compileiq_search, frontier_recipe_ids):
        self.require_complete_search(compileiq_search)
        if self.stage != "single_task_perturbation":
            raise RuntimeError("only a first-stage plan domain can be refined")
        frontier_recipe_ids = tuple(dict.fromkeys(frontier_recipe_ids))
        frontier_recipe_ids = tuple(
            recipe_id for recipe_id in frontier_recipe_ids if recipe_id != self.baseline_recipe_id
        )
        if not frontier_recipe_ids:
            raise ValueError("refinement requires at least one measured nonbaseline plan")
        if len(frontier_recipe_ids) > _MAX_FRONTIER_RECIPES:
            raise ValueError(f"refinement frontier exceeds {_MAX_FRONTIER_RECIPES} recipes")
        frontier = tuple(self._recipe(recipe_id) for recipe_id in frontier_recipe_ids)
        if any(len(recipe.edits) != 1 for recipe in frontier):
            raise ValueError("refinement frontier must contain first-stage perturbations")

        recipes = {
            self.baseline_recipe_id: _PlanRecipe(self._baseline_plan),
            **{recipe.recipe_id: recipe for recipe in frontier},
        }
        for left_index, left in enumerate(frontier):
            for right in frontier[left_index + 1 :]:
                edits = tuple(sorted((*left.edits, *right.edits), key=lambda item: item.sort_key))
                fields = tuple((edit.task_index, edit.field) for edit in edits)
                if len(set(fields)) != len(fields):
                    continue
                plan = self._baseline_plan
                for edit in edits:
                    plan = plan.replace_task(edit.task_index, **{edit.field: edit.value})
                recipes.setdefault(
                    plan.recipe_id,
                    _PlanRecipe(
                        plan=plan,
                        edits=edits,
                        parent_recipe_ids=tuple(sorted((left.recipe_id, right.recipe_id))),
                    ),
                )
        if len(recipes) > 4096:
            raise ValueError("observed-frontier pairwise domain exceeds 4096 recipes")

        refined = object.__new__(type(self))
        refined._initialize(
            kernel=self._kernel,
            sample_args=self._sample_args,
            baseline_plan=self._baseline_plan,
            recipes=recipes,
            eligibility=self._eligibility,
            stage="observed_frontier_pairwise",
            stage_manifest={
                "generation": "baseline_plus_observed_frontier_plus_all_legal_pairs",
                "parent_domain_fingerprint": self.domain_fingerprint,
                "frontier_recipe_ids": tuple(sorted(frontier_recipe_ids)),
                "frontier_limit": _MAX_FRONTIER_RECIPES,
                "candidate_limit": 4096,
                "overflow": "fail_closed_without_truncation",
            },
            capability_components=self._capability_components,
        )
        return refined

    def paired_schedule(self, recipe_ids=None, *, blocks=2):
        if recipe_ids is None:
            recipe_ids = tuple(recipe_id for recipe_id in self.recipe_ids if recipe_id != self.baseline_recipe_id)
        else:
            recipe_ids = tuple(recipe_ids)
            for recipe_id in recipe_ids:
                self._recipe(recipe_id)
        return _paired_schedule(recipe_ids, blocks=blocks)

    def rank_paired(self, measurements, recipe_ids=None, *, blocks=2):
        if recipe_ids is None:
            recipe_ids = tuple(recipe_id for recipe_id in self.recipe_ids if recipe_id != self.baseline_recipe_id)
        else:
            recipe_ids = tuple(recipe_ids)
            for recipe_id in recipe_ids:
                self._recipe(recipe_id)
        return _rank_paired(measurements, recipe_ids, blocks=blocks)

    def qualify(
        self,
        measurements,
        finalist_recipe_ids,
        *,
        correctness,
        memory_stable,
        scope,
        blocks=_MIN_QUALIFICATION_BLOCKS,
    ):
        if blocks < _MIN_QUALIFICATION_BLOCKS:
            raise ValueError("independent qualification requires at least 10 blocks")
        finalist_recipe_ids = tuple(finalist_recipe_ids)
        if not finalist_recipe_ids:
            raise ValueError("qualification requires at least one finalist")
        for recipe_id in finalist_recipe_ids:
            self._recipe(recipe_id)
        if set(correctness) != set(finalist_recipe_ids) or set(memory_stable) != set(finalist_recipe_ids):
            raise ValueError("correctness and memory evidence must cover every exact finalist")
        if not isinstance(scope, dict) or not scope:
            raise ValueError("qualification requires a nonempty exact scope")
        ranked = _rank_paired(measurements, finalist_recipe_ids, blocks=blocks)
        eligible = tuple(
            item
            for item in ranked
            if correctness[item.recipe_id] and memory_stable[item.recipe_id] and item.worst_positive
        )
        scope_id = _identity(
            "forge-offload-plan-qualification-scope-v1:",
            {
                "domain_fingerprint": self.domain_fingerprint,
                "scope": scope,
                "finalists": finalist_recipe_ids,
            },
        )
        if not eligible:
            return OffloadExecutionPlanQualificationDecision(
                status="baseline_retained",
                reason=("no exact finalist passed correctness, memory, and " "worst-positive runtime gates"),
                selected_recipe_id=None,
                scope_id=scope_id,
                evidence=ranked,
            )
        selected = eligible[0]
        return OffloadExecutionPlanQualificationDecision(
            status="qualified",
            reason=("exact complete plan passed independent correctness, memory, " "and worst-positive runtime gates"),
            selected_recipe_id=selected.recipe_id,
            scope_id=scope_id,
            evidence=ranked,
        )

    def manifest(self):
        return {
            "schema": "taichi_forge.compileiq-offload-plan-search.v1",
            **self._transport.manifest(),
            "stage": self.stage,
            "stage_manifest": dict(self._stage_manifest),
            "semantic_fingerprint": self.semantic_fingerprint,
            "semantic_kernel_identity": self._baseline_plan.semantic_kernel_identity,
            "eligibility": tuple(
                {
                    "logical_task_id": logical_task_id,
                    "axes": dict(axes),
                }
                for logical_task_id, axes in self._eligibility
            ),
            "recipes": tuple(dict(self.recipe_manifest(recipe_id)) for recipe_id in self.recipe_ids),
            "compileiq_visibility": "opaque_complete_recipe_tokens_only",
            "qualification": "independent_forge_worst_positive_v1",
            "runtime_admission": "explicit_qualified_scope_only",
            "compile_time": "diagnostic_only_not_a_gate",
        }


def compileiq_offload_execution_plan_search(kernel, *sample_args):
    """Build a baseline-inclusive complete-plan domain for modified CompileIQ."""

    return CompileIQOffloadExecutionPlanSearch(kernel, *sample_args)


__all__ = [
    "CompileIQOffloadExecutionPlanSearch",
    "CompileIQOffloadPlanUnavailableError",
    "OffloadExecutionPlanCandidateEvidence",
    "OffloadExecutionPlanQualificationDecision",
    "OffloadExecutionPlanRecipeSelection",
    "compileiq_offload_execution_plan_search",
]
