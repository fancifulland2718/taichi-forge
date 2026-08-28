"""Private executable-plan optimization identities for Forge Graphs."""

from dataclasses import dataclass
import hashlib
import json

from taichi_forge.graph._ir import graph_ir_to_dict


def _canonical_hash(value):
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _ExecutableOptimizationSpec:
    spec_id: str
    semantic_plan_id: str
    backend: str
    fusion_recipe_ids: tuple
    compilation_identity: str
    execution_identity: str

    def __post_init__(self):
        if not self.spec_id.startswith("executable:"):
            raise ValueError("executable optimization spec ID is invalid")
        if not self.semantic_plan_id.startswith("semantic-plan:"):
            raise ValueError("semantic plan ID is invalid")
        if not isinstance(self.backend, str) or not self.backend:
            raise ValueError("executable optimization backend is invalid")
        if len(set(self.fusion_recipe_ids)) != len(self.fusion_recipe_ids):
            raise ValueError("fusion recipe IDs must be unique")
        if not self.compilation_identity or not self.execution_identity:
            raise ValueError("executable optimization identities are required")

    def to_dict(self):
        return {
            "schema_version": 1,
            "spec_id": self.spec_id,
            "semantic_plan_id": self.semantic_plan_id,
            "backend": self.backend,
            "fusion_recipe_ids": self.fusion_recipe_ids,
            "compilation_identity": self.compilation_identity,
            "execution_identity": self.execution_identity,
        }


@dataclass(frozen=True)
class _ExecutableOptimizationSpace:
    semantic_plan_id: str
    baseline: _ExecutableOptimizationSpec
    candidates: tuple
    selected_spec_id: object
    selection_status: str

    @property
    def selected(self):
        for spec in (self.baseline, *self.candidates):
            if spec.spec_id == self.selected_spec_id:
                return spec
        return None

    def to_dict(self):
        return {
            "schema_version": 1,
            "semantic_plan_id": self.semantic_plan_id,
            "baseline": self.baseline.to_dict(),
            "candidates": tuple(spec.to_dict() for spec in self.candidates),
            "selected_spec_id": self.selected_spec_id,
            "selected": None if self.selected is None else self.selected.to_dict(),
            "selection_status": self.selection_status,
        }


def _make_spec(semantic_plan_id, backend, fusion_recipe_ids):
    fusion_recipe_ids = tuple(fusion_recipe_ids)
    compilation_identity = _canonical_hash(
        {
            "semantic_plan_id": semantic_plan_id,
            "backend": backend,
            "fusion_recipe_ids": fusion_recipe_ids,
        }
    )
    execution_identity = _canonical_hash(
        {
            "compilation_identity": compilation_identity,
            "physical_dispatch_delta": -len(fusion_recipe_ids),
        }
    )
    return _ExecutableOptimizationSpec(
        spec_id=f"executable:{compilation_identity[:24]}",
        semantic_plan_id=semantic_plan_id,
        backend=backend,
        fusion_recipe_ids=fusion_recipe_ids,
        compilation_identity=compilation_identity,
        execution_identity=execution_identity,
    )


def _build_executable_optimization_space(root, fusion_plan, backend):
    semantic_digest = _canonical_hash(graph_ir_to_dict(root))
    semantic_plan_id = f"semantic-plan:{semantic_digest[:24]}"
    baseline = _make_spec(semantic_plan_id, backend, ())
    recipe_ids = tuple(
        recipe.recipe_id for recipe in fusion_plan.candidate_recipes
    )
    candidates = (
        (_make_spec(semantic_plan_id, backend, recipe_ids),)
        if recipe_ids
        else ()
    )
    if fusion_plan.applied_groups == 0:
        selected_spec_id = baseline.spec_id
        selection_status = "selected_baseline"
    elif candidates and fusion_plan.applied_groups == len(recipe_ids):
        selected_spec_id = candidates[0].spec_id
        selection_status = "selected_greedy_pair"
    else:
        selected_spec_id = None
        selection_status = "applied_group_count_mismatch"
    return _ExecutableOptimizationSpace(
        semantic_plan_id=semantic_plan_id,
        baseline=baseline,
        candidates=candidates,
        selected_spec_id=selected_spec_id,
        selection_status=selection_status,
    )


__all__ = [
    "_ExecutableOptimizationSpace",
    "_ExecutableOptimizationSpec",
    "_build_executable_optimization_space",
]
