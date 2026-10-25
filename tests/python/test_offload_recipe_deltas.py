"""Search generation must not retain a full task topology per candidate."""
from types import SimpleNamespace

from taichi_forge.lang._compileiq_offload_execution_plan import (
    CompileIQOffloadExecutionPlanSearch, _PlanEdit, _PlanRecipe,
)
from taichi_forge.lang._offload_execution_plan import _OffloadExecutionPlan, _TaskOptimizationSpec


def baseline_and_manifests(count):
    baseline = _OffloadExecutionPlan("example", tuple(
        _TaskOptimizationSpec(logical_task_id=f"tfl:example:{i}:range_for",
                              task_index=i, task_kind="range_for") for i in range(count)))
    manifests = tuple(SimpleNamespace(
        backend="cuda", task_type="range_for", task_index=i,
        logical_task_id=f"tfl:example:{i}:range_for", range_mapping="grid_stride",
        static_shared_bytes=0, dynamic_shared_bytes=0, source_block_size_explicit=False,
        selected_block_size=128, range_bounds=None, thread_local_bytes=0,
        constant_range_size=65, device_range_dispatch_version=0,
    ) for i in range(count))
    return baseline, manifests


def test_generation_retains_only_baseline_and_immutable_edits(monkeypatch):
    baseline, manifests = baseline_and_manifests(32)
    def forbidden(*args, **kwargs):
        raise AssertionError("generation must not materialize full plans")
    with monkeypatch.context() as patch:
        patch.setattr(_OffloadExecutionPlan, "replace_task", forbidden)
        recipes, eligibility = CompileIQOffloadExecutionPlanSearch._first_stage_recipes(baseline, manifests)
        assert len(recipes) > 300
        assert len(eligibility) == 32
        assert all(recipe.baseline is baseline for recipe in recipes.values())
        assert all(len(recipe.edits) <= 1 for recipe in recipes.values())
    choice = next(recipe for recipe in recipes.values() if recipe.edits)
    before = baseline.identity
    materialized = choice.plan
    edit = choice.edits[0]
    assert materialized == baseline.replace_task(edit.task_index, **{edit.field: edit.value})
    assert baseline.identity == before
    assert choice.plan is not materialized, "inspection must not accumulate full plans"


def test_delta_identity_is_ordered_and_materializes_exact_combination():
    baseline, _ = baseline_and_manifests(3)
    edits = (_PlanEdit(0, "workgroup_size", 64), _PlanEdit(2, "grid_residency_waves", 1))
    recipe = _PlanRecipe(baseline, edits, ("parent-a", "parent-b"))
    expected = baseline.replace_task(0, workgroup_size=64).replace_task(2, grid_residency_waves=1)
    assert recipe.plan.identity == expected.identity
    assert recipe.plan.compilation_identity == expected.compilation_identity
    assert recipe.recipe_id == _PlanRecipe(baseline, edits).recipe_id
    assert recipe.recipe_id != _PlanRecipe(baseline, edits[:1]).recipe_id
    assert _PlanRecipe(baseline).recipe_id == baseline.recipe_id
