from types import SimpleNamespace

import numpy as np

import taichi_forge as ti
from taichi_forge.lang._compileiq_offload_execution_plan import (
    compileiq_offload_execution_plan_search,
)
from tests import test_utils


def _parameters(search, recipe_id):
    return {
        "domain_fingerprint": search.domain_fingerprint,
        "recipe_id": recipe_id,
    }


def _find_single_edit(search, *, task_index, field, value=None):
    for recipe_id in search.recipe_ids:
        edits = search.recipe_manifest(recipe_id)["edits"]
        if len(edits) != 1:
            continue
        edit = edits[0]
        if edit["task_index"] == task_index and edit["field"] == field and (value is None or edit["value"] == value):
            return recipe_id, edit["value"]
    raise AssertionError(f"no {field} recipe for task {task_index}")


def _complete_compileiq_audit(search, *, omit=()):
    omitted = set(omit)
    domain = search.search_space
    records = []
    for ordinal, recipe_id in enumerate(search.recipe_ids):
        if recipe_id in omitted:
            continue
        records.append(
            {
                "param_id": ordinal + 1,
                "schema": "compileiq.opaque-recipe-selection.v1",
                "provider_namespace": domain.provider_namespace,
                "domain_version": domain.domain_version,
                "provider_semantic_fingerprint": (domain.provider_semantic_fingerprint),
                "compileiq_capability_id": domain.compileiq_capability_id,
                "compileiq_core_commit": domain.compileiq_core_commit,
                "compileiq_core_lock": domain.compileiq_core_lock,
                "domain_fingerprint": domain.domain_fingerprint,
                "core_recipe_token": f"ciq-recipe-v1-{ordinal:04d}",
                "recipe_id": recipe_id,
            }
        )
    return SimpleNamespace(
        opaque_recipe_capability=dict(search.capability),
        opaque_recipe_core_provenance={
            "core_commit": search.capability["core_commit"],
            "core_lock": search.capability["core_lock"],
        },
        opaque_recipe_audit_records=tuple(records),
    )


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_compileiq_plan_domain_is_complete_opaque_and_baseline_inclusive():
    count = 1 << 14
    values = ti.ndarray(ti.i32, shape=count)
    stamp = ti.field(ti.i32, shape=())

    @ti.kernel
    def mixed(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        stamp[None] = 7
        for i in range(count):
            out[i] = i * 2
        for i in range(count):
            out[i] = out[i] * 3 + stamp[None]

    search = compileiq_offload_execution_plan_search(mixed, values)
    manifest = search.manifest()
    baseline = search.recipe_manifest(search.baseline_recipe_id)
    task_count = len(baseline["tasks"])

    assert search.capability["package_version"] == (
        "1.0.0dev6+taichiforge.report1"
    )
    assert search.baseline_recipe_id in search.recipe_ids
    assert 3 <= len(search.recipe_ids) <= 4096
    assert all(len(search.recipe_manifest(recipe_id)["tasks"]) == task_count for recipe_id in search.recipe_ids)
    assert any(task["task_kind"] == "serial" for task in baseline["tasks"])
    assert all(
        task == baseline["tasks"][task_index]
        for recipe_id in search.recipe_ids
        for task_index, task in enumerate(search.recipe_manifest(recipe_id)["tasks"])
        if task["task_kind"] != "range_for"
    )
    core_space = search.search_space.to_search_space()
    assert set(core_space) == {"domain_fingerprint", "recipe_id"}
    assert all(token.startswith("ciq-recipe-v1-") for token in core_space["recipe_id"].vals)
    assert not set(core_space["recipe_id"].vals) & set(search.recipe_ids)
    assert manifest["compileiq_visibility"] == ("opaque_complete_recipe_tokens_only")
    assert manifest["compile_time"] == "diagnostic_only_not_a_gate"

    first_range = next(task["task_index"] for task in baseline["tasks"] if task["task_kind"] == "range_for")
    recipe_id, block = _find_single_edit(search, task_index=first_range, field="workgroup_size")
    report = search.materialize(_parameters(search, recipe_id), values)
    assert report.tasks[first_range].selected_block_size == block
    search.bind(_parameters(search, recipe_id))(values)
    ti.sync()
    np.testing.assert_array_equal(values.to_numpy(), np.arange(count, dtype=np.int32) * 6 + 7)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_observed_frontier_builds_and_materializes_exact_pairwise_plan():
    count = 1 << 14
    values = ti.ndarray(ti.i32, shape=count)

    @ti.kernel
    def two_stage(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(count):
            out[i] = i + 3
        for i in range(count):
            out[i] *= 5

    first = compileiq_offload_execution_plan_search(two_stage, values)
    baseline = first.recipe_manifest(first.baseline_recipe_id)
    ranges = tuple(task["task_index"] for task in baseline["tasks"] if task["task_kind"] == "range_for")
    workgroup_id, block = _find_single_edit(first, task_index=ranges[0], field="workgroup_size")
    work_id, target = _find_single_edit(
        first,
        task_index=ranges[1],
        field="range_work_per_thread_target",
        value=4,
    )
    audited = _complete_compileiq_audit(first)
    refined = first.refine(audited, (workgroup_id, work_id))
    combined_id = next(
        recipe_id
        for recipe_id in refined.recipe_ids
        if set(refined.recipe_manifest(recipe_id)["parent_recipe_ids"]) == {workgroup_id, work_id}
    )
    report = refined.materialize(_parameters(refined, combined_id), values)

    assert report.tasks[ranges[0]].selected_block_size == block
    assert report.tasks[ranges[1]].requested_range_work_per_thread_target == target
    assert refined.manifest()["stage_manifest"]["parent_domain_fingerprint"] == (first.domain_fingerprint)
    refined.bind(_parameters(refined, combined_id))(values)
    ti.sync()
    np.testing.assert_array_equal(values.to_numpy(), (np.arange(count, dtype=np.int32) + 3) * 5)

    incomplete = _complete_compileiq_audit(first, omit=(work_id,))
    try:
        first.refine(incomplete, (workgroup_id, work_id))
    except RuntimeError as error:
        assert "complete frozen" in str(error)
    else:
        raise AssertionError("an incomplete CompileIQ search was refined")


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_tls_axis_is_opened_only_for_manifest_proven_reduction_task():
    count = 1 << 14
    values = ti.ndarray(ti.i32, shape=count)
    total = ti.field(ti.i32, shape=())

    @ti.kernel
    def fill_then_reduce(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(count):
            out[i] = i % 13
        for i in range(count):
            total[None] += out[i]

    search = compileiq_offload_execution_plan_search(fill_then_reduce, values)
    baseline = search.recipe_manifest(search.baseline_recipe_id)
    ranges = tuple(task["task_index"] for task in baseline["tasks"] if task["task_kind"] == "range_for")
    tls_recipes = tuple(
        (recipe_id, search.recipe_manifest(recipe_id)["edits"][0])
        for recipe_id in search.recipe_ids
        if len(search.recipe_manifest(recipe_id)["edits"]) == 1
        and search.recipe_manifest(recipe_id)["edits"][0]["field"] == "thread_local"
    )

    assert {edit["task_index"] for _, edit in tls_recipes} == {ranges[1]}
    assert {edit["value"] for _, edit in tls_recipes} == {"on", "off"}
    off_id = next(recipe_id for recipe_id, edit in tls_recipes if edit["value"] == "off")
    report = search.materialize(_parameters(search, off_id), values)
    assert report.tasks[ranges[0]].requested_thread_local_mode == "auto"
    assert report.tasks[ranges[1]].requested_thread_local_mode == "off"
    total[None] = 0
    search.bind(_parameters(search, off_id))(values)
    ti.sync()
    assert total[None] == int(np.sum(np.arange(count, dtype=np.int64) % 13))


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_plan_qualification_keeps_negative_evidence_without_admission():
    values = ti.ndarray(ti.i32, shape=1024)

    @ti.kernel
    def fill(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in out:
            out[i] = i

    search = compileiq_offload_execution_plan_search(fill, values)
    candidate = next(recipe_id for recipe_id in search.recipe_ids if recipe_id != search.baseline_recipe_id)
    scope = {
        "workload": "qualification-contract",
        "device": "test-device",
        "runtime": "test-runtime",
    }
    common = {
        "finalist_recipe_ids": (candidate,),
        "correctness": {candidate: True},
        "memory_stable": {candidate: True},
        "scope": scope,
        "blocks": 10,
    }
    negative = search.qualify(
        {candidate: (1.02,) * 10},
        **common,
    )
    positive = search.qualify(
        {candidate: (0.97,) * 10},
        **common,
    )

    assert negative.status == "baseline_retained"
    assert negative.selected_recipe_id is None
    assert negative.evidence[0].worst_positive is False
    assert positive.status == "qualified"
    assert positive.selected_recipe_id == candidate
    assert positive.scope_id == negative.scope_id


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_modified_compileiq_exhausts_and_materializes_full_plan_domain():
    count = 1 << 12
    values = ti.ndarray(ti.i32, shape=count)

    @ti.kernel
    def fill(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(count):
            out[i] = i * 7 + 3

    plans = compileiq_offload_execution_plan_search(fill, values)
    resets = []

    def reset_workload():
        resets.append(len(resets))

    def measure(bound):
        bound(values)
        ti.sync()
        return float(plans.recipe_ids.index(bound.plan.recipe_id))

    search = plans.compileiq_search(
        lambda parameters: plans.objective(
            parameters,
            reset_workload=reset_workload,
            measure=measure,
        )
    )
    result = search.start()
    coverage = plans.require_complete_search(search)
    selected = plans.select_best_result(search, result)

    assert coverage["complete"] is True
    assert coverage["baseline_observed"] is True
    assert coverage["evaluation_count"] == len(plans.recipe_ids)
    assert len(resets) == len(plans.recipe_ids)
    assert selected.recipe_id == plans.recipe_ids[0]
    np.testing.assert_array_equal(values.to_numpy(), np.arange(count, dtype=np.int32) * 7 + 3)
