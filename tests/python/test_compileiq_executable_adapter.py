import json
import sys
from types import ModuleType

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.graph._compileiq_adapter import (
    _CompileIQExecutableAdapter,
)
from taichi_forge.graph._optimization import (
    _ExecutableOptimizationSpace,
    _GraphFusionQualificationCache,
    _make_spec,
)
from taichi_forge.lang._compileiq_adapter import _CompileIQWinnerScope
from tests import test_utils


_SEMANTIC_PLAN_ID = "semantic-plan:" + "1" * 24
_BACKEND = "cuda"


def _recipe(size, digit):
    return f"fusion:map{size}:{digit * 24}"


def _space(*, selected="baseline", semantic_plan_id=_SEMANTIC_PLAN_ID):
    baseline = _make_spec(semantic_plan_id, _BACKEND, ())
    map2 = _make_spec(
        semantic_plan_id,
        _BACKEND,
        (_recipe(2, "2"), _recipe(2, "3")),
    )
    map3 = _make_spec(
        semantic_plan_id,
        _BACKEND,
        (_recipe(3, "4"), _recipe(2, "5")),
    )
    map4 = _make_spec(
        semantic_plan_id,
        _BACKEND,
        (_recipe(4, "6"),),
    )
    specs = {
        "baseline": baseline,
        "map2": map2,
        "map3": map3,
        "map4": map4,
    }
    return _ExecutableOptimizationSpace(
        semantic_plan_id=semantic_plan_id,
        baseline=baseline,
        candidates=(map2, map3, map4),
        selected_spec_id=specs[selected].spec_id,
        selection_status=(
            "selected_baseline" if selected == "baseline" else "selected_map_recipe"
        ),
    )


def _parameters(name):
    space = _space()
    spec = {
        "baseline": space.baseline,
        "map2": space.candidates[0],
        "map3": space.candidates[1],
        "map4": space.candidates[2],
    }[name]
    return {"forge_executable_spec": spec.spec_id}


def _install_fake_compileiq(monkeypatch):
    package = ModuleType("compileiq")
    search_spaces = ModuleType("compileiq.search_spaces")
    base = ModuleType("compileiq.search_spaces.base")
    base.choice = lambda values: ("choice", tuple(values))
    monkeypatch.setitem(sys.modules, "compileiq", package)
    monkeypatch.setitem(sys.modules, "compileiq.search_spaces", search_spaces)
    monkeypatch.setitem(sys.modules, "compileiq.search_spaces.base", base)


def test_executable_adapter_is_lazy_and_maps_only_stable_specs(monkeypatch):
    for name in tuple(sys.modules):
        if name == "compileiq" or name.startswith("compileiq."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    adapter = _CompileIQExecutableAdapter(_space())

    assert "compileiq" not in sys.modules
    assert len(adapter.spec_ids()) == 4
    assert len(adapter.spec_ids(include_baseline=False)) == 3

    _install_fake_compileiq(monkeypatch)
    assert adapter.search_space() == {
        "forge_executable_spec": ("choice", adapter.spec_ids())
    }

    selection = adapter.select(_parameters("map3"))
    assert selection.materialization_recipe == "map3"
    assert dict(selection.worker_environment) == {
        "TAICHI_FORGE_INTERNAL_MAP_FUSION": "map3"
    }


@pytest.mark.parametrize("recipe", ["baseline", "map2", "map3", "map4"])
def test_executable_adapter_verifies_exact_materialization(recipe):
    adapter = _CompileIQExecutableAdapter(_space())
    parameters = _parameters(recipe)
    actual = _space(selected=recipe)

    selection = adapter.verify_materialized(parameters, actual)
    assert selection.spec_id == actual.selected_spec_id

    mismatched_recipe = "map2" if recipe == "baseline" else "baseline"
    with pytest.raises(ValueError, match="did not select"):
        adapter.verify_materialized(parameters, _space(selected=mismatched_recipe))
    with pytest.raises(ValueError, match="semantic plan"):
        adapter.verify_materialized(
            parameters,
            _space(
                selected=recipe,
                semantic_plan_id="semantic-plan:" + "9" * 24,
            ),
        )


def test_executable_adapter_emits_plain_exact_scope_manifest():
    adapter = _CompileIQExecutableAdapter(_space())
    manifest = adapter.manifest()

    assert manifest["provider"] == "compileiq_user_space"
    assert manifest["semantic_plan_id"] == _SEMANTIC_PLAN_ID
    assert manifest["backend"] == _BACKEND
    assert [spec["materialization_recipe"] for spec in manifest["specs"]] == [
        "baseline",
        "map2",
        "map3",
        "map4",
    ]
    assert all(spec["compilation_identity"] for spec in manifest["specs"])
    assert all(spec["execution_identity"] for spec in manifest["specs"])


def test_executable_adapter_balances_and_ranks_complete_evidence():
    adapter = _CompileIQExecutableAdapter(_space())
    schedule = adapter.paired_schedule(blocks=4)
    candidate_ids = adapter.spec_ids(include_baseline=False)

    assert len(schedule) == 12
    assert [trial.order for trial in schedule[:4]] == [
        ("baseline", "candidate"),
        ("candidate", "baseline"),
        ("baseline", "candidate"),
        ("candidate", "baseline"),
    ]
    ranked = adapter.rank_paired(
        {
            candidate_ids[0]: (0.92, 0.94, 0.91, 0.93),
            candidate_ids[1]: (0.89, 0.90, 0.88, 0.91),
            candidate_ids[2]: (0.86, 1.01, 0.87, 0.88),
        },
        blocks=4,
    )
    assert [item.variant_id for item in ranked] == [
        candidate_ids[1],
        candidate_ids[0],
        candidate_ids[2],
    ]
    assert ranked[0].worst_positive
    assert not ranked[2].worst_positive


def _winner_scope(candidate_id, specialization_id, provider_id):
    return _CompileIQWinnerScope(
        final_candidate_id=candidate_id,
        forge_specialization_id=specialization_id,
        workload_profile_id="tlw1:map-chain",
        shape_scope_id="elements=1048576",
        replay_scope_id="graph-rebuild-fresh-process-v1",
        runtime_scope_id="cuda:uuid:driver",
        compiler_scope_id="llvm20:driver-jit",
        provider_scope_id=provider_id,
        variant_manifest_id="executable-space:sha256",
    )


def test_executable_adapter_qualifies_exact_recipe_and_provider_candidate():
    adapter = _CompileIQExecutableAdapter(_space())
    map2, map3 = adapter.spec_ids(include_baseline=False)[:2]
    finalists = (
        adapter.final_candidate(map2, "driver-baseline"),
        adapter.final_candidate(map3, "acf-map3"),
    )
    candidate_ids = tuple(candidate.identity for candidate in finalists)
    scopes = {
        candidate.identity: _winner_scope(
            candidate.identity,
            adapter.select(
                {"forge_executable_spec": candidate.forge_object_id}
            ).execution_identity,
            candidate.provider_candidate_id,
        )
        for candidate in finalists
    }

    assert len(adapter.qualification_stage(finalists).schedule) == 20
    decision = adapter.qualify(
        {
            candidate_ids[0]: (0.97,) * 10,
            candidate_ids[1]: (0.94,) * 9 + (1.02,),
        },
        finalists,
        scopes=scopes,
        correctness={candidate_ids[0]: True, candidate_ids[1]: True},
        memory_stable={candidate_ids[0]: True, candidate_ids[1]: True},
    )

    assert decision.admitted
    assert decision.selected_candidate_id == candidate_ids[0]
    assert decision.selected_forge_object_kind == "executable_spec"
    assert decision.selected_forge_object_id == map2
    assert decision.selected_forge_variant_id is None
    assert decision.selected_provider_candidate_id == "driver-baseline"
    assert decision.scope_id == scopes[candidate_ids[0]].identity

    mismatched = dict(scopes)
    mismatched[candidate_ids[0]] = _winner_scope(candidate_ids[1], "wrong", "wrong")
    with pytest.raises(ValueError, match="exact final candidate"):
        adapter.qualify(
            {candidate_id: (0.97,) * 10 for candidate_id in candidate_ids},
            finalists,
            scopes=mismatched,
            correctness={candidate_id: True for candidate_id in candidate_ids},
            memory_stable={candidate_id: True for candidate_id in candidate_ids},
        )


def _single_candidate_decision(adapter, *, ratio=0.97, provider="baseline"):
    spec_id = adapter.spec_ids(include_baseline=False)[0]
    finalist = adapter.final_candidate(spec_id, provider)
    candidate_id = finalist.identity
    scope = _winner_scope(
        candidate_id, adapter._specs[spec_id].execution_identity, provider
    )
    decision = adapter.qualify(
        {candidate_id: (ratio,) * 10},
        (finalist,),
        scopes={candidate_id: scope},
        correctness={candidate_id: True},
        memory_stable={candidate_id: True},
    )
    return decision, spec_id


def _qualification_scope():
    return {
        "source_commit": "a" * 40,
        "runtime_scope": {
            "core_commit": "a" * 40,
            "device_uuid": "GPU-test",
            "driver_version": "test-driver",
        },
        "binding_scope": [
            {
                "name": "values",
                "kind": "ndarray",
                "dtype": "f32",
                "rank": 2,
                "element_shape": [],
                "shape_min": [1048576, 3],
                "shape_max": [1048576, 3],
            },
            {
                "name": "count",
                "kind": "scalar",
                "minimum": 1048576,
                "maximum": 1048576,
            },
        ],
        "minimum_expected_replays": 100,
        "evidence_id": "fresh-process-abba:artifact-sha256",
        "runtime_provider_candidate_id": "baseline",
    }


def test_executable_adapter_emits_runtime_cache_only_after_independent_gates():
    adapter = _CompileIQExecutableAdapter(_space())
    decision, spec_id = _single_candidate_decision(adapter)
    cache = adapter.qualification_cache(decision, **_qualification_scope())
    parsed = _GraphFusionQualificationCache.from_dict(cache)

    assert len(parsed.entries) == 1
    entry = parsed.entries[0]
    assert entry.selected_spec_id == spec_id
    assert entry.execution_identity == adapter._specs[spec_id].execution_identity
    assert (
        entry.baseline_execution_identity == adapter._space.baseline.execution_identity
    )
    assert entry.evidence_id.endswith(f"compileiq_scope={decision.scope_id}")

    json_cache = json.loads(
        adapter.qualification_cache_json(decision, **_qualification_scope())
    )
    reparsed = _GraphFusionQualificationCache.from_dict(json_cache)
    assert reparsed.entries[0].identity == entry.identity


def test_executable_adapter_cache_rejects_search_only_or_provider_mismatch():
    adapter = _CompileIQExecutableAdapter(_space())
    rejected, _ = _single_candidate_decision(adapter, ratio=1.01)
    with pytest.raises(ValueError, match="admitted decision"):
        adapter.qualification_cache(rejected, **_qualification_scope())

    admitted, _ = _single_candidate_decision(adapter, provider="acf-provider")
    with pytest.raises(ValueError, match="does not match the runtime provider"):
        adapter.qualification_cache(admitted, **_qualification_scope())


def test_executable_adapter_rejects_unknown_or_ambiguous_recipes():
    space = _space()
    adapter = _CompileIQExecutableAdapter(space)
    with pytest.raises(KeyError, match="require 'forge_executable_spec'"):
        adapter.select({})
    with pytest.raises(KeyError, match="unknown Forge executable spec"):
        adapter.select({"forge_executable_spec": "missing"})

    invalid = _make_spec(_SEMANTIC_PLAN_ID, _BACKEND, ("fusion:unknown",))
    with pytest.raises(ValueError, match="unsupported executable fusion"):
        _CompileIQExecutableAdapter(
            _ExecutableOptimizationSpace(
                semantic_plan_id=_SEMANTIC_PLAN_ID,
                baseline=space.baseline,
                candidates=(invalid,),
                selected_spec_id=space.baseline.spec_id,
                selection_status="selected_baseline",
            )
        )

    duplicate_map2 = _make_spec(
        _SEMANTIC_PLAN_ID,
        _BACKEND,
        (_recipe(2, "7"),),
    )
    with pytest.raises(ValueError, match="unique materialization"):
        _CompileIQExecutableAdapter(
            _ExecutableOptimizationSpace(
                semantic_plan_id=_SEMANTIC_PLAN_ID,
                baseline=space.baseline,
                candidates=(space.candidates[0], duplicate_map2),
                selected_spec_id=space.baseline.spec_id,
                selection_status="selected_baseline",
            )
        )


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_executable_adapter_rebuilds_and_verifies_selected_graph(monkeypatch):
    @ti.kernel
    def stage_one(
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        first: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in source:
            first[i] = source[i] * 2

    @ti.kernel
    def stage_two(
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        first: ti.types.ndarray(dtype=ti.i32, ndim=1),
        second: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in source:
            second[i] = first[i] + 3

    @ti.kernel
    def stage_three(
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        second: ti.types.ndarray(dtype=ti.i32, ndim=1),
        third: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in source:
            third[i] = second[i] * 4

    @ti.kernel
    def stage_four(
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        third: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in source:
            output[i] = third[i] - 5

    symbolic = {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.i32, ndim=1)
        for name in ("source", "first", "second", "third", "output")
    }

    def build():
        builder = ti.graph.GraphBuilder()
        builder.dispatch(stage_one, symbolic["source"], symbolic["first"])
        builder.dispatch(
            stage_two,
            symbolic["source"],
            symbolic["first"],
            symbolic["second"],
        )
        builder.dispatch(
            stage_three,
            symbolic["source"],
            symbolic["second"],
            symbolic["third"],
        )
        builder.dispatch(
            stage_four,
            symbolic["source"],
            symbolic["third"],
            symbolic["output"],
        )
        return builder.compile()

    monkeypatch.setenv("TAICHI_FORGE_INTERNAL_MAP_FUSION", "baseline")
    baseline = build()
    adapter = _CompileIQExecutableAdapter.from_graph(baseline)
    map4 = next(
        item
        for item in adapter.manifest()["specs"]
        if item["materialization_recipe"] == "map4"
    )
    parameters = {"forge_executable_spec": map4["spec_id"]}
    selection = adapter.select(parameters)
    for name, value in selection.worker_environment.items():
        monkeypatch.setenv(name, value)

    materialized = build()
    adapter.verify_materialized_graph(parameters, materialized)
    assert materialized.physical_plan()["physical_dispatch_count"] == 1

    count = 257
    arrays = {name: ti.ndarray(ti.i32, shape=count) for name in symbolic}
    source_np = np.arange(count, dtype=np.int32)
    arrays["source"].from_numpy(source_np)
    materialized.run(arrays)
    np.testing.assert_array_equal(
        arrays["output"].to_numpy(), (source_np * 2 + 3) * 4 - 5
    )
