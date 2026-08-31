import json
import sys
from types import ModuleType

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge._lib import core as ti_core
from taichi_forge.graph._compileiq_adapter import (
    _CompileIQExecutableAdapter,
)
from taichi_forge.graph._optimization import (
    _CUDA_CONDITIONAL_CONTROL_RECIPE_ID,
    _CUDA_MASKED_CONTROL_RECIPE_ID,
    _CUDA_NESTED_DEVICE_UPDATE_CONTROL_RECIPE_ID,
    _CUDA_NESTED_MASKED_CONTROL_RECIPE_ID,
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
        "TAICHI_FORGE_INTERNAL_MAP_FUSION": "map3",
        "TAICHI_FORGE_INTERNAL_STRUCTURED_CONTROL_RECIPE": "auto",
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
    assert "recipe_kind" not in manifest
    assert all("control_recipe_id" not in spec for spec in manifest["specs"])


def _control_space(*, selected="conditional"):
    baseline = _make_spec(
        _SEMANTIC_PLAN_ID,
        _BACKEND,
        (),
        _CUDA_CONDITIONAL_CONTROL_RECIPE_ID,
    )
    masked = _make_spec(
        _SEMANTIC_PLAN_ID,
        _BACKEND,
        (),
        _CUDA_MASKED_CONTROL_RECIPE_ID,
    )
    selected_spec = baseline if selected == "conditional" else masked
    return _ExecutableOptimizationSpace(
        semantic_plan_id=_SEMANTIC_PLAN_ID,
        baseline=baseline,
        candidates=(masked,),
        selected_spec_id=selected_spec.spec_id,
        selection_status=(
            "selected_control_baseline"
            if selected == "conditional"
            else "selected_control_recipe"
        ),
    )


def _nested_control_space(*, selected="device_update"):
    baseline = _make_spec(
        _SEMANTIC_PLAN_ID,
        _BACKEND,
        (),
        _CUDA_NESTED_DEVICE_UPDATE_CONTROL_RECIPE_ID,
    )
    masked = _make_spec(
        _SEMANTIC_PLAN_ID,
        _BACKEND,
        (),
        _CUDA_NESTED_MASKED_CONTROL_RECIPE_ID,
    )
    selected_spec = baseline if selected == "device_update" else masked
    return _ExecutableOptimizationSpace(
        semantic_plan_id=_SEMANTIC_PLAN_ID,
        baseline=baseline,
        candidates=(masked,),
        selected_spec_id=selected_spec.spec_id,
        selection_status=(
            "selected_control_baseline"
            if selected == "device_update"
            else "selected_control_recipe"
        ),
    )


def test_executable_adapter_materializes_exact_structured_control_recipe():
    baseline = _control_space()
    adapter = _CompileIQExecutableAdapter(baseline)
    masked = baseline.candidates[0]
    parameters = {"forge_executable_spec": masked.spec_id}

    selection = adapter.select(parameters)
    assert adapter.recipe_kind == "structured_control"
    assert selection.control_recipe_id == _CUDA_MASKED_CONTROL_RECIPE_ID
    assert dict(selection.worker_environment) == {
        "TAICHI_FORGE_INTERNAL_MAP_FUSION": "baseline",
        "TAICHI_FORGE_INTERNAL_STRUCTURED_CONTROL_RECIPE": (
            "cuda_masked_bounded_graph"
        ),
    }
    assert adapter.verify_materialized(
        parameters, _control_space(selected="masked")
    ) == selection
    with pytest.raises(ValueError, match="did not select"):
        adapter.verify_materialized(parameters, _control_space())

    manifest = adapter.manifest()
    assert manifest["recipe_kind"] == "structured_control"
    assert manifest["runtime_admission"] == (
        "offline_explicit_reconstruction_only"
    )
    assert [spec["control_materialization_recipe"] for spec in manifest["specs"]] == [
        "cuda_conditional_graph",
        "cuda_masked_bounded_graph",
    ]


def test_executable_adapter_materializes_exact_nested_control_recipe():
    baseline = _nested_control_space()
    adapter = _CompileIQExecutableAdapter(baseline)
    masked = baseline.candidates[0]
    parameters = {"forge_executable_spec": masked.spec_id}

    selection = adapter.select(parameters)
    assert adapter.recipe_kind == "structured_control"
    assert adapter.structured_control_domain == "cuda_nested_while_while"
    assert selection.control_recipe_id == (
        _CUDA_NESTED_MASKED_CONTROL_RECIPE_ID
    )
    assert dict(selection.worker_environment) == {
        "TAICHI_FORGE_INTERNAL_MAP_FUSION": "baseline",
        "TAICHI_FORGE_INTERNAL_STRUCTURED_CONTROL_RECIPE": (
            "cuda_nested_masked_bounded"
        ),
    }
    assert adapter.verify_materialized(
        parameters,
        _nested_control_space(selected="masked"),
    ) == selection

    manifest = adapter.manifest()
    assert manifest["recipe_kind"] == "structured_control"
    assert manifest["structured_control_domain"] == (
        "cuda_nested_while_while"
    )
    assert [spec["control_materialization_recipe"] for spec in manifest["specs"]] == [
        "cuda_nested_device_update",
        "cuda_nested_masked_bounded",
    ]


def test_structured_control_qualification_cannot_emit_runtime_cache():
    adapter = _CompileIQExecutableAdapter(_control_space())
    decision, _ = _single_candidate_decision(adapter)
    with pytest.raises(ValueError, match="offline-only"):
        adapter.qualification_cache(decision, **_qualification_scope())


def test_structured_control_recipe_cannot_form_fusion_cartesian_product():
    with pytest.raises(ValueError, match="cannot combine control and fusion"):
        _make_spec(
            _SEMANTIC_PLAN_ID,
            _BACKEND,
            (_recipe(2, "8"),),
            _CUDA_MASKED_CONTROL_RECIPE_ID,
        )


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


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_executable_adapter_rebuilds_cuda_auto_while_routes(monkeypatch):
    capabilities = dict(ti_core.cuda_conditional_graph_capabilities())
    if not capabilities.get("general_graph_exact_control_available", False):
        pytest.skip("general CUDA conditional Graph is unavailable")
    if not capabilities.get("internal_masked_graph_available", False):
        pytest.skip("internal masked CUDA Graph control is unavailable")

    @ti.kernel
    def initialize(
        state: ti.types.ndarray(dtype=ti.i32, ndim=0),
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
    ):
        state[None] = 0
        predicate[None] = 0
        counter[None] = 0

    @ti.kernel
    def condition(
        state: ti.types.ndarray(dtype=ti.i32, ndim=0),
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        target: ti.i32,
    ):
        predicate[None] = int(state[None] < target)

    @ti.kernel
    def step(
        state: ti.types.ndarray(dtype=ti.i32, ndim=0),
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
    ):
        if predicate[None] != 0:
            state[None] += 1
            counter[None] += 1

    def scalar(name):
        return ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.i32, ndim=0)

    state = scalar("state")
    predicate = scalar("predicate")
    counter = scalar("counter")
    target = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "target", ti.i32)

    def build(lowering_mode="auto"):
        builder = ti.graph.GraphBuilder()
        builder.dispatch(initialize, state, predicate, counter)
        condition_region = builder.create_sequential()
        condition_region.dispatch(condition, state, predicate, target)
        body = builder.create_sequential()
        body.dispatch(step, state, predicate, counter)
        builder.while_loop(
            condition_region,
            body,
            predicate=predicate,
            control_inputs=(state, target),
            carried_state=(state,),
            counter=counter,
            max_iterations=8,
            lowering_mode=lowering_mode,
            name="compileiq_control",
        )
        return builder.compile()

    monkeypatch.delenv("TI_GRAPH_CUDA_FORCE_MASKED_CONTROL", raising=False)
    monkeypatch.setenv(
        "TAICHI_FORGE_INTERNAL_STRUCTURED_CONTROL_RECIPE",
        "cuda_conditional_graph",
    )
    baseline = build()
    adapter = _CompileIQExecutableAdapter.from_graph(baseline)
    assert adapter.recipe_kind == "structured_control"
    assert len(adapter.spec_ids()) == 2
    assert baseline._executable_optimization_space.selected_spec_id == (
        adapter.baseline_spec_id
    )

    masked_spec_id = adapter.spec_ids(include_baseline=False)[0]
    parameters = {"forge_executable_spec": masked_spec_id}
    selection = adapter.select(parameters)
    for name, value in selection.worker_environment.items():
        monkeypatch.setenv(name, value)
    materialized = build()
    adapter.verify_materialized_graph(parameters, materialized)

    args = {
        "state": ti.ndarray(ti.i32, shape=()),
        "predicate": ti.ndarray(ti.i32, shape=()),
        "counter": ti.ndarray(ti.i32, shape=()),
        "target": 5,
    }
    materialized.run(args)
    report = materialized.control_flow_stats()[0]
    assert report.lowering == "cuda_masked_bounded_graph"
    assert report.logical_iterations == 5
    assert args["state"].to_numpy()[()] == 5
    assert args["counter"].to_numpy()[()] == 5

    # Public explicit policies remain outside the R5 search domain even when
    # the private worker overlay is present.
    for explicit_mode in ("portable", "native_required"):
        explicit = build(explicit_mode)
        explicit_space = explicit._executable_optimization_space
        assert not explicit_space.baseline.control_recipe_id
        assert all(not spec.control_recipe_id for spec in explicit_space.candidates)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_executable_adapter_rebuilds_cuda_nested_while_routes(monkeypatch):
    capabilities = dict(ti_core.cuda_conditional_graph_capabilities())
    if not capabilities.get("internal_masked_graph_available", False):
        pytest.skip("internal masked CUDA Graph control is unavailable")
    nested_probe = dict(ti_core.cuda_bounded_dispatch_probe())
    if not nested_probe.get("exact_device_grid_available", False):
        pytest.skip("exact CUDA nested device update is unavailable")

    @ti.kernel
    def evaluate_outer(
        counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        target: ti.i32,
    ):
        predicate[None] = int(counter[None] < target)

    @ti.kernel
    def reset_inner(counter: ti.types.ndarray(dtype=ti.i32, ndim=0)):
        counter[None] = 0

    @ti.kernel
    def evaluate_inner(
        counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        target: ti.i32,
    ):
        predicate[None] = int(counter[None] < target)

    @ti.kernel
    def inner_step(
        counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        total: ti.types.ndarray(dtype=ti.i32, ndim=0),
    ):
        if predicate[None] != 0:
            counter[None] += 1
            total[None] += 1

    @ti.kernel
    def outer_step(
        counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
    ):
        if predicate[None] != 0:
            counter[None] += 1

    def scalar(name):
        return ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.i32, ndim=0)

    outer_counter = scalar("outer_counter")
    outer_predicate = scalar("outer_predicate")
    inner_counter = scalar("inner_counter")
    inner_predicate = scalar("inner_predicate")
    total = scalar("total")
    outer_target = ti.graph.Arg(
        ti.graph.ArgKind.SCALAR, "outer_target", ti.i32
    )
    inner_target = ti.graph.Arg(
        ti.graph.ArgKind.SCALAR, "inner_target", ti.i32
    )

    def build(lowering_mode="auto"):
        builder = ti.graph.GraphBuilder()
        outer_condition = builder.create_sequential()
        outer_condition.dispatch(
            evaluate_outer,
            outer_counter,
            outer_predicate,
            outer_target,
        )
        inner_condition = builder.create_sequential()
        inner_condition.dispatch(
            evaluate_inner,
            inner_counter,
            inner_predicate,
            inner_target,
        )
        inner_body = builder.create_sequential()
        inner_body.dispatch(inner_step, inner_counter, inner_predicate, total)
        outer_body = builder.create_sequential()
        outer_body.dispatch(reset_inner, inner_counter)
        outer_body.while_loop(
            inner_condition,
            inner_body,
            predicate=inner_predicate,
            control_inputs=(inner_counter, inner_target),
            carried_state=(inner_counter, total),
            counter=inner_counter,
            max_iterations=8,
            name="compileiq_inner",
        )
        outer_body.dispatch(outer_step, outer_counter, outer_predicate)
        builder.while_loop(
            outer_condition,
            outer_body,
            predicate=outer_predicate,
            control_inputs=(outer_counter, outer_target),
            carried_state=(outer_counter, inner_counter, total),
            counter=outer_counter,
            max_iterations=8,
            lowering_mode=lowering_mode,
            name="compileiq_outer",
        )
        return builder.compile()

    monkeypatch.delenv("TI_GRAPH_CUDA_FORCE_MASKED_CONTROL", raising=False)
    monkeypatch.setenv(
        "TAICHI_FORGE_INTERNAL_STRUCTURED_CONTROL_RECIPE",
        "cuda_nested_device_update",
    )
    baseline = build()
    adapter = _CompileIQExecutableAdapter.from_graph(baseline)
    assert adapter.structured_control_domain == "cuda_nested_while_while"
    assert len(adapter.spec_ids()) == 2

    masked_spec_id = adapter.spec_ids(include_baseline=False)[0]
    parameters = {"forge_executable_spec": masked_spec_id}
    selection = adapter.select(parameters)
    for name, value in selection.worker_environment.items():
        monkeypatch.setenv(name, value)
    materialized = build()
    adapter.verify_materialized_graph(parameters, materialized)
    outer = next(
        node
        for node in materialized._spec.structured_control_nodes
        if node.control_depth == 1
    )
    assert outer._cuda_nested_control_lowering == "cuda_masked_bounded_graph"

    # Physical selection is part of the compiled identity. A later process
    # environment change must not silently mutate this Graph's route.
    monkeypatch.setenv(
        "TAICHI_FORGE_INTERNAL_STRUCTURED_CONTROL_RECIPE",
        "cuda_nested_device_update",
    )
    args = {
        name: ti.ndarray(ti.i32, shape=())
        for name in (
            "outer_counter",
            "outer_predicate",
            "inner_counter",
            "inner_predicate",
            "total",
        )
    }
    for value in args.values():
        value.fill(0)
    ticket = materialized.submit(
        {**args, "outer_target": 2, "inner_target": 3}
    )
    ticket.wait()
    assert args["outer_counter"].to_numpy()[()] == 2
    assert args["inner_counter"].to_numpy()[()] == 3
    assert args["total"].to_numpy()[()] == 6
    assert materialized._graph_stats[0]["last_path"] in (
        "cuda_masked_capture",
        "cuda_masked_replay",
        "cuda_masked_patched_replay",
    )

    # Public explicit policies remain outside the nested search domain even
    # when the private reconstruction selector is present.
    for explicit_mode in ("portable", "native_required"):
        explicit = build(explicit_mode)
        explicit_space = explicit._executable_optimization_space
        assert not explicit_space.baseline.control_recipe_id
        assert all(
            not spec.control_recipe_id for spec in explicit_space.candidates
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
