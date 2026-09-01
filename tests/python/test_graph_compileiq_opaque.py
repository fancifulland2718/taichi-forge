import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge._lib import core as ti_core
from taichi_forge.graph import (
    CompileIQGraphRecipeSearch,
    CompileIQGraphUnavailableError,
    GraphExecutableRecipeSelection,
    compileiq_recipe_search,
)
from taichi_forge.graph import _compileiq_opaque
from taichi_forge.graph._optimization import (
    _CUDA_CONDITIONAL_CONTROL_RECIPE_ID,
    _CUDA_MASKED_CONTROL_RECIPE_ID,
    _CUDA_NESTED_DEVICE_UPDATE_CONTROL_RECIPE_ID,
    _CUDA_NESTED_MASKED_CONTROL_RECIPE_ID,
    _ExecutableOptimizationSpace,
    _GraphFusionQualificationCache,
    _make_spec,
)
from taichi_forge.lang._compileiq_qualification import _CompileIQWinnerScope
from tests import test_utils


_SEMANTIC_PLAN_ID = "semantic-plan:" + "a" * 24
_MAP2_RECIPE_ID = "fusion:map2:" + "b" * 24
_MAP3_RECIPE_ID = "fusion:map3:" + "c" * 24
_MAP4_RECIPE_ID = "fusion:map4:" + "d" * 24
_ROOT = Path(__file__).resolve().parents[2]


def test_compileiq_public_search_surface_is_graph_owned():
    assert ti.graph.compileiq_recipe_search is compileiq_recipe_search
    assert not hasattr(ti, "compileiq_offload_execution_plan_search")
    assert not hasattr(ti.lang, "compileiq_offload_execution_plan_search")
    assert not hasattr(ti, "CompileIQOffloadExecutionPlanSearch")
    assert not hasattr(ti.lang, "CompileIQOffloadExecutionPlanSearch")
    assert not hasattr(ti.algorithms, "compileiq_reduce_provider_search")
    assert not hasattr(ti.algorithms, "compileiq_segmented_scan_search")
    assert not hasattr(ti.algorithms, "CompileIQReduceProviderSearch")
    assert not hasattr(ti.algorithms, "CompileIQSegmentedScanSearch")


class _Literal:
    def __init__(self, value):
        self.value = value


class _Choice:
    def __init__(self, values):
        self.vals = list(values)


class _OpaqueRecipeDomain:
    SCHEMA = "compileiq.opaque-recipe-domain.v1"
    MAX_RECIPE_IDS = 4096
    MAX_FIELD_UTF8_BYTES = 4096
    MAX_CANONICAL_BYTES = 4 * 1024 * 1024

    def __init__(self, **fields):
        for name, value in fields.items():
            setattr(self, name, value)
        self.recipe_ids = tuple(
            sorted(self.recipe_ids, key=lambda value: value.encode())
        )
        payload = json.dumps(
            {
                "provider_namespace": self.provider_namespace,
                "domain_version": self.domain_version,
                "provider_semantic_fingerprint": self.provider_semantic_fingerprint,
                "recipe_ids": self.recipe_ids,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        self.domain_fingerprint = "ciq-domain-v1:" + hashlib.sha256(payload).hexdigest()

    def to_search_space(self):
        return {
            "domain_fingerprint": _Literal(self.domain_fingerprint),
            "recipe_id": _Choice(
                f"ciq-recipe-v1-{ordinal:04d}"
                for ordinal in range(len(self.recipe_ids))
            ),
        }

    def model_dump(self, *, by_alias):
        assert by_alias is True
        return {
            "schema": self.SCHEMA,
            "provider_namespace": self.provider_namespace,
            "domain_version": self.domain_version,
            "provider_semantic_fingerprint": self.provider_semantic_fingerprint,
            "compileiq_capability_id": self.compileiq_capability_id,
            "compileiq_core_commit": self.compileiq_core_commit,
            "compileiq_core_lock": self.compileiq_core_lock,
            "recipe_ids": self.recipe_ids,
        }


class _Worker:
    PROTOCOL = "forge_main_thread_serial_v1"


class _ExhaustiveSearch:
    PROTOCOL = "bounded_exhaustive_main_thread_v1"


class _TargetContract:
    SCHEMA = "compileiq.taichi-forge-opaque-target-contract.v1"


class _Graph:
    def __init__(self, space, *, map_materialization_available=None):
        self._space = space
        self._compileiq_map_materialization_available = map_materialization_available

    @property
    def _executable_optimization_space(self):
        return self._space


def _capability():
    return MappingProxyType(
        {
            "schema": "compileiq.taichi-forge-recipe-search-capability.v1",
            "protocol_revision": 3,
            "fork_build_id": "compileiq-taichi-forge-opaque-recipes.v1.3",
            "package_version": "1.0.0dev3+taichiforge.opaque2",
            "opaque_recipe_domain_schema": "compileiq.opaque-recipe-domain.v1",
            "selection_audit_schema": "compileiq.opaque-recipe-selection.v1",
            "opaque_target_contract_schema": (
                "compileiq.taichi-forge-opaque-target-contract.v1"
            ),
            "opaque_target_selection": (
                "explicit_objectives_constraints_pareto_no_scalarization_v1"
            ),
            "max_recipe_ids": 4096,
            "max_field_utf8_bytes": 4096,
            "max_canonical_bytes": 4 * 1024 * 1024,
            "provider_recipe_ids_cross_core_boundary": False,
            "core_verification": (
                "bundled_manifest_lock_and_platform_hashes_at_search_start_no_override"
            ),
            "opaque_domain_binding": "capability_id_core_commit_core_lock",
            "objective_worker": "forge_main_thread_serial_v1",
            "opaque_recipe_search": "bounded_exhaustive_main_thread_v1",
            "core_manifest_schema_version": 1,
            "core_commit": _compileiq_opaque._EXPECTED_CORE_COMMIT,
            "core_lock": _compileiq_opaque._EXPECTED_CORE_LOCK,
            "capability_id": _compileiq_opaque._EXPECTED_CAPABILITY_ID,
        }
    )


def _space(*, selected="baseline"):
    baseline = _make_spec(_SEMANTIC_PLAN_ID, "cuda", ())
    map2 = _make_spec(_SEMANTIC_PLAN_ID, "cuda", (_MAP2_RECIPE_ID,))
    map3 = _make_spec(_SEMANTIC_PLAN_ID, "cuda", (_MAP3_RECIPE_ID,))
    map4 = _make_spec(_SEMANTIC_PLAN_ID, "cuda", (_MAP4_RECIPE_ID,))
    selected_spec_id = {
        "baseline": baseline.spec_id,
        "map2": map2.spec_id,
        "map3": map3.spec_id,
        "map4": map4.spec_id,
    }[selected]
    return _ExecutableOptimizationSpace(
        semantic_plan_id=_SEMANTIC_PLAN_ID,
        baseline=baseline,
        candidates=(map2, map3, map4),
        selected_spec_id=selected_spec_id,
        selection_status=(
            "selected_baseline" if selected == "baseline" else "selected_map_recipe"
        ),
    )


def _control_space(*, selected="conditional"):
    baseline = _make_spec(
        _SEMANTIC_PLAN_ID,
        "cuda",
        (),
        _CUDA_CONDITIONAL_CONTROL_RECIPE_ID,
    )
    masked = _make_spec(
        _SEMANTIC_PLAN_ID,
        "cuda",
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


def test_map_partition_search_rejects_ambiguous_native_builder_scope(monkeypatch):
    monkeypatch.setattr(_compileiq_opaque, "_validated_compileiq_capability", _capability)
    graph = _Graph(_space(), map_materialization_available=False)

    with pytest.raises(ValueError, match="one Forge-owned source GraphBuilder"):
        CompileIQGraphRecipeSearch(graph)


def _nested_control_space(*, selected="device_update"):
    baseline = _make_spec(
        _SEMANTIC_PLAN_ID,
        "cuda",
        (),
        _CUDA_NESTED_DEVICE_UPDATE_CONTROL_RECIPE_ID,
    )
    masked = _make_spec(
        _SEMANTIC_PLAN_ID,
        "cuda",
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


def _install_reviewed_fork(monkeypatch):
    monkeypatch.setattr(
        _compileiq_opaque,
        "_validated_compileiq_capability",
        lambda: (
            _capability(),
            _OpaqueRecipeDomain,
            _Worker,
            _compileiq_opaque._EXPECTED_PYTHON_SOURCE_LOCK,
        ),
    )


def _parameters(search, recipe_id):
    return {
        "domain_fingerprint": search.domain_fingerprint,
        "recipe_id": recipe_id,
    }


def _compileiq_search_audit(search, recipe_ids):
    records = []
    token_by_recipe = {
        recipe_id: f"ciq-recipe-v1-{ordinal:04d}"
        for ordinal, recipe_id in enumerate(search.recipe_ids)
    }
    for param_id, recipe_id in enumerate(recipe_ids, start=1):
        records.append(
            {
                "param_id": param_id,
                "schema": "compileiq.opaque-recipe-selection.v1",
                "provider_namespace": search.search_space.provider_namespace,
                "domain_version": search.search_space.domain_version,
                "provider_semantic_fingerprint": (
                    search.search_space.provider_semantic_fingerprint
                ),
                "compileiq_capability_id": search.capability["capability_id"],
                "compileiq_core_commit": search.capability["core_commit"],
                "compileiq_core_lock": search.capability["core_lock"],
                "domain_fingerprint": search.domain_fingerprint,
                "core_recipe_token": token_by_recipe[recipe_id],
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


def test_importing_public_graph_api_does_not_import_compileiq():
    script = """
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(root / "python"))
import taichi_forge.graph

loaded = [
    name
    for name in sys.modules
    if name == "compileiq" or name.startswith("compileiq.")
]
assert not loaded, loaded
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(_ROOT)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_public_graph_search_is_baseline_inclusive_and_opaque(monkeypatch):
    _install_reviewed_fork(monkeypatch)

    search = compileiq_recipe_search(_Graph(_space()))

    assert isinstance(search, CompileIQGraphRecipeSearch)
    assert search.baseline_recipe_id in search.recipe_ids
    assert len(search.recipe_ids) == 4
    assert search.worker_type is _Worker
    assert search.python_source_lock == (_compileiq_opaque._EXPECTED_PYTHON_SOURCE_LOCK)
    compiled = search.search_space.to_search_space()
    tokens = compiled["recipe_id"].vals
    assert tokens == [
        "ciq-recipe-v1-0000",
        "ciq-recipe-v1-0001",
        "ciq-recipe-v1-0002",
        "ciq-recipe-v1-0003",
    ]
    assert set(tokens).isdisjoint(search.recipe_ids)

    manifest = search.manifest()
    assert manifest["baseline_recipe_id"] == search.baseline_recipe_id
    assert manifest["recipe_count"] == 4
    assert sum(recipe["is_baseline"] for recipe in manifest["recipes"]) == 1
    assert manifest["qualification"] == "independent_forge_worst_positive_v1"
    assert manifest["runtime_admission"] == "explicit_qualified_cache_only"
    assert manifest["fallback"] == "disabled"
    assert manifest["reviewed_compileiq_distribution"] == {
        "repository": "https://github.com/fancifulland2718/CompileIQ",
        "ref": "refs/heads/forge/opaque-objectives-v1.3",
        "commit": "300c426cf8bef288e926a06ab11431797d4942fa",
        "wheel_sha256": ("1510c2ec7634b379c776103137692a4f3f2f9060cb3f7fd606368c07cd1602da"),
        "runtime_verification": "capability_manifest_and_python_source_lock",
    }
    json.dumps(manifest)


def test_graph_search_forwards_explicit_opaque_target_contract(monkeypatch):
    _install_reviewed_fork(monkeypatch)
    search = compileiq_recipe_search(_Graph(_space()))
    target_contract = object()
    captured = {}

    class _CapturingExhaustiveSearch:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    search._transport._exhaustive_search_type = _CapturingExhaustiveSearch
    objective = lambda _: {"steady_ms": 1.0}

    created = search.compileiq_search(
        objective,
        target_contract=target_contract,
    )

    assert isinstance(created, _CapturingExhaustiveSearch)
    assert captured["objective_function"] is objective
    assert captured["search_space"] is search.search_space
    assert captured["baseline_recipe_id"] == search.baseline_recipe_id
    assert captured["problem_type"] == "min"
    assert captured["target_contract"] is target_contract


def test_decoded_selection_reuses_graph_materialization_identity(monkeypatch):
    _install_reviewed_fork(monkeypatch)
    search = CompileIQGraphRecipeSearch(_Graph(_space()))
    map2_id = next(
        recipe_id
        for recipe_id in search.recipe_ids
        if search.recipe_manifest(recipe_id)["materialization_recipe"] == "map2"
    )
    parameters = _parameters(search, map2_id)

    selection = search.select(parameters)

    assert isinstance(selection, GraphExecutableRecipeSelection)
    assert selection.spec_id == map2_id
    assert dict(search.worker_environment(parameters)) == {
        "TAICHI_FORGE_INTERNAL_MAP_FUSION": "map2",
        "TAICHI_FORGE_INTERNAL_STRUCTURED_CONTROL_RECIPE": "auto",
    }
    verified = search.verify_materialized_graph(
        parameters, _Graph(_space(selected="map2"))
    )
    assert verified.execution_identity == selection.execution_identity
    with pytest.raises(ValueError, match="did not select"):
        search.verify_materialized_graph(parameters, _Graph(_space()))


def test_structured_control_uses_its_own_opaque_domain_and_exact_route(monkeypatch):
    _install_reviewed_fork(monkeypatch)
    search = CompileIQGraphRecipeSearch(_Graph(_control_space()))

    assert len(search.recipe_ids) == 2
    assert search.search_space.provider_namespace == (
        "taichi_forge.graph.structured_control"
    )
    assert search.search_space.domain_version == (
        "structured-control-executable-spec.v1"
    )
    manifest = search.manifest()
    assert manifest["recipe_kind"] == "structured_control"
    assert manifest["runtime_admission"] == (
        "offline_explicit_reconstruction_only"
    )

    masked_id = next(
        recipe_id
        for recipe_id in search.recipe_ids
        if search.recipe_manifest(recipe_id).get("control_recipe_id")
        == _CUDA_MASKED_CONTROL_RECIPE_ID
    )
    parameters = _parameters(search, masked_id)
    assert dict(search.worker_environment(parameters)) == {
        "TAICHI_FORGE_INTERNAL_MAP_FUSION": "baseline",
        "TAICHI_FORGE_INTERNAL_STRUCTURED_CONTROL_RECIPE": (
            "cuda_masked_bounded_graph"
        ),
    }
    verified = search.verify_materialized_graph(
        parameters,
        _Graph(_control_space(selected="masked")),
    )
    assert verified.control_recipe_id == _CUDA_MASKED_CONTROL_RECIPE_ID


def test_nested_control_uses_distinct_opaque_domain_and_exact_route(monkeypatch):
    _install_reviewed_fork(monkeypatch)
    search = CompileIQGraphRecipeSearch(_Graph(_nested_control_space()))

    assert len(search.recipe_ids) == 2
    assert search.search_space.provider_namespace == (
        "taichi_forge.graph.nested_structured_control"
    )
    assert search.search_space.domain_version == (
        "nested-structured-control-executable-spec.v1"
    )
    manifest = search.manifest()
    assert manifest["recipe_kind"] == "structured_control"
    assert manifest["structured_control_domain"] == (
        "cuda_nested_while_while"
    )
    assert manifest["runtime_admission"] == (
        "offline_explicit_reconstruction_only"
    )

    masked_id = next(
        recipe_id
        for recipe_id in search.recipe_ids
        if search.recipe_manifest(recipe_id).get("control_recipe_id")
        == _CUDA_NESTED_MASKED_CONTROL_RECIPE_ID
    )
    parameters = _parameters(search, masked_id)
    assert dict(search.worker_environment(parameters)) == {
        "TAICHI_FORGE_INTERNAL_MAP_FUSION": "baseline",
        "TAICHI_FORGE_INTERNAL_STRUCTURED_CONTROL_RECIPE": (
            "cuda_nested_masked_bounded"
        ),
    }
    verified = search.verify_materialized_graph(
        parameters,
        _Graph(_nested_control_space(selected="masked")),
    )
    assert verified.control_recipe_id == (
        _CUDA_NESTED_MASKED_CONTROL_RECIPE_ID
    )


def test_decoded_selection_fails_closed_on_any_domain_drift(monkeypatch):
    _install_reviewed_fork(monkeypatch)
    search = CompileIQGraphRecipeSearch(_Graph(_space()))
    baseline = _parameters(search, search.baseline_recipe_id)

    assert search.select(baseline).materialization_recipe == "baseline"
    with pytest.raises(ValueError, match="exactly"):
        search.select({"recipe_id": search.baseline_recipe_id})
    with pytest.raises(ValueError, match="another Graph domain"):
        search.select({**baseline, "domain_fingerprint": "ciq-domain-v1:stale"})
    with pytest.raises(KeyError, match="unknown Graph recipe"):
        search.select({**baseline, "recipe_id": "executable:foreign"})
    with pytest.raises(TypeError, match="must be a string"):
        search.select({**baseline, "recipe_id": 1})


def test_best_result_requires_complete_exact_fork_coverage(monkeypatch):
    _install_reviewed_fork(monkeypatch)
    search = CompileIQGraphRecipeSearch(_Graph(_space()))
    candidate_id = next(
        recipe_id
        for recipe_id in search.recipe_ids
        if recipe_id != search.baseline_recipe_id
    )
    incomplete = _compileiq_search_audit(
        search, (search.baseline_recipe_id, candidate_id)
    )

    coverage = search.search_coverage(incomplete)

    assert not coverage["complete"]
    assert coverage["baseline_observed"]
    assert coverage["evaluation_count"] == 2
    with pytest.raises(RuntimeError, match="complete frozen Graph recipe domain"):
        search.require_complete_search(incomplete)

    complete = _compileiq_search_audit(search, search.recipe_ids)
    result = SimpleNamespace(
        get_best_result=lambda: {
            "params": _parameters(search, candidate_id),
            "score_1": 0.9,
        }
    )
    selection = search.select_best_result(complete, result)

    assert search.require_complete_search(complete)["complete"]
    assert selection.spec_id == candidate_id

    complete.opaque_recipe_core_provenance["core_commit"] = "forged"
    with pytest.raises(ValueError, match="verified core provenance"):
        search.search_coverage(complete)


def test_incomplete_partition_domain_refines_only_observed_disjoint_frontier(
    monkeypatch,
):
    _install_reviewed_fork(monkeypatch)
    baseline = _make_spec(_SEMANTIC_PLAN_ID, "cuda", ())
    left = _make_spec(
        _SEMANTIC_PLAN_ID,
        "cuda",
        ("fusion:map2:" + "1" * 24,),
        fusion_source_groups=((0, 1),),
    )
    right = _make_spec(
        _SEMANTIC_PLAN_ID,
        "cuda",
        ("fusion:map2:" + "2" * 24,),
        fusion_source_groups=((3, 4),),
    )
    space = _ExecutableOptimizationSpace(
        semantic_plan_id=_SEMANTIC_PLAN_ID,
        baseline=baseline,
        candidates=(left, right),
        selected_spec_id=baseline.spec_id,
        selection_status="selected_baseline",
        partition_stage="single_phase_perturbation_v1",
        partitions_complete=False,
        partition_combination_count=16,
        partition_candidate_limit=4095,
    )
    search = CompileIQGraphRecipeSearch(_Graph(space))
    observed = _compileiq_search_audit(search, search.recipe_ids)

    refined = search.refine(observed, (left.spec_id, right.spec_id))

    assert len(refined.recipe_ids) == 4
    manifest = refined.manifest()
    assert manifest["partition_stage"] == "observed_frontier_pairwise_v1"
    assert manifest["partition_parent_domain_fingerprint"] == (search.domain_fingerprint)
    assert manifest["partition_frontier_spec_ids"] == tuple(sorted((left.spec_id, right.spec_id)))
    combined = next(recipe for recipe in manifest["recipes"] if recipe["fusion_source_groups"] == ((0, 1), (3, 4)))
    assert combined["materialization_recipe"] == "exact-v1:0,1;3,4"


def test_search_schedule_and_qualification_keep_baseline_as_sentinel(monkeypatch):
    _install_reviewed_fork(monkeypatch)
    search = CompileIQGraphRecipeSearch(_Graph(_space()))

    scheduled = search.paired_schedule(blocks=2)

    assert search.baseline_recipe_id not in {item.variant_id for item in scheduled}
    assert {item.variant_id for item in scheduled} == (
        set(search.recipe_ids) - {search.baseline_recipe_id}
    )
    assert all(
        "baseline" in item.order and "candidate" in item.order for item in scheduled
    )


def test_public_search_requires_independent_qualification_before_runtime_cache(
    monkeypatch,
):
    _install_reviewed_fork(monkeypatch)
    search = CompileIQGraphRecipeSearch(_Graph(_space()))
    recipe_id = next(
        value for value in search.recipe_ids if value != search.baseline_recipe_id
    )
    finalist = search.final_candidate(recipe_id, "baseline")
    candidate_id = finalist.identity
    scope = _CompileIQWinnerScope(
        final_candidate_id=candidate_id,
        forge_specialization_id=search.recipe_manifest(recipe_id)["execution_identity"],
        workload_profile_id="tlw1:map-chain",
        shape_scope_id="elements=1048576",
        replay_scope_id="graph-rebuild-fresh-process-v1",
        runtime_scope_id="cuda:uuid:driver",
        compiler_scope_id="llvm20:driver-jit",
        provider_scope_id="baseline",
        variant_manifest_id="executable-space:sha256",
    )
    decision = search.qualify(
        {candidate_id: (0.97,) * 10},
        (finalist,),
        scopes={candidate_id: scope},
        correctness={candidate_id: True},
        memory_stable={candidate_id: True},
    )

    cache = search.qualification_cache(
        decision,
        source_commit="a" * 40,
        runtime_scope={
            "core_commit": "a" * 40,
            "device_uuid": "GPU-test",
            "driver_version": "test-driver",
        },
        binding_scope=[
            {
                "name": "values",
                "kind": "ndarray",
                "dtype": "f32",
                "rank": 1,
                "element_shape": [],
                "shape_min": [1048576],
                "shape_max": [1048576],
            }
        ],
        minimum_expected_replays=100,
        evidence_id="fresh-process-abba:artifact-sha256",
        runtime_provider_candidate_id="baseline",
    )
    parsed = _GraphFusionQualificationCache.from_dict(cache)

    assert decision.admitted
    assert parsed.entries[0].selected_spec_id == recipe_id
    assert parsed.entries[0].evidence_id.endswith(
        f"compileiq_scope={decision.scope_id}"
    )

    rejected = search.qualify(
        {candidate_id: (1.01,) * 10},
        (finalist,),
        scopes={candidate_id: scope},
        correctness={candidate_id: True},
        memory_stable={candidate_id: True},
    )
    with pytest.raises(ValueError, match="admitted decision"):
        search.qualification_cache(
            rejected,
            source_commit="a" * 40,
            runtime_scope={"device_uuid": "GPU-test"},
            binding_scope=[],
            minimum_expected_replays=100,
            evidence_id="negative-evidence",
            runtime_provider_candidate_id="baseline",
        )


def test_missing_or_different_compileiq_cannot_use_the_public_path(monkeypatch):
    def missing(_):
        raise ImportError("not installed")

    monkeypatch.setattr(_compileiq_opaque, "import_module", missing)
    with pytest.raises(CompileIQGraphUnavailableError, match="reviewed modified"):
        _compileiq_opaque._validated_compileiq_capability()

    capability = dict(_capability())
    capability["fork_build_id"] = "upstream"
    support = SimpleNamespace(
        forge_recipe_search_capability=lambda: SimpleNamespace(
            as_dict=lambda: capability
        ),
        ForgeMainThreadWorker=_Worker,
        ForgeOpaqueRecipeExhaustiveSearchV1=_ExhaustiveSearch,
        ForgeOpaqueTargetContractV1=_TargetContract,
    )
    recipes = SimpleNamespace(OpaqueRecipeDomainV1=_OpaqueRecipeDomain)
    modules = {
        "compileiq.forge_support": support,
        "compileiq.recipes": recipes,
    }
    monkeypatch.setattr(_compileiq_opaque, "import_module", modules.__getitem__)

    with pytest.raises(CompileIQGraphUnavailableError, match="exact reviewed"):
        _compileiq_opaque._validated_compileiq_capability()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_modified_compileiq_exhausts_exact_graph_partitions(monkeypatch):
    count = 257

    @ti.kernel
    def first(
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        temporary: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in source:
            temporary[i] = source[i] * 2

    @ti.kernel
    def second(
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        temporary: ti.types.ndarray(dtype=ti.i32, ndim=1),
        middle: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in source:
            middle[i] = temporary[i] + 3

    @ti.kernel
    def third(
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        middle: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in source:
            output[i] = middle[i] * 4

    symbolic = {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.i32, ndim=1)
        for name in ("source", "temporary", "middle", "output")
    }

    def build():
        builder = ti.graph.GraphBuilder()
        builder.dispatch(first, symbolic["source"], symbolic["temporary"])
        builder.dispatch(
            second,
            symbolic["source"],
            symbolic["temporary"],
            symbolic["middle"],
        )
        builder.dispatch(
            third,
            symbolic["source"],
            symbolic["middle"],
            symbolic["output"],
        )
        return builder.compile()

    monkeypatch.setenv("TAICHI_FORGE_INTERNAL_MAP_FUSION", "baseline")
    baseline = build()
    plans = compileiq_recipe_search(baseline)
    assert len(plans.recipe_ids) == 4

    source = ti.ndarray(ti.i32, shape=count)
    temporary = ti.ndarray(ti.i32, shape=count)
    middle = ti.ndarray(ti.i32, shape=count)
    output = ti.ndarray(ti.i32, shape=count)
    source_np = np.arange(count, dtype=np.int32)
    source.from_numpy(source_np)
    arguments = {
        "source": source,
        "temporary": temporary,
        "middle": middle,
        "output": output,
    }
    materialized = []

    def objective(parameters):
        selection = plans.select(parameters)
        with monkeypatch.context() as environment:
            for name, value in selection.worker_environment.items():
                environment.setenv(name, value)
            graph = build()
        plans.verify_materialized_graph(parameters, graph)
        graph.run(arguments)
        ti.sync()
        materialized.append((selection.spec_id, graph.physical_plan()["physical_dispatch_count"]))
        return float(plans.recipe_ids.index(selection.spec_id))

    compileiq_search = plans.compileiq_search(objective)
    result = compileiq_search.start()
    coverage = plans.require_complete_search(compileiq_search)
    selected = plans.select_best_result(compileiq_search, result)

    assert coverage["complete"]
    assert coverage["evaluation_count"] == len(plans.recipe_ids)
    assert {recipe_id for recipe_id, _ in materialized} == set(plans.recipe_ids)
    assert {dispatches for _, dispatches in materialized} == {1, 2, 3}
    assert selected.spec_id == plans.recipe_ids[0]
    np.testing.assert_array_equal(output.to_numpy(), (source_np * 2 + 3) * 4)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_modified_compileiq_exhausts_complete_graph_bounded_recipes(monkeypatch):
    probe = dict(ti_core.cuda_bounded_dispatch_probe())
    if not probe["exact_device_grid_available"]:
        pytest.skip(probe["unavailable_reason"])

    capacity = 257
    block_dim = 32

    @ti.kernel
    def publish(
        requested: ti.i32,
        extent: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        ti.device_extent_publish(extent, capacity, requested)

    @ti.kernel
    def consume(
        extent: ti.types.ndarray(dtype=ti.i32, ndim=1),
        observed: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        ti.loop_config(block_dim=block_dim)
        for i in range(capacity):
            if i < ti.device_extent_count(extent):
                ti.atomic_add(observed[0], i + 1)

    requested_arg = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "requested", ti.i32)
    extent_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "extent", ti.i32, ndim=1)
    first_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "first", ti.i32, ndim=1)
    second_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "second", ti.i32, ndim=1)

    def build(*, physical_grid="auto", consumer_count=2):
        builder = ti.graph.GraphBuilder()
        builder.dispatch(publish, requested_arg, extent_arg)
        builder.dispatch_bounded(
            consume,
            extent_arg,
            first_arg,
            extent=extent_arg,
            capacity=capacity,
            block_dim=block_dim,
            physical_grid=physical_grid,
        )
        if consumer_count == 2:
            builder.dispatch_bounded(
                consume,
                extent_arg,
                second_arg,
                extent=extent_arg,
                capacity=capacity,
                block_dim=block_dim,
                physical_grid=physical_grid,
            )
        return builder.compile()

    monkeypatch.setenv("TI_CUDA_BOUNDED_DISPATCH_MODE", "auto")
    monkeypatch.setenv("TI_GRAPH_CUDA_BOUNDED_UPDATE_POLICY", "auto")
    baseline = build()
    plans = compileiq_recipe_search(baseline)
    manifest = plans.manifest()
    assert (
        plans.search_space.provider_namespace
        == "taichi_forge.graph.bounded_execution"
    )
    assert plans.search_space.domain_version == "graph-bounded-complete-recipe.v1"
    assert manifest["recipe_kind"] == "graph_bounded_execution"
    assert manifest["runtime_admission"] == "offline_explicit_reconstruction_only"
    assert baseline._compileiq_graph_bounded_status == "complete_recipe_domain"

    expected_strategies = (
        "logical_exact",
        "adaptive_per_node",
        "adaptive_grouped",
        "masked_capacity",
    )
    assert {
        recipe["bounded_recipe_manifest"]["strategy"]
        for recipe in manifest["recipes"]
    } == set(expected_strategies)

    extent = ti.DeviceExtent(capacity)
    first = ti.ndarray(ti.i32, shape=1)
    second = ti.ndarray(ti.i32, shape=1)
    arguments = {
        "requested": 17,
        "extent": extent,
        "first": first,
        "second": second,
    }
    materialized = {}

    def objective(parameters):
        selection = plans.select(parameters)
        strategy = selection.bounded_recipe_manifest.strategy
        with monkeypatch.context() as environment:
            for name, value in selection.worker_environment.items():
                environment.setenv(name, value)
            graph = build()
        plans.verify_materialized_graph(parameters, graph)
        first.fill(0)
        second.fill(0)
        graph.run(arguments)
        ti.sync()
        expected = 17 * 18 // 2
        assert int(first.to_numpy()[0]) == expected
        assert int(second.to_numpy()[0]) == expected
        selected = graph._compileiq_executable_optimization_space.selected
        assert selected.bounded_recipe_manifest.strategy == strategy
        materialized[strategy] = (
            graph.execution_stats().memory.persistent_bounded_control_bytes
        )
        return float(expected_strategies.index(strategy))

    compileiq_search = plans.compileiq_search(objective)
    result = compileiq_search.start()
    coverage = plans.require_complete_search(compileiq_search)
    selected = plans.select_best_result(compileiq_search, result)

    assert coverage["complete"]
    assert coverage["evaluation_count"] == 4
    assert set(materialized) == set(expected_strategies)
    assert materialized["logical_exact"] == 0
    assert materialized["masked_capacity"] == 0
    assert materialized["adaptive_per_node"] > 0
    assert materialized["adaptive_grouped"] > 0
    assert selected.bounded_recipe_manifest.strategy == "logical_exact"

    forced = build(physical_grid="capacity")
    with pytest.raises(ValueError, match="exact map-partition search requires"):
        compileiq_recipe_search(forced)
    assert forced._compileiq_graph_bounded_status == "source_policy_out_of_scope"

    single = build(consumer_count=1)
    single_plans = compileiq_recipe_search(single)
    assert {
        recipe["bounded_recipe_manifest"]["strategy"]
        for recipe in single_plans.manifest()["recipes"]
    } == {"logical_exact", "adaptive_per_node", "masked_capacity"}
