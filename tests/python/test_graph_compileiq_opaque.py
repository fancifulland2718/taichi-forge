import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType, SimpleNamespace

import pytest

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
    _ExecutableOptimizationSpace,
    _GraphFusionQualificationCache,
    _make_spec,
)
from taichi_forge.lang._compileiq_adapter import _CompileIQWinnerScope


_SEMANTIC_PLAN_ID = "semantic-plan:" + "a" * 24
_MAP2_RECIPE_ID = "fusion:map2:" + "b" * 24
_MAP3_RECIPE_ID = "fusion:map3:" + "c" * 24
_MAP4_RECIPE_ID = "fusion:map4:" + "d" * 24
_ROOT = Path(__file__).resolve().parents[2]


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


class _Graph:
    def __init__(self, space):
        self._space = space

    @property
    def _executable_optimization_space(self):
        return self._space


def _capability():
    return MappingProxyType(
        {
            "schema": "compileiq.taichi-forge-recipe-search-capability.v1",
            "protocol_revision": 1,
            "fork_build_id": "compileiq-taichi-forge-opaque-recipes.v1",
            "package_version": "1.0.0dev1+taichiforge.opaque1",
            "opaque_recipe_domain_schema": "compileiq.opaque-recipe-domain.v1",
            "selection_audit_schema": "compileiq.opaque-recipe-selection.v1",
            "max_recipe_ids": 4096,
            "max_field_utf8_bytes": 4096,
            "max_canonical_bytes": 4 * 1024 * 1024,
            "provider_recipe_ids_cross_core_boundary": False,
            "core_verification": (
                "bundled_manifest_lock_and_platform_hashes_at_search_start_no_override"
            ),
            "opaque_domain_binding": "capability_id_core_commit_core_lock",
            "objective_worker": "forge_main_thread_serial_v1",
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
        "ref": "refs/heads/forge/opaque-recipes-v1",
        "commit": "b36f2d2abcb8234f3f12818a38e14172d990b79a",
        "wheel_sha256": (
            "04b550cc12d7ef652c479db63447717d4b071ab7e21ee58ab50133e962d70470"
        ),
        "runtime_verification": "capability_manifest_and_python_source_lock",
    }
    json.dumps(manifest)


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
    )
    recipes = SimpleNamespace(OpaqueRecipeDomainV1=_OpaqueRecipeDomain)
    modules = {
        "compileiq.forge_support": support,
        "compileiq.recipes": recipes,
    }
    monkeypatch.setattr(_compileiq_opaque, "import_module", modules.__getitem__)

    with pytest.raises(CompileIQGraphUnavailableError, match="exact reviewed"):
        _compileiq_opaque._validated_compileiq_capability()
