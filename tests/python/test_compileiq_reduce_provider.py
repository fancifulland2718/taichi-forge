import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge import _compileiq_opaque as _opaque_transport
from taichi_forge.algorithms._compileiq_opaque import (
    CompileIQReduceProviderSearch,
    CompileIQReduceProviderSelection,
    compileiq_reduce_provider_search,
)
from taichi_forge.algorithms import _compileiq_opaque
from taichi_forge.lang._compileiq_qualification import _CompileIQWinnerScope
from tests import test_utils


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


def _capability():
    return MappingProxyType(
        {
            "schema": "compileiq.taichi-forge-recipe-search-capability.v1",
            "protocol_revision": 2,
            "fork_build_id": "compileiq-taichi-forge-opaque-recipes.v1.2",
            "package_version": "1.0.0dev3+taichiforge.opaque1",
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
            "opaque_recipe_search": "bounded_exhaustive_main_thread_v1",
            "core_manifest_schema_version": 1,
            "core_commit": _opaque_transport._EXPECTED_CORE_COMMIT,
            "core_lock": _opaque_transport._EXPECTED_CORE_LOCK,
            "capability_id": _opaque_transport._EXPECTED_CAPABILITY_ID,
        }
    )


def _scope(size=4096):
    return {
        "schema": "taichi_forge.algorithms.reduce-provider-scope.v1",
        "taichi_commit": "a" * 40,
        "backend": "cuda",
        "operation": "sum",
        "dtype": "i32",
        "size": size,
        "input": {
            "storage": "dense_field",
            "shape": (size,),
            "offset": 0,
            "stride": 1,
        },
        "output": {"storage": "scalar_field", "shape": ()},
        "field_private_threshold": 65536,
        "providers": (
            {
                "method": "cuda_device",
                "backend": "cuda",
                "program_available": True,
                "provider_probes": ("cuda_device_reduce_available",),
                "implementation": "native",
                "dependency_class": "driver",
            },
            {
                "method": "field_atomic",
                "backend": "cuda",
                "program_available": True,
                "provider_probes": (),
                "implementation": "fallback",
                "dependency_class": "builtin",
            },
        ),
    }


def _install_reviewed_fork(monkeypatch):
    monkeypatch.setattr(
        _compileiq_opaque,
        "_validated_compileiq_capability",
        lambda: (
            _capability(),
            _OpaqueRecipeDomain,
            _Worker,
            _opaque_transport._EXPECTED_PYTHON_SOURCE_LOCK,
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


def test_importing_algorithms_does_not_import_compileiq():
    script = """
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(root / "python"))
import taichi_forge.algorithms

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


def test_reduce_provider_search_is_baseline_inclusive_and_opaque(monkeypatch):
    _install_reviewed_fork(monkeypatch)
    monkeypatch.setattr(_compileiq_opaque, "_reduce_scope", lambda *_: _scope())

    search = compileiq_reduce_provider_search(object(), object())

    assert isinstance(search, CompileIQReduceProviderSearch)
    assert search.baseline_recipe_id in search.recipe_ids
    assert len(search.recipe_ids) == 2
    assert search.worker_type is _Worker
    assert search.python_source_lock == (_opaque_transport._EXPECTED_PYTHON_SOURCE_LOCK)
    tokens = search.search_space.to_search_space()["recipe_id"].vals
    assert tokens == ["ciq-recipe-v1-0000", "ciq-recipe-v1-0001"]
    assert set(tokens).isdisjoint(search.recipe_ids)

    selections = {
        search.select(_parameters(search, recipe_id)).method
        for recipe_id in search.recipe_ids
    }
    assert selections == {"cuda_device", "field_atomic"}
    assert isinstance(
        search.select(_parameters(search, search.baseline_recipe_id)),
        CompileIQReduceProviderSelection,
    )
    manifest = search.manifest()
    assert manifest["recipe_count"] == 2
    assert sum(recipe["is_baseline"] for recipe in manifest["recipes"]) == 1
    assert manifest["runtime_admission"] == (
        "explicit_selection_only_no_auto_policy_mutation"
    )
    mutable_scope = dict(search.scope)
    mutable_scope["input"]["shape"] = (1,)
    assert search.scope["input"]["shape"] == (4096,)
    with pytest.raises(ValueError, match="baseline is a sentinel"):
        search.final_candidate(search.baseline_recipe_id)
    json.dumps(manifest)


def test_reduce_provider_search_requires_complete_audited_coverage(monkeypatch):
    _install_reviewed_fork(monkeypatch)
    monkeypatch.setattr(_compileiq_opaque, "_reduce_scope", lambda *_: _scope())
    search = CompileIQReduceProviderSearch(object(), object())
    candidate = next(
        recipe_id
        for recipe_id in search.recipe_ids
        if recipe_id != search.baseline_recipe_id
    )
    incomplete = _compileiq_search_audit(search, (search.baseline_recipe_id,))

    assert not search.search_coverage(incomplete)["complete"]
    with pytest.raises(RuntimeError, match="complete frozen reduce provider recipe"):
        search.require_complete_search(incomplete)

    complete = _compileiq_search_audit(search, search.recipe_ids)
    result = SimpleNamespace(
        get_best_result=lambda: {
            "params": _parameters(search, candidate),
            "score_1": 0.9,
        }
    )
    selection = search.select_best_result(complete, result)
    assert selection.method == "field_atomic"


def test_reduce_provider_qualification_keeps_auto_policy_unchanged(monkeypatch):
    _install_reviewed_fork(monkeypatch)
    monkeypatch.setattr(_compileiq_opaque, "_reduce_scope", lambda *_: _scope())
    search = CompileIQReduceProviderSearch(object(), object())
    recipe_id = next(
        value for value in search.recipe_ids if value != search.baseline_recipe_id
    )
    finalist = search.final_candidate(recipe_id)
    scope = _CompileIQWinnerScope(
        final_candidate_id=finalist.identity,
        forge_specialization_id=search.domain_fingerprint,
        workload_profile_id="reduce-replay-v1",
        shape_scope_id="elements=4096",
        replay_scope_id="fresh-process-abba-v1",
        runtime_scope_id="cuda:uuid:driver",
        compiler_scope_id="llvm20:driver-jit",
        provider_scope_id="cuda-device-vs-field-atomic",
        variant_manifest_id=search.domain_fingerprint,
    )
    decision = search.qualify(
        {finalist.identity: (0.95,) * 10},
        (finalist,),
        scopes={finalist.identity: scope},
        correctness={finalist.identity: True},
        memory_stable={finalist.identity: True},
    )

    assert decision.admitted
    assert decision.selected_forge_object_kind == "primitive_provider_recipe"
    assert decision.selected_forge_object_id == recipe_id
    assert decision.selected_provider_candidate_id == "field_atomic"
    assert search.manifest()["runtime_admission"].endswith("no_auto_policy_mutation")


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_reduce_provider_search_materializes_both_exact_routes(monkeypatch):
    _install_reviewed_fork(monkeypatch)
    size = 4096
    values = ti.field(ti.i32, shape=size)
    output = ti.field(ti.i32, shape=())
    values.from_numpy(np.arange(size, dtype=np.int32) % 97 - 48)
    expected = int(np.sum(values.to_numpy(), dtype=np.int64))
    search = compileiq_reduce_provider_search(values, output)

    observed = {}
    for recipe_id in search.recipe_ids:
        selection = search.execute(_parameters(search, recipe_id), values, output)
        ti.sync()
        observed[selection.method] = int(output[None])

    assert observed == {
        "cuda_device": expected,
        "field_atomic": expected,
    }
