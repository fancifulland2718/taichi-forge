import hashlib
import json
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge import _compileiq_opaque as _opaque_transport
from taichi_forge.algorithms import (
    CompileIQSegmentedScanSearch,
    CompileIQSegmentedScanSelection,
    CompileIQSegmentedScanUnavailableError,
    compileiq_segmented_scan_search,
)
from taichi_forge.algorithms import _compileiq_segmented_scan
from taichi_forge.lang._compileiq_qualification import _CompileIQWinnerScope
from tests import test_utils


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
        self.recipe_ids = tuple(sorted(self.recipe_ids, key=lambda value: value.encode()))
        payload = json.dumps(
            {
                "provider_namespace": self.provider_namespace,
                "domain_version": self.domain_version,
                "provider_semantic_fingerprint": (self.provider_semantic_fingerprint),
                "recipe_ids": self.recipe_ids,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        self.domain_fingerprint = "ciq-domain-v1:" + hashlib.sha256(payload).hexdigest()

    def to_search_space(self):
        return {
            "domain_fingerprint": _Literal(self.domain_fingerprint),
            "recipe_id": _Choice(f"ciq-recipe-v1-{ordinal:04d}" for ordinal in range(len(self.recipe_ids))),
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
            "core_verification": ("bundled_manifest_lock_and_platform_hashes_at_search_start_no_override"),
            "opaque_domain_binding": "capability_id_core_commit_core_lock",
            "objective_worker": "forge_main_thread_serial_v1",
            "opaque_recipe_search": "bounded_exhaustive_main_thread_v1",
            "core_manifest_schema_version": 1,
            "core_commit": _opaque_transport._EXPECTED_CORE_COMMIT,
            "core_lock": _opaque_transport._EXPECTED_CORE_LOCK,
            "capability_id": _opaque_transport._EXPECTED_CAPABILITY_ID,
        }
    )


def _scope(*, baseline_method="serial", inclusive=True):
    return {
        "schema": "taichi_forge.algorithms.segmented-scan-scope.v1",
        "taichi_commit": "a" * 40,
        "backend": "cuda",
        "operation": "sum",
        "dtype": "i32",
        "inclusive": inclusive,
        "source_method": "auto",
        "baseline_method": baseline_method,
        "input": {"storage": "plain_ndarray", "shape": (4096,)},
        "output": {
            "storage": "plain_ndarray",
            "shape": (4096,),
            "aliases_input": False,
        },
        "layout": {
            "encoding": "offsets",
            "capacity": 4096,
            "num_items": 4096,
            "num_segments": 64,
            "max_segment_length": 64,
            "topology_fingerprint": "forge-segmented-layout-v1:" + "b" * 64,
        },
        "auto_threshold": {
            "minimum_items": 65536,
            "minimum_max_segment_length": 4096,
        },
        "provider_source_lock": "forge-segmented-scan-source-v1:" + "c" * 64,
        "provider_source_files": (),
        "providers": (
            {
                "method": "serial",
                "backend": "cuda",
                "implementation": "segment_local_jit",
                "dependency_class": "builtin",
            },
            {
                "method": "global_scan",
                "backend": "cuda",
                "implementation": "native_scan_plus_segment_correction",
                "dependency_class": "driver",
                "provider_probe": "cuda_device_scan_available",
            },
        ),
    }


def _install_reviewed_fork(monkeypatch):
    monkeypatch.setattr(
        _compileiq_segmented_scan,
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
    token_by_recipe = {recipe_id: f"ciq-recipe-v1-{ordinal:04d}" for ordinal, recipe_id in enumerate(search.recipe_ids)}
    records = [
        {
            "param_id": param_id,
            "schema": "compileiq.opaque-recipe-selection.v1",
            "provider_namespace": search.search_space.provider_namespace,
            "domain_version": search.search_space.domain_version,
            "provider_semantic_fingerprint": (search.search_space.provider_semantic_fingerprint),
            "compileiq_capability_id": search.capability["capability_id"],
            "compileiq_core_commit": search.capability["core_commit"],
            "compileiq_core_lock": search.capability["core_lock"],
            "domain_fingerprint": search.domain_fingerprint,
            "core_recipe_token": token_by_recipe[recipe_id],
            "recipe_id": recipe_id,
        }
        for param_id, recipe_id in enumerate(recipe_ids, start=1)
    ]
    return SimpleNamespace(
        opaque_recipe_capability=dict(search.capability),
        opaque_recipe_core_provenance={
            "core_commit": search.capability["core_commit"],
            "core_lock": search.capability["core_lock"],
        },
        opaque_recipe_audit_records=tuple(records),
    )


def test_segmented_scan_search_is_dynamic_baseline_inclusive_and_opaque(
    monkeypatch,
):
    _install_reviewed_fork(monkeypatch)
    monkeypatch.setattr(
        _compileiq_segmented_scan,
        "_segmented_scan_scope",
        lambda *_: _scope(),
    )
    search = compileiq_segmented_scan_search(object(), object(), object())

    assert isinstance(search, CompileIQSegmentedScanSearch)
    assert search.baseline_recipe_id == "segmented-scan:serial:v1"
    assert search.baseline_recipe_id in search.recipe_ids
    assert len(search.recipe_ids) == 2
    assert search.search_space.provider_namespace == ("taichi_forge.algorithms.segmented_scan")
    assert search.search_space.domain_version == ("cuda-integer-immutable-layout.v1")
    tokens = search.search_space.to_search_space()["recipe_id"].vals
    assert tokens == ["ciq-recipe-v1-0000", "ciq-recipe-v1-0001"]
    assert set(tokens).isdisjoint(search.recipe_ids)
    assert {search.select(_parameters(search, recipe_id)).method for recipe_id in search.recipe_ids} == {
        "serial",
        "global_scan",
    }
    assert isinstance(
        search.select(_parameters(search, search.baseline_recipe_id)),
        CompileIQSegmentedScanSelection,
    )
    manifest = search.manifest()
    assert manifest["recipe_count"] == 2
    assert sum(recipe["is_baseline"] for recipe in manifest["recipes"]) == 1
    assert manifest["runtime_admission"] == ("explicit_selection_only_no_auto_policy_mutation")
    mutable_scope = dict(search.scope)
    mutable_scope["layout"]["capacity"] = 1
    assert search.scope["layout"]["capacity"] == 4096
    json.dumps(manifest)

    monkeypatch.setattr(
        _compileiq_segmented_scan,
        "_segmented_scan_scope",
        lambda *_: _scope(baseline_method="global_scan"),
    )
    global_search = CompileIQSegmentedScanSearch(object(), object(), object())
    assert global_search.baseline_recipe_id == ("segmented-scan:cuda-global-scan:v1")


def test_segmented_scan_search_requires_complete_audited_coverage(monkeypatch):
    _install_reviewed_fork(monkeypatch)
    monkeypatch.setattr(
        _compileiq_segmented_scan,
        "_segmented_scan_scope",
        lambda *_: _scope(),
    )
    search = CompileIQSegmentedScanSearch(object(), object(), object())
    candidate = next(recipe_id for recipe_id in search.recipe_ids if recipe_id != search.baseline_recipe_id)
    incomplete = _compileiq_search_audit(search, (search.baseline_recipe_id,))

    assert not search.search_coverage(incomplete)["complete"]
    with pytest.raises(RuntimeError, match="complete frozen segmented-scan"):
        search.require_complete_search(incomplete)

    complete = _compileiq_search_audit(search, search.recipe_ids)
    result = SimpleNamespace(
        get_best_result=lambda: {
            "params": _parameters(search, candidate),
            "score_1": 0.9,
        }
    )
    assert search.select_best_result(complete, result).method == "global_scan"


def test_segmented_scan_execute_is_exact_and_keeps_auto_unchanged(monkeypatch):
    _install_reviewed_fork(monkeypatch)
    scope = _scope(inclusive=False)
    monkeypatch.setattr(
        _compileiq_segmented_scan,
        "_segmented_scan_scope",
        lambda *_: copy_scope(scope),
    )
    calls = []
    monkeypatch.setattr(
        _compileiq_segmented_scan,
        "experimental_segmented_scan",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    search = CompileIQSegmentedScanSearch(object(), object(), object(), inclusive=False)
    candidate = next(recipe_id for recipe_id in search.recipe_ids if recipe_id != search.baseline_recipe_id)
    selection = search.execute(
        _parameters(search, candidate),
        object(),
        object(),
        object(),
    )
    assert selection.method == "global_scan"
    assert calls[0][1]["method"] == "global_scan"
    assert calls[0][1]["inclusive"] is False
    assert search.manifest()["scope"]["source_method"] == "auto"


def copy_scope(value):
    return json.loads(json.dumps(value))


def test_segmented_scan_qualification_is_explicit_only(monkeypatch):
    _install_reviewed_fork(monkeypatch)
    monkeypatch.setattr(
        _compileiq_segmented_scan,
        "_segmented_scan_scope",
        lambda *_: _scope(),
    )
    search = CompileIQSegmentedScanSearch(object(), object(), object())
    recipe_id = next(value for value in search.recipe_ids if value != search.baseline_recipe_id)
    finalist = search.final_candidate(recipe_id)
    scope = _CompileIQWinnerScope(
        final_candidate_id=finalist.identity,
        forge_specialization_id=search.domain_fingerprint,
        workload_profile_id="integer-segmented-scan-v1",
        shape_scope_id="items=4096:segments=64:max=64",
        replay_scope_id="fresh-process-abba-v1",
        runtime_scope_id="cuda:uuid:driver",
        compiler_scope_id="llvm20:driver-jit",
        provider_scope_id="serial-vs-global-scan",
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
    assert decision.selected_provider_candidate_id == "global_scan"
    assert search.manifest()["runtime_admission"].endswith("no_auto_policy_mutation")


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_segmented_scan_search_materializes_both_routes_and_fails_on_drift(
    monkeypatch,
):
    _install_reviewed_fork(monkeypatch)
    n = 4096
    offsets = np.arange(0, n + 1, 64, dtype=np.int32)
    values_np = (np.arange(n, dtype=np.int32) % 7) + 1
    expected = np.empty_like(values_np)
    for begin, end in zip(offsets[:-1], offsets[1:]):
        expected[begin:end] = np.cumsum(values_np[begin:end], dtype=np.int32)
    values = ti.ndarray(ti.i32, shape=n)
    output = ti.ndarray(ti.i32, shape=n)
    values.from_numpy(values_np)
    layout = ti.algorithms.SegmentedLayout.from_offsets(offsets)
    search = compileiq_segmented_scan_search(values, layout, output)

    assert search.baseline_recipe_id == "segmented-scan:serial:v1"
    auto_output = ti.ndarray(ti.i32, shape=n)
    auto_workspace = ti.algorithms.SegmentedWorkspace(
        max_items=n,
        max_segments=len(offsets) - 1,
    )
    ti.algorithms.experimental_segmented_scan(
        values,
        layout,
        auto_output,
        method="auto",
        workspace=auto_workspace,
    )
    ti.sync()
    assert auto_workspace.last_scan_method == "serial"
    observed = {}
    for recipe_id in search.recipe_ids:
        workspace = ti.algorithms.SegmentedWorkspace(max_items=n, max_segments=len(offsets) - 1)
        selection = search.execute(
            _parameters(search, recipe_id),
            values,
            layout,
            output,
            workspace=workspace,
        )
        ti.sync()
        observed[selection.method] = (
            np.array_equal(output.to_numpy(), expected),
            workspace.last_scan_method,
        )
    assert observed == {
        "serial": (True, "serial"),
        "global_scan": (True, "global_scan"),
    }

    drifted = ti.algorithms.SegmentedLayout.from_offsets(np.asarray([0, 32, n], dtype=np.int32))
    with pytest.raises(ValueError, match="does not match the frozen domain"):
        search.execute(
            _parameters(search, search.baseline_recipe_id),
            values,
            drifted,
            output,
        )


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_segmented_scan_search_materializes_u32_exclusive_and_rejects_in_place(
    monkeypatch,
):
    _install_reviewed_fork(monkeypatch)
    n = 4096
    offsets = np.arange(0, n + 1, 64, dtype=np.int32)
    values_np = ((np.arange(n, dtype=np.uint32) % 11) + 1).astype(np.uint32)
    expected = _expected_segmented(values_np, offsets, inclusive=False)
    values = ti.ndarray(ti.u32, shape=n)
    output = ti.ndarray(ti.u32, shape=n)
    values.from_numpy(values_np)
    layout = ti.algorithms.SegmentedLayout.from_offsets(offsets)
    search = compileiq_segmented_scan_search(
        values,
        layout,
        output,
        inclusive=False,
    )

    for recipe_id in search.recipe_ids:
        workspace = ti.algorithms.SegmentedWorkspace(
            max_items=n,
            max_segments=len(offsets) - 1,
        )
        search.execute(
            _parameters(search, recipe_id),
            values,
            layout,
            output,
            workspace=workspace,
        )
        ti.sync()
        assert np.array_equal(output.to_numpy(), expected)

    with pytest.raises(
        CompileIQSegmentedScanUnavailableError,
        match="disjoint plain 1D i32/u32",
    ):
        compileiq_segmented_scan_search(
            values,
            layout,
            values,
            inclusive=False,
        )


def _expected_segmented(values, offsets, *, inclusive):
    expected = np.empty_like(values)
    for begin, end in zip(offsets[:-1], offsets[1:]):
        segment = np.cumsum(values[begin:end], dtype=values.dtype)
        if inclusive:
            expected[begin:end] = segment
        else:
            expected[begin] = 0
            expected[begin + 1 : end] = segment[:-1]
    return expected


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_segmented_scan_search_baseline_matches_large_auto_and_rejects_float(
    monkeypatch,
):
    _install_reviewed_fork(monkeypatch)
    n = 65536
    offsets = np.arange(0, n + 1, 4096, dtype=np.int32)
    values = ti.ndarray(ti.i32, shape=n)
    output = ti.ndarray(ti.i32, shape=n)
    layout = ti.algorithms.SegmentedLayout.from_offsets(offsets)
    search = compileiq_segmented_scan_search(values, layout, output)
    assert search.baseline_recipe_id == "segmented-scan:cuda-global-scan:v1"
    auto_workspace = ti.algorithms.SegmentedWorkspace(
        max_items=n,
        max_segments=len(offsets) - 1,
    )
    ti.algorithms.experimental_segmented_scan(
        values,
        layout,
        output,
        method="auto",
        workspace=auto_workspace,
    )
    ti.sync()
    assert auto_workspace.last_scan_method == "global_scan"

    float_values = ti.ndarray(ti.f32, shape=n)
    float_output = ti.ndarray(ti.f32, shape=n)
    with pytest.raises(
        CompileIQSegmentedScanUnavailableError,
        match="plain 1D i32/u32",
    ):
        compileiq_segmented_scan_search(
            float_values,
            layout,
            float_output,
        )
