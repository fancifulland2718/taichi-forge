"""Variant coverage, actual code reuse, bounded builds and cold selection."""

import json
import os
import struct
import subprocess
import sys
from zipfile import ZipFile

import pytest
import taichi_forge as ti
from taichi_forge.aot import BuildBudget, Component, ComponentBuildError, RootOption, Variant, build_variants
from taichi_forge.lang import kernel_impl
from taichi_forge_runtime.aot import Runtime, inspect_component_variants
from tests import test_utils
from tests.python.test_aot_component_configuration import _ConfiguredModel, _configured


def test_package_variant_frontier_keeps_one_pending_copy_per_selection(monkeypatch):
    from collections import deque
    from types import SimpleNamespace
    from taichi_forge.aot import _package_variants as variants
    from taichi_forge_runtime._static_values import static_key

    queued = set()

    class UniqueFrontier(deque):
        def append(self, selection):
            identity = tuple(sorted(selection.items()))
            assert identity not in queued, "Repeated selections retain duplicate frontier work"
            queued.add(identity)
            super().append(selection)

        def __init__(self, items):
            super().__init__()
            for item in items:
                self.append(item)

    monkeypatch.setattr(variants, "deque", UniqueFrontier)
    owner = SimpleNamespace(a=True, b=True, c=True, unused=True)
    schema = {name: dict(kind="static", choices=[static_key(False), static_key(True)])
              for name in vars(owner)}
    calls = []

    class Compiler:
        def compile(self, entry, *args, owner, template_args):
            reads = ("a", "b", "c") if owner.a else ("a",)
            captured = {name: static_key(getattr(owner, name)) for name in reads}
            calls.append(captured)
            return SimpleNamespace(static=captured, objects={})

    codes = variants.compile_static_variants(Compiler(), None, (), owner, schema)
    assert len(codes) == len(calls) == 5
    assert len(queued) <= 8 and all(not dict(item)["unused"] for item in queued)
    assert vars(owner) == dict(a=True, b=True, c=True, unused=True)
    assert {(tuple(code.static["b"]), tuple(code.static["c"])) for code in codes if "b" in code.static} == {
        (tuple(static_key(b)), tuple(static_key(c))) for b in (False, True) for c in (False, True)
    }


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_candidate_static_rejection_precedes_storage_validation(monkeypatch):
    from taichi_forge.aot import _component

    owner, environment = _ConfiguredModel(7), _configured()
    code = environment.compile(owner.evaluate)

    def storage_validation(*args, **kwargs):
        raise ValueError("storage validation is still required for matching candidates")

    monkeypatch.setattr(_component, "_dimension_value", storage_validation)
    owner.scale = False
    assert code.dependency_mismatch(environment, owner) == "static value 'scale'"
    owner.scale = True
    with pytest.raises(ValueError, match="storage validation is still required"):
        code.dependency_mismatch(environment, owner)


@ti.data_oriented
class _VariantModel(_ConfiguredModel):
    def record_options(self, builder, *, include_capacity=False):
        builder.dispatch(self.evaluate)
        if include_capacity:
            builder.dispatch(self.capacity_pass)

    def record_configuration(self, builder, *, configuration):
        builder.dispatch(self.evaluate)


def _requests():
    env = Component(**_configured().declarations, include_capacity=RootOption(choices=(False, True)))
    first, other = _VariantModel(7), _VariantModel(37, unused_mode="b")
    return [
        Variant(env, first.record_options),
        Variant(env, first.record_options, include_capacity=True),
        Variant(env, other.record_options, include_capacity=True),
    ]


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_variant_family_collects_distinct_declared_domains_without_cartesian_coverage(tmp_path):
    models = [_VariantModel(7, unused_mode=mode) for mode in ("a", "b")]
    built = build_variants([
        Variant(_configured(scale=(True,), unused=(model.unused_mode,)), model.record)
        for model in models
    ])
    assert len(built.report["kernel_variants"]) == 1
    module = ti.aot.Module()
    manifest = built.export(module, "collected")
    assert manifest["version"] == 2
    assert manifest["required_capabilities"] == ["component_variant_selection_v2"]
    assert len(manifest["cases"]) == 2
    assert len(manifest["exported_kernel_entries"]) == 1
    artifact = tmp_path / "collected.tcm"
    module.archive(str(artifact))
    built.close()
    info = inspect_component_variants(artifact, "collected")
    with Runtime("cpu") as runtime, runtime.load_module(artifact) as loaded:
        family = loaded.component_variants("collected")
        for mode in ("a", "b"):
            configuration = dict(scale=True, extra_stage=False, unused_mode=mode)
            code = family.select(configuration)
            assert code.contract == info.select(configuration).contract
            assert code.contract["configuration"]["unused_mode"]["choices"] == [["str", mode]]
            with code.instantiate(dict(capacity=11, active=9), configuration=configuration) as instance:
                instance.initialize(zero=code.contract["resources"])
                with instance.prepare(values=dict(gravity=[3, 0, 0])) as call:
                    call.run()
                assert struct.unpack("<11f", instance.resource("x").read()) == (6,) * 9 + (0,) * 2
        for selector in (info, family):
            with pytest.raises(ValueError, match="No exported component variant.*extra_stage"):
                selector.select(dict(scale=True, extra_stage=True, unused_mode="a"))
            with pytest.raises(ValueError, match="outside its declared domain"):
                selector.select(dict(scale=1, extra_stage=False, unused_mode="a"))
    # A family cannot broaden its domain independently of its compiled roots.
    invalid = tmp_path / "invented_domain.tcm"
    with ZipFile(artifact) as source, ZipFile(invalid, "w") as target:
        for name in source.namelist():
            data = source.read(name)
            if name == "component_families/collected.json":
                facts = json.loads(data)
                facts["domain"]["unused_mode"].append(["str", "invented"])
                data = json.dumps(facts)
            target.writestr(name, data)
    with pytest.raises(ValueError, match="domain differs from its actual compiled roots"):
        inspect_component_variants(invalid, "collected")
    with Runtime("cpu") as runtime, runtime.load_module(invalid) as loaded:
        with pytest.raises(ValueError, match="domain differs from its actual compiled roots"):
            loaded.component_variants("collected")


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_build_has_no_implicit_configuration_or_kernel_limit():
    # Cross the former default configuration ceiling while compiling just one
    # numerical entry. Host choices still have an explicit finite legal domain.
    choices = tuple(range(65))
    env = Component(**_configured().declarations, configuration=RootOption(choices=choices))
    model = _VariantModel(2)
    built = build_variants(Variant(env, model.record_configuration, configuration=value) for value in choices)
    report = built.report
    assert len(report["completed_configurations"]) == 65
    assert len(report["kernel_variants"]) == 1
    assert next(iter(report["kernel_variants"].values()))["uses"] == 65
    assert report["budget"] == dict(max_configurations=None, max_kernel_variants=None, max_seconds=None)
    assert not report["missing_configurations"]


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_variant_coverage_uses_existing_code_and_execution_plans(tmp_path):
    built = build_variants(_requests(), budget=BuildBudget(max_configurations=3, max_kernel_variants=2))
    assert len(built.report["kernel_variants"]) == 2
    assert sorted(row["uses"] for row in built.report["kernel_variants"].values()) == [2, 3]
    assert not built.report["missing_configurations"]
    module = ti.aot.Module()
    manifest = built.export(module, "configured")
    assert len(manifest["cases"]) == 3 and len(manifest["exported_kernel_entries"]) == 2
    artifact = tmp_path / "variants.tcm"
    module.archive(str(artifact))
    built.close()
    built.close()
    with pytest.raises(RuntimeError, match="family is closed"):
        built.select({})
    with pytest.raises(RuntimeError, match="family is closed"):
        built.export(module, "closed_family")
    info = inspect_component_variants(artifact, "configured")
    assert info.target == "x64"
    assert info.contract == manifest
    with Runtime("cpu") as runtime:
        loaded = runtime.load_module(artifact)
        variants = loaded.component_variants("configured")
        assert variants.configuration_sources() == info.configuration_sources()
        base = dict(scale=True, extra_stage=False, unused_mode="a", include_capacity=False)
        with pytest.raises(ValueError, match="No exported component variant"):
            variants.select({**base, "unused_mode": "b"})
        with pytest.raises(ValueError, match="outside its declared domain"):
            variants.select({**base, "scale": 1})
        for mode, capacity in (("a", False), ("b", True)):
            configuration = {**base, "unused_mode": mode, "include_capacity": capacity}
            selected = variants.select(configuration)
            assert info.select(configuration).contract == selected.contract
            with variants.instantiate(dict(capacity=41, active=37), configuration=configuration) as instance:
                instance.initialize(zero=selected.contract["resources"])
                with instance.prepare(values=dict(gravity=[5, 7, 9])) as call:
                    call.run()
                assert struct.unpack("<41f", instance.resource("x").read()) == tuple([10] * 37 + [0] * 4)
                if capacity:
                    assert struct.unpack("<41f", instance.resource("y").read()) == tuple(range(41))
        loaded.close()
        with pytest.raises(RuntimeError, match="closed"):
            variants.select(base)
    # JSON formatting is not an ABI limit. Reuse the compiled artifact to cross
    # the former 4 MiB ceiling without generating thousands of numerical entries.
    large = tmp_path / "large_contract.tcm"
    with ZipFile(artifact) as source, ZipFile(large, "w") as target:
        for name in source.namelist():
            data = source.read(name)
            if name == "component_families/configured.json":
                data += b"\n" * (4 * 1024 * 1024)
            target.writestr(name, data)
    with Runtime("cpu") as runtime:
        assert len(runtime.load_module(large).component_variants("configured").contract["cases"]) == 3
    # The selector validates references/coverage against actual root contracts.
    bad = tmp_path / "inconsistent.tcm"
    with ZipFile(artifact) as source, ZipFile(bad, "w") as target:
        for name in source.namelist():
            data = source.read(name)
            if name == "component_families/configured.json":
                contract = json.loads(data)
                contract["cases"][0]["semantic_key"] = "wrong"
                data = json.dumps(contract)
            target.writestr(name, data)
    with Runtime("cpu") as runtime:
        with pytest.raises(ValueError, match="actual compiled root"):
            runtime.load_module(bad).component_variants("configured")
    with pytest.raises(ValueError, match="actual compiled root"):
        inspect_component_variants(bad, "configured")


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_family_inspection_preserves_cold_selection_and_source_provenance(tmp_path):
    built = build_variants(_requests())
    module = ti.aot.Module()
    manifest = built.export(module, "configured")
    artifact = tmp_path / "variants.tcm"
    module.archive(str(artifact), shared_component_contracts=True, shared_execution_plans=True)
    info = inspect_component_variants(artifact, "configured")
    base = dict(scale=True, extra_stage=False, unused_mode="a", include_capacity=False)
    with pytest.raises(ValueError, match="configuration inputs differ"):
        info.select({"scale": True})
    with pytest.raises(ValueError, match="outside its declared domain"):
        info.select({**base, "scale": 1})
    with pytest.raises(ValueError, match="No exported component variant"):
        info.select({**base, "unused_mode": "b"})
    sources = info.configuration_sources()
    sources["scale"]["path"].append("changed")
    assert info.configuration_sources()["scale"]["path"] == ["scale"]
    info.contract["cases"].clear()
    assert len(info.contract["cases"]) == 3
    with ZipFile(artifact) as archive:
        directory = tmp_path / "directory"
        archive.extractall(directory)
    assert inspect_component_variants(directory, "configured").select(base).contract == info.select(base).contract
    # A fresh consumer cannot import the numerical frontend or load native code.
    probe = r'''
import ctypes, importlib.abc, sys
class NoFrontend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('taichi', 'taichi_forge'):
            raise AssertionError('numerical frontend imported')
sys.meta_path.insert(0, NoFrontend())
def no_native(*args, **kwargs):
    raise AssertionError('native library loaded')
ctypes.CDLL = no_native
if hasattr(ctypes, 'WinDLL'):
    ctypes.WinDLL = no_native
from taichi_forge_runtime.aot import inspect_component_variants
family = inspect_component_variants(sys.argv[1], 'configured')
configuration = dict(scale=True, extra_stage=False, unused_mode='a', include_capacity=False)
assert family.configuration_sources()['scale']['path'] == ['scale']
family.select(configuration).check_configuration(configuration)
'''
    import taichi_forge_runtime
    from pathlib import Path
    env = dict(os.environ, PYTHONPATH=str(Path(taichi_forge_runtime.__file__).resolve().parent.parent))
    subprocess.run([sys.executable, "-B", "-c", probe, str(artifact)], env=env,
                   check=True, capture_output=True, text=True)

    family_path = directory / "component_families/configured.json"
    changed = json.loads(json.dumps(manifest))
    changed["build_report"]["completed_configurations"].pop()
    family_path.write_text(json.dumps(changed), encoding="utf8")
    with pytest.raises(ValueError, match="coverage"):
        inspect_component_variants(directory, "configured")
    family_path.write_text(json.dumps(manifest), encoding="utf8")
    # Use ordinary decoded documents to isolate provenance from storage checks.
    from taichi_forge_runtime._static_values import read_static_key
    for case in manifest["cases"]:
        values = {name: read_static_key(value) for name, value in case["configuration"].items()}
        facts = info.select(values).contract
        if case is manifest["cases"][0]:
            facts["compilation"]["configuration_sources"]["scale"]["path"] = ["other_scale"]
        (directory / "components" / (case["root"] + ".json")).write_text(json.dumps(facts), encoding="utf8")
    changed_info = inspect_component_variants(directory, "configured")
    changed_info.select(base).check_configuration(base)
    with pytest.raises(ValueError, match="different configuration source paths"):
        changed_info.configuration_sources()
    # Do not weaken loaded Plan/kernel validation to match the cold query scope.
    changed = json.loads(json.dumps(manifest))
    changed["exported_kernel_entries"] = []
    family_path.write_text(json.dumps(changed), encoding="utf8")
    inspect_component_variants(directory, "configured").select(base)
    with Runtime("cpu") as runtime, runtime.load_module(directory) as loaded:
        with pytest.raises(ValueError, match="coverage"):
            loaded.component_variants("configured")


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_build_budget_reports_missing_cases_and_reuses_complete_definitions(monkeypatch):
    requests = _requests()
    lowerings = []
    original = kernel_impl.transform_tree

    def observe(tree, ctx):
        if ctx.is_kernel:
            lowerings.append(ctx.func.func.__name__)
        return original(tree, ctx)

    monkeypatch.setattr(kernel_impl, "transform_tree", observe)
    with pytest.raises(ComponentBuildError, match="kernel variant budget") as caught:
        build_variants(requests, budget=BuildBudget(max_kernel_variants=1))
    report = caught.value.report
    assert report["request_admission_complete"]
    assert report["blocked_variant"]["entry"][1].endswith("capacity_pass")
    assert len(report["completed_configurations"]) == 1 and len(report["missing_configurations"]) == 2
    assert lowerings == ["evaluate", "capacity_pass"]
    retry = build_variants(requests, budget=BuildBudget(max_kernel_variants=2))
    assert len(retry.report["kernel_variants"]) == 2 and not retry.report["missing_configurations"]
    assert lowerings == ["evaluate", "capacity_pass"]
    with pytest.raises(ComponentBuildError, match="configuration budget") as caught:
        build_variants(requests, budget=BuildBudget(max_configurations=1))
    assert not caught.value.report["request_admission_complete"]
    with pytest.raises(ComponentBuildError, match="time budget"):
        build_variants(requests, budget=BuildBudget(max_seconds=1e-12))
    with pytest.raises(ComponentBuildError, match="recording argument 'include_capacity'.*outside its declared domain"):
        build_variants([Variant(requests[0].component, requests[0].record, include_capacity=1)])
