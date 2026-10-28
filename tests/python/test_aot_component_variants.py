"""Variant coverage, actual code reuse, bounded builds and cold selection."""

import json
import struct
from zipfile import ZipFile

import pytest
import taichi_forge as ti
from taichi_forge.aot._component import Component, RootOption
from taichi_forge.aot._component_variants import BuildBudget, ComponentBuildError, Variant, build_variants
from taichi_forge.lang import kernel_impl
from taichi_forge_runtime.aot import Runtime
from tests import test_utils
from tests.python.test_aot_component_configuration import _ConfiguredModel, _configured


@ti.data_oriented
class _VariantModel(_ConfiguredModel):
    def record_options(self, builder, *, include_capacity=False):
        builder.dispatch(self.evaluate)
        if include_capacity:
            builder.dispatch(self.capacity_pass)


def _requests():
    env = Component(**_configured().declarations, include_capacity=RootOption(choices=(False, True)))
    first, other = _VariantModel(7), _VariantModel(37, unused_mode="b")
    return [
        Variant(env, first.record_options),
        Variant(env, first.record_options, include_capacity=True),
        Variant(env, other.record_options, include_capacity=True),
    ]


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
    with Runtime("cpu") as runtime:
        loaded = runtime.load_module(artifact)
        variants = loaded.component_variants("configured")
        base = dict(scale=True, extra_stage=False, unused_mode="a", include_capacity=False)
        with pytest.raises(ValueError, match="No exported component variant"):
            variants.select({**base, "unused_mode": "b"})
        with pytest.raises(ValueError, match="outside its declared domain"):
            variants.select({**base, "scale": 1})
        for mode, capacity in (("a", False), ("b", True)):
            configuration = {**base, "unused_mode": mode, "include_capacity": capacity}
            selected = variants.select(configuration)
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
