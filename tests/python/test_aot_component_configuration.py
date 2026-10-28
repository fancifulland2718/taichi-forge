"""Finite root admission is separate from actual per-kernel specialization."""

import struct

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.aot._component import Component, StaticValue
from taichi_forge.lang import kernel_impl
from taichi_forge_runtime.aot import Runtime
from tests import test_utils
from tests.python.test_aot_component_dense import _DenseModel, _environment


@ti.data_oriented
class _ConfiguredModel(_DenseModel):
    def __init__(self, n, *, scale=True, extra_stage=False, unused_mode="a"):
        super().__init__(n, n - 1)
        self.scale, self.extra_stage, self.unused_mode = scale, extra_stage, unused_mode

    @ti.kernel
    def evaluate(self):
        for i in range(self.active):
            value = self.gravity[0]
            if ti.static(self.scale):
                value *= 2
            self.x[i] = value

    def record(self, builder):
        builder.dispatch(self.evaluate)
        if self.extra_stage:
            builder.dispatch(self.capacity_pass)

    def mutating_record(self, builder):
        builder.dispatch(self.evaluate)
        self.unused_mode = "b"


def _configured(*, scale=(False, True), unused=("a", "b")):
    return Component(
        **_environment().declarations,
        scale=StaticValue(choices=scale),
        extra_stage=StaticValue(choices=(False, True)),
        unused_mode=StaticValue(choices=unused),
    )


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_actual_specializations_and_python_root_configuration(tmp_path, monkeypatch):
    lowerings = []
    original = kernel_impl.transform_tree

    def observe(tree, ctx):
        if ctx.is_kernel:
            lowerings.append(ctx.func.func.__name__)
        return original(tree, ctx)

    first, other = _ConfiguredModel(7), _ConfiguredModel(37, unused_mode="b")
    monkeypatch.setattr(kernel_impl, "transform_tree", observe)
    env = _configured()
    enabled = env.compile_graph(first.record)
    reused = env.compile_graph(other.record)
    widened = _configured(unused=("a", "b", "c")).compile_graph(other.record)
    assert enabled.contract["semantic_key"] == reused.contract["semantic_key"] == widened.contract["semantic_key"]
    assert lowerings == ["evaluate"]
    assert dict(enabled.contract["entries"][0]["static"]) == {"scale": ["bool", True]}
    assert "extra_stage" in enabled.contract["configuration"]  # Host construction choice is still admitted.
    first.extra_stage = True
    extended = env.compile_graph(first.record)
    assert extended.contract["semantic_key"] != enabled.contract["semantic_key"]
    assert extended.contract["entries"][0]["semantic_key"] == enabled.contract["entries"][0]["semantic_key"]
    assert lowerings == ["evaluate", "capacity_pass"]
    first.scale = False
    disabled = env.compile_graph(first.record)
    assert lowerings == ["evaluate", "capacity_pass", "evaluate"]
    assert disabled.contract["entries"][1]["semantic_key"] == extended.contract["entries"][1]["semantic_key"]
    # Two valid finite choices share unrelated compiled entries; no Cartesian
    # enumeration or whole-model fingerprint drives specialization.
    module = ti.aot.Module()
    extended.export(module, "enabled")
    disabled.export(module, "disabled")
    artifact = tmp_path / "variants.tcm"
    module.archive(str(artifact))
    with Runtime("cpu") as runtime:
        loaded = runtime.load_module(artifact)
        component = loaded.component("enabled")
        dims = dict(capacity=41, active=37)
        config = dict(scale=True, extra_stage=True, unused_mode="a")
        with pytest.raises(ValueError, match="configuration inputs differ"):
            component.instantiate(dims)
        with pytest.raises(ValueError, match="different compiled variant"):
            component.instantiate(dims, configuration={**config, "scale": False})
        with pytest.raises(ValueError, match="outside its declared domain"):
            component.instantiate(dims, configuration={**config, "scale": 1})
        with pytest.raises(ValueError, match="different compiled variant"):
            component.instantiate(dims, configuration={**config, "extra_stage": False})
        for name, factor in (("enabled", 2), ("disabled", 1)):
            selected = loaded.component(name)
            with selected.instantiate(dims, configuration={**config, "scale": name == "enabled"}) as instance:
                instance.initialize(zero=selected.contract["resources"])
                with instance.prepare(values=dict(gravity=[5, 7, 9])) as call:
                    call.run()
                actual = struct.unpack("<41f", instance.resource("x").read())
                np.testing.assert_array_equal(actual, [5 * factor] * 37 + [0] * 4)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_domain_rejection_on_cached_and_cold_paths():
    model = _ConfiguredModel(7)
    code = _configured().compile(model.evaluate)
    with pytest.raises(ti.TaichiCompilationError, match="self.scale.*outside its declared domain"):
        _configured(scale=(False,)).compile(model.evaluate)
    with pytest.raises(ti.TaichiCompilationError, match="self.scale.*outside its declared domain"):
        code.run(_configured(scale=(False,)), model)
    model.scale = 2
    count = len(model.evaluate._primal._component_candidates)
    with pytest.raises(ti.TaichiCompilationError, match="self.scale.*outside its declared domain"):
        _configured().compile(model.evaluate)
    assert len(model.evaluate._primal._component_candidates) == count
    model.scale = False
    assert _configured().compile(model.evaluate).semantic_key != code.semantic_key
    with pytest.raises(ti.TaichiCompilationError, match="bound normal recording method"):
        _configured().compile_graph(lambda builder: model.record(builder))
    with pytest.raises(ValueError, match="nonempty and distinct"):
        StaticValue(choices=(False, False))
    with pytest.raises(ti.TaichiCompilationError, match="configuration changed while recording"):
        _configured().compile_graph(model.mutating_record)
    assert _configured().compile_graph(model.record).contract["configuration"]["unused_mode"]["selected"] == ["str", "b"]
