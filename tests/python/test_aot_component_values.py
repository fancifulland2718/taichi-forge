"""Normal value roles feed the existing capture ABI without numerical wrappers."""

from dataclasses import dataclass
import struct

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.aot._component import Component, Record, RuntimeValue, StaticValue
from taichi_forge_runtime.aot import Runtime
from tests import test_utils


@dataclass(frozen=True, slots=True)
class _Parameters:
    gain: float
    enabled: bool


@ti.data_oriented
class _Model:
    def __init__(self, n, *, gain=2.0, enabled=True, extra=False, kernel_only=False):
        self.n = n
        self.gravity = ti.Vector([3.0, 5.0, 7.0])
        self.parameters = _Parameters(gain, enabled)
        builder = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=builder)
        scope.output = scope.field(ti.f32, shape=scope.dim("n"))
        scope.value("gravity", ti.types.vector(3, ti.f32))
        record = scope.record("parameters", self.parameters)
        record.value("gain", ti.f32)
        record.static("enabled", choices=(False, True), kernel_only=kernel_only)
        if extra:
            self.unused = 123
            scope.value("unused", ti.i32)
        self.tree = builder.finalize()

    @ti.func
    def value(self, i):
        result = self.gravity[0] + i
        if ti.static(self.parameters.enabled):
            result *= self.parameters.gain
        return result

    @ti.kernel
    def step(self):
        for i in range(self.n):
            self.output[i] = self.value(i)

    def record(self, builder):
        builder.dispatch(self.step)


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_normal_roles_reuse_actual_capture_and_export_contract(tmp_path):
    first, second = _Model(3), _Model(19, gain=4, extra=True)
    environment, other = Component.from_owner(first), Component.from_owner(second)
    code = environment.compile(first.step)
    assert other.compile(second.step) is code
    assert "unused" not in code.dynamic
    assert type(first.parameters) is _Parameters
    assert type(second.parameters.gain) is int  # Declaration did not wrap/cast it.
    second.gravity[0] = 11
    code.run(other, second)
    expected = (11 + np.arange(19)) * 4
    np.testing.assert_array_equal(second.output.to_numpy(), expected)
    second.output.fill(0)
    second.step()
    np.testing.assert_array_equal(second.output.to_numpy(), expected)
    alternate = _Model(7, enabled=False)
    assert Component.from_owner(alternate).compile(alternate.step).semantic_key != code.semantic_key

    root = environment.compile_graph(first.record)
    assert "unused" not in root.contract["values"]
    module = ti.aot.Module()
    root.export(module, "normal")
    artifact = tmp_path / "normal.tcm"
    module.archive(str(artifact))
    backend = "cuda" if ti.cfg.arch == ti.cuda else "cpu"
    configuration = {name: True for name in root.contract["configuration"]}
    values = {name: ([13, 17, 19] if name == "gravity" else 5) for name in root.contract["values"]}
    dimensions = {name: 37 for name, schema in root.contract["dimensions"].items() if len(schema) == 4}
    ti.reset()
    with Runtime(backend) as runtime:
        component = runtime.load_module(artifact).component("normal")
        with component.instantiate(dimensions, configuration=configuration) as instance:
            instance.initialize(zero=component.contract["resources"])
            with instance.prepare(values=values) as call:
                call.run()
            actual = struct.unpack("<37f", instance.resource("output").read())
            np.testing.assert_array_equal(actual, (13 + np.arange(37)) * 5)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_normal_value_declarations_reject_conflicts_without_getters():
    model = _Model(3)
    scope = ti.experimental.ResourceScope(model)
    scope.value("gravity", ti.types.vector(3, ti.f32))  # Equivalent redeclaration is idempotent.
    with pytest.raises(ValueError, match="different value declaration"):
        scope.value("gravity", ti.f32)
    with pytest.raises(ValueError, match="resource or dimension role"):
        scope.value("n", ti.i32)
    with pytest.raises(ValueError, match="value role"):
        scope.dim("gravity")
    with pytest.raises(ValueError, match="Duplicate normal component declaration"):
        Component.from_owner(model, gravity=RuntimeValue(ti.f32))
    model.mode = 0
    with pytest.raises(ti.TaichiCompilationError, match="outside its declared domain"):
        scope.static("mode", choices=(False, True))
    with pytest.raises(ValueError, match="finite choices"):
        scope.static("mode", choices=None)
    assert "mode" not in scope._registry["values"]

    class PropertyOwner:
        @property
        def gain(self):
            raise AssertionError("must not execute this getter")

    with pytest.raises(ti.TaichiCompilationError, match="descriptor"):
        ti.experimental.ResourceScope(PropertyOwner()).value("gain", ti.f32)

    # Frozen records are read through built-in slots, never rewritten.
    record = scope.record("parameters", model.parameters)
    record.value("gain", ti.f32)
    with pytest.raises(TypeError, match="cannot modify"):
        record.gain = 2
    assert model.parameters.gain == 2.0


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_component_can_only_narrow_normal_static_domains():
    model = _Model(3)
    normal = Component.from_owner(model)
    code = normal.compile(model.step)
    narrowed = Component.from_owner(model, parameters=Record(enabled=StaticValue(choices=(True,))))
    assert narrowed.compile(model.step) is code
    with pytest.raises(ti.TaichiCompilationError, match="outside its declared domain"):
        Component.from_owner(model, parameters=Record(enabled=StaticValue(choices=(False,)))).compile(model.step)
    for choices in (None, (0, 1), (True, 2)):
        with pytest.raises(ValueError, match="must narrow"):
            Component.from_owner(model, parameters=Record(enabled=StaticValue(choices=choices)))
    with pytest.raises(ValueError, match="Duplicate normal component declaration"):
        Component.from_owner(model, parameters=Record(gain=StaticValue(choices=(2.0,))))


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_kernel_only_policy_preserves_used_guards_without_unrelated_domains(tmp_path):
    from taichi_forge.aot._component import RootOption
    from taichi_forge.aot._component_variants import ComponentBuildError, Variant, build_variants

    with pytest.raises(ValueError, match="Recording arguments"):
        RootOption(choices=(False, True), kernel_only=True)

    first, second = _Model(3, kernel_only=True), _Model(19, enabled=False, kernel_only=True)
    for index, model in enumerate((first, second)):
        model.other_layout = index + 7
        scope = ti.experimental.ResourceScope(model)
        scope.static("other_layout", choices=(model.other_layout,), kernel_only=True)
        scope.static("other_layout", choices=(model.other_layout,), kernel_only=True)
        with pytest.raises(ValueError, match="different value declaration"):
            scope.static("other_layout", choices=(model.other_layout,))
    environments = [Component.from_owner(model) for model in (first, second)]
    with pytest.raises(ValueError, match="must narrow"):
        Component.from_owner(first, parameters=Record(enabled=StaticValue(choices=(True,))))
    narrowed = Component.from_owner(first, parameters=Record(enabled=StaticValue(choices=(True,), kernel_only=True)))
    assert narrowed.compile(first.step) is environments[0].compile(first.step)
    family = build_variants([Variant(env, model.record) for env, model in zip(environments, (first, second))])
    assert len(family.report["kernel_variants"]) == 2
    for configuration in family.report["requested_configurations"]:
        assert "other_layout" not in configuration
        assert len(configuration) == 1  # The used numerical guard is required.
    module = ti.aot.Module()
    manifest = family.export(module, "normal")
    assert "other_layout" not in manifest["domain"]
    name = next(iter(manifest["domain"]))
    artifact = tmp_path / "kernel_only.tcm"
    module.archive(str(artifact))
    with Runtime("cpu") as runtime:
        loaded = runtime.load_module(artifact).component_variants("normal")
        for enabled in (False, True):
            component = loaded.select({name: enabled})
            dimensions = {key: 11 for key, schema in component.contract["dimensions"].items() if len(schema) == 4}
            with pytest.raises(ValueError, match="different compiled variant"):
                component.instantiate(dimensions, configuration={name: not enabled})
            with component.instantiate(dimensions, configuration={name: enabled}) as instance:
                instance.initialize(zero=component.contract["resources"])
                values = {key: ([13, 17, 19] if key == "gravity" else 5) for key in component.contract["values"]}
                with instance.prepare(values=values) as call:
                    call.run()
                actual = struct.unpack("<11f", instance.resource("output").read())
                np.testing.assert_array_equal(actual, (13 + np.arange(11)) * (5 if enabled else 1))

    duplicate = _Model(9, kernel_only=True)
    duplicate.other_layout = 99
    ti.experimental.ResourceScope(duplicate).static("other_layout", choices=(99,), kernel_only=True)
    duplicate_environment = Component.from_owner(duplicate)
    with pytest.raises(ComponentBuildError, match="ambiguous recorded roots"):
        build_variants([Variant(environments[0], first.record), Variant(duplicate_environment, duplicate.record)])
    # Failed family publication retires its Graphs, not reusable definitions.
    retry = build_variants([Variant(environments[0], first.record)])
    assert len(retry.report["kernel_variants"]) == 1
    assert duplicate_environment.compile(duplicate.step) is environments[0].compile(first.step)
