"""Declared record paths keep numeric resources separate from static presence."""

from dataclasses import dataclass
import struct

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.aot._component import Component, DenseField, Dimension, Record, RuntimeValue, StaticValue
from taichi_forge_runtime.aot import Runtime
from tests import test_utils


@dataclass(frozen=True, slots=True)
class _Buffers:
    capacity: int
    active: int
    x: object
    alias: object
    y: object
    gain: float
    unused: int = 0


@ti.data_oriented
class _Model:
    def __init__(self, capacity, active, gain=2.0, *, present=True):
        self.capacity = capacity
        self.output = ti.field(ti.f32)
        builder = ti.FieldsBuilder()
        builder.dense(ti.i, capacity).place(self.output)
        self.output_tree = builder.finalize()
        self.buffers = None
        if present:
            x, y = ti.field(ti.f32), ti.field(ti.f32)
            builder = ti.FieldsBuilder()
            builder.dense(ti.i, capacity).place(x)
            builder.dense(ti.i, capacity).place(y)
            self.tree = builder.finalize()
            self.buffers = _Buffers(capacity, active, x, x, y, gain)

    @ti.func
    def value(self, i):
        return self.buffers.alias[i] + self.buffers.gain

    @ti.kernel
    def step(self):
        if ti.static(self.buffers is not None):
            for i in range(self.buffers.active):
                self.buffers.x[i] = i
                self.buffers.y[i] = self.value(i)
                self.output[i] = self.buffers.y[i]
        else:
            self.output[0] = -1

    def record(self, builder):
        builder.dispatch(self.step)

    @ti.kernel
    def object_identity(self):
        if ti.static(self.buffers is self.buffers):
            self.output[0] = 42


def _environment(*, extra=False):
    return Component(
        capacity=Dimension(),
        output=DenseField(ti.f32, ("capacity",), "output"),
        buffers=Record(
            optional=True,
            capacity=Dimension(),
            active=Dimension("active", "capacity"),
            x=DenseField(ti.f32, ("capacity",), "state"),
            alias=DenseField(ti.f32, ("capacity",), "state"),
            y=DenseField(ti.f32, ("capacity",), "state"),
            gain=RuntimeValue(ti.f32),
            **({"unused": StaticValue(choices=(0, 1))} if extra else {}),
        ),
    )


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_record_capture_preserves_dynamic_leaves_and_optional_variant(tmp_path):
    first, second = _Model(7, 5), _Model(19, 17, gain=3)
    environment = _environment()
    code = environment.compile(first.step)
    assert _environment(extra=True).compile(second.step) is code
    assert code.contract["objects"] == [["buffers", True]]
    assert not any("unused" in path for path in code.contract["access_paths"].values())
    code.run(environment, second)
    np.testing.assert_array_equal(second.output.to_numpy(), list(np.arange(17) + 3) + [0, 0])
    absent = _Model(7, 0, present=False)
    alternate = environment.compile(absent.step)
    assert alternate.semantic_key != code.semantic_key
    alternate.run(environment, absent)
    assert absent.output[0] == -1
    assert "configuration" not in _environment(extra=True).compile_graph(absent.record).contract
    with pytest.raises(ValueError, match="dependencies"):
        code.run(environment, absent)
    with pytest.raises(ti.TaichiCompilationError, match="presence comparisons"):
        environment.compile(first.object_identity)

    # An unchecked property must not become a new compile-time evaluation path.
    class PropertyRecord:
        active = 17

        @property
        def capacity(self):
            raise AssertionError("must not execute this getter")

    second.buffers = PropertyRecord()
    with pytest.raises(ti.TaichiCompilationError, match="descriptor"):
        environment.compile(second.step)

    root = environment.compile_graph(first.record)
    bounded = _environment(extra=True).compile_graph(first.record)
    assert bounded.contract["semantic_key"] == root.contract["semantic_key"]
    assert len(bounded.contract["configuration"]) == 1
    facts = root.contract["compilation"]["entries"][0]
    assert facts["record_presence"] == {"buffers": True}
    assert ["buffers", "gain"] in facts["capture_paths"].values()
    module = ti.aot.Module()
    root.export(module, "record_root")
    path = tmp_path / "record.tcm"
    module.archive(str(path))
    backend = "cuda" if ti.cfg.arch == ti.cuda else "cpu"
    dimensions = {
        name: (17 if "active" in name else 19)
        for name, schema in root.contract["dimensions"].items()
        if len(schema) == 4
    }
    ti.reset()
    with Runtime(backend) as runtime:
        component = runtime.load_module(path).component("record_root")
        with component.instantiate(dimensions) as instance:
            instance.initialize(zero=component.contract["resources"])
            with instance.prepare(values={name: 5 for name in component.contract["values"]}) as prepared:
                prepared.run()
            assert struct.unpack("<19f", instance.resource("output").read()) == tuple(list(range(5, 22)) + [0, 0])


class _DeclaredBuffers:
    def __init__(self, capacity, active, gain):
        self.capacity, self.active, self.gain = capacity, active, gain
        builder = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=builder)
        scope.dim("active")
        scope.x = scope.field(ti.f32, shape=scope.dim("capacity"))
        scope.alias = self.x
        scope.y = scope.field(ti.f32, shape=scope.dim("capacity"))
        self.tree = builder.finalize()


@ti.data_oriented
class _DeclaredModel(_Model):
    def __init__(self, capacity, active, gain=2):
        self.capacity = capacity
        builder = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=builder)
        scope.output = scope.field(ti.f32, shape=scope.dim("capacity"))
        self.output_tree = builder.finalize()
        scope.buffers = _DeclaredBuffers(capacity, active, gain)


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_normal_child_declarations_form_the_original_numerical_closure():
    first, second = _DeclaredModel(7, 5), _DeclaredModel(19, 17, 9)
    environment = Component.from_owner(first, buffers=Record(gain=RuntimeValue(ti.f32)))
    other = Component.from_owner(second, buffers=Record(gain=RuntimeValue(ti.f32)))
    code = environment.compile(first.step)
    assert other.compile(second.step) is code
    code.run(other, second)
    np.testing.assert_array_equal(second.output.to_numpy(), list(range(9, 26)) + [0, 0])
    assert len(code.contract["groups"]) == 2
    # One explicit declaration edge cannot recursively enumerate an owner cycle.
    ti.experimental.ResourceScope(first.buffers).parent = first
    with pytest.raises(ti.TaichiCompilationError, match="acyclic"):
        Component.from_owner(first, buffers=Record(gain=RuntimeValue(ti.f32)))
