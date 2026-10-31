"""Repeated and nested normal recordings keep their private workspaces distinct."""
import struct

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.aot._component import Component
from taichi_forge_runtime.aot import Runtime
from tests import test_utils


@ti.kernel
def _seed7(scratch: ti.types.ndarray(ti.i32, 0)):
    scratch[None] = 7


@ti.kernel
def _seed11(scratch: ti.types.ndarray(ti.i32, 0)):
    scratch[None] = 11


@ti.kernel
def _reset(scratch: ti.types.ndarray(ti.i32, 0)):
    scratch[None] = 0


@ti.kernel
def _advance(scratch: ti.types.ndarray(ti.i32, 0)):
    scratch[None] += 1


@ti.data_oriented
class _Model:
    def __init__(self):
        builder = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=builder)
        scope.output = scope.field(ti.i32, shape=2)
        self.tree = builder.finalize()

    @ti.kernel
    def first(self, scratch: ti.types.ndarray(ti.i32, 0)):
        self.output[0] = scratch[None]

    @ti.kernel
    def second(self, scratch: ti.types.ndarray(ti.i32, 0)):
        self.output[1] = scratch[None]

    def record(self, builder):
        container = builder.create_sequential(private_scope=True)
        for _ in range(2):
            outer = container.create_sequential(private_scope=True)
            a = outer.private_ndarray("scratch", ti.i32, ())
            counter = outer.private_ndarray("counter", ti.i32, ())
            outer.dispatch(_seed7, a)
            outer.dispatch(_reset, counter)
            inner = outer.create_sequential(private_scope=True)
            b = inner.private_ndarray("scratch", ti.i32, ())
            inner.dispatch(_seed11, b)
            inner.dispatch(self.second, b, template_args={"self": self})
            inner.dispatch(_advance, counter)
            condition = outer.create_sequential()
            condition.dispatch(_seed7, a)
            outer.while_loop(condition, inner, predicate=a, counter=counter,
                             max_iterations=1, masked_execution=True)
            outer.dispatch(self.first, a, template_args={"self": self})
            container.append(outer)
        builder.append(container)


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_nested_and_repeated_private_regions_freeze_without_aliasing(tmp_path):
    model = _Model()
    builder = ti.graph.GraphBuilder()
    model.record(builder)
    graph = builder.compile()
    graph.run({})
    np.testing.assert_array_equal(model.output.to_numpy(), [7, 11])
    env = Component.from_owner(model)
    root = env.compile_graph(model.record)
    other = _Model()
    assert Component.from_owner(other).compile_graph(other.record).contract["semantic_key"] == root.contract["semantic_key"]
    assert not root.contract["explicit"]
    assert len(root.contract["internal_arguments"]) == 6
    module = ti.aot.Module()
    root.export(module, "regions")
    artifact = tmp_path / "regions.tcm"
    module.archive(str(artifact))
    with Runtime("cuda" if ti.cfg.arch == ti.cuda else "cpu") as runtime:
        component = runtime.load_module(artifact).component("regions")
        with component.instantiate({}) as instance:
            instance.initialize(zero=component.contract["resources"])
            with instance.prepare() as call:
                call.run()
            assert struct.unpack("<2i", instance.resource("output").read()) == (7, 11)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_private_region_option_is_explicit_and_inherits_scope():
    builder = ti.graph.GraphBuilder()
    with pytest.raises(TypeError, match="private_scope"):
        builder.create_sequential(private_scope="yes")
    outer = builder.create_sequential(private_scope=True)
    inner = outer.create_sequential()
    a = outer.private_ndarray("first", ti.i32, ())
    b = inner.private_ndarray("second", ti.i32, ())
    assert a.name.removesuffix("first") == b.name.removesuffix("second")
    with pytest.raises(ti.TaichiRuntimeError, match="nonempty"):
        outer.private_ndarray("", ti.i32, ())


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_sequence_append_is_a_snapshot_and_rejects_conflicts_atomically():
    model = _Model()
    builder = ti.graph.GraphBuilder()
    parent = builder.create_sequential()
    scratch = parent.private_ndarray("scratch", ti.i32, ())
    child = parent.create_sequential()
    child.dispatch(_seed7, scratch)
    parent.append(child)
    child.dispatch(_seed11, scratch)  # Must not change the appended recording.
    conflict = builder.create_sequential()
    conflict.private_ndarray("unrelated", ti.i32, ())
    conflict.private_ndarray("scratch", ti.i32, ())
    with pytest.raises(ti.TaichiRuntimeError, match="conflicting fixed"):
        parent.append(conflict)
    assert "unrelated" not in parent._fixed_runtime_args
    with pytest.raises(ti.TaichiRuntimeError, match="distinct Sequential"):
        parent.append(parent)
    parent.dispatch(model.first, scratch, template_args={"self": model})
    builder.append(parent)
    builder.compile().run({})
    assert model.output[0] == 7


@ti.data_oriented
class _LifecycleModel(_Model):
    @ti.kernel
    def initialize_kernel(self):
        self.output[0] = 7

    def initialize(self, builder):
        builder.dispatch(self.initialize_kernel)

    def adjust(self, builder):
        scratch = builder.private_ndarray("scratch", ti.i32, ())
        builder.dispatch(_seed7, scratch)
        builder.dispatch(self.first, scratch, template_args={"self": self})

    def record(self, builder):
        # Existing normal recordings need not invent per-AOT-operation names.
        region = builder.create_sequential()
        scratch = region.private_ndarray("scratch", ti.i32, ())
        counter = region.private_ndarray("counter", ti.i32, ())
        region.dispatch(_reset, counter)
        condition, body = region.create_sequential(), region.create_sequential()
        condition.dispatch(_seed7, scratch)
        body.dispatch(_seed11, scratch)
        body.dispatch(self.second, scratch, template_args={"self": self})
        body.dispatch(_advance, counter)
        region.while_loop(condition, body, predicate=scratch, counter=counter,
                          max_iterations=1, masked_execution=True)
        builder.append(region)


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_lifecycle_phases_isolate_root_and_nested_private_names(tmp_path):
    owner = _LifecycleModel()
    root = Component.from_owner(owner).compile_graph(owner.record,
        initializer=owner.initialize, operations={"repeat": owner.record, "adjust": owner.adjust})
    assert len(root.contract["internal_arguments"]) == 5
    module = ti.aot.Module()
    root.export(module, "phases")
    artifact = tmp_path / "phases.tcm"
    module.archive(str(artifact))
    with Runtime("cuda" if ti.cfg.arch == ti.cuda else "cpu") as runtime:
        component = runtime.load_module(artifact).component("phases")
        with component.instantiate({}) as instance:
            instance.initialize(zero=component.contract["resources"])
            instance.initialize_state()
            assert struct.unpack("<2i", instance.resource("output").read()) == (7, 0)
            with instance.prepare() as main, instance.prepare(operation="repeat") as repeat:
                main.run()
                repeat.run()
                assert struct.unpack("<2i", instance.resource("output").read()) == (7, 11)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_private_recording_scope_restores_after_failure():
    builder = ti.graph.GraphBuilder()
    with pytest.raises(RuntimeError, match="recording failed"):
        with builder._private_recording_scope():
            first = builder.private_ndarray("scratch", ti.i32, ())
            child = builder.create_sequential()
            second = child.private_ndarray("counter", ti.i32, ())
            assert first.name.removesuffix("scratch") == second.name.removesuffix("counter")
            raise RuntimeError("recording failed")
    assert builder.private_ndarray("scratch", ti.i32, ()).name == "scratch"
