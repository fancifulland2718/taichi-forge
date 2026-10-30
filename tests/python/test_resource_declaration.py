import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.aot._component import Component
from tests import test_utils


@ti.data_oriented
class _Declared:
    def __init__(self, count):
        self.count = count
        builder = ti.FieldsBuilder()
        state = ti.experimental.ResourceScope(self, builder=builder)
        state.x = state.field(ti.f32, shape=state.dim("count").maximum(1))
        state.y = state.field(ti.f32, shape=state.dim("count") + 1)
        state.alias = state.x
        self.tree = builder.finalize()

    @ti.kernel
    def fill(self):
        for i in range(self.count):
            self.alias[i] = i + 2
        for i in self.y:
            self.y[i] = i * 3


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_normal_declarations_preserve_relations_and_actual_root():
    from taichi_forge.aot._component import Dimension
    from taichi_forge.lang._resource_declaration import Extent

    expression = ["max", ["dim", "count"], ["const", 1]]
    extent, dimension = Extent(expression), Dimension(expression=expression)
    expression[2][1] = 9
    assert extent.expression == dimension.expression == ("max", ("dim", "count"), ("const", 1))
    small = _Declared(3)
    env = Component.from_owner(small)
    code = env.compile(small.fill)
    large = _Declared(17)
    large.unrelated_count = 11
    independent = ti.FieldsBuilder()
    unrelated = ti.experimental.ResourceScope(large, builder=independent)
    unrelated.unused = unrelated.field(ti.i32, shape=unrelated.dim("unrelated_count"))
    large.independent_tree = independent.finalize()
    large_env = Component.from_owner(large)
    assert large_env.compile(large.fill) is code
    code.run(large_env, large)
    np.testing.assert_array_equal(large.x.to_numpy(), np.arange(17) + 2)
    np.testing.assert_array_equal(large.y.to_numpy(), np.arange(18) * 3)
    assert len(code.contract["groups"]) == 1
    assert "count" in code.contract["dimensions"]
    assert "unrelated_count" not in code.contract["dimensions"]
    assert len(code.contract["captures"]) == 3
    assert code.contract["declaration_sources"]["alias"][0].endswith("test_resource_declaration.py")
    empty = _Declared(0)
    empty_env = Component.from_owner(empty)
    assert empty_env.compile(empty.fill) is code
    code.run(empty_env, empty)
    assert empty.x.shape == (1,)
    ordinary = _Declared(17)
    ordinary.fill()
    np.testing.assert_array_equal(large.y.to_numpy(), ordinary.y.to_numpy())
    large.x = ti.field(ti.f32, shape=17)
    with pytest.raises(ti.TaichiCompilationError, match="replaced without redeclaration"):
        Component.from_owner(large)


@ti.data_oriented
class _Derived:
    def __init__(self, count):
        self.count = count
        self.facts = max(1, count * 2)
        self.rows = self.facts * 6
        builder = ti.FieldsBuilder()
        state = ti.experimental.ResourceScope(self, builder=builder)
        facts = state.derived_dim("facts", (state.dim("count") * 2).maximum(1))
        rows = state.derived_dim("rows", facts * 6)
        state.x = state.field(ti.i32, shape=facts)
        state.y = state.field(ti.i32, shape=rows)
        self.tree = builder.finalize()

    @ti.kernel
    def fill(self):
        for i in range(self.facts):
            self.x[i] = i + self.count
        for i in range(self.rows):
            self.y[i] = self.x[i // 6] + i

    def record(self, builder):
        builder.dispatch(self.fill)


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_named_derived_dimensions_share_loop_and_instance_relations(tmp_path):
    from taichi_forge_runtime.aot import Runtime

    first = _Derived(3)
    env = Component.from_owner(first)
    code = env.compile(first.fill)
    second = _Derived(17)
    other = Component.from_owner(second)
    assert other.compile(second.fill) is code
    code.run(other, second)
    expected = np.arange(second.rows) // 6 + 17 + np.arange(second.rows)
    np.testing.assert_array_equal(second.y.to_numpy(), expected)
    second.rows += 1
    with pytest.raises(ti.TaichiCompilationError, match="Derived dimension self.rows"):
        other.compile(second.fill)
    with pytest.raises(ti.TaichiCompilationError, match="Derived dimension self.rows"):
        code.run(other, second)
    second.rows -= 1
    assert other.compile(second.fill) is code

    root = env.compile_graph(first.record)
    assert {name for name, schema in root.contract["dimensions"].items() if len(schema) == 4} == {"count"}
    module = ti.aot.Module()
    root.export(module, "derived")
    path = tmp_path / "derived.tcm"
    module.archive(str(path))
    backend = "cuda" if ti.cfg.arch == ti.cuda else "cpu"
    ti.reset()
    with Runtime(backend) as runtime:
        component = runtime.load_module(path).component("derived")
        with pytest.raises(ValueError, match="dimension inputs differ"):
            component.instantiate(dict(count=9, rows=1))
        for count in (0, 37):
            with component.instantiate(dict(count=count)) as instance:
                instance.initialize(zero=component.contract["resources"])
                with instance.prepare() as prepared:
                    prepared.run()
                facts = max(1, count * 2)
                np.testing.assert_array_equal(
                    np.frombuffer(instance.resource("y").read(), dtype=np.int32),
                    np.arange(facts * 6) // 6 + count + np.arange(facts * 6),
                )


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_derived_declaration_failure_does_not_publish_relations():
    from types import SimpleNamespace
    from taichi_forge.lang._resource_declaration import Extent

    owner = SimpleNamespace(count=3, facts=7, rows=42)
    scope = ti.experimental.ResourceScope(owner)
    count = scope.dim("count")
    with pytest.raises(ValueError, match="does not match"):
        scope.derived_dim("facts", count * 2)
    assert "facts" not in owner._forge_resource_declarations["dimensions"]
    owner.facts = 6
    facts = scope.derived_dim("facts", count * 2)
    assert scope.derived_dim("facts", count * 2) == facts
    with pytest.raises(ValueError, match="another expression"):
        scope.derived_dim("facts", count + 3)
    with pytest.raises(ValueError, match="Cyclic"):
        scope.derived_dim("count", facts // 2)
    with pytest.raises(ValueError, match="undeclared"):
        scope.derived_dim("rows", Extent(("dim", "missing")))
    owner.rows = 36
    scope.derived_dim("rows", facts * 6)
    env = Component.from_owner(owner)
    assert env.declarations["rows"].expression == ("mul", ("dim", "facts"), ("const", 6))


@ti.data_oriented
class _NestedDerived:
    def __init__(self, count):
        scope = ti.experimental.ResourceScope(self)
        scope.buffers = _Derived(count)

    @ti.kernel
    def fill(self):
        for i in range(self.buffers.rows):
            self.buffers.y[i] = i + self.buffers.facts


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_derived_record_paths_keep_their_original_source_guard():
    model = _NestedDerived(2)
    env = Component.from_owner(model)
    code = env.compile(model.fill)
    other = _NestedDerived(7)
    other_env = Component.from_owner(other)
    assert other_env.compile(other.fill) is code
    code.run(other_env, other)
    np.testing.assert_array_equal(other.buffers.y.to_numpy(), np.arange(84) + 14)
    other.buffers.facts = 15
    with pytest.raises(ti.TaichiCompilationError, match="self.buffers.facts.*declared at"):
        code.run(other_env, other)


@ti.data_oriented
class _DefinitionConstants:
    OFFSET = 3
    UNUSED = 17

    def __init__(self, count, mode=True):
        self.count, self.mode = count, mode
        builder = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=builder)
        scope.constant("OFFSET")
        scope.constant("OFFSET")  # Repeated identical declarations remain legal.
        scope.constant("UNUSED")
        scope.static("mode", choices=(False, True))
        scope.output = scope.field(ti.i32, shape=scope.dim("count"))
        self.tree = builder.finalize()

    @ti.kernel
    def evaluate(self):
        for i in range(self.count):
            self.output[i] = self.OFFSET
            if ti.static(self.mode):
                self.output[i] += 2

    def record(self, builder):
        builder.dispatch(self.evaluate)

    def mutate_definition(self, builder):
        self.record(builder)
        self.UNUSED += 1


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_definition_constants_specialize_code_without_consumer_configuration(tmp_path):
    from taichi_forge_runtime.aot import Runtime

    first, other = _DefinitionConstants(3), _DefinitionConstants(7)
    other.UNUSED = 19
    env, other_env = Component.from_owner(first), Component.from_owner(other)
    root = env.compile_graph(first.record)
    reused = other_env.compile_graph(other.record)
    assert root.contract["semantic_key"] == reused.contract["semantic_key"]
    assert set(root.contract["configuration"]) == {"mode"}
    assert dict(root.contract["entries"][0]["static"])["OFFSET"] == ["int", 3]
    assert "UNUSED" not in dict(root.contract["entries"][0]["static"])
    original = env.compile(first.evaluate)
    other.OFFSET = 5
    changed = other_env.compile(other.evaluate)
    assert changed is not original
    changed.run(other_env, other)
    np.testing.assert_array_equal(other.output.to_numpy(), 7)
    with pytest.raises(ValueError, match="different value declaration"):
        ti.experimental.ResourceScope(first).static("OFFSET", choices=(3,))
    with pytest.raises(ti.TaichiCompilationError, match="definition constants changed"):
        env.compile_graph(first.mutate_definition)

    module = ti.aot.Module()
    root.export(module, "step")
    artifact = tmp_path / "constant.tcm"
    module.archive(str(artifact))
    arch = "cuda" if ti.cfg.arch == ti.cuda else "cpu"
    ti.reset()
    with Runtime(arch) as runtime, runtime.load_module(artifact) as loaded:
        component = loaded.component("step")
        with pytest.raises(ValueError, match="configuration inputs differ"):
            component.instantiate(dict(count=11), configuration=dict(mode=True, OFFSET=3))
        with component.instantiate(dict(count=11), configuration=dict(mode=True)) as instance:
            instance.initialize(zero=component.contract["resources"])
            with instance.prepare() as call:
                call.run()
            np.testing.assert_array_equal(np.frombuffer(instance.resource("output").read(), np.int32), 5)
