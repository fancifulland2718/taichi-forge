"""Explicit immutable module constants participate in actual-use cache guards."""
from typing import Final

import numpy as np
import pytest
import taichi_forge as ti

from taichi_forge.aot._component import Array, Component
from taichi_forge_runtime.aot import Runtime
from tests import test_utils


MATERIAL: Final[int] = 3
FACTORS: Final = (2, 5)
UNUSED: Final[int] = 999
UNDECLARED = 7
MUTABLE: Final = [1, 2]
NAMES: Final = ("a", "b", "a", "c")


@ti.data_oriented
class _Angles:
    def __init__(self, count):
        scope = ti.experimental.ResourceScope(self)
        self.count = count
        scope.output = scope.vector_field(2, ti.f32, shape=scope.dim("count"))

    @ti.kernel
    def convert(self):
        for i in self.output:
            self.output[i] = ti.Vector([
                ti.math.radians(15.0 * i), ti.math.degrees(0.125 * i)
            ])

    def record(self, builder):
        builder.dispatch(self.convert, template_args={"self": self})


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_standard_angle_helpers_export_with_library_constant_guards(tmp_path):
    producer = _Angles(3)
    environment = Component.from_owner(producer)
    code = environment.compile(producer.convert)
    assert {item[2] for item in code.contract["constants"]} == {"pi"}
    assert environment.compile(producer.convert) is code
    module = ti.aot.Module()
    environment.compile_graph(producer.record).export(module, "angles")
    artifact = tmp_path / "angles.tcm"
    module.archive(str(artifact))
    backend = "cuda" if ti.cfg.arch == ti.cuda else "cpu"
    ti.reset()
    with Runtime(backend) as runtime:
        component = runtime.load_module(artifact).component("angles")
        with component.instantiate(dict(count=17)) as instance:
            instance.initialize(zero=("output",))
            with instance.prepare() as call:
                call.run()
            actual = np.frombuffer(instance.resource("output").read(), np.float32).reshape(17, 2)
            expected = np.stack((np.deg2rad(np.arange(17) * 15.0),
                                 np.rad2deg(np.arange(17) * 0.125)), axis=1)
            np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=2e-7)


@ti.func
def _material_value(i):
    return MATERIAL + FACTORS[0] + i


@ti.data_oriented
class _State:
    def __init__(self, count, declared=False):
        if declared:
            scope = ti.experimental.ResourceScope(self)
            scope.output = scope.field(ti.i32, shape=count)
        else:
            self.output = ti.ndarray(ti.i32, count)

    @ti.kernel
    def step(self):
        for i in range(self.output.shape[0]):
            self.output[i] = _material_value(0) + i

    @ti.kernel
    def undeclared(self):
        self.output[0] = UNDECLARED

    @ti.kernel
    def mutable(self):
        self.output[0] = MUTABLE[0]

    @ti.kernel
    def tuple_queries(self):
        index = ti.static(NAMES.index("b", 1, 3))
        count = ti.static(NAMES.count("a"))
        nested = ti.static(((1, 2), (3, 4)).index((3, 4)))
        self.output[0], self.output[1], self.output[2] = index, count, nested

    @ti.kernel
    def mutable_call(self):
        self.output[0] = ti.static([2, 1].pop())

    @ti.kernel
    def device_tuple_query(self):
        self.output[0] = ti.static((1, 2).index(self.output[0]))

    def record(self, builder):
        builder.dispatch(self.step, template_args={"self": self})


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_static_tuple_queries_keep_constant_guards_and_reject_mutable_calls(monkeypatch):
    state = _State(3, declared=True)
    environment = Component.from_owner(state)
    code = environment.compile(state.tuple_queries)
    for run in (state.tuple_queries, lambda: code.run(environment, state)):
        run()
        np.testing.assert_array_equal(state.output.to_numpy(), [1, 2, 1])
    assert {item[2] for item in code.contract["constants"]} == {"NAMES"}
    monkeypatch.setitem(globals(), "NAMES", ("b", "a", "b", "c"))
    assert not code.matches(environment, state)
    revised = environment.compile(state.tuple_queries)
    assert revised.semantic_key != code.semantic_key
    revised.run(environment, state)
    np.testing.assert_array_equal(state.output.to_numpy(), [2, 1, 1])
    with pytest.raises(ti.TaichiCompilationError, match="Uncontrolled Python calls"):
        environment.compile(state.mutable_call)
    with pytest.raises(ti.TaichiCompilationError, match="immutable scalar values or tuples"):
        environment.compile(state.device_tuple_query)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_final_constants_guard_values_declarations_and_failure_recovery(monkeypatch):
    state = _State(5)
    environment = Component(output=Array(ti.i32, 1))
    code = environment.compile(state.step)
    code.run(environment, state)
    np.testing.assert_array_equal(state.output.to_numpy(), np.arange(5) + 5)
    assert code.dynamic == ("output",)
    assert {item[2] for item in code.contract["constants"]} == {"MATERIAL", "FACTORS"}
    monkeypatch.setitem(globals(), "UNUSED", 11)
    assert environment.compile(state.step) is code
    monkeypatch.setitem(globals(), "MATERIAL", 8)
    assert not code.matches(environment, state)
    revised = environment.compile(state.step)
    assert revised.semantic_key != code.semantic_key
    revised.run(environment, state)
    np.testing.assert_array_equal(state.output.to_numpy(), np.arange(5) + 10)
    annotations = globals()["__annotations__"]
    monkeypatch.delitem(annotations, "MATERIAL")
    with pytest.raises(ti.TaichiCompilationError, match="undeclared global/closure.*MATERIAL"):
        environment.compile(state.step)
    monkeypatch.setitem(annotations, "MATERIAL", Final[int])
    assert environment.compile(state.step) is revised
    with pytest.raises(ti.TaichiCompilationError, match="undeclared global/closure.*UNDECLARED"):
        environment.compile(state.undeclared)
    with pytest.raises(ti.TaichiCompilationError, match="immutable scalar values or tuples"):
        environment.compile(state.mutable)
    # An unrelated module Final must not authorize a same-named closure value.
    MATERIAL = 19

    @ti.kernel
    def closed(output: ti.types.ndarray(ti.i32, 1)):
        output[0] = MATERIAL

    with pytest.raises(ti.TaichiCompilationError, match="undeclared global/closure.*MATERIAL"):
        Component().compile(closed, state.output, owner=state)


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_final_constants_keep_original_definition_in_loaded_component(tmp_path):
    state = _State(7, declared=True)
    state.step()
    np.testing.assert_array_equal(state.output.to_numpy(), np.arange(7) + 5)
    environment = Component.from_owner(state)
    root = environment.compile_graph(state.record)
    assert {item[2] for item in root.contract["entries"][0]["constants"]} == {"MATERIAL", "FACTORS"}
    module = ti.aot.Module()
    root.export(module, "step")
    artifact = tmp_path / "constants.tcm"
    module.archive(str(artifact))
    backend = "cuda" if ti.cfg.arch == ti.cuda else "cpu"
    ti.reset()
    with Runtime(backend) as runtime:
        component = runtime.load_module(artifact).component("step")
        with component.instantiate({}) as instance:
            instance.initialize(zero=("output",))
            with instance.prepare() as call:
                call.initialize()
                call.run()
            np.testing.assert_array_equal(np.frombuffer(instance.resource("output").read(), np.int32),
                                          np.arange(7) + 5)
