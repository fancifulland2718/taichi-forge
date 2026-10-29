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

    def record(self, builder):
        builder.dispatch(self.step, template_args={"self": self})


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
