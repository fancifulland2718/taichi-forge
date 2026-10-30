"""Static resource selection preserves runtime storage, shape and values."""

from dataclasses import dataclass

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.aot._component import Component, RuntimeValue, StaticValue
from taichi_forge_runtime.aot import Runtime
from tests import test_utils


@dataclass
class _History:
    count: int


@ti.data_oriented
class _State:
    def __init__(self, count, offset=7):
        self.count, self.offset = count, offset
        self.selected = "current"
        builder = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=builder)
        scope.current = scope.field(ti.i32, shape=scope.dim("count"))
        scope.alias = scope.current
        self.tree = builder.finalize()
        history = _History(count)
        history_builder = ti.FieldsBuilder()
        old = ti.experimental.ResourceScope(history, builder=history_builder)
        old.previous = old.field(ti.i32, shape=old.dim("count"))
        scope.history = history
        self.history_tree = history_builder.finalize()
        self.getter_calls = 0

    @property
    def trap(self):
        self.getter_calls += 1
        return self.current

    @ti.func
    def update(self, i):
        ti.static(getattr(self, "alias"))[i] += self.offset

    @ti.kernel
    def step(self):
        for suffix in ti.static(("current",)):
            selected = ti.static(getattr(self, suffix))
            previous = ti.static(getattr(self.history, "previous"))
            for i in selected:
                selected[i] = previous[i] + i
                getattr(self, "update")(i)

    @ti.kernel
    def by_config(self):
        selected = ti.static(getattr(self, self.selected))
        for i in range(self.count):
            selected[i] += self.offset

    @ti.kernel
    def static_count(self):
        for i in ti.static(range(getattr(self, "count"))):
            self.current[i] = 1

    @ti.kernel
    def static_shape(self):
        for i in ti.static(range(getattr(self, "current").shape[0])):
            self.current[i] = 1

    @ti.kernel
    def static_load(self):
        self.current[0] = ti.static(getattr(self, "current")[0])

    @ti.kernel
    def dynamic_name(self):
        getattr(self, self.count)[0] = 1

    @ti.kernel
    def unsafe_getter(self):
        getattr(self, "trap")[0] = 1

    @ti.kernel
    def default_value(self):
        getattr(self, "missing", self.current)[0] = 1

    @ti.kernel
    def declared_defaults(self):
        target = ti.static(getattr(self, self.selected, None))
        previous = ti.static(getattr(self.history, "previous", None))
        for i in range(getattr(self, "count", 0)):
            target[i] = previous[i] + getattr(self, "offset", -101)

    @ti.kernel
    def unsafe_getter_default(self):
        getattr(self, "trap", self.current)[0] = 1

    @ti.kernel
    def undeclared_default(self):
        self.current[0] = getattr(self, "getter_calls", -1)

    def record(self, builder):
        builder.dispatch(self.step)


def _environment(state, **extra):
    return Component.from_owner(state, offset=RuntimeValue(ti.i32), selected=StaticValue(), **extra)


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_static_resources_use_original_capture_contract_and_loaded_binding(tmp_path):
    first, second = _State(3), _State(37, 11)
    env, other = _environment(first), _environment(second)
    code = env.compile(first.step)
    assert other.compile(second.step) is code
    assert code.contract["members"]["alias"] == code.contract["members"]["current"]
    assert sum(name in ("alias", "current") for name in code.dynamic) == 1
    second.history.previous.from_numpy(np.arange(37, dtype=np.int32) * 2)
    for execute in (lambda: code.run(other, second), second.step):
        second.current.fill(-19)
        execute()
        np.testing.assert_array_equal(second.current.to_numpy(), np.arange(37) * 3 + 11)
    configured = env.compile(first.by_config)
    first.selected = "alias"
    assert env.compile(first.by_config) is not configured  # Selection is a real static dependency.
    first.selected = "current"
    root = env.compile_graph(first.record)
    module = ti.aot.Module()
    root.export(module, "history")
    path = tmp_path / "history.tcm"
    module.archive(str(path))
    backend = "cuda" if ti.cfg.arch == ti.cuda else "cpu"
    ti.reset()
    with Runtime(backend) as runtime:
        component = runtime.load_module(path).component("history")
        contract = component.contract
        dimensions = {name: 41 for name, declaration in contract["dimensions"].items() if len(declaration) == 4}
        previous_name = next(
            name for entry in contract["compilation"]["entries"]
            for name, path in entry["capture_paths"].items() if path == ["history", "previous"])
        with component.instantiate(dimensions) as instance:
            instance.initialize(zero=component.contract["resources"])
            instance.resource(previous_name).write((np.arange(41, dtype=np.int32) * 4).tobytes())
            with instance.prepare(values=dict(offset=-3)) as call:
                call.run()
            np.testing.assert_array_equal(
                np.frombuffer(instance.resource("current").read(), dtype=np.int32), np.arange(41) * 5 - 3)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_static_resource_selection_does_not_constant_fold_data_or_invoke_getters():
    owner = _State(3)
    env = _environment(owner, trap=RuntimeValue(ti.i32))
    for entry, message in (
        (owner.static_count, "runtime capture self.count"),
        (owner.static_shape, "compile-time|integer|interpreted"),
        (owner.static_load, "compile-time"),
        (owner.dynamic_name, "compile-time string"),
        (owner.unsafe_getter, "descriptor|property"),
        (owner.default_value, "undeclared capture self.missing"),
        (owner.unsafe_getter_default, "descriptor|property"),
        (owner.undeclared_default, "undeclared capture self.getter_calls"),
    ):
        with pytest.raises(ti.TaichiCompilationError, match=message):
            env.compile(entry)
    assert owner.getter_calls == 0
    code = env.compile(owner.step)
    code.run(env, owner)
    np.testing.assert_array_equal(owner.current.to_numpy(), [7, 8, 9])


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_getattr_default_preserves_declared_capture_and_static_guards():
    first, second = _State(3), _State(17, 29)
    env, other = _environment(first), _environment(second)
    code = env.compile(first.declared_defaults)
    assert other.compile(second.declared_defaults) is code
    second.history.previous.from_numpy(np.arange(17, dtype=np.int32) * 2)
    for execute in (lambda: code.run(other, second), second.declared_defaults):
        second.current.fill(-19)
        execute()
        np.testing.assert_array_equal(second.current.to_numpy(), np.arange(17) * 2 + 29)
    second.selected = "alias"
    assert other.compile(second.declared_defaults) is not code
