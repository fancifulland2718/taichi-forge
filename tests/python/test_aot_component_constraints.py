"""Algorithm bounds follow actual resource use and precede allocation/submission."""

import struct
from dataclasses import dataclass

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.aot._component import Component
from taichi_forge_runtime.aot import Runtime
from tests import test_utils


@dataclass(frozen=True)
class _Counts:
    active: int


@ti.data_oriented
class _ParentBoundedModel:
    def __init__(self, capacity, active):
        self.capacity = capacity
        scope = ti.experimental.ResourceScope(self)
        record = scope.record("state", _Counts(active))
        record.require_le("parent_capacity", record.dim("active"), record.parent_dim("capacity"), when="active")
        builder = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=builder)
        scope.output = scope.field(ti.i32, shape=scope.dim("capacity"))
        self.tree = builder.finalize()

    @ti.kernel
    def step(self):
        for i in range(self.state.active):
            self.output[i] = i + 1

    def record(self, builder):
        builder.dispatch(self.step)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_record_constraints_resolve_parent_dimensions_before_and_after_export(tmp_path):
    with pytest.raises(ValueError, match="parent_capacity.*5 > 4"):
        _ParentBoundedModel(4, 5)
    model = _ParentBoundedModel(4, 3)
    env = Component.from_owner(model)
    root = env.compile_graph(model.record)
    module = ti.aot.Module()
    root.export(module, "parent_bounded")
    artifact = tmp_path / "parent_bounded.tcm"
    module.archive(str(artifact))
    active = next(name for name in root.contract["dimensions"] if env._paths.get(name) == ("state", "active"))
    with Runtime("cpu") as runtime:
        component = runtime.load_module(artifact).component("parent_bounded")
        with pytest.raises(ValueError, match="parent_capacity.*8 > 7"):
            component.instantiate({"capacity": 7, active: 8})
        with component.instantiate({"capacity": 7, active: 5}) as instance:
            instance.initialize(zero=component.contract["resources"])
            with instance.prepare() as call:
                call.run()
            assert struct.unpack("<7i", instance.resource("output").read()) == (1, 2, 3, 4, 5, 0, 0)


@ti.data_oriented
class _BoundedModel:
    def __init__(self, capacity, active, *, budget=8):
        self.capacity, self.active, self.unused, self.padding = capacity, active, 0, 0
        builder = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=builder)
        scope.output = scope.field(ti.i32, shape=scope.dim("capacity"))
        scope.value("unused", ti.i32)
        scope.require_le("active_capacity", scope.dim("active"), scope.dim("capacity"), when="output")
        scope.require_le("algorithm_budget", scope.dim("capacity"), budget, when="output")
        scope.require_le("reserved_budget", scope.dim("capacity") + scope.dim("padding"), budget, when="output")
        scope.require_le("unrelated_budget", scope.dim("capacity"), capacity, when="unused")
        self.tree = builder.finalize()

    @ti.kernel
    def step(self):
        for i in range(self.active):
            self.output[i] = i + 1

    def record(self, builder):
        builder.dispatch(self.step)


@ti.data_oriented
class _NestedModel:
    def __init__(self):
        ti.experimental.ResourceScope(self).state = _BoundedModel(4, 3)

    @ti.kernel
    def step(self):
        for i in range(self.state.active):
            self.state.output[i] = i + 1

    def record(self, builder):
        builder.dispatch(self.step)


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_constraints_guard_real_calls_and_frozen_instances(tmp_path, monkeypatch):
    first, second = _BoundedModel(4, 3), _BoundedModel(7, 5, budget=16)
    env, other = Component.from_owner(first), Component.from_owner(second)
    code = env.compile(first.step)
    assert other.compile(second.step) is code  # Admission did not widen the ABI/code key.
    assert "padding" not in code.used  # Constraint-only dimensions do not become numerical inputs.
    first.output.fill(-1)
    first.active = 5
    with pytest.raises(ValueError, match="active_capacity.*5 > 4.*declared at"):
        code.run(env, first)
    with pytest.raises(ValueError, match="active_capacity"):
        env.compile(first.step)  # Cached definitions cannot bypass current bounds.
    np.testing.assert_array_equal(first.output.to_numpy(), -1)
    first.active = 3
    code.run(env, first)
    np.testing.assert_array_equal(first.output.to_numpy(), [1, 2, 3, -1])
    root = env.compile_graph(first.record)
    assert root.contract["version"] == 5
    assert set(root.contract["constraints"]) == {"active_capacity", "algorithm_budget", "reserved_budget"}
    assert "unrelated_budget" not in root.contract["compilation"]["constraint_sources"]
    module = ti.aot.Module()
    root.export(module, "bounded")
    artifact = tmp_path / "bounded.tcm"
    module.archive(str(artifact))
    arch = "cuda" if ti.cfg.arch == ti.cuda else "cpu"
    with Runtime(arch) as runtime:
        component = runtime.load_module(artifact).component("bounded")
        dimensions = {name: 7 for name, schema in component.contract["dimensions"].items() if len(schema) == 4}
        dimensions["active"] = 5
        dimensions["padding"] = 0
        allocate = runtime.dense_snode_group
        allocations = []

        def track_allocation(*args, **kwargs):
            allocations.append(1)
            return allocate(*args, **kwargs)

        monkeypatch.setattr(runtime, "dense_snode_group", track_allocation)
        with pytest.raises(ValueError, match="algorithm_budget.*9 > 8"):
            component.instantiate({**dimensions, "capacity": 9})
        assert not allocations  # Rejected before allocating any native group.
        with component.instantiate(dimensions) as instance:
            instance.initialize(zero=component.contract["resources"])
            with pytest.raises(ValueError, match="active_capacity.*8 > 7"):
                instance.prepare(dimensions={"active": 8})
            with pytest.raises(ValueError, match="reserved_budget.*9 > 8"):
                instance.prepare(dimensions={"padding": 2})
            assert not instance._prepared
            for active in (0, 5, 7):
                with instance.prepare(dimensions={"active": active}) as call:
                    call.run()
                expected = list(range(1, active + 1)) + [0] * (7 - active)
                assert struct.unpack("<7i", instance.resource("output").read()) == tuple(expected)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_constraints_in_records_keep_qualified_dimension_sources(tmp_path):
    model = _NestedModel()
    env = Component.from_owner(model)
    root = env.compile_graph(model.record)
    assert len(root.contract["constraints"]) == 3
    module = ti.aot.Module()
    root.export(module, "nested")
    artifact = tmp_path / "nested.tcm"
    module.archive(str(artifact))
    with Runtime("cpu") as runtime:
        component = runtime.load_module(artifact).component("nested")
        dimensions = {name: 7 for name, schema in component.contract["dimensions"].items() if len(schema) == 4}
        active = next(name for name in dimensions if env._paths[name] == ("state", "active"))
        padding = next(name for name in dimensions if env._paths[name] == ("state", "padding"))
        dimensions[padding] = 0
        with pytest.raises(ValueError, match="nested.*8 > 7.*declared at"):
            component.instantiate({**dimensions, active: 8})
        with component.instantiate(dimensions) as instance:
            instance.initialize(zero=component.contract["resources"])
            with instance.prepare() as call:
                call.run()
            output = next(iter(component.contract["resources"]))
            assert struct.unpack("<7i", instance.resource(output).read()) == tuple(range(1, 8))
