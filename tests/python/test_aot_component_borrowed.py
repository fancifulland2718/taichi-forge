"""Foreign Field bindings preserve complete source layout and explicit relations."""

import gc
import struct
import weakref

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.aot._component import Component
from taichi_forge_runtime.aot import Runtime
from tests import test_utils


class _Source:
    def __init__(self, n, m):
        self.n, self.m = n, m
        builder = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=builder)
        scope.prefix = scope.field(ti.f32, shape=scope.dim("m"))
        scope.payload = scope.field(ti.f32, shape=scope.dim("n"))
        scope.suffix = scope.field(ti.i32, shape=scope.dim("m") + 1)
        self.tree = builder.finalize()


@ti.data_oriented
class _Consumer:
    def __init__(self, source, active):
        self.capacity, self.padding, self.active = source.n, source.m, active
        scope = ti.experimental.ResourceScope(self)
        scope.dim("active")
        mapping = {"n": scope.dim("capacity"), "m": scope.dim("padding")}
        scope.borrow("state", source.payload, dimensions=mapping)
        scope.borrow("alias", source.payload, dimensions=mapping)

    @ti.func
    def value(self, i):
        return self.alias[i] + 1

    @ti.kernel
    def step(self):
        for i in range(self.active):
            self.state[i] = i * 3
            self.state[i] = self.value(i)

    def record(self, builder):
        builder.dispatch(self.step)


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_borrow_preserves_source_siblings_without_retaining_owner(tmp_path):
    source, larger = _Source(7, 3), _Source(19, 11)
    first, second = _Consumer(source, 5), _Consumer(larger, 17)
    environment, other = Component.from_owner(first), Component.from_owner(second)
    code = environment.compile(first.step)
    assert other.compile(second.step) is code
    unused = _Source(2, 4)
    second.unused_n, second.unused_m = 2, 4
    scope = ti.experimental.ResourceScope(second)
    scope.borrow("aaa_unused_alias", second.state, dimensions={"n": scope.dim("capacity"), "m": scope.dim("padding")})
    scope.borrow(
        "unused",
        unused.payload,
        dimensions={"n": scope.dim("unused_n"), "m": scope.dim("unused_m")},
    )
    other = Component.from_owner(second)
    assert other.compile(second.step) is code
    assert "unused_m" not in code.contract["dimensions"]
    assert len(code.contract["groups"]) == 1
    assert "padding" in code.contract["dimensions"]
    assert not any(name.startswith("__borrowed_") for name, _ in code.contract["captures"])
    assert len(code.contract["captures"]) == 2  # active count and one canonical Field
    larger.prefix.fill(-8)
    larger.suffix.fill(23)
    second.state.fill(-99)
    code.run(other, second)
    np.testing.assert_array_equal(second.state.to_numpy(), list(np.arange(17) * 3 + 1) + [-99, -99])
    np.testing.assert_array_equal(larger.prefix.to_numpy(), [-8] * 11)
    np.testing.assert_array_equal(larger.suffix.to_numpy(), [23] * 12)
    second.state.fill(-99)
    second.step()  # original JIT uses exactly the same borrowed Fields
    np.testing.assert_array_equal(second.state.to_numpy(), list(np.arange(17) * 3 + 1) + [-99, -99])

    # Complete declared storage stays reachable; the numerical source owner does not.
    reference = weakref.ref(source)
    del source
    gc.collect()
    assert reference() is None
    root = environment.compile_graph(first.record)
    assert set(root.contract["resources"]) == {"payload"}
    assert len(root.contract["groups"]) == 1
    module = ti.aot.Module()
    root.export(module, "borrowed")
    path = tmp_path / "borrowed.tcm"
    module.archive(str(path))
    backend = "cuda" if ti.cfg.arch == ti.cuda else "cpu"
    ti.reset()
    with Runtime(backend) as runtime:
        component = runtime.load_module(path).component("borrowed")
        with component.instantiate({"capacity": 23, "padding": 13, "active": 21}) as instance:
            instance.initialize(zero=["payload"])
            with instance.prepare() as prepared:
                prepared.run()
            assert struct.unpack("<23f", instance.resource("payload").read()) == tuple(
                list(np.arange(21) * 3 + 1) + [0, 0]
            )


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_borrow_rejects_incomplete_and_conflicting_group_relations():
    source = _Source(7, 3)
    owner = _Consumer(source, 5)
    scope = ti.experimental.ResourceScope(owner)
    scope.borrow("state", source.payload, dimensions={"n": scope.dim("capacity")})
    with pytest.raises(ti.TaichiCompilationError, match="whole source group dimensions"):
        Component.from_owner(owner)
    scope.borrow("state", source.payload, dimensions={"n": scope.dim("capacity"), "m": scope.dim("capacity")})
    with pytest.raises(ti.TaichiCompilationError, match="conflicting dimension relations"):
        Component.from_owner(owner)
    scope.borrow("state", source.payload, dimensions={"n": scope.dim("capacity"), "m": scope.dim("padding")})
    environment = Component.from_owner(owner)
    code = environment.compile(owner.step)
    owner.padding = 4
    with pytest.raises(ti.TaichiCompilationError, match="shape differs"):
        code.run(environment, owner)
    owner.padding = 3
    owner.state = source.prefix
    with pytest.raises(ti.TaichiCompilationError, match="source anchor identity"):
        code.run(environment, owner)
    plain = ti.field(ti.f32, shape=7)
    with pytest.raises(ValueError, match="explicit declared FieldsBuilder"):
        scope.borrow("plain", plain, dimensions={})


@ti.data_oriented
class _RecordModel:
    def __init__(self, source, active):
        ti.experimental.ResourceScope(self).binding = _Consumer(source, active)

    @ti.kernel
    def step(self):
        for i in range(self.binding.active):
            self.binding.alias[i] = i + 8


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_borrowed_group_dimensions_follow_the_declared_record_namespace():
    first, second = _RecordModel(_Source(7, 3), 5), _RecordModel(_Source(19, 11), 17)
    environment, other = Component.from_owner(first), Component.from_owner(second)
    code = environment.compile(first.step)
    assert other.compile(second.step) is code
    code.run(other, second)
    np.testing.assert_array_equal(second.binding.state.to_numpy(), list(range(8, 25)) + [0, 0])
    assert ["binding", "padding"] in code.contract["access_paths"].values()
