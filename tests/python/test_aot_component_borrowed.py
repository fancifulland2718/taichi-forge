"""Foreign Field bindings preserve complete source layout and explicit relations."""

import gc
import struct
import weakref
from dataclasses import dataclass, FrozenInstanceError

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.aot._component import Component
from taichi_forge_runtime.aot import Runtime
from tests import test_utils


@ti.data_oriented
class _InputSource:
    def __init__(self, n, padding):
        self.n, self.padding = n, padding
        builder = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=builder)
        node = scope.dense(ti.i, scope.dim("n"))
        scope.payload = scope.field(ti.f32, shape=scope.dim("n"), placement=node)
        scope.neighbor = scope.field(ti.f32, shape=scope.dim("n"), placement=node)
        scope.padding_state = scope.field(ti.i32, shape=scope.dim("padding"))
        self.tree = builder.finalize()

    @ti.kernel
    def advance(self):
        for i in self.payload:
            self.payload[i] += 1

    def record(self, builder):
        builder.dispatch(self.advance)


@ti.data_oriented
class _InputReader:
    def __init__(self, source):
        self.count = source.n
        scope = ti.experimental.ResourceScope(self)
        scope.borrow("state", source.payload, dimensions={"n": scope.dim("count")}, external=True)
        scope.borrow("alias", source.payload, dimensions={"n": scope.dim("count")}, external=True)
        scope.output = scope.ndarray(ti.f32, shape=scope.dim("count"), initialization="zero")

    @ti.kernel
    def render(self):
        for i in self.state:
            self.output[i] = self.state[i] + self.alias[i]

    @ti.kernel
    def bad_write(self):
        self.alias[0] = 3

    def record(self, builder):
        builder.dispatch(self.render)

    def bad_record(self, builder):
        builder.dispatch(self.bad_write)


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_external_field_input_keeps_independent_roots_and_source_lifetime(tmp_path):
    source = _InputSource(1, 2)
    reader = _InputReader(source)
    environment = Component.from_owner(reader)
    code = environment.compile(reader.render)
    other = _InputReader(_InputSource(13, 29))
    assert Component.from_owner(other).compile(other.render) is code
    scope = ti.experimental.ResourceScope(other)
    scope.borrow("aaa_unused", other.state, dimensions={"n": scope.dim("count")}, external=True)
    assert Component.from_owner(other).compile(other.render) is code
    assert not code.contract["groups"]
    assert "padding" not in code.contract["dimensions"]
    module = ti.aot.Module()
    Component.from_owner(source).compile_graph(source.record, host_resources=(source.neighbor, source.padding_state)).export(module, "solver")
    root = environment.compile_graph(reader.record)
    assert not root.contract["groups"]
    assert len(root.contract["resources"]) == 2  # input alias + own output
    root.export(module, "renderer")
    path = tmp_path / "independent.tcm"
    module.archive(str(path))
    backend = "cuda" if ti.cfg.arch == ti.cuda else "cpu"
    ti.reset()
    with Runtime(backend) as runtime, runtime.load_module(path) as loaded:
        solver = loaded.component("solver")
        renderer = loaded.component("renderer")
        with solver.instantiate(dict(n=13, padding=29)) as state:
            state.initialize(dict(payload=np.arange(13, dtype=np.float32),
                                  neighbor=np.full(13, -7, np.float32), padding_state=np.full(29, 17, np.int32)))
            binding = state.borrow("payload")
            with pytest.raises(ValueError, match="cover"):
                renderer.instantiate(dict(count=13))
            with pytest.raises(ValueError, match="dtype/shape"):
                renderer.instantiate(dict(count=7), inputs={"state": binding})
            with pytest.raises(ValueError, match="distinct"):
                renderer.instantiate(dict(count=13), inputs={"state": binding, "alias": binding})
            with pytest.raises(ValueError, match="dtype/shape"):
                renderer.instantiate(dict(count=13), inputs={"state": state.borrow("padding_state")})
            with Runtime(backend) as foreign, foreign.load_module(path) as foreign_module:
                with pytest.raises(ValueError, match="different AOT runtime"):
                    foreign_module.component("renderer").instantiate(dict(count=13), inputs={"state": binding})
            frame = renderer.instantiate(dict(count=13), inputs={"state": binding})
            assert frame.layout == {}
            assert frame.nbytes == 13 * 4
            with pytest.raises(ValueError, match="source owner"):
                frame.initialize(zero=["state"])
            with frame.prepare() as draw:
                draw.run()
                np.testing.assert_array_equal(np.frombuffer(frame.resource("output").read(), np.float32), np.arange(13) * 2)
                with pytest.raises(RuntimeError, match="cold boundary"):
                    state.initialize(zero=["payload"])
                with state.prepare() as step:
                    step.run()
                draw.run()
                np.testing.assert_array_equal(np.frombuffer(frame.resource("output").read(), np.float32), (np.arange(13) + 1) * 2)
                with solver.instantiate(dict(n=13, padding=29)) as peer:
                    peer.initialize(zero=solver.contract["resources"])
                    with renderer.instantiate(dict(count=13), inputs={"state": peer.borrow("payload")}) as peer_frame:
                        with peer_frame.prepare() as peer_draw:
                            peer_draw.run()
                            draw.run()
                        np.testing.assert_array_equal(np.frombuffer(peer_frame.resource("output").read(), np.float32), [0] * 13)
                        np.testing.assert_array_equal(np.frombuffer(frame.resource("output").read(), np.float32), (np.arange(13) + 1) * 2)
            frame.close()
            np.testing.assert_array_equal(np.frombuffer(state.resource("neighbor").read(), np.float32), [-7] * 13)
            np.testing.assert_array_equal(np.frombuffer(state.resource("padding_state").read(), np.int32), [17] * 29)
            with state.prepare() as step:
                step.run()
            state.initialize(zero=["payload"])
            with pytest.raises(RuntimeError, match="stale"):
                renderer.instantiate(dict(count=13), inputs={"state": binding})
            with renderer.instantiate(dict(count=13), inputs={"state": state.borrow("payload")}) as frame:
                draw = frame.prepare()
                state.close()
                with pytest.raises(RuntimeError):
                    draw.run()
                draw.close()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_external_field_input_rejects_writes():
    reader = _InputReader(_InputSource(7, 3))
    with pytest.raises(ti.TaichiCompilationError, match="read-only"):
        Component.from_owner(reader).compile_graph(reader.bad_record)


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
        sources = component.input_sources()
        assert {tuple(source["path"]) for source in sources["dimensions"].values()} == {
            ("capacity",), ("padding",), ("active",)}
        for source in sources["resources"]["payload"].values():
            assert source["kind"] == "attribute"
            assert source["path"] in (["state"], ["alias"])
        with component.instantiate({"capacity": 23, "padding": 13, "active": 21}) as instance:
            instance.initialize(zero=["payload"])
            with instance.prepare() as prepared:
                prepared.run()
            assert struct.unpack("<23f", instance.resource("payload").read()) == tuple(
                list(np.arange(21) * 3 + 1) + [0, 0]
            )


@ti.data_oriented
class _PairConsumer(_Consumer):
    def __init__(self, source):
        super().__init__(source, 5)
        scope = ti.experimental.ResourceScope(self)
        scope.borrow("padding_state", source.prefix,
                     dimensions={"n": scope.dim("capacity"), "m": scope.dim("padding")})

    @ti.kernel
    def copy_prefix(self):
        self.padding_state[0] = self.state[0]

    def record(self, builder):
        super().record(builder)
        builder.dispatch(self.copy_prefix)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_borrow_sources_keep_anchor_paths_for_distinct_members(tmp_path):
    owner = _PairConsumer(_Source(7, 3))
    root = Component.from_owner(owner).compile_graph(owner.record)
    module = ti.aot.Module()
    root.export(module, "pair")
    path = tmp_path / "pair.tcm"
    module.archive(str(path))
    ti.reset()
    with Runtime("cpu") as runtime, runtime.load_module(path) as loaded:
        component = loaded.component("pair")
        sources = component.input_sources()["resources"]["prefix"].values()
        assert dict(kind="attribute", path=["padding_state"]) in sources
        # Virtual siblings affect the layout but are not additional inputs.
        # In particular, the payload anchor cannot supply the prefix data.
        assert all(source == dict(kind="attribute", path=["padding_state"]) for source in sources)
        with component.instantiate(dict(capacity=23, padding=13, active=21)) as instance:
            instance.initialize(zero=component.contract["resources"])
            with instance.prepare() as prepared:
                prepared.run()
            assert struct.unpack("<13f", instance.resource("prefix").read()) == (1,) + (0,) * 12


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


@dataclass(frozen=True, slots=True)
class _FrozenBinding:
    count: int
    payload: object


@ti.data_oriented
class _FrozenConsumer:
    def __init__(self, source):
        self.padding = source.m if source else 0
        scope = ti.experimental.ResourceScope(self)
        self.binding = _FrozenBinding(source.n, source.payload) if source else None
        record = scope.record("binding", self.binding, optional=True)
        if source:
            record.borrow(
                "payload",
                self.binding.payload,
                dimensions={"n": record.dim("count"), "m": record.parent_dim("padding")},
            )

    @ti.kernel
    def step(self):
        if ti.static(self.binding is not None):
            for i in range(self.binding.count):
                self.binding.payload[i] = i + 6


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_frozen_record_borrow_uses_parent_relations_without_mutation():
    first, second = _FrozenConsumer(_Source(7, 3)), _FrozenConsumer(_Source(19, 11))
    environment, other = Component.from_owner(first), Component.from_owner(second)
    code = environment.compile(first.step)
    assert other.compile(second.step) is code
    assert not hasattr(first.binding, "_forge_resource_declarations")
    with pytest.raises(FrozenInstanceError):
        first.binding.count = 10
    code.run(other, second)
    np.testing.assert_array_equal(second.binding.payload.to_numpy(), np.arange(19) + 6)
    assert "padding" in code.contract["dimensions"]
    absent = _FrozenConsumer(None)
    empty = Component.from_owner(absent)
    alternative = empty.compile(absent.step)
    assert alternative.semantic_key != code.semantic_key
    alternative.run(empty, absent)
    assert alternative.contract["captures"] == []
    record = ti.experimental.ResourceScope(second).record("binding", second.binding, optional=True)
    with pytest.raises(TypeError, match="cannot modify"):
        record.count = 12
    with pytest.raises(ValueError, match="existing Field"):
        record.borrow("payload", first.binding.payload, dimensions={})
