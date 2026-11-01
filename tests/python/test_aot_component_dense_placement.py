"""Normal interleaved Dense placement retains symbolic extent provenance."""
from types import SimpleNamespace

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge_runtime.aot import Runtime
from tests import test_utils


@ti.data_oriented
class _Records:
    def __init__(self, n):
        self.n = n
        builder = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=builder, initialization="zero")
        cell = scope.dense(ti.i, scope.dim("n"))
        scope.index = scope.field(ti.i32, shape=scope.dim("n"), placement=cell)
        scope.vector = scope.vector_field(3, ti.f32, shape=scope.dim("n"), placement=cell)
        scope.sentinel = scope.field(ti.i32, shape=scope.dim("n"), placement=cell)
        scope.alias = self.vector
        # A following Dense block must move with the first cell's new extent.
        scope.tail = scope.field(ti.i32, shape=scope.dim("n"))
        self.tree = builder.finalize()

    @ti.kernel
    def step(self):
        for i in self.index:
            self.index[i] = i + 1
            self.alias[i] += ti.Vector([i, i + 2, -i])
            self.tail[i] = i + 13

    def record(self, builder):
        builder.dispatch(self.step)

    @ti.kernel
    def initialize_kernel(self):
        for i in self.sentinel:
            self.sentinel[i] = -31

    def initialize(self, builder):
        builder.dispatch(self.initialize_kernel)


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_declared_dense_placement_grows_without_relayout(tmp_path):
    small = _Records(1)
    environment = ti.aot.Component.from_owner(small)
    code = environment.compile_graph(small.record, initializer=small.initialize)
    grown = _Records(37)
    grown_code = ti.aot.Component.from_owner(grown).compile_graph(grown.record, initializer=grown.initialize)
    assert grown_code.contract["semantic_key"] == code.contract["semantic_key"]
    grown.step()
    reference = [grown.index.to_numpy(), grown.vector.to_numpy(), grown.tail.to_numpy()]
    module = ti.aot.Module()
    code.export(module, "step")
    artifact = tmp_path / "dense-placement.tcm"
    module.archive(str(artifact))
    backend = "cuda" if ti.cfg.arch == ti.cuda else "cpu"
    ti.reset()
    with Runtime(backend) as runtime, runtime.load_module(artifact) as module:
        component = module.component("step")
        with component.instantiate(dict(n=37)) as instance:
            # Preserve an unaccessed interleaved sibling, not just active outputs.
            instance.initialize_state()
            with instance.prepare() as call:
                call.run()
            for name, dtype, expected in zip(("index", "vector", "tail"),
                                             (np.int32, np.float32, np.int32), reference):
                actual = np.frombuffer(instance.resource(name).read(), dtype).reshape(expected.shape)
                np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(
                np.frombuffer(instance.resource("sentinel").read(), np.int32), np.full(37, -31))


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_dense_placement_checks_provenance_before_member_allocation():
    owner = SimpleNamespace(n=4, unrelated=4)
    builder = ti.FieldsBuilder()
    scope = ti.experimental.ResourceScope(owner, builder=builder)
    cell = scope.dense(ti.i, scope.dim("n"))
    with pytest.raises(ValueError, match="extent relations"):
        scope.field(ti.i32, shape=scope.dim("unrelated"), placement=cell)
    other = ti.experimental.ResourceScope(SimpleNamespace(n=4))
    with pytest.raises(ValueError, match="another declaration namespace"):
        other.field(ti.i32, shape=other.dim("n"), placement=cell)
    owner.n = 5
    with pytest.raises(ValueError, match="changed after"):
        scope.field(ti.i32, shape=scope.dim("n"), placement=cell)
    owner.n = 4
    scope.x = scope.field(ti.i32, shape=scope.dim("n"), placement=cell)
    builder.finalize()
