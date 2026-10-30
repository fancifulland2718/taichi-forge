"""Original child kernels share declared roots through instance call bindings."""

import gc
import weakref

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge_runtime.aot import Runtime
from taichi_forge.aot._component import _capture_id
from tests import test_utils


@ti.data_oriented
class _Worker:
    def __init__(self, parent, gain):
        self.gain = gain
        builder = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=builder)
        scope.value("gain", ti.f32)
        scope.local = scope.field(ti.f32, shape=())
        self.tree = builder.finalize()
        scope.shared = parent.shared

    @ti.func
    def increment(self):
        return self.gain + self.local[None]

    @ti.kernel
    def step(self):
        for i in self.shared:
            self.shared[i] += self.increment()
        self.local[None] += self.gain


@ti.data_oriented
class _Parent:
    def __init__(self, n, left=2., right=3.):
        self.n = n
        builder = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=builder)
        scope.shared = scope.field(ti.f32, shape=scope.dim("n"))
        self.tree = builder.finalize()
        scope.left = _Worker(self, left)
        scope.right = _Worker(self, right)

    @ti.kernel
    def tick(self):
        for i in self.shared:
            self.shared[i] += 1

    def record(self, builder):
        builder.dispatch(self.tick)
        sequence = builder.create_sequential()
        sequence.dispatch(self.left.step)
        sequence.dispatch(self.right.step)
        builder.append(sequence)


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_child_graph_preserves_shared_roots_and_independent_instances(tmp_path):
    first, second = _Parent(3), _Parent(19, 5., 7.)
    env, other = ti.aot.Component.from_owner(first), ti.aot.Component.from_owner(second)
    child_code = env.compile(first.left.step, owner=first)
    assert other.compile(second.left.step, owner=second) is child_code
    child_code.run(other, second)
    np.testing.assert_array_equal(second.shared.to_numpy(), 5)
    np.testing.assert_array_equal(second.left.local.to_numpy(), 5)
    np.testing.assert_array_equal(second.right.local.to_numpy(), 0)
    root = env.compile_graph(first.record)
    assert other.compile_graph(second.record).contract["semantic_key"] == root.contract["semantic_key"]
    contract = root.contract
    assert len(contract["groups"]) == 3
    assert len(contract["resources"]) == 3
    assert {"shared", _capture_id(("left", "shared")), _capture_id(("right", "shared"))} <= set(
        contract["resources"]["shared"]["aliases"]
    )
    local_names = {side: next(name for name, spec in contract["resources"].items()
                             if _capture_id((side, "local")) in spec["aliases"])
                   for side in ("left", "right")}
    assert local_names["left"] != local_names["right"]
    module = ti.aot.Module()
    root.export(module, "parent")
    archive = tmp_path / "parent.tcm"
    module.archive(str(archive))
    refs = [weakref.ref(value) for value in (first, first.left, first.right)]
    del first, env
    gc.collect()
    assert all(ref() is None for ref in refs)
    backend = "cuda" if ti.cfg.arch == ti.cuda else "cpu"
    ti.reset()
    with Runtime(backend) as runtime:
        loaded = runtime.load_module(archive)
        component = loaded.component("parent")
        with component.instantiate({"n": 7}) as a, component.instantiate({"n": 37}) as b:
            for instance in (a, b):
                instance.initialize(zero=contract["resources"])
            values_a = {_capture_id((side, "gain")): value for side, value in (("left", 2.), ("right", 3.))}
            values_b = {_capture_id((side, "gain")): value for side, value in (("left", 5.), ("right", 7.))}
            with a.prepare(values=values_a) as pa, b.prepare(values=values_b) as pb:
                pa.run()
                pb.run()
                pa.run()
                np.testing.assert_array_equal(np.frombuffer(a.resource("shared").read(), dtype=np.float32), 17)
                np.testing.assert_array_equal(np.frombuffer(b.resource("shared").read(), dtype=np.float32), 13)
                for side, expected in (("left", 4.), ("right", 6.)):
                    np.testing.assert_array_equal(np.frombuffer(a.resource(local_names[side]).read(), dtype=np.float32), expected)
            a.close()
            with b.prepare(values=values_b) as pb:
                pb.run()
            np.testing.assert_array_equal(np.frombuffer(b.resource("shared").read(), dtype=np.float32), 38)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_child_entry_rejects_undeclared_and_ambiguous_owners():
    root, stranger = _Parent(3), _Parent(3)
    env = ti.aot.Component.from_owner(root)
    with pytest.raises(ti.TaichiCompilationError, match="exactly one declared.*found 0"):
        env.compile(stranger.left.step, owner=root)
    ti.experimental.ResourceScope(root).alias = root.left
    env = ti.aot.Component.from_owner(root)
    with pytest.raises(ti.TaichiCompilationError, match="exactly one declared.*found 2"):
        env.compile(root.left.step, owner=root)
