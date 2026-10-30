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
    def __init__(self, n, left=2., right=3., worker_type=_Worker):
        self.n = n
        builder = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=builder)
        scope.shared = scope.field(ti.f32, shape=scope.dim("n"))
        self.tree = builder.finalize()
        scope.left = worker_type(self, left)
        scope.right = worker_type(self, right)

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


@ti.kernel
def _step_worker(worker: ti.template()):
    for i in worker.shared:
        worker.shared[i] += worker.increment()
    worker.local[None] += worker.gain


class _TemplateParent(_Parent):
    def record(self, builder):
        builder.dispatch(self.tick)
        sequence = builder.create_sequential()
        sequence.dispatch(_step_worker, template_args={"worker": self.left})
        sequence.dispatch(_step_worker, template_args={"worker": self.right})
        builder.append(sequence)


@ti.kernel
def _ignore_worker(worker: ti.template()) -> ti.i32:
    return 7


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_nested_template_record_keeps_path_and_only_actual_dependencies():
    class Envelope:
        def __init__(self, n):
            ti.experimental.ResourceScope(self).inner = _Parent(n)

    first, second = Envelope(3), Envelope(19)
    env, other = ti.aot.Component.from_owner(first), ti.aot.Component.from_owner(second)
    code = env.compile(_step_worker, owner=first, template_args={"worker": first.inner.left})
    assert other.compile(_step_worker, owner=second, template_args={"worker": second.inner.left}) is code
    code.run(other, second)
    np.testing.assert_array_equal(second.inner.shared.to_numpy(), 2.)
    unused = env.compile(_ignore_worker, owner=first, template_args={"worker": first.inner.left})
    assert env.compile(_ignore_worker, owner=first, template_args={"worker": first.inner.right}) is unused
    assert not unused.contract["captures"]
    assert not unused.contract["groups"]


@pytest.mark.parametrize("parent_type", (_Parent, _TemplateParent))
@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_child_graph_preserves_shared_roots_and_independent_instances(tmp_path, parent_type):
    first, second = parent_type(3), parent_type(19, 5., 7.)
    env, other = ti.aot.Component.from_owner(first), ti.aot.Component.from_owner(second)
    child_code = env.compile(first.left.step, owner=first)
    assert other.compile(second.left.step, owner=second) is child_code
    template_code = env.compile(_step_worker, owner=first, template_args={"worker": first.left})
    assert other.compile(_step_worker, owner=second, template_args={"worker": second.left}) is template_code
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
    with pytest.raises(ti.TaichiCompilationError, match="immutable"):
        env.compile(_step_worker, owner=root, template_args={"worker": stranger.left})
    ti.experimental.ResourceScope(root).alias = root.left
    env = ti.aot.Component.from_owner(root)
    with pytest.raises(ti.TaichiCompilationError, match="exactly one declared.*found 2"):
        env.compile(root.left.step, owner=root)
    with pytest.raises(ti.TaichiCompilationError, match="exactly one declared resource record path"):
        env.compile(_step_worker, owner=root, template_args={"worker": root.left})


class _BorrowedWorker(_Worker):
    def __init__(self, parent, gain):
        self.gain, self.count = gain, parent.n
        builder = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=builder)
        scope.value("gain", ti.f32)
        scope.local = scope.field(ti.f32, shape=())
        self.tree = builder.finalize()
        scope.borrow("shared", parent.shared, dimensions={"n": scope.dim("count")})


class _SourceWorker(_BorrowedWorker):
    def __init__(self, parent, gain):
        super().__init__(parent, gain)
        scope = ti.experimental.ResourceScope(self)
        scope.borrow("shared", parent.shared, source_scope=True, dimensions={"n": scope.dim("count")})

    @ti.kernel
    def clear(self):
        self.local[None] = 0


class _SourceParent(_Parent):
    def __init__(self, n, padding):
        self.n, self.padding = n, padding
        builder = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=builder)
        scope.shared = scope.field(ti.f32, shape=scope.dim("n"))
        scope.unrelated = scope.field(ti.i32, shape=scope.dim("padding"))
        self.tree = builder.finalize()
        scope.left = _SourceWorker(self, 2.)
        scope.right = _SourceWorker(self, 3.)


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_source_scope_borrow_keeps_layout_only_dimensions_and_unused_links(tmp_path):
    model = _SourceParent(3, 5)
    child = ti.aot.Component.from_owner(model.left)
    local = child.compile(model.left.clear)
    assert not local.contract["dimensions"]
    with pytest.raises(ti.TaichiCompilationError, match="requires its declared allocation scope"):
        child.compile(model.left.step)
    env = ti.aot.Component.from_owner(model)
    root = env.compile_graph(model.record)
    other = _SourceParent(7, 13)
    assert ti.aot.Component.from_owner(other).compile(other.left.step, owner=other) is env.compile(model.left.step, owner=model)
    assert child.compile(model.left.clear) is local  # rejected capture did not publish a partial entry
    assert "padding" in root.contract["dimensions"]
    module = ti.aot.Module()
    root.export(module, "source_parent")
    path = tmp_path / "source_parent.tcm"
    module.archive(str(path))
    backend = "cuda" if ti.cfg.arch == ti.cuda else "cpu"
    ti.reset()
    with Runtime(backend) as runtime:
        component = runtime.load_module(path).component("source_parent")
        dims = {"n": 19, "padding": 37, _capture_id(("left", "count")): 19, _capture_id(("right", "count")): 19}
        values = {_capture_id((side, "gain")): value for side, value in (("left", 2.), ("right", 3.))}
        with component.instantiate(dims) as instance:
            instance.initialize(zero=component.contract["resources"])
            with instance.prepare(values=values) as prepared:
                prepared.run()
            np.testing.assert_array_equal(np.frombuffer(instance.resource("shared").read(), dtype=np.float32), 6)
            with pytest.raises(ValueError, match="Dimension constraint"):
                instance.prepare(values=values, dimensions={_capture_id(("left", "count")): 20})


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_child_borrow_uses_source_layout_and_checks_explicit_dimension_relation(tmp_path):
    model = _Parent(3, worker_type=_BorrowedWorker)
    env = ti.aot.Component.from_owner(model)
    root = env.compile_graph(model.record)
    assert len(root.contract["groups"]) == 3
    assert len(root.contract["resources"]) == 3
    assert len(root.contract["constraints"]) == 4
    model.left.count = 4
    with pytest.raises(ValueError, match="Dimension constraint"):
        env.compile(model.left.step, owner=model)
    model.left.count = 3
    module = ti.aot.Module()
    root.export(module, "borrowed_parent")
    path = tmp_path / "borrowed_parent.tcm"
    module.archive(str(path))
    backend = "cuda" if ti.cfg.arch == ti.cuda else "cpu"
    ti.reset()
    with Runtime(backend) as runtime:
        component = runtime.load_module(path).component("borrowed_parent")
        dims = {"n": 19, _capture_id(("left", "count")): 19, _capture_id(("right", "count")): 19}
        values = {_capture_id((side, "gain")): value for side, value in (("left", 2.), ("right", 3.))}
        with component.instantiate(dims) as instance:
            instance.initialize(zero=component.contract["resources"])
            with instance.prepare(values=values) as prepared:
                prepared.run()
            np.testing.assert_array_equal(np.frombuffer(instance.resource("shared").read(), dtype=np.float32), 6)
            with pytest.raises(ValueError, match="Dimension constraint"):
                instance.prepare(values=values, dimensions={_capture_id(("left", "count")): 20})
