"""Fixed normal records keep their shared layout in component instances."""

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.aot._component import Component
from taichi_forge_runtime.aot import Runtime
from tests import test_utils


@ti.data_oriented
class _State:
    def __init__(self):
        builder = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=builder)
        scope.state = scope.struct_field(
            dict(before=ti.i32, vector=ti.types.vector(3, ti.f32), matrix=ti.types.matrix(2, 2, ti.f32), after=ti.i32),
            shape=(),
        )
        scope.alias = self.state.vector
        self.tree = builder.finalize()

    @ti.kernel
    def step(self):
        self.alias[None] += ti.Vector([1, 2, 3])
        self.state.matrix[None][1, 1] += self.state.vector[None][2]

    def record(self, builder):
        builder.dispatch(self.step)


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_normal_fixed_record_deploys_shared_layout_and_independent_instances(tmp_path):
    first, second = _State(), _State()
    environment = Component.from_owner(first)
    code = environment.compile(first.step)
    assert Component.from_owner(second).compile(second.step) is code
    assert len(code.contract["groups"]) == 1
    assert len(code.dynamic) == 2  # alias and original vector are one capture
    second.state.before.fill(31)
    second.state.after.fill(37)
    second.state.vector.fill(10)
    second.state.matrix.fill(20)
    code.run(Component.from_owner(second), second)
    np.testing.assert_array_equal(second.state.vector.to_numpy(), [11, 12, 13])
    np.testing.assert_array_equal(second.state.matrix.to_numpy(), [[20, 20], [20, 33]])
    assert second.state.before[None] == 31 and second.state.after[None] == 37
    second.step()  # identical normal JIT definition and record placement
    np.testing.assert_array_equal(second.state.vector.to_numpy(), [12, 14, 16])
    np.testing.assert_array_equal(second.state.matrix.to_numpy(), [[20, 20], [20, 49]])
    root = environment.compile_graph(first.record)
    assert len(root.contract["resources"]) == 2
    module = ti.aot.Module()
    root.export(module, "record")
    archive = tmp_path / "record.tcm"
    module.archive(str(archive))
    backend = "cuda" if ti.cfg.arch == ti.cuda else "cpu"
    ti.reset()
    with Runtime(backend) as runtime:
        component = runtime.load_module(archive).component("record")
        names = component.contract["resources"]
        vector = next(name for name, resource in names.items() if resource["element_shape"] == [3])
        matrix = next(name for name, resource in names.items() if resource["element_shape"] == [2, 2])
        with component.instantiate({}) as a, component.instantiate({}) as b:
            a.initialize(zero=names)
            b.initialize({vector: np.array([100, 200, 300], dtype=np.float32).tobytes()}, zero=(matrix,))
            with a.prepare() as run_a, b.prepare() as run_b:
                run_a.run()
                run_b.run()
                run_a.run()
                runtime.wait()
                np.testing.assert_array_equal(np.frombuffer(a.resource(vector).read(), dtype=np.float32), [2, 4, 6])
                np.testing.assert_array_equal(np.frombuffer(b.resource(vector).read(), dtype=np.float32), [101, 202, 303])
                np.testing.assert_array_equal(np.frombuffer(a.resource(matrix).read(), dtype=np.float32), [0, 0, 0, 9])
                np.testing.assert_array_equal(np.frombuffer(b.resource(matrix).read(), dtype=np.float32), [0, 0, 0, 303])


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_record_declaration_rejects_unqualified_layout_before_allocation():
    scope = ti.experimental.ResourceScope(type("Owner", (), {})())
    for kwargs in (dict(), dict(shape=(), layout=ti.Layout.SOA), dict(shape=(), needs_grad=True)):
        with pytest.raises(ValueError, match="flat plain AOS"):
            scope.struct_field(dict(value=ti.f32), **kwargs)


@ti.data_oriented
class _RecordArray:
    def __init__(self, capacity, active):
        self.capacity, self.active = capacity, active
        builder = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=builder)
        capacity_dim = scope.dim("capacity")
        scope.dim("active")
        scope.state = scope.struct_field(
            dict(before=ti.i32, vector=ti.types.vector(3, ti.f32), matrix=ti.types.matrix(2, 2, ti.f32), after=ti.i32),
            shape=capacity_dim,
        )
        scope.alias = self.state.vector
        self.tree = builder.finalize()

    @ti.func
    def advance(self, i):
        self.alias[i] += ti.Vector([1, 2, 3])
        self.state.matrix[i][1, 1] += self.state.vector[i][2]

    @ti.kernel
    def step(self):
        for i in range(self.active):
            self.advance(i)

    def record(self, builder):
        builder.dispatch(self.step)


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_interleaved_record_component_preserves_stride_and_instance_state(tmp_path):
    producer = _RecordArray(7, 7)
    environment = Component.from_owner(producer)
    code = environment.compile(producer.step)
    larger = _RecordArray(37, 31)
    assert Component.from_owner(larger).compile(larger.step) is code
    larger.state.before.fill(71)
    larger.state.after.fill(73)
    larger.state.vector.fill(10)
    larger.state.matrix.fill(20)
    # Existing JIT affine ABI and the Runtime borrower must agree on stride.
    code.run(Component.from_owner(larger), larger)
    expected = np.full((37, 3), 10, dtype=np.float32)
    expected[:31] += [1, 2, 3]
    np.testing.assert_array_equal(larger.state.vector.to_numpy(), expected)
    with ti.aot.borrow_runtime() as borrowed:
        view = borrowed.import_dense_view(ti.experimental.ndarray_view(larger.state.vector))
        np.testing.assert_array_equal(np.frombuffer(view.read(), np.float32).reshape(37, 3), expected)
        view.write(np.full((37, 3), 5, dtype=np.float32))
        view.close()
    larger.step()
    expected.fill(5)
    expected[:31] += [1, 2, 3]
    np.testing.assert_array_equal(larger.state.vector.to_numpy(), expected)
    assert (larger.state.before.to_numpy() == 71).all()
    assert (larger.state.after.to_numpy() == 73).all()
    root = environment.compile_graph(producer.record)
    assert len(root.contract["groups"]) == 1
    assert len(root.contract["resources"]) == 2
    module = ti.aot.Module()
    root.export(module, "records")
    archive = tmp_path / "records.tcm"
    module.archive(str(archive))
    backend = "cuda" if ti.cfg.arch == ti.cuda else "cpu"
    ti.reset()
    with Runtime(backend) as runtime:
        loaded = runtime.load_module(archive)
        component = loaded.component("records")
        domain = runtime.execution_domain() if backend == "cuda" else None
        names = component.contract["resources"]
        vector = next(name for name, resource in names.items() if resource["element_shape"] == [3])
        matrix = next(name for name, resource in names.items() if resource["element_shape"] == [2, 2])
        with component.instantiate(dict(capacity=37, active=31)) as a, component.instantiate(dict(capacity=65, active=63)) as b:
            for instance, size in ((a, 37), (b, 65)):
                instance.initialize({
                    vector: np.full((size, 3), 10, dtype=np.float32).tobytes(),
                    matrix: np.full((size, 2, 2), 20, dtype=np.float32).tobytes(),
                })
            with a.prepare(execution_domain=domain) as run_a, b.prepare(execution_domain=domain) as run_b:
                if domain is None:
                    run_a.run()
                    run_b.run()
                    run_a.run()
                    runtime.wait()
                else:
                    run_a.initialize()
                    run_b.initialize()
                    assert run_a.info["execution_kind"] == run_b.info["execution_kind"] == "cuda_graph"
                    run_a.submit()
                    run_b.submit()
                    run_a.submit().wait()
                for instance, size, active, repeats in ((a, 37, 31, 2), (b, 65, 63, 1)):
                    expect_v = np.full((size, 3), 10, dtype=np.float32)
                    expect_v[:active] += repeats * np.array([1, 2, 3])
                    expect_m = np.full((size, 2, 2), 20, dtype=np.float32)
                    expect_m[:active, 1, 1] += 13 if repeats == 1 else 29
                    np.testing.assert_array_equal(np.frombuffer(instance.resource(vector).read(), np.float32).reshape(size, 3), expect_v)
                    np.testing.assert_array_equal(np.frombuffer(instance.resource(matrix).read(), np.float32).reshape(size, 2, 2), expect_m)
            a.close()
            with b.prepare(dimensions=dict(active=0), execution_domain=domain) as empty:
                if domain is None:
                    empty.run()
                else:
                    assert not domain.info["closed"]
                    empty.initialize()
                    empty.submit().wait()
            np.testing.assert_array_equal(np.frombuffer(b.resource(vector).read(), np.float32).reshape(65, 3), expect_v)
