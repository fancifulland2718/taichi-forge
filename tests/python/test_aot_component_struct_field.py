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
    for kwargs in (dict(shape=3), dict(shape=(), layout=ti.Layout.SOA), dict(shape=(), needs_grad=True)):
        with pytest.raises(ValueError, match="flat rank-zero plain AOS"):
            scope.struct_field(dict(value=ti.f32), **kwargs)
