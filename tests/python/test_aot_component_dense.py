import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.aot._component import Component, DenseField, Dimension, RuntimeValue
from tests import test_utils


@ti.data_oriented
class _DenseModel:
    def __init__(self, capacity, active, gravity=(2, 3, 4)):
        self.capacity = capacity
        self.active = active
        self.gravity = ti.Vector(gravity)
        self.x = ti.field(ti.f32)
        self.y = ti.field(ti.f32)
        first = ti.FieldsBuilder()
        first.dense(ti.i, capacity).place(self.x)
        first.dense(ti.i, capacity).place(self.y)
        self.first = first.finalize()
        self.alias_x = self.x
        self.v = ti.Vector.field(3, ti.f32)
        self.m = ti.Matrix.field(2, 2, ti.f32)
        second = ti.FieldsBuilder()
        second.dense(ti.i, capacity).place(self.v)
        second.dense(ti.i, capacity).place(self.m)
        self.second = second.finalize()
        self.x.fill(-11)
        self.y.fill(-17)
        self.v.fill(-23)
        self.m.fill(-29)

    @ti.func
    def value(self, i):
        return self.alias_x[i] + self.gravity[0]

    @ti.kernel
    def active_pass(self):
        for i in range(self.active):
            self.x[i] = self.value(i)
            self.y[i] = 2 * self.x[i]
            self.v[i] = ti.Vector([self.x[i], self.y[i], self.gravity[1]])
            self.m[i] = ti.Matrix([[self.x[i], 0.0], [0.0, self.gravity[2]]])

    @ti.kernel
    def capacity_pass(self):
        for i in range(self.y.shape[0]):
            self.y[i] = i
        for i in self.v:
            self.v[i] = ti.Vector([i, i + 1, i + 2])


def _environment():
    return Component(
        capacity=Dimension(),
        active=Dimension("active", "capacity"),
        gravity=RuntimeValue(ti.types.vector(3, ti.f32)),
        x=DenseField(ti.f32, ("capacity",), "state"),
        alias_x=DenseField(ti.f32, ("capacity",), "state"),
        y=DenseField(ti.f32, ("capacity",), "state"),
        v=DenseField(ti.types.vector(3, ti.f32), ("capacity",), "work"),
        m=DenseField(ti.types.matrix(2, 2, ti.f32), ("capacity",), "work"),
    )


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_original_fields_preserve_groups_aliases_and_dynamic_ranges():
    producer = _DenseModel(7, 7)
    env = _environment()
    code = env.compile(producer.active_pass)
    capacity_code = env.compile(producer.capacity_pass)
    assert len(code.contract["groups"]) == 2
    assert code.contract["members"]["x"] == code.contract["members"]["alias_x"]
    assert sum(name in ("x", "alias_x") for name in code.dynamic) == 1
    bigger = _DenseModel(41, 37, gravity=(5, 7, 9))
    empty = _DenseModel(129, 0)
    assert env.compile(bigger.active_pass) is code
    assert env.compile(empty.active_pass) is code
    assert env.compile(bigger.capacity_pass) is capacity_code
    assert env.compile(empty.capacity_pass) is capacity_code
    runtime = ti.lang.impl.get_runtime()
    runtime.set_kernel_executable_lifecycle_telemetry_enabled(True)
    runtime.debug_kernel_executable_lifecycle_stats(True)
    try:
        code.run(env, bigger)
        code.run(env, empty)
        code.run(env, producer)
        assert runtime.debug_kernel_executable_lifecycle_stats()["compiler_invocations"] == 1
    finally:
        runtime.set_kernel_executable_lifecycle_telemetry_enabled(False)
    assert (producer.x.to_numpy() == -9).all()
    x = bigger.x.to_numpy()
    np.testing.assert_array_equal(x[:37], np.full(37, -6))
    np.testing.assert_array_equal(x[37:], np.full(4, -11))
    np.testing.assert_array_equal(bigger.y.to_numpy()[37:], np.full(4, -17))
    assert (empty.x.to_numpy() == -11).all()
    np.testing.assert_array_equal(bigger.v.to_numpy()[36], [-6, -12, 7])
    np.testing.assert_array_equal(bigger.m.to_numpy()[36], [[-6, 0], [0, 9]])
    assert (bigger.m.to_numpy()[37:] == -29).all()

    # Same numerical definition and normal Field construction on ordinary JIT.
    ordinary = _DenseModel(41, 37, gravity=(5, 7, 9))
    ordinary.active_pass()
    for name in ("x", "y", "v", "m"):
        np.testing.assert_array_equal(getattr(bigger, name).to_numpy(), getattr(ordinary, name).to_numpy())
    capacity_code.run(env, empty)
    np.testing.assert_array_equal(empty.y.to_numpy(), np.arange(129))
    np.testing.assert_array_equal(empty.v.to_numpy()[-1], [128, 129, 130])

    before = bigger.x.to_numpy()
    bigger.active = 42
    with pytest.raises(ti.TaichiCompilationError, match="exceeds capacity"):
        code.run(env, bigger)
    np.testing.assert_array_equal(bigger.x.to_numpy(), before)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_group_cannot_be_split_or_infer_dimension_from_equal_values():
    model = _DenseModel(7, 7)
    env = _environment()
    code = env.compile(model.active_pass)
    replacement = _DenseModel(7, 7)
    model.y = replacement.y
    with pytest.raises(ti.TaichiCompilationError, match="one actual SNode root"):
        code.run(env, model)

    model = _DenseModel(7, 7)
    model.same_number = model.capacity
    declarations = dict(env.declarations)
    declarations["same_number"] = Dimension()
    declarations["alias_x"] = DenseField(ti.f32, ("same_number",), "state")
    with pytest.raises(ti.TaichiCompilationError, match="Conflicting dimension provenance"):
        Component(**declarations).compile(model.active_pass)

    @ti.data_oriented
    class Scalars:
        def __init__(self):
            self.a = ti.field(ti.i32)
            self.b = ti.field(ti.i32)
            builder = ti.FieldsBuilder()
            builder.place(self.a, self.b)
            self.tree = builder.finalize()

        @ti.kernel
        def step(self):
            self.a[None] += self.b[None]

    scalar = Scalars()
    with pytest.raises(ti.TaichiCompilationError, match="cannot split the same SNode root"):
        Component(a=DenseField(ti.i32, (), "a"), b=DenseField(ti.i32, (), "b")).compile(scalar.step)
