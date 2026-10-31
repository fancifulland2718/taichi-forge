"""Vector/Matrix constructors copy runtime tensors without losing dtype or shape."""
import numpy as np
import taichi_forge as ti
from taichi_forge.aot._component import Component
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_tensor_constructor_copies_values_and_honors_dtype():
    vector = ti.Vector.field(3, ti.f32, shape=())
    matrix = ti.Matrix.field(2, 2, ti.f32, shape=())

    @ti.kernel
    def copy_values(value: ti.types.vector(3, ti.f32), coefficients: ti.types.matrix(2, 2, ti.f32)) -> ti.f32:
        a = ti.Vector(value)
        b = ti.Matrix(coefficients, dt=ti.i32)
        a[0] = 91
        b[0, 0] = 97
        vector[None] = a
        matrix[None] = b
        return value[0] + coefficients[0, 0]

    assert copy_values(ti.Vector([1.5, 2.5, 3.5]), ti.Matrix([[4.5, 5.5], [6.5, 7.5]])) == 6
    np.testing.assert_array_equal(vector.to_numpy(), [91, 2.5, 3.5])
    np.testing.assert_array_equal(matrix.to_numpy(), [[97, 5], [6, 7]])


@ti.data_oriented
class _Bounds:
    def __init__(self, lower):
        self.lower = lower
        scope = ti.experimental.ResourceScope(self)
        scope.value("lower", ti.types.vector(3, ti.f32))
        scope.out = scope.vector_field(3, ti.f32, shape=())

    @ti.func
    def bounds(self):
        return ti.Vector(self.lower)

    @ti.kernel
    def step(self):
        self.out[None] = self.bounds() + 1


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_original_tuple_constructor_accepts_dynamic_component_vector():
    first = _Bounds((1.0, 2.0, 3.0))
    code = Component.from_owner(first).compile(first.step)
    other = _Bounds((-4.0, 5.0, -6.0))
    environment = Component.from_owner(other)
    assert environment.compile(other.step) is code
    code.run(environment, other)
    np.testing.assert_array_equal(other.out.to_numpy(), [-3, 6, -5])
    other.out.fill(0)
    other.step()
    np.testing.assert_array_equal(other.out.to_numpy(), [-3, 6, -5])
