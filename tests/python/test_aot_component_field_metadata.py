"""Captured Field scalar dtype survives lowering and inline helper calls."""

import numpy as np
import taichi_forge as ti
from taichi_forge.aot._component import Component
from tests import test_utils


@ti.data_oriented
class _TypedState:
    def __init__(self, count, dtype):
        self.count = count
        builder = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=builder)
        scope.scalar = scope.field(dtype, shape=scope.dim("count"))
        scope.vector = scope.vector_field(3, dtype=dtype, shape=scope.dim("count"))
        scope.matrix = scope.matrix_field(2, 2, dtype=dtype, shape=scope.dim("count"))
        self.tree = builder.finalize()

    @ti.func
    def cast_value(self, field: ti.template(), value):
        return ti.cast(value, field.dtype)

    @ti.kernel
    def populate(self):
        for i in range(self.count):
            self.scalar[i] = self.cast_value(self.scalar, i + .5)
            self.vector[i] = ti.Vector([i, i + 1, i + 2], dt=self.vector.dtype)
            value = ti.Matrix.zero(self.matrix.dtype, 2, 2)
            value[0, 0], value[1, 1] = i + 1, i + 2
            self.matrix[i] = value


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_captured_dtype_is_static_metadata_not_a_sample_value():
    first = _TypedState(3, ti.u32)
    code = Component.from_owner(first).compile(first.populate)
    for count, dtype, host_type in ((37, ti.u32, np.uint32), (7, ti.f32, np.float32)):
        state = _TypedState(count, dtype)
        environment = Component.from_owner(state)
        current = environment.compile(state.populate)
        assert (current is code) == (dtype == ti.u32)
        index = np.arange(count)
        for execute in (lambda: current.run(environment, state), state.populate):
            state.scalar.fill(91)
            state.vector.fill(93)
            state.matrix.fill(97)
            execute()
            np.testing.assert_array_equal(state.scalar.to_numpy(), (index + .5).astype(host_type))
            np.testing.assert_array_equal(state.vector.to_numpy(), np.stack([index, index + 1, index + 2], axis=1))
            expected = np.zeros((count, 2, 2), dtype=host_type)
            expected[:, 0, 0], expected[:, 1, 1] = index + 1, index + 2
            np.testing.assert_array_equal(state.matrix.to_numpy(), expected)
