"""Captured Field scalar dtype survives lowering and inline helper calls."""

import numpy as np
import pytest
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

    @ti.func
    def load_reference(self, field, index):
        return field[index]

    @ti.func
    def store_reference(self, field, index, value):
        field[index] = value

    @ti.func
    def update_reference(self, field, index):
        value = self.load_reference(field, index)
        self.store_reference(field, index, value + 1)

    @ti.kernel
    def update_references(self):
        for i in range(self.count):
            self.update_reference(self.scalar, i)
            self.update_reference(self.vector, i)
            self.update_reference(self.matrix, i)


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


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_captured_fields_remain_references_in_unannotated_inline_parameters():
    first = _TypedState(3, ti.f32)
    code = Component.from_owner(first).compile(first.update_references)
    state = _TypedState(37, ti.f32)
    environment = Component.from_owner(state)
    assert environment.compile(state.update_references) is code
    for execute in (lambda: code.run(environment, state), state.update_references):
        for field in (state.scalar, state.vector, state.matrix):
            field.fill(41)
        execute()
        for field in (state.scalar, state.vector, state.matrix):
            np.testing.assert_array_equal(field.to_numpy(), 42)

    # Explicit ndarray arguments use the same borrowed-reference semantics;
    # this must not rely on a component-specific tag or a template annotation.
    @ti.kernel
    def ordinary_array(array: ti.types.ndarray(ti.f32, 1)):
        for i in range(array.shape[0]):
            first.update_reference(array, i)

    array = ti.ndarray(ti.f32, 5)
    array.fill(7)
    ordinary_array(array)
    np.testing.assert_array_equal(array.to_numpy(), 8)


@ti.kernel
def _pack_fields(fields: ti.template(), offsets: ti.template(), output: ti.types.ndarray(ndim=2)):
    for row in range(output.shape[0]):
        for column in ti.static(range(len(fields))):
            if ti.static(hasattr(fields[column], 'n')):
                for axis in ti.static(range(getattr(fields[column], 'n'))):
                    output[row, offsets[column] + axis] = fields[column][row][axis]
            else:
                output[row, offsets[column]] = fields[column][row]


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_tuple_fields_preserve_qualified_element_introspection():
    for dtype in (ti.i32, ti.f32):
        state = _TypedState(7, dtype)
        state.populate()
        array = ti.ndarray(dtype, (7, 4))
        env = Component.from_owner(state)
        code = env.compile(_pack_fields, array, owner=state,
                           template_args=dict(fields=(state.scalar, state.vector), offsets=(0, 1)))
        expected = np.column_stack((state.scalar.to_numpy(), state.vector.to_numpy()))
        for execute in (lambda: code.run(env, state, array),
                        lambda: _pack_fields((state.scalar, state.vector), (0, 1), array)):
            execute()
            np.testing.assert_array_equal(array.to_numpy(), expected)

    @ti.kernel
    def unqualified(owner: ti.template()):
        if ti.static(hasattr(owner, 'scalar')):
            owner.scalar[0] = 1

    with pytest.raises(ti.TaichiCompilationError, match="hasattr requires a captured Field"):
        env.compile(unqualified, owner=state, template_args=dict(owner=state))
