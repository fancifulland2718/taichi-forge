"""Struct offsets apply to the final placement, including nested AD storage."""

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("layout", [ti.Layout.AOS, ti.Layout.SOA])
@pytest.mark.parametrize("shape,offset", [((3,), (-2,)), ((2, 3), (4, -1))])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, cpu_max_num_threads=4)
def test_nested_struct_offset_primal_grad_and_dual(shape, offset, layout):
    record = ti.types.struct(
        x=ti.f32, inner=ti.types.struct(v=ti.types.vector(2, ti.f32)),
        m=ti.types.matrix(2, 2, ti.f32))
    with pytest.raises(ti.TaichiSyntaxError, match="dimensionality"):
        record.field(shape=(3,), offset=(1, 2))
    with pytest.raises(ti.TaichiSyntaxError, match="shape cannot"):
        record.field(offset=1)
    state = record.field(shape=shape, offset=offset, layout=layout, needs_grad=True, needs_dual=True)

    @ti.kernel
    def initialize():
        for index in ti.grouped(state):
            linear = 0
            for axis in ti.static(range(len(shape))):
                linear = linear * shape[axis] + index[axis] - offset[axis]
            state[index].x = linear + 1
            state[index].inner.v = ti.Vector([linear + 2, linear + 3])
            state[index].m = ti.Matrix([[linear + 4, linear + 5], [linear + 6, linear + 7]])
            state.x.grad[index] = linear + 8
            state.inner.v.dual[index] = ti.Vector([linear + 9, linear + 10])

    initialize()
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    np.testing.assert_array_equal(state.x.to_numpy(), values + 1)
    np.testing.assert_array_equal(state.inner.v.to_numpy(), np.stack([values + 2, values + 3], axis=-1))
    np.testing.assert_array_equal(state.m.to_numpy(),
        np.stack([values + i for i in (4, 5, 6, 7)], axis=-1).reshape(*shape, 2, 2))
    np.testing.assert_array_equal(state.x.grad.to_numpy(), values + 8)
    np.testing.assert_array_equal(state.inner.v.dual.to_numpy(),
        np.stack([values + 9, values + 10], axis=-1))
