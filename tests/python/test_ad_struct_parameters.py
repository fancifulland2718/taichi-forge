"""Compiler-generated parameter copies must not be reversed as user code."""

import numpy as np
import pytest
import taichi_forge as ti

from tests import test_utils


def parameter_type():
    inner = ti.types.struct(scale=ti.f32, code=ti.i32)
    record = ti.types.struct(inner=inner, bias=ti.types.vector(3, ti.f32),
                             basis=ti.types.matrix(2, 2, ti.f32))
    return record, record(inner=inner(scale=2, code=7), bias=[1, -2, 3],
                          basis=[[3, 0], [0, 4]])


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_nested_struct_parameters_reverse_loop_order():
    record, value = parameter_type()
    x = ti.ndarray(ti.f32, 3, needs_grad=True)
    temporary = ti.ndarray(ti.f32, 3, needs_grad=True)
    loss = ti.ndarray(ti.f32, 1, needs_grad=True)

    @ti.kernel
    def energy(p: record, source: ti.types.ndarray(),
               middle: ti.types.ndarray(), result: ti.types.ndarray()):
        for i in range(3):
            middle[i] = p.inner.scale * source[i] + p.bias[i]
        for i in range(3):
            result[0] += p.basis[i % 2, i % 2] * middle[i] * middle[i]

    x.from_numpy(np.array([1, -2, 3], dtype=np.float32))
    x.grad.fill(0)
    temporary.grad.fill(0)
    loss.fill(0)
    loss.grad.fill(1)
    energy(value, x, temporary, loss)
    energy.grad(value, x, temporary, loss)
    assert loss.to_numpy() == pytest.approx([414])
    assert x.grad.to_numpy() == pytest.approx([36, -96, 108])


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_struct_parameter_copies_preserve_mutation_semantics():
    record, value = parameter_type()

    @ti.kernel
    def modify(p: record) -> ti.f32:
        p.inner.scale += 5
        p.bias[0] = p.inner.scale
        return p.inner.scale + p.bias[0] + p.basis[1, 1]

    assert modify(value) == 18
    assert value.inner.scale == 2 and value.bias[0] == 1
    assert modify(value) == 18


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_struct_parameter_prefix_does_not_hide_user_statements():
    record, value = parameter_type()
    loss = ti.field(ti.f32, shape=(), needs_grad=True)

    @ti.kernel
    def invalid(p: record):
        loss[None] = p.inner.scale
        for i in range(3):
            loss[None] += p.bias[i]

    with pytest.raises(RuntimeError, match="Mixed usage of for-loops"):
        invalid.grad(value)


@pytest.mark.parametrize("member", ["scalar", "tensor"])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_reverse_parameter_mutation_requires_a_loop_local_copy(member):
    record, value = parameter_type()
    x = ti.ndarray(ti.f32, 3, needs_grad=True)
    loss = ti.ndarray(ti.f32, 1, needs_grad=True)

    @ti.kernel
    def invalid(p: record, source: ti.types.ndarray(), result: ti.types.ndarray()):
        for i in range(1):
            if ti.static(member == "scalar"):
                p.inner.scale = source[0] * source[0]
            else:
                p.bias[0] = source[0] * source[0]
        for i in range(1):
            result[0] = p.inner.scale * p.inner.scale + p.bias[0] * p.bias[0]

    with pytest.raises(RuntimeError, match="constants across loops"):
        invalid.grad(value, x, loss)

    @ti.kernel
    def local(p: record, source: ti.types.ndarray(), result: ti.types.ndarray()):
        for i in range(3):
            copy = p
            if ti.static(member == "scalar"):
                copy.inner.scale *= source[i]
                result[0] += copy.inner.scale * copy.inner.scale
            else:
                copy.bias[0] = copy.inner.scale * source[i]
                result[0] += copy.bias[0] * copy.bias[0]

    x.from_numpy(np.array([1, -2, 3], dtype=np.float32))
    x.grad.fill(0)
    loss.fill(0)
    loss.grad.fill(1)
    local(value, x, loss)
    local.grad(value, x, loss)
    assert loss.to_numpy() == pytest.approx([56])
    assert x.grad.to_numpy() == pytest.approx([8, -16, 24])
