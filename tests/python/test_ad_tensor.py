import functools

import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils

has_autograd = False

try:
    import autograd.numpy as np
    from autograd import grad

    has_autograd = True
except:
    pass


def if_has_autograd(func):
    # functools.wraps is nececssary for pytest parametrization to work
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if has_autograd:
            func(*args, **kwargs)

    return wrapper


@if_has_autograd
@test_utils.test()
def test_ad_tensor_store_load():
    x = ti.Vector.field(4, dtype=ti.f32, shape=(), needs_grad=True)
    y = ti.Vector.field(4, dtype=ti.f32, shape=(), needs_grad=True)

    @ti.kernel
    def test(tmp: ti.f32):
        b = ti.Vector([tmp, tmp, tmp, tmp])
        b[0] = tmp * 4
        y[None] = b * x[None]

    y.grad.fill(2.0)
    test.grad(10)

    assert (x.grad.to_numpy() == [80.0, 20.0, 20.0, 20.0]).all()


@test_utils.test()
def test_reverse_mode_local_vector_dynamic_read_routes_gradient():
    n = 12
    x = ti.field(ti.f32, shape=n, needs_grad=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)
    x.from_numpy(np.linspace(0.25, 3.0, n, dtype=np.float32))

    @ti.kernel
    def forward():
        for i in range(n):
            value = x[i]
            local = ti.Vector([value, 2.0 * value, 3.0 * value])
            loss[None] += local[(i + 1) % 3]

    with ti.ad.Tape(loss):
        forward()

    expected = np.array([2.0, 3.0, 1.0] * 4, dtype=np.float32)
    np.testing.assert_array_equal(x.grad.to_numpy(), expected)


@test_utils.test()
def test_reverse_mode_local_matrix_dynamic_read_routes_gradient():
    n = 8
    x = ti.field(ti.f32, shape=n, needs_grad=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)
    x.from_numpy(np.arange(1, n + 1, dtype=np.float32))

    @ti.kernel
    def forward():
        for i in range(n):
            value = x[i]
            local = ti.Matrix(
                [[value, 2.0 * value], [3.0 * value, 4.0 * value]]
            )
            loss[None] += local[(i // 2) % 2, i % 2]

    with ti.ad.Tape(loss):
        forward()

    expected = np.array([1.0, 2.0, 3.0, 4.0] * 2, dtype=np.float32)
    np.testing.assert_array_equal(x.grad.to_numpy(), expected)
