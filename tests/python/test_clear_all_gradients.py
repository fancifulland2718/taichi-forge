from taichi_forge.lang import impl

import numpy as np
import taichi_forge as ti
from tests import test_utils


@test_utils.test(exclude=[ti.vulkan, ti.dx11])
def test_clear_all_gradients():
    x = ti.field(ti.f32)
    y = ti.field(ti.f32)
    z = ti.field(ti.f32)
    w = ti.field(ti.f32)

    n = 128

    ti.root.place(x)
    ti.root.dense(ti.i, n).place(y)
    ti.root.dense(ti.i, n).dense(ti.j, n).place(z, w)
    ti.root.lazy_grad()

    x.grad[None] = 3
    for i in range(n):
        y.grad[i] = 3
        for j in range(n):
            z.grad[i, j] = 5
            w.grad[i, j] = 6

    ti.ad.clear_all_gradients()

    assert x.grad[None] == 0
    for i in range(n):
        assert y.grad[i] == 0
        for j in range(n):
            assert z.grad[i, j] == 0
            assert w.grad[i, j] == 0

    compiled_functions = impl.get_runtime().get_num_compiled_functions()
    ti.ad.clear_all_gradients()
    assert impl.get_runtime().get_num_compiled_functions() == compiled_functions


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_clear_all_gradients_dense_native_backends():
    n = 256
    x = ti.field(ti.f32, shape=(n,), needs_grad=True)

    x.grad.from_numpy(np.full((n,), 7, dtype=np.float32))
    ti.ad.clear_all_gradients()
    np.testing.assert_array_equal(x.grad.to_numpy(), np.zeros((n,), dtype=np.float32))

    compiled_functions = impl.get_runtime().get_num_compiled_functions()
    ti.ad.clear_all_gradients()
    assert impl.get_runtime().get_num_compiled_functions() == compiled_functions
    np.testing.assert_array_equal(x.grad.to_numpy(), np.zeros((n,), dtype=np.float32))


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_clear_all_gradients_matrix_field_dense_native_backends():
    n = 257
    x = ti.Matrix.field(2, 3, ti.f32, shape=(n,), needs_grad=True)
    values = np.arange(n * 6, dtype=np.float32).reshape(n, 2, 3) + 1

    x.grad.from_numpy(values)
    ti.ad.clear_all_gradients()
    np.testing.assert_array_equal(
        x.grad.to_numpy(), np.zeros((n, 2, 3), dtype=np.float32)
    )

    compiled_functions = impl.get_runtime().get_num_compiled_functions()
    x.grad.from_numpy(values)
    ti.ad.clear_all_gradients()
    assert impl.get_runtime().get_num_compiled_functions() == compiled_functions
    np.testing.assert_array_equal(
        x.grad.to_numpy(), np.zeros((n, 2, 3), dtype=np.float32)
    )
