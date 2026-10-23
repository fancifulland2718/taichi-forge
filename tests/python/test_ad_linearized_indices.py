"""Dependent field indices and branch-local values survive batched type checks."""

import numpy as np
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_multidimensional_fields_branch_and_adjoint():
    x = ti.Vector.field(3, ti.f32, needs_grad=True)
    y = ti.field(ti.f32, needs_grad=True)
    ti.root.dense(ti.i, 4).dense(ti.j, 8).place(x, x.grad, y, y.grad)

    @ti.kernel
    def evaluate():
        for i, j in y:
            value = 0.0
            for k in ti.static(range(3)):
                sample = x[(i + k) % 4, (j + 2 * k) % 8][k]
                value += sample * sample
            weight = 1.0
            if (i + j) % 2:
                weight = 2.0
            y[i, j] = weight * value

    source = np.arange(96, dtype=np.float32).reshape(4, 8, 3) / 32 - 1.5
    x.from_numpy(source)
    x.grad.fill(0)
    y.grad.fill(1)
    evaluate()
    evaluate.grad()
    forward = np.zeros((4, 8), np.float32)
    gradient = np.zeros_like(source)
    for i in range(4):
        for j in range(8):
            weight = 1 + (i + j) % 2
            for k in range(3):
                index = ((i + k) % 4, (j + 2 * k) % 8, k)
                forward[i, j] += weight * source[index] ** 2
                gradient[index] += 2 * weight * source[index]
    np.testing.assert_allclose(y.to_numpy(), forward, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(x.grad.to_numpy(), gradient, rtol=1e-6, atol=1e-6)
