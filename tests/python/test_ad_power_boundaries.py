"""Integer powers retain their domain through conditional reverse replay."""

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("advanced", [False, True])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_conditional_integer_power_base_gradients(advanced):
    ti.lang.impl.current_cfg().advanced_optimization = advanced
    values = np.array([-2, -2, 0, 0, 2, -0.5, 0, -2], np.float32)
    powers = np.array([2, 3, 0, 1, -1, 2, 2, 3], np.int32)
    x = ti.field(ti.f32, shape=8, needs_grad=True, needs_dual=True)
    y = ti.field(ti.f32, shape=8, needs_grad=True, needs_dual=True)
    exponent = ti.field(ti.i32, shape=8)
    mask = ti.field(ti.i32, shape=8)

    @ti.kernel
    def power():
        for i in range(8):
            if mask[i]:
                y[i] = x[i] ** exponent[i]
            else:
                y[i] = 0

    @ti.kernel
    def square():
        for i in range(8):
            if mask[i]:
                y[i] = x[i] ** 2
            else:
                y[i] = 0

    x.from_numpy(values)
    exponent.from_numpy(powers)
    mask.from_numpy(np.array([1] * 7 + [0], np.int32))
    for kernel, expected, derivative in (
        (power, [4, -8, 1, 0, 0.5, 0.25, 0, 0], [-4, 12, 0, 1, -0.25, -1, 0, 0]),
        (square, [4, 4, 0, 0, 4, 0.25, 0, 0], [-4, -4, 0, 0, 4, -1, 0, 0]),
    ):
        x.grad.fill(0)
        y.grad.fill(1)
        kernel()
        kernel.grad()
        np.testing.assert_allclose(y.to_numpy(), expected, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(x.grad.to_numpy(), derivative, rtol=2e-6, atol=2e-6)
        with ti.ad.FwdMode(loss=y, param=x, seed=[1] * 8):
            kernel()
        np.testing.assert_allclose(y.dual.to_numpy(), derivative, rtol=2e-6, atol=2e-6)
