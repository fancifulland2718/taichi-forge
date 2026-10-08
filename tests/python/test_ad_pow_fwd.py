import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("compile_tier", ["fast", "balanced", "full"])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_constant_power_forward_derivative_at_zero_and_negative_inputs(compile_tier):
    ti.cfg.compile_tier = compile_tier
    ti.cfg.advanced_optimization = compile_tier != "fast"
    values = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
    x = ti.field(ti.f32, shape=5)
    y = ti.field(ti.f32, shape=5)
    ti.root.lazy_dual()
    x.from_numpy(values)

    @ti.kernel
    def calculate(exponent: ti.template()):
        for i in x:
            y[i] = x[i] ** exponent

    for exponent in (0, 1, 2, 3, 2.0):
        with ti.ad.FwdMode(loss=y, param=x, seed=[1.0] * 5):
            calculate(exponent)
        expected = np.zeros(5, np.float32) if exponent == 0 else exponent * values ** (exponent - 1)
        np.testing.assert_allclose(y.to_numpy(), values**exponent, atol=1e-6)
        np.testing.assert_allclose(y.dual.to_numpy(), expected, atol=1e-6)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], compile_tier="fast", offline_cache=False)
def test_variable_power_forward_derivative_keeps_both_tangents():
    x = ti.field(ti.f32, shape=2)
    y = ti.field(ti.f32, shape=2)
    ti.root.lazy_dual()
    x[0], x[1] = 2, 3

    @ti.kernel
    def calculate():
        y[0] = x[0] ** x[1]
        y[1] = 2.0 ** x[1]

    for seed in ([1.0, 0.0], [0.0, 1.0], [1.0, 1.0]):
        with ti.ad.FwdMode(loss=y, param=x, seed=seed):
            calculate()
        expected = [12 * seed[0] + 8 * np.log(2) * seed[1], 8 * np.log(2) * seed[1]]
        np.testing.assert_allclose(y.dual.to_numpy(), expected, rtol=1e-5)
