import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


def check_unary_tensor_derivatives(dtype, matrix):
    x = ti.field(dtype, shape=(), needs_dual=True)
    y = ti.field(dtype, shape=(5, 4), needs_dual=True)
    x[None] = 0.1

    @ti.func
    def make_value(value):
        if ti.static(matrix):
            return ti.Matrix([[value, 2 * value], [3 * value, 4 * value]])
        else:
            return ti.Vector([value, 2 * value, 3 * value, 4 * value])

    @ti.kernel
    def evaluate():
        v = make_value(x[None])
        for op in ti.static(range(5)):
            r = v
            if ti.static(op == 0):
                r = ti.tanh(v)
            elif ti.static(op == 1):
                r = ti.sqrt(v)
            elif ti.static(op == 2):
                r = ti.asin(v)
            elif ti.static(op == 3):
                r = ti.acos(v)
            else:
                r = ti.rsqrt(v)
            for i in ti.static(range(4)):
                if ti.static(matrix):
                    y[op, i] = r[i // 2, i % 2]
                else:
                    y[op, i] = r[i]

    scales = np.arange(1, 5)
    values = scales * x[None]
    expected_primal = np.array(
        [np.tanh(values), np.sqrt(values), np.arcsin(values), np.arccos(values), 1 / np.sqrt(values)]
    )
    expected_derivative = scales * np.array(
        [
            1 - np.tanh(values) ** 2,
            0.5 / np.sqrt(values),
            1 / np.sqrt(1 - values**2),
            -1 / np.sqrt(1 - values**2),
            -0.5 / values**1.5,
        ]
    )
    tolerance = 3e-6 if dtype == ti.f32 else 1e-12
    for seed in (1.0, -0.5):
        with ti.ad.FwdMode(loss=y, param=x, seed=[seed]):
            evaluate()
        np.testing.assert_allclose(y.to_numpy(), expected_primal, rtol=tolerance)
        np.testing.assert_allclose(y.dual.to_numpy(), seed * expected_derivative, rtol=tolerance)


@pytest.mark.parametrize("compile_tier", ["fast", "balanced"])
@pytest.mark.parametrize("matrix", [False, True])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_forward_unary_tensor_types(compile_tier, matrix):
    ti.cfg.compile_tier = compile_tier
    ti.cfg.advanced_optimization = compile_tier != "fast"
    check_unary_tensor_derivatives(ti.f32, matrix)


@pytest.mark.parametrize("matrix", [False, True])
@test_utils.test(arch=[ti.cpu, ti.cuda], require=ti.extension.data64, compile_tier="fast", offline_cache=False)
def test_forward_unary_tensor_double_precision(matrix):
    check_unary_tensor_derivatives(ti.f64, matrix)
