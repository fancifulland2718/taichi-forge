import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


def check_runtime_powers(source, values, exponents):
    x = ti.field(ti.f32, shape=len(values), needs_dual=True)
    y = ti.field(ti.f32, shape=len(values), needs_dual=True)
    passive = ti.field(ti.f32, shape=())
    x.from_numpy(np.asarray(values, dtype=np.float32))

    @ti.kernel
    def evaluate(exponent: ti.f32, adjustment: ti.f32):
        for i in x:
            power = exponent
            if ti.static(source == "derived_arg"):
                power = exponent + adjustment
            elif ti.static(source == "field"):
                power = passive[None]
            y[i] = x[i] ** power

    values = np.asarray(values, dtype=np.float32)
    for exponent in exponents:
        passive[None] = exponent
        for seed in (1.0, -0.5):
            with ti.ad.FwdMode(loss=y, param=x, seed=[seed] * len(values)):
                evaluate(exponent, 0.0)
            expected = np.zeros_like(values) if exponent == 0 else seed * exponent * values ** (exponent - 1)
            np.testing.assert_allclose(y.to_numpy(), values**exponent, atol=1e-6)
            np.testing.assert_allclose(y.dual.to_numpy(), expected, atol=1e-6)


@pytest.mark.parametrize("source", ["arg", "derived_arg", "field"])
@pytest.mark.parametrize("compile_tier", ["fast", "balanced"])
@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_passive_runtime_power_at_nonpositive_bases(source, compile_tier):
    ti.cfg.compile_tier = compile_tier
    ti.cfg.advanced_optimization = compile_tier != "fast"
    check_runtime_powers(source, [-2, 0, 2], [0, 1, 2, 3])


@pytest.mark.parametrize("source", ["arg", "derived_arg", "field"])
@pytest.mark.parametrize("compile_tier", ["fast", "balanced"])
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_passive_runtime_power_in_vulkan_pow_domain(source, compile_tier):
    ti.cfg.compile_tier = compile_tier
    ti.cfg.advanced_optimization = compile_tier != "fast"
    # GLSL pow does not define negative bases or zero raised to <= 0.
    check_runtime_powers(source, [0, 2], [1, 2, 3])


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], compile_tier="fast", offline_cache=False)
def test_runtime_vector_power_keeps_active_exponent_tangent():
    x = ti.field(ti.f32, shape=2, needs_dual=True)
    y = ti.Vector.field(2, ti.f32, shape=(), needs_dual=True)
    x[0], x[1] = 2, 3

    @ti.kernel
    def evaluate():
        y[None] = ti.Vector([x[0], 2 * x[0]]) ** ti.Vector([x[1], x[1]])

    for seed in ([1, 0], [0, 1], [1, 1]):
        with ti.ad.FwdMode(loss=[y.get_scalar_field(i) for i in range(2)], param=x, seed=seed):
            evaluate()
        expected = [12 * seed[0] + 8 * np.log(2) * seed[1], 96 * seed[0] + 64 * np.log(4) * seed[1]]
        np.testing.assert_allclose(y[None].to_numpy(), [8, 64], rtol=1e-5)
        np.testing.assert_allclose(y.dual[None].to_numpy(), expected, rtol=1e-5)
