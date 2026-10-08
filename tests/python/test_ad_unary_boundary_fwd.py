import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("compile_tier", ["fast", "balanced"])
@pytest.mark.parametrize("source", ["arg", "derived_arg", "field"])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_passive_unary_boundary_has_zero_tangent(source, compile_tier):
    ti.cfg.compile_tier = compile_tier
    ti.cfg.advanced_optimization = compile_tier != "fast"
    x = ti.field(ti.f32, shape=(), needs_dual=True)
    y = ti.field(ti.f32, shape=3, needs_dual=True)
    passive = ti.field(ti.f32, shape=2)
    x[None] = 2

    @ti.kernel
    def evaluate(c0: ti.f32, c1: ti.f32, adjustment: ti.f32):
        v0, v1 = c0, c1
        if ti.static(source == "derived_arg"):
            v0, v1 = c0 + adjustment, c1 + adjustment
        elif ti.static(source == "field"):
            v0, v1 = passive[0], passive[1]
        y[0] = x[None] + ti.sqrt(v0)
        y[1] = x[None] + ti.asin(v1)
        y[2] = x[None] + ti.acos(v1)

    for c0, c1 in [(4.0, 0.25), (0.0, 1.0), (0.0, -1.0)]:
        passive[0], passive[1] = c0, c1
        for seed in (1.0, -0.5):
            with ti.ad.FwdMode(loss=y, param=x, seed=[seed]):
                evaluate(c0, c1, 0.0)
            expected = 2 + np.array([np.sqrt(c0), np.arcsin(c1), np.arccos(c1)])
            np.testing.assert_allclose(y.to_numpy(), expected, rtol=2e-6)
            np.testing.assert_allclose(y.dual.to_numpy(), seed, atol=1e-6)


@pytest.mark.parametrize("compile_tier", ["fast", "balanced"])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_unary_tensor_masks_only_zero_tangent_lanes(compile_tier):
    ti.cfg.compile_tier = compile_tier
    ti.cfg.advanced_optimization = compile_tier != "fast"
    x = ti.field(ti.f32, shape=2, needs_dual=True)
    y = ti.field(ti.f32, shape=(3, 2), needs_dual=True)
    x[0], x[1] = 0, 0.25

    @ti.kernel
    def evaluate():
        v = ti.Vector([x[0], x[1]])
        roots = ti.sqrt(v)
        arcsines = ti.asin(v + ti.Vector([1.0, 0.0]))
        arccosines = ti.acos(v + ti.Vector([1.0, 0.0]))
        for i in ti.static(range(2)):
            y[0, i] = x[1] + roots[i]
            y[1, i] = x[1] + arcsines[i]
            y[2, i] = x[1] + arccosines[i]

    for seed in (1.0, -0.5):
        with ti.ad.FwdMode(loss=y, param=x, seed=[0, seed]):
            evaluate()
        expected_primal = 0.25 + np.array([[0, 0.5], [np.pi / 2, np.arcsin(0.25)], [0, np.arccos(0.25)]])
        expected_dual = [[1, 2], [1, 1 + 1 / np.sqrt(1 - 0.25**2)], [1, 1 - 1 / np.sqrt(1 - 0.25**2)]]
        np.testing.assert_allclose(y.to_numpy(), expected_primal, rtol=2e-6)
        np.testing.assert_allclose(y.dual.to_numpy(), seed * np.array(expected_dual), atol=2e-6)


@test_utils.test(arch=ti.cpu, compile_tier="fast", fast_math=False, offline_cache=False)
def test_nonzero_unary_tangent_keeps_singular_derivative():
    x = ti.field(ti.f32, shape=2, needs_dual=True)
    y = ti.field(ti.f32, shape=3, needs_dual=True)
    x[0], x[1] = 0, 1

    @ti.kernel
    def evaluate():
        y[0] = ti.sqrt(x[0])
        y[1] = ti.asin(x[1])
        y[2] = ti.acos(x[1])

    with ti.ad.FwdMode(loss=y, param=x, seed=[1, 1]):
        evaluate()
    np.testing.assert_array_equal(y.dual.to_numpy(), [np.inf, np.inf, -np.inf])


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], compile_tier="fast", offline_cache=False)
def test_passive_rsqrt_does_not_evaluate_overflowing_derivative():
    x = ti.field(ti.f32, shape=(), needs_dual=True)
    y = ti.field(ti.f32, shape=(), needs_dual=True)
    x[None] = 2

    @ti.kernel
    def evaluate(value: ti.f32):
        y[None] = x[None] + ti.rsqrt(value)

    with ti.ad.FwdMode(loss=y, param=x):
        evaluate(1e-26)  # Finite primal; its local f32 derivative overflows.
    assert np.isfinite(y[None])
    np.testing.assert_allclose(y[None], 1e13, rtol=2e-6)
    assert y.dual[None] == 1
