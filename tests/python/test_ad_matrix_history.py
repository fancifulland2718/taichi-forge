import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("tier,advanced", [("fast", False), ("fast", True), ("full", True)])
@test_utils.test(arch=[ti.cpu, ti.cuda], require=ti.extension.adstack, offline_cache=False)
def test_nonlinear_matrix_history_in_nested_loops(tier, advanced):
    ti.cfg.compile_tier = tier
    ti.cfg.advanced_optimization = advanced
    x = ti.Matrix.field(3, 3, ti.f32, shape=2, needs_grad=True, needs_dual=True)
    y = ti.field(ti.f32, shape=2, needs_grad=True, needs_dual=True)

    @ti.func
    def density(f):
        e = 0.5 * (f.transpose() @ f - ti.Matrix.identity(ti.f32, 3))
        return (e * e).sum()

    @ti.kernel
    def energy():
        for i in x:
            total = 0.0
            for j in range(2):
                for k in range(2):
                    f = x[i] + (0.01 * j + 0.02 * k) * ti.Matrix.identity(ti.f32, 3)
                    total += density(f)
            y[i] = total

    perturbation = np.arange(18).reshape(2, 3, 3) * 0.007 - 0.03
    for offset, seed in [(0.05, 1.0), (-0.07, -0.5)]:
        values = (np.eye(3) + perturbation + offset).astype(np.float32)
        x.from_numpy(values)
        x.grad.fill(0)
        y.grad.fill(seed)
        energy()
        energy.grad()
        expected, gradient = np.zeros(2), np.zeros_like(values, dtype=np.float64)
        for j in range(2):
            for k in range(2):
                f = values.astype(np.float64) + (0.01 * j + 0.02 * k) * np.eye(3)
                e = 0.5 * (np.swapaxes(f, -1, -2) @ f - np.eye(3))
                expected += (e * e).sum(axis=(1, 2))
                gradient += 2 * f @ e
        np.testing.assert_allclose(y.to_numpy(), expected, rtol=2e-5, atol=1e-6)
        np.testing.assert_allclose(x.grad.to_numpy(), seed * gradient, rtol=2e-5, atol=1e-6)
        direction = (perturbation + 0.1).astype(np.float32)
        with ti.ad.FwdMode(loss=y, param=x, seed=direction):
            energy()
        np.testing.assert_allclose(y.to_numpy(), expected, rtol=2e-5, atol=1e-6)
        np.testing.assert_allclose(
            y.dual.to_numpy(), (gradient * direction).sum(axis=(1, 2)), rtol=2e-5, atol=1e-6
        )
