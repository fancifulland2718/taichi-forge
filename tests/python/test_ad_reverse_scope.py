import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("tier,advanced", [("fast", False), ("fast", True), ("full", True)])
@pytest.mark.parametrize("conditional", [False, True])
@test_utils.test(arch=[ti.cpu, ti.cuda], require=ti.extension.adstack, offline_cache=False)
def test_nonlinear_nested_loop_adjoint_scope(tier, advanced, conditional):
    ti.cfg.compile_tier = tier
    ti.cfg.advanced_optimization = advanced
    x = ti.field(ti.f32, shape=4, needs_grad=True, needs_dual=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True, needs_dual=True)

    @ti.kernel
    def energy():
        for i in x:
            total = 0.0
            for j in range(2):
                for k in range(2):
                    for q in ti.static(range(4)):
                        f = x[i] + 0.01 * j + 0.02 * k + 0.005 * q
                        value = (f * f * f + ti.sin(f)) / (q + 1)
                        if ti.static(conditional):
                            if k == 0:
                                total += value
                            else:
                                total += 0.5 * value
                        else:
                            total += value
            loss[None] += total

    for offset in [0.0, -0.15]:
        values = (np.linspace(0.1, 0.8, 4) + offset).astype(np.float32)
        x.from_numpy(values)
        expected, gradient = 0.0, np.zeros(4)
        for j in range(2):
            for k in range(2):
                for q in range(4):
                    f = values.astype(np.float64) + 0.01 * j + 0.02 * k + 0.005 * q
                    weight = 0.5 if conditional and k else 1.0
                    expected += weight * ((f**3 + np.sin(f)) / (q + 1)).sum()
                    gradient += weight * (3 * f * f + np.cos(f)) / (q + 1)
        # Both direct reverse calls and Tape reuse compiled kernels with new inputs.
        for tape in [False, True]:
            loss[None] = 0
            x.grad.fill(0)
            if tape:
                with ti.ad.Tape(loss):
                    energy()
            else:
                energy()
                loss.grad[None] = 1
                energy.grad()
            np.testing.assert_allclose(loss[None], expected, rtol=3e-5, atol=3e-6)
            np.testing.assert_allclose(x.grad.to_numpy(), gradient, rtol=3e-5, atol=3e-6)
        direction = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
        loss[None] = 0
        with ti.ad.FwdMode(loss=loss, param=x, seed=direction):
            energy()
        np.testing.assert_allclose(loss[None], expected, rtol=3e-5, atol=3e-6)
        np.testing.assert_allclose(loss.dual[None], gradient @ direction, rtol=3e-5, atol=3e-6)
