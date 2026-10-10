import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


def check_matrix_component_adjoint(tier, advanced, nested, dtype=ti.f32):
    ti.cfg.compile_tier = tier
    ti.cfg.advanced_optimization = advanced
    x = ti.Matrix.field(3, 2, dtype, shape=2, needs_grad=True, needs_dual=True)
    loss = ti.field(dtype, shape=(), needs_grad=True, needs_dual=True)

    @ti.func
    def density(f):
        # Repeated reads must accumulate into the same component; the other
        # components retain their own gradients. Include a local matrix store.
        g = f * f
        g[1, 0] = f[1, 0] * f[2, 1]
        return g.sum() + f[0, 1] * f[2, 0] + f[0, 1] * f[1, 1]

    @ti.kernel
    def energy():
        for i in x:
            total = ti.cast(0.0, dtype)
            if ti.static(nested):
                for j in range(2):
                    for k in range(2):
                        f = x[i] + 0.05 * j - 0.02 * k
                        total += density(f)
            else:
                total = density(x[i])
            loss[None] += total

    direction = np.linspace(-0.2, 0.3, 12).reshape(2, 3, 2).astype(np.float32)
    for shift in [0.0, -0.3]:
        values = (np.linspace(-0.4, 0.7, 12).reshape(2, 3, 2) + shift).astype(np.float32)
        expected, gradient = 0.0, np.zeros_like(values, dtype=np.float64)
        offsets = [0.05 * j - 0.02 * k for j in range(2) for k in range(2)] if nested else [0.0]
        for offset in offsets:
            f = values.astype(np.float64) + offset
            g = f * f
            g[:, 1, 0] = f[:, 1, 0] * f[:, 2, 1]
            expected += g.sum() + (f[:, 0, 1] * (f[:, 2, 0] + f[:, 1, 1])).sum()
            grad = 2 * f
            grad[:, 1, 0] = f[:, 2, 1]
            grad[:, 2, 1] += f[:, 1, 0]
            grad[:, 0, 1] += f[:, 2, 0] + f[:, 1, 1]
            grad[:, 2, 0] += f[:, 0, 1]
            grad[:, 1, 1] += f[:, 0, 1]
            gradient += grad
        x.from_numpy(values)
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
            np.testing.assert_allclose(loss[None], expected, rtol=2e-5, atol=1e-6)
            np.testing.assert_allclose(x.grad.to_numpy(), gradient, rtol=2e-5, atol=1e-6)
        loss[None] = 0
        with ti.ad.FwdMode(loss=loss, param=x, seed=direction):
            energy()
        np.testing.assert_allclose(loss[None], expected, rtol=2e-5, atol=1e-6)
        np.testing.assert_allclose(loss.dual[None], (gradient * direction).sum(), rtol=2e-5, atol=1e-6)


@pytest.mark.parametrize("tier,advanced", [("fast", False), ("fast", True), ("full", True)])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_matrix_component_adjoint(tier, advanced):
    check_matrix_component_adjoint(tier, advanced, nested=False)


@pytest.mark.parametrize("tier,advanced", [("fast", False), ("fast", True), ("full", True)])
@test_utils.test(arch=[ti.cpu, ti.cuda], require=ti.extension.adstack, offline_cache=False)
def test_matrix_component_adjoint_with_history(tier, advanced):
    check_matrix_component_adjoint(tier, advanced, nested=True)


@pytest.mark.parametrize("tier,advanced", [("fast", False), ("fast", True), ("full", True)])
@test_utils.test(
    arch=[ti.cpu, ti.cuda],
    require=[ti.extension.adstack, ti.extension.data64],
    offline_cache=False,
)
def test_matrix_component_adjoint_with_history_f64(tier, advanced):
    check_matrix_component_adjoint(tier, advanced, nested=True, dtype=ti.f64)
