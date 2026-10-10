import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("iterations", [2, 4, 8])
@pytest.mark.parametrize("square", [False, True])
@test_utils.test(
    arch=[ti.cpu, ti.cuda],
    require=ti.extension.adstack,
    compile_tier="fast",
    advanced_optimization=False,
    ad_stack_size=0,
    offline_cache=False,
)
def test_matrix_reduction_stack_in_finite_nested_loops(iterations, square):
    x = ti.Matrix.field(3, 3, ti.f32, shape=1, needs_grad=True)
    y = ti.field(ti.f32, shape=1, needs_grad=True)

    @ti.kernel
    def energy():
        for i in x:
            total = 0.0
            for j in range(iterations):
                for k in range(iterations):
                    f = x[i] + (0.01 * j + 0.02 * k) * ti.Matrix.identity(ti.f32, 3)
                    if ti.static(square):
                        total += (f * f).sum()
                    else:
                        total += f.sum()
            y[i] = total

    # Each matrix reduction pushes multiple intermediate sums. Even 2x2
    # iterations need 41 entries, exceeding the old fallback of 32.
    for offset, seed in [(0.05, 1.0), (-0.07, -0.5)]:
        values = (np.eye(3) + offset).astype(np.float32)[None]
        x.from_numpy(values)
        x.grad.fill(0)
        y.grad.fill(seed)
        energy()
        energy.grad()
        expected, gradient = 0.0, np.zeros_like(values)
        for j in range(iterations):
            for k in range(iterations):
                f = values + (0.01 * j + 0.02 * k) * np.eye(3)
                expected += (f * f).sum() if square else f.sum()
                gradient += 2 * f if square else np.ones_like(f)
        np.testing.assert_allclose(y[0], expected, rtol=2e-5, atol=1e-6)
        np.testing.assert_allclose(x.grad.to_numpy(), seed * gradient, rtol=2e-5, atol=1e-6)
