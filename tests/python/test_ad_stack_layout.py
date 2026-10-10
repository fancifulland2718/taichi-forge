import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("dtype", [ti.f32, ti.f64])
@pytest.mark.parametrize("capacity", [0, 2])
@pytest.mark.parametrize(
    "tier,advanced,llvm_level",
    [("fast", False, 3), ("fast", True, 3), ("full", True, 1), ("full", True, 2), ("full", True, 3)],
)
@test_utils.test(arch=[ti.cpu, ti.cuda], require=ti.extension.adstack, offline_cache=False)
def test_reverse_branch_uses_last_stack_adjoint(dtype, capacity, tier, advanced, llvm_level):
    ti.cfg.compile_tier = tier
    ti.cfg.advanced_optimization = advanced
    ti.cfg.llvm_opt_level = llvm_level
    ti.cfg.ad_stack_size = capacity
    x = ti.field(dtype, shape=4, needs_grad=True)
    y = ti.field(dtype, shape=4, needs_grad=True)

    @ti.kernel
    def evaluate():
        for i in x:
            fractional = x[i] - ti.floor(x[i]) if x[i] > 0 else x[i] - ti.ceil(x[i])
            y[i] = fractional * fractional

    # Initial local zero and branch assignment fill both stack entries.
    # Repeat with changed inputs/seeds to exercise the same compiled adjoint.
    for values, seed in [([-0.25, -0.75, 0.25, 0.75], 1.0), ([0.5, -1.25, 1.75, -0.5], -0.5)]:
        values = np.asarray(values, dtype=np.float64 if dtype == ti.f64 else np.float32)
        x.from_numpy(values)
        x.grad.fill(0)
        y.grad.fill(seed)
        evaluate()
        evaluate.grad()
        fractional = np.modf(values)[0]
        np.testing.assert_allclose(y.to_numpy(), fractional**2, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(x.grad.to_numpy(), 2 * seed * fractional, rtol=1e-6, atol=1e-7)
