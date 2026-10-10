import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("bounds", ["constant", "argument", "local"])
@pytest.mark.parametrize("capacity", [0, 32])
@pytest.mark.parametrize("tier,advanced", [("fast", False), ("fast", True), ("full", True)])
@test_utils.test(arch=[ti.cpu, ti.cuda], require=ti.extension.adstack, offline_cache=False)
def test_reverse_nested_loop_bounds(bounds, capacity, tier, advanced):
    ti.cfg.compile_tier = tier
    ti.cfg.advanced_optimization = advanced
    ti.cfg.ad_stack_size = capacity
    x = ti.field(ti.f32, shape=2, needs_grad=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)

    @ti.kernel
    def evaluate(begin: ti.i32, end: ti.i32):
        for i in x:
            for j in range(2):
                lo, hi = 1, 4
                if ti.static(bounds == "argument"):
                    lo, hi = begin, end
                elif ti.static(bounds == "local"):
                    lo, hi = begin + j, end + j
                total = 0.0
                for k in range(lo, hi):
                    total += (k + 1) * x[i] * x[i]
                loss[None] += total

    # Reuse primal/adjoint kernels after changing inputs and dynamic bounds.
    # The final case is empty and must not reuse an earlier gradient.
    for begin, end, values in [(0, 3, [0.5, -0.75]), (2, 4, [-0.25, 1.5]), (2, 2, [1.0, -0.5])]:
        values = np.asarray(values, dtype=np.float32)
        x.from_numpy(values)
        weight = 0
        for j in range(2):
            lo, hi = (1, 4) if bounds == "constant" else (begin, end)
            if bounds == "local":
                lo, hi = lo + j, hi + j
            weight += sum(k + 1 for k in range(lo, hi))
        for tape in [False, True]:
            loss[None] = 0
            x.grad.fill(0)
            if tape:
                with ti.ad.Tape(loss=loss):
                    evaluate(begin, end)
            else:
                evaluate(begin, end)
                loss.grad[None] = 1
                evaluate.grad(begin, end)
            np.testing.assert_allclose(loss[None], weight * np.dot(values, values), rtol=1e-6)
            np.testing.assert_allclose(x.grad.to_numpy(), 2 * weight * values, rtol=1e-6, atol=1e-7)
