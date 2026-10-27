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


@pytest.mark.parametrize("cfg_enabled", [False, True])
@pytest.mark.parametrize("fast_math", [False, True])
@test_utils.test(
    arch=[ti.cpu, ti.cuda],
    require=ti.extension.adstack,
    compile_tier="fast",
    advanced_optimization=False,
    offline_cache=False,
)
def test_fast_reverse_nested_branch_recurrence(cfg_enabled, fast_math):
    ti.cfg.cfg_optimization = cfg_enabled
    ti.cfg.fast_math = fast_math
    n = 8
    x = ti.field(ti.f32, shape=n, needs_grad=True)
    y = ti.field(ti.f32, shape=n, needs_grad=True)

    @ti.kernel
    def evaluate(end: ti.i32):
        for i in range(n):
            state = x[i]
            for j in range(end):
                scale = 0.75
                if (i + j) % 2 == 0:
                    scale = 1.125
                state = scale * state + 0.125 * x[(i + 1) % n]
            y[i] = state * state

    # Independent f64 recurrence followed by an analytic square derivative.
    # Cover loop-carried values, branch assignments, cross-element atomic
    # accumulation and reuse with empty loop bounds.
    for end in [6, 3, 0]:
        values = np.linspace(-0.7, 0.9, n, dtype=np.float32)
        seeds = np.array([1, -2, 0.5, -1, 2, 0.25, -0.5, 1], dtype=np.float32)
        matrix = np.eye(n)
        for i in range(n):
            for j in range(end):
                matrix[i] *= 1.125 if (i + j) % 2 == 0 else 0.75
                matrix[i, (i + 1) % n] += 0.125
        state = matrix @ values.astype(np.float64)
        expected = matrix.T @ (2 * state * seeds)
        x.from_numpy(values)
        x.grad.fill(0)
        y.grad.from_numpy(seeds)
        evaluate(end)
        evaluate.grad(end)
        np.testing.assert_allclose(y.to_numpy(), state**2, rtol=2e-6, atol=1e-7)
        np.testing.assert_allclose(x.grad.to_numpy(), expected, rtol=3e-6, atol=1e-7)
