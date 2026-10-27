import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("cfg_enabled", [False, True])
@pytest.mark.parametrize("fast_math", [False, True])
@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    compile_tier="fast",
    advanced_optimization=False,
    offline_cache=False,
)
def test_fast_reverse_nested_conditional_field_reads(cfg_enabled, fast_math):
    ti.cfg.cfg_optimization = cfg_enabled
    ti.cfg.fast_math = fast_math
    n = 8
    x = ti.field(ti.f32, shape=(n, n), needs_grad=True)
    y = ti.field(ti.f32, shape=(n, n), needs_grad=True)

    @ti.func
    def read(i, j):
        value = 0.0
        if 0 <= i < n and 0 <= j < n:
            value = x[i, j]
        return value

    @ti.kernel
    def evaluate(end: ti.i32):
        for i in range(n):
            for j in range(end):
                v = read(i - 1, j) + 2.0 * read(i, j) - read(i + 1, j) + 0.5 * read(i, j) + 0.25 * read(i, j - 1)
                y[i, j] = v * v + 0.125 * read(i, j)

    values = (np.arange(n * n).reshape(n, n) - 31) / 32
    weights = (np.arange(n * n).reshape(n, n) + 1) / 64
    # Different bounds reuse the same specialization, including an empty loop.
    # The reference applies the transpose stencil directly, including edges
    # and contributions from neighboring outputs, without using Taichi AD.
    for end in (n, 5, 0):
        x.from_numpy(values.astype(np.float32))
        y.fill(0)
        x.grad.fill(0)
        y.grad.from_numpy(weights.astype(np.float32))
        evaluate(end)
        evaluate.grad(end)
        expected_y, expected_grad = np.zeros_like(values), np.zeros_like(values)
        for i in range(n):
            for j in range(end):
                terms = [(i - 1, j, 1.0), (i, j, 2.5), (i + 1, j, -1.0), (i, j - 1, 0.25)]
                valid = [(ii, jj, c) for ii, jj, c in terms if 0 <= ii < n and 0 <= jj < n]
                value = sum(c * values[ii, jj] for ii, jj, c in valid)
                expected_y[i, j] = value * value + 0.125 * values[i, j]
                for ii, jj, c in valid:
                    expected_grad[ii, jj] += 2 * value * c * weights[i, j]
                expected_grad[i, j] += 0.125 * weights[i, j]
        np.testing.assert_allclose(y.to_numpy(), expected_y, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(x.grad.to_numpy(), expected_grad, rtol=2e-6, atol=2e-6)
