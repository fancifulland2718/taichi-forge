import math

import numpy as np
import pytest
from taichi_forge.lang import impl

import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cuda, ti.cpu])
def test_local_matrix_non_constant_index_real_matrix():
    N = 1
    x = ti.Vector.field(3, float, shape=1)

    @ti.kernel
    def test_invariant_cache():
        for i in range(1):
            x[i][1] = x[i][1] + 1.0
            for j in range(1):
                x[i][1] = x[i][1] - 5.0
                for z in range(1):
                    idx = 0
                    if z == 0:
                        idx = 1
                    x_print = x[i][idx]

                    assert x_print == x[i][1]

    test_invariant_cache()


@test_utils.test(arch=ti.cpu, compile_tier="full", cache_loop_invariant_global_vars=False)
def test_full_cpu_loop_uses_current_scalar_arguments_and_array_values():
    values = ti.ndarray(ti.i32, shape=8)

    @ti.kernel
    def accumulate(data: ti.types.ndarray(ti.i32, ndim=1), steps: ti.i32, increment: ti.i32):
        for i in range(8):
            for _ in range(steps):
                data[i] = 2 * data[i] + increment

    values.fill(1)
    accumulate(values, 3, 2)
    accumulate(values, 0, 100)
    accumulate(values, 1, -1)
    assert values.to_numpy().tolist() == [43] * 8


@pytest.mark.parametrize("dtype", [ti.f32, ti.f64])
@test_utils.test(arch=ti.cpu, advanced_optimization=False)
def test_scalar_argument_trig_across_launches(dtype):
    out = ti.field(dtype, shape=(7, 2))

    @ti.kernel
    def evaluate(angle: dtype, count: ti.i32):
        for i in range(7):
            for j in range(count):
                out[i, 0] += ti.sin(angle)
                out[i, 1] += ti.cos(angle)

    for angle, count in ((0.0, 2), (-0.75, 5), (2.25, 0), (2.25, 3)):
        out.fill(0)
        evaluate(angle, count)
        expected = np.tile([count * math.sin(angle), count * math.cos(angle)], (7, 1))
        np.testing.assert_allclose(out.to_numpy(), expected, rtol=1e-6 if dtype == ti.f32 else 1e-13)
