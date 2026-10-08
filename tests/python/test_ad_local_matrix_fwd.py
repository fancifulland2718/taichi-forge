import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("compile_tier", ["fast", "balanced"])
@pytest.mark.parametrize("matrix", [False, True])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_dynamic_local_component_tangent_aliases_storage(compile_tier, matrix):
    ti.cfg.compile_tier = compile_tier
    ti.cfg.advanced_optimization = compile_tier != "fast"
    x = ti.field(ti.f32, shape=(), needs_dual=True)
    y = ti.field(ti.f32, shape=5, needs_dual=True)
    x[None] = 2

    @ti.func
    def make_value(v):
        if ti.static(matrix):
            return ti.Matrix([[v, 2 * v], [3 * v, 4 * v]])
        else:
            return ti.Vector([v, 2 * v, 3 * v, 4 * v])

    @ti.kernel
    def evaluate(index: ti.i32):
        for row in range(1):
            value = make_value(x[None])
            saved = value
            before = 0.0
            if ti.static(matrix):
                before = value[index // 2, index % 2]
                value[index // 2, index % 2] = 3 * x[None]
            else:
                before = value[index]
                value[index] = 3 * x[None]
            after = value.sum()
            for _ in range(2):
                if ti.static(matrix):
                    value[index // 2, index % 2] += x[None]
                else:
                    value[index] += x[None]
            after_updates = value.sum()
            if ti.static(matrix):
                value[index // 2, index % 2] = 7.0
            else:
                value[index] = 7.0
            # Integer component pointers must remain inactive in forward AD.
            indices = ti.Vector([0, 0, 0, 0])
            indices[index] = 1
            y[0] = before
            y[1] = after
            y[2] = after_updates
            y[3] = value.sum() + indices[index]
            y[4] = saved.sum()

    for index in range(4):
        coefficient = index + 1
        for seed in (1.0, -0.5):
            with ti.ad.FwdMode(loss=y, param=x, seed=[seed]):
                evaluate(index)
            expected_dual = np.array([coefficient, 13 - coefficient, 15 - coefficient, 10 - coefficient, 10])
            expected_primal = 2 * expected_dual
            expected_primal[3] += 8
            np.testing.assert_allclose(y.to_numpy(), expected_primal)
            np.testing.assert_allclose(y.dual.to_numpy(), seed * expected_dual)
