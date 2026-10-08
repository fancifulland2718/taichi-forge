import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("compile_tier", ["fast", "balanced", "full"])
@pytest.mark.parametrize("layout", [ti.Layout.AOS, ti.Layout.SOA])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_forward_matrix_constructor_loads_component_tangents(compile_tier, layout):
    ti.cfg.compile_tier = compile_tier
    ti.cfg.advanced_optimization = compile_tier != "fast"
    param = ti.Matrix.field(2, 2, ti.f32, shape=(), needs_dual=True, layout=layout)
    loss = ti.field(ti.f32, shape=(), needs_dual=True)
    param[None] = [[2, 3], [4, 5]]

    @ti.kernel
    def evaluate():
        value = param[None]
        # Mix derivatives of different components with a constant tangent.
        mixed = ti.Matrix([[value[0, 0] * value[1, 1], 7.0], [value[0, 1], -value[1, 0]]])
        loss[None] = mixed[0, 0] + mixed[0, 1] + 3.0 * mixed[1, 0] + 4.0 * mixed[1, 1]

    for seed in ([1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0], [-2.0, 1.0, 0.0, 3.0]):
        with ti.ad.FwdMode(loss=loss, param=param, seed=seed):
            evaluate()
        assert loss[None] == 10.0
        assert loss.dual[None] == 5 * seed[0] + 3 * seed[1] - 4 * seed[2] + 2 * seed[3]
        np.testing.assert_array_equal(param.dual.to_numpy(), np.zeros((2, 2)))
