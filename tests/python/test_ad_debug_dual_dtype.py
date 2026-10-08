import pytest

import taichi_forge as ti
from taichi_forge._lib import core
from tests import test_utils


@pytest.mark.parametrize("dtype", [ti.f32, ti.f64])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], debug=True, offline_cache=False)
def test_debug_fields_keep_floating_point_dual_dtype(dtype):
    if dtype == ti.f64 and not core.is_extension_supported(ti.cfg.arch, ti.extension.data64):
        pytest.skip("Backend does not advertise data64 support")
    x = ti.field(dtype, shape=(), needs_grad=True, needs_dual=True)
    loss = ti.field(dtype, shape=(), needs_grad=True, needs_dual=True)
    assert x.dual.dtype == dtype
    assert loss.dual.dtype == dtype
    x[None] = 1.5

    @ti.kernel
    def evaluate():
        loss[None] = x[None] * x[None]

    with ti.ad.FwdMode(loss=loss, param=x, seed=[0.5]):
        evaluate()
    assert loss[None] == 2.25
    assert loss.dual[None] == 1.5

    # The adjoint checkbit still has its backend-specific integer dtype.
    with ti.ad.Tape(loss, validation=True):
        evaluate()
    assert x.grad[None] == 3.0
