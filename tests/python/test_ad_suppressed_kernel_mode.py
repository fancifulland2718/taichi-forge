import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.lang.enums import AutodiffMode
from tests import test_utils


@pytest.mark.parametrize("decorator", [ti.ad.grad_replaced, ti.ad.no_grad])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_custom_scope_reuses_forward_kernel_as_primal(decorator):
    x = ti.field(ti.f32, shape=4, needs_dual=True)
    y = ti.field(ti.f32, shape=4, needs_dual=True)
    x.fill(2)

    @ti.kernel
    def forward(scale: ti.f32):
        for i in x:
            y[i] += scale * x[i]

    @ti.ad.no_grad
    def inner():
        forward(1)

    @decorator
    def suppressed():
        inner()
        forward(scale=1)
        with pytest.raises(ti.TaichiRuntimeTypeError):
            forward("invalid scalar")

    with ti.ad.FwdMode(loss=y, param=x, seed=[1] * 4):
        forward(1)
        suppressed()
        assert forward._primal.autodiff_mode == AutodiffMode.FORWARD
        np.testing.assert_array_equal(y.dual.to_numpy(), 1)
        forward(1)
    np.testing.assert_array_equal(y.to_numpy(), 8)
    np.testing.assert_array_equal(y.dual.to_numpy(), 2)
    assert forward._primal.autodiff_mode == AutodiffMode.NONE


@pytest.mark.parametrize("decorator", [ti.ad.grad_replaced, ti.ad.no_grad])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], debug=True, offline_cache=False)
def test_custom_scope_reuses_validation_kernel_as_primal(decorator):
    x = ti.field(ti.f32, shape=(), needs_grad=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)
    scratch = ti.field(ti.f32, shape=(), needs_grad=True)
    x[None] = 2

    @ti.kernel
    def assign(read_existing: ti.i32):
        if read_existing:
            loss[None] += scratch[None]
        scratch[None] = x[None]

    @ti.kernel
    def finish():
        loss[None] += scratch[None]

    @decorator
    def suppressed():
        # Read then overwrite is owned by the custom derivative, so this
        # invocation must skip automatic global-data-access validation.
        assign(1)

    if decorator is ti.ad.grad_replaced:

        @ti.ad.grad_for(suppressed)
        def suppressed_grad():
            pass

    with ti.ad.Tape(loss, validation=True):
        assign(0)
        suppressed()
        assert assign._primal.autodiff_mode == AutodiffMode.VALIDATION
        finish()
    assert loss[None] == 4
    assert x.grad[None] == 1
    assert assign._primal.autodiff_mode == AutodiffMode.NONE
