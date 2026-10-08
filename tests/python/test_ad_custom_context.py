import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("outer_kind", ["custom", "no_grad"])
@pytest.mark.parametrize("inner_kind", ["custom", "no_grad"])
@pytest.mark.parametrize("caught_error", [False, True])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_nested_custom_gradients_record_only_outer_call(outer_kind, inner_kind, caught_error):
    x = ti.field(ti.f32, shape=4, needs_grad=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)
    x.fill(2)

    @ti.kernel
    def forward():
        for i in x:
            loss[None] += x[i]

    def inner_body():
        forward()
        if caught_error:
            raise RuntimeError("inner failure")

    inner = (ti.ad.grad_replaced if inner_kind == "custom" else ti.ad.no_grad)(inner_body)
    if inner_kind == "custom":

        @ti.ad.grad_for(inner)
        def inner_grad():
            forward.grad()

    def outer_body():
        if caught_error:
            with pytest.raises(RuntimeError, match="inner failure"):
                inner()
        else:
            inner()
        forward()

    outer = (ti.ad.grad_replaced if outer_kind == "custom" else ti.ad.no_grad)(outer_body)
    if outer_kind == "custom":

        @ti.ad.grad_for(outer)
        def outer_grad():
            # The outer callback owns its complete derivative, even if it
            # explicitly chooses a derivative for an inner no_grad operation.
            forward.grad()
            forward.grad()

    with ti.ad.Tape(loss) as tape:
        outer()
        forward()
        assert len(tape.calls) == 2
    assert loss[None] == 24
    np.testing.assert_array_equal(x.grad.to_numpy(), 3 if outer_kind == "custom" else 1)
    assert not ti.lang.impl.get_runtime().grad_replaced


@pytest.mark.parametrize("decorator", [ti.ad.grad_replaced, ti.ad.no_grad])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_custom_gradient_recording_failure_restores_runtime(decorator, monkeypatch):
    x = ti.field(ti.f32, shape=(), needs_grad=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)
    x[None] = 2
    invoked = []

    @decorator
    def custom():
        invoked.append(True)

    @ti.kernel
    def forward():
        loss[None] += x[None]

    def reject_record(*args):
        raise RuntimeError("recording failed")

    with ti.ad.Tape(loss) as tape:
        with monkeypatch.context() as patch:
            patch.setattr(tape, "insert", reject_record)
            with pytest.raises(RuntimeError, match="recording failed"):
                custom()
        assert not invoked
        assert not ti.lang.impl.get_runtime().grad_replaced
        forward()
    assert x.grad[None] == 1
