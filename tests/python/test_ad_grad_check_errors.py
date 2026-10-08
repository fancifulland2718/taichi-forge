import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils


@pytest.mark.parametrize("failure_phase", ["positive", "negative", "comparison"])
@test_utils.test(arch=[ti.cpu, ti.cuda], require=ti.extension.data64, offline_cache=False)
def test_grad_check_restores_fields_after_failure(failure_phase):
    x = ti.field(ti.f64, shape=(), needs_grad=True)
    loss = ti.field(ti.f64, shape=(), needs_grad=True)
    marker = ti.field(ti.i32, shape=())
    x[None] = 0.05
    marker[None] = 10
    calls = []
    failure = ValueError("input is outside the custom operation's domain")

    @ti.kernel
    def forward():
        loss[None] += x[None] * x[None]

    @ti.ad.grad_replaced
    def evaluate():
        calls.append(x[None])
        marker[None] += 1
        forward()
        if (failure_phase == "positive" and x[None] > 0.1) or (failure_phase == "negative" and x[None] < 0):
            raise failure

    @ti.ad.grad_for(evaluate)
    def backward():
        forward.grad()
        if failure_phase == "comparison":
            x.grad[None] *= 2  # Deliberately incorrect custom derivative.

    tape = ti.ad.Tape(loss, grad_check=[x])
    tape.grad_checker.eps_range = [0.125, 0.03125]
    error_type = AssertionError if failure_phase == "comparison" else ValueError
    original_rand = np.random.rand
    np.random.rand = lambda *shape: np.ones(shape)
    try:
        with pytest.raises(error_type) as caught:
            with tape:
                evaluate()
    finally:
        np.random.rand = original_rand
    if failure_phase == "comparison":
        assert "Grad check failed" in str(caught.value)
        assert len(calls) == 5
    else:
        assert caught.value is failure
        assert len(calls) == (2 if failure_phase == "positive" else 3)
    assert x[None] == 0.05
    assert loss[None] == pytest.approx(0.0025)
    assert marker[None] == 11
    assert x.grad[None] == pytest.approx(0.2 if failure_phase == "comparison" else 0.1)
    assert impl.get_runtime().target_tape is None
    assert not impl.get_runtime().grad_replaced

    # A subsequent Tape still executes normally and does not replay the failed one.
    x[None] = 0.2
    with ti.ad.Tape(loss):
        forward()
    assert loss[None] == pytest.approx(0.04)
    assert x.grad[None] == pytest.approx(0.4)
