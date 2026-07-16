import pytest

import taichi_forge as ti
from tests import test_utils


@test_utils.test(offline_cache=False)
def test_first_order_reverse_and_forward_match_finite_difference():
    x = ti.field(ti.f32, shape=(), needs_grad=True, needs_dual=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True, needs_dual=True)

    @ti.kernel
    def evaluate():
        loss[None] = x[None] * x[None] + 3.0 * x[None]

    def primal(value):
        x[None] = value
        evaluate()
        return loss[None]

    center = 1.25
    epsilon = 1.0e-2
    finite_difference = (primal(center + epsilon) - primal(center - epsilon)) / (
        2.0 * epsilon
    )

    x[None] = center
    with ti.ad.Tape(loss=loss):
        evaluate()
    reverse = x.grad[None]

    x[None] = center
    with ti.ad.FwdMode(loss=loss, param=x, seed=[1.0]):
        evaluate()
    forward = loss.dual[None]

    assert reverse == pytest.approx(finite_difference, abs=2.0e-3)
    assert forward == pytest.approx(finite_difference, abs=2.0e-3)


@test_utils.test(offline_cache=False)
def test_forward_on_reverse_fails_before_launch():
    x = ti.field(ti.f32, shape=(), needs_grad=True, needs_dual=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True, needs_dual=True)

    @ti.kernel
    def square():
        loss[None] = x[None] * x[None]

    with ti.ad.FwdMode(loss=loss, param=x, seed=[1.0]):
        with pytest.raises(
            ti.TaichiRuntimeError,
            match="Forward-on-reverse automatic differentiation is not supported",
        ):
            square.grad()

    assert ti.lang.impl.get_runtime().fwd_mode_manager is None


@test_utils.test(offline_cache=False)
def test_manual_reverse_inside_tape_fails_without_running_adjoints():
    x = ti.field(ti.f32, shape=(), needs_grad=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)

    @ti.kernel
    def square():
        loss[None] = x[None] * x[None]

    x[None] = 3.0
    with pytest.raises(
        ti.TaichiRuntimeError,
        match="Manual reverse kernel execution inside ti.ad.Tape",
    ):
        with ti.ad.Tape(loss=loss):
            square()
            square.grad()

    assert x.grad[None] == 0.0
    assert ti.lang.impl.get_runtime().target_tape is None


@test_utils.test(offline_cache=False)
def test_nested_automatic_ad_contexts_fail_fast():
    x = ti.field(ti.f32, shape=(), needs_grad=True, needs_dual=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True, needs_dual=True)

    with ti.ad.FwdMode(loss=loss, param=x, seed=[1.0]):
        with pytest.raises(
            ti.TaichiRuntimeError,
            match="another automatic AD context",
        ):
            with ti.ad.Tape(loss=loss):
                pass

    with ti.ad.Tape(loss=loss):
        with pytest.raises(
            ti.TaichiRuntimeError,
            match="another automatic AD context",
        ):
            with ti.ad.FwdMode(loss=loss, param=x, seed=[1.0]):
                pass

    runtime = ti.lang.impl.get_runtime()
    assert runtime.target_tape is None
    assert runtime.fwd_mode_manager is None


@test_utils.test(offline_cache=False)
def test_dynamic_early_return_is_rejected_in_ad_context():
    x = ti.field(ti.f32, shape=(), needs_grad=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)

    @ti.kernel
    def conditional(flag: ti.i32):
        if flag != 0:
            return
        loss[None] = x[None]

    with pytest.raises(
        ti.TaichiCompilationError,
        match="Return inside non-static if/for is not supported",
    ):
        with ti.ad.Tape(loss=loss):
            conditional(1)

    assert ti.lang.impl.get_runtime().target_tape is None
