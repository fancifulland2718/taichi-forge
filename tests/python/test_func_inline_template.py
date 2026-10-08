import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_repeated_inline_calls_refresh_closure_and_static_callbacks():
    output = ti.field(ti.i32, shape=4)
    calls = []
    gain = 2

    def observe(tag):
        calls.append(tag)
        return gain + tag

    @ti.func
    def helper(value, tag: ti.template()):
        amount = ti.static(observe(tag))
        return value + amount + gain

    @ti.func
    def nested(value, tag: ti.template()):
        return helper(value, tag) + helper(value + 1, tag + 1)

    @ti.kernel
    def evaluate(specialization: ti.template()):
        for i in output:
            value = i
            for j in ti.static(range(4)):
                value = nested(value, specialization + j)
            output[i] = value

    for specialization in (0, 10):
        gain += 1
        start = len(calls)
        evaluate(specialization)
        expected = np.arange(4)
        for j in range(4):
            expected = 2 * expected + 4 * gain + 2 * (specialization + j) + 2
        np.testing.assert_array_equal(output.to_numpy(), expected)
        assert calls[start:] == [specialization + j + offset for j in range(4) for offset in (0, 1)]
        # An already-compiled kernel executes without revisiting Python callbacks.
        evaluate(specialization)
        assert len(calls) == start + 8


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_recursive_inline_calls_have_independent_ast_state():
    output = ti.field(ti.i32, shape=4)

    @ti.func
    def recursive(value, depth: ti.template()):
        if ti.static(depth == 0):
            return value
        else:
            return recursive(value + depth, depth - 1) + recursive(value, depth - 1)

    @ti.kernel
    def evaluate():
        for i in output:
            output[i] = recursive(i, 3) + recursive(i, 1)

    evaluate()
    np.testing.assert_array_equal(output.to_numpy(), 10 * np.arange(4) + 25)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_inline_template_recovers_after_compilation_error():
    output = ti.field(ti.i32, shape=())

    @ti.func
    def helper(value, fail: ti.template()):
        if ti.static(fail):
            return missing_inline_name  # noqa: F821
        else:
            return 3 * value

    @ti.kernel
    def evaluate(fail: ti.template()):
        output[None] = helper(7, fail)

    with pytest.raises(ti.TaichiNameError, match="missing_inline_name") as caught:
        evaluate(True)
    assert "in helper" in str(caught.value)
    evaluate(False)
    assert output[None] == 21


@pytest.mark.parametrize("mode", ["forward", "reverse"])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_inline_template_keeps_derivatives(mode):
    x = ti.field(ti.f32, shape=4, needs_grad=True, needs_dual=True)
    y = ti.field(ti.f32, shape=4, needs_grad=True, needs_dual=True)
    x.fill(0.125)

    @ti.func
    def helper(value, tag: ti.template()):
        return value * value + (tag + 1) * value

    @ti.kernel
    def evaluate():
        for i in range(4):
            y[i] = helper(helper(helper(x[i], 0), 1), 2)

    expected, derivative = 0.125, 1.0
    for tag in range(3):
        derivative *= 2 * expected + tag + 1
        expected = expected * expected + (tag + 1) * expected
    if mode == "forward":
        with ti.ad.FwdMode(loss=y, param=x, seed=[1.0] * 4):
            evaluate()
        np.testing.assert_allclose(y.dual.to_numpy(), derivative, rtol=1e-6)
    else:
        evaluate()
        y.grad.fill(1)
        evaluate.grad()
        np.testing.assert_allclose(x.grad.to_numpy(), derivative, rtol=1e-6)
    np.testing.assert_allclose(y.to_numpy(), expected, rtol=1e-6)
