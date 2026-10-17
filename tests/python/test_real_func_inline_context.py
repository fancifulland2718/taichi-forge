import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("tier", ["fast", "balanced", "full"])
@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_real_func_inline_control_flow_stays_in_current_callable(tier):
    @ti.func
    def helper(x: ti.i32, n: ti.i32):
        result = x
        for i in range(n):
            result += i
        if x > 0:
            result *= 2
        else:
            result -= 3
        i = 0
        while i < 2:
            result += 1
            i += 1
        return result

    @ti.func
    def nested(x: ti.i32, n: ti.i32):
        return helper(x, n) + 1

    @ti.real_func
    def real(x: ti.i32, n: ti.i32) -> ti.i32:
        result = nested(x, n)
        for _ in range(2):
            result += x
        return result

    @ti.kernel(opt_level=tier)
    def run(x: ti.i32, n: ti.i32) -> ti.i32:
        result = real(x, n)
        return result + helper(x - 1, n)

    def reference(x, n):
        result = x + sum(range(n))
        return (result * 2 if x > 0 else result - 3) + 2

    for x in [-2, 0, 3]:
        for n in [0, 3]:
            assert run(x, n) == reference(x, n) + 1 + 2 * x + reference(x - 1, n)


@pytest.mark.parametrize("tier", ["fast", "balanced", "full"])
@pytest.mark.parametrize("calls", [1, 32])
@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_real_func_inline_tensor_calls_preserve_values_and_returns(tier, calls):
    vec = ti.types.vector(2, ti.f32)
    mat = ti.types.matrix(2, 2, ti.f32)

    @ti.func
    def transform(a: mat, v: vec) -> vec:
        v[1] += 1
        for i, j in ti.ndrange(2, 2):
            a[i, j] += i + j
        result = vec(0)
        for i, j in ti.ndrange(2, 2):
            result[i] += a[i, j] * v[j]
        return result

    @ti.func
    def nested(a: mat, v: vec) -> vec:
        return transform(a, v) + transform(a, v)

    @ti.real_func
    def real(a: mat, v: vec) -> vec:
        result = nested(a, v)
        return result + vec(a[0, 1], v[1])

    output = vec.field(shape=4)

    @ti.kernel(opt_level=tier)
    def run():
        for i in range(4):
            result = vec(0)
            for j in ti.static(range(calls)):
                result += real(mat(1, 2, 3, 4), vec(i + j, i + j + 1))
            output[i] = result

    run()
    expected = [
        [calls * (8 * i + 14) + 8 * sum(range(calls)), calls * (21 * i + 25) + 21 * sum(range(calls))]
        for i in range(4)
    ]
    np.testing.assert_array_equal(output.to_numpy(), expected)
    assert len(real.func.taichi_functions) == 1


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False, inline_ir_cache=True)
def test_real_func_inline_context_survives_tier_changes_and_reset():
    arch = ti.lang.impl.current_cfg().arch

    @ti.func
    def helper(x: ti.i32):
        for i in range(3):
            x += i
        if x < 0:
            x = -x
        return x

    @ti.real_func
    def real(x: ti.i32) -> ti.i32:
        return helper(x)

    @ti.kernel(opt_level="fast")
    def fast(x: ti.i32) -> ti.i32:
        result = real(x)
        return result + helper(x + 1)

    @ti.kernel(opt_level="full")
    def full(x: ti.i32) -> ti.i32:
        result = real(x)
        return result + helper(x + 1)

    for _ in range(2):
        for run in [fast, full, fast]:
            for x in [-10, 5]:
                assert run(x) == abs(x + 3) + abs(x + 4)
        assert len(real.func.taichi_functions) == 2
        ti.reset()
        ti.init(arch=arch, enable_fallback=False, offline_cache=False, inline_ir_cache=True)
