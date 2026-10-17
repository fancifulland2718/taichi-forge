import pytest

import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("tier", ["fast", "balanced", "full"])
@pytest.mark.parametrize("kind", ["scalar", "vector", "struct", "tuple"])
@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_real_func_return_values_survive_following_inline_loop(tier, kind):
    vec = ti.types.vector(2, ti.i32)
    record = ti.types.struct(a=ti.i32, b=vec)

    @ti.real_func
    def scalar(x: ti.i32) -> ti.i32:
        return x + 1

    @ti.real_func
    def vector(x: ti.i32) -> vec:
        return vec(x + 1, x + 2)

    @ti.real_func
    def structure(x: ti.i32) -> record:
        return record(a=x + 1, b=vec(x + 2, x + 3))

    @ti.real_func
    def multiple(x: ti.i32) -> (ti.i32, ti.i32):
        return x + 1, x + 2

    @ti.func
    def loop(x: ti.i32):
        for i in range(3):
            x += i
        return x

    @ti.func
    def combine(pair, value):
        return pair[0] + pair[1] + value

    @ti.kernel(opt_level=tier)
    def run(x: ti.i32) -> ti.i32:
        result = 0
        if ti.static(kind == "scalar"):
            result = scalar(x) + loop(x)
        elif ti.static(kind == "vector"):
            result = vector(x)[1] + loop(x)
        elif ti.static(kind == "struct"):
            result = structure(x).a + loop(x)
        else:
            result = combine(multiple(x), loop(x))
        return result

    for x in [-3, 5]:
        expected = {"scalar": 2 * x + 4, "vector": 2 * x + 5, "struct": 2 * x + 4, "tuple": 3 * x + 6}
        assert run(x) == expected[kind]
