import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_nested_branch_values_and_side_effects():
    calls = ti.field(ti.i32, shape=2)

    @ti.func
    def choose(flag, inner):
        if flag:
            if inner:
                value = 7
            else:
                value = 9
            calls[0] += 1
        else:
            value = -3
            calls[1] += 1
        return value

    @ti.kernel
    def run(flag: ti.i32, inner: ti.i32) -> ti.i32:
        return choose(flag, inner)

    assert [run(0, 0), run(1, 0), run(1, 1)] == [-3, 9, 7]
    assert calls.to_numpy().tolist() == [2, 1]


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_branch_aggregate_values():
    pair = ti.types.struct(value=ti.i32, vector=ti.types.vector(2, ti.f32))

    @ti.kernel
    def run(flag: ti.i32) -> ti.f32:
        if flag:
            a, b = 2, 3
            vector = ti.Vector([1.0, 2.0])
            matrix = ti.Matrix([[1, 2], [3, 4]])
            record = pair(value=7, vector=ti.Vector([4.0, 5.0]))
        else:
            a, b = 5, 6
            vector = ti.Vector([3.0, 4.0])
            matrix = ti.Matrix([[5, 6], [7, 8]])
            record = pair(value=8, vector=ti.Vector([6.0, 7.0]))
        vector[0] += 1.0
        record.value += a
        return a + b + vector.sum() + matrix.sum() + record.value + record.vector.sum()

    assert run(1) == 37.0
    assert run(0) == 71.0


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_branch_merge_inside_loop_and_existing_variable():
    @ti.kernel
    def run(limit: ti.i32) -> ti.i32:
        total = 0
        ti.loop_config(serialize=True)
        for i in range(limit):
            if i % 2:
                if i == 3:
                    continue
                value = i * 2
            else:
                value = -i
            total += value
        if limit > 0:
            total += 1
        return total

    for limit in (0, 3, 8):
        expected = sum(2 * i if i % 2 else -i for i in range(limit) if i != 3)
        assert run(limit) == expected + int(limit > 0)


@pytest.mark.parametrize('case', ['single', 'nested', 'loop', 'static', 'sibling'])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_branch_merge_requires_both_paths(case):
    @ti.kernel
    def run(flag: ti.i32) -> ti.i32:
        if ti.static(case == 'single'):
            if flag:
                value = 1
            return value
        elif ti.static(case == 'nested'):
            if flag:
                if flag > 1:
                    value = 1
            else:
                value = 2
            return value
        elif ti.static(case == 'loop'):
            for i in range(flag):
                if i % 2:
                    value = 1
                else:
                    value = 2
            return value
        elif ti.static(case == 'static'):
            if ti.static(True):
                value = 1
            return value
        else:
            if flag:
                value = 1
            else:
                value = value + 1
            return value

    with pytest.raises(ti.TaichiNameError, match='Name "value" is not defined'):
        run(1)


@pytest.mark.parametrize('case', ['dtype', 'shape'])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_branch_merge_rejects_different_types(case):
    @ti.kernel
    def run(flag: ti.i32):
        if ti.static(case == 'dtype'):
            if flag:
                value = 1
            else:
                value = 2.0
            print(value)
        else:
            if flag:
                value = ti.Vector([1, 2])
            else:
                value = ti.Vector([3, 4, 5])
            print(value)

    with pytest.raises(ti.TaichiTypeError, match='different types in if/else'):
        run(1)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_static_branch_objects_stay_local():
    @ti.kernel
    def run(flag: ti.i32) -> ti.i32:
        result = 0
        if flag:
            items = ti.static([1, 2])
            result = items[0]
        else:
            items = ti.static([3, 4, 5])
            result = items[0]
        return result

    assert run(1) == 1
    assert run(0) == 3


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_branch_merge_reverse_gradient():
    x = ti.field(ti.f32, shape=4, needs_grad=True)
    y = ti.field(ti.f32, shape=4, needs_grad=True)

    @ti.kernel
    def run():
        for i in x:
            if x[i] > 0:
                value = x[i] * x[i]
            else:
                value = x[i] * x[i] * x[i]
            y[i] = value

    values = np.array([-2.0, -1.0, 0.5, 1.0], dtype=np.float32)
    x.from_numpy(values)
    y.grad.fill(1)
    run()
    run.grad()
    np.testing.assert_allclose(y.to_numpy(), np.where(values > 0, values**2, values**3))
    np.testing.assert_allclose(x.grad.to_numpy(), np.where(values > 0, 2 * values, 3 * values**2))
