import numpy as np

import taichi_forge as ti
from tests import test_utils


def _extent(kernel, *args):
    return [t.constant_range_size for t in kernel.task_manifest(*args) if t.task_type == "range_for"]


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_independent_time_planes_parallelize_before_autodiff():
    rows, cols = 64, 80
    field = ti.field(ti.f32, shape=(2, rows, cols), needs_grad=True)

    @ti.kernel
    def advance(t: ti.i32):
        for i in range(rows):
            for j in range(cols):
                field[t, i, j] = 2 * field[t - 1, i, (j + 1) % cols]

    assert _extent(advance, 1) == [rows * cols]
    assert _extent(advance.grad, 1) == [rows * cols]
    values = np.arange(2 * rows * cols, dtype=np.float32).reshape(2, rows, cols)
    field.from_numpy(values)
    advance(1)
    np.testing.assert_array_equal(field.to_numpy()[1], 2 * np.roll(values[0], -1, axis=1))
    seeds = np.zeros_like(values)
    seeds[1] = np.arange(cols)
    field.grad.from_numpy(seeds)
    advance.grad(1)
    np.testing.assert_array_equal(field.grad.to_numpy()[0], 2 * np.roll(seeds[1], 1, axis=1))


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_inner_recurrence_unknown_planes_and_launch_policy_keep_row_order():
    n = 64
    field = ti.field(ti.i32, shape=(2, n, n))

    @ti.kernel
    def advance(dst: ti.i32, src: ti.i32):
        for i in range(n):
            for j in range(n):
                if j > 0:
                    field[dst, i, j] = field[src, i, j - 1] + 1

    @ti.kernel
    def prefix():
        for i in range(n):
            total = 0
            for j in range(n):
                total += j
                field[0, i, j] = total

    @ti.kernel
    def with_block():
        ti.loop_config(block_dim=32)
        for i in range(n):
            for j in range(n):
                field[1, i, j] = i + j

    @ti.kernel
    def transposed():
        for i in range(n):
            for j in range(n):
                field[0, j, i] = 3

    assert _extent(advance, 0, 0) == [n]
    assert _extent(prefix) == [n]
    assert _extent(with_block) == [n]
    assert _extent(transposed) == [n]
    advance(0, 0)
    np.testing.assert_array_equal(field.to_numpy()[0], np.tile(np.arange(n), (n, 1)))
    prefix()
    np.testing.assert_array_equal(field.to_numpy()[0], np.tile(np.arange(n).cumsum(), (n, 1)))
