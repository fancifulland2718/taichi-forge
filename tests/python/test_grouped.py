import numpy as np

import taichi_forge as ti
from tests import test_utils


@test_utils.test()
def test_vector_index():
    val = ti.field(ti.i32)

    n = 4
    m = 7
    p = 11

    ti.root.dense(ti.i, n).dense(ti.j, m).dense(ti.k, p).place(val)

    @ti.kernel
    def test():
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    I = ti.Vector([i, j, k])
                    val[I] = i + j * 2 + k * 3

    test()

    for i in range(n):
        for j in range(m):
            for k in range(p):
                assert val[i, j, k] == i + j * 2 + k * 3


@test_utils.test()
def test_grouped():
    val = ti.field(ti.i32)

    n = 4
    m = 8
    p = 16

    ti.root.dense(ti.i, n).dense(ti.j, m).dense(ti.k, p).place(val)

    @ti.kernel
    def test():
        for I in ti.grouped(val):
            val[I] = I[0] + I[1] * 2 + I[2] * 3

    test()

    for i in range(n):
        for j in range(m):
            for k in range(p):
                assert val[i, j, k] == i + j * 2 + k * 3


@test_utils.test()
def test_grouped_atomic_addresses_and_return_values():
    rows, cols = 17, 19
    cells = ti.field(ti.i32, shape=(rows, cols))
    tickets = ti.Vector.field(2, ti.i32, shape=(rows, cols))
    buckets = ti.field(ti.i32, shape=(rows, 1))

    @ti.kernel
    def update(shared_col: ti.i32):
        for I in ti.grouped(cells):
            previous = ti.atomic_add(cells[I], 3)
            row, col = I[0], shared_col
            ticket = ti.atomic_add(buckets[ti.Vector([row, col])], 1)
            tickets[I] += ti.Vector([previous, ticket])

    cells.fill(7)
    update(0)
    values = tickets.to_numpy()
    assert np.all(cells.to_numpy() == 10)
    assert np.all(values[:, :, 0] == 7)
    assert np.all(buckets.to_numpy() == cols)
    assert np.all(np.sort(values[:, :, 1], axis=1) == np.arange(cols))


@test_utils.test()
def test_vector_field_index_captures_mutable_coordinates():
    cells = ti.field(ti.i32, shape=(3, 4))

    @ti.kernel
    def update():
        row, col = 1, 2
        ti.atomic_add(cells[ti.Vector([row, col])], ti.atomic_add(row, 1))
        cells[row, col] += 4

    update()
    expected = np.zeros((3, 4), dtype=np.int32)
    expected[1, 2], expected[2, 2] = 1, 4
    np.testing.assert_array_equal(cells.to_numpy(), expected)


@test_utils.test()
def test_grouped_ndrange():
    val = ti.field(ti.i32)

    n = 4
    m = 8

    ti.root.dense(ti.ij, (n, m)).place(val)

    x0 = 2
    y0 = 3
    x1 = 1
    y1 = 6

    @ti.kernel
    def test():
        for I in ti.grouped(ti.ndrange((x0, y0), (x1, y1))):
            val[I] = I[0] + I[1] * 2

    test()

    for i in range(n):
        for j in range(m):
            assert val[i, j] == (i + j * 2 if x0 <= i < y0 and x1 <= j < y1 else 0)


@test_utils.test()
def test_static_grouped_ndrange():
    val = ti.field(ti.i32)

    n = 4
    m = 8

    ti.root.dense(ti.ij, (n, m)).place(val)

    x0 = 2
    y0 = 3
    x1 = 1
    y1 = 6

    @ti.kernel
    def test():
        for I in ti.static(ti.grouped(ti.ndrange((x0, y0), (x1, y1)))):
            val[I] = I[0] + I[1] * 2

    test()

    for i in range(n):
        for j in range(m):
            assert val[i, j] == (i + j * 2 if x0 <= i < y0 and x1 <= j < y1 else 0)


@test_utils.test()
def test_grouped_ndrange_starred():
    val = ti.field(ti.i32)

    n = 4
    m = 8
    p = 16
    dim = 3

    ti.root.dense(ti.ijk, (n, m, p)).place(val)

    @ti.kernel
    def test():
        for I in ti.grouped(ti.ndrange(*(((0, n),) * dim))):
            val[I] = I[0] + I[1] * 2 + I[2] * 3

    test()

    for i in range(n):
        for j in range(m):
            for k in range(p):
                assert val[i, j, k] == (i + j * 2 + k * 3 if j < n and k < n else 0)


@test_utils.test()
def test_grouped_ndrange_0d():
    val = ti.field(ti.i32, shape=())

    @ti.kernel
    def test():
        for I in ti.grouped(ti.ndrange()):
            val[I] = 42

    test()

    assert val[None] == 42


@test_utils.test()
def test_static_grouped_ndrange_0d():
    val = ti.field(ti.i32, shape=())

    @ti.kernel
    def test():
        for I in ti.static(ti.grouped(ti.ndrange())):
            val[I] = 42

    test()

    assert val[None] == 42


@test_utils.test()
def test_static_grouped_func():
    K = 3
    dim = 2

    v = ti.Vector.field(K, dtype=ti.i32, shape=((K,) * dim))

    def stencil_range():
        return ti.ndrange(*((K,) * (dim + 1)))

    @ti.kernel
    def p2g():
        for I in ti.static(ti.grouped(stencil_range())):
            v[I[0], I[1]][I[2]] = I[0] + I[1] * 3 + I[2] * 10

    p2g()

    for i in range(K):
        for j in range(K):
            for k in range(K):
                assert v[i, j][k] == i + j * 3 + k * 10
