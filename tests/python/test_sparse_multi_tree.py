import pytest

import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_pointer():
    e = ti.Vector.field(2, dtype=int, shape=16)

    e[0] = ti.Vector([0, 0])

    a = ti.field(float, shape=512)
    b = ti.field(dtype=float)
    ti.root.pointer(ti.i, 32).dense(ti.i, 16).place(b)

    @ti.kernel
    def test():
        for i in a:
            a[i] = i
        for i in a:
            b[i] += a[i]

    test()
    ti.sync()

    b_np = b.to_numpy()
    for i in range(512):
        assert b_np[i] == i


@test_utils.test(
    arch=ti.cuda,
    offline_cache=False,
    cuda_sparse_pool_auto_size=True,
    cuda_sparse_per_snode_pool=True,
    cuda_pointer_deterministic_slot=True,
    cuda_listgen_reuse=True,
)
def test_cuda_independent_sparse_trees_keep_existing_pool_alive():
    n = 32
    block = 8
    x = ti.field(ti.f32)
    ptr_x = ti.root.pointer(ti.ijk, n // block)
    ptr_x.bitmasked(ti.ijk, block).place(x)

    @ti.kernel
    def fill_x(v: ti.f32):
        for i, j, k in ti.ndrange(n, n, n):
            if (i * 13 + j * 7 + k * 3) % 29 == 0:
                x[i, j, k] = v

    @ti.kernel
    def sum_x() -> ti.f32:
        s = 0.0
        for I in ti.grouped(x):
            s += x[I]
        return s

    fill_x(1.0)
    before = sum_x()
    ti.sync()

    y = ti.field(ti.f32)
    ptr_y = ti.root.pointer(ti.ijk, n // block)
    ptr_y.bitmasked(ti.ijk, block).place(y)

    @ti.kernel
    def fill_y(v: ti.f32):
        for i, j, k in ti.ndrange(n, n, n):
            if (i * 5 + j * 11 + k * 17) % 31 == 0:
                y[i, j, k] = v

    @ti.kernel
    def sum_y() -> ti.f32:
        s = 0.0
        for I in ti.grouped(y):
            s += y[I]
        return s

    fill_y(2.0)
    second = sum_y()
    ti.sync()

    fill_x(3.0)
    after = sum_x()
    ti.sync()

    assert before > 0.0
    assert second > 0.0
    assert abs(after - before * 3.0) < 1e-3


@test_utils.test(
    arch=ti.cuda,
    offline_cache=False,
    cuda_sparse_pool_auto_size=True,
    cuda_sparse_per_snode_pool=True,
)
def test_cuda_nested_pointer_auto_pool_uses_global_cell_bound():
    n = 128
    x = ti.field(ti.i32)
    ti.root.pointer(ti.i, n).pointer(ti.j, n).place(x)

    @ti.kernel
    def activate_all():
        for i, j in ti.ndrange(n, n):
            x[i, j] = i * n + j + 1

    @ti.kernel
    def checksum() -> ti.i32:
        total = 0
        for i, j in x:
            total += x[i, j]
        return total

    activate_all()
    ti.sync()

    count = n * n
    assert checksum() == count * (count + 1) // 2
