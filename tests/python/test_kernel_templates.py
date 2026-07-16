import gc
import weakref

import pytest
import taichi_forge as ti
from tests import test_utils


@test_utils.test()
def test_kernel_template_basic():
    x = ti.field(ti.i32)
    y = ti.field(ti.f32)

    n = 16

    ti.root.dense(ti.i, n).place(x, y)

    @ti.kernel
    def inc(a: ti.template(), b: ti.template()):
        for i in a:
            a[i] += b

    inc(x, 1)
    inc(y, 2)

    for i in range(n):
        assert x[i] == 1
        assert y[i] == 2

    @ti.kernel
    def inc2(z: ti.i32, a: ti.template(), b: ti.i32):
        for i in a:
            a[i] += b + z

    inc2(10, x, 1)
    for i in range(n):
        assert x[i] == 12


@test_utils.test()
def test_kernel_template_gradient():
    x = ti.field(ti.f32)
    y = ti.field(ti.f32)
    z = ti.field(ti.f32)
    loss = ti.field(ti.f32)

    ti.root.dense(ti.i, 16).place(x, y, z)
    ti.root.place(loss)
    ti.root.lazy_grad()

    @ti.kernel
    def double(a: ti.template(), b: ti.template()):
        for i in range(16):
            b[i] = a[i] * 2 + 1

    @ti.kernel
    def compute_loss():
        for i in range(16):
            ti.atomic_add(loss[None], z[i])

    for i in range(16):
        x[i] = i

    with ti.ad.Tape(loss):
        double(x, y)
        double(y, z)
        compute_loss()

    for i in range(16):
        assert z[i] == i * 4 + 3
        assert x.grad[i] == 4


@test_utils.test()
def test_func_template():
    a = [ti.field(dtype=ti.f32) for _ in range(2)]
    b = [ti.field(dtype=ti.f32) for _ in range(2)]

    for l in range(2):
        ti.root.dense(ti.ij, 16).place(a[l], b[l])

    @ti.func
    def sample(x: ti.template(), l: ti.template(), I):
        return x[l][I]

    @ti.kernel
    def fill(l: ti.template()):
        for I in ti.grouped(a[l]):
            a[l][I] = l

    @ti.kernel
    def aTob(l: ti.template()):
        for I in ti.grouped(b[l]):
            b[l][I] = sample(a, l, I)

    for l in range(2):
        fill(l)
        aTob(l)

    for l in range(2):
        for i in range(16):
            for j in range(16):
                assert b[l][i, j] == l


@test_utils.test()
def test_func_template2():
    a = ti.field(dtype=ti.f32)
    b = ti.field(dtype=ti.f32)

    ti.root.dense(ti.ij, 16).place(a, b)

    @ti.func
    def sample(x: ti.template(), I):
        return x[I]

    @ti.kernel
    def fill():
        for I in ti.grouped(a):
            a[I] = 1.0

    @ti.kernel
    def aTob():
        for I in ti.grouped(b):
            b[I] = sample(a, I)

    for l in range(2):
        fill()
        aTob()

    for i in range(16):
        for j in range(16):
            assert b[i, j] == 1.0


@test_utils.test(arch=ti.cpu)
def test_dead_kernel_definitions_leave_runtime_registry():
    runtime = ti.lang.impl.get_runtime()
    baseline = len(runtime.kernels)

    @ti.kernel
    def temporary(value: ti.i32) -> ti.i32:
        return value

    primal_ref = weakref.ref(temporary._primal)
    adjoint_ref = weakref.ref(temporary._adjoint)
    assert len(runtime.kernels) == baseline + 2

    del temporary
    gc.collect()

    assert primal_ref() is None
    assert adjoint_ref() is None
    assert len(runtime.kernels) == baseline


@test_utils.test(arch=ti.cpu, kernel_specialization_limit=2)
def test_kernel_specialization_budget_preserves_cache_hits():
    @ti.kernel
    def identity(value: ti.template()) -> ti.i32:
        return value

    assert identity(11) == 11
    assert identity(22) == 22
    with pytest.raises(ti.TaichiRuntimeError, match="kernel_specialization_limit=2"):
        identity(33)

    # Reaching the budget blocks only a new cache miss. Existing compiled
    # specializations remain valid for Graph and asynchronous launch users.
    assert identity(22) == 22


@test_utils.test(arch=ti.cpu, kernel_specialization_limit=1)
def test_dead_kernel_definition_cannot_reclaim_native_specialization_budget():
    @ti.kernel
    def temporary() -> ti.i32:
        return 1

    assert temporary() == 1
    del temporary
    gc.collect()

    @ti.kernel
    def replacement() -> ti.i32:
        return 2

    with pytest.raises(ti.TaichiRuntimeError, match="kernel_specialization_limit=1"):
        replacement()
