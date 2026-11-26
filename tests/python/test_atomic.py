import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils

n = 128


def run_atomic_add_global_case(vartype, step, valproc=lambda x: x):
    x = ti.field(vartype)
    y = ti.field(vartype)
    c = ti.field(vartype)

    ti.root.dense(ti.i, n).place(x, y)
    ti.root.place(c)

    # Make Taichi correctly infer the type
    # TODO: Taichi seems to treat numpy.int32 as a float type, fix that.
    init_ck = 0 if vartype == ti.i32 else 0.0

    @ti.kernel
    def func():
        ck = init_ck
        for i in range(n):
            x[i] = ti.atomic_add(c[None], step)
            y[i] = ti.atomic_add(ck, step)

    func()

    assert valproc(c[None]) == n * step
    x_actual = sorted(x.to_numpy())
    y_actual = sorted(y.to_numpy())
    expect = [i * step for i in range(n)]
    for xa, ya, e in zip(x_actual, y_actual, expect):
        print(xa, ya, e)
        assert valproc(xa) == e
        assert valproc(ya) == e


@test_utils.test()
def test_atomic_add_global_i32():
    run_atomic_add_global_case(ti.i32, 42)


@test_utils.test()
def test_atomic_add_global_f32():
    run_atomic_add_global_case(ti.f32, 4.2, valproc=lambda x: test_utils.approx(x, rel=1e-5))


@test_utils.test(arch=[ti.cpu, ti.cuda], require=ti.extension.data64)
def test_atomic_add_global_f64():
    # A binary-exact increment isolates f64 atomic lowering from rounding-order
    # differences between serial CPU and parallel CUDA execution.
    run_atomic_add_global_case(
        ti.f64,
        0.25,
        valproc=lambda x: test_utils.approx(x, rel=0.0, abs=1e-12),
    )


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_atomic_min_max_uint():
    x = ti.field(ti.u64, shape=100)

    @ti.kernel
    def test0():
        for I in x:
            x[I] = 0
        x[1] = ti.cast(1, ti.u64) << 63
        for I in x:
            ti.atomic_max(x[0], x[I])

    test0()
    assert x[0] == 9223372036854775808

    @ti.kernel
    def test1():
        for I in x:
            x[I] = ti.cast(1, ti.u64) << 63
        x[1] = 100
        for I in x:
            ti.atomic_min(x[0], x[I])

    test1()
    assert x[0] == 100


@test_utils.test()
def test_atomic_add_expr_evaled():
    c = ti.field(ti.i32)
    step = 42

    ti.root.place(c)

    @ti.kernel
    def func():
        for i in range(n):
            # this is an expr with side effect, make sure it's not optimized out.
            ti.atomic_add(c[None], step)

    func()

    assert c[None] == n * step


@test_utils.test(arch=[ti.cuda, ti.vulkan])
def test_grouped_field_atomic_add_preserves_rounding():
    shape = (17, 19)
    scalar = ti.field(ti.f32, shape=shape)
    vector = ti.Vector.field(2, ti.f32, shape=shape)
    terms = ti.field(ti.f32, shape=shape)
    reference = ti.field(ti.f32, shape=shape[0])

    @ti.kernel
    def update(scale: ti.f32):
        for I in ti.grouped(scalar):
            term = terms[I] * scale
            scalar[I] += term
            vector[I] += ti.Vector([term, term])

    @ti.kernel
    def update_reference(scale: ti.f32):
        for I in ti.grouped(terms):
            if I[1] == 0:
                reference[I[0]] += terms[I] * scale

    # The squared value rounds before addition. Fusion would produce 1.
    for initial, value, scale in ((-16785408, 4097, 4097), (0, 2**-127, 1)):
        scalar.fill(initial)
        vector.fill(initial)
        reference.fill(initial)
        terms.from_numpy(np.full(shape, value, dtype=np.float32))
        update(scale)
        update_reference(scale)
        expected = reference.to_numpy().view(np.uint32)
        assert np.all(scalar.to_numpy().view(np.uint32) == expected[:, None])
        assert np.all(vector.to_numpy().view(np.uint32) == expected[:, None, None])


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_grouped_field_atomics_special_values():
    bits = np.array([0, 0x80000000, 0x7FC12345, 1, 0x3F800000], dtype=np.uint32)
    n = len(bits)
    source = np.repeat(bits, n).reshape(n, n).view(np.float32)
    target = ti.field(ti.f32, shape=(n, n))
    reference = ti.field(ti.f32, shape=(n, n))
    terms = ti.field(ti.f32, shape=(n, n))
    terms.from_numpy(np.tile(bits, n).reshape(n, n).view(np.float32))

    @ti.func
    def add_negative(dst: ti.template(), x):
        return ti.atomic_add(dst, -x)

    @ti.kernel
    def update(op: ti.template(), planes: ti.i32):
        for I in ti.grouped(target):
            op(target[I], terms[I])
        for i, j, k in ti.ndrange(n, n, planes):
            if k == 0:
                op(reference[i, j], terms[i, j])

    for op in (ti.atomic_add, add_negative, ti.atomic_sub, ti.atomic_min, ti.atomic_max):
        target.from_numpy(source)
        reference.from_numpy(source)
        update(op, 2)
        np.testing.assert_array_equal(target.to_numpy().view(np.uint32), reference.to_numpy().view(np.uint32))


@test_utils.test()
def test_atomic_add_demoted():
    # Ensure demoted atomics do not crash the program.
    x = ti.field(ti.i32)
    y = ti.field(ti.i32)
    step = 42

    ti.root.dense(ti.i, n).place(x, y)

    @ti.kernel
    def func():
        for i in range(n):
            s = i
            # Both adds should get demoted.
            x[i] = ti.atomic_add(s, step)
            y[i] = ti.atomic_add(s, step)

    func()

    for i in range(n):
        assert x[i] == i
        assert y[i] == i + step


@test_utils.test()
def test_atomic_add_with_local_store_simplify1():
    # Test for the following LocalStoreStmt simplification case:
    #
    # local store [$a <- ...]
    # atomic add ($a, ...)
    # local store [$a <- ...]
    #
    # Specifically, the second store should not suppress the first one, because
    # atomic_add can return value.
    x = ti.field(ti.i32)
    y = ti.field(ti.i32)
    step = 42

    ti.root.dense(ti.i, n).place(x, y)

    @ti.kernel
    def func():
        for i in range(n):
            # do a local store
            j = i
            x[i] = ti.atomic_add(j, step)
            # do another local store, make sure the previous one is not optimized out
            j = x[i]
            y[i] = j

    func()

    for i in range(n):
        assert x[i] == i
        assert y[i] == i


@test_utils.test()
def test_atomic_add_with_local_store_simplify2():
    # Test for the following LocalStoreStmt simplification case:
    #
    # local store [$a <- ...]
    # atomic add ($a, ...)
    #
    # Specifically, the local store should not be removed, because
    # atomic_add can return its value.
    x = ti.field(ti.i32)
    step = 42

    ti.root.dense(ti.i, n).place(x)

    @ti.kernel
    def func():
        for i in range(n):
            j = i
            x[i] = ti.atomic_add(j, step)

    func()

    for i in range(n):
        assert x[i] == i


@test_utils.test()
def test_atomic_add_with_if_simplify():
    # Make sure IfStmt simplification doesn't move stmts depending on the result
    # of atomic_add()
    x = ti.field(ti.i32)
    step = 42

    ti.root.dense(ti.i, n).place(x)

    boundary = n / 2

    @ti.kernel
    def func():
        for i in range(n):
            if i > boundary:
                # A sequence of commands designed such that atomic_add() is the only
                # thing to decide whether the if branch can be simplified.
                s = i
                j = ti.atomic_add(s, s)
                k = j + s
                x[i] = k
            else:
                # If we look at the IR, this branch should be simplified, since nobody
                # is using atomic_add's result.
                ti.atomic_add(x[i], i)
                x[i] += step

    func()

    for i in range(n):
        expect = i * 3 if i > boundary else (i + step)
        assert x[i] == expect


@test_utils.test()
def test_local_atomic_with_if():
    ret = ti.field(dtype=ti.i32, shape=())

    @ti.kernel
    def test():
        if True:
            x = 0
            x += 1
            ret[None] = x

    test()
    assert ret[None] == 1


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_atomic_sub_with_type_promotion():
    # Test Case 1
    @ti.kernel
    def test_u16_sub_u8() -> ti.uint16:
        x: ti.uint16 = 1000
        y: ti.uint8 = 255

        ti.atomic_sub(x, y)
        return x

    res = test_u16_sub_u8()
    assert res == 745

    # Test Case 2
    @ti.kernel
    def test_u8_sub_u16() -> ti.uint8:
        x: ti.uint8 = 255
        y: ti.uint16 = 100

        ti.atomic_sub(x, y)
        return x

    res = test_u8_sub_u16()
    assert res == 155

    # Test Case 3
    A = ti.field(ti.uint8, shape=())
    B = ti.field(ti.uint16, shape=())

    @ti.kernel
    def test_with_field():
        v: ti.uint16 = 1000
        v -= A[None]
        B[None] = v

    A[None] = 255
    test_with_field()
    assert B[None] == 745


@test_utils.test()
def test_atomic_sub_expr_evaled():
    c = ti.field(ti.i32)
    step = 42

    ti.root.place(c)

    @ti.kernel
    def func():
        for i in range(n):
            # this is an expr with side effect, make sure it's not optimized out.
            ti.atomic_sub(c[None], step)

    func()

    assert c[None] == -n * step


@test_utils.test()
def test_atomic_mul_expr_evaled():
    c = ti.field(ti.i32)
    base = 2

    ti.root.place(c)

    @ti.kernel
    def func():
        c[None] = 1
        for i in range(16):
            # this is an expr with side effect, make sure it's not optimized out.
            ti.atomic_mul(c[None], base)

    func()

    assert c[None] == base**16


@test_utils.test()
def test_atomic_max_expr_evaled():
    c = ti.field(ti.i32)
    step = 42

    ti.root.place(c)

    @ti.kernel
    def func():
        for i in range(n):
            # this is an expr with side effect, make sure it's not optimized out.
            ti.atomic_max(c[None], i * step)

    func()

    assert c[None] == (n - 1) * step


@test_utils.test()
def test_atomic_min_expr_evaled():
    c = ti.field(ti.i32)
    step = 42

    ti.root.place(c)

    @ti.kernel
    def func():
        c[None] = 1000
        for i in range(n):
            # this is an expr with side effect, make sure it's not optimized out.
            ti.atomic_min(c[None], i * step)

    func()

    assert c[None] == 0


@test_utils.test()
def test_atomic_and_expr_evaled():
    c = ti.field(ti.i32)
    step = 42

    ti.root.place(c)

    max_int = 2147483647

    @ti.kernel
    def func():
        c[None] = 1023
        for i in range(10):
            # this is an expr with side effect, make sure it's not optimized out.
            ti.atomic_and(c[None], max_int - 2**i)

    func()

    assert c[None] == 0


@test_utils.test()
def test_atomic_or_expr_evaled():
    c = ti.field(ti.i32)
    step = 42

    ti.root.place(c)

    @ti.kernel
    def func():
        c[None] = 0
        for i in range(10):
            # this is an expr with side effect, make sure it's not optimized out.
            ti.atomic_or(c[None], 2**i)

    func()

    assert c[None] == 1023


@test_utils.test()
def test_atomic_xor_expr_evaled():
    c = ti.field(ti.i32)
    step = 42

    ti.root.place(c)

    @ti.kernel
    def func():
        c[None] = 1023
        for i in range(10):
            # this is an expr with side effect, make sure it's not optimized out.
            ti.atomic_xor(c[None], 2**i)

    func()

    assert c[None] == 0


@test_utils.test()
def test_atomic_min_rvalue_as_frist_op():
    @ti.kernel
    def func():
        y = ti.Vector([1, 2, 3])
        z = ti.atomic_min([3, 2, 1], y)

    with pytest.raises(ti.TaichiSyntaxError) as e:
        func()

    assert "atomic_min" in str(e.value)
    assert "cannot use a non-writable target as the first operand of" in str(e.value)


@test_utils.test()
def test_atomic_max_f32():
    @ti.kernel
    def max_kernel() -> ti.f32:
        x = -1000.0
        for i in range(1, 20):
            ti.atomic_max(x, -ti.f32(i))

        return x

    assert max_kernel() == -1.0


@test_utils.test()
def test_atomic_mul_f32():
    @ti.kernel
    def mul_kernel() -> ti.f32:
        x = 1.0
        for i in range(1, 8):
            ti.atomic_mul(x, ti.f32(i))

        return x

    assert mul_kernel() == 5040.0
