import taichi_forge as ti
from tests import test_utils


@test_utils.test()
def test_normal_grad():
    x = ti.field(ti.f32)
    loss = ti.field(ti.f32)

    n = 128

    ti.root.dense(ti.i, n).place(x)
    ti.root.place(loss)
    ti.root.lazy_grad()

    @ti.kernel
    def func():
        for i in range(n):
            loss[None] += x[i] ** 2

    for i in range(n):
        x[i] = i

    with ti.ad.Tape(loss):
        func()

    for i in range(n):
        assert x.grad[i] == i * 2


@test_utils.test()
def test_stop_grad():
    x = ti.field(ti.f32)
    loss = ti.field(ti.f32)

    n = 128

    ti.root.dense(ti.i, n).place(x)
    ti.root.place(loss)
    ti.root.lazy_grad()

    @ti.kernel
    def func():
        for i in range(n):
            ti.stop_grad(x)
            loss[None] += x[i] ** 2

    for i in range(n):
        x[i] = i

    with ti.ad.Tape(loss):
        func()

    for i in range(n):
        assert x.grad[i] == 0


@test_utils.test()
def test_stop_grad2():
    x = ti.field(ti.f32)
    loss = ti.field(ti.f32)

    n = 128

    ti.root.dense(ti.i, n).place(x)
    ti.root.place(loss)
    ti.root.lazy_grad()

    @ti.kernel
    def func():
        # Two loops, one with stop grad on without
        for i in range(n):
            ti.stop_grad(x)
            loss[None] += x[i] ** 2
        for i in range(n):
            loss[None] += x[i] ** 2

    for i in range(n):
        x[i] = i

    with ti.ad.Tape(loss):
        func()

    # If without stop, grad x.grad[i] = i * 4
    for i in range(n):
        assert x.grad[i] == i * 2


@test_utils.test(arch=[ti.cpu, ti.cuda], require=ti.extension.adstack, offline_cache=False)
def test_stop_grad_inside_reversed_loop():
    x = ti.field(ti.f32, shape=2, needs_grad=True)
    y = ti.field(ti.f32, shape=2, needs_grad=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)

    @ti.kernel
    def energy():
        for i in x:
            total = x[i] * x[i]
            for j in range(2):
                ti.stop_grad(x)
                total += x[i] * x[i] + (j + 1) * y[i] * y[i]
            loss[None] += total

    x.fill(0.5)
    y.fill(0.25)
    with ti.ad.Tape(loss):
        energy()
    assert loss[None] == 1.875
    for i in range(2):
        assert x.grad[i] == 1
        assert y.grad[i] == 1.5
