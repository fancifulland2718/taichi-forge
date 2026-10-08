import numpy as np

import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_tape_custom_gradients_keep_keyword_arguments():
    x = ti.field(ti.f32, shape=4, needs_grad=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)
    x.fill(2)
    seen = []

    @ti.kernel
    def forward(scale: ti.f32):
        for i in x:
            loss[None] += scale * x[i]

    @ti.ad.grad_replaced
    def custom(multiplier=1, *, scale=1):
        forward(multiplier * scale)

    @ti.ad.grad_for(custom)
    def backward(multiplier=1, *, scale=1):
        seen.append((multiplier, scale))
        forward.grad(multiplier * scale)

    with ti.ad.Tape(loss):
        custom(scale=3)
        forward(1)
        custom(2, scale=4)
        custom(multiplier=3, scale=5)
    assert seen == [(3, 5), (2, 4), (1, 3)]
    assert loss[None] == 216
    np.testing.assert_array_equal(x.grad.to_numpy(), 27)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_tape_custom_method_keeps_required_keyword_only_argument():
    @ti.data_oriented
    class Model:
        def __init__(self):
            self.x = ti.field(ti.f32, shape=(), needs_grad=True)
            self.loss = ti.field(ti.f32, shape=(), needs_grad=True)

        @ti.kernel
        def kernel(self, scale: ti.f32):
            self.loss[None] += scale * self.x[None]

        @ti.ad.grad_replaced
        def evaluate(self, *, scale):
            self.kernel(scale)

        @ti.ad.grad_for(evaluate)
        def backward(self, *, scale):
            self.kernel.grad(scale)

    model = Model()
    model.x[None] = 2
    with ti.ad.Tape(model.loss):
        model.evaluate(scale=3)
    assert model.loss[None] == 6
    assert model.x.grad[None] == 3


@test_utils.test(arch=[ti.cpu, ti.cuda], require=ti.extension.data64, offline_cache=False)
def test_gradient_checker_replays_custom_keyword_arguments():
    x = ti.field(ti.f64, shape=(), needs_grad=True)
    loss = ti.field(ti.f64, shape=(), needs_grad=True)
    x[None] = 1.25

    @ti.kernel
    def evaluate(scale: ti.f64):
        loss[None] += scale * x[None] * x[None]

    @ti.kernel
    def add_constant(bias: ti.f64):
        loss[None] += bias

    @ti.ad.grad_replaced
    def custom(*, scale=1.0):
        evaluate(scale)

    @ti.ad.grad_for(custom)
    def custom_grad(*, scale=1.0):
        evaluate.grad(scale)

    @ti.ad.no_grad
    def constant(*, bias=1.0):
        add_constant(bias)

    with ti.ad.Tape(loss, grad_check=[x]):
        constant(bias=7.0)
        custom(scale=3.0)
    assert x[None] == 1.25
    assert x.grad[None] == 7.5
    assert loss[None] == 11.6875
