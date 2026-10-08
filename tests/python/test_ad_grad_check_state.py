import numpy as np

import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda], require=ti.extension.data64, offline_cache=False)
def test_grad_check_preserves_accumulated_loss():
    x = ti.field(ti.f64, shape=(), needs_grad=True)
    loss = ti.field(ti.f64, shape=(), needs_grad=True)
    x[None] = 2
    loss[None] = 99

    @ti.kernel
    def evaluate():
        loss[None] += x[None] * x[None]

    # The next Tape must not pick up either the initial loss or the last result.
    for _ in range(2):
        with ti.ad.Tape(loss, grad_check=[x]):
            evaluate()
        assert x[None] == 2
        assert x.grad[None] == 4
        assert loss[None] == 4


@test_utils.test(arch=[ti.cpu, ti.cuda], require=ti.extension.data64, offline_cache=False)
def test_grad_check_snapshots_at_entry_and_restores_without_replay():
    x = ti.field(ti.f64, shape=(), needs_grad=True)
    loss = ti.field(ti.f64, shape=(), needs_grad=True)
    x[None] = 0.5
    tape = ti.ad.Tape(loss, grad_check=[x])
    # Both changed inputs and fields created after construction belong to entry.
    marker = ti.field(ti.i32, shape=())
    marker[None] = 10
    x[None] = 2
    loss[None] = 99
    calls = []

    @ti.kernel
    def forward():
        loss[None] += x[None] * x[None]

    @ti.ad.grad_replaced
    def evaluate():
        calls.append(x[None])
        marker[None] += 1
        forward()

    @ti.ad.grad_for(evaluate)
    def backward():
        forward.grad()

    # Quadratic central differences pass on the first positive/negative pair.
    original_rand = np.random.rand
    np.random.rand = lambda *shape: np.ones(shape)
    try:
        with tape:
            evaluate()
    finally:
        np.random.rand = original_rand
    assert calls == [2, 2.125, 1.875]
    assert marker[None] == 11
    assert x[None] == 2
    assert x.grad[None] == 4
    assert loss[None] == 4
