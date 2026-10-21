"""Repeated resource member names retain independent bindings and gradients."""

import numpy as np

import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_argpack_repeated_resource_members_and_adjoint(tmp_path):
    member = ti.types.argpack(values=ti.types.ndarray(ti.f32, 1, needs_grad=True))
    nested = ti.types.argpack(child=member)
    x, y = ti.ndarray(ti.f32, 1, needs_grad=True), ti.ndarray(ti.f32, 1, needs_grad=True)
    loss = ti.ndarray(ti.f32, 1, needs_grad=True)
    x.fill(3)
    y.fill(7)
    a, b = nested(child=member(values=x)), nested(child=member(values=y))

    @ti.kernel
    def read(left: nested, right: nested) -> ti.f32:
        return 10 * left.child.values[0] + right.child.values[0]

    @ti.kernel
    def energy(left: nested, right: nested, total: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in left.child.values:
            total[0] += left.child.values[i] ** 2 + 3 * right.child.values[i]

    assert read(a, b) == 37
    assert read(b, a) == 73
    loss.fill(0)
    loss.grad.fill(1)
    x.grad.fill(0)
    y.grad.fill(0)
    energy(a, b, loss)
    energy.grad(a, b, loss)
    np.testing.assert_allclose(loss.to_numpy(), [30])
    np.testing.assert_allclose(x.grad.to_numpy(), [6])
    np.testing.assert_allclose(y.grad.to_numpy(), [3])
    module = ti.aot.Module()
    module.add_kernel(read)
    module.add_kernel(energy)
    module.add_kernel(energy.grad, name="backward")
    module.archive(str(tmp_path / "members.tcm"))
    with ti.aot.borrow_runtime() as runtime:
        loaded = runtime.load_module(tmp_path / "members.tcm")
        for name in ("read", "energy", "backward"):
            assert loaded.kernel(name)
