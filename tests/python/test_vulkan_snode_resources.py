import pytest

import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("pool", ["inline", "independent", "chunked"])
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_sparse_tree_resources_retire_without_affecting_other_trees(pool):
    ti.cfg.vulkan_pointer_independent_pool = pool != "inline"
    ti.cfg.vulkan_pointer_allocator_kind = "chunked" if pool == "chunked" else "bump"
    ti.cfg.vulkan_pointer_max_chunks = 2 if pool == "chunked" else 1
    weight = ti.field(ti.f32, shape=(), needs_grad=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)
    weight[None] = 2

    def create_tree():
        value = ti.field(ti.i32)
        fields = ti.FieldsBuilder()
        fields.pointer(ti.i, 4).dense(ti.i, 2).place(value)
        return value, fields.finalize()

    @ti.kernel
    def seed(value: ti.template(), shift: ti.i32):
        for i in range(3):
            value[3 * i] = 3 * i + 1 + shift

    @ti.kernel
    def energy(value: ti.template()):
        for i in value:
            loss[None] += ti.cast(value[i], ti.f32) * weight[None]

    def check(value, total):
        loss[None] = 0
        weight.grad[None] = 0
        loss.grad[None] = 1
        energy(value)
        energy.grad(value)
        assert loss[None] == 2 * total
        assert weight.grad[None] == total

    survivor, surviving_tree = create_tree()
    seed(survivor, 10)
    for shift in (0, 2):
        transient, tree = create_tree()
        seed(transient, shift)
        check(transient, 12 + 3 * shift)
        check(survivor, 42)
        tree.destroy()
        check(survivor, 42)
    surviving_tree.destroy()
