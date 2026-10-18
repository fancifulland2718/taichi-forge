import pytest

import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("skip_intermediate", [False, True])
@test_utils.test(
    arch=ti.vulkan,
    offline_cache=False,
    vulkan_sparse_experimental=True,
    vulkan_spv_stats=True,
    vulkan_spv_stats_filter="all",
    vulkan_listgen_reuse=True,
)
def test_sparse_clear_and_gc_do_not_emit_shaders(skip_intermediate):
    ti.cfg.spirv_skip_intermediate_listgen = skip_intermediate
    values = ti.field(ti.f32, needs_grad=True)
    blocks = ti.root.pointer(ti.i, 2)
    blocks.dense(ti.i, 2).bitmasked(ti.i, 2).place(values, values.grad)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)

    @ti.kernel
    def activate(scale: ti.f32):
        values[2] = scale
        values[6] = 2 * scale
        values.grad[2] = 0
        values.grad[6] = 0

    @ti.kernel
    def energy():
        for i in values:
            loss[None] += 0.5 * values[i] * values[i]

    @ti.kernel
    def deactivate():
        for i in range(2):
            ti.deactivate(blocks, i)

    for iteration, scale in enumerate((2, 3)):
        deactivate()
        if iteration == 0:
            stats = ti.lang.runtime_ops.get_last_spv_stats()
            assert [task["type"] for task in stats] == ["range_for"]
        activate(scale)
        loss[None] = 0
        energy()
        if iteration == 0:
            stats = ti.lang.runtime_ops.get_last_spv_stats()
            # List construction and traversal are real work; the old separate
            # ClearList shaders performed no work. Keep all listgens when the
            # independent intermediate-list optimization is disabled.
            assert [task["type"] for task in stats] == ["listgen"] * (1 if skip_intermediate else 3) + ["struct_for"]
        assert loss[None] == 2.5 * scale * scale
        loss.grad[None] = 1
        energy.grad()
        assert values.grad[2] == scale
        assert values.grad[6] == 2 * scale

    deactivate()
    loss[None] = 0
    energy()
    assert loss[None] == 0
