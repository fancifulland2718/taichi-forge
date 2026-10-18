"""Pointer bump and recycled-pop paths leave the same structured SPIR-V loop."""

import taichi_forge as ti
from tests import test_utils


@test_utils.test(
    arch=ti.vulkan,
    offline_cache=False,
    vulkan_pointer_independent_pool=True,
    vulkan_pointer_deterministic_slot=False,
)
def test_pointer_freelist_structured_bump_pop_and_payload_reset(monkeypatch):
    monkeypatch.setenv("TI_VULKAN_VALIDATE_SPIRV", "1")
    values = ti.field(ti.f32)
    pointer = ti.root.pointer(ti.i, 4)
    pointer.bitmasked(ti.i, 2).place(values)

    @ti.kernel
    def seed():
        values[2] = 1
        values[3] = 8
        values[5] = 2

    @ti.kernel
    def recycle():
        ti.deactivate(pointer, 1)
        values[0] = 4

    @ti.kernel
    def clear():
        for i in range(4):
            ti.deactivate(pointer, i)

    @ti.kernel
    def total() -> ti.f32:
        result = 0.0
        for i in values:
            result += values[i]
        return result

    for _ in range(2):
        seed()
        assert total() == 11
        recycle()
        assert total() == 6  # The other bitmasked child was cleared on reuse.
        clear()
        assert total() == 0
