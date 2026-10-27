import pytest

import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("dtype", [ti.f32, ti.i32, ti.u32], ids=["f32", "i32", "u32"])
@pytest.mark.parametrize("first", ["read", "write"])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_scalar_accessors_create_only_used_direction(dtype, first):
    value = ti.field(dtype, shape=())
    prog = ti.lang.impl.get_runtime().prog
    before = prog._debug_kernel_definition_count()
    for _ in range(2):
        if first == "read":
            assert value[None] == 0
        else:
            value[None] = 7
        assert prog._debug_kernel_definition_count() == before + 1
    if first == "read":
        value[None] = 7
    assert value[None] == 7
    assert prog._debug_kernel_definition_count() == before + 2


@pytest.mark.parametrize("first", ["read", "write"])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_scalar_accessor_one_direction_is_retired_with_tree(first):
    prog = ti.lang.impl.get_runtime().prog
    before = prog._debug_kernel_definition_count()
    for _ in range(2):
        value = ti.field(ti.i32)
        builder = ti.FieldsBuilder()
        builder.place(value)
        tree = builder.finalize()
        if first == "read":
            assert value[None] == 0
        else:
            value[None] = 7
        assert prog._debug_kernel_definition_count() == before + 1
        tree.destroy()
        assert prog._debug_kernel_definition_count() == before
