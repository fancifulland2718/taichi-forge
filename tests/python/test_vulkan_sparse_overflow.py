import pytest

import taichi_forge as ti
from tests import test_utils


def _run_pointer_pool_overflow():
    x = ti.field(ti.i32)
    outer = ti.root.pointer(ti.i, 2)
    outer.pointer(ti.j, 2, vk_max_active=2).place(x)

    @ti.kernel
    def fill_pool():
        for j in range(2):
            x[0, j] = j + 1

    @ti.kernel
    def activate_beyond_pool():
        x[1, 0] = 3

    fill_pool()
    ti.sync()
    assert x[0, 0] == 1
    assert x[0, 1] == 2

    activate_beyond_pool()
    with pytest.raises(Exception, match="Pointer SNode pool overflow"):
        ti.sync()


@test_utils.test(
    arch=ti.vulkan,
    vulkan_sparse_experimental=True,
    offline_cache=False,
)
def test_pointer_pool_overflow_raises_on_sync():
    _run_pointer_pool_overflow()


@test_utils.test(
    arch=ti.vulkan,
    vulkan_sparse_experimental=True,
    vulkan_pointer_cas_marker=False,
    offline_cache=False,
)
def test_pointer_pool_overflow_legacy_protocol_raises_on_sync():
    _run_pointer_pool_overflow()


@test_utils.test(
    arch=ti.vulkan,
    vulkan_sparse_experimental=True,
    offline_cache=False,
)
def test_dynamic_append_overflow_raises_on_sync():
    x = ti.field(ti.i32)
    ti.root.dynamic(ti.i, 2, chunk_size=2).place(x)

    @ti.kernel
    def fill_capacity():
        for i in range(2):
            ti.append(x.parent(), [], i + 1)

    @ti.kernel
    def append_beyond_capacity():
        ti.append(x.parent(), [], 3)

    fill_capacity()
    ti.sync()
    assert sorted([x[0], x[1]]) == [1, 2]

    append_beyond_capacity()
    with pytest.raises(Exception, match="Dynamic SNode capacity overflow"):
        ti.sync()


@test_utils.test(
    arch=ti.vulkan,
    vulkan_sparse_experimental=True,
    offline_cache=False,
)
def test_dynamic_activate_overflow_raises_on_sync():
    x = ti.field(ti.i32)
    ti.root.dynamic(ti.i, 2, chunk_size=2).place(x)

    @ti.kernel
    def fill_capacity():
        x[0] = 1
        x[1] = 2

    @ti.kernel
    def activate_beyond_capacity():
        x[2] = 3

    fill_capacity()
    ti.sync()
    assert x[0] == 1
    assert x[1] == 2

    activate_beyond_capacity()
    with pytest.raises(Exception, match="Dynamic SNode capacity overflow"):
        ti.sync()
