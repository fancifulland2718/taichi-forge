import pytest

import taichi_forge as ti
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.lang.snode import _select_hash_snode_capacity
from tests import test_utils


def test_hash_snode_capacity_selector():
    assert _select_hash_snode_capacity(
        logical_elements=1024,
        expected_active=17,
        default_load_factor=0.5,
    ) == (64, 0.5)
    assert _select_hash_snode_capacity(
        logical_elements=1024,
        max_active=17,
        hash_load_factor=0.75,
        default_load_factor=0.5,
    ) == (32, 0.75)
    with pytest.warns(UserWarning):
        assert _select_hash_snode_capacity(
            logical_elements=1024,
            capacity=17,
            default_load_factor=0.5,
        ) == (32, None)

    with pytest.raises(TaichiRuntimeError):
        _select_hash_snode_capacity(
            logical_elements=1024,
            max_active=8,
            expected_active=8,
            default_load_factor=0.5,
        )


@test_utils.test(arch=ti.cpu, require=ti.extension.sparse)
def test_hash_snode_requires_gate():
    x = ti.field(ti.i32)
    with pytest.raises(TaichiRuntimeError):
        ti.root.hash(ti.i, 16, max_active=4).place(x)


@test_utils.test(
    arch=ti.cpu,
    require=ti.extension.sparse,
    hash_snode_experimental=True,
    offline_cache=False,
)
def test_hash_snode_direct_place_struct_for():
    x = ti.field(ti.i32)
    count = ti.field(ti.i32, shape=())
    key_sum = ti.field(ti.i32, shape=())
    value_sum = ti.field(ti.i32, shape=())

    ti.root.hash(ti.i, 128, max_active=8).place(x)

    @ti.kernel
    def write():
        x[3] = 8
        x[41] = 12
        x[96] = 5

    @ti.kernel
    def collect():
        for i in x:
            count[None] += 1
            key_sum[None] += i
            value_sum[None] += x[i]

    write()
    collect()

    assert count[None] == 3
    assert key_sum[None] == 3 + 41 + 96
    assert value_sum[None] == 8 + 12 + 5


@test_utils.test(
    arch=ti.cpu,
    require=ti.extension.sparse,
    hash_snode_experimental=True,
    offline_cache=False,
)
def test_hash_snode_is_active_and_deactivate():
    x = ti.field(ti.i32)
    active_count = ti.field(ti.i32, shape=())
    value_sum = ti.field(ti.i32, shape=())

    h = ti.root.hash(ti.i, 128, capacity=16)
    h.place(x)

    @ti.kernel
    def write():
        x[5] = 9
        x[33] = 2

    @ti.kernel
    def count_active():
        for i in range(64):
            active_count[None] += ti.is_active(h, [i])
            value_sum[None] += x[i]

    @ti.kernel
    def remove_one():
        ti.deactivate(h, [5])

    write()
    count_active()
    assert active_count[None] == 2
    assert value_sum[None] == 11

    active_count[None] = 0
    value_sum[None] = 0
    remove_one()
    count_active()
    assert active_count[None] == 1
    assert value_sum[None] == 2


@test_utils.test(
    arch=ti.cpu,
    require=ti.extension.sparse,
    hash_snode_experimental=True,
    offline_cache=False,
)
def test_hash_snode_dense_child_coordinates_2d():
    x = ti.field(ti.i32)
    count = ti.field(ti.i32, shape=())
    coord_sum = ti.field(ti.i32, shape=())
    value_sum = ti.field(ti.i32, shape=())

    ti.root.hash(ti.ij, (32, 32), max_active=8).dense(ti.k, 4).place(x)

    @ti.kernel
    def write():
        x[2, 3, 1] = 7
        x[5, 7, 2] = 11

    @ti.kernel
    def collect():
        for i, j, k in x:
            count[None] += 1
            coord_sum[None] += i * 100 + j * 10 + k
            value_sum[None] += x[i, j, k]

    write()
    collect()

    assert count[None] == 8
    assert coord_sum[None] == ((2 * 100 + 3 * 10) * 4 + 6) + (
        (5 * 100 + 7 * 10) * 4 + 6
    )
    assert value_sum[None] == 18


@test_utils.test(
    arch=ti.cpu,
    require=ti.extension.sparse,
    hash_snode_experimental=True,
    hash_snode_default_load_factor=1.0,
    offline_cache=False,
)
def test_hash_snode_expected_active_and_default_load_factor():
    x = ti.field(ti.i32)
    count = ti.field(ti.i32, shape=())
    value_sum = ti.field(ti.i32, shape=())

    ti.root.hash(ti.i, 32, expected_active=4).place(x)

    @ti.kernel
    def write():
        x[1] = 3
        x[7] = 5
        x[13] = 9

    @ti.kernel
    def collect():
        for i in x:
            count[None] += 1
            value_sum[None] += x[i]

    write()
    collect()

    assert count[None] == 3
    assert value_sum[None] == 17


@test_utils.test(
    arch=ti.cpu,
    require=ti.extension.sparse,
    hash_snode_experimental=True,
    offline_cache=False,
)
def test_hash_snode_cpu_overflow_raises():
    x = ti.field(ti.i32)
    ti.root.hash(ti.i, 64, capacity=2).place(x)

    @ti.kernel
    def write_too_many():
        x[1] = 1
        x[2] = 2
        x[3] = 3

    with pytest.raises(Exception, match="Hash SNode table overflow"):
        write_too_many()


@test_utils.test(
    arch=ti.cuda,
    require=ti.extension.sparse,
    hash_snode_experimental=True,
    offline_cache=False,
    cuda_sparse_pool_auto_size=True,
)
def test_hash_snode_cuda_direct_place_struct_for():
    x = ti.field(ti.i32)
    count = ti.field(ti.i32, shape=())
    key_sum = ti.field(ti.i32, shape=())
    value_sum = ti.field(ti.i32, shape=())

    ti.root.hash(ti.i, 256, max_active=32).place(x)

    @ti.kernel
    def write():
        for i in range(32):
            key = i * 7 + 3
            x[key] = key + 1

    @ti.kernel
    def collect():
        for i in x:
            count[None] += 1
            key_sum[None] += i
            value_sum[None] += x[i]

    write()
    collect()

    expected_keys = sum(i * 7 + 3 for i in range(32))
    assert count[None] == 32
    assert key_sum[None] == expected_keys
    assert value_sum[None] == expected_keys + 32


@test_utils.test(
    arch=ti.cuda,
    require=ti.extension.sparse,
    hash_snode_experimental=True,
    offline_cache=False,
    cuda_sparse_pool_auto_size=True,
)
def test_hash_snode_cuda_is_active_and_deactivate():
    x = ti.field(ti.i32)
    active_count = ti.field(ti.i32, shape=())
    value_sum = ti.field(ti.i32, shape=())

    h = ti.root.hash(ti.i, 128, capacity=32)
    h.place(x)

    @ti.kernel
    def write():
        x[5] = 9
        x[33] = 2

    @ti.kernel
    def count_active():
        for i in range(64):
            active_count[None] += ti.is_active(h, [i])
            value_sum[None] += x[i]

    @ti.kernel
    def remove_one():
        ti.deactivate(h, [5])

    write()
    count_active()
    assert active_count[None] == 2
    assert value_sum[None] == 11

    active_count[None] = 0
    value_sum[None] = 0
    remove_one()
    count_active()
    assert active_count[None] == 1
    assert value_sum[None] == 2


@test_utils.test(
    arch=ti.cuda,
    require=ti.extension.sparse,
    hash_snode_experimental=True,
    offline_cache=False,
    cuda_sparse_pool_auto_size=True,
)
def test_hash_snode_cuda_dense_child_coordinates_2d():
    x = ti.field(ti.i32)
    count = ti.field(ti.i32, shape=())
    coord_sum = ti.field(ti.i32, shape=())
    value_sum = ti.field(ti.i32, shape=())

    ti.root.hash(ti.ij, (32, 32), max_active=8).dense(ti.k, 4).place(x)

    @ti.kernel
    def write():
        x[2, 3, 1] = 7
        x[5, 7, 2] = 11

    @ti.kernel
    def collect():
        for i, j, k in x:
            count[None] += 1
            coord_sum[None] += i * 100 + j * 10 + k
            value_sum[None] += x[i, j, k]

    write()
    collect()

    assert count[None] == 8
    assert coord_sum[None] == ((2 * 100 + 3 * 10) * 4 + 6) + (
        (5 * 100 + 7 * 10) * 4 + 6
    )
    assert value_sum[None] == 18


@test_utils.test(
    arch=ti.cuda,
    require=ti.extension.sparse,
    hash_snode_experimental=True,
    offline_cache=False,
    cuda_sparse_pool_auto_size=True,
)
def test_hash_snode_cuda_overflow_raises():
    x = ti.field(ti.i32)
    ti.root.hash(ti.i, 64, capacity=2).place(x)

    @ti.kernel
    def write_too_many():
        x[1] = 1
        x[2] = 2
        x[3] = 3

    with pytest.raises(Exception, match="Hash SNode table overflow"):
        write_too_many()


@test_utils.test(
    arch=ti.vulkan,
    hash_snode_experimental=True,
    vulkan_sparse_experimental=True,
    offline_cache=False,
    vulkan_listgen_dynamic_size=True,
)
def test_hash_snode_vulkan_direct_place_struct_for():
    x = ti.field(ti.i32)
    count = ti.field(ti.i32, shape=())
    key_sum = ti.field(ti.i32, shape=())
    value_sum = ti.field(ti.i32, shape=())

    ti.root.hash(ti.i, 256, max_active=32).place(x)

    @ti.kernel
    def write():
        for i in range(32):
            key = i * 7 + 3
            x[key] = key + 1

    @ti.kernel
    def collect():
        for i in x:
            count[None] += 1
            key_sum[None] += i
            value_sum[None] += x[i]

    write()
    collect()

    expected_keys = sum(i * 7 + 3 for i in range(32))
    assert count[None] == 32
    assert key_sum[None] == expected_keys
    assert value_sum[None] == expected_keys + 32


@test_utils.test(
    arch=ti.vulkan,
    hash_snode_experimental=True,
    vulkan_sparse_experimental=True,
    offline_cache=False,
    vulkan_listgen_dynamic_size=True,
)
def test_hash_snode_vulkan_struct_for_keys_only():
    x = ti.field(ti.i32)
    count = ti.field(ti.i32, shape=())
    key_sum = ti.field(ti.i32, shape=())

    ti.root.hash(ti.i, 256, max_active=32).place(x)

    @ti.kernel
    def write():
        for i in range(32):
            key = i * 7 + 3
            x[key] = key + 1

    @ti.kernel
    def collect():
        for i in x:
            count[None] += 1
            key_sum[None] += i

    write()
    collect()

    expected_keys = sum(i * 7 + 3 for i in range(32))
    assert count[None] == 32
    assert key_sum[None] == expected_keys


@test_utils.test(
    arch=ti.vulkan,
    hash_snode_experimental=True,
    vulkan_sparse_experimental=True,
    offline_cache=False,
    vulkan_listgen_dynamic_size=True,
)
def test_hash_snode_vulkan_is_active_and_deactivate():
    x = ti.field(ti.i32)
    active_count = ti.field(ti.i32, shape=())
    value_sum = ti.field(ti.i32, shape=())

    h = ti.root.hash(ti.i, 128, capacity=32)
    h.place(x)

    @ti.kernel
    def write():
        x[5] = 9
        x[33] = 2

    @ti.kernel
    def count_active():
        for i in range(64):
            active_count[None] += ti.is_active(h, [i])
            value_sum[None] += x[i]

    @ti.kernel
    def remove_one():
        ti.deactivate(h, [5])

    write()
    count_active()
    assert active_count[None] == 2
    assert value_sum[None] == 11

    active_count[None] = 0
    value_sum[None] = 0
    remove_one()
    count_active()
    assert active_count[None] == 1
    assert value_sum[None] == 2


@test_utils.test(
    arch=ti.vulkan,
    hash_snode_experimental=True,
    vulkan_sparse_experimental=True,
    offline_cache=False,
    vulkan_listgen_dynamic_size=True,
)
def test_hash_snode_vulkan_dense_child_coordinates_2d():
    x = ti.field(ti.i32)
    count = ti.field(ti.i32, shape=())
    coord_sum = ti.field(ti.i32, shape=())
    value_sum = ti.field(ti.i32, shape=())

    ti.root.hash(ti.ij, (32, 32), max_active=8).dense(ti.k, 4).place(x)

    @ti.kernel
    def write():
        x[2, 3, 1] = 7
        x[5, 7, 2] = 11

    @ti.kernel
    def collect():
        for i, j, k in x:
            count[None] += 1
            coord_sum[None] += i * 100 + j * 10 + k
            value_sum[None] += x[i, j, k]

    write()
    collect()

    assert count[None] == 8
    assert coord_sum[None] == ((2 * 100 + 3 * 10) * 4 + 6) + (
        (5 * 100 + 7 * 10) * 4 + 6
    )
    assert value_sum[None] == 18


@test_utils.test(
    arch=ti.vulkan,
    hash_snode_experimental=True,
    vulkan_sparse_experimental=True,
    offline_cache=False,
    vulkan_listgen_dynamic_size=True,
)
def test_hash_snode_vulkan_overflow_raises_on_sync():
    x = ti.field(ti.i32)
    ti.root.hash(ti.i, 64, capacity=2).place(x)

    @ti.kernel
    def write_too_many():
        x[1] = 1
        x[2] = 2
        x[3] = 3

    write_too_many()
    with pytest.raises(Exception, match="Hash SNode table overflow"):
        ti.sync()
