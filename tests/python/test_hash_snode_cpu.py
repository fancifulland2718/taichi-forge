import pytest

import taichi_forge as ti
from taichi_forge.lang import impl
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.lang.snode import _select_hash_snode_capacity
from tests import test_utils


def _hash_mix_u32(x):
    x &= 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    x ^= x >> 16
    return x & 0xFFFFFFFF


def _collision_keys(capacity, count, bucket=0, domain=4096):
    keys = []
    for k in range(domain):
        if (_hash_mix_u32(k) & (capacity - 1)) == bucket:
            keys.append(k)
            if len(keys) == count:
                return tuple(keys)
    raise RuntimeError("not enough collision keys")


def test_hash_snode_capacity_selector():
    assert _select_hash_snode_capacity(
        logical_elements=1024,
        expected_active=17,
        default_load_factor=0.5,
    ) == (64, 0.5, 17)
    assert _select_hash_snode_capacity(
        logical_elements=1024,
        max_active=17,
        hash_load_factor=0.75,
        default_load_factor=0.5,
    ) == (32, 0.75, 17)
    with pytest.warns(UserWarning):
        assert _select_hash_snode_capacity(
            logical_elements=1024,
            capacity=17,
            default_load_factor=0.5,
        ) == (32, None, None)

    with pytest.raises(TaichiRuntimeError):
        _select_hash_snode_capacity(
            logical_elements=1024,
            max_active=8,
            expected_active=8,
            default_load_factor=0.5,
        )


@test_utils.test(
    arch=ti.cpu,
    require=ti.extension.sparse,
    hash_snode_experimental=False,
)
def test_hash_snode_can_be_disabled_by_flag():
    x = ti.field(ti.i32)
    with pytest.raises(TaichiRuntimeError, match="disabled"):
        ti.root.hash(ti.i, 16, max_active=4).place(x)


@test_utils.test(
    arch=ti.cpu,
    require=ti.extension.sparse,
    offline_cache=False,
)
def test_hash_snode_hash_dense_smoke_with_default_enabled():
    x = ti.field(ti.i32)
    h = ti.root.hash(ti.i, 16, capacity=8)

    h.dense(ti.i, 2).place(x)


@test_utils.test(
    arch=ti.cpu,
    require=ti.extension.sparse,
    hash_snode_experimental=True,
    offline_cache=False,
)
def test_hash_snode_diagnostic_errors_for_capacity_and_key_limits():
    with pytest.raises(TaichiRuntimeError, match="32-bit flattened keys"):
        ti.root.hash(ti.ij, (1 << 16, 1 << 16), capacity=8)

    with pytest.raises(TaichiRuntimeError, match="expected_active, max_active, or capacity"):
        ti.root.hash(ti.i, 16)

    with pytest.raises(TaichiRuntimeError, match="hash_load_factor"):
        ti.root.hash(ti.i, 16, expected_active=4, hash_load_factor=0)


def _run_hash_snode_explicit_activate_and_deactivate_all():
    x = ti.field(ti.i32)
    count = ti.field(ti.i32, shape=())
    key_sum = ti.field(ti.i32, shape=())
    value_sum = ti.field(ti.i32, shape=())
    active_count = ti.field(ti.i32, shape=())

    h = ti.root.hash(ti.i, 64, capacity=8)
    h.place(x)

    @ti.kernel
    def activate_only():
        ti.activate(h, [3])
        ti.activate(h, [9])

    @ti.kernel
    def write_after_clear():
        x[4] = 11

    @ti.kernel
    def collect():
        count[None] = 0
        key_sum[None] = 0
        value_sum[None] = 0
        active_count[None] = 0
        for i in range(16):
            active_count[None] += ti.is_active(h, [i])
        for i in x:
            count[None] += 1
            key_sum[None] += i
            value_sum[None] += x[i]

    activate_only()
    collect()
    assert count[None] == 2
    assert key_sum[None] == 12
    assert value_sum[None] == 0
    assert active_count[None] == 2

    h.deactivate_all()
    collect()
    assert count[None] == 0
    assert key_sum[None] == 0
    assert value_sum[None] == 0
    assert active_count[None] == 0

    write_after_clear()
    collect()
    assert count[None] == 1
    assert key_sum[None] == 4
    assert value_sum[None] == 11
    assert active_count[None] == 1


def _run_hash_snode_tombstone_reuse():
    x = ti.field(ti.i32)
    count = ti.field(ti.i32, shape=())
    key_sum = ti.field(ti.i32, shape=())
    value_sum = ti.field(ti.i32, shape=())

    h = ti.root.hash(ti.i, 128, capacity=4)
    h.place(x)

    @ti.kernel
    def fill_table():
        x[1] = 10
        x[9] = 90
        x[17] = 170
        x[25] = 250

    @ti.kernel
    def remove_two():
        ti.deactivate(h, [9])
        ti.deactivate(h, [17])

    @ti.kernel
    def reuse_slots():
        x[33] = 330
        x[41] = 410

    @ti.kernel
    def collect():
        count[None] = 0
        key_sum[None] = 0
        value_sum[None] = 0
        for i in x:
            count[None] += 1
            key_sum[None] += i
            value_sum[None] += x[i]

    fill_table()
    remove_two()
    reuse_slots()
    collect()

    assert count[None] == 4
    assert key_sum[None] == 1 + 25 + 33 + 41
    assert value_sum[None] == 10 + 250 + 330 + 410


def _run_hash_snode_concurrent_same_key_activation():
    x = ti.field(ti.i32)
    count = ti.field(ti.i32, shape=())
    key_sum = ti.field(ti.i32, shape=())
    value_sum = ti.field(ti.i32, shape=())
    active = ti.field(ti.i32, shape=())

    h = ti.root.hash(ti.i, 128, capacity=16)
    h.place(x)

    @ti.kernel
    def write_same_key():
        for i in range(128):
            x[17] = i + 1

    @ti.kernel
    def collect():
        count[None] = 0
        key_sum[None] = 0
        value_sum[None] = 0
        active[None] = ti.is_active(h, [17])
        for i in x:
            count[None] += 1
            key_sum[None] += i
            value_sum[None] += x[i]

    write_same_key()
    collect()

    assert active[None] == 1
    assert count[None] == 1
    assert key_sum[None] == 17
    assert value_sum[None] > 0


def _run_hash_snode_collision_probe_chain_reuse():
    capacity = 8
    keys = _collision_keys(capacity, 12)
    initial = keys[:8]
    removed = keys[1::2][:4]
    kept = tuple(k for k in initial if k not in removed)
    added = keys[8:12]
    expected_keys = kept + added

    x = ti.field(ti.i32)
    count = ti.field(ti.i32, shape=())
    key_sum = ti.field(ti.i32, shape=())
    value_sum = ti.field(ti.i32, shape=())
    active_count = ti.field(ti.i32, shape=())

    h = ti.root.hash(ti.i, 4096, capacity=capacity)
    h.place(x)

    @ti.kernel
    def fill_full_probe_chain():
        for p in ti.static(range(8)):
            x[initial[p]] = p + 1

    @ti.kernel
    def remove_alternating():
        for p in ti.static(range(4)):
            ti.deactivate(h, [removed[p]])

    @ti.kernel
    def reuse_tombstones():
        for p in ti.static(range(4)):
            x[added[p]] = 100 + p

    @ti.kernel
    def collect():
        count[None] = 0
        key_sum[None] = 0
        value_sum[None] = 0
        active_count[None] = 0
        for p in ti.static(range(8)):
            active_count[None] += ti.is_active(h, [expected_keys[p]])
        for i in x:
            count[None] += 1
            key_sum[None] += i
            value_sum[None] += x[i]

    fill_full_probe_chain()
    remove_alternating()
    reuse_tombstones()
    collect()

    expected_values = 1 + 3 + 5 + 7 + 100 + 101 + 102 + 103
    assert active_count[None] == 8
    assert count[None] == 8
    assert key_sum[None] == sum(expected_keys)
    assert value_sum[None] == expected_values


def _run_hash_snode_active_list_pure_insert():
    x = ti.field(ti.i32)
    count = ti.field(ti.i32, shape=())
    key_sum = ti.field(ti.i32, shape=())
    value_sum = ti.field(ti.i32, shape=())

    ti.root.hash(ti.i, 512, capacity=64).place(x)

    @ti.kernel
    def write():
        for p in range(16):
            key = p * 17 + 5
            x[key] = key + 3

    @ti.kernel
    def collect():
        count[None] = 0
        key_sum[None] = 0
        value_sum[None] = 0
        for i in x:
            count[None] += 1
            key_sum[None] += i
            value_sum[None] += x[i]

    write()
    collect()

    expected_keys = sum(p * 17 + 5 for p in range(16))
    assert count[None] == 16
    assert key_sum[None] == expected_keys
    assert value_sum[None] == expected_keys + 16 * 3


def _run_hash_snode_active_list_churn_fallback():
    x = ti.field(ti.i32)
    count = ti.field(ti.i32, shape=())
    key_sum = ti.field(ti.i32, shape=())
    value_sum = ti.field(ti.i32, shape=())
    active_count = ti.field(ti.i32, shape=())

    h = ti.root.hash(ti.i, 512, capacity=16)
    h.place(x)

    @ti.kernel
    def write_initial():
        x[3] = 10
        x[19] = 20
        x[35] = 30
        x[51] = 40

    @ti.kernel
    def churn():
        ti.deactivate(h, [19])
        ti.deactivate(h, [35])
        x[67] = 70

    @ti.kernel
    def collect():
        count[None] = 0
        key_sum[None] = 0
        value_sum[None] = 0
        active_count[None] = 0
        active_count[None] += ti.is_active(h, [3])
        active_count[None] += ti.is_active(h, [51])
        active_count[None] += ti.is_active(h, [67])
        for i in x:
            count[None] += 1
            key_sum[None] += i
            value_sum[None] += x[i]

    write_initial()
    churn()
    collect()

    assert active_count[None] == 3
    assert count[None] == 3
    assert key_sum[None] == 3 + 51 + 67
    assert value_sum[None] == 10 + 40 + 70


def _run_hash_snode_active_list_reuse_all_tombstones():
    capacity = 8
    keys = _collision_keys(capacity, 6)
    initial = keys[:4]
    removed = keys[1:3]
    added = keys[4:6]
    expected_keys = (initial[0], initial[3], added[0], added[1])

    x = ti.field(ti.i32)
    count = ti.field(ti.i32, shape=())
    key_sum = ti.field(ti.i32, shape=())
    value_sum = ti.field(ti.i32, shape=())
    active_count = ti.field(ti.i32, shape=())

    h = ti.root.hash(ti.i, 4096, capacity=capacity)
    h.place(x)

    @ti.kernel
    def fill_initial():
        for p in ti.static(range(4)):
            x[initial[p]] = p + 1

    @ti.kernel
    def remove_middle():
        for p in ti.static(range(2)):
            ti.deactivate(h, [removed[p]])

    @ti.kernel
    def reuse_all_tombstones():
        for p in ti.static(range(2)):
            x[added[p]] = 100 + p

    @ti.kernel
    def collect():
        count[None] = 0
        key_sum[None] = 0
        value_sum[None] = 0
        active_count[None] = 0
        for p in ti.static(range(4)):
            active_count[None] += ti.is_active(h, [expected_keys[p]])
        for i in x:
            count[None] += 1
            key_sum[None] += i
            value_sum[None] += x[i]

    fill_initial()
    remove_middle()
    reuse_all_tombstones()
    collect()

    assert active_count[None] == 4
    assert count[None] == 4
    assert key_sum[None] == sum(expected_keys)
    assert value_sum[None] == 1 + 4 + 100 + 101


def _run_hash_snode_bitmasked_child():
    x = ti.field(ti.i32)
    count = ti.field(ti.i32, shape=())
    coord_sum = ti.field(ti.i32, shape=())
    value_sum = ti.field(ti.i32, shape=())
    active_bits = ti.field(ti.i32, shape=())
    active_hash = ti.field(ti.i32, shape=())

    h = ti.root.hash(ti.i, 64, capacity=16)
    b = h.bitmasked(ti.j, 4)
    b.place(x)

    @ti.kernel
    def write_and_activate():
        x[3, 1] = 31
        x[7, 2] = 72
        ti.activate(h, [11])
        ti.activate(b, [11, 3])

    @ti.kernel
    def remove_one_child():
        ti.deactivate(b, [7, 2])

    @ti.kernel
    def collect():
        count[None] = 0
        coord_sum[None] = 0
        value_sum[None] = 0
        active_bits[None] = 0
        active_hash[None] = 0
        active_bits[None] += ti.is_active(b, [3, 1])
        active_bits[None] += ti.is_active(b, [7, 2])
        active_bits[None] += ti.is_active(b, [11, 3])
        active_hash[None] += ti.is_active(h, [3])
        active_hash[None] += ti.is_active(h, [7])
        active_hash[None] += ti.is_active(h, [11])
        for i, j in x:
            count[None] += 1
            coord_sum[None] += i * 10 + j
            value_sum[None] += x[i, j]

    write_and_activate()
    remove_one_child()
    collect()

    assert active_bits[None] == 2
    assert active_hash[None] == 3
    assert count[None] == 2
    assert coord_sum[None] == 3 * 10 + 1 + 11 * 10 + 3
    assert value_sum[None] == 31


def _run_hash_snode_dynamic_child():
    x = ti.field(ti.i32)
    count = ti.field(ti.i32, shape=())
    coord_sum = ti.field(ti.i32, shape=())
    value_sum = ti.field(ti.i32, shape=())
    len3 = ti.field(ti.i32, shape=())
    len7 = ti.field(ti.i32, shape=())
    active_hash = ti.field(ti.i32, shape=())

    h = ti.root.hash(ti.i, 64, capacity=16)
    d = h.dynamic(ti.j, 8, chunk_size=4)
    d.place(x)

    @ti.kernel
    def append_values():
        ti.activate(h, [3])
        ti.activate(h, [7])
        ti.append(d, 3, 31)
        ti.append(d, 3, 32)
        ti.append(d, 7, 70)

    @ti.kernel
    def clear_key3():
        ti.deactivate(d, [3])

    @ti.kernel
    def collect():
        count[None] = 0
        coord_sum[None] = 0
        value_sum[None] = 0
        len3[None] = ti.length(d, 3)
        len7[None] = ti.length(d, 7)
        active_hash[None] = 0
        active_hash[None] += ti.is_active(h, [3])
        active_hash[None] += ti.is_active(h, [7])
        for i, j in x:
            count[None] += 1
            coord_sum[None] += i * 10 + j
            value_sum[None] += x[i, j]

    append_values()
    collect()
    assert active_hash[None] == 2
    assert len3[None] == 2
    assert len7[None] == 1
    assert count[None] == 3
    assert coord_sum[None] == 3 * 10 + 0 + 3 * 10 + 1 + 7 * 10 + 0
    assert value_sum[None] == 31 + 32 + 70

    clear_key3()
    collect()
    assert active_hash[None] == 2
    assert len3[None] == 0
    assert len7[None] == 1
    assert count[None] == 1
    assert coord_sum[None] == 7 * 10
    assert value_sum[None] == 70


def _run_hash_snode_pointer_child():
    x = ti.field(ti.i32)
    count = ti.field(ti.i32, shape=())
    coord_sum = ti.field(ti.i32, shape=())
    value_sum = ti.field(ti.i32, shape=())
    active_hash = ti.field(ti.i32, shape=())
    active_ptr = ti.field(ti.i32, shape=())

    h = ti.root.hash(ti.i, 64, capacity=16)
    p = h.pointer(ti.j, 4)
    p.place(x)

    @ti.kernel
    def write_initial():
        x[3, 1] = 31
        x[3, 3] = 33
        x[7, 2] = 72

    @ti.kernel
    def remove_one_pointer_child():
        ti.deactivate(p, [3, 1])

    @ti.kernel
    def remove_hash_and_rewrite():
        ti.deactivate(h, [3])
        x[3, 2] = 302

    @ti.kernel
    def collect():
        count[None] = 0
        coord_sum[None] = 0
        value_sum[None] = 0
        active_hash[None] = 0
        active_ptr[None] = 0
        active_hash[None] += ti.is_active(h, [3])
        active_hash[None] += ti.is_active(h, [7])
        active_ptr[None] += ti.is_active(p, [3, 1])
        active_ptr[None] += ti.is_active(p, [3, 2])
        active_ptr[None] += ti.is_active(p, [3, 3])
        active_ptr[None] += ti.is_active(p, [7, 2])
        for i, j in x:
            count[None] += 1
            coord_sum[None] += i * 10 + j
            value_sum[None] += x[i, j]

    write_initial()
    remove_one_pointer_child()
    collect()
    assert active_hash[None] == 2
    assert active_ptr[None] == 2
    assert count[None] == 2
    assert coord_sum[None] == 3 * 10 + 3 + 7 * 10 + 2
    assert value_sum[None] == 33 + 72

    remove_hash_and_rewrite()
    collect()
    assert active_hash[None] == 2
    assert active_ptr[None] == 2
    assert count[None] == 2
    assert coord_sum[None] == 3 * 10 + 2 + 7 * 10 + 2
    assert value_sum[None] == 302 + 72


def _run_hash_snode_nested_hash(compact_child_pool=False):
    x = ti.field(ti.i32)
    count = ti.field(ti.i32, shape=())
    coord_sum = ti.field(ti.i32, shape=())
    value_sum = ti.field(ti.i32, shape=())
    active_outer = ti.field(ti.i32, shape=())
    active_inner = ti.field(ti.i32, shape=())

    if compact_child_pool:
        outer = ti.root.hash(ti.i, 64, expected_active=4, hash_load_factor=0.5)
    else:
        outer = ti.root.hash(ti.i, 64, capacity=16)
    inner = outer.hash(ti.j, 64, capacity=8)
    inner.place(x)

    @ti.kernel
    def write_initial():
        x[3, 1] = 31
        x[3, 2] = 32
        x[7, 5] = 75
        x[11, 0] = 110
        ti.activate(inner, [11, 6])

    @ti.kernel
    def remove_one_inner():
        ti.deactivate(inner, [3, 2])

    @ti.kernel
    def remove_outer_and_rewrite():
        ti.deactivate(outer, [3])
        x[3, 4] = 304

    @ti.kernel
    def collect():
        count[None] = 0
        coord_sum[None] = 0
        value_sum[None] = 0
        active_outer[None] = 0
        active_inner[None] = 0
        active_outer[None] += ti.is_active(outer, [3])
        active_outer[None] += ti.is_active(outer, [7])
        active_outer[None] += ti.is_active(outer, [11])
        active_inner[None] += ti.is_active(inner, [3, 1])
        active_inner[None] += ti.is_active(inner, [3, 2])
        active_inner[None] += ti.is_active(inner, [3, 4])
        active_inner[None] += ti.is_active(inner, [7, 5])
        active_inner[None] += ti.is_active(inner, [11, 0])
        active_inner[None] += ti.is_active(inner, [11, 6])
        for i, j in x:
            count[None] += 1
            coord_sum[None] += i * 10 + j
            value_sum[None] += x[i, j]

    write_initial()
    remove_one_inner()
    collect()
    assert active_outer[None] == 3
    assert active_inner[None] == 4
    assert count[None] == 4
    assert (
        coord_sum[None]
        == 3 * 10 + 1 + 7 * 10 + 5 + 11 * 10 + 0 + 11 * 10 + 6
    )
    assert value_sum[None] == 31 + 75 + 110

    remove_outer_and_rewrite()
    collect()
    assert active_outer[None] == 3
    assert active_inner[None] == 4
    assert count[None] == 4
    assert (
        coord_sum[None]
        == 3 * 10 + 4 + 7 * 10 + 5 + 11 * 10 + 0 + 11 * 10 + 6
    )
    assert value_sum[None] == 304 + 75 + 110


def _run_hash_snode_nested_hash_compact_child_pool_overflow():
    x = ti.field(ti.i32)

    outer = ti.root.hash(ti.i, 64, expected_active=2, hash_load_factor=0.5)
    inner = outer.hash(ti.j, 64, capacity=4)
    inner.place(x)

    @ti.kernel
    def write_too_many_parent_keys():
        x[3, 1] = 31
        x[7, 1] = 71
        x[11, 1] = 111

    with pytest.raises(
        Exception,
        match="Hash SNode compact child pool overflow|Hash SNode table overflow",
    ):
        write_too_many_parent_keys()
        ti.sync()


def _run_hash_snode_hash_under_pointer():
    x = ti.field(ti.i32)
    count = ti.field(ti.i32, shape=())
    coord_sum = ti.field(ti.i32, shape=())
    value_sum = ti.field(ti.i32, shape=())
    active_ptr = ti.field(ti.i32, shape=())
    active_hash = ti.field(ti.i32, shape=())

    p = ti.root.pointer(ti.i, 8)
    h = p.hash(ti.j, 64, capacity=8)
    h.place(x)

    @ti.kernel
    def write_initial():
        x[3, 5] = 35
        x[3, 6] = 36
        x[6, 2] = 62

    @ti.kernel
    def remove_one_hash_key():
        ti.deactivate(h, [3, 5])

    @ti.kernel
    def remove_pointer_and_rewrite():
        ti.deactivate(p, [3])
        x[3, 1] = 301

    @ti.kernel
    def collect():
        count[None] = 0
        coord_sum[None] = 0
        value_sum[None] = 0
        active_ptr[None] = 0
        active_hash[None] = 0
        active_ptr[None] += ti.is_active(p, [3])
        active_ptr[None] += ti.is_active(p, [6])
        active_hash[None] += ti.is_active(h, [3, 1])
        active_hash[None] += ti.is_active(h, [3, 5])
        active_hash[None] += ti.is_active(h, [3, 6])
        active_hash[None] += ti.is_active(h, [6, 2])
        for i, j in x:
            count[None] += 1
            coord_sum[None] += i * 10 + j
            value_sum[None] += x[i, j]

    write_initial()
    remove_one_hash_key()
    collect()
    assert active_ptr[None] == 2
    assert active_hash[None] == 2
    assert count[None] == 2
    assert coord_sum[None] == 3 * 10 + 6 + 6 * 10 + 2
    assert value_sum[None] == 36 + 62

    remove_pointer_and_rewrite()
    collect()
    assert active_ptr[None] == 2
    assert active_hash[None] == 2
    assert count[None] == 2
    assert coord_sum[None] == 3 * 10 + 1 + 6 * 10 + 2
    assert value_sum[None] == 301 + 62


def _run_hash_snode_hash_under_dynamic():
    x = ti.field(ti.i32)
    count = ti.field(ti.i32, shape=())
    coord_sum = ti.field(ti.i32, shape=())
    value_sum = ti.field(ti.i32, shape=())
    len_root = ti.field(ti.i32, shape=())
    active_hash = ti.field(ti.i32, shape=())

    d = ti.root.dynamic(ti.i, 8, chunk_size=4)
    h = d.hash(ti.j, 64, capacity=8)
    h.place(x)

    @ti.kernel
    def write_initial():
        x[3, 5] = 35
        x[3, 6] = 36
        x[7, 1] = 71

    @ti.kernel
    def remove_one_hash_key():
        ti.deactivate(h, [3, 5])

    @ti.kernel
    def clear_dynamic_and_rewrite():
        ti.deactivate(d, [])
        x[3, 2] = 302

    @ti.kernel
    def collect():
        count[None] = 0
        coord_sum[None] = 0
        value_sum[None] = 0
        len_root[None] = ti.length(d, [])
        active_hash[None] = 0
        active_hash[None] += ti.is_active(h, [3, 2])
        active_hash[None] += ti.is_active(h, [3, 5])
        active_hash[None] += ti.is_active(h, [3, 6])
        active_hash[None] += ti.is_active(h, [7, 1])
        for i, j in x:
            count[None] += 1
            coord_sum[None] += i * 10 + j
            value_sum[None] += x[i, j]

    write_initial()
    remove_one_hash_key()
    collect()
    assert len_root[None] == 8
    assert active_hash[None] == 2
    assert count[None] == 2
    assert coord_sum[None] == 3 * 10 + 6 + 7 * 10 + 1
    assert value_sum[None] == 36 + 71

    clear_dynamic_and_rewrite()
    collect()
    assert len_root[None] == 4
    assert active_hash[None] == 1
    assert count[None] == 1
    assert coord_sum[None] == 3 * 10 + 2
    assert value_sum[None] == 302


def _run_hash_snode_runtime_probe_stats():
    from taichi_forge.lang import impl

    capacity = 8
    keys = _collision_keys(capacity, 4)
    x = ti.field(ti.i32)
    total = ti.field(ti.i32, shape=())

    h = ti.root.hash(ti.i, 4096, capacity=capacity)
    h.place(x)

    @ti.kernel
    def write_colliding_keys():
        for p in ti.static(range(4)):
            x[keys[p]] = p + 1

    @ti.kernel
    def read_colliding_keys():
        total[None] = 0
        for p in ti.static(range(4)):
            total[None] += x[keys[p]]

    write_colliding_keys()
    ti.sync()
    prog = impl.get_runtime().prog

    prog.reset_hash_snode_probe_stats()
    write_colliding_keys()
    ti.sync()
    insert_stats = dict(prog.get_hash_snode_probe_stats())
    assert insert_stats["insert_count"] >= 4
    assert insert_stats["insert_total"] >= insert_stats["insert_count"]
    assert insert_stats["insert_max"] >= 2
    assert insert_stats["lookup_count"] == 0
    assert insert_stats["lookup_total"] == 0
    assert insert_stats["lookup_max"] == 0

    prog.reset_hash_snode_probe_stats()
    read_colliding_keys()
    ti.sync()
    lookup_stats = dict(prog.get_hash_snode_probe_stats())
    assert total[None] == 1 + 2 + 3 + 4
    assert lookup_stats["lookup_count"] >= 4
    assert lookup_stats["lookup_total"] >= lookup_stats["lookup_count"]
    assert lookup_stats["lookup_max"] >= 2


def _run_hash_snode_runtime_probe_stats_default_off():
    from taichi_forge.lang import impl

    capacity = 8
    keys = _collision_keys(capacity, 4)
    x = ti.field(ti.i32)

    ti.root.hash(ti.i, 4096, capacity=capacity).place(x)

    @ti.kernel
    def write_colliding_keys():
        for p in ti.static(range(4)):
            x[keys[p]] = p + 1

    prog = impl.get_runtime().prog
    prog.reset_hash_snode_probe_stats()
    write_colliding_keys()
    ti.sync()
    stats = dict(prog.get_hash_snode_probe_stats())
    assert stats["insert_count"] == 0
    assert stats["insert_total"] == 0
    assert stats["insert_max"] == 0
    assert stats["lookup_count"] == 0
    assert stats["lookup_total"] == 0
    assert stats["lookup_max"] == 0


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


@pytest.mark.run_in_serial
@test_utils.test(
    arch=[ti.cpu, ti.cuda],
    require=ti.extension.sparse,
    hash_snode_experimental=True,
    offline_cache=False,
    cuda_sparse_pool_auto_size=True,
)
def test_hash_snode_runtime_error_checks_are_per_kernel(monkeypatch):
    arch = impl.current_cfg().arch
    ti.reset()
    monkeypatch.setenv("TI_DEBUG_ORDINARY_LAUNCH_ATTRIBUTION", "1")
    ti.init(
        arch=arch,
        hash_snode_experimental=True,
        offline_cache=False,
        cuda_sparse_pool_auto_size=True,
    )

    dense = ti.field(ti.i32, shape=2)
    hashed = ti.field(ti.i32)
    ti.root.hash(ti.i, 16, capacity=8).place(hashed)

    @ti.kernel
    def activate_hash():
        hashed[3] = 7

    @ti.kernel
    def dense_only():
        dense[0] += 1

    @ti.kernel
    def read_hash_only():
        dense[1] = hashed[3]

    # Compile all three capability classes before measuring launch decisions.
    activate_hash()
    dense_only()
    read_hash_only()
    ti.sync()

    program = impl.get_runtime().prog
    program._debug_reset_ordinary_launch_attribution()
    for _ in range(8):
        dense_only()
    ti.sync()
    stats = dict(program._debug_ordinary_launch_attribution())
    assert stats["runtime_error_checks"] == 0
    assert stats["runtime_error_check_elisions"] == 8

    program._debug_reset_ordinary_launch_attribution()
    read_hash_only()
    ti.sync()
    stats = dict(program._debug_ordinary_launch_attribution())
    assert stats["runtime_error_checks"] == 0
    assert stats["runtime_error_check_elisions"] == 1
    assert dense[1] == 7

    program._debug_reset_ordinary_launch_attribution()
    activate_hash()
    stats = dict(program._debug_ordinary_launch_attribution())
    assert stats["runtime_error_checks"] == 1
    assert stats["runtime_error_check_elisions"] == 0


@test_utils.test(
    arch=ti.cpu,
    require=ti.extension.sparse,
    hash_snode_experimental=True,
    offline_cache=False,
)
def test_hash_snode_cpu_explicit_activate_and_deactivate_all():
    _run_hash_snode_explicit_activate_and_deactivate_all()


@test_utils.test(
    arch=ti.cpu,
    require=ti.extension.sparse,
    hash_snode_experimental=True,
    offline_cache=False,
)
def test_hash_snode_cpu_tombstone_reuse():
    _run_hash_snode_tombstone_reuse()


@test_utils.test(
    arch=ti.cpu,
    require=ti.extension.sparse,
    hash_snode_experimental=True,
    offline_cache=False,
)
def test_hash_snode_cpu_concurrent_same_key_activation():
    _run_hash_snode_concurrent_same_key_activation()


@test_utils.test(
    arch=ti.cpu,
    require=ti.extension.sparse,
    hash_snode_experimental=True,
    offline_cache=False,
)
def test_hash_snode_cpu_collision_probe_chain_reuse():
    _run_hash_snode_collision_probe_chain_reuse()


@test_utils.test(
    arch=ti.cpu,
    require=ti.extension.sparse,
    hash_snode_experimental=True,
    hash_snode_active_list=True,
    offline_cache=False,
)
def test_hash_snode_cpu_active_list_pure_insert():
    _run_hash_snode_active_list_pure_insert()


@test_utils.test(
    arch=ti.cpu,
    require=ti.extension.sparse,
    hash_snode_experimental=True,
    hash_snode_active_list=True,
    offline_cache=False,
)
def test_hash_snode_cpu_active_list_churn_fallback():
    _run_hash_snode_active_list_churn_fallback()


@test_utils.test(
    arch=ti.cpu,
    require=ti.extension.sparse,
    hash_snode_experimental=True,
    hash_snode_active_list=True,
    offline_cache=False,
)
def test_hash_snode_cpu_active_list_reuse_all_tombstones():
    _run_hash_snode_active_list_reuse_all_tombstones()


@test_utils.test(
    arch=ti.cpu,
    require=ti.extension.sparse,
    hash_snode_experimental=True,
    hash_snode_active_list=True,
    offline_cache=False,
)
def test_hash_snode_cpu_active_list_concurrent_same_key_activation():
    _run_hash_snode_concurrent_same_key_activation()


@test_utils.test(
    arch=ti.cpu,
    require=ti.extension.sparse,
    hash_snode_experimental=True,
    offline_cache=False,
)
def test_hash_snode_cpu_bitmasked_child():
    _run_hash_snode_bitmasked_child()


@test_utils.test(
    arch=ti.cpu,
    require=ti.extension.sparse,
    hash_snode_experimental=True,
    offline_cache=False,
)
def test_hash_snode_cpu_dynamic_child():
    _run_hash_snode_dynamic_child()


@test_utils.test(
    arch=ti.cpu,
    require=ti.extension.sparse,
    hash_snode_experimental=True,
    offline_cache=False,
)
def test_hash_snode_cpu_pointer_child():
    _run_hash_snode_pointer_child()


@test_utils.test(
    arch=ti.cpu,
    require=ti.extension.sparse,
    hash_snode_experimental=True,
    offline_cache=False,
)
def test_hash_snode_cpu_nested_hash():
    _run_hash_snode_nested_hash()


@test_utils.test(
    arch=ti.cpu,
    require=ti.extension.sparse,
    hash_snode_experimental=True,
    hash_snode_compact_child_pool=True,
    offline_cache=False,
)
def test_hash_snode_cpu_nested_hash_compact_child_pool():
    _run_hash_snode_nested_hash(compact_child_pool=True)


@test_utils.test(
    arch=ti.cpu,
    require=ti.extension.sparse,
    hash_snode_experimental=True,
    hash_snode_compact_child_pool=True,
    offline_cache=False,
)
def test_hash_snode_cpu_nested_hash_compact_child_pool_overflow():
    _run_hash_snode_nested_hash_compact_child_pool_overflow()


@test_utils.test(
    arch=ti.cpu,
    require=ti.extension.sparse,
    hash_snode_experimental=True,
    offline_cache=False,
)
def test_hash_snode_cpu_hash_under_pointer():
    _run_hash_snode_hash_under_pointer()


@test_utils.test(
    arch=ti.cpu,
    require=ti.extension.sparse,
    hash_snode_experimental=True,
    offline_cache=False,
)
def test_hash_snode_cpu_hash_under_dynamic():
    _run_hash_snode_hash_under_dynamic()


@test_utils.test(
    arch=ti.cpu,
    require=ti.extension.sparse,
    hash_snode_experimental=True,
    hash_snode_diagnostics=True,
    offline_cache=False,
)
def test_hash_snode_cpu_runtime_probe_stats():
    _run_hash_snode_runtime_probe_stats()


@test_utils.test(
    arch=ti.cpu,
    require=ti.extension.sparse,
    hash_snode_experimental=True,
    offline_cache=False,
)
def test_hash_snode_cpu_runtime_probe_stats_default_off():
    _run_hash_snode_runtime_probe_stats_default_off()


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
    arch=ti.cuda,
    require=ti.extension.sparse,
    hash_snode_experimental=True,
    offline_cache=False,
    cuda_sparse_pool_auto_size=True,
)
def test_hash_snode_cuda_explicit_activate_and_deactivate_all():
    _run_hash_snode_explicit_activate_and_deactivate_all()


@test_utils.test(
    arch=ti.cuda,
    require=ti.extension.sparse,
    hash_snode_experimental=True,
    offline_cache=False,
    cuda_sparse_pool_auto_size=True,
)
def test_hash_snode_cuda_tombstone_reuse():
    _run_hash_snode_tombstone_reuse()


@test_utils.test(
    arch=ti.cuda,
    require=ti.extension.sparse,
    hash_snode_experimental=True,
    offline_cache=False,
    cuda_sparse_pool_auto_size=True,
)
def test_hash_snode_cuda_concurrent_same_key_activation():
    _run_hash_snode_concurrent_same_key_activation()


@test_utils.test(
    arch=ti.cuda,
    require=ti.extension.sparse,
    hash_snode_experimental=True,
    offline_cache=False,
    cuda_sparse_pool_auto_size=True,
)
def test_hash_snode_cuda_collision_probe_chain_reuse():
    _run_hash_snode_collision_probe_chain_reuse()


@test_utils.test(
    arch=ti.cuda,
    require=ti.extension.sparse,
    hash_snode_experimental=True,
    hash_snode_active_list=True,
    offline_cache=False,
    cuda_sparse_pool_auto_size=True,
)
def test_hash_snode_cuda_active_list_pure_insert():
    _run_hash_snode_active_list_pure_insert()


@test_utils.test(
    arch=ti.cuda,
    require=ti.extension.sparse,
    hash_snode_experimental=True,
    hash_snode_active_list=True,
    offline_cache=False,
    cuda_sparse_pool_auto_size=True,
)
def test_hash_snode_cuda_active_list_churn_fallback():
    _run_hash_snode_active_list_churn_fallback()


@test_utils.test(
    arch=ti.cuda,
    require=ti.extension.sparse,
    hash_snode_experimental=True,
    hash_snode_active_list=True,
    offline_cache=False,
    cuda_sparse_pool_auto_size=True,
)
def test_hash_snode_cuda_active_list_reuse_all_tombstones():
    _run_hash_snode_active_list_reuse_all_tombstones()


@test_utils.test(
    arch=ti.cuda,
    require=ti.extension.sparse,
    hash_snode_experimental=True,
    hash_snode_active_list=True,
    offline_cache=False,
    cuda_sparse_pool_auto_size=True,
)
def test_hash_snode_cuda_active_list_concurrent_same_key_activation():
    _run_hash_snode_concurrent_same_key_activation()


@test_utils.test(
    arch=ti.cuda,
    require=ti.extension.sparse,
    hash_snode_experimental=True,
    offline_cache=False,
    cuda_sparse_pool_auto_size=True,
)
def test_hash_snode_cuda_bitmasked_child():
    _run_hash_snode_bitmasked_child()


@test_utils.test(
    arch=ti.cuda,
    require=ti.extension.sparse,
    hash_snode_experimental=True,
    offline_cache=False,
    cuda_sparse_pool_auto_size=True,
)
def test_hash_snode_cuda_dynamic_child():
    _run_hash_snode_dynamic_child()


@test_utils.test(
    arch=ti.cuda,
    require=ti.extension.sparse,
    hash_snode_experimental=True,
    offline_cache=False,
    cuda_sparse_pool_auto_size=True,
)
def test_hash_snode_cuda_pointer_child():
    _run_hash_snode_pointer_child()


@test_utils.test(
    arch=ti.cuda,
    require=ti.extension.sparse,
    hash_snode_experimental=True,
    offline_cache=False,
    cuda_sparse_pool_auto_size=True,
)
def test_hash_snode_cuda_nested_hash():
    _run_hash_snode_nested_hash()


@test_utils.test(
    arch=ti.cuda,
    require=ti.extension.sparse,
    hash_snode_experimental=True,
    hash_snode_compact_child_pool=True,
    offline_cache=False,
    cuda_sparse_pool_auto_size=True,
)
def test_hash_snode_cuda_nested_hash_compact_child_pool():
    _run_hash_snode_nested_hash(compact_child_pool=True)


@test_utils.test(
    arch=ti.cuda,
    require=ti.extension.sparse,
    hash_snode_experimental=True,
    hash_snode_compact_child_pool=True,
    offline_cache=False,
    cuda_sparse_pool_auto_size=True,
)
def test_hash_snode_cuda_nested_hash_compact_child_pool_overflow():
    _run_hash_snode_nested_hash_compact_child_pool_overflow()


@test_utils.test(
    arch=ti.cuda,
    require=ti.extension.sparse,
    hash_snode_experimental=True,
    hash_snode_active_list=True,
    offline_cache=False,
    cuda_sparse_pool_auto_size=True,
)
def test_hash_snode_cuda_nested_hash_active_list():
    _run_hash_snode_nested_hash()


@test_utils.test(
    arch=ti.cuda,
    require=ti.extension.sparse,
    hash_snode_experimental=True,
    offline_cache=False,
    cuda_sparse_pool_auto_size=True,
)
def test_hash_snode_cuda_hash_under_pointer():
    _run_hash_snode_hash_under_pointer()


@test_utils.test(
    arch=ti.cuda,
    require=ti.extension.sparse,
    hash_snode_experimental=True,
    offline_cache=False,
    cuda_sparse_pool_auto_size=True,
)
def test_hash_snode_cuda_hash_under_dynamic():
    _run_hash_snode_hash_under_dynamic()


@test_utils.test(
    arch=ti.cuda,
    require=ti.extension.sparse,
    hash_snode_experimental=True,
    hash_snode_diagnostics=True,
    offline_cache=False,
    cuda_sparse_pool_auto_size=True,
)
def test_hash_snode_cuda_runtime_probe_stats():
    _run_hash_snode_runtime_probe_stats()


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


@test_utils.test(
    arch=ti.vulkan,
    hash_snode_experimental=True,
    vulkan_sparse_experimental=True,
    offline_cache=False,
    vulkan_listgen_dynamic_size=True,
)
def test_hash_snode_vulkan_explicit_activate_and_deactivate_all():
    _run_hash_snode_explicit_activate_and_deactivate_all()


@test_utils.test(
    arch=ti.vulkan,
    hash_snode_experimental=True,
    vulkan_sparse_experimental=True,
    offline_cache=False,
    vulkan_listgen_dynamic_size=True,
)
def test_hash_snode_vulkan_tombstone_reuse():
    _run_hash_snode_tombstone_reuse()


@test_utils.test(
    arch=ti.vulkan,
    hash_snode_experimental=True,
    vulkan_sparse_experimental=True,
    offline_cache=False,
    vulkan_listgen_dynamic_size=True,
)
def test_hash_snode_vulkan_concurrent_same_key_activation():
    _run_hash_snode_concurrent_same_key_activation()


@test_utils.test(
    arch=ti.vulkan,
    hash_snode_experimental=True,
    vulkan_sparse_experimental=True,
    offline_cache=False,
    vulkan_listgen_dynamic_size=True,
)
def test_hash_snode_vulkan_collision_probe_chain_reuse():
    _run_hash_snode_collision_probe_chain_reuse()


@test_utils.test(
    arch=ti.vulkan,
    hash_snode_experimental=True,
    hash_snode_active_list=True,
    vulkan_sparse_experimental=True,
    offline_cache=False,
    vulkan_listgen_dynamic_size=True,
)
def test_hash_snode_vulkan_active_list_pure_insert():
    _run_hash_snode_active_list_pure_insert()


@test_utils.test(
    arch=ti.vulkan,
    hash_snode_experimental=True,
    hash_snode_active_list=True,
    vulkan_sparse_experimental=True,
    offline_cache=False,
    vulkan_listgen_dynamic_size=True,
)
def test_hash_snode_vulkan_active_list_churn_fallback():
    _run_hash_snode_active_list_churn_fallback()


@test_utils.test(
    arch=ti.vulkan,
    hash_snode_experimental=True,
    hash_snode_active_list=True,
    vulkan_sparse_experimental=True,
    offline_cache=False,
    vulkan_listgen_dynamic_size=True,
)
def test_hash_snode_vulkan_active_list_reuse_all_tombstones():
    _run_hash_snode_active_list_reuse_all_tombstones()


@test_utils.test(
    arch=ti.vulkan,
    hash_snode_experimental=True,
    hash_snode_active_list=True,
    vulkan_sparse_experimental=True,
    offline_cache=False,
    vulkan_listgen_dynamic_size=True,
)
def test_hash_snode_vulkan_active_list_concurrent_same_key_activation():
    _run_hash_snode_concurrent_same_key_activation()


@test_utils.test(
    arch=ti.vulkan,
    hash_snode_experimental=True,
    vulkan_sparse_experimental=True,
    offline_cache=False,
    vulkan_listgen_dynamic_size=True,
)
def test_hash_snode_vulkan_bitmasked_child():
    _run_hash_snode_bitmasked_child()


@test_utils.test(
    arch=ti.vulkan,
    hash_snode_experimental=True,
    vulkan_sparse_experimental=True,
    offline_cache=False,
    vulkan_listgen_dynamic_size=True,
)
def test_hash_snode_vulkan_dynamic_child():
    _run_hash_snode_dynamic_child()


@test_utils.test(
    arch=ti.vulkan,
    hash_snode_experimental=True,
    vulkan_sparse_experimental=True,
    offline_cache=False,
    vulkan_listgen_dynamic_size=True,
)
def test_hash_snode_vulkan_pointer_child():
    _run_hash_snode_pointer_child()


@test_utils.test(
    arch=ti.vulkan,
    hash_snode_experimental=True,
    vulkan_sparse_experimental=True,
    offline_cache=False,
    vulkan_listgen_dynamic_size=True,
)
def test_hash_snode_vulkan_nested_hash():
    _run_hash_snode_nested_hash()


@test_utils.test(
    arch=ti.vulkan,
    hash_snode_experimental=True,
    hash_snode_compact_child_pool=True,
    vulkan_sparse_experimental=True,
    offline_cache=False,
    vulkan_listgen_dynamic_size=True,
)
def test_hash_snode_vulkan_nested_hash_compact_child_pool():
    _run_hash_snode_nested_hash(compact_child_pool=True)


@test_utils.test(
    arch=ti.vulkan,
    hash_snode_experimental=True,
    hash_snode_compact_child_pool=True,
    vulkan_sparse_experimental=True,
    offline_cache=False,
    vulkan_listgen_dynamic_size=True,
)
def test_hash_snode_vulkan_nested_hash_compact_child_pool_overflow():
    _run_hash_snode_nested_hash_compact_child_pool_overflow()


@test_utils.test(
    arch=ti.vulkan,
    hash_snode_experimental=True,
    vulkan_sparse_experimental=True,
    offline_cache=False,
    vulkan_listgen_dynamic_size=True,
)
def test_hash_snode_vulkan_hash_under_pointer():
    _run_hash_snode_hash_under_pointer()


@test_utils.test(
    arch=ti.vulkan,
    hash_snode_experimental=True,
    vulkan_sparse_experimental=True,
    offline_cache=False,
    vulkan_listgen_dynamic_size=True,
)
def test_hash_snode_vulkan_hash_under_dynamic():
    _run_hash_snode_hash_under_dynamic()


@test_utils.test(
    arch=ti.vulkan,
    hash_snode_experimental=True,
    vulkan_sparse_experimental=True,
    offline_cache=False,
    vulkan_listgen_dynamic_size=True,
)
def test_hash_snode_vulkan_nonroot_overflow_raises_on_sync():
    x = ti.field(ti.i32)

    p = ti.root.pointer(ti.i, 2)
    h = p.hash(ti.j, 8, capacity=2)
    h.place(x)

    @ti.kernel
    def write_too_many():
        x[0, 0] = 1
        x[0, 1] = 2
        x[0, 2] = 3

    write_too_many()
    with pytest.raises(Exception, match="Hash SNode table overflow"):
        ti.sync()
