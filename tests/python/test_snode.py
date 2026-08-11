import struct

import taichi_forge as ti
from tests import test_utils
from taichi_forge.lang import impl


@ti.kernel
def some_kernel(_: ti.template()): ...


@test_utils.test(cpu_max_num_threads=1)
def test_get_snode_tree_id():
    s = ti.field(int, shape=())
    some_kernel(s)
    assert s.snode._snode_tree_id == 0

    s = ti.field(int, shape=())
    some_kernel(s)
    assert s.snode._snode_tree_id == 1

    s = ti.field(int, shape=())
    some_kernel(s)
    assert s.snode._snode_tree_id == 2


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_global_snode_ids_above_legacy_1024_limit():
    # Each scalar field materialized separately contributes root/dense/place
    # SNodes to the Program-global id space. Keep every field alive so this
    # matches applications that assemble many independent subsystems.
    prefix_fields = []
    for _ in range(340):
        prefix_fields.append(ti.field(ti.i32, shape=1))
        impl.get_runtime().materialize()

    src = ti.field(ti.i32, shape=16)
    dst = ti.field(ti.i32, shape=16)
    # 1024 is already outside the legacy table's valid [0, 1023] range.
    assert dst.snode.ptr.id >= 1024

    @ti.kernel
    def copy():
        for i in dst:
            src[i] = i * 3 + 1
            dst[i] = src[i]

    copy()
    assert (dst.to_numpy() == [i * 3 + 1 for i in range(16)]).all()


@test_utils.test(
    arch=[ti.cpu, ti.cuda],
    require=ti.extension.sparse,
    hash_snode_experimental=True,
    offline_cache=False,
)
def test_single_tree_above_legacy_4096_snode_limit_and_sparse_lifecycle():
    builder = ti.FieldsBuilder()
    storage = builder.dense(ti.i, 1)
    last = None
    # Keep pointer/dynamic/hash state above the former table boundary instead
    # of qualifying only dense places. root + dense + 4090 places + the three
    # sparse node/place pairs = 4098 runtime-addressable SNodes.
    for _ in range(4090):
        last = ti.field(ti.i32)
        storage.place(last)
    pointer_value = ti.field(ti.i32)
    dynamic_value = ti.field(ti.i32)
    hash_value = ti.field(ti.i32)
    pointer = builder.pointer(ti.i, 16)
    pointer.place(pointer_value)
    dynamic = builder.dynamic(ti.i, 16, 8)
    dynamic.place(dynamic_value)
    hash_node = builder.hash(ti.i, 64, capacity=8)
    hash_node.place(hash_value)
    tree = builder.finalize()

    @ti.kernel
    def populate():
        last[0] = 73
        pointer_value[3] = 11
        ti.append(dynamic, [], 13)
        hash_value[5] = 17

    @ti.kernel
    def reduce() -> ti.i32:
        total = last[0]
        for i in pointer_value:
            total += pointer_value[i]
        for i in dynamic_value:
            total += dynamic_value[i]
        for i in hash_value:
            total += hash_value[i]
        return total

    populate()
    assert reduce() == 114
    assert hash_value.snode.ptr.id >= 4096
    memory = dict(
        impl.get_runtime().prog._debug_sparse_snode_tree_stats(tree.id)["memory"]
    )
    assert memory["runtime_state_reserved_bytes"] == 40 + 4098 * 48

    pointer.deactivate_all()
    dynamic.deactivate_all()
    hash_node.deactivate_all()
    assert reduce() == 73

    old_tree_id = tree.id
    tree.destroy()

    # Global diagnostic ids keep increasing, while the runtime directory and
    # per-tree local state safely reuse the destroyed tree slot.
    replacement_builder = ti.FieldsBuilder()
    replacement = ti.field(ti.i32)
    replacement_builder.dense(ti.i, 1).place(replacement)
    replacement_tree = replacement_builder.finalize()
    assert replacement_tree.id == old_tree_id
    assert replacement.snode.ptr.id > last.snode.ptr.id
    replacement[0] = 91
    assert replacement[0] == 91
    replacement_tree.destroy()


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_concurrent_snode_trees_above_legacy_512_limit():
    prog = impl.get_runtime().prog
    impl.get_runtime().materialize()
    baseline = dict(prog._debug_snode_runtime_directory_stats())
    assert baseline["available"]
    trees = []
    fields = []
    try:
        for _ in range(513):
            builder = ti.FieldsBuilder()
            field = ti.field(ti.i32)
            builder.dense(ti.i, 1).place(field)
            trees.append(builder.finalize())
            fields.append(field)

        expanded = dict(prog._debug_snode_runtime_directory_stats())
        assert expanded["capacity"] >= baseline["active_tree_count"] + 513
        assert expanded["capacity"] & (expanded["capacity"] - 1) == 0
        assert expanded["active_tree_count"] == baseline["active_tree_count"] + 513
        assert expanded["reserved_bytes"] == (
            expanded["capacity"] * struct.calcsize("P")
        )
        assert expanded["growth_events"] >= baseline["growth_events"]

        fields[0][0] = 17
        fields[-1][0] = 29
        assert fields[0][0] == 17
        assert fields[-1][0] == 29
    finally:
        for tree in reversed(trees):
            tree.destroy()
    retired = dict(prog._debug_snode_runtime_directory_stats())
    assert retired["active_tree_count"] == baseline["active_tree_count"]
    assert retired["capacity"] == expanded["capacity"]
    assert retired["reserved_bytes"] == expanded["reserved_bytes"]
    assert retired["growth_events"] == expanded["growth_events"]


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_snode_runtime_directory_diagnostic_is_backend_honest():
    impl.get_runtime().materialize()
    stats = dict(impl.get_runtime().prog._debug_snode_runtime_directory_stats())
    assert stats == {
        "available": False,
        "host_visible": False,
        "capacity": 0,
        "active_tree_count": 0,
        "reserved_bytes": 0,
        "growth_events": 0,
    }


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_snode_metadata_retention_is_bounded_by_peak_live_topology():
    prog = impl.get_runtime().prog
    impl.get_runtime().materialize()
    baseline = dict(prog._debug_snode_metadata_stats())

    def churn_once():
        builder = ti.FieldsBuilder()
        value = ti.field(ti.i32)
        builder.dense(ti.i, 4).place(value)
        tree = builder.finalize()
        tree.destroy()

    churn_once()
    warmed = dict(prog._debug_snode_metadata_stats())
    for _ in range(31):
        churn_once()
    final = dict(prog._debug_snode_metadata_stats())

    assert final["tree_slots"] == warmed["tree_slots"]
    assert final["active_tree_count"] == baseline["active_tree_count"]
    assert final["retired_tree_shells"] == warmed["retired_tree_shells"]
    assert final["retired_snode_count"] == warmed["retired_snode_count"]
    assert final["tree_inline_bytes_lower_bound"] == warmed[
        "tree_inline_bytes_lower_bound"
    ]
    assert final["snode_inline_bytes_lower_bound"] == warmed[
        "snode_inline_bytes_lower_bound"
    ]
    assert final["generation_table_bytes"] == warmed["generation_table_bytes"]
    assert final["active_table_bytes"] == warmed["active_table_bytes"]
    assert final["global_snode_ids_issued"] > warmed["global_snode_ids_issued"]
    assert final["logical_bytes_are_lower_bounds"]
