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


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_single_tree_above_legacy_4096_snode_limit_and_reuse():
    builder = ti.FieldsBuilder()
    storage = builder.dense(ti.i, 1)
    last = None
    # root + dense + 4095 places = 4097 runtime-addressable SNodes.
    for _ in range(4095):
        last = ti.field(ti.i32)
        storage.place(last)
    tree = builder.finalize()

    last[0] = 73
    assert last[0] == 73
    assert last.snode.ptr.id >= 4096

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
    trees = []
    fields = []
    try:
        for _ in range(513):
            builder = ti.FieldsBuilder()
            field = ti.field(ti.i32)
            builder.dense(ti.i, 1).place(field)
            trees.append(builder.finalize())
            fields.append(field)

        fields[0][0] = 17
        fields[-1][0] = 29
        assert fields[0][0] == 17
        assert fields[-1][0] == 29
    finally:
        for tree in reversed(trees):
            tree.destroy()
