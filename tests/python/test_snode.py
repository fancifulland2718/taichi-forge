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
def test_snode_id_above_previous_4096_limit():
    # Keep this in one tree so the test exercises the SNode-id-indexed runtime
    # tables without approaching the independent LLVM SNode-tree limit.
    fields = [ti.field(ti.i32) for _ in range(4100)]
    builder = ti.FieldsBuilder()
    dense = builder.dense(ti.i, 1)
    for field in fields:
        dense.place(field)
    tree = builder.finalize()

    target = fields[-1]
    assert target.snode.ptr.id >= 4096

    @ti.kernel
    def write_target():
        target[0] = 37

    write_target()
    assert target[0] == 37
    tree.destroy()
