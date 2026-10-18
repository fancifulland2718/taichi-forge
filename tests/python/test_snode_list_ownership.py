"""List storage follows the materialized tree, not later compile settings."""

import taichi_forge as ti
import pytest
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False, demote_dense_struct_fors=False)
def test_dense_list_storage_keeps_its_creation_contract():
    cfg = ti.lang.impl.current_cfg()
    program = ti.lang.impl.get_runtime().prog
    trees = []

    def make_tree(demote):
        cfg.demote_dense_struct_fors = demote
        values = ti.field(ti.i32)
        builder = ti.FieldsBuilder()
        builder.dense(ti.i, 4).dense(ti.i, 2).place(values)
        tree = builder.finalize()
        values[0] = 7  # Materialize under this tree's original contract.
        trees.append((tree, demote))
        return values, tree

    def memory(tree):
        return dict(program._debug_sparse_snode_tree_stats(tree.id))["memory"]

    try:
        listed, list_tree = make_tree(False)

        @ti.kernel
        def total() -> ti.i32:
            result = 0
            for i in listed:
                result += listed[i]
            return result

        assert total() == 7
        initial = dict(memory(list_tree))
        assert initial["active_list_reserved_bytes"] > 0
        plain, plain_tree = make_tree(True)
        plain_initial = dict(memory(plain_tree))
        assert plain_initial["active_list_reserved_bytes"] == 0
        for demote in (True, False):
            cfg.demote_dense_struct_fors = demote
            for tree, expected in ((list_tree, initial), (plain_tree, plain_initial)):
                current = dict(memory(tree))
                for key in ("active_list_reserved_bytes", "active_list_used_bytes", "runtime_metadata_requested_bytes"):
                    assert current[key] == expected[key], (demote, key, current[key], expected[key])
        # Retire both kinds with the opposite current setting.
        cfg.demote_dense_struct_fors = True
        list_tree.destroy()
        trees.remove((list_tree, False))
        assert plain[0] == 7
        cfg.demote_dense_struct_fors = False
        plain_tree.destroy()
        trees.remove((plain_tree, True))
    finally:
        for tree, demote in trees:
            cfg.demote_dense_struct_fors = demote
            tree.destroy()
        cfg.demote_dense_struct_fors = False


@test_utils.test(arch=ti.cuda, offline_cache=False, demote_dense_struct_fors=False)
def test_dense_list_trees_have_independent_cuda_backing_pools():
    _check_cuda_dense_list_pools(0, 1 << 20)


@test_utils.test(arch=ti.cuda, offline_cache=False, demote_dense_struct_fors=False, cuda_sparse_pool_size_floor_MiB=1)
def test_dense_list_pool_respects_explicit_floor():
    _check_cuda_dense_list_pools(1 << 20, 2 << 20)


@test_utils.test(arch=ti.cuda, offline_cache=False, demote_dense_struct_fors=False, cuda_sparse_per_snode_pool=False)
def test_dense_list_trees_keep_explicit_shared_pool_policy():
    _check_cuda_dense_list_pools(0, 0)


def _check_cuda_dense_list_pools(minimum, maximum):
    trees = []
    fields = []
    for index, shape in enumerate(((4, 2), (8, 8), (2, 4, 8))):
        values = ti.field(ti.i32)
        builder = ti.FieldsBuilder()
        node = builder
        for size in shape:
            node = node.dense(ti.i, size)
        node.place(values)
        tree = builder.finalize()
        trees.append(tree)
        fields.append(values)
        values[0] = 7 + index
        memory = dict(ti.lang.impl.get_runtime().prog._debug_sparse_snode_tree_stats(tree.id))["memory"]
        # These small traversal trees must not reserve a full runtime-sized
        # pool. This is a regression bound for this fixture, not a user limit.
        if maximum:
            assert minimum <= memory["sparse_pool_reserved_bytes"] < maximum
        else:
            assert memory["sparse_pool_reserved_bytes"] == 0  # Shared pool is not tree-owned.

    @ti.kernel
    def total(values: ti.template()) -> ti.i32:
        result = 0
        for i in values:
            result += values[i]
        return result

    for index, values in enumerate(fields):
        assert total(values) == 7 + index
    trees[1].destroy()
    assert total(fields[0]) == 7
    assert total(fields[2]) == 9
    trees[0].destroy()
    trees[2].destroy()


@pytest.mark.parametrize("floor_mib", [0, 1])
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_packed_trees_budget_only_their_list_storage(floor_mib):
    ti.cfg.cuda_sparse_pool_size_floor_MiB = floor_mib
    fields, trees = [], []
    for layout in ("bitpacked", "quant_array"):
        x = ti.field(ti.types.quant.int(7, signed=True))
        builder = ti.FieldsBuilder()
        if layout == "bitpacked":
            packed = ti.BitpackedFields(max_num_bits=32)
            packed.place(x)
            builder.dense(ti.i, 8).place(packed)
        else:
            builder.dense(ti.i, 2).quant_array(ti.i, 4, max_num_bits=32).place(x)
        tree = builder.finalize()
        fields.append(x)
        trees.append(tree)
        x[0], x[7] = 11, -5
        memory = dict(ti.lang.impl.get_runtime().prog._debug_sparse_snode_tree_stats(tree.id))["memory"]
        # Fixture bound, not a user capacity limit. Explicit floors still apply.
        assert floor_mib * (1 << 20) <= memory["sparse_pool_reserved_bytes"] < (floor_mib + 1) * (1 << 20)

    @ti.kernel
    def total(x: ti.template()) -> ti.i32:
        result = 0
        for i in x:
            result += x[i]
        return result

    assert total(fields[0]) == total(fields[1]) == 6
    trees[0].destroy()
    assert total(fields[1]) == 6
    trees[1].destroy()
