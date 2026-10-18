"""List storage follows the materialized tree, not later compile settings."""

import taichi_forge as ti
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
