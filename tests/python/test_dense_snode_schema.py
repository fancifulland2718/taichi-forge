import json

import pytest

import taichi_forge as ti
from taichi_forge._lib import core
from tests import test_utils


def _group(n):
    x = ti.field(ti.f32)
    vector = ti.Vector.field(3, ti.f64)
    matrix = ti.Matrix.field(2, 2, ti.f32)
    scalar = ti.field(ti.i32)
    builder = ti.FieldsBuilder()
    blocks = [builder.dense(ti.i, n) for _ in range(3)]
    blocks[0].place(x)
    blocks[1].place(vector)
    blocks[2].place(matrix)
    builder.place(scalar)
    schema = core.DenseSNodeSchema.capture(builder.ptr, [(block.ptr, 0, "N") for block in blocks])
    tree = builder.finalize()
    return schema, tree, builder.ptr


def _layout(root):
    # Query actual compiled SNodes, not a second implementation of padding.
    result = []

    def visit(node, path):
        result.append(
            (
                path,
                str(node.type),
                str(node.data_type()),
                tuple(node.get_shape_along_axis(i) for i in range(node.num_active_indices())),
                node.cell_size_bytes,
                node.offset_bytes_in_parent_cell,
            )
        )
        for index in range(node.get_num_ch()):
            visit(node.get_ch(index), path + (index,))

    visit(root, ())
    return result


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_parameterized_schema_uses_normal_snode_layout():
    schema, producer, producer_root = _group(7)
    frozen = schema.to_json()
    program = ti.lang.impl.get_runtime().prog
    producer_layout = _layout(producer_root)
    for n in (37, 113):
        expected_schema, expected, expected_root = _group(n)
        assert expected_schema.to_json() == frozen
        instance = core.DenseSNodeSchema.from_json(frozen).instantiate(program, {"N": n})
        try:
            assert _layout(instance.root()) == _layout(expected_root)
            assert (
                instance.root().get_ch(1).offset_bytes_in_parent_cell
                != producer_root.get_ch(1).offset_bytes_in_parent_cell
            )
            assert _layout(producer_root) == producer_layout
            assert instance.id() != producer.id
        finally:
            instance.destroy_snode_tree(program)
            expected.destroy()
    producer.destroy()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_schema_rejects_unknown_dimensions_and_unqualified_layouts():
    schema, producer, _ = _group(7)
    program = ti.lang.impl.get_runtime().prog
    active = program.get_active_snode_tree_ids()
    for bindings in ({}, {"N": 0}, {"N": -1}, {"N": 7, "unused": 1}):
        with pytest.raises(ValueError):
            schema.instantiate(program, bindings)
        assert program.get_active_snode_tree_ids() == active
    data = json.loads(schema.to_json())
    data["version"] = 99
    with pytest.raises(ValueError, match="version"):
        core.DenseSNodeSchema.from_json(json.dumps(data))
    data = json.loads(schema.to_json())
    # Reopening an earlier sibling would change the stable node projection.
    data["nodes"][-1]["parent"] = 1
    with pytest.raises(ValueError, match="preorder"):
        core.DenseSNodeSchema.from_json(json.dumps(data))
    data = json.loads(schema.to_json())
    data["nodes"][1]["extents"].append({"axis": 1, "constant": 2, "parameter": ""})
    too_large = core.DenseSNodeSchema.from_json(json.dumps(data))
    with pytest.raises(ValueError, match="i32 indexing range"):
        too_large.instantiate(program, {"N": 2**30})
    assert program.get_active_snode_tree_ids() == active
    field = ti.field(ti.i32)
    builder = ti.FieldsBuilder()
    outer = builder.dense(ti.i, 2)
    outer.dense(ti.i, 4).place(field)
    with pytest.raises(ValueError, match="root/dense/place"):
        core.DenseSNodeSchema.capture(builder.ptr, [])
    builder.finalize().destroy()
    producer.destroy()
