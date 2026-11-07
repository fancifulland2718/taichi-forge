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


def _tensor_group(n, dofs):
    builder = ti.FieldsBuilder()
    row = builder.dense(ti.i, n)
    prefix = ti.field(ti.u8)
    values = ti.Vector.field(dofs, ti.f32)
    fixed = ti.Vector.field(3, ti.f64)
    row.place(prefix, values, fixed)
    matrix_block = builder.dense(ti.i, n)
    matrix = ti.Matrix.field(2, dofs, ti.f32)
    matrix_block.place(matrix)
    parameters = [(row.ptr, 0, "N"), (matrix_block.ptr, 0, "N")]
    components = lambda field: [var.ptr.snode() for var in field._get_field_members()]
    tensors = [(components(values), [dofs], ["D"]), (components(matrix), [2, dofs], ["", "D"])]
    schema = core.DenseSNodeSchema.capture_tensors(builder.ptr, parameters, tensors)
    tree = builder.finalize()
    return schema, tree, builder.ptr, parameters, tensors


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_tensor_member_dimensions_keep_one_authoritative_layout_recipe():
    schema, producer, root, _, _ = _tensor_group(3, 7)
    frozen = schema.to_json()
    assert json.loads(frozen)["version"] == 2
    program = ti.lang.impl.get_runtime().prog
    baseline = _layout(root)
    for n, dofs in ((5, 1), (2, 19), (7, 65)):
        expected, reference, reference_root, _, _ = _tensor_group(n, dofs)
        assert expected.to_json() == frozen
        materialized = core.DenseSNodeSchema.from_json(frozen).instantiate(program, {"N": n, "D": dofs})
        try:
            assert _layout(materialized.root()) == _layout(reference_root)
            assert _layout(root) == baseline
        finally:
            materialized.destroy_snode_tree(program)
            reference.destroy()
    producer.destroy()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_tensor_recipe_rejects_unproven_components_and_invalid_extents():
    schema, producer, root, parameters, tensors = _tensor_group(3, 7)
    program = ti.lang.impl.get_runtime().prog
    active = program.get_active_snode_tree_ids()
    leaves, shape, names = tensors[0]
    for invalid in (
        [(list(reversed(leaves)), shape, names)],
        [(leaves[:-1], shape, names)],
        [(leaves, shape, ["N"])],
        [tensors[0], tensors[0]],
        [(leaves, [7, 1, 1], ["D", "", ""])],
    ):
        with pytest.raises(ValueError):
            core.DenseSNodeSchema.capture_tensors(root, parameters, invalid)
    for dimensions in (
        {"N": 3},
        {"N": 3, "D": 0},
        {"N": 3, "D": -1},
        {"N": 3, "D": 7, "unknown": 1},
        {"N": 3, "D": 2**31 - 1},
    ):
        with pytest.raises(ValueError):
            schema.instantiate(program, dimensions)
        assert program.get_active_snode_tree_ids() == active
    data = json.loads(schema.to_json())
    data["version"] = 1
    with pytest.raises(ValueError, match="version 2 tensor"):
        core.DenseSNodeSchema.from_json(json.dumps(data))
    data = json.loads(schema.to_json())
    tensor = next(node for node in data["nodes"] if node["type"] == int(core.SNodeType.place) and node["extents"])
    tensor["extents"][0]["axis"] = 1
    with pytest.raises(ValueError, match="consecutive"):
        core.DenseSNodeSchema.from_json(json.dumps(data))
    producer.destroy()


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
    with pytest.raises(ValueError, match="repeated axes"):
        core.DenseSNodeSchema.capture(builder.ptr, [])
    builder.finalize().destroy()
    producer.destroy()
