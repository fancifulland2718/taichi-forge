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


def _soa_tensor_group(n, dofs):
    builder = ti.FieldsBuilder()
    prefix = ti.field(ti.u8)
    builder.dense(ti.i, n).place(prefix)
    vector = ti.Vector.field(dofs, ti.f32)
    matrix = ti.Matrix.field(2, dofs, ti.f64)
    parameters = [(builder.ptr.get_ch(0), 0, "N")]
    tensors = []
    for field, shape, names in (
        (vector, [dofs], ["D"]),
        (matrix, [2, dofs], ["", "D"]),
    ):
        for i in range(field.n):
            for j in range(field.m):
                outer = builder.dense(ti.j, 3)
                inner = outer.dense(ti.i, n)
                inner.place(field.get_scalar_field(i, j))
                parameters.append((inner.ptr, 0, "N"))
        tensors.append(
            ([var.ptr.snode() for var in field._get_field_members()], shape, names)
        )
    schema, members = core.DenseSNodeSchema.capture_tensor_layout_members(
        builder.ptr, parameters, tensors, [True, True]
    )
    tree = builder.finalize()
    return schema, members, tree, builder.ptr, parameters, tensors


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_soa_model_tensor_recipe_reuses_native_dense_branch_layout():
    schema, members, producer, root, parameters, tensors = _soa_tensor_group(2, 1)
    frozen = schema.to_json()
    assert json.loads(frozen)["version"] == 3
    selectors = [
        tuple(dict.fromkeys(members[node.id] for node in t[0])) for t in tensors
    ]
    program = ti.lang.impl.get_runtime().prog
    for n, dofs in ((5, 7), (3, 19), (7, 1)):
        expected, mapping, reference, reference_root, _, definitions = (
            _soa_tensor_group(n, dofs)
        )
        assert expected.to_json() == frozen
        assert [
            tuple(dict.fromkeys(mapping[node.id] for node in t[0])) for t in definitions
        ] == selectors
        instance = core.DenseSNodeSchema.from_json(frozen).instantiate(
            program, {"N": n, "D": dofs}
        )
        try:
            assert _layout(instance.root()) == _layout(reference_root)
        finally:
            instance.destroy_snode_tree(program)
            reference.destroy()
    active = program.get_active_snode_tree_ids()
    for bindings in ({"N": 2, "D": 0}, {"N": 2, "D": 2**31 - 1}):
        with pytest.raises(ValueError):
            schema.instantiate(program, bindings)
        assert program.get_active_snode_tree_ids() == active
    producer.destroy()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_soa_tensor_recipe_rejects_nonmatching_branches_and_nested_repetition():
    schema, _, producer, root, parameters, tensors = _soa_tensor_group(2, 3)
    leaves, shape, names = tensors[0]
    for invalid, layouts in (
        ([(list(reversed(leaves)), shape, names)], [True]),
        ([tensors[0], tensors[0]], [True, True]),
        ([tensors[0]], []),
    ):
        with pytest.raises(ValueError):
            core.DenseSNodeSchema.capture_tensor_layout_members(
                root, parameters, invalid, layouts
            )
    changed = list(parameters)
    node, axis, _ = changed[2]
    changed[2] = (node, axis, "OtherN")
    with pytest.raises(ValueError, match="different index or scalar layouts"):
        core.DenseSNodeSchema.capture_tensor_layout_members(
            root, changed, [tensors[0]], [True]
        )
    data = json.loads(schema.to_json())
    branch_index = next(
        i for i, node in enumerate(data["nodes"]) if node["tensor_extents"]
    )
    data["nodes"][branch_index + 1]["tensor_extents"] = data["nodes"][branch_index][
        "tensor_extents"
    ]
    with pytest.raises(ValueError, match="another tensor repetition"):
        core.DenseSNodeSchema.from_json(json.dumps(data))
    producer.destroy()


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_tensor_member_dimensions_keep_one_authoritative_layout_recipe():
    schema, producer, root, parameters, tensors = _tensor_group(3, 7)
    projected, members = core.DenseSNodeSchema.capture_tensor_members(root, parameters, tensors)
    assert projected.to_json() == schema.to_json()
    selectors = tuple(tuple(dict.fromkeys(members[node.id] for node in tensor[0])) for tensor in tensors)
    frozen = schema.to_json()
    assert json.loads(frozen)["version"] == 2
    program = ti.lang.impl.get_runtime().prog
    baseline = _layout(root)
    for n, dofs in ((5, 1), (2, 19), (7, 65)):
        expected, reference, reference_root, reference_parameters, reference_tensors = _tensor_group(n, dofs)
        assert expected.to_json() == frozen
        _, reference_members = core.DenseSNodeSchema.capture_tensor_members(
            reference_root, reference_parameters, reference_tensors)
        assert tuple(tuple(dict.fromkeys(reference_members[node.id] for node in tensor[0]))
                     for tensor in reference_tensors) == selectors
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
