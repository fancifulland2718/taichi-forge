from dataclasses import replace

import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils
from taichi_forge.lang import impl
from taichi_forge.lang._storage_view import (
    StorageRequirement,
    _flatten_storage_to_scalar_vector,
    analyze_storage_alias,
    describe_storage,
    qualify_storage,
    shadow_validate_dense_field_descriptor,
    shadow_validate_primitive_view,
    validate_storage_owner,
)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_storage_view_describes_existing_dense_storage_without_copy():
    scalar_array = ti.ndarray(ti.f32, shape=(4, 3))
    vector_array = ti.Vector.ndarray(3, ti.f32, shape=4)
    payload = ti.types.struct(
        scalar=ti.f32,
        vector=ti.types.vector(2, ti.f32),
    )
    struct_array = ti.ndarray(payload, shape=4)
    scalar_member = struct_array.field("scalar")
    vector_member = struct_array.field("vector")
    scalar_field = ti.field(ti.f32, shape=(4, 3))
    vector_field = ti.Vector.field(3, ti.f32, shape=4)

    program = impl.get_runtime().prog
    resources_before = program._debug_ndarray_resource_stats()
    scalar_array_view = describe_storage(scalar_array)
    vector_array_view = describe_storage(vector_array)
    scalar_member_view = describe_storage(scalar_member)
    vector_member_view = describe_storage(vector_member)
    scalar_field_view = describe_storage(scalar_field)
    vector_field_view = describe_storage(vector_field)
    resources_after = program._debug_ndarray_resource_stats()
    assert resources_after["live"] == resources_before["live"]
    assert resources_after["created_total"] == resources_before["created_total"]

    for description in (
        scalar_array_view,
        vector_array_view,
        scalar_member_view,
        vector_member_view,
        scalar_field_view,
        vector_field_view,
    ):
        assert description.supported
        assert validate_storage_owner(description) == "kNone"

    assert tuple(scalar_array_view.descriptor.index_shape) == (4, 3)
    assert tuple(scalar_array_view.descriptor.element_shape) == ()
    assert scalar_array_view.properties["compact_contiguous"]
    assert scalar_array_view.properties["ndarray_abi_compatible"]

    assert tuple(vector_array_view.descriptor.index_shape) == (4,)
    assert tuple(vector_array_view.descriptor.element_shape) == (3,)
    assert vector_array_view.properties["array_layout"] == "kAos"
    assert vector_array_view.properties["record_stride"] == 12

    assert not scalar_member_view.properties["compact_contiguous"]
    assert scalar_member_view.properties["single_record_stride_compatible"]
    assert scalar_member_view.properties["record_stride"] == 12
    assert scalar_member_view.descriptor.byte_offset == 0
    assert tuple(vector_member_view.descriptor.element_shape) == (2,)
    assert vector_member_view.descriptor.byte_offset == 4
    assert vector_member_view.properties["record_stride"] == 12

    assert tuple(scalar_field_view.descriptor.index_shape) == (4, 3)
    assert scalar_field_view.descriptor.source_kind == "kDenseScalarField"
    assert scalar_field_view.descriptor.tree_identity is not None
    assert tuple(vector_field_view.descriptor.element_shape) == (3,)
    assert vector_field_view.descriptor.source_kind == "kDensePackedField"
    assert vector_field_view.properties["array_layout"] == "kAos"

    flat_vector_field = _flatten_storage_to_scalar_vector(vector_field_view)
    assert flat_vector_field.supported
    assert tuple(flat_vector_field.descriptor.index_shape) == (12,)
    assert tuple(flat_vector_field.descriptor.element_shape) == ()
    assert flat_vector_field.descriptor.byte_offset == (
        vector_field_view.descriptor.byte_offset
    )
    assert flat_vector_field.descriptor.owner_kind == "kSNodePayload"
    assert flat_vector_field.properties["compact_contiguous"]

    direct_record = qualify_storage(
        scalar_member_view,
        StorageRequirement(
            accept_single_record_stride=True,
            require_unique_mapping=True,
        ),
    )
    assert direct_record["supported"]
    assert direct_record["execution_mode"] == "kDirectAffine"
    assert not direct_record["requires_materialization"]

    readonly = describe_storage(scalar_array, access="readonly")
    writable = qualify_storage(
        readonly,
        StorageRequirement(require_writable=True),
    )
    assert not writable["supported"]
    assert writable["reason"] == "kReadOnlySource"

    assert analyze_storage_alias(scalar_array_view, scalar_array_view) == "kProvenOverlap"
    assert analyze_storage_alias(scalar_array_view, vector_array_view) == "kProvenDisjoint"
    assert analyze_storage_alias(scalar_member_view, vector_member_view) == "kUnknown"


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_storage_view_rejects_sparse_field_without_materializing():
    sparse = ti.field(ti.f32)
    builder = ti.FieldsBuilder()
    builder.pointer(ti.i, 8).place(sparse)
    tree = builder.finalize()
    try:
        description = describe_storage(sparse)
        assert not description.supported
        assert description.failure_reason == "kUnsupportedLayout"
    finally:
        tree.destroy()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_storage_view_ndarray_owner_rejects_reused_registry_slot():
    old = ti.ndarray(ti.f32, shape=8)
    old_description = describe_storage(old)
    assert old_description.supported
    old_identity = old_description.descriptor.resource_identity

    program = impl.get_runtime().prog
    old_native = old.arr
    old._invalidate_runtime()
    program.delete_ndarray(old_native)
    assert validate_storage_owner(old_description) == "kStaleOwner"

    replacement = ti.ndarray(ti.f32, shape=8)
    replacement_description = describe_storage(replacement)
    replacement_identity = replacement_description.descriptor.resource_identity
    assert replacement_identity[:3] == old_identity[:3]
    assert replacement_identity[3] != old_identity[3]
    assert validate_storage_owner(old_description) == "kStaleOwner"
    assert validate_storage_owner(replacement_description) == "kNone"


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_storage_view_snode_owner_tracks_tree_generation():
    old_field = ti.field(ti.f32)
    old_builder = ti.FieldsBuilder()
    old_builder.dense(ti.i, 8).place(old_field)
    old_tree = old_builder.finalize()
    old_tree_id = int(old_tree.ptr.id())
    old_description = describe_storage(old_field)
    assert old_description.supported

    old_tree.destroy()
    assert validate_storage_owner(old_description) == "kStaleOwner"

    replacement_field = ti.field(ti.f32)
    replacement_builder = ti.FieldsBuilder()
    replacement_builder.dense(ti.i, 8).place(replacement_field)
    replacement_tree = replacement_builder.finalize()
    try:
        assert int(replacement_tree.ptr.id()) == old_tree_id
        replacement_description = describe_storage(replacement_field)
        assert replacement_description.supported
        assert validate_storage_owner(old_description) == "kRetiredGeneration"
        assert validate_storage_owner(replacement_description) == "kNone"
    finally:
        replacement_tree.destroy()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_storage_view_shadow_checks_existing_algorithm_descriptors(monkeypatch):
    from taichi_forge.algorithms import _algorithms
    from taichi_forge.lang import _storage_view
    from taichi_forge.linalg import _vector_io

    monkeypatch.setattr(_storage_view, "STORAGE_VIEW_SHADOW_MODE", "error")

    array = ti.Vector.ndarray(3, ti.f32, shape=4)
    primitive = _algorithms._primitive_view_legacy(array)
    description = shadow_validate_primitive_view(array, primitive)
    assert description.supported

    authoritative = _algorithms._primitive_view(array)
    assert authoritative.description is not None
    assert authoritative.description.descriptor.fingerprint == (
        describe_storage(array).descriptor.fingerprint
    )

    field = ti.Vector.field(3, ti.f32, shape=4)
    legacy_field = _vector_io._describe_value_field_legacy(field, "value")
    description = shadow_validate_dense_field_descriptor(field, legacy_field)
    assert description.supported

    wrong_shape = replace(legacy_field, index_shape=(5,))
    with pytest.raises(RuntimeError, match="index_shape"):
        shadow_validate_dense_field_descriptor(field, wrong_shape)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_native_plan_hot_replay_does_not_reparse_storage(monkeypatch):
    from taichi_forge.algorithms import _algorithms

    program = impl.get_runtime().prog
    if not program.cpu_transform_available():
        pytest.skip("CPU native transform is unavailable")

    source = ti.field(ti.i32, shape=32)
    output = ti.field(ti.i32, shape=32)
    source.fill(3)
    workspace = ti.algorithms.TransformWorkspace(max_items=32)
    ti.algorithms.experimental_transform(
        source,
        output,
        scale=2,
        bias=1,
        method="cpu_native",
        workspace=workspace,
    )

    descriptor_builds = 0
    original_describe_storage = _algorithms.describe_storage

    def count_descriptions(value):
        nonlocal descriptor_builds
        descriptor_builds += 1
        return original_describe_storage(value)

    monkeypatch.setattr(_algorithms, "describe_storage", count_descriptions)
    for _ in range(4):
        ti.algorithms.experimental_transform(
            source,
            output,
            scale=2,
            bias=1,
            method="cpu_native",
            workspace=workspace,
        )

    assert descriptor_builds == 0
    assert (output.to_numpy() == 7).all()


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    exclude=[(ti.vulkan, "Darwin")],
    offline_cache=False,
)
def test_ndarray_view_binds_field_and_ndarray_without_temporary_storage():
    @ti.kernel
    def add_bias(values: ti.types.ndarray(dtype=ti.f32, ndim=1), bias: ti.f32):
        for i in values:
            values[i] += bias

    field = ti.field(ti.f32, shape=16)
    array = ti.ndarray(ti.f32, shape=16)
    field.fill(2.0)
    array.fill(5.0)

    program = impl.get_runtime().prog
    before = program._debug_ndarray_resource_stats()
    binding_before = program._debug_dense_storage_binding_stats()
    field_view = ti.experimental.ndarray_view(field)
    array_view = ti.experimental.ndarray_view(array)
    for view in (field_view, array_view):
        qualification = view.runtime_argument.qualification
        assert qualification["describable"]
        assert qualification["bindable"]
        assert qualification["zero_copy_qualified"]
        assert qualification["reason"] == "kNone"
        assert view.runtime_argument.stable_signature != 0
    after_describe = program._debug_ndarray_resource_stats()
    assert after_describe["live"] == before["live"]
    assert after_describe["created_total"] == before["created_total"]

    add_bias(field_view, 3.0)
    add_bias(array_view, 7.0)
    assert (field.to_numpy() == 5.0).all()
    assert (array.to_numpy() == 12.0).all()

    after_launch = program._debug_ndarray_resource_stats()
    assert after_launch["live"] == before["live"]
    assert after_launch["created_total"] == before["created_total"]
    binding_after = program._debug_dense_storage_binding_stats()
    assert binding_after["direct_submissions"] == binding_before["direct_submissions"] + 2
    assert binding_after["resolved_bindings"] == binding_before["resolved_bindings"] + 2
    assert binding_after["field_bindings"] == binding_before["field_bindings"] + 1
    assert binding_after["ndarray_bindings"] == binding_before["ndarray_bindings"] + 1
    assert binding_after["temporary_allocations"] == 0
    assert binding_after["temporary_bytes"] == 0


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    exclude=[(ti.vulkan, "Darwin")],
    offline_cache=False,
)
def test_graph_ndarray_prepares_runtime_storage_without_temporary_allocations():
    @ti.kernel
    def increment(values: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in values:
            values[i] += i + 1

    values = ti.ndarray(ti.i32, shape=32)
    values.fill(0)
    symbolic = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "values", ti.i32, ndim=1
    )
    builder = ti.graph.GraphBuilder()
    builder.dispatch(increment, symbolic)
    builder.dispatch(increment, symbolic)
    graph = builder.compile()
    graph.execution_stats()

    program = impl.get_runtime().prog
    before = program._debug_dense_storage_binding_stats()
    resources_before = program._debug_ndarray_resource_stats()
    graph.run({"values": values})
    graph.run({"values": values})
    ti.sync()

    np.testing.assert_array_equal(
        values.to_numpy(), (np.arange(32, dtype=np.int32) + 1) * 4
    )
    after = program._debug_dense_storage_binding_stats()
    resources_after = program._debug_ndarray_resource_stats()
    assert after["resolved_bindings"] >= before["resolved_bindings"] + 2
    assert after["ndarray_bindings"] >= before["ndarray_bindings"] + 2
    assert after["temporary_allocations"] == 0
    assert after["temporary_bytes"] == 0
    assert resources_after["created_total"] == resources_before["created_total"]
    assert resources_after["live"] <= resources_before["live"]

    arch = impl.current_cfg().arch
    if arch == ti.cuda:
        stats = graph._graph_stats[0]
        assert stats["captures"] == 1
        assert stats["exact_replays"] == 1
        assert stats["ordinary_fallbacks"] == 0
        assert stats["last_path"] == "cuda_exact_replay"
    elif arch == ti.vulkan:
        stats = graph._graph_stats[0]
        assert stats["records"] == 2
        assert stats["ordinary_fallbacks"] == 0
        assert stats["last_path"] == "vulkan_record"


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    exclude=[(ti.vulkan, "Darwin")],
    offline_cache=False,
)
def test_graph_automatically_normalizes_dense_field_and_view_without_copy():
    @ti.kernel
    def increment(values: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in values:
            values[i] += i + 1

    symbolic = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "values", ti.i32, ndim=1
    )
    builder = ti.graph.GraphBuilder()
    builder.dispatch(increment, symbolic)
    builder.dispatch(increment, symbolic)
    graph = builder.compile()
    graph.execution_stats()

    automatic = ti.field(ti.i32, shape=32)
    explicit_field = ti.field(ti.i32, shape=32)
    explicit = ti.experimental.ndarray_view(explicit_field)
    program = impl.get_runtime().prog
    bindings_before = program._debug_dense_storage_binding_stats()
    resources_before = program._debug_ndarray_resource_stats()

    for runtime_value, field in (
        (automatic, automatic),
        (explicit, explicit_field),
    ):
        for run_index in range(9):
            if run_index == 8:
                ti.sync()
            graph.run({"values": runtime_value})
        ti.sync()
        np.testing.assert_array_equal(
            field.to_numpy(), (np.arange(32, dtype=np.int32) + 1) * 18
        )

    bindings_after = program._debug_dense_storage_binding_stats()
    resources_after = program._debug_ndarray_resource_stats()
    assert bindings_after["field_bindings"] >= (
        bindings_before["field_bindings"] + 18
    )
    assert bindings_after["temporary_allocations"] == 0
    assert bindings_after["temporary_bytes"] == 0
    assert resources_after["created_total"] == resources_before["created_total"]
    assert resources_after["live"] == resources_before["live"]

    arch = impl.current_cfg().arch
    if arch == ti.cuda:
        stats = graph._graph_stats[0]
        assert stats["ordinary_fallbacks"] == 18
        assert stats["captures"] == 0
        assert stats["last_path"] == "ordinary_fallback"
        assert stats["last_fallback_reason"] == "unsupported_arguments"
    elif arch == ti.vulkan:
        stats = graph._graph_stats[0]
        assert stats["records"] == 16
        assert stats["replays"] == 2
        assert stats["ordinary_fallbacks"] == 0
        assert stats["last_path"] == "vulkan_replay"


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    exclude=[(ti.vulkan, "Darwin")],
    offline_cache=False,
)
def test_graph_automatically_normalizes_packed_dense_field():
    vec3 = ti.types.vector(3, ti.f32)

    @ti.kernel
    def update(values: ti.types.ndarray(dtype=vec3, ndim=1)):
        for i in values:
            values[i] += vec3(1.0, 2.0, 3.0)

    symbolic = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "values", vec3, ndim=1
    )
    builder = ti.graph.GraphBuilder()
    builder.dispatch(update, symbolic)
    graph = builder.compile()

    values = ti.Vector.field(3, ti.f32, shape=8)
    values.fill(4.0)
    graph.run({"values": values})
    ti.sync()
    result = values.to_numpy()
    np.testing.assert_array_equal(result[:, 0], np.full(8, 5.0))
    np.testing.assert_array_equal(result[:, 1], np.full(8, 6.0))
    np.testing.assert_array_equal(result[:, 2], np.full(8, 7.0))


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_graph_dense_field_runtime_argument_validates_declared_type_and_rank():
    @ti.kernel
    def touch(values: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in values:
            values[i] += 1

    symbolic = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "values", ti.i32, ndim=1
    )
    builder = ti.graph.GraphBuilder()
    builder.dispatch(touch, symbolic)
    graph = builder.compile()

    wrong_dtype = ti.field(ti.f32, shape=8)
    with pytest.raises(RuntimeError, match="dtype"):
        graph.run({"values": wrong_dtype})

    wrong_rank = ti.field(ti.i32, shape=(2, 4))
    with pytest.raises(RuntimeError, match="field_dim"):
        graph.run({"values": wrong_rank})

    wrong_element = ti.Vector.field(2, ti.i32, shape=8)
    with pytest.raises(RuntimeError, match="element rank"):
        graph.run({"values": wrong_element})


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_graph_dense_field_runtime_argument_rejects_retired_tree():
    @ti.kernel
    def touch(values: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for i in values:
            values[i] += 1.0

    symbolic = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "values", ti.f32, ndim=1
    )
    builder = ti.graph.GraphBuilder()
    builder.dispatch(touch, symbolic)
    graph = builder.compile()

    values = ti.field(ti.f32)
    fields_builder = ti.FieldsBuilder()
    fields_builder.dense(ti.i, 8).place(values)
    tree = fields_builder.finalize()
    graph.run({"values": values})
    tree.destroy()
    with pytest.raises(
        RuntimeError, match="retired SNodeTree|generation"
    ):
        graph.run({"values": values})

@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    exclude=[(ti.vulkan, "Darwin")],
    offline_cache=False,
)
def test_ndarray_view_supports_canonical_packed_compound_fields():
    vec3 = ti.types.vector(3, ti.f32)
    mat2 = ti.types.matrix(2, 2, ti.f32)

    @ti.kernel
    def update_vector(values: ti.types.ndarray(dtype=vec3, ndim=1)):
        for i in values:
            values[i] = values[i] + vec3(1.0, 2.0, 3.0)

    @ti.kernel
    def update_matrix(values: ti.types.ndarray(dtype=mat2, ndim=1)):
        for i in values:
            values[i][0, 0] += 1.0
            values[i][0, 1] += 2.0
            values[i][1, 0] += 3.0
            values[i][1, 1] += 4.0

    field = ti.Vector.field(3, ti.f32, shape=8)
    field.fill(4.0)
    view = ti.experimental.ndarray_view(field)
    update_vector(view)
    result = field.to_numpy()
    assert (result[:, 0] == 5.0).all()
    assert (result[:, 1] == 6.0).all()
    assert (result[:, 2] == 7.0).all()

    matrix_field = ti.Matrix.field(2, 2, ti.f32, shape=8)
    matrix_field.fill(5.0)
    matrix_view = ti.experimental.ndarray_view(matrix_field)
    update_matrix(matrix_view)
    matrix_result = matrix_field.to_numpy()
    assert (matrix_result[:, 0, 0] == 6.0).all()
    assert (matrix_result[:, 0, 1] == 7.0).all()
    assert (matrix_result[:, 1, 0] == 8.0).all()
    assert (matrix_result[:, 1, 1] == 9.0).all()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_ndarray_view_rejects_stale_owners_before_launch():
    @ti.kernel
    def touch(values: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for i in values:
            values[i] += 1.0

    field = ti.field(ti.f32)
    builder = ti.FieldsBuilder()
    builder.dense(ti.i, 8).place(field)
    tree = builder.finalize()
    field_view = ti.experimental.ndarray_view(field)
    touch(field_view)
    tree.destroy()
    with pytest.raises(RuntimeError, match="retired SNodeTree|generation"):
        touch(field_view)

    array = ti.ndarray(ti.f32, shape=8)
    array_view = ti.experimental.ndarray_view(array)
    touch(array_view)
    program = impl.get_runtime().prog
    native = array.arr
    array._invalidate_runtime()
    program.delete_ndarray(native)
    with pytest.raises(RuntimeError, match="stale or retired Ndarray"):
        touch(array_view)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_ndarray_view_rejects_padded_dense_layout_without_materializing():
    lhs = ti.field(ti.f32)
    rhs = ti.field(ti.f32)
    builder = ti.FieldsBuilder()
    builder.dense(ti.i, 8).place(lhs, rhs)
    tree = builder.finalize()
    try:
        with pytest.raises(ValueError, match="zero-copy ndarray view"):
            ti.experimental.ndarray_view(lhs)
    finally:
        tree.destroy()
