from dataclasses import replace

import pytest

import taichi_forge as ti
from tests import test_utils
from taichi_forge.lang import impl
from taichi_forge.lang._storage_view import (
    StorageRequirement,
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

    field = ti.Vector.field(3, ti.f32, shape=4)
    legacy_field = _vector_io._describe_value_field_legacy(field, "value")
    description = shadow_validate_dense_field_descriptor(field, legacy_field)
    assert description.supported

    wrong_shape = replace(legacy_field, index_shape=(5,))
    with pytest.raises(RuntimeError, match="index_shape"):
        shadow_validate_dense_field_descriptor(field, wrong_shape)
