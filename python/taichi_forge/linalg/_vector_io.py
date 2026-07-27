"""Runtime-bound dense-field vector views for experimental linear algebra.

This module deliberately describes storage only.  It does not assign physical
meaning to field indices or construct active sets.  Supported fields are
converted to the scalar-flat ndarray ABI at an operation boundary and remain
device resident throughout that conversion.
"""

from dataclasses import dataclass
import hashlib
import math
from types import MappingProxyType

import numpy as np

from taichi_forge._lib import core as _ti_core
from taichi_forge.lang import impl
from taichi_forge.lang._ndarray import ScalarNdarray
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.lang._storage_view import (
    _flatten_storage_to_scalar_vector,
    describe_storage,
)
from taichi_forge.lang.field import (
    ScalarField,
    _dense_host_copy_value_type,
    _dense_native_field_layout_supported,
)
from taichi_forge.lang.impl import grouped, static
from taichi_forge.lang.kernel_impl import func, kernel
from taichi_forge.lang.matrix import MatrixField
from taichi_forge.types import f32, f64, i32
from taichi_forge.types import ndarray_type
from taichi_forge.types.annotations import template


_VECTOR_IO_SCHEMA = "taichi_forge.linalg.vector_io.v1"
_SUPPORTED_VALUE_DTYPES = (f32, f64)
_IMPLICIT_VIEW_CACHE_LIMIT = 16
_TRANSFER_PLAN_CACHE_LIMIT = 32


@func
def _scalar_field_load_flat(field: template(), item_index: i32):
    offset = static(
        field.snode.ptr.offset
        if len(field.snode.ptr.offset) != 0
        else [0] * len(field.shape)
    )
    if static(len(field.shape) == 1):
        return field[item_index + static(offset[0])]
    if static(len(field.shape) == 2):
        j = item_index % static(field.shape[1])
        i = item_index // static(field.shape[1])
        return field[
            i + static(offset[0]),
            j + static(offset[1]),
        ]
    k = item_index % static(field.shape[2])
    prefix = item_index // static(field.shape[2])
    j = prefix % static(field.shape[1])
    i = prefix // static(field.shape[1])
    return field[
        i + static(offset[0]),
        j + static(offset[1]),
        k + static(offset[2]),
    ]


@func
def _scalar_field_store_flat(
    field: template(), item_index: i32, value: template()
):
    offset = static(
        field.snode.ptr.offset
        if len(field.snode.ptr.offset) != 0
        else [0] * len(field.shape)
    )
    if static(len(field.shape) == 1):
        field[item_index + static(offset[0])] = value
    elif static(len(field.shape) == 2):
        j = item_index % static(field.shape[1])
        i = item_index // static(field.shape[1])
        field[
            i + static(offset[0]),
            j + static(offset[1]),
        ] = value
    else:
        k = item_index % static(field.shape[2])
        prefix = item_index // static(field.shape[2])
        j = prefix % static(field.shape[1])
        i = prefix // static(field.shape[1])
        field[
            i + static(offset[0]),
            j + static(offset[1]),
            k + static(offset[2]),
        ] = value


@func
def _tensor_field_load_flat(field: template(), item_index: i32):
    offset = static(
        field.snode.ptr.offset
        if len(field.snode.ptr.offset) != 0
        else [0] * len(field.shape)
    )
    if static(len(field.shape) == 1):
        return field[item_index + static(offset[0])]
    if static(len(field.shape) == 2):
        j = item_index % static(field.shape[1])
        i = item_index // static(field.shape[1])
        return field[
            i + static(offset[0]),
            j + static(offset[1]),
        ]
    k = item_index % static(field.shape[2])
    prefix = item_index // static(field.shape[2])
    j = prefix % static(field.shape[1])
    i = prefix // static(field.shape[1])
    return field[
        i + static(offset[0]),
        j + static(offset[1]),
        k + static(offset[2]),
    ]


@func
def _tensor_field_store_lane_flat(
    field: template(), item_index: i32, lane: i32, value: template()
):
    offset = static(
        field.snode.ptr.offset
        if len(field.snode.ptr.offset) != 0
        else [0] * len(field.shape)
    )
    if static(len(field.shape) == 1):
        if static(field.ndim == 1):
            for p in static(range(field.n)):
                if lane == p:
                    field[item_index + static(offset[0])][p] = value
        else:
            for p in static(range(field.n)):
                for q in static(range(field.m)):
                    if lane == p * field.m + q:
                        field[item_index + static(offset[0])][p, q] = value
    elif static(len(field.shape) == 2):
        j = item_index % static(field.shape[1])
        i = item_index // static(field.shape[1])
        if static(field.ndim == 1):
            for p in static(range(field.n)):
                if lane == p:
                    field[
                        i + static(offset[0]),
                        j + static(offset[1]),
                    ][p] = value
        else:
            for p in static(range(field.n)):
                for q in static(range(field.m)):
                    if lane == p * field.m + q:
                        field[
                            i + static(offset[0]),
                            j + static(offset[1]),
                        ][p, q] = value
    else:
        k = item_index % static(field.shape[2])
        prefix = item_index // static(field.shape[2])
        j = prefix % static(field.shape[1])
        i = prefix // static(field.shape[1])
        if static(field.ndim == 1):
            for p in static(range(field.n)):
                if lane == p:
                    field[
                        i + static(offset[0]),
                        j + static(offset[1]),
                        k + static(offset[2]),
                    ][p] = value
        else:
            for p in static(range(field.n)):
                for q in static(range(field.m)):
                    if lane == p * field.m + q:
                        field[
                            i + static(offset[0]),
                            j + static(offset[1]),
                            k + static(offset[2]),
                        ][p, q] = value


@kernel
def _pack_scalar_field(
    field: template(), output: ndarray_type.ndarray(ndim=1)
):
    offset = static(
        field.snode.ptr.offset
        if len(field.snode.ptr.offset) != 0
        else [0] * len(field.shape)
    )
    for I in grouped(field):
        linear = 0
        for axis in static(range(len(field.shape))):
            linear = (
                linear * static(field.shape[axis])
                + I[axis]
                - static(offset[axis])
            )
        output[linear] = field[I]


@kernel
def _unpack_scalar_field(
    input_arr: ndarray_type.ndarray(ndim=1), field: template()
):
    offset = static(
        field.snode.ptr.offset
        if len(field.snode.ptr.offset) != 0
        else [0] * len(field.shape)
    )
    for I in grouped(field):
        linear = 0
        for axis in static(range(len(field.shape))):
            linear = (
                linear * static(field.shape[axis])
                + I[axis]
                - static(offset[axis])
            )
        field[I] = input_arr[linear]


@kernel
def _pack_tensor_field(
    field: template(), output: ndarray_type.ndarray(ndim=1)
):
    offset = static(
        field.snode.ptr.offset
        if len(field.snode.ptr.offset) != 0
        else [0] * len(field.shape)
    )
    lanes = static(field.n if field.ndim == 1 else field.n * field.m)
    for I in grouped(field):
        item = 0
        for axis in static(range(len(field.shape))):
            item = (
                item * static(field.shape[axis])
                + I[axis]
                - static(offset[axis])
            )
        if static(field.ndim == 1):
            for p in static(range(field.n)):
                output[item * lanes + p] = field[I][p]
        else:
            for p in static(range(field.n)):
                for q in static(range(field.m)):
                    output[item * lanes + p * field.m + q] = field[I][p, q]


@kernel
def _unpack_tensor_field(
    input_arr: ndarray_type.ndarray(ndim=1), field: template()
):
    offset = static(
        field.snode.ptr.offset
        if len(field.snode.ptr.offset) != 0
        else [0] * len(field.shape)
    )
    lanes = static(field.n if field.ndim == 1 else field.n * field.m)
    for I in grouped(field):
        item = 0
        for axis in static(range(len(field.shape))):
            item = (
                item * static(field.shape[axis])
                + I[axis]
                - static(offset[axis])
            )
        if static(field.ndim == 1):
            value = field[I]
            for p in static(range(field.n)):
                value[p] = input_arr[item * lanes + p]
            field[I] = value
        else:
            value = field[I]
            for p in static(range(field.n)):
                for q in static(range(field.m)):
                    value[p, q] = input_arr[
                        item * lanes + p * field.m + q
                    ]
            field[I] = value


@kernel
def _gather_scalar_field(
    field: template(),
    indices: ndarray_type.ndarray(dtype=i32, ndim=1),
    output: ndarray_type.ndarray(ndim=1),
):
    for index in range(indices.shape[0]):
        output[index] = _scalar_field_load_flat(field, indices[index])


@kernel
def _scatter_scalar_field(
    input_arr: ndarray_type.ndarray(ndim=1),
    indices: ndarray_type.ndarray(dtype=i32, ndim=1),
    field: template(),
):
    for index in range(indices.shape[0]):
        _scalar_field_store_flat(field, indices[index], input_arr[index])


@kernel
def _gather_tensor_field(
    field: template(),
    indices: ndarray_type.ndarray(dtype=i32, ndim=1),
    output: ndarray_type.ndarray(ndim=1),
):
    lanes = static(field.n if field.ndim == 1 else field.n * field.m)
    for index in range(indices.shape[0]):
        scalar_index = indices[index]
        item_index = scalar_index // lanes
        lane = scalar_index % lanes
        value = _tensor_field_load_flat(field, item_index)
        output[index] = 0
        if static(field.ndim == 1):
            for p in static(range(field.n)):
                if lane == p:
                    output[index] = value[p]
        else:
            for p in static(range(field.n)):
                for q in static(range(field.m)):
                    if lane == p * field.m + q:
                        output[index] = value[p, q]


@kernel
def _scatter_tensor_field(
    input_arr: ndarray_type.ndarray(ndim=1),
    indices: ndarray_type.ndarray(dtype=i32, ndim=1),
    field: template(),
):
    lanes = static(field.n if field.ndim == 1 else field.n * field.m)
    for index in range(indices.shape[0]):
        scalar_index = indices[index]
        item_index = scalar_index // lanes
        lane = scalar_index % lanes
        _tensor_field_store_lane_flat(
            field, item_index, lane, input_arr[index]
        )


@dataclass(frozen=True)
class _DenseFieldDescriptor:
    storage_kind: str
    dtype: object
    index_shape: tuple
    element_shape: tuple
    item_count: int
    lane_count: int
    scalar_extent: int
    tree_id: int
    node_ids: tuple
    byte_offset: int
    item_stride: int
    runtime_generation: int
    program_identity: int
    storage_description: object = None
    flat_storage_description: object = None

    @property
    def alias_owner_key(self):
        return (
            self.program_identity,
            self.runtime_generation,
            self.tree_id,
            self.node_ids,
        )

    @property
    def descriptor_key(self):
        return (
            self.storage_kind,
            str(self.dtype),
            self.index_shape,
            self.element_shape,
            self.scalar_extent,
            self.tree_id,
            self.node_ids,
            self.byte_offset,
            self.item_stride,
            self.runtime_generation,
            self.program_identity,
        )


def _current_program():
    program = impl.get_runtime().prog
    if program is None:
        raise TaichiRuntimeError(
            "ti.init() must be called before constructing a vector view"
        )
    return program


def _dtype_bytes(dtype):
    if dtype == f64:
        return 8
    if dtype in (f32, i32):
        return 4
    return 0


def _shape_tuple(value):
    shape = value.shape
    if shape is None:
        raise TaichiRuntimeError(
            "dense vector field must be placed before it can be used"
        )
    if isinstance(shape, tuple):
        return tuple(int(extent) for extent in shape)
    if isinstance(shape, list):
        return tuple(int(extent) for extent in shape)
    return (int(shape),)


def _require_dense_index_shape(shape, role):
    if len(shape) not in (1, 2, 3) or any(extent <= 0 for extent in shape):
        raise TaichiRuntimeError(
            f"{role} must have a non-empty 1D, 2D, or 3D dense shape"
        )


def _describe_scalar_field(field, role, allowed_dtypes):
    if not isinstance(field, ScalarField):
        raise TaichiRuntimeError(f"{role} must be a scalar Taichi field")
    if field.dtype not in allowed_dtypes:
        names = ", ".join(str(dtype) for dtype in allowed_dtypes)
        raise TaichiRuntimeError(
            f"{role} must have dtype {names}, got {field.dtype}"
        )
    shape = _shape_tuple(field)
    _require_dense_index_shape(shape, role)
    snode = field.snode
    parent = snode.parent()
    root = snode.parent(2)
    if (
        parent is None
        or parent.ptr.type != _ti_core.SNodeType.dense
        or root is None
        or root.ptr.type != _ti_core.SNodeType.root
        or not _dense_native_field_layout_supported(field)
    ):
        raise TaichiRuntimeError(
            f"{role} must use a root-dense-place SNode layout"
        )
    value_bytes = _dtype_bytes(field.dtype)
    if value_bytes == 0:
        raise TaichiRuntimeError(f"{role} uses an unsupported dtype")
    item_count = math.prod(shape)
    program = _current_program()
    return _DenseFieldDescriptor(
        storage_kind="dense_scalar",
        dtype=field.dtype,
        index_shape=shape,
        element_shape=(),
        item_count=item_count,
        lane_count=1,
        scalar_extent=item_count,
        tree_id=int(snode.ptr.get_snode_tree_id()),
        node_ids=(int(snode.ptr.id),),
        byte_offset=int(snode._offset_bytes_in_parent_cell),
        item_stride=int(parent._cell_size_bytes),
        runtime_generation=int(impl.runtime_generation()),
        program_identity=id(program),
    )


def _describe_matrix_field(field, role):
    if not isinstance(field, MatrixField):
        raise TaichiRuntimeError(
            f"{role} must be a scalar, vector, or matrix Taichi field"
        )
    if field.dtype not in _SUPPORTED_VALUE_DTYPES:
        raise TaichiRuntimeError(
            f"{role} must have dtype ti.f32 or ti.f64, got {field.dtype}"
        )
    shape = _shape_tuple(field)
    _require_dense_index_shape(shape, role)
    indices = tuple(field._native_dense_component_indices())
    components = tuple(field.get_scalar_field(*index) for index in indices)
    if not components:
        raise TaichiRuntimeError(f"{role} has no scalar components")
    value_bytes = _dtype_bytes(field.dtype)
    lane_count = len(components)
    expected_stride = value_bytes * lane_count
    snodes = tuple(component.snode for component in components)
    first_parent = snodes[0].parent()
    first_root = snodes[0].parent(2)
    if (
        first_parent is None
        or first_parent.ptr.type != _ti_core.SNodeType.dense
        or first_root is None
        or first_root.ptr.type != _ti_core.SNodeType.root
        or first_parent._cell_size_bytes != expected_stride
        or snodes[0]._offset_bytes_in_parent_cell != 0
        or not _dense_native_field_layout_supported(components[0])
    ):
        raise TaichiRuntimeError(
            f"{role} must use a canonical packed root-dense-place layout"
        )
    parent_key = (
        int(first_parent.ptr.get_snode_tree_id()),
        int(first_parent.ptr.id),
    )
    for lane, snode in enumerate(snodes):
        parent = snode.parent()
        if (
            parent is None
            or parent.ptr.type != _ti_core.SNodeType.dense
            or (
                int(parent.ptr.get_snode_tree_id()),
                int(parent.ptr.id),
            )
            != parent_key
            or snode._offset_bytes_in_parent_cell != lane * value_bytes
        ):
            raise TaichiRuntimeError(
                f"{role} must use a canonical packed root-dense-place layout"
            )
    item_count = math.prod(shape)
    element_shape = (
        (int(field.n),)
        if int(field.ndim) == 1
        else (int(field.n), int(field.m))
    )
    program = _current_program()
    return _DenseFieldDescriptor(
        storage_kind="dense_packed",
        dtype=field.dtype,
        index_shape=shape,
        element_shape=element_shape,
        item_count=item_count,
        lane_count=lane_count,
        scalar_extent=item_count * lane_count,
        tree_id=int(snodes[0].ptr.get_snode_tree_id()),
        node_ids=tuple(int(snode.ptr.id) for snode in snodes),
        byte_offset=0,
        item_stride=expected_stride,
        runtime_generation=int(impl.runtime_generation()),
        program_identity=id(program),
    )


def _describe_value_field_legacy(field, role):
    impl.get_runtime().materialize()
    if isinstance(field, ScalarField):
        return _describe_scalar_field(
            field, role, allowed_dtypes=_SUPPORTED_VALUE_DTYPES
        )
    return _describe_matrix_field(field, role)


def _describe_value_field(field, role):
    """Describe vector semantics from the shared physical descriptor."""

    if not isinstance(field, (ScalarField, MatrixField)):
        raise TaichiRuntimeError(
            f"{role} must be a scalar, vector, or matrix Taichi field"
        )
    if field.dtype not in _SUPPORTED_VALUE_DTYPES:
        raise TaichiRuntimeError(
            f"{role} must have dtype ti.f32 or ti.f64, got {field.dtype}"
        )
    shape = _shape_tuple(field)
    _require_dense_index_shape(shape, role)
    description = describe_storage(field)
    if not description.supported:
        raise TaichiRuntimeError(
            f"{role} must use a supported root-dense-place SNode layout "
            f"({description.failure_reason})"
        )
    descriptor = description.descriptor
    source_kind = descriptor.source_kind
    if source_kind not in ("kDenseScalarField", "kDensePackedField"):
        raise TaichiRuntimeError(
            f"{role} must use a supported root-dense-place SNode layout"
        )
    if tuple(int(extent) for extent in descriptor.index_shape) != shape:
        raise TaichiRuntimeError(f"{role} storage shape is inconsistent")

    properties = description.properties
    item_count = int(properties["item_count"])
    scalar_extent = int(properties["scalar_count"])
    if item_count <= 0 or scalar_extent <= 0 or scalar_extent % item_count:
        raise TaichiRuntimeError(f"{role} storage extent is inconsistent")
    lane_count = scalar_extent // item_count
    element_shape = tuple(
        int(extent) for extent in descriptor.element_shape
    )
    if isinstance(field, ScalarField):
        if source_kind != "kDenseScalarField" or element_shape or lane_count != 1:
            raise TaichiRuntimeError(f"{role} scalar field layout is inconsistent")
        storage_kind = "dense_scalar"
    else:
        expected_element_shape = (
            (int(field.n),)
            if int(field.ndim) == 1
            else (int(field.n), int(field.m))
        )
        if (
            source_kind != "kDensePackedField"
            or element_shape != expected_element_shape
            or lane_count != math.prod(expected_element_shape)
        ):
            raise TaichiRuntimeError(
                f"{role} must use a canonical packed root-dense-place layout"
            )
        storage_kind = "dense_packed"

    flat_description = _flatten_storage_to_scalar_vector(description)
    if not flat_description.supported:
        flat_description = None
    tree_identity = descriptor.tree_identity
    if tree_identity is None:
        raise TaichiRuntimeError(f"{role} has no live SNodeTree owner")
    program = _current_program()
    return _DenseFieldDescriptor(
        storage_kind=storage_kind,
        dtype=descriptor.scalar_type,
        index_shape=shape,
        element_shape=element_shape,
        item_count=item_count,
        lane_count=lane_count,
        scalar_extent=scalar_extent,
        tree_id=int(tree_identity[0]),
        node_ids=tuple(int(value) for value in descriptor.component_snode_ids),
        byte_offset=int(descriptor.byte_offset),
        item_stride=int(properties["record_stride"]),
        runtime_generation=int(impl.runtime_generation()),
        program_identity=id(program),
        storage_description=description,
        flat_storage_description=flat_description,
    )



def _validate_indices_ndarray(indices, role):
    program = _current_program()
    if (
        not isinstance(indices, ScalarNdarray)
        or indices.arr is None
        or indices._runtime_prog is not program
        or indices.dtype != i32
        or len(indices.shape) != 1
    ):
        raise TaichiRuntimeError(
            f"{role} must be a one-dimensional ti.i32 ndarray"
        )
    return indices


def _snapshot_indices(indices, source_extent):
    role = "vector_view indices"
    if isinstance(indices, ScalarNdarray):
        source = _validate_indices_ndarray(indices, role)
        extent = int(source.shape[0])
        if extent <= 0:
            raise TaichiRuntimeError(f"{role} must be non-empty")
        snapshot = ScalarNdarray(i32, (extent,))
        snapshot.copy_from(source)
    elif isinstance(indices, ScalarField):
        descriptor = _describe_scalar_field(
            indices, role, allowed_dtypes=(i32,)
        )
        if len(descriptor.index_shape) != 1:
            raise TaichiRuntimeError(f"{role} field must be one-dimensional")
        extent = descriptor.scalar_extent
        snapshot = ScalarNdarray(i32, (extent,))
        _pack_scalar_field(indices, snapshot)
    else:
        raise TaichiRuntimeError(
            f"{role} must be a one-dimensional ti.i32 ndarray or dense field"
        )

    host_indices = np.asarray(snapshot.to_numpy(), dtype=np.int32)
    if np.any(host_indices < 0) or np.any(host_indices >= source_extent):
        raise TaichiRuntimeError(
            "vector_view indices must be within the scalar-flat field extent"
        )
    if np.unique(host_indices).size != host_indices.size:
        raise TaichiRuntimeError(
            "vector_view indices must be unique for deterministic scatter"
        )
    digest = hashlib.sha256(host_indices.tobytes()).hexdigest()
    return snapshot, digest


class VectorView:
    """A runtime-bound scalar-flat view of a supported dense Taichi field."""

    _TOKEN = object()

    def __init__(self, token, field, descriptor, indices, indices_digest):
        if token is not self._TOKEN:
            raise TypeError("Use ti.linalg.experimental.vector_view()")
        self._field = field
        self._descriptor = descriptor
        self._indices = indices
        self._indexed = indices is not None
        self._indices_digest = indices_digest
        self._program = _current_program()
        self._runtime_generation = int(impl.runtime_generation())
        self._retired = False
        self._retirement_pending = None
        self.dtype = descriptor.dtype
        self.index_shape = descriptor.index_shape
        self.element_shape = descriptor.element_shape
        self.source_scalar_extent = descriptor.scalar_extent
        self.scalar_extent = (
            descriptor.scalar_extent
            if indices is None
            else int(indices.shape[0])
        )
        self.shape = (self.scalar_extent,)
        impl.get_runtime().register_runtime_object(self)

    @property
    def indexed(self):
        return self._indexed

    @property
    def _direct_storage_description(self):
        if self.indexed:
            return None
        return self._descriptor.flat_storage_description

    @property
    def _direct_storage_descriptor(self):
        description = self._direct_storage_description
        return None if description is None else description.descriptor

    @property
    def metadata(self):
        return MappingProxyType(
            {
                "schema": _VECTOR_IO_SCHEMA,
                "storage_kind": self._descriptor.storage_kind,
                "layout_kind": (
                    "indexed_scalar_flat"
                    if self.indexed
                    else "full_scalar_flat"
                ),
                "execution_mode": (
                    "provider_qualified_direct_or_staged"
                    if self._direct_storage_description is not None
                    else "device_staged"
                ),
                "value_host_transfer": False,
                "zero_copy_candidate": (
                    self._direct_storage_description is not None
                ),
                "zero_copy": False,
                "index_validation": (
                    "host_once_immutable_snapshot"
                    if self.indexed
                    else "not_applicable"
                ),
                "dtype": str(self.dtype),
                "index_shape": self.index_shape,
                "element_shape": self.element_shape,
                "source_scalar_extent": self.source_scalar_extent,
                "scalar_extent": self.scalar_extent,
                "tree_id": self._descriptor.tree_id,
                "node_ids": self._descriptor.node_ids,
                "byte_offset": self._descriptor.byte_offset,
                "item_stride": self._descriptor.item_stride,
                "indices_digest": self._indices_digest,
            }
        )

    @property
    def _alias_owner_key(self):
        return self._descriptor.alias_owner_key

    @property
    def _exact_view_key(self):
        return (
            self._descriptor.alias_owner_key,
            self._indices_digest if self.indexed else None,
        )

    def _ensure_valid(self, role="VectorView"):
        if self._retired:
            raise TaichiRuntimeError(
                f"{role} references a destroyed SNodeTree"
            )
        if (
            self._program is None
            or self._program is not impl.get_runtime().prog
            or self._runtime_generation != impl.runtime_generation()
        ):
            raise TaichiRuntimeError(
                f"{role} belongs to an inactive or different Taichi runtime"
            )
        return self

    def _retire_snode_tree(self, dependency):
        dependency = tuple(int(value) for value in dependency)
        if dependency[0] != self._descriptor.tree_id or self._retired:
            return False
        self._retired = True
        self._retirement_pending = dependency
        return True

    def _cancel_snode_tree_retirement(self, dependency):
        dependency = tuple(int(value) for value in dependency)
        if self._retirement_pending == dependency:
            self._retired = False
            self._retirement_pending = None

    def _invalidate_runtime(self):
        self._retired = True
        self._retirement_pending = None
        self._program = None
        self._indices = None

    def _pack_to(self, output):
        self._ensure_valid("VectorView input")
        if self.indexed:
            if self._descriptor.storage_kind == "dense_scalar":
                _gather_scalar_field(self._field, self._indices, output)
            else:
                _gather_tensor_field(self._field, self._indices, output)
        elif self._descriptor.storage_kind == "dense_scalar":
            _pack_scalar_field(self._field, output)
        else:
            _pack_tensor_field(self._field, output)

    def _unpack_from(self, input_arr):
        self._ensure_valid("VectorView output")
        if self.indexed:
            if self._descriptor.storage_kind == "dense_scalar":
                _scatter_scalar_field(input_arr, self._indices, self._field)
            else:
                _scatter_tensor_field(input_arr, self._indices, self._field)
        elif self._descriptor.storage_kind == "dense_scalar":
            _unpack_scalar_field(input_arr, self._field)
        else:
            _unpack_tensor_field(input_arr, self._field)


def vector_view(field, *, indices=None):
    """Returns a scalar-flat, runtime-bound view of a dense Taichi field.

    ``indices`` is an optional one-dimensional ``ti.i32`` ndarray or dense
    scalar field.  It is validated and snapshotted at construction, so the view
    has immutable subset topology.  Indices address the source field's
    canonical scalar-flat order and must be in range and unique.
    """

    if isinstance(field, VectorView):
        if indices is not None:
            raise TaichiRuntimeError(
                "indices cannot be supplied when field is already a VectorView"
            )
        return field._ensure_valid()
    descriptor = _describe_value_field(field, "vector_view field")
    indices_snapshot = None
    indices_digest = None
    if indices is not None:
        indices_snapshot, indices_digest = _snapshot_indices(
            indices, descriptor.scalar_extent
        )
    return VectorView(
        VectorView._TOKEN,
        field,
        descriptor,
        indices_snapshot,
        indices_digest,
    )


def _is_dense_field_value(value):
    return isinstance(value, (ScalarField, MatrixField))


def _as_vector_view(value, role):
    if isinstance(value, VectorView):
        return value._ensure_valid(role)
    if _is_dense_field_value(value):
        descriptor = _describe_value_field(value, role)
        return VectorView(
            VectorView._TOKEN, value, descriptor, None, None
        )
    return None



def _native_transfer_arch_supported():
    return impl.current_cfg().arch in (
        _ti_core.Arch.x64,
        _ti_core.Arch.arm64,
        _ti_core.Arch.cuda,
        _ti_core.Arch.vulkan,
    )


class _NativeVectorTransfer:
    """One direct contiguous field/staging bulk copy."""

    @classmethod
    def try_create(cls, view, staging, direction, allow_native_bulk):
        descriptor = view._descriptor
        program = _current_program()
        method_name = (
            "_copy_dense_field_to_ndarray"
            if direction == "pack"
            else "_copy_ndarray_to_dense_field"
        )
        if (
            not allow_native_bulk
            or view.indexed
            or not _native_transfer_arch_supported()
            or not hasattr(program, method_name)
            or descriptor.item_stride
            != descriptor.lane_count * _dtype_bytes(descriptor.dtype)
        ):
            return None
        return cls(view, staging, direction, program, method_name)

    def __init__(self, view, staging, direction, program, method_name):
        self._field = view._field
        self._staging = staging
        self._exact_view_key = view._exact_view_key
        self._direction = direction
        self._program = program
        self._method = getattr(program, method_name)
        self._value_type = _dense_host_copy_value_type(view.dtype)
        self._item_count = view._descriptor.item_count
        self._lane_count = view._descriptor.lane_count
        if view._descriptor.storage_kind == "dense_scalar":
            self._snode = self._field.snode.ptr
        else:
            first_index = next(
                iter(self._field._native_dense_component_indices())
            )
            self._snode = self._field.get_scalar_field(
                *first_index
            ).snode.ptr

    def matches(self, view, staging, direction):
        return (
            direction == self._direction
            and view._field is self._field
            and view._exact_view_key == self._exact_view_key
            and staging is self._staging
            and _current_program() is self._program
        )

    def run(self, view):
        view._ensure_valid("VectorView transfer")
        if self._direction == "pack":
            self._method(
                self._staging.arr,
                self._snode,
                self._value_type,
                self._item_count,
                self._lane_count,
            )
        else:
            self._method(
                self._snode,
                self._staging.arr,
                self._value_type,
                self._item_count,
                self._lane_count,
            )


class _CompiledVectorTransfer:
    """One replayable field/staging conversion with fixed resource identity."""

    def __init__(self, view, staging, direction):
        from taichi_forge.graph import (  # pylint: disable=import-outside-toplevel
            Arg,
            ArgKind,
            GraphBuilder,
        )

        self._field = view._field
        self._indices = view._indices
        self._staging = staging
        self._exact_view_key = view._exact_view_key
        self._direction = direction

        staging_arg = Arg(ArgKind.NDARRAY, "staging", view.dtype, ndim=1)
        builder = GraphBuilder()
        template_args = {"field": self._field}
        if view.indexed:
            indices_arg = Arg(ArgKind.NDARRAY, "indices", i32, ndim=1)
            if direction == "pack":
                transfer_kernel = (
                    _gather_scalar_field
                    if view._descriptor.storage_kind == "dense_scalar"
                    else _gather_tensor_field
                )
                builder.dispatch(
                    transfer_kernel,
                    indices_arg,
                    staging_arg,
                    template_args=template_args,
                )
            else:
                transfer_kernel = (
                    _scatter_scalar_field
                    if view._descriptor.storage_kind == "dense_scalar"
                    else _scatter_tensor_field
                )
                builder.dispatch(
                    transfer_kernel,
                    staging_arg,
                    indices_arg,
                    template_args=template_args,
                )
            self._arguments = {
                "indices": self._indices,
                "staging": self._staging,
            }
        else:
            if direction == "pack":
                transfer_kernel = (
                    _pack_scalar_field
                    if view._descriptor.storage_kind == "dense_scalar"
                    else _pack_tensor_field
                )
            else:
                transfer_kernel = (
                    _unpack_scalar_field
                    if view._descriptor.storage_kind == "dense_scalar"
                    else _unpack_tensor_field
                )
            builder.dispatch(
                transfer_kernel,
                staging_arg,
                template_args=template_args,
            )
            self._arguments = {"staging": self._staging}
        self._graph = builder.compile()

    def matches(self, view, staging, direction):
        return (
            direction == self._direction
            and view._field is self._field
            and view._indices is self._indices
            and view._exact_view_key == self._exact_view_key
            and staging is self._staging
        )

    def run(self, view):
        view._ensure_valid("VectorView transfer")
        self._graph.run(self._arguments)


def vector_io_capabilities():
    """Returns the backend-neutral dense vector I/O capability contract."""

    return {
        "schema": _VECTOR_IO_SCHEMA,
        "ndarray": {
            "execution_mode": "direct",
            "layouts": ("scalar_1d",),
            "value_host_transfer": False,
            "zero_copy": True,
        },
        "dense_field": {
            "execution_mode": "provider_qualified",
            "full_field_execution_modes": (
                "direct_contiguous",
                "device_staged",
            ),
            "dtypes": ("f32", "f64"),
            "layouts": (
                "root_dense_scalar_1d_2d_3d",
                "root_dense_packed_vector_matrix_1d_2d_3d",
                "indexed_scalar_flat",
            ),
            "index_topology": "immutable_validated_snapshot",
            "index_validation": "host_once_at_view_construction",
            "indexed_output_policy": "unique_indices_required",
            "value_host_transfer": False,
            "staging_reuse": "operator_or_plan_owned",
            "conversion_scope": "apply_or_solve_boundary_only",
            "conversion_submission": (
                "native_bulk_copy_or_compiled_graph_replay"
            ),
            "zero_copy": False,
            "zero_copy_condition": (
                "canonical full field and provider dense_storage_operands"
            ),
        },
        "sparse_snode": {
            "execution_mode": "unavailable",
            "value_host_transfer": False,
            "zero_copy": False,
            "reason": "sparse SNode vector storage is outside this contract",
        },
    }


class _VectorIOCache:
    """Reusable scalar ndarray staging and operation telemetry."""

    def __init__(self, *, allow_native_bulk):
        self._allow_native_bulk = bool(allow_native_bulk)
        self._buffers = {}
        self._implicit_views = {}
        self._transfer_plans = {}
        self._stats = {
            "schema": _VECTOR_IO_SCHEMA,
            "staging_buffer_builds": 0,
            "staging_buffer_reuses": 0,
            "staging_reserved_bytes": 0,
            "implicit_view_builds": 0,
            "implicit_view_reuses": 0,
            "implicit_view_evictions": 0,
            "transfer_plan_builds": 0,
            "transfer_plan_reuses": 0,
            "transfer_plan_evictions": 0,
            "transfer_graph_submissions": 0,
            "transfer_native_submissions": 0,
            "pack_calls": 0,
            "unpack_calls": 0,
            "indexed_gather_calls": 0,
            "indexed_scatter_calls": 0,
            "packed_logical_bytes": 0,
            "unpacked_logical_bytes": 0,
            "direct_bindings": 0,
            "direct_dense_field_submissions": 0,
            "completion_syncs": 0,
            "coalesced_operator_syncs": 0,
            "last_input_storage": "unavailable",
            "last_output_storage": "unavailable",
            "last_input_execution_mode": "unavailable",
            "last_output_execution_mode": "unavailable",
        }

    @staticmethod
    def _value_bytes(dtype):
        return _dtype_bytes(dtype)

    def buffer(self, role, dtype, size):
        key = (str(role), str(dtype), int(size))
        buffer = self._buffers.get(key)
        if buffer is not None:
            self._stats["staging_buffer_reuses"] += 1
            return buffer
        buffer = ScalarNdarray(dtype, (int(size),))
        self._buffers[key] = buffer
        self._stats["staging_buffer_builds"] += 1
        self._stats["staging_reserved_bytes"] += (
            int(size) * self._value_bytes(dtype)
        )
        return buffer

    def view(self, value, role):
        if isinstance(value, VectorView):
            return value._ensure_valid(role)
        if not _is_dense_field_value(value):
            return None
        key = id(value)
        cached = self._implicit_views.get(key)
        if cached is not None and cached._field is value:
            self._stats["implicit_view_reuses"] += 1
            return cached._ensure_valid(role)
        descriptor = _describe_value_field(value, role)
        view = VectorView(VectorView._TOKEN, value, descriptor, None, None)
        if len(self._implicit_views) >= _IMPLICIT_VIEW_CACHE_LIMIT:
            self._implicit_views.pop(next(iter(self._implicit_views)))
            self._stats["implicit_view_evictions"] += 1
        self._implicit_views[key] = view
        self._stats["implicit_view_builds"] += 1
        return view

    def _transfer(self, view, staging, direction):
        key = (
            direction,
            view._exact_view_key,
            id(view._indices) if view.indexed else 0,
            id(staging),
        )
        plan = self._transfer_plans.get(key)
        if plan is not None and plan.matches(view, staging, direction):
            self._stats["transfer_plan_reuses"] += 1
        else:
            plan = _NativeVectorTransfer.try_create(
                view,
                staging,
                direction,
                self._allow_native_bulk,
            )
            if plan is None:
                plan = _CompiledVectorTransfer(view, staging, direction)
            if len(self._transfer_plans) >= _TRANSFER_PLAN_CACHE_LIMIT:
                self._transfer_plans.pop(next(iter(self._transfer_plans)))
                self._stats["transfer_plan_evictions"] += 1
            self._transfer_plans[key] = plan
            self._stats["transfer_plan_builds"] += 1
        plan.run(view)
        if isinstance(plan, _NativeVectorTransfer):
            self._stats["transfer_native_submissions"] += 1
        else:
            self._stats["transfer_graph_submissions"] += 1

    def pack(self, view, role, dtype, size):
        staging = self.buffer(role, dtype, size)
        self._transfer(view, staging, "pack")
        logical_bytes = int(size) * self._value_bytes(dtype)
        self._stats["pack_calls"] += 1
        self._stats["packed_logical_bytes"] += logical_bytes
        if view.indexed:
            self._stats["indexed_gather_calls"] += 1
        self._stats["last_input_storage"] = (
            "indexed_dense_field" if view.indexed else "dense_field"
        )
        self._stats["last_input_execution_mode"] = "device_staged"
        return staging

    def unpack(self, staging, view, dtype, size):
        self._transfer(view, staging, "unpack")
        logical_bytes = int(size) * self._value_bytes(dtype)
        self._stats["unpack_calls"] += 1
        self._stats["unpacked_logical_bytes"] += logical_bytes
        if view.indexed:
            self._stats["indexed_scatter_calls"] += 1
        self._stats["last_output_storage"] = (
            "indexed_dense_field" if view.indexed else "dense_field"
        )
        self._stats["last_output_execution_mode"] = "device_staged"

    def record_direct_input(self):
        self._stats["direct_bindings"] += 1
        self._stats["last_input_storage"] = "ndarray"
        self._stats["last_input_execution_mode"] = "direct"

    def record_direct_output(self):
        self._stats["direct_bindings"] += 1
        self._stats["last_output_storage"] = "ndarray"
        self._stats["last_output_execution_mode"] = "direct"

    def record_direct_dense_fields(self):
        self._stats["direct_bindings"] += 2
        self._stats["direct_dense_field_submissions"] += 1
        self._stats["last_input_storage"] = "dense_field"
        self._stats["last_output_storage"] = "dense_field"
        self._stats["last_input_execution_mode"] = "direct_contiguous"
        self._stats["last_output_execution_mode"] = "direct_contiguous"

    def record_completion_sync(self):
        self._stats["completion_syncs"] += 1

    def record_coalesced_operator_sync(self):
        self._stats["coalesced_operator_syncs"] += 1

    def statistics(self):
        return dict(self._stats)


__all__ = ["VectorView", "vector_io_capabilities", "vector_view"]
