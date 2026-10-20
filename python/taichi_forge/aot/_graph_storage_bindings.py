"""Cold shared bindings for retained arrays and module-owned Field roots."""

from math import prod

from taichi_forge._lib import core
from taichi_forge.lang._ndarray import ScalarNdarray, StructNdarray, StructNdarrayScalarMemberView
from taichi_forge.lang._storage_view import DenseNdarrayView, describe_storage
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.lang.field import ScalarField
from taichi_forge.lang.matrix import MatrixField, MatrixNdarray, VectorNdarray
from ._graph_array_data import ArrayDataLayout, ArraySnapshot

_SOURCE_NDARRAY_TYPES = (ScalarNdarray, VectorNdarray, MatrixNdarray)


def packed_description(owner):
    description = describe_storage(owner)
    if not description.supported or not all(
        description.properties[key] for key in ("compact_contiguous", "ndarray_abi_compatible", "aligned")
    ):
        raise TaichiRuntimeError("AOT retained arrays require aligned packed dense storage")
    return description.descriptor


def _field_key(descriptor):
    return (
        tuple(descriptor.tree_identity),
        descriptor.byte_offset,
        str(descriptor.scalar_type),
        tuple(descriptor.index_shape),
        tuple(descriptor.element_shape),
    )


class RetainedArrayBindings:
    def __init__(self, module, context):
        self.module = module
        self.context = context
        self.fields = {}
        self.tree_ids = set()
        self.private_arrays = {}
        self.private_roles = {}
        self.private_usages = {}
        self.snapshots = {}
        self.existing = {_field_key(packed_description(module._fields[name])): name for name in module._array_fields}

    def require(self, owner):
        context = self.context
        if type(owner) in (*_SOURCE_NDARRAY_TYPES, StructNdarray):
            identity = context.source_arrays.identity(owner)
            if identity in self.private_arrays:
                return self.private_arrays[identity]
            if context.source_arrays.find(owner) is None and context.snapshot_state:
                return self.declare_snapshot(owner, mutable=True)
            return context.source_arrays.require(owner)
        if type(owner) is StructNdarrayScalarMemberView:
            if owner.component is not None:
                raise TaichiRuntimeError("AOT structure member export requires a scalar leaf path")
            return context._resource(
                owner, dict(kind="ndarray_member", array=self.require(owner.base),
                            path=".".join(owner.path)),
            )
        if type(owner) is DenseNdarrayView:
            selected = packed_description(owner)
            parent = packed_description(owner._source)
            binding = self.require(owner._source)
            if selected.scalar_type != parent.scalar_type or tuple(selected.element_shape) != tuple(
                parent.element_shape
            ):
                raise TaichiRuntimeError("AOT retained views cannot reinterpret their parent layout")
            delta = selected.byte_offset - parent.byte_offset
            element_bytes = core.data_type_size(parent.scalar_type) * prod(parent.element_shape)
            if delta < 0 or delta % element_bytes:
                raise TaichiRuntimeError("AOT retained view offset must select packed parent elements")
            if delta == 0 and tuple(selected.index_shape) == tuple(parent.index_shape):
                return binding
            return context._resource(
                owner,
                dict(
                    kind="ndarray_view",
                    array=binding,
                    offset=delta // element_bytes,
                    shape=list(selected.index_shape),
                ),
            )
        if type(owner) not in (ScalarField, MatrixField):
            raise TaichiRuntimeError("AOT retained array requires a named ndarray or a qualified dense Field/view")
        descriptor = packed_description(owner)
        dependency = tuple(descriptor.tree_identity)
        context.field_state((dependency,))
        self.tree_ids.add(dependency[0])
        key = _field_key(descriptor)
        previous = self.fields.get(key)
        if previous is not None:
            return previous[2]
        view = self.existing.get(key) or context._name("field")
        resource = context._resource(owner, dict(kind="module_array", view=view))
        self.fields[key] = (view, owner, resource)
        return resource

    def declare_scratch(self, owner, *, usage=None):
        """A concrete recipe proves a full overwrite before any workspace read."""
        identity = self._private_identity(owner, "scratch")
        required_usage = tuple(sorted(usage or ()))
        if self.private_usages.setdefault(identity, required_usage) != required_usage:
            raise TaichiRuntimeError("AOT scratch allocation usage changed within one export")
        if identity not in self.private_arrays:
            layout = self.data_layout(owner)
            self.private_arrays[identity] = self.context._resource(
                owner,
                dict(
                    kind="ndarray",
                    **layout.resource_fields(),
                    **({"usage": list(required_usage)} if usage is not None else {}),
                ),
            )
        return self.private_arrays[identity]

    def declare_data(self, owner, data):
        """A concrete recipe supplies immutable construction-time array data."""
        identity = self._private_identity(owner, "data")
        layout = self.data_layout(owner)
        if len(data) != layout.byte_size:
            raise ValueError("AOT construction data does not match its complete array layout")
        resource = self.context.data_resource(
            owner,
            data,
            **layout.resource_fields(),
            readonly=True,
        )
        previous = self.private_arrays.setdefault(identity, resource)
        if previous != resource:
            raise TaichiRuntimeError("AOT private array reconstruction changed within one export")
        return resource

    def declare_snapshot(self, owner, *, mutable=False):
        """Capture private recipe storage with the complete export closure.

        The concrete exporter proves immutability; arbitrary mutable application
        state still requires the existing explicit snapshot contract. No device
        transfer occurs until the whole Graph has passed structural preflight.
        """
        if mutable and not self.context.snapshot_state:
            raise TaichiRuntimeError("AOT captured arrays require explicit snapshot_state=True at export")
        identity = self._private_identity(owner, "state" if mutable else "data")
        if identity not in self.private_arrays:
            name = self.context._name("resource")
            self.snapshots[identity] = ArraySnapshot(owner, name, self.data_layout(owner))
            self.private_arrays[identity] = name
        return self.private_arrays[identity]

    def data_layout(self, owner):
        if type(owner) is StructNdarray:
            return ArrayDataLayout(
                tuple(owner.shape), core.data_type_size(owner.dtype),
                layout=self.context.source_arrays.structure_layout(owner.struct_type))
        descriptor = packed_description(owner)
        return ArrayDataLayout(
            tuple(descriptor.index_shape),
            core.data_type_size(descriptor.scalar_type) * prod(descriptor.element_shape),
            dtype=str(descriptor.scalar_type), element_shape=tuple(descriptor.element_shape))

    def _private_identity(self, owner, role):
        identity = self.context.source_arrays.identity(owner)
        if self.context.source_arrays.find(owner) is not None:
            raise TaichiRuntimeError("AOT private recipe storage cannot also be an external source binding")
        if self.private_roles.setdefault(identity, role) != role:
            raise TaichiRuntimeError("AOT private storage cannot alias a different reconstruction role")
        return identity

    def bind_private(self, name, owner):
        context = self.context
        if name in context.private_bindings and context.private_bindings[name] is not owner:
            raise TaichiRuntimeError("AOT retained array conflicts with a private binding")
        context._private_names[name] = self.require(owner)
        context.private_bindings[name] = owner

    def publish(self):
        # Called only after the entire Graph has passed structural preflight.
        for view, owner, _ in self.fields.values():
            if view not in self.module._array_fields:
                self.module.add_field(view, owner, as_array=True)
