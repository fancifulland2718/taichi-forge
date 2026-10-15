"""Cold shared bindings for retained arrays and module-owned Field roots."""

from math import prod

from taichi_forge._lib import core
from taichi_forge.lang._ndarray import ScalarNdarray
from taichi_forge.lang._storage_view import DenseNdarrayView, describe_storage
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.lang.field import ScalarField
from taichi_forge.lang.matrix import MatrixField


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
        self.scratch = {}
        self.existing = {_field_key(packed_description(module._fields[name])): name for name in module._array_fields}

    def require(self, owner):
        context = self.context
        if type(owner) is ScalarNdarray:
            identity = context.source_arrays.identity(owner)
            if identity in self.scratch:
                return self.scratch[identity]
            return context.source_arrays.require(owner)
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

    def declare_scratch(self, owner):
        """A concrete recipe proves a full overwrite before any workspace read."""
        identity = self.context.source_arrays.identity(owner)
        if self.context.source_arrays.find(owner) is not None:
            raise TaichiRuntimeError("AOT private scratch cannot also be an external source binding")
        if identity not in self.scratch:
            descriptor = packed_description(owner)
            self.scratch[identity] = self.context._resource(
                owner,
                dict(
                    kind="ndarray",
                    dtype=str(descriptor.scalar_type),
                    shape=list(descriptor.index_shape),
                    element_shape=list(descriptor.element_shape),
                ),
            )
        return self.scratch[identity]

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
