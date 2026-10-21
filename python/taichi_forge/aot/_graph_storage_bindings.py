"""Deployment-owned Field publication and retained-array snapshots."""

from taichi_forge.graph._lowering_array_data import ArraySnapshot
from taichi_forge.graph._lowering_arrays import (
    RetainedArrayBindings as _RetainedArrayBindings,
    _SOURCE_NDARRAY_TYPES,
    _field_key,
    field_description,
    packed_description,
)
from taichi_forge.lang.exception import TaichiRuntimeError


class RetainedArrayBindings(_RetainedArrayBindings):
    def __init__(self, module, context):
        super().__init__(context)
        self.module = module
        self.fields = {}
        self.tree_ids = set()
        self.snapshots = {}
        self.existing = {_field_key(field_description(module._fields[name])): name for name in module._array_fields}

    def _reference_field(self, owner):
        context = self.context
        descriptor = field_description(owner)
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

    def publish(self):
        # Called only after the entire Graph has passed structural preflight.
        for view, owner, _ in self.fields.values():
            if view not in self.module._array_fields:
                self.module.add_field(view, owner, as_array=True)
