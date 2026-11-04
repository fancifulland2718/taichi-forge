"""Qualified ray resource ports projected into the existing native Plan lowering."""

import json

from taichi_forge.graph._ir import (
    GraphAccess,
    NativeCallNode,
    ResourceEffect,
    RuntimeBinding,
)
from taichi_forge.graph._native import NativeGraphExecutable
from taichi_forge.lang.exception import TaichiCompilationError, TaichiRuntimeError


def capture_instance_transforms(recording, capture):
    from taichi_forge.graph import Arg, ArgKind

    if recording.scene_binding is None:
        raise TaichiCompilationError(
            "Component refit requires an explicit scene binding on record_refit_transforms()"
        )
    recording.validate_graph_lifetime()
    capture.explicit_argument(
        Arg(ArgKind.ACCELERATION_STRUCTURE, recording.scene_binding)
    )
    return _ComponentRayDefinition(
        dict(
            kind="native",
            provider="vulkan_ray",
            operation="instance_refit",
            version=1,
            parameters={},
            bindings=dict(
                scene=recording.scene_binding, transforms=recording.transforms
            ),
        )
    )


class _ComponentRayDefinition(NativeGraphExecutable):
    """No producer handles, topology, arrays or device command survive capture."""

    control_kind = "native_ray"

    def __init__(self, operation):
        from taichi_forge.aot._component import _digest
        from taichi_forge_runtime.aot._component_native import operation_bindings

        operation_bindings(operation)
        self._operation = json.dumps(operation, sort_keys=True)
        self.semantic_key = _digest(operation)
        self.scene, self.transforms = (
            operation["bindings"]["scene"],
            operation["bindings"]["transforms"],
        )

    @property
    def operations(self):
        return [json.loads(self._operation)]

    @property
    def runtime_arg_schema(self):
        return (
            RuntimeBinding(self.scene, "acceleration_structure"),
            RuntimeBinding(self.transforms, "ndarray"),
        )

    @property
    def resource_effects(self):
        return (
            ResourceEffect(self.scene, GraphAccess.WRITE),
            ResourceEffect(self.transforms, GraphAccess.READ),
        )

    @property
    def graph_ir_node(self):
        return NativeCallNode(
            name="component_ray_" + self.semantic_key,
            bindings=self.runtime_arg_schema,
            effects=self.resource_effects,
            opaque=False,
            semantic_fingerprint=self.semantic_key,
        )

    def _export_aot(self, context):
        from taichi_forge_runtime.aot._component_native import map_operation_bindings

        context.operations.append(
            map_operation_bindings(
                self.operations[0], lambda value: context.binding(value, resource=True)
            )
        )

    def run(self, runtime_args=None):
        raise TaichiRuntimeError(
            "A component native definition must execute through its exported instance plan"
        )
