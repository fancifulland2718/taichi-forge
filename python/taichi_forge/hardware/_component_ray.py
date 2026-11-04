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


def _optix_scene(recording, capture):
    from taichi_forge.graph import Arg, ArgKind

    if recording.scene_binding is None:
        raise TaichiCompilationError(
            "Component OptiX recording requires an explicit scene binding"
        )
    recording.validate_graph_lifetime()
    capture.explicit_argument(
        Arg(ArgKind.ACCELERATION_STRUCTURE, recording.scene_binding)
    )


def capture_optix_refit(recording, capture):
    _optix_scene(recording, capture)
    if recording.transforms is None:
        raise TaichiCompilationError("Component OptiX refit requires device transforms")
    return _ComponentRayDefinition(
        dict(
            kind="native",
            provider="optix",
            operation="instance_refit",
            version=1,
            parameters={},
            bindings=dict(
                scene=recording.scene_binding, transforms=recording.transforms
            ),
        )
    )


def capture_optix_query(recording, capture):
    _optix_scene(recording, capture)
    if recording.hit_indices is None and not recording._occlusion:
        raise TaichiCompilationError(
            "Component OptiX queries require typed hits or compact occlusion"
        )
    if any(mask is not None for mask in recording.alpha_masks or ()) or getattr(
        recording.scene, "_has_micromaps", False
    ):
        raise TaichiCompilationError(
            "Component OptiX masks and micromaps need a qualified resource contract"
        )
    rules = recording.face_rules or ("two_sided",)
    if any(not isinstance(rule, str) or rule != rules[0] for rule in rules):
        raise TaichiCompilationError(
            "Component OptiX queries currently require uniform face rules"
        )
    bindings = dict(
        scene=recording.scene_binding, rays=recording.rays, hits=recording.hits
    )
    if recording.hit_indices is not None:
        bindings["hit_indices"] = recording.hit_indices
    # The ordinary recording validates exact whole-array ray counts. This
    # version resolves that same count from each consumer's prepared binding.
    return _ComponentRayDefinition(
        dict(
            kind="native",
            provider="optix",
            operation="query",
            version=4,
            parameters=dict(
                occlusion=recording._occlusion,
                any_hit=recording.any_hit,
                front_only=rules[0] == "front_only",
                graph_fixed_query=recording.graph_fixed_query,
            ),
            bindings=bindings,
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
        self._ports = tuple(operation["bindings"].items())
        self._refit = operation["operation"] == "instance_refit"

    @property
    def operations(self):
        return [json.loads(self._operation)]

    @property
    def runtime_arg_schema(self):
        return tuple(
            RuntimeBinding(
                name, "acceleration_structure" if port == "scene" else "ndarray"
            )
            for port, name in self._ports
        )

    @property
    def resource_effects(self):
        return tuple(
            ResourceEffect(
                name,
                (
                    GraphAccess.WRITE
                    if port in ("hits", "hit_indices")
                    or port == "scene"
                    and self._refit
                    else GraphAccess.READ
                ),
            )
            for port, name in self._ports
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
