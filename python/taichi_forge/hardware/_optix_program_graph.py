"""Graph adapter for an explicitly initialized, fixed OptiX launch packet."""

import hashlib
import json

from taichi_forge.graph._ir import GraphAccess, ResourceEffect
from taichi_forge.graph._native import BackendCommandRecording
from taichi_forge.hardware._native_adapter import native_recording_node, static_resource_effect
from taichi_forge.hardware._ray_identity import RayResourceIdentity, identify_ray_recording, program_sbt_contract
from taichi_forge.lang.exception import TaichiRuntimeError


class _PreparedProgramRecording(BackendCommandRecording):
    def __init__(self, launch):
        launch._require_initialized()
        source = launch.recording
        super().__init__(
            backend="cuda",
            binding_names=source.binding_names,
            command_count=1,
            workspace_ownership="provider_generation",
            barrier_policy="internal",
            replay_mode="rerecord",
            no_host_readback=True,
        )
        object.__setattr__(self, "launch", launch)
        scene_names = tuple(sorted(source.scenes))
        resource = RayResourceIdentity(
            "optix_program_resources",
            children=tuple(source.scenes[name]._effect_name for name in scene_names),
            scene_names=scene_names,
        )
        identify_ray_recording(
            self,
            "optix_program_launch",
            resource,
            program_id=source.program.program_id,
            dimensions=source.dimensions,
            parameters=source._parameters.to_dict(),
            raygen=source.raygen.to_dict(),
            miss=tuple(item.to_dict() for item in source.miss),
            hit=tuple(item.to_dict() for item in source.hit),
        )
        object.__setattr__(self, "_source_resource_plan_json", resource._plan_json)

    def _graph_source_contract(self):
        source = self.launch.recording
        parameters = source._parameters.to_dict()
        parameters.pop("scalar_bytes")
        parameters["scalar_sha256"] = hashlib.sha256(source._parameters.scalar_bytes).hexdigest()
        return {
            "kind": "optix_program",
            "source": "provider_declared_not_measured",
            "program": source.program.to_dict(),
            "launch": {
                "dimensions": source.dimensions,
                "parameters": parameters,
                "sbt": {
                    name: program_sbt_contract(records)
                    for name, records in (("raygen", (source.raygen,)), ("miss", source.miss), ("hit", source.hit))
                },
            },
            "resources": json.loads(self._source_resource_plan_json),
            "execution": {
                "initialization": "explicit_before_graph_binding",
                "ordinary": "runtime_ordered_rerecord",
                "capture": "unavailable",
                "capture_reason": "optix_program_launch_not_capture_supported",
                "compiler": "caller_supplied_optix_ptx",
            },
        }

    @property
    def resource_effects(self):
        source = self.launch.recording
        access = {"read": GraphAccess.READ, "write": GraphAccess.WRITE, "read_write": GraphAccess.READ_WRITE}
        return tuple(
            ResourceEffect(name, access[mode]) for name, (kind, mode) in source._resources.items() if kind != "scene"
        ) + tuple(
            static_resource_effect(scene._effect_name, GraphAccess.READ)
            for scene in dict.fromkeys(source.scenes.values())
        )

    def validate_graph_bindings(self, bindings):
        self.launch._require_initialized()
        if any(bindings[name] is not self.launch._bindings[name] for name in self.binding_names):
            raise TaichiRuntimeError(
                "OptiX Graph recording has fixed bindings; prepare and initialize a new launch for replacement resources"
            )

    def prepare_graph_execute(self, bindings):
        # Cold binding proof only. Never allocate, initialize or upload here.
        self.validate_graph_bindings(bindings)
        return self.launch.run

    def execute(self, bindings):
        return self.prepare_graph_execute(bindings)()

    def _graph_provider_memory_dependencies(self):
        return (self.launch,)

    def _as_graph_native_node(self):
        return native_recording_node(
            self,
            runtime_bindings=tuple(
                (name, "texture" if self.launch.recording._resources[name][0] == "texture" else "ndarray")
                for name in self.binding_names
            ),
            lifetime_leases=(self.launch,),
            debug_info={
                "kind": "optix_program_prepared_launch",
                "execution": "runtime_ordered_native_launch",
                "initialization": "explicit_before_graph_binding",
                "capture": "unavailable",
                "capture_reason": "optix_program_launch_not_capture_supported",
            },
            # Exact object/layout binding does not change within a BindingVersion.
            # Native prepared storage and the launch callable own retirement.
            publish_time_binding_validation_stable=True,
        )
