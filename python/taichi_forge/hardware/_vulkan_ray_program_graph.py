"""Existing Graph adapter for an explicitly initialized Vulkan RT packet."""

from taichi_forge.graph._ir import GraphAccess, ResourceEffect
from taichi_forge.graph._native import BackendCommandRecording
from taichi_forge.hardware._native_adapter import native_recording_node, static_resource_effect
from taichi_forge.hardware._ray_identity import RayResourceIdentity, identify_ray_recording
from taichi_forge.lang.exception import TaichiRuntimeError


class _PreparedVulkanProgramRecording(BackendCommandRecording):
    def __init__(self, launch):
        launch._require_initialized()
        source = launch.recording
        names = tuple(name for name, binding in source.bindings.items() if binding.kind != "scene")
        super().__init__(
            backend="vulkan",
            binding_names=names,
            command_count=1,
            queue="compute",
            stream_binding="runtime_ordered",
            workspace_ownership="provider_generation",
            barrier_policy="internal",
            replay_mode="rerecord",
            no_host_readback=True,
        )
        object.__setattr__(self, "launch", launch)
        scene_names = tuple(sorted(name for name, binding in source.bindings.items() if binding.kind == "scene"))
        resource = RayResourceIdentity(
            "vulkan_program_resources",
            children=tuple(launch._bindings[name]._effect_name for name in scene_names),
            scene_names=scene_names,
        )
        identify_ray_recording(self, "vulkan_program_launch", resource, program=source.to_dict())

    @property
    def resource_effects(self):
        source = self.launch.recording
        access = {"read": GraphAccess.READ, "write": GraphAccess.WRITE, "read_write": GraphAccess.READ_WRITE}
        return tuple(
            ResourceEffect(name, access[binding.access])
            for name, binding in source.bindings.items()
            if binding.kind != "scene"
        ) + tuple(
            static_resource_effect(self.launch._bindings[name]._effect_name, GraphAccess.READ)
            for name, binding in source.bindings.items()
            if binding.kind == "scene"
        )

    def validate_graph_bindings(self, bindings):
        self.launch._require_initialized()
        if any(bindings[name] is not self.launch._bindings[name] for name in self.binding_names):
            raise TaichiRuntimeError("Vulkan ray Graph uses fixed bindings; prepare a new launch to replace resources")

    def prepare_graph_execute(self, bindings):
        self.validate_graph_bindings(bindings)
        # The same callable also exposes its cold native fixed-command factory.
        return self.launch

    def execute(self, bindings):
        return self.prepare_graph_execute(bindings)()

    def _vulkan_graph_command(self):
        # A fixed initialized packet, not a probe-only command sequence.
        return self.launch._vulkan_graph_command()

    def _graph_provider_memory_dependencies(self):
        return (self.launch,)

    def _as_graph_native_node(self):
        return native_recording_node(
            self,
            runtime_bindings=tuple(
                (name, "texture" if self.launch.recording.bindings[name].kind.endswith("image") else "ndarray")
                for name in self.binding_names
            ),
            lifetime_leases=(self.launch,),
            debug_info={
                "kind": "vulkan_ray_program_prepared_launch",
                "initialization": "explicit_before_graph_binding",
                "execution": "runtime_ordered_trace_rays",
                "fixed_command_available": True,
            },
            publish_time_binding_validation_stable=True,
        )
