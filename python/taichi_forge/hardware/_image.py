"""Low-level runtime-ordered image commands."""

from taichi_forge._lib import core as _ti_core
from taichi_forge.graph._ir import GraphAccess, ResourceEffect, RuntimeBinding
from taichi_forge.graph._native import (
    BackendCommandGraphAction,
    BackendCommandRecording,
    NativeGraphExecutable,
    NativeGraphNode,
)
from taichi_forge.lang import impl
from taichi_forge.lang._texture import Texture
from taichi_forge.lang.exception import TaichiRuntimeError


def _active_backend():
    arch = _ti_core.arch_name(impl.current_cfg().arch)
    return "cpu" if arch in ("x64", "arm64") else arch


class VulkanImageCopyRecording(BackendCommandRecording):
    """One reusable whole-image Vulkan color copy command."""

    def __init__(self, *, source="source", destination="destination"):
        for value, label in (
            (source, "source"),
            (destination, "destination"),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} binding must be a nonempty string")
        if source == destination:
            raise ValueError("source and destination bindings must differ")
        program = impl.get_runtime().prog
        if program is None:
            raise TaichiRuntimeError(
                "Vulkan image copy requires an initialized Taichi runtime"
            )
        backend = _active_backend()
        if backend != "vulkan":
            raise TaichiRuntimeError(
                "Vulkan image copy requires the Vulkan backend; the active "
                f"backend is {backend}"
            )
        super().__init__(
            backend="vulkan",
            binding_names=(source, destination),
            command_count=1,
            queue="compute",
            stream_binding="runtime_ordered",
            barrier_policy="internal",
            workspace_ownership="none",
            replay_mode="rerecord",
            no_host_readback=True,
        )
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "destination", destination)
        object.__setattr__(self, "_runtime_prog", program)
        object.__setattr__(
            self, "_runtime_generation", int(impl.runtime_generation())
        )

    @property
    def resource_effects(self):
        return (
            ResourceEffect(self.source, GraphAccess.READ),
            ResourceEffect(self.destination, GraphAccess.WRITE),
        )

    def execute(self, bindings):
        required = frozenset(self.binding_names)
        provided = frozenset(bindings)
        if provided != required:
            missing = sorted(required.difference(provided))
            unexpected = sorted(provided.difference(required))
            details = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if unexpected:
                details.append("unexpected " + ", ".join(unexpected))
            raise TaichiRuntimeError(
                "Vulkan image-copy bindings do not match the recording: "
                + "; ".join(details)
            )
        self.validate_graph_lifetime()
        source = bindings[self.source]
        destination = bindings[self.destination]
        if not isinstance(source, Texture) or not isinstance(destination, Texture):
            raise TaichiRuntimeError(
                "Vulkan image-copy bindings must be Taichi Textures"
            )
        self._runtime_prog._vulkan_copy_texture(destination.tex, source.tex)
        return destination

    def validate_graph_lifetime(self):
        if (
            impl.get_runtime().prog is not self._runtime_prog
            or int(impl.runtime_generation()) != self._runtime_generation
        ):
            raise TaichiRuntimeError(
                "Vulkan image-copy recording belongs to a previous Taichi "
                "runtime generation"
            )

    def _as_graph_native_node(self):
        return _VulkanImageCopyNode(self)


class _VulkanImageCopyExecutable(NativeGraphExecutable):
    def __init__(self, recording):
        self._recording = recording
        self._action = BackendCommandGraphAction(recording)

    def run(self, runtime_args):
        return self._recording.execute(runtime_args)

    @property
    def runtime_arg_schema(self):
        return tuple(
            RuntimeBinding(name, "texture")
            for name in self._recording.binding_names
        )

    @property
    def resource_effects(self):
        return self._recording.resource_effects

    @property
    def recordable_action(self):
        return self._action

    @property
    def debug_info(self):
        return {"kind": "vulkan_image_copy"}


class _VulkanImageCopyNode(NativeGraphNode):
    def __init__(self, recording):
        self._recording = recording

    def compile(self):
        return _VulkanImageCopyExecutable(self._recording)


def copy(destination, source):
    """Copy one complete, format-matched Vulkan color texture."""

    recording = VulkanImageCopyRecording()
    return recording.execute({"source": source, "destination": destination})


__all__ = ["VulkanImageCopyRecording", "copy"]
