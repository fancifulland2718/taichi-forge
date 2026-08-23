"""Low-level Vulkan graphics pipelines and draw-command recordings.

This module deliberately stops below renderer semantics.  Callers provide
SPIR-V, vertex layouts, buffers, and attachments; Forge only owns the native
pipeline object and runtime ordering needed to record the draw.
"""

from dataclasses import dataclass
from types import MappingProxyType

from taichi_forge._lib import core as _ti_core
from taichi_forge.graph._ir import GraphAccess, ResourceEffect, RuntimeBinding
from taichi_forge.graph._native import (
    BackendCommandGraphAction,
    BackendCommandRecording,
    NativeGraphExecutable,
    NativeGraphNode,
)
from taichi_forge.hardware._memory import HardwareMemoryComponent, make_memory_report
from taichi_forge.lang import impl
from taichi_forge.lang._ndarray import Ndarray
from taichi_forge.lang._texture import Texture
from taichi_forge.lang.exception import TaichiRuntimeError


_TOPOLOGIES = {"triangles": 0, "lines": 1, "points": 2}
_POLYGON_MODES = {"fill": 0, "line": 1, "point": 2}
_CULL_MODES = {
    "none": (False, False),
    "front": (True, False),
    "back": (False, True),
}


def _active_backend():
    arch = _ti_core.arch_name(impl.current_cfg().arch)
    return "cpu" if arch in ("x64", "arm64") else arch


def _u32(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    minimum = 1 if positive else 0
    if not minimum <= value <= 0xFFFFFFFF:
        qualifier = "positive " if positive else "nonnegative "
        raise ValueError(f"{name} must be a {qualifier}uint32 value")
    return value


def _i32(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not -(1 << 31) <= value < (1 << 31):
        raise ValueError(f"{name} must be a signed int32 value")
    return value


def _name(value, label):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _bytes(value, label):
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"{label} must be a bytes-like SPIR-V binary")
    value = bytes(value)
    if not value or len(value) % 4:
        raise ValueError(f"{label} must be nonempty and four-byte aligned")
    return value


@dataclass(frozen=True)
class VertexBinding:
    """One Vulkan vertex-buffer binding declaration."""

    binding: int
    stride: int
    instance: bool = False

    def __post_init__(self):
        _u32(self.binding, "binding")
        _u32(self.stride, "stride", positive=True)
        if not isinstance(self.instance, bool):
            raise TypeError("instance must be a bool")


@dataclass(frozen=True)
class VertexAttribute:
    """One shader input location sourced from a vertex binding."""

    location: int
    binding: int
    format: object
    offset: int = 0

    def __post_init__(self):
        _u32(self.location, "location")
        _u32(self.binding, "binding")
        _u32(self.offset, "offset")
        if not isinstance(self.format, _ti_core.Format):
            raise TypeError("format must be a ti.Format value")


@dataclass(frozen=True)
class Draw:
    """Immutable direct or indexed draw range."""

    element_count: int
    instance_count: int = 1
    first_vertex: int = 0
    first_index: int = 0
    first_instance: int = 0
    vertex_offset: int = 0
    index_bounds: tuple | None = None

    def __post_init__(self):
        _u32(self.element_count, "element_count", positive=True)
        _u32(self.instance_count, "instance_count", positive=True)
        _u32(self.first_vertex, "first_vertex")
        _u32(self.first_index, "first_index")
        _u32(self.first_instance, "first_instance")
        _i32(self.vertex_offset, "vertex_offset")
        if self.index_bounds is not None:
            if not isinstance(self.index_bounds, (tuple, list)):
                raise TypeError("index_bounds must contain minimum and maximum indices")
            if len(self.index_bounds) != 2:
                raise ValueError(
                    "index_bounds must contain minimum and maximum indices"
                )
            index_bounds = tuple(
                _u32(value, f"index_bounds[{index}]")
                for index, value in enumerate(self.index_bounds)
            )
            if index_bounds[0] > index_bounds[1]:
                raise ValueError("index_bounds minimum must not exceed maximum")
            object.__setattr__(self, "index_bounds", index_bounds)


class VulkanGraphicsDrawRecording(BackendCommandRecording):
    """One immutable Vulkan graphics draw with runtime-bound resources."""

    def __init__(
        self,
        pipeline,
        draw,
        *,
        color="color",
        vertex_buffers=None,
        depth=None,
        index_buffer=None,
        clear_color=(0.0, 0.0, 0.0, 1.0),
        viewport=None,
    ):
        if not isinstance(pipeline, VulkanGraphicsPipeline):
            raise TypeError(
                "Vulkan graphics recordings require a VulkanGraphicsPipeline"
            )
        if not isinstance(draw, Draw):
            raise TypeError("draw must be a ti.hardware.graphics.Draw value")
        color = _name(color, "color binding")
        if depth is not None:
            depth = _name(depth, "depth binding")
        if index_buffer is not None:
            index_buffer = _name(index_buffer, "index-buffer binding")
            if draw.index_bounds is None:
                raise ValueError("indexed draws require declared index_bounds")
            if draw.first_vertex != 0:
                raise ValueError(
                    "indexed draws use vertex_offset instead of first_vertex"
                )
        else:
            if draw.index_bounds is not None:
                raise ValueError("index_bounds require an indexed draw")
            if draw.vertex_offset != 0:
                raise ValueError("vertex_offset requires an indexed draw")
            if draw.first_index != 0:
                raise ValueError("first_index requires an indexed draw")
        if not isinstance(vertex_buffers, dict):
            raise TypeError("vertex_buffers must map binding integers to names")
        normalized_vertices = {}
        for binding, name in vertex_buffers.items():
            binding = _u32(binding, "vertex-buffer binding")
            normalized_vertices[binding] = _name(name, "vertex-buffer name")
        required_bindings = frozenset(item.binding for item in pipeline.vertex_bindings)
        if frozenset(normalized_vertices) != required_bindings:
            raise ValueError(
                "vertex_buffers must bind exactly the pipeline vertex bindings"
            )

        binding_names = [color]
        if depth is not None:
            binding_names.append(depth)
        binding_names.extend(normalized_vertices[key] for key in sorted(normalized_vertices))
        if index_buffer is not None:
            binding_names.append(index_buffer)
        if len(binding_names) != len(set(binding_names)):
            raise ValueError("graphics resource binding names must be unique")

        clear_color = tuple(float(component) for component in clear_color)
        if len(clear_color) != 4:
            raise ValueError("clear_color must contain four values")
        if viewport is None:
            viewport = (0, 0, 0, 0)
        else:
            viewport = tuple(
                _u32(component, f"viewport[{index}]")
                for index, component in enumerate(viewport)
            )
            if len(viewport) != 4:
                raise ValueError("viewport must contain x, y, width, and height")

        super().__init__(
            backend="vulkan",
            binding_names=tuple(binding_names),
            command_count=1,
            queue="graphics",
            stream_binding="runtime_ordered",
            barrier_policy="internal",
            workspace_ownership="provider_generation",
            replay_mode="rerecord",
            no_host_readback=True,
        )
        object.__setattr__(self, "pipeline", pipeline)
        object.__setattr__(self, "draw", draw)
        object.__setattr__(self, "color", color)
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "vertex_buffers", MappingProxyType(normalized_vertices))
        object.__setattr__(self, "index_buffer", index_buffer)
        object.__setattr__(self, "clear_color", clear_color)
        object.__setattr__(self, "viewport", viewport)

    @property
    def resource_effects(self):
        effects = [ResourceEffect(self.color, GraphAccess.WRITE)]
        if self.depth is not None:
            effects.append(ResourceEffect(self.depth, GraphAccess.WRITE))
        effects.extend(
            ResourceEffect(name, GraphAccess.READ)
            for _, name in sorted(self.vertex_buffers.items())
        )
        if self.index_buffer is not None:
            effects.append(ResourceEffect(self.index_buffer, GraphAccess.READ))
        return tuple(effects)

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
                "Vulkan graphics bindings do not match the recording: "
                + "; ".join(details)
            )
        self.validate_graph_lifetime()
        color = bindings[self.color]
        depth = None if self.depth is None else bindings[self.depth]
        if not isinstance(color, Texture):
            raise TaichiRuntimeError("graphics color binding must be a Texture")
        if depth is not None and not isinstance(depth, Texture):
            raise TaichiRuntimeError("graphics depth binding must be a Texture")
        if any(
            not isinstance(bindings[name], Ndarray)
            for name in self.vertex_buffers.values()
        ):
            raise TaichiRuntimeError("graphics vertex bindings must be Taichi ndarrays")
        vertices = tuple(
            (binding, bindings[name].arr)
            for binding, name in sorted(self.vertex_buffers.items())
        )
        index = None if self.index_buffer is None else bindings[self.index_buffer]
        if index is not None and not isinstance(index, Ndarray):
            raise TaichiRuntimeError("graphics index binding must be a Taichi ndarray")
        draw = self.draw
        index_min, index_max = (
            (0, 0) if draw.index_bounds is None else draw.index_bounds
        )
        self.pipeline._runtime_prog._vulkan_graphics_draw(
            self.pipeline._handle,
            color.tex,
            None if depth is None else depth.tex,
            vertices,
            None if index is None else index.arr,
            draw.element_count,
            draw.instance_count,
            draw.first_vertex,
            draw.first_index,
            draw.first_instance,
            draw.vertex_offset,
            index_min,
            index_max,
            index is not None,
            self.clear_color,
            self.viewport,
        )
        return color

    def validate_graph_lifetime(self):
        self.pipeline._validate_lifetime()

    def memory_report(self):
        return self.pipeline.memory_report()

    def _as_graph_native_node(self):
        return _VulkanGraphicsDrawNode(self)


class _VulkanGraphicsDrawExecutable(NativeGraphExecutable):
    def __init__(self, recording):
        self._recording = recording
        self._action = BackendCommandGraphAction(recording)

    def run(self, runtime_args):
        return self._recording.execute(runtime_args)

    @property
    def runtime_arg_schema(self):
        texture_names = {self._recording.color, self._recording.depth}
        return tuple(
            RuntimeBinding(name, "texture" if name in texture_names else "ndarray")
            for name in self._recording.binding_names
        )

    @property
    def resource_effects(self):
        return self._recording.resource_effects

    @property
    def lifetime_leases(self):
        return (self._recording.pipeline,)

    @property
    def recordable_action(self):
        return self._action

    @property
    def debug_info(self):
        return {
            "kind": "vulkan_graphics_draw",
            "indexed": self._recording.index_buffer is not None,
            "vertex_binding_count": len(self._recording.vertex_buffers),
        }


class _VulkanGraphicsDrawNode(NativeGraphNode):
    def __init__(self, recording):
        self._recording = recording

    def compile(self):
        return _VulkanGraphicsDrawExecutable(self._recording)


class VulkanGraphicsPipeline:
    """A caller-defined Vulkan raster pipeline, without renderer semantics."""

    def __init__(
        self,
        vertex_spirv,
        fragment_spirv,
        *,
        vertex_bindings,
        vertex_attributes,
        topology="triangles",
        polygon_mode="fill",
        cull_mode="none",
        depth_test=False,
        depth_write=False,
        blending=False,
        name="",
    ):
        program = impl.get_runtime().prog
        if program is None:
            raise TaichiRuntimeError(
                "VulkanGraphicsPipeline requires an initialized Taichi runtime"
            )
        if _active_backend() != "vulkan":
            raise TaichiRuntimeError(
                "VulkanGraphicsPipeline requires the Vulkan backend; the active "
                f"backend is {_active_backend()}"
            )
        if not program.vulkan_graphics_pipeline_available():
            raise TaichiRuntimeError(
                "Vulkan graphics commands are unavailable in this build/runtime"
            )
        vertex_bindings = tuple(vertex_bindings)
        vertex_attributes = tuple(vertex_attributes)
        if not vertex_bindings or not all(
            isinstance(item, VertexBinding) for item in vertex_bindings
        ):
            raise TypeError("vertex_bindings must contain VertexBinding values")
        if not vertex_attributes or not all(
            isinstance(item, VertexAttribute) for item in vertex_attributes
        ):
            raise TypeError("vertex_attributes must contain VertexAttribute values")
        try:
            topology_value = _TOPOLOGIES[topology]
        except (KeyError, TypeError) as exc:
            raise ValueError("unsupported graphics topology") from exc
        try:
            polygon_value = _POLYGON_MODES[polygon_mode]
        except (KeyError, TypeError) as exc:
            raise ValueError("unsupported graphics polygon_mode") from exc
        try:
            front_cull, back_cull = _CULL_MODES[cull_mode]
        except (KeyError, TypeError) as exc:
            raise ValueError("unsupported graphics cull_mode") from exc
        if not isinstance(name, str):
            raise TypeError("name must be a string")

        self._runtime_prog = program
        self._runtime_generation = int(impl.runtime_generation())
        self.vertex_bindings = vertex_bindings
        self.vertex_attributes = vertex_attributes
        self._handle = int(
            program._create_vulkan_graphics_pipeline(
                _bytes(vertex_spirv, "vertex_spirv"),
                _bytes(fragment_spirv, "fragment_spirv"),
                tuple(
                    (item.binding, item.stride, item.instance)
                    for item in vertex_bindings
                ),
                tuple(
                    (item.location, item.binding, item.format, item.offset)
                    for item in vertex_attributes
                ),
                topology_value,
                polygon_value,
                front_cull,
                back_cull,
                bool(depth_test),
                bool(depth_write),
                bool(blending),
                name,
            )
        )

    @property
    def closed(self):
        return self._handle is None

    def record(self, draw, **kwargs):
        self._validate_lifetime()
        return VulkanGraphicsDrawRecording(self, draw, **kwargs)

    def draw(
        self,
        color,
        vertex_buffers,
        *,
        draw,
        depth=None,
        index_buffer=None,
        clear_color=(0.0, 0.0, 0.0, 1.0),
        viewport=None,
    ):
        symbolic_vertices = {
            binding: f"vertex_{binding}" for binding in vertex_buffers
        }
        recording = self.record(
            draw,
            color="color",
            depth=None if depth is None else "depth",
            vertex_buffers=symbolic_vertices,
            index_buffer=None if index_buffer is None else "index",
            clear_color=clear_color,
            viewport=viewport,
        )
        bindings = {"color": color}
        if depth is not None:
            bindings["depth"] = depth
        bindings.update(
            (symbolic_vertices[binding], resource)
            for binding, resource in vertex_buffers.items()
        )
        if index_buffer is not None:
            bindings["index"] = index_buffer
        recording.execute(bindings)
        return color

    def _validate_lifetime(self):
        if self._handle is None:
            raise TaichiRuntimeError("VulkanGraphicsPipeline has been closed")
        if (
            impl.get_runtime().prog is not self._runtime_prog
            or int(impl.runtime_generation()) != self._runtime_generation
        ):
            raise TaichiRuntimeError(
                "VulkanGraphicsPipeline belongs to a previous Taichi runtime generation"
            )

    def validate_graph_lifetime(self):
        self._validate_lifetime()

    def memory_report(self):
        handle_present = self._handle is not None
        runtime_valid = handle_present and (
            impl.get_runtime().prog is self._runtime_prog
            and int(impl.runtime_generation()) == self._runtime_generation
        )
        return make_memory_report(
            "vulkan_graphics_pipeline",
            "vulkan",
            (
                HardwareMemoryComponent(
                    "pipeline_shader_modules_and_driver_state",
                    None,
                    False,
                    "provider_generation",
                    "driver",
                    resident=runtime_valid,
                ),
            ),
            lifecycle_state=(
                "ready"
                if runtime_valid
                else "closed"
                if not handle_present
                else "runtime_invalid"
            ),
            ownership_scope="pipeline_generation",
        )

    def _graph_provider_memory_report(self):
        return self.memory_report()

    def close(self):
        if self._handle is None:
            return None
        handle = self._handle
        self._handle = None
        if (
            impl.get_runtime().prog is self._runtime_prog
            and int(impl.runtime_generation()) == self._runtime_generation
        ):
            self._runtime_prog._destroy_vulkan_graphics_pipeline(handle)
        return None

    destroy = close

    def __enter__(self):
        self._validate_lifetime()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def is_available():
    """Return whether the initialized runtime accepts Vulkan graphics draws."""

    program = impl.get_runtime().prog
    return bool(
        program is not None
        and _active_backend() == "vulkan"
        and program.vulkan_graphics_pipeline_available()
    )


__all__ = [
    "Draw",
    "VertexAttribute",
    "VertexBinding",
    "VulkanGraphicsDrawRecording",
    "VulkanGraphicsPipeline",
    "is_available",
]
