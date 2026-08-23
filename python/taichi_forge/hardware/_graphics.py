"""Low-level Vulkan graphics pipelines and draw-command recordings.

This module deliberately stops below renderer semantics.  Callers provide
SPIR-V, vertex layouts, buffers, and attachments; Forge only owns the native
pipeline object and runtime ordering needed to record the draw.
"""

from dataclasses import dataclass
from types import MappingProxyType

from taichi_forge._lib import core as _ti_core
from taichi_forge.graph._ir import GraphAccess, ResourceEffect
from taichi_forge.graph._native import BackendCommandRecording
from taichi_forge._hardware_telemetry import instrument_hardware_recording
from taichi_forge.hardware._memory import HardwareMemoryComponent, make_memory_report
from taichi_forge.hardware._native_adapter import (
    native_recording_node,
    runtime_generation_matches,
    validate_exact_bindings,
    validate_runtime_generation,
)
from taichi_forge.hardware._runtime import active_backend
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
_ATTACHMENT_LOAD_OPS = frozenset(("clear", "load"))
_ATTACHMENT_STORE_OPS = frozenset(("store",))
_SHADER_BUFFER_KINDS = frozenset(("uniform", "storage"))
_SHADER_BUFFER_ACCESSES = frozenset(("read", "write", "read_write"))

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


@dataclass(frozen=True)
class ShaderBufferBinding:
    """One SPIR-V descriptor buffer declared by a graphics pipeline."""

    set_index: int
    binding: int
    kind: str = "storage"
    access: str = "read"

    def __post_init__(self):
        _u32(self.set_index, "set_index")
        _u32(self.binding, "binding")
        if self.kind not in _SHADER_BUFFER_KINDS:
            raise ValueError("shader buffer kind must be 'uniform' or 'storage'")
        if self.access not in _SHADER_BUFFER_ACCESSES:
            raise ValueError(
                "shader buffer access must be 'read', 'write', or 'read_write'"
            )
        if self.kind == "uniform" and self.access != "read":
            raise ValueError("uniform shader buffers are read-only")


@dataclass(frozen=True)
class GraphicsPassDraw:
    """One pipeline and its symbolic resources inside a graphics pass."""

    pipeline: object
    draw: Draw
    vertex_buffers: object
    index_buffer: str | None = None
    shader_buffers: object = None

    def __post_init__(self):
        if not isinstance(self.pipeline, VulkanGraphicsPipeline):
            raise TypeError("pipeline must be a VulkanGraphicsPipeline")
        if not isinstance(self.draw, Draw):
            raise TypeError("draw must be a ti.hardware.graphics.Draw value")
        if not isinstance(self.vertex_buffers, dict):
            raise TypeError("vertex_buffers must map binding integers to names")
        vertices = {}
        for binding, name in self.vertex_buffers.items():
            vertices[_u32(binding, "vertex-buffer binding")] = _name(
                name, "vertex-buffer name"
            )
        required_vertices = frozenset(
            item.binding for item in self.pipeline.vertex_bindings
        )
        if frozenset(vertices) != required_vertices:
            raise ValueError(
                "vertex_buffers must bind exactly the pipeline vertex bindings"
            )

        index_buffer = self.index_buffer
        if index_buffer is not None:
            index_buffer = _name(index_buffer, "index-buffer binding")
            if self.draw.index_bounds is None:
                raise ValueError("indexed draws require declared index_bounds")
            if self.draw.first_vertex != 0:
                raise ValueError(
                    "indexed draws use vertex_offset instead of first_vertex"
                )
        else:
            if self.draw.index_bounds is not None:
                raise ValueError("index_bounds require an indexed draw")
            if self.draw.vertex_offset != 0:
                raise ValueError("vertex_offset requires an indexed draw")
            if self.draw.first_index != 0:
                raise ValueError("first_index requires an indexed draw")

        shader_buffers = {} if self.shader_buffers is None else self.shader_buffers
        if not isinstance(shader_buffers, dict):
            raise TypeError(
                "shader_buffers must map (set_index, binding) pairs to names"
            )
        normalized_shader = {}
        for key, name in shader_buffers.items():
            if not isinstance(key, (tuple, list)) or len(key) != 2:
                raise TypeError("shader-buffer keys must contain set_index and binding")
            normalized_key = (
                _u32(key[0], "shader-buffer set_index"),
                _u32(key[1], "shader-buffer binding"),
            )
            normalized_shader[normalized_key] = _name(name, "shader-buffer name")
        required_shader = frozenset(
            (item.set_index, item.binding)
            for item in self.pipeline.shader_buffer_bindings
        )
        if frozenset(normalized_shader) != required_shader:
            raise ValueError(
                "shader_buffers must bind exactly the pipeline shader buffers"
            )

        object.__setattr__(self, "vertex_buffers", MappingProxyType(vertices))
        object.__setattr__(self, "index_buffer", index_buffer)
        object.__setattr__(self, "shader_buffers", MappingProxyType(normalized_shader))


@instrument_hardware_recording("raster.draw.vulkan")
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
        if pipeline.shader_buffer_bindings:
            raise ValueError(
                "pipelines with shader buffers require pass_draw()/record_pass()"
            )
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
        binding_names.extend(
            normalized_vertices[key] for key in sorted(normalized_vertices)
        )
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
        object.__setattr__(
            self, "vertex_buffers", MappingProxyType(normalized_vertices)
        )
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
        validate_exact_bindings(self, bindings, "Vulkan graphics")
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
        raw_draw = (
            self.pipeline._handle,
            vertices,
            None if index is None else index.arr,
            (),
            draw.element_count,
            draw.instance_count,
            draw.first_vertex,
            draw.first_index,
            draw.first_instance,
            draw.vertex_offset,
            index_min,
            index_max,
            index is not None,
        )
        self.pipeline._runtime_prog._vulkan_graphics_pass(
            color.tex,
            None if depth is None else depth.tex,
            (raw_draw,),
            True,
            depth is not None,
            self.clear_color,
            self.viewport,
        )
        return color

    def validate_graph_lifetime(self):
        self.pipeline._validate_lifetime()

    def memory_report(self):
        return self.pipeline.memory_report()

    def _as_graph_native_node(self):
        return native_recording_node(
            self,
            runtime_bindings=lambda item: tuple(
                (
                    name,
                    "texture"
                    if name in {item.color, item.depth}
                    else "ndarray",
                )
                for name in item.binding_names
            ),
            lifetime_leases=lambda item: (item.pipeline,),
            debug_info=lambda item: {
                "kind": "vulkan_graphics_draw",
                "indexed": item.index_buffer is not None,
                "vertex_binding_count": len(item.vertex_buffers),
            },
        )


def _merge_graphics_access(left, right):
    if left is None or left == right:
        return right if left is None else left
    if GraphAccess.READ_WRITE in (left, right):
        return GraphAccess.READ_WRITE
    return GraphAccess.READ_WRITE


@instrument_hardware_recording("raster.draw.vulkan")
class VulkanGraphicsPassRecording(BackendCommandRecording):
    """One renderer-neutral Vulkan render pass containing one or more draws."""

    def __init__(
        self,
        draws,
        *,
        color="color",
        depth=None,
        color_load_op="clear",
        color_store_op="store",
        depth_load_op="clear",
        depth_store_op="store",
        clear_color=(0.0, 0.0, 0.0, 1.0),
        viewport=None,
    ):
        draws = tuple(draws)
        if not draws or len(draws) > (1 << 20):
            raise ValueError("graphics passes require 1 to 1048576 draws")
        if not all(isinstance(item, GraphicsPassDraw) for item in draws):
            raise TypeError("draws must contain GraphicsPassDraw values")
        for item in draws:
            item.pipeline._validate_lifetime()

        color = _name(color, "color binding")
        if depth is not None:
            depth = _name(depth, "depth binding")
            if depth == color:
                raise ValueError("color and depth bindings must be different")
        if color_load_op not in _ATTACHMENT_LOAD_OPS:
            raise ValueError("color_load_op must be 'clear' or 'load'")
        if depth_load_op not in _ATTACHMENT_LOAD_OPS:
            raise ValueError("depth_load_op must be 'clear' or 'load'")
        if color_store_op not in _ATTACHMENT_STORE_OPS:
            raise ValueError("the current Vulkan RHI only supports color store")
        if depth_store_op not in _ATTACHMENT_STORE_OPS:
            raise ValueError("the current Vulkan RHI only supports depth store")

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

        effects = {}
        effects[color] = (
            GraphAccess.WRITE if color_load_op == "clear" else GraphAccess.READ_WRITE
        )
        if depth is not None:
            effects[depth] = (
                GraphAccess.WRITE
                if depth_load_op == "clear"
                else GraphAccess.READ_WRITE
            )

        pipelines = []
        pipeline_ids = set()
        attachment_names = frozenset(
            name for name in (color, depth) if name is not None
        )
        for item in draws:
            if id(item.pipeline) not in pipeline_ids:
                pipelines.append(item.pipeline)
                pipeline_ids.add(id(item.pipeline))
            for name in item.vertex_buffers.values():
                if name in attachment_names:
                    raise ValueError(
                        "graphics attachment bindings cannot also name ndarray resources"
                    )
                effects[name] = _merge_graphics_access(
                    effects.get(name), GraphAccess.READ
                )
            if item.index_buffer is not None:
                if item.index_buffer in attachment_names:
                    raise ValueError(
                        "graphics attachment bindings cannot also name ndarray resources"
                    )
                effects[item.index_buffer] = _merge_graphics_access(
                    effects.get(item.index_buffer), GraphAccess.READ
                )
            for key, name in item.shader_buffers.items():
                if name in attachment_names:
                    raise ValueError(
                        "graphics attachment bindings cannot also name ndarray resources"
                    )
                declaration = item.pipeline._shader_buffer_by_key[key]
                access = {
                    "read": GraphAccess.READ,
                    "write": GraphAccess.WRITE,
                    "read_write": GraphAccess.READ_WRITE,
                }[declaration.access]
                effects[name] = _merge_graphics_access(effects.get(name), access)

        ndarray_names = frozenset(effects).difference(attachment_names)

        super().__init__(
            backend="vulkan",
            binding_names=tuple(effects),
            command_count=1,
            queue="graphics",
            stream_binding="runtime_ordered",
            barrier_policy="internal",
            workspace_ownership="provider_generation",
            replay_mode="rerecord",
            no_host_readback=True,
        )
        object.__setattr__(self, "draws", draws)
        object.__setattr__(self, "pipelines", tuple(pipelines))
        object.__setattr__(self, "color", color)
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "color_load_op", color_load_op)
        object.__setattr__(self, "color_store_op", color_store_op)
        object.__setattr__(self, "depth_load_op", depth_load_op)
        object.__setattr__(self, "depth_store_op", depth_store_op)
        object.__setattr__(self, "clear_color", clear_color)
        object.__setattr__(self, "viewport", viewport)
        object.__setattr__(
            self,
            "_resource_effects",
            tuple(ResourceEffect(name, access) for name, access in effects.items()),
        )
        object.__setattr__(self, "_ndarray_names", ndarray_names)

    @property
    def resource_effects(self):
        return self._resource_effects

    def execute(self, bindings):
        validate_exact_bindings(self, bindings, "Vulkan graphics pass")
        self.validate_graph_lifetime()
        color = bindings[self.color]
        depth = None if self.depth is None else bindings[self.depth]
        if not isinstance(color, Texture):
            raise TaichiRuntimeError("graphics color binding must be a Texture")
        if depth is not None and not isinstance(depth, Texture):
            raise TaichiRuntimeError("graphics depth binding must be a Texture")
        if any(not isinstance(bindings[name], Ndarray) for name in self._ndarray_names):
            raise TaichiRuntimeError(
                "graphics vertex, index, and shader bindings must be Taichi ndarrays"
            )

        raw_draws = []
        for item in self.draws:
            vertices = tuple(
                (binding, bindings[name].arr)
                for binding, name in sorted(item.vertex_buffers.items())
            )
            index = (
                None if item.index_buffer is None else bindings[item.index_buffer].arr
            )
            shader_buffers = []
            for key, name in sorted(item.shader_buffers.items()):
                declaration = item.pipeline._shader_buffer_by_key[key]
                shader_buffers.append(
                    (
                        key[0],
                        key[1],
                        bindings[name].arr,
                        declaration.kind == "storage",
                    )
                )
            draw = item.draw
            index_min, index_max = (
                (0, 0) if draw.index_bounds is None else draw.index_bounds
            )
            raw_draws.append(
                (
                    item.pipeline._handle,
                    vertices,
                    index,
                    tuple(shader_buffers),
                    draw.element_count,
                    draw.instance_count,
                    draw.first_vertex,
                    draw.first_index,
                    draw.first_instance,
                    draw.vertex_offset,
                    index_min,
                    index_max,
                    index is not None,
                )
            )

        self.pipelines[0]._runtime_prog._vulkan_graphics_pass(
            color.tex,
            None if depth is None else depth.tex,
            tuple(raw_draws),
            self.color_load_op == "clear",
            depth is not None and self.depth_load_op == "clear",
            self.clear_color,
            self.viewport,
        )
        return color

    def validate_graph_lifetime(self):
        for pipeline in self.pipelines:
            pipeline._validate_lifetime()
        programs = {id(pipeline._runtime_prog) for pipeline in self.pipelines}
        if len(programs) != 1:
            raise TaichiRuntimeError(
                "all graphics-pass pipelines must belong to one runtime"
            )

    def memory_report(self):
        reports = tuple(pipeline.memory_report() for pipeline in self.pipelines)
        return make_memory_report(
            "vulkan_graphics_pass",
            "vulkan",
            tuple(
                HardwareMemoryComponent(
                    f"pipeline_{index}_shader_modules_and_driver_state",
                    None,
                    False,
                    "provider_generation",
                    "driver",
                    resident=report.lifecycle_state == "ready",
                )
                for index, report in enumerate(reports)
            ),
            lifecycle_state=(
                "ready"
                if all(report.lifecycle_state == "ready" for report in reports)
                else "runtime_invalid"
            ),
            ownership_scope="pass_generation",
        )

    def _as_graph_native_node(self):
        return native_recording_node(
            self,
            runtime_bindings=lambda item: tuple(
                (
                    name,
                    "texture"
                    if name in {item.color, item.depth}
                    else "ndarray",
                )
                for name in item.binding_names
            ),
            lifetime_leases=lambda item: item.pipelines,
            debug_info=lambda item: {
                "kind": "vulkan_graphics_pass",
                "draw_count": len(item.draws),
                "pipeline_count": len(item.pipelines),
                "indexed_draw_count": sum(
                    draw.index_buffer is not None for draw in item.draws
                ),
                "color_load_op": item.color_load_op,
                "depth_load_op": item.depth_load_op,
            },
        )


class VulkanGraphicsPipeline:
    """A caller-defined Vulkan raster pipeline, without renderer semantics."""

    def __init__(
        self,
        vertex_spirv,
        fragment_spirv,
        *,
        vertex_bindings,
        vertex_attributes,
        shader_buffer_bindings=(),
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
        if active_backend() != "vulkan":
            raise TaichiRuntimeError(
                "VulkanGraphicsPipeline requires the Vulkan backend; the active "
                f"backend is {active_backend()}"
            )
        if not program.vulkan_graphics_pipeline_available():
            raise TaichiRuntimeError(
                "Vulkan graphics commands are unavailable in this build/runtime"
            )
        vertex_bindings = tuple(vertex_bindings)
        vertex_attributes = tuple(vertex_attributes)
        shader_buffer_bindings = tuple(shader_buffer_bindings)
        if not vertex_bindings or not all(
            isinstance(item, VertexBinding) for item in vertex_bindings
        ):
            raise TypeError("vertex_bindings must contain VertexBinding values")
        if not vertex_attributes or not all(
            isinstance(item, VertexAttribute) for item in vertex_attributes
        ):
            raise TypeError("vertex_attributes must contain VertexAttribute values")
        if not all(
            isinstance(item, ShaderBufferBinding) for item in shader_buffer_bindings
        ):
            raise TypeError(
                "shader_buffer_bindings must contain ShaderBufferBinding values"
            )
        shader_buffer_by_key = {
            (item.set_index, item.binding): item for item in shader_buffer_bindings
        }
        if len(shader_buffer_by_key) != len(shader_buffer_bindings):
            raise ValueError("shader buffer set/binding pairs must be unique")
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
        self.shader_buffer_bindings = shader_buffer_bindings
        self._shader_buffer_by_key = MappingProxyType(shader_buffer_by_key)
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

    def pass_draw(
        self,
        draw,
        *,
        vertex_buffers,
        index_buffer=None,
        shader_buffers=None,
    ):
        self._validate_lifetime()
        return GraphicsPassDraw(
            self,
            draw,
            vertex_buffers,
            index_buffer=index_buffer,
            shader_buffers=shader_buffers,
        )

    def record_pass(self, draws, **kwargs):
        self._validate_lifetime()
        recording = VulkanGraphicsPassRecording(draws, **kwargs)
        if self not in recording.pipelines:
            raise ValueError("record_pass draws must include the owning pipeline")
        return recording

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
        symbolic_vertices = {binding: f"vertex_{binding}" for binding in vertex_buffers}
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
        validate_runtime_generation(
            self,
            "VulkanGraphicsPipeline belongs to a previous Taichi runtime generation",
        )

    def validate_graph_lifetime(self):
        self._validate_lifetime()

    def memory_report(self):
        handle_present = self._handle is not None
        runtime_valid = handle_present and runtime_generation_matches(self)
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
                else "closed" if not handle_present else "runtime_invalid"
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
        if runtime_generation_matches(self):
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
        and active_backend() == "vulkan"
        and program.vulkan_graphics_pipeline_available()
    )


__all__ = [
    "Draw",
    "GraphicsPassDraw",
    "ShaderBufferBinding",
    "VertexAttribute",
    "VertexBinding",
    "VulkanGraphicsDrawRecording",
    "VulkanGraphicsPassRecording",
    "VulkanGraphicsPipeline",
    "is_available",
]
