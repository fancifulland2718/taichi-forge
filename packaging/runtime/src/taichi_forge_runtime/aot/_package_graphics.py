"""Package Host constructors for direct Vulkan vertex draws."""

from contextlib import ExitStack
from dataclasses import asdict, dataclass
from functools import partial
import struct

from ._package_native import NativeObject, NativeRecording
from ._runtime import _name


@dataclass(frozen=True)
class VertexBinding:
    binding: int
    stride: int
    instance: bool = False


@dataclass(frozen=True)
class VertexAttribute:
    location: int
    binding: int
    format: str
    offset: int = 0


@dataclass(frozen=True)
class Draw:
    element_count: int
    instance_count: int = 1
    first_vertex: int = 0
    first_index: int = 0
    first_instance: int = 0
    vertex_offset: int = 0


@dataclass(frozen=True)
class _PassDraw:
    pipeline: object
    draw: Draw
    vertices: tuple


class _Pipeline(NativeObject):
    def __init__(self, package, vertex_spirv, fragment_spirv, *, vertex_bindings, vertex_attributes,
                 depth_test=False, depth_write=False, depth_compare="greater_equal"):
        runtime = package.runtime
        vertices = tuple(vertex_bindings)
        parameters = dict(shader_count=2, shaders=[dict(stage="vertex"), dict(stage="fragment")],
                          topology="triangles", polygon_mode="fill", depth_test=depth_test,
                          depth_write=depth_write, depth_compare=depth_compare,
                          vertices=[asdict(v) for v in vertices], attributes=[asdict(v) for v in vertex_attributes])
        with ExitStack() as inputs:
            shaders = {f"shader_{i}": inputs.enter_context(runtime.host_blob(code))
                       for i, code in enumerate((vertex_spirv, fragment_spirv))}
            resource = runtime.native_resource("vulkan_graphics", "pipeline", parameters=parameters, bindings=shaders)
            # The factory reports which input handles the pipeline still owns.
            # NativeResource retains those handles until its own retirement.
            inputs.pop_all()
        self._vertex_slots = frozenset(v.binding for v in vertices)
        super().__init__(package, resource)

    def pass_draw(self, draw, *, vertex_buffers):
        self._check()
        if not isinstance(draw, Draw) or set(vertex_buffers) != self._vertex_slots:
            raise ValueError("Direct draw requires every declared vertex binding")
        for name in vertex_buffers.values():
            _name(name)
        return _PassDraw(self, draw, tuple(sorted(vertex_buffers.items())))

    def record_pass(self, draws, *, color="color", colors=None, depth=None, clear_depth=0.0):
        self._check()
        targets = (color,) if colors is None else tuple(colors)
        for target in targets:
            _name(target)
        if depth is not None:
            _name(depth)
        if not targets and depth is None:
            raise ValueError("Vulkan pass requires a color or depth attachment")
        if len(set(targets)) != len(targets) or depth in targets:
            raise ValueError("Vulkan attachments must be distinct")
        bindings, schemas, effects = {}, {}, {}

        def port(key, value, kind, effect):
            bindings[key], schemas[key], effects[key] = value, kind, effect

        # The pipeline is captured directly. External vertex buffers are byte
        # ranges; a kernel sharing a port can provide a more precise array type.
        port("pipeline_0", self, dict(kind="native_resource"), "read")
        slots, descriptions = {}, []
        for item in draws:
            if not isinstance(item, _PassDraw) or item.pipeline is not self:
                raise ValueError("Package direct draws must use the recording pipeline")
            vertices = []
            for binding, name in item.vertices:
                if name not in slots:
                    slots[name] = len(slots)
                    port(f"array_{slots[name]}", name, dict(kind="ndarray"), "read")
                vertices.append(dict(binding=binding, array=slots[name]))
            descriptions.append(dict(pipeline=0, vertices=vertices, buffers=[], images=[], **asdict(item.draw)))
        if not descriptions:
            raise ValueError("Vulkan pass requires at least one draw")
        for i, target in enumerate(targets):
            port(f"color_{i}", target, dict(kind="texture", ndim=2), "write")
        if depth is not None:
            port("depth", depth, dict(kind="texture", ndim=2), "write")
        clear = struct.unpack("<4I", struct.pack("<4f", 0, 0, 0, 1))
        params = dict(pipeline_count=1, array_count=len(slots), image_count=0, color_count=len(targets),
                      colors=[dict(clear=True, clear_bits=clear) for _ in targets],
                      clear_depth=clear_depth, depth_clear=True, draws=descriptions)
        return NativeRecording(self.package, "vulkan_graphics", "pass_direct", parameters=params,
                               bindings=bindings, schemas=schemas, effects=effects)


class GraphicsAPI:
    VertexBinding = VertexBinding
    VertexAttribute = VertexAttribute
    Draw = Draw

    def __init__(self, package):
        self.VulkanGraphicsPipeline = partial(_Pipeline, package)
