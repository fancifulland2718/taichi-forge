"""Immutable Vulkan pipeline state shared by creation and reconstruction."""

from dataclasses import dataclass, field
import hashlib
import struct

from taichi_forge._lib import core
from taichi_forge.graph._native_definition import NativeDefinition


@dataclass(frozen=True)
class GraphicsPipelineDefinition:
    shaders: tuple
    vertices: tuple
    attributes: tuple
    topology: int
    polygon_mode: int
    front_cull: bool
    back_cull: bool
    depth_test: bool
    depth_write: bool
    depth: tuple
    targets: tuple
    blending: bool
    native: NativeDefinition = field(init=False)

    def __post_init__(self):
        spec = pipeline_description(
            self.shaders,
            self.vertices,
            self.attributes,
            topology=self.topology,
            polygon_mode=self.polygon_mode,
            front_cull=self.front_cull,
            back_cull=self.back_cull,
            depth_test=self.depth_test,
            depth_write=self.depth_write,
            depth=self.depth,
            targets=self.targets,
            blending=self.blending,
        )
        version = 2 if any(shader.stage == "mesh" for shader in self.shaders) else 1
        object.__setattr__(self, "native", NativeDefinition("vulkan_graphics", "pipeline", spec, version))

    def driver_arguments(self, name):
        return (
            self.topology,
            self.polygon_mode,
            self.front_cull,
            self.back_cull,
            self.depth_test,
            self.depth_write,
            self.blending,
            name,
            *self.depth,
            self.targets,
        )


def pipeline_description(
    shaders,
    vertices,
    attributes,
    *,
    topology,
    polygon_mode,
    front_cull,
    back_cull,
    depth_test,
    depth_write,
    depth,
    targets,
    blending,
):
    """The shared cold code/layout contract, without constructing a GPU pipeline."""
    from ._graphics import _BLEND_FACTORS, _BLEND_OPS, _DEPTH_COMPARE_OPS, _POLYGON_MODES, _TOPOLOGIES

    if blending and not targets:
        targets = ((True, 15, 6, 7, 0, 6, 7, 0),)
    factors, ops = tuple(_BLEND_FACTORS), tuple(_BLEND_OPS)
    spec = dict(
        shader_count=len(shaders),
        shaders=[dict(stage=value.stage) for value in shaders],
        vertices=[dict(binding=v.binding, stride=v.stride, instance=v.instance) for v in vertices],
        attributes=[
            dict(location=v.location, binding=v.binding, format=v.format.name, offset=v.offset) for v in attributes
        ],
        topology=tuple(_TOPOLOGIES)[topology],
        polygon_mode=tuple(_POLYGON_MODES)[polygon_mode],
        front_face_cull=bool(front_cull),
        back_face_cull=bool(back_cull),
        depth_test=bool(depth_test),
        depth_write=bool(depth_write),
        depth_compare=tuple(_DEPTH_COMPARE_OPS)[depth[0]],
        depth_bias_constant=depth[1],
        depth_bias_slope=depth[2],
        color_targets=[
            dict(
                blending=v[0],
                write_mask=v[1],
                src_color_factor=factors[v[2]],
                dst_color_factor=factors[v[3]],
                color_op=ops[v[4]],
                src_alpha_factor=factors[v[5]],
                dst_alpha_factor=factors[v[6]],
                alpha_op=ops[v[7]],
            )
            for v in targets
        ],
    )
    # Match native pipeline_contract v1's typed, length-delimited serialization.
    digest = hashlib.sha256()

    def number(value):
        digest.update(int(value).to_bytes(8, "little"))

    def text(value):
        encoded = value.encode()
        number(len(encoded))
        digest.update(encoded)

    text("forge-vulkan-graphics-v1")
    number(len(shaders))
    for shader in shaders:
        number({"vertex": 2, "fragment": 1, "task": 6, "mesh": 7}[shader.stage])
        text(shader.code_sha256)
    for value in (topology, polygon_mode, front_cull, back_cull, depth_test, depth_write, depth[0]):
        number(value)
    for value in depth[1:]:
        number(struct.unpack("<I", struct.pack("<f", value))[0])
    number(len(targets))
    for v in targets:
        for i in (0, 1, 4, 2, 3, 7, 5, 6):
            number(v[i])
    number(len(spec["vertices"]))
    for v in spec["vertices"]:
        for key in ("binding", "stride", "instance"):
            number(v[key])
    number(len(spec["attributes"]))
    for v in spec["attributes"]:
        for value in (v["location"], v["binding"], int(core.Format.__members__[v["format"]]), v["offset"]):
            number(value)
    spec["expected_pipeline"] = digest.hexdigest()
    return spec
