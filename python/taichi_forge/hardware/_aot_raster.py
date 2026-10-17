"""High-level raster lowering into shared kernels and native graphics passes."""

import math
from pathlib import Path
import struct

from taichi_forge._lib import core
from taichi_forge.lang.exception import TaichiRuntimeError

from ._aot_vulkan_graphics import _clear_bits, pipeline_description
from ._graphics import VertexAttribute, VertexBinding
from ._shader_artifact import SpirvShader


def _count(value):
    # The existing scene bridge accepts float32 counts and stores signed ints.
    # Preserve that conversion; reject ranges that cannot represent a draw.
    value = struct.unpack("<f", struct.pack("<f", value))[0]
    if not math.isfinite(value) or not 0 <= value <= 0x7FFFFFFF:
        raise TaichiRuntimeError("AOT raster count/offset exceeds the native nonnegative int32 range")
    return int(value)


def _draw_range(draw):
    args = draw._arguments
    count = draw._buffers["vertices"].shape[0]
    result = dict(first_vertex=0, first_index=0, first_instance=0, vertex_offset=0, instance_count=1)
    if draw.kind == "particles":
        result.update(element_count=6, instance_count=_count(args[-2]), first_instance=_count(args[-1]) % count)
        if result["instance_count"] + result["first_instance"] > count:
            raise TaichiRuntimeError("AOT particle range exceeds its center buffer")
        return result
    ni, first_index, nv, first_vertex = map(_count, args[-5:-1])
    first_vertex %= count
    if (indices := draw._buffers.get("indices")) is not None:
        if not indices.shape[0]:
            raise TaichiRuntimeError("AOT indexed raster requires a nonempty index buffer")
        first_index %= indices.shape[0]
        if ni + first_index > indices.shape[0]:
            raise TaichiRuntimeError("AOT raster index range exceeds its index buffer")
        # This is the source draw's valid-index contract, not a host scan or a
        # claim that mutable device indices were inspected. Rebinding retains it.
        result.update(
            element_count=ni,
            first_index=first_index,
            vertex_offset=first_vertex,
            index_min=0,
            index_max=count - first_vertex - 1,
        )
    else:
        if nv + first_vertex > count:
            raise TaichiRuntimeError("AOT raster vertex range exceeds its vertex buffer")
        result.update(element_count=nv, first_vertex=first_vertex)
    if draw.kind == "mesh_instance" and "transforms" in draw._buffers:
        instances, first = _count(args[6]), _count(args[7])
        available = draw._buffers["transforms"].shape[0] - first
        if available < 0:
            raise TaichiRuntimeError("AOT raster instance offset exceeds its transform buffer")
        result.update(instance_count=min(instances, available), first_instance=first)
    return result


def preflight(recording, context):
    cache = context._native_export_facts.setdefault("raster_recordings", {})
    if id(recording) in cache:
        return cache[id(recording)]
    if context.arch != core.Arch.vulkan:
        raise TaichiRuntimeError("AOT RasterPass requires Vulkan")
    recording.validate_graph_lifetime()
    if any(draw.kind not in ("mesh", "mesh_instance", "particles") for draw in recording._prepared_draws):
        raise TaichiRuntimeError("AOT RasterPass line expansion requires its compute reconstruction recipe")
    if not context.snapshot_state:
        raise TaichiRuntimeError("AOT raster camera/light state requires explicit snapshot_state=True at export")
    if any(draw.has_host_upload for draw in recording._prepared_draws):
        raise TaichiRuntimeError("AOT RasterPass requires device geometry")
    owner = recording._owner
    for image in (owner.color_texture, owner.depth_texture):
        context.source_images.qualify_attachment(image)
    width, height = owner.resolution
    scene = recording._camera.ptr._raster_scene_uniforms(width, height, recording._ambient, len(recording._point_lights))
    lights = b"".join(core._raster_point_light(position, color) for position, color in recording._point_lights)
    result = (width, height, scene, lights or bytes(32), tuple(_draw_range(draw) for draw in recording._prepared_draws))
    cache[id(recording)] = result
    return result


def _pipeline(context, kind, wireframe):
    cache = context._native_export_facts.setdefault("raster_pipelines", {})
    key = (kind == "particles", wireframe)
    if key not in cache:
        name = "Particles" if key[0] else "Mesh"
        directory = Path(__file__).parent.parent / "shaders"
        shaders = tuple(
            SpirvShader.from_file(directory / f"{name}_vk_{suffix}.spv", stage=stage)
            for suffix, stage in (("vert", "vertex"), ("frag", "fragment"))
        )
        spec = pipeline_description(
            shaders,
            (VertexBinding(0, 48, instance=key[0]),),
            tuple(
                VertexAttribute(i, 0, fmt, offset)
                for i, (fmt, offset) in enumerate(
                    (
                        (core.Format.rgb32f, 0),
                        (core.Format.rgb32f, 12),
                        (core.Format.rg32f, 24),
                        (core.Format.rgba32f, 32),
                    )
                )
            ),
            topology=0,
            polygon_mode=int(wireframe),
            front_cull=False,
            back_cull=False,
            depth_test=True,
            depth_write=True,
            depth=(6, 0.0, 0.0),
            targets=(),
            blending=True,
        )
        resource = context.resource(
            object(),
            "vulkan_graphics",
            "pipeline",
            parameters=spec,
            bindings={f"shader_{i}": shader._export_aot_resource(context) for i, shader in enumerate(shaders)},
        )
        cache[key] = resource, spec["expected_pipeline"]
    return cache[key]


def export(recording, context):
    width, height, scene, lights, ranges = preflight(recording, context)

    def immutable(data, usage):
        return context.data_resource(object(), data, dtype="u32", shape=[len(data) // 4], readonly=True, usage=(usage,))

    scene_bindings = None
    groups = {name: {} for name in ("pipeline", "array", "image", "scene")}
    bindings = {}

    def index(group, resource):
        values = groups[group]
        if resource not in values:
            values[resource] = len(values)
            bindings[f"{group}_{values[resource]}"] = resource
        return values[resource]

    draws = []
    for draw, selected in zip(recording._prepared_draws, ranges, strict=True):
        if not selected["element_count"] or not selected["instance_count"]:
            continue
        buffers = draw._export_aot_preparation(context)
        if scene_bindings is None:
            scene_bindings = immutable(scene, "uniform"), immutable(lights, "storage")
        args = draw._resolved_arguments
        particles = draw.kind == "particles"
        pipeline, contract = _pipeline(context, draw.kind, bool(args.get("show_wireframe", False)))
        uniform = (
            core._raster_particle_uniforms(
                tuple(args["color"]),
                args["per_vertex_color"] is not None,
                args["per_vertex_radius"] is not None,
                args["radius"],
            )
            if particles
            else core._raster_mesh_uniforms(
                tuple(args["color"]), args["per_vertex_color"] is not None, args["two_sided"], "transforms" in buffers
            )
        )
        shader_buffers = [immutable(uniform, "uniform"), *scene_bindings]
        if not particles:
            shader_buffers.append(buffers.get("transforms") or immutable(bytes(64), "storage"))
        description = dict(
            selected,
            pipeline=index("pipeline", pipeline),
            expected_pipeline=contract,
            vertices=[dict(binding=0, array=index("array", buffers["vertices"]))],
            buffers=[
                dict(set=0, binding=i, storage=i >= 2, array=index("array", value))
                for i, value in enumerate(shader_buffers)
            ],
            images=[],
        )
        if "indices" in buffers:
            description["index"] = index("array", buffers["indices"])
        draws.append(description)
    owner = recording._owner
    bindings["color_0"] = context.source_images.retained_attachment(context, owner.color_texture, first_clear=True)
    bindings["depth"] = context.source_images.retained_attachment(context, owner.depth_texture, first_clear=True)
    context.native(
        "vulkan_graphics",
        "pass_direct",
        version=5,
        bindings=bindings,
        parameters=dict(
            **{name + "_count": len(values) for name, values in groups.items()},
            color_count=1,
            colors=[dict(clear=True, clear_bits=_clear_bits(core.Format.rgba8, (*owner._background_color, 1.0)))],
            depth_clear=True,
            clear_depth=0.0,
            viewport=[0, 0, width, height],
            draws=draws,
            attachment_contract=dict(width=width, height=height, colors=["rgba8"], depth="depth32f"),
        ),
    )
