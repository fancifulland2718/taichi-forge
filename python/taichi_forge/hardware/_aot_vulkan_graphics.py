"""Cold raster reconstruction through the shared native resource registry."""

import hashlib
import json
import math
import struct

from taichi_forge._lib import core
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.types.texture_type import FORMAT2TY_CH, is_float_sampled_texture_format

from ._shader_artifact import SpirvShader, _digest


def freeze_pipeline(
    pipeline, *, topology, polygon_mode, front_cull, back_cull, depth_test, depth_write, depth, targets, blending
):
    """Snapshot the exact creation state; never inspect mutable attributes at export."""
    from ._graphics import _BLEND_FACTORS, _BLEND_OPS, _DEPTH_COMPARE_OPS, _POLYGON_MODES, _TOPOLOGIES

    if blending and not targets:
        targets = ((True, 15, 6, 7, 0, 6, 7, 0),)
    factors, ops = tuple(_BLEND_FACTORS), tuple(_BLEND_OPS)
    spec = dict(
        shader_count=len(pipeline._shader_artifacts),
        shaders=[dict(stage=value.stage) for value in pipeline._shader_artifacts],
        vertices=[dict(binding=v.binding, stride=v.stride, instance=v.instance) for v in pipeline.vertex_bindings],
        attributes=[
            dict(location=v.location, binding=v.binding, format=v.format.name, offset=v.offset)
            for v in pipeline.vertex_attributes
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
    number(len(pipeline._shader_artifacts))
    for shader in pipeline._shader_artifacts:
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
    return json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _pipeline(context, pipeline):
    cache = context._native_export_facts.setdefault("vulkan_graphics_pipelines", {})
    if id(pipeline) in cache:
        return cache[id(pipeline)]
    pipeline._validate_lifetime()
    spec = json.loads(pipeline._aot_pipeline_spec)
    if (public := context.public_resource(pipeline)) is not None:
        cache[id(pipeline)] = public, spec["expected_pipeline"]
        return cache[id(pipeline)]

    def build():
        facts = pipeline.shader_artifacts()
        ports = tuple(f"shader_{i}" for i in range(len(facts)))
        inputs = context.resource_inputs.require(pipeline, ports, asset_ports=ports)
        for port, fact in zip(ports, facts, strict=True):
            asset = inputs[port]
            if not isinstance(asset, str):
                if type(asset) is not SpirvShader or _digest(asset.to_dict()) != _digest(fact):
                    raise TaichiRuntimeError("AOT graphics shader asset differs from the source pipeline contract")
                inputs[port] = asset._export_aot_resource(context)
        version = 2 if any(shader["stage"] == "mesh" for shader in spec["shaders"]) else 1
        return context.resource(
            pipeline, "vulkan_graphics", "pipeline", version=version, parameters=spec, bindings=inputs
        )

    cache[id(pipeline)] = (
        context.resource_once(pipeline, build, policy="vulkan_graphics_code"),
        spec["expected_pipeline"],
    )
    return cache[id(pipeline)]


def _parts(recording):
    from ._graphics import ColorAttachment, GraphicsPassDraw, VulkanGraphicsDrawRecording

    if type(recording) is VulkanGraphicsDrawRecording:
        return (
            (
                GraphicsPassDraw(
                    recording.pipeline,
                    recording.draw,
                    dict(recording.vertex_buffers),
                    index_buffer=recording.index_buffer,
                ),
            ),
            (ColorAttachment(recording.color, clear_value=recording.clear_color),),
            "clear",
        )
    return recording.draws, recording.colors, recording.depth_load_op


def preflight(recording, context):
    from ._graphics import Draw, IndirectDraw, MeshDraw, VulkanGraphicsPipeline, VulkanMeshPipeline

    cache = context._native_export_facts.setdefault("vulkan_graphics_recordings", {})
    if id(recording) in cache:
        return cache[id(recording)]
    if context.arch != core.Arch.vulkan:
        raise TaichiRuntimeError("AOT graphics recording requires Vulkan")
    parts = _parts(recording)
    draws, colors, _ = parts
    for item in draws:
        if (type(item.pipeline), type(item.draw)) not in (
            (VulkanGraphicsPipeline, Draw),
            (VulkanGraphicsPipeline, IndirectDraw),
            (VulkanMeshPipeline, MeshDraw),
        ):
            raise TaichiRuntimeError("AOT graphics recording requires classic raster or direct mesh draws")
    if any(type(item.draw) is IndirectDraw for item in draws) and any(type(item.draw) is MeshDraw for item in draws):
        raise TaichiRuntimeError("AOT pass_indirect cannot mix mesh and classic indirect draws")
    for name in (*(value.name for value in colors), recording.depth):
        if name is not None:
            context.source_images.require_attachment(name)
    cache[id(recording)] = parts
    return parts


def _clear_bits(fmt, values):
    if is_float_sampled_texture_format(fmt):
        if any(not math.isfinite(value) for value in values):
            raise TaichiRuntimeError("AOT color clear requires finite float32 values")
        return list(struct.unpack("<4I", struct.pack("<4f", *values)))
    dtype, _ = FORMAT2TY_CH[fmt]
    signed = str(dtype).startswith("i")
    bits = core.data_type_size(dtype) * 8
    low, high = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)
    if any(not isinstance(value, int) or not low <= value <= high for value in values):
        raise TaichiRuntimeError("AOT integer clear is outside the texture component range")
    return [value & 0xFFFFFFFF for value in values]


def export(recording, context):
    from ._graphics import IndirectDraw, MeshDraw, ShaderBufferArrayBinding

    draws, colors, depth_load = preflight(recording, context)
    bindings, groups, descriptions = {}, {name: {} for name in ("pipeline", "array", "image")}, []
    has_scenes = any(item.shader_acceleration_structures for item in draws)
    has_mesh = any(type(item.draw) is MeshDraw for item in draws)
    has_indirect = any(type(item.draw) is IndirectDraw for item in draws)
    has_arrays = any(
        isinstance(value, ShaderBufferArrayBinding) for item in draws for value in item.pipeline.shader_buffer_bindings
    )
    if has_scenes or has_arrays or has_mesh or has_indirect:
        groups["scene"] = {}

    def index(group, resource):
        entries = groups[group]
        if resource not in entries:
            entries[resource] = len(entries)
            bindings[f"{group}_{entries[resource]}"] = resource
        return entries[resource]

    def buffer_description(key, name, declaration):
        if isinstance(declaration, ShaderBufferArrayBinding):
            return dict(
                set=key[0],
                binding=key[1],
                storage=True,
                arrays=[index("array", context.binding(element)) for element in name],
            )
        return dict(
            set=key[0],
            binding=key[1],
            storage=declaration.kind == "storage",
            array=index("array", context.binding(name)),
        )

    for i, color in enumerate(colors):
        bindings[f"color_{i}"] = context.source_images.resource(
            context, color.name, first_clear=color.load_op == "clear"
        )
    if recording.depth is not None:
        bindings["depth"] = context.source_images.resource(context, recording.depth, first_clear=depth_load == "clear")
    for item in draws:
        pipeline, contract = _pipeline(context, item.pipeline)
        draw = item.draw
        result = dict(
            pipeline=index("pipeline", pipeline),
            expected_pipeline=contract,
            vertices=[
                dict(binding=binding, array=index("array", context.binding(name)))
                for binding, name in item._ordered_vertex_buffers
            ],
            buffers=[
                buffer_description(key, name, declaration) for key, name, declaration in item._ordered_shader_buffers
            ],
            images=[
                dict(set=key[0], binding=key[1], image=index("image", context.source_images.resource(context, name)))
                for key, name in item._ordered_shader_images
            ],
        )
        if type(draw) is MeshDraw:
            result["mesh"] = [draw.group_count_x, draw.group_count_y, draw.group_count_z]
        elif type(draw) is IndirectDraw:
            result["indirect"] = dict(
                commands=index("array", context.binding(item.indirect_buffer)),
                command_offset=draw.command_offset,
                count_offset=draw.count_offset or 0,
                max_draw_count=draw.max_draw_count,
                stride=draw.command_stride or (20 if item.index_buffer is not None else 16),
                vertex_record_limit=draw.vertex_record_limit,
                instance_record_limit=draw.instance_record_limit,
                index_element_limit=draw.index_element_limit,
                first_instance_may_be_nonzero=draw.first_instance_may_be_nonzero,
            )
            if item.count_buffer is not None:
                result["indirect"]["count"] = index("array", context.binding(item.count_buffer))
        else:
            result.update(
                element_count=draw.element_count,
                instance_count=draw.instance_count,
                first_vertex=draw.first_vertex,
                first_index=draw.first_index,
                first_instance=draw.first_instance,
                vertex_offset=draw.vertex_offset,
            )
        if item.index_buffer is not None:
            result["index"] = index("array", context.binding(item.index_buffer))
            if type(draw) is not IndirectDraw:
                result.update(index_min=draw.index_bounds[0], index_max=draw.index_bounds[1])
        if item.shader_acceleration_structures:
            result["scenes"] = [
                dict(
                    set=key[0],
                    binding=key[1],
                    scene=index("scene", context.source_resources.acceleration_structures.resource(context, name)),
                )
                for key, name in item._ordered_shader_acceleration_structures
            ]
        descriptions.append(result)
    parameters = dict(
        **{name + "_count": len(values) for name, values in groups.items()},
        color_count=len(colors),
        colors=[
            dict(
                clear=value.load_op == "clear",
                clear_bits=_clear_bits(
                    context.source_images.require(value.name)._creation_metadata[0], value.clear_value
                ),
            )
            for value in colors
        ],
        depth_clear=depth_load == "clear",
        clear_depth=recording.clear_depth,
        viewport=list(recording.viewport),
        draws=descriptions,
    )
    context.native(
        "vulkan_graphics",
        "pass_indirect" if has_indirect else "pass_direct",
        version=1 if has_indirect else (4 if has_mesh else (3 if has_arrays else (2 if has_scenes else 1))),
        parameters=parameters,
        bindings=bindings,
    )
    return ()
