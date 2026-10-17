"""Cold export of initialized, fixed Vulkan RT launches; no driver snapshots."""

import hashlib

from taichi_forge.lang.exception import TaichiRuntimeError

from ._shader_artifact import SpirvShader, _digest
from ._aot_vulkan_ray import _scene


def _contract(spec, facts):
    """Match native pipeline_contract v1, without retaining JIT shader bytes."""
    digest = hashlib.sha256()

    def integer(value, width=4):
        digest.update(value.to_bytes(width, "little"))

    def text(value):
        encoded = value.encode("utf-8")
        integer(len(encoded), 8)
        digest.update(encoded)

    text("forge-vulkan-ray-program-v1")
    integer(len(facts), 8)
    for shader in facts:
        integer(("raygen", "miss", "closest_hit", "any_hit").index(shader["stage"]))
        text(shader["entry_point"])
        text(shader["code_sha256"])
    integer(len(spec["groups"]), 8)
    for group in spec["groups"]:
        integer(("raygen", "miss", "triangles").index(group["kind"]))
        for field in ("general", "closest_hit", "any_hit"):
            integer(group.get(field, 0xFFFFFFFF))
    integer(spec["max_trace_depth"])
    integer(int(spec["allow_opacity_micromaps"]))
    return digest.hexdigest()


def pipeline_spec(program):
    manifest = program.to_dict()
    facts, indices, groups = [], {}, []

    def shader_index(fact):
        if fact is None:
            return 0xFFFFFFFF
        key = _digest(fact)
        if key not in indices:
            indices[key] = len(facts)
            facts.append(fact)
        return indices[key]

    # Preserve the original sorted group order and artifact deduplication.
    for group in manifest["groups"]:
        item = dict(kind=group["kind"])
        if group["kind"] == "triangles":
            item.update(zip(("closest_hit", "any_hit"), map(shader_index, group["shaders"])))
        else:
            item["general"] = shader_index(group["shaders"])
        groups.append(item)
    spec = dict(
        shader_count=len(facts),
        shaders=[dict(stage=f["stage"], entry=f["entry_point"]) for f in facts],
        groups=groups,
        max_trace_depth=manifest["max_trace_depth"],
        allow_opacity_micromaps=manifest["opacity_micromaps"],
    )
    spec["expected_pipeline"] = _contract(spec, facts)
    return spec, facts


def _pipeline(context, program):
    program._validate_lifetime()
    spec, facts = pipeline_spec(program)
    if (public := context.public_resource(program)) is not None:
        return public, spec["expected_pipeline"]

    def build():
        ports = tuple(f"shader_{i}" for i in range(len(facts)))
        inputs = context.resource_inputs.require(program, ports, asset_ports=ports)
        for port, fact in zip(ports, facts):
            asset = inputs[port]
            if not isinstance(asset, str):
                if type(asset) is not SpirvShader or _digest(asset.to_dict()) != _digest(fact):
                    raise TaichiRuntimeError("AOT ray shader asset differs from the source pipeline contract")
                inputs[port] = asset._export_aot_resource(context)
        return context.resource(program, "vulkan_ray_program", "pipeline", parameters=spec, bindings=inputs)

    return context.resource_once(program, build, policy="vulkan_ray_program_code"), spec["expected_pipeline"]


def export_launch(recording, context):
    launch = recording.launch
    launch._require_initialized()
    source = launch.recording
    pipeline, contract = _pipeline(context, source.program)
    payload = bytearray()

    def section(data):
        item = dict(offset=len(payload), size=len(data))
        payload.extend(data)
        return item

    def record(value):
        return dict(group=source.program._groups[value.group][0], **section(value.data))

    parameters = dict(
        dimensions=list(source.dimensions),
        raygen=record(source.raygen),
        miss=[record(v) for v in source.miss],
        hit=[record(v) for v in source.hit],
        push_constants=section(source.push_constants),
        expected_pipeline=contract,
        buffers=[],
        images=[],
        scenes=[],
    )
    bindings = dict(pipeline=pipeline, payload=context.host_blob_resource(bytes(payload)))
    for name, binding in source.bindings.items():
        owner = launch._bindings[name]
        point = dict(set=binding.set, binding=binding.binding)
        if binding.kind == "scene":
            group, resource = "scene", _scene(context, owner)
        elif binding.kind.endswith("image"):
            if context.source_images.require(name) is not owner:
                raise TaichiRuntimeError("AOT ray image source differs from its fixed launch binding")
            group, resource = "image", context.source_images.resource(context, name)
            point.update(kind=binding.kind, access=binding.access, mip_level=binding.mip_level)
        else:
            resource = context.source_arrays.require(owner)
            if resource != name:
                raise TaichiRuntimeError("AOT fixed ray buffer requires its Graph name in source_bindings")
            group = "buffer"
            point.update(kind=binding.kind, access=binding.access)
        items = parameters[group + "s"]
        bindings[f"{group}_{len(items)}"] = resource
        items.append(point)
    for group in ("buffer", "image", "scene"):
        parameters[group + "_count"] = len(parameters[group + "s"])
    context.native("vulkan_ray_program", "launch", parameters=parameters, bindings=bindings)
    return ()
