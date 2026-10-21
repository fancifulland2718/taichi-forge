"""Cold export of initialized, fixed Vulkan RT launches; no driver snapshots."""


from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.graph._native_definition import NativeResourceBinding, NativeResourceReference

from ._shader_artifact import SpirvShader, _digest
from ._aot_vulkan_ray import _scene


def _pipeline(context, program):
    program._validate_lifetime()
    definition = program._definition
    spec, facts = definition.native.parameters, definition.asset_facts()
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
        return NativeResourceBinding(
            program,
            definition.native,
            resources=tuple((port, NativeResourceReference(value)) for port, value in inputs.items()),
        ).lower(context)

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
