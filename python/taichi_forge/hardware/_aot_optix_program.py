"""Cold program/packet export through the shared native resource catalog."""

from math import prod

from taichi_forge._lib import core
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.graph._native_definition import NativeResourceBinding, NativeResourceReference

from ._aot_optix import _provider, _scene
from ._shader_artifact import PtxModule, _digest


def _program(context, program):
    cache = context._native_export_facts.setdefault("optix_programs", {})
    if id(program) in cache:
        return cache[id(program)]
    program._validate_lifetime()
    definition = program._definition
    spec, facts = definition.native.parameters, definition.asset_facts()
    context.resource_links(program, dict(context=program.provider))
    if (public := context.public_resource(program)) is not None:
        cache[id(program)] = public, spec["expected_program"]
        return cache[id(program)]

    def build():
        ports = tuple(f"module_{i}" for i in range(len(facts)))
        inputs = context.resource_inputs.require(program, ports, asset_ports=ports)
        for port, fact in zip(ports, facts, strict=True):
            asset = inputs[port]
            if not isinstance(asset, str):
                if type(asset) is not PtxModule or _digest(asset.to_dict()) != _digest(fact):
                    raise TaichiRuntimeError("AOT OptiX PTX asset differs from the source program contract")
                inputs[port] = asset._export_aot_resource(context)
        inputs["context"] = _provider(context, program.provider)
        return NativeResourceBinding(
            program,
            definition.native,
            resources=tuple((port, NativeResourceReference(value)) for port, value in inputs.items()),
        ).lower(context)

    cache[id(program)] = context.resource_once(program, build, policy="optix_program_code"), spec["expected_program"]
    return cache[id(program)]


def export_launch(recording, context):
    launch = recording.launch
    launch._require_initialized()
    source = launch.recording

    def build():
        program, contract = _program(context, source.program)
        bindings = dict(program=program)
        counts = dict(buffer=0, texture=0, scene=0)
        locations, layouts = {}, []
        for name, (kind, access) in source._resources.items():
            index = counts[kind]
            locations[name] = dict(kind=kind, index=index)
            counts[kind] += 1
            if kind == "scene":
                bindings[f"scene_{index}"] = _scene(context, source.scenes[name])
            elif kind == "texture":
                if context.source_images.require(name) is not launch._bindings[name]:
                    raise TaichiRuntimeError("AOT OptiX image differs from its fixed launch binding")
                bindings[f"image_{index}"] = context.source_images.resource(context, name)
            else:
                owner = launch._bindings[name]
                if context.source_arrays.require(owner) != name:
                    raise TaichiRuntimeError("AOT fixed OptiX buffer requires its Graph name in source_bindings")
                dtype, shape, element_shape = context.source_arrays.layouts[name]
                scalar_bytes = core.data_type_size(dtype)
                layouts.append(
                    dict(
                        dtype=str(dtype),
                        shape=list(shape),
                        element_shape=list(element_shape),
                        record_bytes=scalar_bytes * prod(element_shape),
                        alignment=scalar_bytes,
                        access=access,
                    )
                )
                bindings[f"array_{index}"] = name
        packet = bytearray()
        sections = {}

        def payload(data):
            # Repeated immutable SBT values share artifact bytes, while the
            # adapter still allocates each actual SBT record required by OptiX.
            value = sections.get(data.scalar_bytes)
            if value is None:
                value = dict(offset=len(packet), size=len(data.scalar_bytes))
                packet.extend(data.scalar_bytes)
                sections[data.scalar_bytes] = value
            return dict(
                value, references=[dict(locations[name], offset=field.offset) for field, name in data.references]
            )

        def record(value):
            return dict(group=source.program._groups[value.group][0], **payload(value._data))

        parameters = dict(
            expected_program=contract,
            dimensions=list(source.dimensions),
            parameters=payload(source._parameters),
            raygen=record(source.raygen),
            miss=[record(v) for v in source.miss],
            hit=[record(v) for v in source.hit],
            array_count=counts["buffer"],
            image_count=counts["texture"],
            scene_count=counts["scene"],
            array_layouts=layouts,
        )
        bindings["data"] = context.host_blob_resource(bytes(packet))
        return context.resource(launch, "optix", "prepared_launch", parameters=parameters, bindings=bindings)

    resource = context.resource_once(launch, build, policy="optix_fixed_launch")
    context.native("optix", "launch", bindings=dict(launch=resource), parameters={})
    return ()
