"""Cold program/packet export through the shared native resource catalog."""

import hashlib
from math import prod

from taichi_forge._lib import core
from taichi_forge.lang.exception import TaichiRuntimeError

from ._aot_optix import _provider, _scene
from ._shader_artifact import PtxModule, _digest


def program_spec(program):
    manifest = program.to_dict()
    facts = manifest["modules"]

    def entry(value):
        return None if value is None else dict(module=value[0], name=value[1])

    spec = dict(
        module_count=len(facts),
        module_sha256=[f["code_sha256"] for f in facts],
        groups=[
            dict(name=g["name"], kind=g["kind"], entry=entry(g["entry"]), any_hit=entry(g["any_hit"]))
            for g in manifest["groups"]
        ],
        parameter_name=manifest["parameter_name"],
        parameter_size=manifest["parameters"]["size"],
        **{
            key: manifest[key]
            for key in ("payload_count", "attribute_count", "max_trace_depth", "allow_opacity_micromaps")
        },
    )
    # native_optix_program.cpp program_contract v1. Driver/build identity does
    # not belong in a durable code/layout contract.
    digest = hashlib.sha256()

    def number(value):
        digest.update(int(value).to_bytes(8, "little"))

    def text(value):
        data = value.encode("utf-8")
        number(len(data))
        digest.update(data)

    text("forge-optix-program-v1")
    number(spec["module_count"])
    for value in spec["module_sha256"]:
        text(value)
    number(len(spec["groups"]))
    for group in spec["groups"]:
        text(group["name"])
        number(group["kind"])
        for key in ("entry", "any_hit"):
            item = group[key] or {}
            number(item.get("module", 0))
            text(item.get("name", ""))
    text(spec["parameter_name"])
    for key in ("parameter_size", "payload_count", "attribute_count", "max_trace_depth", "allow_opacity_micromaps"):
        number(spec[key])
    spec["expected_program"] = digest.hexdigest()
    return spec, facts


def _program(context, program):
    cache = context._native_export_facts.setdefault("optix_programs", {})
    if id(program) in cache:
        return cache[id(program)]
    program._validate_lifetime()
    spec, facts = program_spec(program)
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
        return context.resource(program, "optix", "program", parameters=spec, bindings=inputs)

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
