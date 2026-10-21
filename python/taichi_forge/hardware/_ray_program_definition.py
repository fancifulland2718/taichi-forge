"""Cold ray program contracts; code assets and live pipeline owners stay separate."""

from dataclasses import dataclass
import hashlib

from taichi_forge.graph._native_definition import NativeDefinition, _freeze, _parameters
from ._shader_artifact import _digest


@dataclass(frozen=True)
class RayProgramDefinition:
    native: NativeDefinition
    manifest: object
    assets: tuple

    def __post_init__(self):
        object.__setattr__(self, "manifest", _freeze(self.manifest))
        object.__setattr__(self, "assets", tuple(_freeze(item) for item in self.assets))

    def to_dict(self):
        return _parameters(self.manifest)

    def asset_facts(self):
        return tuple(_parameters(item) for item in self.assets)


def optix_program_definition(manifest):
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
    return RayProgramDefinition(NativeDefinition("optix", "program", spec), manifest, facts)


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


def vulkan_program_definition(manifest):
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
    return RayProgramDefinition(NativeDefinition("vulkan_ray_program", "pipeline", spec), manifest, facts)
