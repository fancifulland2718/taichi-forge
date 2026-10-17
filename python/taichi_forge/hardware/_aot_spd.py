"""Frozen SPD code and fixed image relationships for cold Graph export."""

import hashlib
import json
import struct

from taichi_forge._lib import core
from taichi_forge.lang.exception import TaichiRuntimeError


def freeze(plan, words):
    code = struct.pack(f"<{len(words)}I", *words)
    digest = hashlib.sha256()

    def text(value):
        encoded = value.encode()
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)

    text("forge.spd.plan.v1")
    text(plan._reduction)
    for image in (plan._source, plan._output):
        fmt, shape, levels, _ = image._creation_metadata
        text(fmt.name)
        digest.update(struct.pack("<4Q", *shape, 1, levels))
    text(hashlib.sha256(code).hexdigest())
    return code, json.dumps(dict(reduction=plan._reduction, expected_plan=digest.hexdigest()), sort_keys=True)


def preflight(recording, context):
    if context.arch != core.Arch.vulkan:
        raise TaichiRuntimeError("AOT SPD recording requires Vulkan")
    recording.plan.validate_graph_lifetime()
    images = context.source_images
    for name, owner in zip(recording.binding_names, (recording.plan._source, recording.plan._output), strict=True):
        # Same fixed owner qualification as JIT; do not infer from shape alone.
        images.require_storage(name, owner)


def export(recording, context):
    preflight(recording, context)
    plan = recording.plan
    spec = json.loads(plan._aot_plan_spec)
    images = context.source_images
    source = images.resource(context, recording.binding_names[0])
    output = images.resource(context, recording.binding_names[1], full_overwrite=True)
    context.resource_links(plan, dict(source=plan._source, output=plan._output))
    resource = context.public_resource(plan)
    if resource is None:
        resource = context.resource_once(
            plan,
            lambda: context.resource(
                plan,
                "spd",
                "plan",
                parameters=spec,
                bindings=dict(source=source, output=output, shader=context.host_blob_resource(plan._aot_shader)),
            ),
            policy="spd_frozen_code_and_images",
        )
    context.native(
        "spd",
        "run",
        parameters=dict(expected_plan=spec["expected_plan"]),
        bindings=dict(plan=resource, source=source, output=output),
    )
    context.reconstructed_binding_validators.add(id(recording))
    return ()
