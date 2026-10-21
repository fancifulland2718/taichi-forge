"""Frozen SPD code and fixed image relationships for cold Graph export."""

from taichi_forge._lib import core
from taichi_forge.graph._native_definition import NativeResourceBinding, NativeResourceReference
from taichi_forge.lang.exception import TaichiRuntimeError


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
    definition = plan._definition
    images = context.source_images
    source = images.resource(context, recording.binding_names[0])
    output = images.resource(context, recording.binding_names[1], full_overwrite=True)
    context.resource_links(plan, dict(source=plan._source, output=plan._output))
    resource = context.public_resource(plan)
    if resource is None:
        resource = NativeResourceBinding(
            plan,
            definition.native,
            resources=tuple(
                (port, NativeResourceReference(symbol))
                for port, symbol in (
                    ("source", source),
                    ("output", output),
                    ("shader", definition.shader._export_aot_resource(context)),
                )
            ),
        )
    else:
        resource = NativeResourceReference(resource)
    definition.bind(resource, recording.binding_names, (source, output)).lower(context)
    context.reconstructed_binding_validators.add(id(recording))
    return ()
