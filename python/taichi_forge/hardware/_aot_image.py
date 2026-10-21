"""Cold region transfer reconstruction; shared native preparation owns safety."""

from taichi_forge._lib import core
from taichi_forge.lang.exception import TaichiRuntimeError


def _full(owner, region):
    _, shape, levels, _ = owner._creation_metadata
    return (
        levels == 1
        and region["mip_level"] == 0
        and not any(region["offset"])
        and region["extent"] == [*shape, *([1] * (3 - len(shape)))]
        and region["base_layer"] == 0
        and region["layer_count"] == 1
    )


def preflight(recording, context):
    from ._image import (
        VulkanBufferToImageRecording,
        VulkanImageBlitRecording,
        VulkanImageCopyRecording,
        VulkanImageToBufferRecording,
    )

    if context.arch != core.Arch.vulkan:
        raise TaichiRuntimeError("AOT image transfers require Vulkan")
    cache = context._native_export_facts.setdefault("image_transfers", {})
    if id(recording) in cache:
        return cache[id(recording)]
    recording.validate_graph_lifetime()
    images = context.source_images
    kind = type(recording)
    if kind in (VulkanImageCopyRecording, VulkanImageBlitRecording):
        source, destination = (images.require_transfer(name) for name in (recording.source, recording.destination))
        definition = recording._definition.resolve(source, destination)
        parameters = definition.parameter_dict()
        result = (
            definition,
            recording.source,
            recording.destination,
            _full(destination, parameters["destination_region"]),
        )
    elif kind in (VulkanBufferToImageRecording, VulkanImageToBufferRecording):
        upload = kind is VulkanBufferToImageRecording
        name = recording.destination if upload else recording.source
        owner = images.require_transfer(name)
        definition = recording._definition.resolve(owner)
        result = definition, name, None, upload and _full(owner, definition.parameter_dict()["region"])
    else:
        raise TaichiRuntimeError("AOT image transfer requires a concrete qualified recording")
    cache[id(recording)] = result
    return result


def export(recording, context):
    definition, source, destination, overwrite = preflight(recording, context)
    images = context.source_images
    if destination is not None:
        bindings = dict(
            source=images.resource(context, source),
            destination=images.resource(context, destination, full_overwrite=overwrite),
        )
        recording._native_binding(definition, bindings["source"], bindings["destination"]).lower(context)
    else:
        image = images.resource(context, source, full_overwrite=overwrite)
        # Preserve array admission before the default adapter resolves symbols.
        context.binding(recording.source if definition.operation == "buffer_to_image" else recording.destination)
        # Native preparation rechecks the actual finite dense view. Merely
        # claiming this validator without reconstructing it is not sufficient.
        context.reconstructed_binding_validators.add(id(recording))
        recording._native_binding(definition, image).lower(context)
    return ()
