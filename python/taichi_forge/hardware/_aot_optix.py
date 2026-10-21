"""Reconstruct qualified OptiX recordings through the common AOT factories."""

from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.graph._ir import GraphAccess
from taichi_forge.graph._native_definition import (
    NativeArrayBinding,
    NativeDefinition,
    NativeOperationBinding,
    NativeResourceBinding,
    NativeResourceReference,
)

def export_micromap(asset, context):
    from ._optix_micromap import OptixOpacityMicromap

    if type(asset) is not OptixOpacityMicromap:
        raise TaichiRuntimeError("AOT OptiX micromap requires a qualified baked asset")
    if (public := context.public_resource(asset)) is not None:
        return public

    def build():
        bindings = dict(
            data=context.host_blob_resource(asset.data), descriptors=context.host_blob_resource(asset.descriptors)
        )
        if asset.triangle_indices is not None:
            bindings["indices"] = context.host_blob_resource(asset.triangle_indices)
        return NativeResourceBinding(
            asset,
            NativeDefinition("optix", "opacity_micromap", dict(triangle_count=asset.triangle_count)),
            resources=tuple((port, NativeResourceReference(value)) for port, value in bindings.items()),
        ).lower(context)

    return context.resource_once(asset, build, policy="optix_baked_opacity_micromap")


def _provider(context, provider):
    provider._validate_lifetime()
    if (public := context.public_resource(provider)) is not None:
        return public
    return context.resource_once(
        provider,
        lambda: NativeResourceBinding(provider, provider._definition).lower(context),
        policy="optix_context",
    )


def _geometry(context, geometry, *, retained_by=None):
    from ._optix import OptixTriangleGAS, OptixTriangleScene

    if type(geometry) not in (OptixTriangleScene, OptixTriangleGAS):
        raise TaichiRuntimeError("AOT OptiX geometry requires a qualified triangle resource")
    if retained_by is None:
        geometry._validate_lifetime()
    else:
        # An IAS retains the native GAS even after its original public handle
        # closes. Fresh reconstruction only needs immutable metadata and seeds.
        retained_by._validate_lifetime()
        # The caller enumerates this IAS's frozen topology. Do not search that
        # topology again for every instance (quadratic for large scenes).
        if geometry.provider is not retained_by.provider:
            raise TaichiRuntimeError("AOT OptiX GAS and instance scene require the same provider")
    shared = type(geometry) is OptixTriangleGAS
    context.resource_links(geometry, dict(context=geometry.provider))
    if (public := context.public_resource(geometry)) is not None:
        return public

    def build():
        # AS contents cannot be recovered from a retained input array: it may
        # have changed since the last build/refit. Require a fresh consumer seed.
        has_micromap = shared and geometry._micromap_id is not None
        ports = ("vertices", "indices", "opacity_micromap") if has_micromap else ("vertices", "indices")
        inputs = context.resource_inputs.require(geometry, ports, asset_ports=("opacity_micromap",))
        if has_micromap and not isinstance(inputs["opacity_micromap"], str):
            from ._optix_micromap import OptixOpacityMicromap

            asset = inputs["opacity_micromap"]
            if type(asset) is not OptixOpacityMicromap or asset.fingerprint != geometry._micromap_id:
                raise TaichiRuntimeError("AOT OptiX baked micromap must match the source GAS fingerprint")
            inputs["opacity_micromap"] = export_micromap(asset, context)
        dependencies = {k: v for k, v in inputs.items() if k not in ("vertices", "indices")}
        dependencies["context"] = _provider(context, geometry.provider)
        return NativeResourceBinding(
            geometry,
            geometry._definition,
            arrays=tuple(NativeArrayBinding(port, inputs[port], GraphAccess.READ) for port in ("vertices", "indices")),
            resources=tuple((port, NativeResourceReference(value)) for port, value in dependencies.items()),
        ).lower(context)

    return context.resource_once(geometry, build, policy="optix_consumer_seed")


def _scene(context, scene):
    from ._optix import OptixTriangleScene, OptixInstanceScene

    if type(scene) is OptixTriangleScene:
        return _geometry(context, scene)
    if type(scene) is not OptixInstanceScene:
        raise TaichiRuntimeError("AOT OptiX recording export requires a qualified scene")
    scene._validate_lifetime()
    context.resource_links(
        scene,
        dict(
            context=scene.provider,
            **{f"gas_{i}": gas for i, gas in enumerate(scene._topology)},
        ),
    )
    for gas in dict.fromkeys(scene._topology):
        context.resource_links(gas, dict(context=scene.provider))
    if (public := context.public_resource(scene)) is not None:
        return public

    def build():
        inputs = context.resource_inputs.require(scene, ("transforms",))
        bindings = dict(context=_provider(context, scene.provider), **inputs)
        for ordinal, gas in enumerate(scene._topology):
            bindings[f"gas_{ordinal}"] = _geometry(context, gas, retained_by=scene)
        return NativeResourceBinding(
            scene,
            scene._definition,
            arrays=(NativeArrayBinding("transforms", inputs["transforms"], GraphAccess.READ),),
            resources=tuple(
                (port, NativeResourceReference(value)) for port, value in bindings.items() if port != "transforms"
            ),
        ).lower(context)

    return context.resource_once(scene, build, policy="optix_consumer_seed")


def export_query(recording, context):
    from taichi_forge._lib import core

    recording.validate_graph_lifetime()
    if recording.hit_indices is None and not recording._occlusion:
        raise TaichiRuntimeError("AOT OptiX recording export requires typed or compact occlusion queries")
    definition = recording._query_definition
    resources = [("scene", NativeResourceReference(_scene(context, recording.scene)))]
    for i, mask in enumerate(definition.masks):
        texture = context.source_images.require(mask.texture)
        fmt, shape, mip_levels, _ = texture._creation_metadata
        if len(shape) != 2 or mip_levels != 1 or fmt not in (core.Format.r32f, core.Format.rgba32f):
            raise TaichiRuntimeError("AOT OptiX alpha requires a single-level 2D r32f/rgba32f texture")
        if fmt == core.Format.r32f and mask.channel != 0:
            raise ValueError("OptiX r32f alpha textures require channel=0")
        resources.append((f"image_{i}", NativeResourceReference(context.source_images.resource(context, mask.texture))))
    NativeOperationBinding(
        definition.native, definition.arrays(recording.rays, recording.hits, recording.hit_indices), resources=resources
    ).lower(context)
    return ()


def export_refit(recording, context):
    recording.validate_graph_lifetime()
    context.native("optix", "refit", bindings=dict(scene=_scene(context, recording.scene), vertices=recording.vertices))
    return ()


def export_gas_refit(recording, context):
    recording.validate_graph_lifetime()
    context.native(
        "optix", "refit", bindings=dict(scene=_geometry(context, recording.gas), vertices=recording.vertices)
    )
    return ()


def export_instance_refit(recording, context):
    recording.validate_graph_lifetime()
    bindings = dict(scene=_scene(context, recording.scene))
    if recording.transforms is not None:
        bindings["transforms"] = recording.transforms
    context.native("optix", "instance_refit", bindings=bindings)
    return ()
