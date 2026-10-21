"""Reconstruct qualified OptiX recordings through the common AOT factories."""

from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.graph._ir import GraphAccess
from taichi_forge.graph._native_definition import (
    NativeArrayBinding,
    NativeDefinition,
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
        return context.resource(
            scene,
            "optix",
            "instance_scene",
            version=2 if any(scene._sbt_offsets) else 1,
            bindings=bindings,
            parameters=dict(
                instance_count=scene.instance_count,
                allow_update=scene.allow_update,
                custom_indices=[instance.custom_index for instance in scene._instances],
                visibility_masks=[instance.mask for instance in scene._instances],
                opaque=[instance.opaque for instance in scene._instances],
                **(dict(sbt_offsets=list(scene._sbt_offsets)) if any(scene._sbt_offsets) else {}),
            ),
        )

    return context.resource_once(scene, build, policy="optix_consumer_seed")


def export_query(recording, context):
    from taichi_forge._lib import core

    from ._optix import OptixFaceRuleTable, OptixInstanceScene

    recording.validate_graph_lifetime()
    if recording.hit_indices is None and not recording._occlusion:
        raise TaichiRuntimeError("AOT OptiX recording export requires typed or compact occlusion queries")
    bindings = dict(scene=_scene(context, recording.scene), rays=recording.rays, hits=recording.hits)
    if recording.hit_indices is not None:
        bindings["hit_indices"] = recording.hit_indices
    rules = []
    table_count = 0
    count = recording.scene.instance_count if type(recording.scene) is OptixInstanceScene else 1
    for rule in recording.face_rules or ("two_sided",) * count:
        if isinstance(rule, OptixFaceRuleTable):
            bindings[f"faces_{table_count}"] = rule.binding
            table_count += 1
            rules.append("table")
        else:
            rules.append(rule)
    masks = []
    mask_count = 0
    for mask in recording.alpha_masks or ():
        if mask is None:
            masks.append(None)
            continue
        texture = context.source_images.require(mask.texture)
        fmt, shape, mip_levels, _ = texture._creation_metadata
        if len(shape) != 2 or mip_levels != 1 or fmt not in (core.Format.r32f, core.Format.rgba32f):
            raise TaichiRuntimeError("AOT OptiX alpha requires a single-level 2D r32f/rgba32f texture")
        if fmt == core.Format.r32f and mask.channel != 0:
            raise ValueError("OptiX r32f alpha textures require channel=0")
        bindings[f"uvs_{mask_count}"] = mask.uvs
        bindings[f"image_{mask_count}"] = context.source_images.resource(context, mask.texture)
        masks.append(dict(cutoff=mask.cutoff, channel=mask.channel))
        mask_count += 1
    parameters = dict(
        ray_count=recording.ray_count,
        occlusion=recording._occlusion,
        any_hit=recording.any_hit,
        face_rules=rules,
        face_table_count=table_count,
    )
    if mask_count:
        parameters.update(alpha_masks=masks, mask_count=mask_count)
    context.native(
        "optix",
        "query",
        version=3 if mask_count else 2,
        bindings=bindings,
        parameters=parameters,
    )
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
