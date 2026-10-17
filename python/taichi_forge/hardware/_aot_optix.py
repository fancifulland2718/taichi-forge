"""Reconstruct qualified OptiX recordings through the common AOT factories."""

from taichi_forge.lang.exception import TaichiRuntimeError


def _provider(context, provider):
    provider._validate_lifetime()
    return context.resource_once(
        provider,
        lambda: context.resource(provider, "optix", "context", parameters=dict(validation=provider._validation)),
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
    if shared and geometry._micromap_id is not None:
        raise TaichiRuntimeError("AOT OptiX recording export does not yet reconstruct OMM geometry")

    def build():
        # AS contents cannot be recovered from a retained input array: it may
        # have changed since the last build/refit. Require a fresh consumer seed.
        inputs = context.resource_inputs.require(geometry, ("vertices", "indices"))
        return context.resource(
            geometry,
            "optix",
            "triangle_gas" if shared else "triangle_scene",
            parameters=dict(
                vertex_count=geometry.vertex_count,
                triangle_count=geometry.triangle_count,
                allow_update=geometry.allow_update,
            ),
            bindings=dict(context=_provider(context, geometry.provider), **inputs),
        )

    return context.resource_once(geometry, build, policy="optix_consumer_seed")


def _scene(context, scene):
    from ._optix import OptixTriangleScene, OptixInstanceScene

    if type(scene) is OptixTriangleScene:
        return _geometry(context, scene)
    if type(scene) is not OptixInstanceScene:
        raise TaichiRuntimeError("AOT OptiX recording export requires a qualified scene")
    scene._validate_lifetime()
    if scene._has_micromaps or any(scene._sbt_offsets):
        raise TaichiRuntimeError("AOT OptiX recording export does not yet reconstruct OMM or nonzero SBT offsets")

    def build():
        inputs = context.resource_inputs.require(scene, ("transforms",))
        bindings = dict(context=_provider(context, scene.provider), **inputs)
        for ordinal, gas in enumerate(scene._topology):
            bindings[f"gas_{ordinal}"] = _geometry(context, gas, retained_by=scene)
        return context.resource(
            scene,
            "optix",
            "instance_scene",
            bindings=bindings,
            parameters=dict(
                instance_count=scene.instance_count,
                allow_update=scene.allow_update,
                custom_indices=[instance.custom_index for instance in scene._instances],
                visibility_masks=[instance.mask for instance in scene._instances],
                opaque=[instance.opaque for instance in scene._instances],
            ),
        )

    return context.resource_once(scene, build, policy="optix_consumer_seed")


def export_query(recording, context):
    from ._optix import OptixFaceRuleTable, OptixInstanceScene

    recording.validate_graph_lifetime()
    if recording._micromap or any(mask is not None for mask in recording.alpha_masks or ()):
        raise TaichiRuntimeError("AOT OptiX recording export does not yet reconstruct alpha/OMM inputs")
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
    context.native(
        "optix",
        "query",
        version=2,
        bindings=bindings,
        parameters=dict(
            ray_count=recording.ray_count,
            occlusion=recording._occlusion,
            any_hit=recording.any_hit,
            face_rules=rules,
            face_table_count=table_count,
        ),
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
