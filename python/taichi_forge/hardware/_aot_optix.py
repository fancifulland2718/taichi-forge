"""Reconstruct qualified OptiX recordings through the common AOT factories."""

from taichi_forge.lang.exception import TaichiRuntimeError


def _scene(context, scene):
    from ._optix import OptixTriangleScene

    if type(scene) is not OptixTriangleScene:
        raise TaichiRuntimeError("AOT OptiX recording export currently requires a triangle scene")
    scene._validate_lifetime()

    def build():
        # AS contents cannot be recovered from a retained input array: it may
        # have changed since the last build/refit. Require a fresh consumer seed.
        inputs = context.resource_inputs.require(scene, ("vertices", "indices"))
        provider = scene.provider
        provider_name = context.resource_once(
            provider,
            lambda: context.resource(provider, "optix", "context", parameters=dict(validation=provider._validation)),
            policy="optix_context",
        )
        return context.resource(
            scene,
            "optix",
            "triangle_scene",
            parameters=dict(
                vertex_count=scene.vertex_count, triangle_count=scene.triangle_count, allow_update=scene.allow_update
            ),
            bindings=dict(context=provider_name, **inputs),
        )

    return context.resource_once(scene, build, policy="optix_consumer_seed")


def export_query(recording, context):
    from ._optix import OptixFaceRuleTable

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
    for rule in recording.face_rules or ("two_sided",):
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
