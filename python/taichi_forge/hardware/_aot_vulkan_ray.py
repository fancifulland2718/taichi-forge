"""Lower qualified Vulkan ray recordings through the shared AOT factories."""

from taichi_forge.lang.exception import TaichiRuntimeError


def _scene(context, scene):
    from ._ray import TriangleScene

    if type(scene) is not TriangleScene:
        raise TaichiRuntimeError("AOT Vulkan ray export currently requires a qualified TriangleScene")
    scene._validate_lifetime()
    if (public := context.public_resource(scene)) is not None:
        return public

    def build():
        # Source arrays may have changed since AS construction/refit. Require
        # consumer seeds instead of pretending to snapshot the live AS state.
        inputs = context.resource_inputs.require(scene, ("vertices", "indices"))
        return context.resource(
            scene,
            "vulkan_ray",
            "triangle_scene",
            parameters=dict(vertex_count=scene.vertex_count, triangle_count=scene.triangle_count),
            bindings=inputs,
        )

    return context.resource_once(scene, build, policy="vulkan_ray_consumer_seed")


def export_query(recording, context):
    recording.validate_graph_lifetime()
    bindings = dict(scene=_scene(context, recording.scene), rays=recording.rays, hits=recording.hits)
    if recording.hit_indices is not None:
        bindings["hit_indices"] = recording.hit_indices
    context.native("vulkan_ray", "query", parameters=dict(ray_count=recording.ray_count), bindings=bindings)
    return ()


def export_refit(recording, context):
    recording.validate_graph_lifetime()
    context.native(
        "vulkan_ray", "refit", bindings=dict(scene=_scene(context, recording.scene), vertices=recording.vertices)
    )
    return ()
