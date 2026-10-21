"""Lower qualified Vulkan ray recordings through the shared AOT factories."""

from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.graph._ir import GraphAccess
from taichi_forge.graph._native_definition import (
    NativeArrayBinding,
    NativeDefinition,
    NativeResourceBinding,
    NativeResourceReference,
)


def export_micromap(asset, context):
    from ._vulkan_micromap import VulkanOpacityMicromap

    if type(asset) is not VulkanOpacityMicromap:
        raise TaichiRuntimeError("AOT Vulkan micromap requires a qualified baked asset")
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
            NativeDefinition("vulkan_ray", "opacity_micromap", dict(triangle_count=asset.triangle_count)),
            resources=tuple((port, NativeResourceReference(value)) for port, value in bindings.items()),
        ).lower(context)

    return context.resource_once(asset, build, policy="vulkan_baked_opacity_micromap")


def _geometry(context, scene, *, retained_by=None):
    from ._ray import TriangleBLAS, TriangleScene

    if type(scene) not in (TriangleScene, TriangleBLAS):
        raise TaichiRuntimeError("AOT Vulkan geometry requires a qualified triangle resource")
    if retained_by is None:
        scene._validate_lifetime()
    else:
        # A live TLAS retains its BLAS cores after public BLAS handles close.
        # The caller walks that frozen topology; no repeated topology search.
        scene._validate_runtime_identity()
        if scene._runtime_prog is not retained_by._runtime_prog:
            raise TaichiRuntimeError("AOT Vulkan BLAS and TLAS require the same runtime")
    shared = type(scene) is TriangleBLAS
    if (public := context.public_resource(scene)) is not None:
        return public

    def build():
        # Source arrays may have changed since AS construction/refit. Require
        # consumer seeds instead of pretending to snapshot the live AS state.
        has_micromap = shared and scene._micromap_id is not None
        ports = ("vertices", "indices", "opacity_micromap") if has_micromap else ("vertices", "indices")
        inputs = context.resource_inputs.require(scene, ports, asset_ports=("opacity_micromap",))
        if has_micromap and not isinstance(inputs["opacity_micromap"], str):
            from ._vulkan_micromap import VulkanOpacityMicromap

            asset = inputs["opacity_micromap"]
            if type(asset) is not VulkanOpacityMicromap or asset.fingerprint != scene._micromap_id:
                raise TaichiRuntimeError("AOT Vulkan baked micromap must match the source BLAS fingerprint")
            inputs["opacity_micromap"] = export_micromap(asset, context)
        return NativeResourceBinding(
            scene,
            scene._definition,
            arrays=tuple(NativeArrayBinding(port, inputs[port], GraphAccess.READ) for port in ("vertices", "indices")),
            resources=tuple(
                (port, NativeResourceReference(value))
                for port, value in inputs.items()
                if port not in ("vertices", "indices")
            ),
        ).lower(context)

    return context.resource_once(scene, build, policy="vulkan_ray_consumer_seed")


def _scene(context, scene):
    from ._ray import InstanceTLAS, TriangleScene

    if type(scene) is TriangleScene:
        return _geometry(context, scene)
    if type(scene) is not InstanceTLAS:
        raise TaichiRuntimeError("AOT Vulkan query export requires a qualified scene")
    scene._validate_lifetime()
    context.resource_links(scene, {f"blas_{i}": blas for i, blas in enumerate(scene._topology)})
    if (public := context.public_resource(scene)) is not None:
        return public

    def build():
        inputs = context.resource_inputs.require(scene, ("transforms",))
        bindings = dict(inputs)
        for ordinal, blas in enumerate(scene._topology):
            bindings[f"blas_{ordinal}"] = _geometry(context, blas, retained_by=scene)
        # Host build/refit can replace custom IDs and masks, whereas device
        # refits change only transforms. Preserve the current host seed policy;
        # these mutable facts do not belong in the creation definition.
        metadata = dict(
            custom_indices=[instance.custom_index for instance in scene._instances],
            visibility_masks=[instance.mask for instance in scene._instances],
        )
        return NativeResourceBinding(
            scene,
            scene._definition,
            arrays=(NativeArrayBinding("transforms", inputs["transforms"], GraphAccess.READ),),
            resources=tuple(
                (port, NativeResourceReference(value)) for port, value in bindings.items() if port != "transforms"
            ),
        ).lower(context, requirements=metadata)

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


def export_blas_refit(recording, context):
    recording.validate_graph_lifetime()
    context.native(
        "vulkan_ray", "refit", bindings=dict(scene=_geometry(context, recording.blas), vertices=recording.vertices)
    )
    return ()


def export_blas_build(recording, context):
    recording.validate_graph_lifetime()
    context.native(
        "vulkan_ray",
        "blas_build",
        bindings=dict(scene=_geometry(context, recording.blas), vertices=recording.vertices, indices=recording.indices),
    )
    return ()


def export_instance_update(recording, context):
    recording.validate_graph_lifetime()
    context.native(
        "vulkan_ray",
        "instance_update",
        bindings=dict(scene=_scene(context, recording.tlas)),
        parameters=dict(
            transforms=[list(instance.transform) for instance in recording.instances],
            custom_indices=[instance.custom_index for instance in recording.instances],
            visibility_masks=[instance.mask for instance in recording.instances],
            update=recording.update,
        ),
    )
    return ()


def export_instance_transforms(recording, context):
    recording.validate_graph_lifetime()
    context.native(
        "vulkan_ray",
        "instance_refit",
        bindings=dict(scene=_scene(context, recording.tlas), transforms=recording.transforms),
    )
    return ()
