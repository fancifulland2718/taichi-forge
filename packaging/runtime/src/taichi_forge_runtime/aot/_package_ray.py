"""Ray acceleration structure Host factories backed by the Runtime native catalog.

No numerical frontend or producer scene is loaded. Geometry seeds are consumer
inputs; contexts, acceleration structures and calls retain the native leases.
"""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import os

from .._optix_adapters import bundled_optix_adapters
from ._package_native import NativeObject, NativeRecording


def _array(dtype, rank):
    return dict(kind="ndarray", element=dict(kind="scalar", dtype=dtype, element_shape=[]), ndim=rank)


@dataclass(frozen=True)
class OptixFaceRuleTable:
    binding: str

    def __post_init__(self):
        from ._runtime import _name
        _name(self.binding)


@dataclass(frozen=True)
class OptixRayInstance:
    gas: object
    transform: object = None
    mask: int = 255
    custom_index: int = 0
    opaque: bool = False
    sbt_record_offset: int = 0

    def __post_init__(self):
        for name, maximum in (("mask", 0xFF), ("custom_index", 0xFFFFFF), ("sbt_record_offset", 0xFFFFFFFF)):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= maximum:
                raise ValueError(f"OptiX instance {name} must be an integer in [0, {maximum}]")
        if type(self.opaque) is not bool:
            raise TypeError("OptiX instance opaque must be a bool")


class _VulkanBLAS(NativeObject):
    pass


@dataclass(frozen=True)
class RayInstance:
    blas: _VulkanBLAS
    transform: tuple = (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0)
    mask: int = 255
    custom_index: int = 0
    sbt_record_offset: int = 0

    def __post_init__(self):
        import numpy as np
        if not isinstance(self.blas, _VulkanBLAS):
            raise TypeError("Vulkan instance requires a triangle BLAS")
        for name, maximum in (("mask", 255), ("custom_index", 0xffffff), ("sbt_record_offset", 0xffffff)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"Vulkan instance {name} must be an integer")
            if not 0 <= value <= maximum:
                raise ValueError(f"Vulkan instance {name} is outside its native range")
        transform = np.asarray(self.transform, dtype=np.float32)
        if transform.shape not in ((12,), (3, 4)) or not np.isfinite(transform).all():
            raise ValueError("Vulkan instance requires a finite row-major 3 by 4 transform")
        object.__setattr__(self, "transform", tuple(transform.ravel()))


class _VulkanTLAS(NativeObject):
    def __init__(self, package, resource, count):
        super().__init__(package, resource)
        self.instance_count = count

    def record_refit_transforms(self, *, transforms="transforms", scene=None):
        self._check()
        if not isinstance(transforms, str) or not transforms or "\0" in transforms:
            raise ValueError("Vulkan transform binding requires a nonempty name")
        if scene is not None and (not isinstance(scene, str) or not scene or "\0" in scene or scene == transforms):
            raise ValueError("Vulkan scene binding requires a distinct nonempty name")
        recording = NativeRecording(self.package, "vulkan_ray", "instance_refit",
            bindings=dict(scene=self if scene is None else scene, transforms=transforms),
            schemas=dict(scene=dict(kind="acceleration_structure"), transforms=_array("f32", 3)),
            effects=dict(scene="read_write", transforms="read"))
        recording.extents = {"transforms": (self.instance_count, 3, 4)}
        if scene is not None:
            recording.fixed_owners = {"scene": self.resource}
        return recording

    def memory_report(self):
        return SimpleNamespace(to_dict=lambda: dict(complete=False, known_resident_requested_bytes=0,
            unknown_components=["Vulkan acceleration structures"]))


class RayAPI:
    RayInstance = RayInstance

    def __init__(self, package):
        self.package = package
        self.OptixRayInstance = OptixRayInstance
        self.OptixFaceRuleTable = OptixFaceRuleTable

    def is_available(self):
        runtime = self.package.runtime
        return runtime.arch == "vulkan" and bool(runtime.device_capabilities().get("spirv_has_ray_query"))

    def TriangleBLAS(self, vertices, indices, *, opaque=None):
        self.package._check()
        if opaque is not None and not isinstance(opaque, bool):
            raise TypeError("opaque must be a bool or None")
        runtime = self.package.runtime
        vertices, indices = (runtime._adapt_package_array(x) for x in (vertices, indices))
        for array, dtype in ((vertices, "f32"), (indices, "i32")):
            shape, element = tuple(array.shape), tuple(array.element_shape)
            triples = (len(shape) == 1 and element == (3,)) or (len(shape) == 2 and shape[1] == 3 and not element)
            if array.dtype != dtype or not triples or shape[0] <= 0:
                raise ValueError(f"Vulkan triangles require nonempty scalar or packed triples of {dtype}")
        resource = runtime.native_resource("vulkan_ray", "triangle_blas",
            parameters=dict(vertex_count=vertices.shape[0], triangle_count=indices.shape[0], opaque=opaque is not False),
            bindings=dict(vertices=vertices, indices=indices))
        return _VulkanBLAS(self.package, resource)

    def InstanceTLAS(self, instances):
        import numpy as np
        self.package._check()
        instances = tuple(instances)
        runtime = self.package.runtime
        if not instances or any(not isinstance(x, RayInstance) or x.blas.resource._runtime is not runtime
                                for x in instances):
            raise ValueError("Vulkan instances require geometry from this Runtime")
        transforms = np.asarray([x.transform for x in instances], dtype=np.float32).reshape(-1, 3, 4)
        poses = runtime.ndarray("f32", transforms.shape)
        try:
            poses.write(transforms.tobytes())
            resource = runtime.native_resource("vulkan_ray", "instance_tlas",
                parameters=dict(instance_count=len(instances), custom_indices=[x.custom_index for x in instances],
                                visibility_masks=[x.mask for x in instances],
                                sbt_record_offsets=[x.sbt_record_offset for x in instances]),
                bindings=dict(transforms=poses, **{f"blas_{i}": x.blas.resource for i, x in enumerate(instances)}))
        except BaseException:
            poses.close()
            raise
        return _VulkanTLAS(self.package, resource, len(instances))

    def load_optix_provider(self, library_path=None, *, validation=False,
                            provider_path=None, required_features=()):
        self.package._check()
        if self.package.runtime.arch != "cuda":
            raise ValueError("OptiX requires a CUDA Runtime")
        # These features are required by the existing native OptiX factory.
        # Additional feature families need their own qualified factory routes.
        supported = {"instances", "typed_hits", "alpha_mask", "occlusion",
                     "face_filter_typed", "face_filter_occlusion", "face_filter_per_primitive"}
        if isinstance(required_features, str) or not set(required_features) <= supported:
            raise ValueError("Required OptiX features are outside this Runtime factory contract")
        if type(validation) is not bool:
            raise TypeError("OptiX validation must be bool")
        runtime = self.package.runtime
        with runtime._lock:
            configured = runtime._provider_paths.get(b"optix")
            if provider_path is None and configured is not None:
                provider_path = os.fsdecode(configured[0])
                if library_path is None and configured[1] is not None:
                    library_path = os.fsdecode(configured[1])
            paths = [Path(provider_path)] if provider_path is not None else list(bundled_optix_adapters())
            if not paths:
                raise RuntimeError("No bundled OptiX provider adapter; supply provider_path")
            vendor = library_path or os.environ.get("TAICHI_FORGE_OPTIX_LIBRARY")
            try:
                for index, path in enumerate(paths):
                    runtime.configure_provider("optix", adapter_path=path, library_path=vendor)
                    # Let the final candidate report the original driver error.
                    if index + 1 < len(paths) and not runtime.probe_native_resource("optix_context")["available"]:
                        continue
                    resource = runtime.native_resource("optix", "context", parameters=dict(validation=validation))
                    return _Provider(self.package, resource, path)
            except BaseException:
                previous_path, previous_vendor = configured or (None, None)
                runtime.configure_provider("optix",
                    adapter_path=os.fsdecode(previous_path) if previous_path else None,
                    library_path=os.fsdecode(previous_vendor) if previous_vendor else None)
                raise


class _Provider(NativeObject):
    def __init__(self, package, resource, path):
        super().__init__(package, resource)
        self.identity = dict(provider="optix", adapter_path=str(Path(path).resolve()))

    def triangle_gas(self, vertices, indices, *, allow_update=False):
        self._check()
        runtime = self.package.runtime
        vertices, indices = (runtime._adapt_package_array(x) for x in (vertices, indices))
        resource = runtime.native_resource("optix", "triangle_gas",
            parameters=dict(vertex_count=vertices.shape[0], triangle_count=indices.shape[0], allow_update=allow_update),
            bindings=dict(context=self.resource, vertices=vertices, indices=indices))
        return _Geometry(self.package, resource, self)

    def instance_scene(self, instances, *, allow_update=True):
        import numpy as np
        self._check()
        instances = tuple(instances)
        if not instances or any(not isinstance(x, OptixRayInstance) or not isinstance(x.gas, _Geometry)
                                or x.gas.provider is not self for x in instances):
            raise ValueError("Instances require geometry from this OptiX context")
        transforms = np.asarray([np.eye(3, 4) if x.transform is None else np.asarray(x.transform).reshape(3, 4)
                                 for x in instances], dtype=np.float32)
        if transforms.shape != (len(instances), 3, 4) or not np.isfinite(transforms).all():
            raise ValueError("Instance transforms require finite 3 by 4 matrices")
        runtime = self.package.runtime
        poses = runtime.ndarray("f32", transforms.shape)
        try:
            poses.write(transforms.tobytes())
            offsets = [x.sbt_record_offset for x in instances]
            resource = runtime.native_resource("optix", "instance_scene", version=2 if any(offsets) else 1,
                parameters=dict(instance_count=len(instances), allow_update=allow_update,
                    custom_indices=[x.custom_index for x in instances], visibility_masks=[x.mask for x in instances],
                    opaque=[x.opaque for x in instances], **(dict(sbt_offsets=offsets) if any(offsets) else {})),
                bindings=dict(context=self.resource, transforms=poses,
                              **{f"gas_{i}": x.gas.resource for i, x in enumerate(instances)}))
            result = _Scene(self.package, resource, self, len(instances))
        except BaseException:
            poses.close()
            raise
        result._seed = poses
        return result


class _Geometry(NativeObject):
    def __init__(self, package, resource, provider):
        super().__init__(package, resource)
        self.provider = provider


class _Scene(_Geometry):
    def __init__(self, package, resource, provider, count):
        super().__init__(package, resource, provider)
        self.instance_count = count
        self._seed = None

    def close(self):
        super().close()
        if self._seed is not None:
            self._seed.close()
            self._seed = None

    def _query(self, count, rays, hits, ids, scene, face_rules, any_hit, occlusion, graph_fixed_query):
        self._check()
        if type(count) is not int or count <= 0 or any(type(x) is not bool for x in (any_hit, graph_fixed_query)):
            raise ValueError("Invalid OptiX query count or policy")
        bindings = dict(scene=self if scene is None else scene, rays=rays, hits=hits)
        schemas = dict(scene=dict(kind="native_resource"), rays=_array("f32", 2),
                       hits=_array("i32" if occlusion else "f32", 1 if occlusion else 2))
        effects = dict(scene="read", rays="read", hits="write")
        parameters = dict(occlusion=occlusion, any_hit=any_hit, graph_fixed_query=graph_fixed_query)
        if isinstance(face_rules, OptixFaceRuleTable):
            version = 5
            bindings["faces"], schemas["faces"], effects["faces"] = face_rules.binding, _array("i32", 1), "read"
        else:
            rules = tuple(face_rules or ("two_sided",) * self.instance_count)
            if len(rules) != self.instance_count or len(set(rules)) != 1 or rules[0] not in ("two_sided", "front_only"):
                raise ValueError("Mixed face rules require a live OptixFaceRuleTable")
            version = 4
            parameters["front_only"] = rules[0] == "front_only"
        if ids is not None:
            bindings["hit_indices"], schemas["hit_indices"], effects["hit_indices"] = ids, _array("i32", 2), "write"
        result = NativeRecording(self.package, "optix", "query", version=version,
            parameters=parameters, bindings=bindings, schemas=schemas, effects=effects)
        # Whole-array schemas derive their extent at prepare. Require the
        # requested count at the same binding boundary, never scan ray data.
        result.extents = {"rays": count}
        return result

    def record_typed(self, ray_count, *, rays="rays", hits="hits", hit_indices="hit_indices",
                     scene=None, face_rules=None, any_hit=False, graph_fixed_query=True):
        return self._query(ray_count, rays, hits, hit_indices, scene, face_rules, any_hit, False, graph_fixed_query)

    def record_occlusion(self, ray_count, *, rays="rays", occluded="occluded", scene=None,
                         face_rules=None, graph_fixed_query=True):
        return self._query(ray_count, rays, occluded, None, scene, face_rules, True, True, graph_fixed_query)

    def record_refit_transforms(self, *, transforms, scene=None):
        self._check()
        return NativeRecording(self.package, "optix", "instance_refit",
            bindings=dict(scene=self if scene is None else scene, transforms=transforms),
            schemas=dict(scene=dict(kind="native_resource"), transforms=_array("f32", 3)),
            effects=dict(scene="read_write", transforms="read"))

    def memory_report(self):
        # The current public native handle has no per-resource accounting API.
        return SimpleNamespace(to_dict=lambda: dict(complete=False, known_resident_requested_bytes=0,
            unknown_components=["OptiX acceleration structures and provider storage"]))
