"""OptiX Host factories backed by the existing Runtime native catalog.

No numerical frontend or producer scene is loaded. Geometry seeds are consumer
inputs; contexts, acceleration structures and calls retain the native leases.
"""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import os

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


class RayAPI:
    def __init__(self, package):
        self.package = package
        self.OptixRayInstance = OptixRayInstance
        self.OptixFaceRuleTable = OptixFaceRuleTable

    def is_available(self):
        # This predicate denotes Vulkan inline ray queries, not OptiX.
        return False

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
        paths = [Path(provider_path)] if provider_path else []
        if not paths:
            for name in ("taichi_forge_runtime", "taichi_forge"):
                try:
                    spec = importlib.util.find_spec(name)
                except ImportError:
                    continue
                for root in (() if spec is None else spec.submodule_search_locations or ()):
                    directory = Path(root) / "_lib" / "hardware_providers"
                    suffix = ".dll" if os.name == "nt" else ".so"
                    paths.extend(sorted(directory.glob("*taichi_forge_optix_provider_abi1_optix118" + suffix)))
        if not paths:
            raise RuntimeError("No bundled OptiX provider adapter; supply provider_path")
        # Configuration belongs to this Runtime; creation validates the actual
        # adapter ABI/features and retains its library independent of later changes.
        runtime = self.package.runtime
        errors = []
        for path in paths:
            runtime.configure_provider("optix", adapter_path=path,
                                       library_path=library_path or os.environ.get("TAICHI_FORGE_OPTIX_LIBRARY"))
            try:
                resource = runtime.native_resource("optix", "context", parameters=dict(validation=validation))
            except Exception as exc:
                errors.append(f"{path}: {exc}")
                continue
            return _Provider(self.package, resource, path)
        raise RuntimeError("Cannot create OptiX context: " + "; ".join(errors))


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
