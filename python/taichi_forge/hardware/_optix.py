"""Optional user-built OptiX provider behind the Forge C ABI."""

import ctypes
from dataclasses import dataclass
import os
from pathlib import Path
from types import MappingProxyType
import weakref

from taichi_forge._hardware_telemetry import (
    hardware_failure_phase,
    instrument_hardware_recording,
)
from taichi_forge.graph._ir import GraphAccess, ResourceEffect
from taichi_forge.graph._native import BackendCommandRecording
from taichi_forge.hardware._memory import HardwareMemoryComponent, make_memory_report
from taichi_forge.hardware._native_adapter import (
    native_recording_node,
    runtime_generation_matches,
    static_resource_effect,
    validate_exact_bindings,
    validate_runtime_generation,
)
from taichi_forge.hardware._runtime import active_backend
from taichi_forge.lang import impl
from taichi_forge.lang._ndarray import Ndarray
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.types.primitive_types import f32, i32


PROVIDER_ABI_VERSION = 1
PROVIDER_ABI_NAME = "taichi-forge-optix-provider-c-abi1"
PROVIDER_QUERY_SYMBOL = "taichi_forge_optix_provider_query"
SUPPORTED_OPTIX_ABIS = (93, 105)

_SUCCESS = 0
_REQUIRED_FEATURES = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)
_loaded_providers = weakref.WeakSet()


class _ProviderInfo(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("provider_abi_version", ctypes.c_uint32),
        ("optix_abi_version", ctypes.c_uint32),
        ("optix_version", ctypes.c_uint32),
        ("features", ctypes.c_uint64),
        ("provider_name", ctypes.c_char_p),
        ("build_identity", ctypes.c_char_p),
    ]


class _ContextDesc(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("device_ordinal", ctypes.c_uint32),
        ("cuda_context", ctypes.c_uint64),
        ("validation_mode", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class _TriangleSceneDesc(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("vertex_count", ctypes.c_uint32),
        ("triangle_count", ctypes.c_uint32),
        ("allow_update", ctypes.c_uint32),
        ("vertices", ctypes.c_uint64),
        ("indices", ctypes.c_uint64),
        ("cuda_stream", ctypes.c_uint64),
    ]


class _TraceDesc(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("ray_count", ctypes.c_uint32),
        ("rays", ctypes.c_uint64),
        ("hits", ctypes.c_uint64),
        ("cuda_stream", ctypes.c_uint64),
    ]


class _SceneMemory(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("gas_bytes", ctypes.c_uint64),
        ("ias_bytes", ctypes.c_uint64),
        ("build_update_scratch_bytes", ctypes.c_uint64),
        ("instance_bytes", ctypes.c_uint64),
        ("launch_params_bytes", ctypes.c_uint64),
        ("shared_pipeline_sbt_bytes", ctypes.c_uint64),
    ]


_CreateContext = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.POINTER(_ContextDesc), ctypes.POINTER(ctypes.c_void_p)
)
_DestroyContext = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
_CreateScene = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.POINTER(_TriangleSceneDesc),
    ctypes.POINTER(ctypes.c_void_p),
)
_UpdateScene = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(_TriangleSceneDesc)
)
_Trace = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(_TraceDesc))
_GetSceneMemory = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(_SceneMemory)
)
_DestroyScene = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
_GetLastError = ctypes.CFUNCTYPE(
    ctypes.c_size_t, ctypes.POINTER(ctypes.c_char), ctypes.c_size_t
)


class _ProviderApi(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("provider_abi_version", ctypes.c_uint32),
        ("info", _ProviderInfo),
        ("create_context", _CreateContext),
        ("destroy_context", _DestroyContext),
        ("create_triangle_scene", _CreateScene),
        ("update_triangle_scene", _UpdateScene),
        ("trace", _Trace),
        ("get_scene_memory", _GetSceneMemory),
        ("destroy_triangle_scene", _DestroyScene),
        ("get_last_error", _GetLastError),
    ]


@dataclass(frozen=True)
class _LoadedApi:
    library: object
    path: str
    api: _ProviderApi


def _decode(value):
    return "" if not value else value.decode("utf-8", errors="replace")


def _provider_error(api):
    required = int(api.get_last_error(None, 0))
    if required <= 1:
        return "optional OptiX provider call failed"
    buffer = ctypes.create_string_buffer(required)
    api.get_last_error(buffer, required)
    return buffer.value.decode("utf-8", errors="replace")


def _check_api(api):
    if api.struct_size < ctypes.sizeof(_ProviderApi):
        raise RuntimeError("OptiX provider returned a truncated Forge API table")
    if api.provider_abi_version != PROVIDER_ABI_VERSION:
        raise RuntimeError("OptiX provider returned a mismatched Forge ABI")
    if api.info.struct_size < ctypes.sizeof(_ProviderInfo):
        raise RuntimeError("OptiX provider returned truncated identity facts")
    if api.info.provider_abi_version != PROVIDER_ABI_VERSION:
        raise RuntimeError("OptiX provider identity uses a mismatched Forge ABI")
    if api.info.optix_abi_version not in SUPPORTED_OPTIX_ABIS:
        raise RuntimeError(
            "OptiX provider SDK ABI is outside Forge's qualified source range"
        )
    if int(api.info.features) & _REQUIRED_FEATURES != _REQUIRED_FEATURES:
        raise RuntimeError("OptiX provider does not implement the complete ray ABI")
    for name in (
        "create_context",
        "destroy_context",
        "create_triangle_scene",
        "update_triangle_scene",
        "trace",
        "get_scene_memory",
        "destroy_triangle_scene",
        "get_last_error",
    ):
        if not bool(getattr(api, name)):
            raise RuntimeError(f"OptiX provider is missing ABI entry {name}")


def _load_library(path):
    return ctypes.CDLL(path)


def _query_provider(path):
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError("OptiX provider path must be a string or path-like value")
    resolved = str(Path(path).expanduser().resolve())
    library = _load_library(resolved)
    try:
        query = getattr(library, PROVIDER_QUERY_SYMBOL)
    except AttributeError as exc:
        raise RuntimeError("OptiX provider query symbol is missing") from exc
    query.argtypes = [
        ctypes.c_uint32,
        ctypes.c_size_t,
        ctypes.POINTER(_ProviderApi),
    ]
    query.restype = ctypes.c_int
    api = _ProviderApi()
    result = int(
        query(PROVIDER_ABI_VERSION, ctypes.sizeof(_ProviderApi), ctypes.byref(api))
    )
    if result != _SUCCESS:
        message = (
            _provider_error(api)
            if bool(api.get_last_error)
            else f"provider query failed with result {result}"
        )
        raise RuntimeError(message)
    _check_api(api)
    return _LoadedApi(library, resolved, api)


def _format_optix_version(value):
    value = int(value)
    return f"{value // 10000}.{(value // 100) % 100}.{value % 100}"


def probe_provider(path=None):
    """Query only the Forge provider table; never initialize OptiX or CUDA."""

    if path is None:
        path = os.environ.get("TAICHI_FORGE_OPTIX_PROVIDER")
    native_facts = {
        "probe_policy": "explicit_transient_plugin_query",
        "provider_enablement_changed": False,
        "provider_selection_changed": False,
        "execution_qualified": False,
        "supported_optix_abi_versions": SUPPORTED_OPTIX_ABIS,
    }
    result = {
        "provider_id": "optix",
        "external_component_probed": False,
        "discovery": "missing",
        "unavailable_reason": "provider_library_path_required",
        "provider_abi": PROVIDER_ABI_NAME,
        "provider_version": None,
        "last_error": None,
        "failure_scope": None,
        "native_facts": native_facts,
    }
    if not path:
        return result
    result["external_component_probed"] = True
    try:
        loaded = _query_provider(path)
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        result.update(
            discovery="incompatible",
            unavailable_reason="provider_abi_query_failed",
            last_error=str(exc) or type(exc).__name__,
            failure_scope="provider",
        )
        native_facts["library_loaded_transiently"] = False
        return result
    info = loaded.api.info
    result.update(
        discovery="present",
        unavailable_reason="execution_not_qualified",
        provider_version=_format_optix_version(info.optix_version),
    )
    native_facts.update(
        library_candidate=loaded.path,
        library_loaded_transiently=True,
        query_only=True,
        optix_abi_version=int(info.optix_abi_version),
        optix_version=int(info.optix_version),
        provider_name=_decode(info.provider_name),
        build_identity=_decode(info.build_identity),
        feature_bits=int(info.features),
    )
    return result


def passive_status():
    loaded = tuple(provider for provider in _loaded_providers if not provider.closed)
    native_facts = {
        "status_policy": "passive_loaded_optix_plugins",
        "external_component_probed": False,
        "provider_enablement_changed": False,
        "provider_selection_changed": False,
        "loaded_provider_count": len(loaded),
    }
    if not loaded:
        return {
            "provider_id": "optix",
            "library_loaded": False,
            "provider_abi": PROVIDER_ABI_NAME,
            "provider_version": None,
            "native_facts": native_facts,
        }
    provider = loaded[0]
    native_facts.update(provider.identity)
    return {
        "provider_id": "optix",
        "library_loaded": True,
        "provider_abi": PROVIDER_ABI_NAME,
        "provider_version": provider.identity["provider_version"],
        "native_facts": native_facts,
    }


def _item_count(value, width, dtype, name):
    if not isinstance(value, Ndarray):
        raise TaichiRuntimeError(f"OptiX ray {name} must be a Taichi ndarray")
    shape = tuple(value.shape)
    element_shape = tuple(value.element_shape)
    if value.dtype != dtype:
        raise TaichiRuntimeError(f"OptiX ray {name} must use dtype {dtype}")
    if element_shape == () and len(shape) == 2 and shape[1] == width:
        count = shape[0]
    elif element_shape == (width,) and len(shape) == 1:
        count = shape[0]
    else:
        raise TaichiRuntimeError(
            f"OptiX ray {name} must have scalar shape (N, {width}) or "
            f"AOS vector-{width} shape (N,)"
        )
    if count <= 0:
        raise TaichiRuntimeError(f"OptiX ray {name} must not be empty")
    return count


def _device_pointer(value):
    return int(value.arr.device_allocation_ptr())


class OptixProvider:
    """Explicit owner of one user-built OptiX provider and CUDA context view."""

    def __init__(self, library_path, *, validation=False):
        program = impl.get_runtime().prog
        if program is None or active_backend() != "cuda":
            raise TaichiRuntimeError(
                "OptixProvider requires an initialized Taichi CUDA runtime"
            )
        if not isinstance(validation, bool):
            raise TypeError("validation must be a bool")
        with hardware_failure_phase("provider_load_failure"):
            loaded = _query_provider(library_path)
        with hardware_failure_phase("provider_plan_failure"):
            context = ctypes.c_void_p()
            desc = _ContextDesc(ctypes.sizeof(_ContextDesc), 0, 0, int(validation), 0)
            result = int(
                loaded.api.create_context(ctypes.byref(desc), ctypes.byref(context))
            )
            if result != _SUCCESS or not context.value:
                raise TaichiRuntimeError(_provider_error(loaded.api))
        self._loaded = loaded
        self._context = context
        self._runtime_prog = program
        self._runtime_generation = int(impl.runtime_generation())
        self._scenes = weakref.WeakSet()
        info = loaded.api.info
        self.identity = MappingProxyType(
            {
                "library_candidate": loaded.path,
                "provider_abi": PROVIDER_ABI_NAME,
                "provider_version": _format_optix_version(info.optix_version),
                "optix_abi_version": int(info.optix_abi_version),
                "optix_version": int(info.optix_version),
                "provider_name": _decode(info.provider_name),
                "build_identity": _decode(info.build_identity),
                "feature_bits": int(info.features),
            }
        )
        _loaded_providers.add(self)

    @property
    def closed(self):
        return self._context is None

    def triangle_scene(self, vertices, indices, *, allow_update=True):
        self._validate_lifetime()
        return OptixTriangleScene(self, vertices, indices, allow_update=allow_update)

    def _validate_lifetime(self):
        if self._context is None:
            raise TaichiRuntimeError("OptixProvider has been closed")
        validate_runtime_generation(
            self, "OptixProvider belongs to a previous Taichi runtime generation"
        )

    def close(self):
        if self._context is None:
            return None
        live = tuple(scene for scene in self._scenes if not scene.closed)
        if live:
            raise TaichiRuntimeError(
                "OptixProvider cannot close while triangle scenes are live"
            )
        context = self._context
        self._context = None
        if runtime_generation_matches(self):
            self._runtime_prog.synchronize()
            result = int(self._loaded.api.destroy_context(context))
            if result != _SUCCESS:
                self._context = context
                raise TaichiRuntimeError(_provider_error(self._loaded.api))
        return None

    destroy = close

    def __enter__(self):
        self._validate_lifetime()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


@instrument_hardware_recording("ray.query.batch.optix")
class OptixRayQueryRecording(BackendCommandRecording):
    """One runtime-ordered OptiX launch against a fixed scene generation."""

    def __init__(self, scene, ray_count, *, rays="rays", hits="hits"):
        if not isinstance(scene, OptixTriangleScene):
            raise TypeError("OptiX ray query recording requires an OptixTriangleScene")
        if (
            isinstance(ray_count, bool)
            or not isinstance(ray_count, int)
            or not 1 <= ray_count <= 0xFFFFFFFF
        ):
            raise ValueError("OptiX ray count must be in [1, UINT32_MAX]")
        if any(not isinstance(name, str) or not name for name in (rays, hits)):
            raise ValueError("OptiX ray bindings must be nonempty strings")
        if rays == hits:
            raise ValueError("OptiX ray bindings must be unique")
        super().__init__(
            backend="cuda",
            binding_names=(rays, hits),
            command_count=1,
            queue="compute",
            stream_binding="runtime_ordered",
            barrier_policy="internal",
            workspace_ownership="provider_generation",
            replay_mode="rerecord",
            no_host_readback=True,
        )
        object.__setattr__(self, "scene", scene)
        object.__setattr__(self, "ray_count", ray_count)
        object.__setattr__(self, "rays", rays)
        object.__setattr__(self, "hits", hits)

    @property
    def resource_effects(self):
        return (
            ResourceEffect(self.rays, GraphAccess.READ),
            ResourceEffect(self.hits, GraphAccess.WRITE),
            static_resource_effect(self.scene._effect_name, GraphAccess.READ),
        )

    def execute(self, bindings):
        validate_exact_bindings(self, bindings, "OptiX ray query")
        self.validate_graph_lifetime()
        rays = bindings[self.rays]
        hits = bindings[self.hits]
        if _item_count(rays, 8, f32, self.rays) != self.ray_count:
            raise TaichiRuntimeError("OptiX ray binding has the wrong ray count")
        if _item_count(hits, 4, f32, self.hits) != self.ray_count:
            raise TaichiRuntimeError("OptiX hit binding has the wrong ray count")
        self.scene._execute_query(rays, hits, self.ray_count)

    def validate_graph_lifetime(self):
        self.scene._validate_lifetime()

    def memory_report(self):
        return self.scene.memory_report()

    def _as_graph_native_node(self):
        return native_recording_node(
            self,
            lifetime_leases=lambda item: (item.scene, item.scene.provider),
            debug_info=lambda item: {
                "kind": "optix_triangle_ray_query",
                "ray_count": item.ray_count,
                "provider_abi": PROVIDER_ABI_NAME,
            },
        )


@instrument_hardware_recording("ray.as_refit.optix")
class OptixRayRefitRecording(BackendCommandRecording):
    """One fixed-topology OptiX GAS update."""

    def __init__(self, scene, *, vertices="vertices"):
        if not isinstance(scene, OptixTriangleScene):
            raise TypeError("OptiX refit recording requires an OptixTriangleScene")
        if not isinstance(vertices, str) or not vertices:
            raise ValueError("OptiX refit binding must be a nonempty string")
        super().__init__(
            backend="cuda",
            binding_names=(vertices,),
            command_count=1,
            queue="compute",
            stream_binding="runtime_ordered",
            barrier_policy="internal",
            workspace_ownership="provider_generation",
            replay_mode="rerecord",
            no_host_readback=True,
        )
        object.__setattr__(self, "scene", scene)
        object.__setattr__(self, "vertices", vertices)

    @property
    def resource_effects(self):
        return (
            ResourceEffect(self.vertices, GraphAccess.READ),
            static_resource_effect(self.scene._effect_name, GraphAccess.WRITE),
        )

    def execute(self, bindings):
        validate_exact_bindings(self, bindings, "OptiX ray refit")
        self.validate_graph_lifetime()
        self.scene.refit(bindings[self.vertices])

    def validate_graph_lifetime(self):
        self.scene._validate_lifetime()

    def memory_report(self):
        return self.scene.memory_report()

    def _as_graph_native_node(self):
        return native_recording_node(
            self,
            lifetime_leases=lambda item: (item.scene, item.scene.provider),
            debug_info=lambda item: {
                "kind": "optix_triangle_gas_update",
                "vertex_count": item.scene.vertex_count,
            },
        )


class OptixTriangleScene:
    """Fixed-topology triangle GAS plus identity IAS owned by a provider."""

    def __init__(self, provider, vertices, indices, *, allow_update=True):
        if not isinstance(provider, OptixProvider):
            raise TypeError("provider must be an OptixProvider")
        if not isinstance(allow_update, bool):
            raise TypeError("allow_update must be a bool")
        provider._validate_lifetime()
        vertex_count = _item_count(vertices, 3, f32, "vertices")
        triangle_count = _item_count(indices, 3, i32, "indices")
        scene = ctypes.c_void_p()
        desc = _TriangleSceneDesc(
            ctypes.sizeof(_TriangleSceneDesc),
            vertex_count,
            triangle_count,
            int(allow_update),
            _device_pointer(vertices),
            _device_pointer(indices),
            0,
        )
        with hardware_failure_phase("provider_plan_failure"):
            result = int(
                provider._loaded.api.create_triangle_scene(
                    provider._context, ctypes.byref(desc), ctypes.byref(scene)
                )
            )
            if result != _SUCCESS or not scene.value:
                raise TaichiRuntimeError(_provider_error(provider._loaded.api))
        self.provider = provider
        self._scene = scene
        self._runtime_prog = provider._runtime_prog
        self._runtime_generation = provider._runtime_generation
        self.vertex_count = vertex_count
        self.triangle_count = triangle_count
        self.allow_update = allow_update
        self._indices = indices
        self._effect_name = (
            f"optix-ray-scene:{self._runtime_generation}:{int(scene.value)}"
        )
        memory = _SceneMemory()
        memory.struct_size = ctypes.sizeof(_SceneMemory)
        result = int(provider._loaded.api.get_scene_memory(scene, ctypes.byref(memory)))
        if result != _SUCCESS:
            provider._loaded.api.destroy_triangle_scene(scene)
            self._scene = None
            raise TaichiRuntimeError(_provider_error(provider._loaded.api))
        self._memory = memory
        provider._scenes.add(self)

    @property
    def closed(self):
        return self._scene is None

    def record(self, ray_count, *, rays="rays", hits="hits"):
        self._validate_lifetime()
        return OptixRayQueryRecording(self, ray_count, rays=rays, hits=hits)

    def trace(self, rays, hits):
        ray_count = _item_count(rays, 8, f32, "rays")
        self.record(ray_count).execute({"rays": rays, "hits": hits})
        return hits

    def record_refit(self, *, vertices="vertices"):
        self._validate_lifetime()
        if not self.allow_update:
            raise TaichiRuntimeError("OptiX scene was not created for updates")
        return OptixRayRefitRecording(self, vertices=vertices)

    def refit(self, vertices):
        self._validate_lifetime()
        if not self.allow_update:
            raise TaichiRuntimeError("OptiX scene was not created for updates")
        if _item_count(vertices, 3, f32, "vertices") != self.vertex_count:
            raise TaichiRuntimeError("OptiX refit must preserve the vertex count")
        desc = _TriangleSceneDesc(
            ctypes.sizeof(_TriangleSceneDesc),
            self.vertex_count,
            self.triangle_count,
            1,
            _device_pointer(vertices),
            _device_pointer(self._indices),
            0,
        )
        with hardware_failure_phase("provider_execution_failure"):
            result = int(
                self.provider._loaded.api.update_triangle_scene(
                    self._scene, ctypes.byref(desc)
                )
            )
            if result != _SUCCESS:
                raise TaichiRuntimeError(_provider_error(self.provider._loaded.api))
        return self

    def _execute_query(self, rays, hits, ray_count):
        self._validate_lifetime()
        desc = _TraceDesc(
            ctypes.sizeof(_TraceDesc),
            ray_count,
            _device_pointer(rays),
            _device_pointer(hits),
            0,
        )
        with hardware_failure_phase("provider_execution_failure"):
            result = int(
                self.provider._loaded.api.trace(self._scene, ctypes.byref(desc))
            )
            if result != _SUCCESS:
                raise TaichiRuntimeError(_provider_error(self.provider._loaded.api))

    def _validate_lifetime(self):
        if self._scene is None:
            raise TaichiRuntimeError("OptixTriangleScene has been closed")
        self.provider._validate_lifetime()
        validate_runtime_generation(
            self,
            "OptixTriangleScene belongs to a previous Taichi runtime generation",
        )

    def memory_report(self):
        resident = self._scene is not None and runtime_generation_matches(self)
        memory = self._memory
        components = (
            HardwareMemoryComponent(
                "gas_storage",
                int(memory.gas_bytes),
                True,
                "provider_generation",
                "provider",
                resident=resident,
            ),
            HardwareMemoryComponent(
                "identity_ias_storage",
                int(memory.ias_bytes),
                True,
                "provider_generation",
                "provider",
                resident=resident,
            ),
            HardwareMemoryComponent(
                "build_update_scratch",
                int(memory.build_update_scratch_bytes),
                True,
                "provider_generation",
                "provider",
                resident=resident,
            ),
            HardwareMemoryComponent(
                "instance_and_launch_params",
                int(memory.instance_bytes + memory.launch_params_bytes),
                True,
                "provider_generation",
                "provider",
                resident=resident,
            ),
            HardwareMemoryComponent(
                "shared_pipeline_sbt",
                int(memory.shared_pipeline_sbt_bytes),
                True,
                "runtime",
                "provider",
                resident=resident,
            ),
            HardwareMemoryComponent(
                "optix_driver_context_state",
                None,
                False,
                "runtime",
                "driver",
                resident=resident,
            ),
        )
        return make_memory_report(
            "optix_triangle_ray",
            "cuda",
            components,
            lifecycle_state="ready" if resident else "closed",
            ownership_scope="provider_context_and_scene_generation",
        )

    def _graph_provider_memory_report(self):
        return self.memory_report()

    def close(self):
        if self._scene is None:
            return None
        scene = self._scene
        self._scene = None
        if runtime_generation_matches(self):
            self._runtime_prog.synchronize()
            result = int(self.provider._loaded.api.destroy_triangle_scene(scene))
            if result != _SUCCESS:
                self._scene = scene
                raise TaichiRuntimeError(_provider_error(self.provider._loaded.api))
        return None

    destroy = close

    def __enter__(self):
        self._validate_lifetime()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def load_provider(library_path, *, validation=False):
    return OptixProvider(library_path, validation=validation)


def is_loaded():
    return bool(passive_status()["library_loaded"])


__all__ = [
    "OptixProvider",
    "OptixRayQueryRecording",
    "OptixRayRefitRecording",
    "OptixTriangleScene",
    "PROVIDER_ABI_NAME",
    "PROVIDER_ABI_VERSION",
    "SUPPORTED_OPTIX_ABIS",
    "is_loaded",
    "load_provider",
    "passive_status",
    "probe_provider",
]
