"""Optional OptiX runtime behind bundled Forge C-ABI adapters."""

import ctypes
from dataclasses import dataclass
from functools import partial
import importlib.util
import math
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
from taichi_forge.hardware._ray_identity import RayResourceIdentity, identify_ray_recording
from taichi_forge.hardware._ray_memory import (
    aggregate_ray_memory, ray_resource_resident,
)
from taichi_forge.hardware._optix_micromap import (
    OptixOpacityMicromap,
    _MicromapDesc,
    _MicromapMemory,
)
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
from taichi_forge.lang._storage_view import describe_storage
from taichi_forge.lang._texture import Texture
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.types.primitive_types import f32, i32, u32


PROVIDER_ABI_VERSION = 1
PROVIDER_ABI_NAME = "taichi-forge-optix-provider-c-abi1"
PROVIDER_QUERY_SYMBOL = "taichi_forge_optix_provider_query"
SUPPORTED_OPTIX_ABIS = (93, 105, 118)

_SUCCESS = 0
_OPTIX_UNAVAILABLE = 4
_REQUIRED_FEATURES = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)
_TYPED_HITS = 1 << 6
_WORD_ALIGNED_QUERY_STORAGE = 1 << 7
_SHARED_TRIANGLE_GAS = 1 << 8
_MULTI_INSTANCE_IAS = 1 << 9
_DEVICE_INSTANCE_TRANSFORM_UPDATE = 1 << 10
_ALPHA_MASK = 1 << 11
_INSTANCE_OPACITY = 1 << 12
_OPACITY_MICROMAP_IMPORT = 1 << 13
_INSTANCE_FEATURES = (
    _SHARED_TRIANGLE_GAS | _MULTI_INSTANCE_IAS | _DEVICE_INSTANCE_TRANSFORM_UPDATE
)
_loaded_providers = weakref.WeakSet()


def _provider_filename(optix_abi):
    stem = f"taichi_forge_optix_provider_abi1_optix{optix_abi}"
    if os.name == "nt":
        return f"{stem}.dll"
    return f"lib{stem}.so"


def _runtime_package_roots():
    roots = []
    spec = importlib.util.find_spec("taichi_forge_runtime")
    if spec is not None and spec.submodule_search_locations is not None:
        roots.extend(Path(path) for path in spec.submodule_search_locations)
    roots.append(Path(__file__).resolve().parents[1])
    return roots


def _bundled_provider_candidates():
    candidates = []
    seen = set()
    for root in _runtime_package_roots():
        directory = root / "_lib" / "hardware_providers"
        for optix_abi in reversed(SUPPORTED_OPTIX_ABIS):
            candidate = directory / _provider_filename(optix_abi)
            key = os.path.normcase(str(candidate))
            if key not in seen and candidate.is_file():
                candidates.append(str(candidate))
                seen.add(key)
    return tuple(candidates)


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
        ("runtime_library_path", ctypes.c_char_p),
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


class _TypedTraceDesc(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("ray_count", ctypes.c_uint32),
        ("rays", ctypes.c_uint64),
        ("hits", ctypes.c_uint64),
        ("hit_indices", ctypes.c_uint64),
        ("cuda_stream", ctypes.c_uint64),
    ]


class _AlphaTraceDesc(ctypes.Structure):
    _fields_ = _TypedTraceDesc._fields_ + [
        ("masks", ctypes.c_uint64),
        ("launch_params", ctypes.c_uint64),
        ("mask_count", ctypes.c_uint32),
        ("any_hit", ctypes.c_uint32),
    ]


class _AlphaMask(ctypes.Structure):
    _fields_ = [
        ("uvs", ctypes.c_uint64),
        ("indices", ctypes.c_uint64),
        ("texture", ctypes.c_uint64),
        ("cutoff", ctypes.c_float),
        ("channel", ctypes.c_uint32),
    ]


class _InstanceDesc(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("gas", ctypes.c_void_p),
        ("transform", ctypes.c_float * 12),
        ("custom_index", ctypes.c_uint32),
        ("visibility_mask", ctypes.c_uint32),
    ]


class _InstanceSceneDesc(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("instance_count", ctypes.c_uint32),
        ("allow_update", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("instances", ctypes.POINTER(_InstanceDesc)),
        ("cuda_stream", ctypes.c_uint64),
    ]


class _InstanceUpdateDesc(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("instance_count", ctypes.c_uint32),
        ("transforms", ctypes.c_uint64),
        ("cuda_stream", ctypes.c_uint64),
    ]


_ProbeRuntime = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p)
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
_PrepareTyped = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
_TraceTyped = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(_TypedTraceDesc)
)
_CreateTriangleGas = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.POINTER(_TriangleSceneDesc),
    ctypes.POINTER(ctypes.c_void_p),
)
_UpdateTriangleGas = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(_TriangleSceneDesc)
)
_GetTriangleGasMemory = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(_SceneMemory)
)
_DestroyTriangleGas = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
_CreateInstanceScene = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.POINTER(_InstanceSceneDesc),
    ctypes.POINTER(ctypes.c_void_p),
)
_UpdateInstanceScene = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(_InstanceUpdateDesc)
)
_TraceInstanceScene = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(_TraceDesc)
)
_TraceInstanceSceneTyped = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(_TypedTraceDesc)
)
_GetInstanceSceneMemory = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(_SceneMemory)
)
_DestroyInstanceScene = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
_GetLastError = ctypes.CFUNCTYPE(
    ctypes.c_size_t, ctypes.POINTER(ctypes.c_char), ctypes.c_size_t
)
_TraceAlpha = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(_AlphaTraceDesc)
)
_CreateMicromapGas = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.POINTER(_TriangleSceneDesc),
    ctypes.POINTER(_MicromapDesc),
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(_MicromapMemory),
)


class _ProviderApi(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("provider_abi_version", ctypes.c_uint32),
        ("info", _ProviderInfo),
        ("probe_runtime", _ProbeRuntime),
        ("create_context", _CreateContext),
        ("destroy_context", _DestroyContext),
        ("create_triangle_scene", _CreateScene),
        ("update_triangle_scene", _UpdateScene),
        ("trace", _Trace),
        ("get_scene_memory", _GetSceneMemory),
        ("destroy_triangle_scene", _DestroyScene),
        ("get_last_error", _GetLastError),
        ("prepare_typed", _PrepareTyped),
        ("trace_typed", _TraceTyped),
        ("create_triangle_gas", _CreateTriangleGas),
        ("update_triangle_gas", _UpdateTriangleGas),
        ("get_triangle_gas_memory", _GetTriangleGasMemory),
        ("destroy_triangle_gas", _DestroyTriangleGas),
        ("create_instance_scene", _CreateInstanceScene),
        ("update_instance_scene", _UpdateInstanceScene),
        ("trace_instance_scene", _TraceInstanceScene),
        ("trace_instance_scene_typed", _TraceInstanceSceneTyped),
        ("get_instance_scene_memory", _GetInstanceSceneMemory),
        ("destroy_instance_scene", _DestroyInstanceScene),
        ("prepare_alpha", _PrepareTyped),
        ("trace_alpha", _TraceAlpha),
        ("trace_instance_alpha", _TraceAlpha),
        ("create_triangle_gas_micromap", _CreateMicromapGas),
        ("trace_instance_micromap", _TraceAlpha),
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
    if api.struct_size < _ProviderApi.prepare_typed.offset:
        raise RuntimeError("OptiX provider returned a truncated Forge API table")
    if api.provider_abi_version != PROVIDER_ABI_VERSION:
        raise RuntimeError("OptiX provider returned a mismatched Forge ABI")
    if api.info.struct_size < ctypes.sizeof(_ProviderInfo):
        raise RuntimeError("OptiX provider returned truncated identity facts")
    if api.info.provider_abi_version != PROVIDER_ABI_VERSION:
        raise RuntimeError("OptiX provider identity uses a mismatched Forge ABI")
    if api.info.optix_abi_version not in SUPPORTED_OPTIX_ABIS:
        raise RuntimeError(
            "OptiX provider SDK ABI is outside Forge's bundled adapter range"
        )
    if int(api.info.features) & _REQUIRED_FEATURES != _REQUIRED_FEATURES:
        raise RuntimeError("OptiX provider does not implement the complete ray ABI")
    for name in (
        "probe_runtime",
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


def _api_has(api, name):
    return int(api.struct_size) >= getattr(_ProviderApi, name).offset + ctypes.sizeof(
        ctypes.c_void_p
    ) and bool(getattr(api, name))


def _require_instance_api(api):
    if int(api.info.features) & _INSTANCE_FEATURES != _INSTANCE_FEATURES:
        raise TaichiRuntimeError(
            "OptiX adapter does not support shared GAS instance scenes; "
            "use a newer Forge adapter"
        )
    for name in (
        "create_triangle_gas",
        "update_triangle_gas",
        "get_triangle_gas_memory",
        "destroy_triangle_gas",
        "create_instance_scene",
        "update_instance_scene",
        "trace_instance_scene",
        "trace_instance_scene_typed",
        "get_instance_scene_memory",
        "destroy_instance_scene",
    ):
        if not _api_has(api, name):
            raise TaichiRuntimeError(
                "OptiX adapter instance feature table is truncated or incomplete"
            )


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


def _resolved_path(path):
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError("OptiX library path must be a string or path-like value")
    return str(Path(path).expanduser().resolve())


def _provider_and_runtime_candidates(library_path=None):
    runtime_path = os.environ.get("TAICHI_FORGE_OPTIX_LIBRARY") or None
    if library_path is not None:
        runtime_path = _resolved_path(library_path)
    if runtime_path:
        runtime_path = _resolved_path(runtime_path)
    return _bundled_provider_candidates(), runtime_path, "forge_runtime_wheel"


def _provider_candidates_for_load(library_path=None, provider_path=None):
    candidates, runtime_path, provider_source = _provider_and_runtime_candidates(
        library_path
    )
    if provider_path is not None:
        candidates = (_resolved_path(provider_path),)
        provider_source = "explicit_adapter_path"
    return candidates, runtime_path, provider_source


def _runtime_library_argument(runtime_path):
    return None if runtime_path is None else os.fsencode(runtime_path)


def _probe_provider_runtime(loaded, runtime_path):
    result = int(loaded.api.probe_runtime(_runtime_library_argument(runtime_path)))
    if result != _SUCCESS:
        raise RuntimeError(_provider_error(loaded.api))
    return True


def _format_optix_version(value):
    value = int(value)
    return f"{value // 10000}.{(value // 100) % 100}.{value % 100}"


def probe_provider(path=None):
    """Probe bundled adapters and the vendor runtime without retaining them."""

    native_facts = {
        "probe_policy": "transient_adapter_and_vendor_runtime_query",
        "provider_enablement_changed": False,
        "provider_selection_changed": False,
        "execution_qualified": False,
        "supported_optix_abi_versions": SUPPORTED_OPTIX_ABIS,
    }
    result = {
        "provider_id": "optix",
        "external_component_probed": False,
        "discovery": "missing",
        "unavailable_reason": "bundled_provider_adapter_not_installed",
        "provider_abi": PROVIDER_ABI_NAME,
        "provider_version": None,
        "last_error": None,
        "failure_scope": None,
        "native_facts": native_facts,
    }
    try:
        candidates, runtime_path, provider_source = _provider_and_runtime_candidates(
            path
        )
    except (OSError, TypeError, ValueError) as exc:
        result.update(
            discovery="incompatible",
            unavailable_reason="library_path_resolution_failed",
            last_error=str(exc) or type(exc).__name__,
            failure_scope="provider",
        )
        return result
    native_facts.update(
        provider_source=provider_source,
        vendor_library_candidate=runtime_path or "system_default",
        provider_candidates=tuple(candidates),
    )
    if not candidates:
        return result
    result["external_component_probed"] = True
    failures = []
    for candidate in candidates:
        try:
            loaded = _query_provider(candidate)
            runtime_compatible = _probe_provider_runtime(loaded, runtime_path)
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            failures.append(f"{candidate}: {str(exc) or type(exc).__name__}")
            continue
        info = loaded.api.info
        result.update(
            discovery="present",
            unavailable_reason="execution_not_qualified",
            provider_version=_format_optix_version(info.optix_version),
        )
        native_facts.update(
            library_candidate=loaded.path,
            library_loaded_transiently=True,
            runtime_probe_only=True,
            context_created=False,
            vendor_runtime_abi_compatible=runtime_compatible,
            optix_abi_version=int(info.optix_abi_version),
            optix_version=int(info.optix_version),
            provider_name=_decode(info.provider_name),
            build_identity=_decode(info.build_identity),
            feature_bits=int(info.features),
        )
        if failures:
            native_facts["rejected_newer_candidates"] = tuple(failures)
        return result
    result.update(
        discovery="incompatible",
        unavailable_reason="no_compatible_optix_provider",
        last_error="; ".join(failures),
        failure_scope="provider",
    )
    native_facts["library_loaded_transiently"] = False
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


def _ray_storage(value, width, dtypes, name, *, access="readwrite"):
    description = describe_storage(value, access=access)
    descriptor = description.descriptor
    if descriptor is None or not description.supported:
        raise TaichiRuntimeError(f"OptiX ray {name} requires canonical dense storage")
    shape = tuple(descriptor.index_shape)
    element_shape = tuple(descriptor.element_shape)
    if descriptor.scalar_type not in dtypes:
        raise TaichiRuntimeError(f"OptiX ray {name} must use dtype {dtypes}")
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
    return description, count


_IDENTITY_TRANSFORM_3X4 = (
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
)


def _normalize_instance_transform(transform):
    try:
        values = tuple(transform)
    except TypeError as exc:
        raise TypeError("OptiX instance transform must be iterable") from exc
    if len(values) == 3 and all(hasattr(row, "__iter__") for row in values):
        values = tuple(value for row in values for value in row)
    if len(values) != 12:
        raise ValueError("OptiX instance transform must be a row-major 3x4 matrix")
    try:
        values = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise TypeError("OptiX instance transform must be numeric") from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError("OptiX instance transform must be finite")
    values = tuple(float(ctypes.c_float(value).value) for value in values)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("OptiX instance transform must be finite in f32")
    a, b, c, _, d, e, f, _, g, h, i, _ = values
    determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if not math.isfinite(determinant) or determinant == 0.0:
        raise ValueError("OptiX instance transform upper 3x3 must be invertible")
    return values


def _invoke_checked(api, function, *args):
    if int(function(*args)) != _SUCCESS:
        raise TaichiRuntimeError(_provider_error(api))


@dataclass(frozen=True)
class _PreparedOptixCall:
    owner: object
    storage: object
    call: object
    owners: tuple

    def __call__(self):
        self.owner._validate_lifetime()
        with hardware_failure_phase("provider_execution_failure"):
            self.owner._runtime_prog._invoke_external_cuda_prepared(
                self.storage, self.call
            )


def _prepare_storage(owner, values, descriptions, writable, textures=()):
    packet = owner._runtime_prog._prepare_external_cuda_storage(
        tuple(item.descriptor for item in descriptions),
        writable,
        tuple(texture.tex for texture in textures),
    )
    owners = (
        *values,
        *descriptions,
        *(value.arr for value in values if isinstance(value, Ndarray)),
        *textures,
    )
    return packet, owners


class OptixProvider:
    """Owner of one bundled OptiX adapter and CUDA context view."""

    def __init__(self, library_path=None, *, validation=False, provider_path=None):
        program = impl.get_runtime().prog
        if program is None or active_backend() != "cuda":
            raise TaichiRuntimeError(
                "OptixProvider requires an initialized Taichi CUDA runtime"
            )
        if not isinstance(validation, bool):
            raise TypeError("validation must be a bool")
        with hardware_failure_phase("provider_load_failure"):
            candidates, runtime_path, provider_source = (
                _provider_candidates_for_load(library_path, provider_path)
            )
            if not candidates:
                raise TaichiRuntimeError(
                    "Forge runtime wheel does not contain an OptiX provider adapter"
                )
            queried_candidates = []
            load_failures = []
            for candidate in candidates:
                try:
                    queried_candidates.append(_query_provider(candidate))
                except (
                    AttributeError,
                    OSError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ) as exc:
                    load_failures.append(
                        f"{candidate}: {str(exc) or type(exc).__name__}"
                    )
            if not queried_candidates:
                raise TaichiRuntimeError(
                    "no loadable OptiX provider adapter: " + "; ".join(load_failures)
                )
        failures = list(load_failures)
        loaded = None
        context = None
        attempted = []
        with hardware_failure_phase("provider_plan_failure"):
            for candidate_loaded in queried_candidates:
                attempted.append(candidate_loaded.path)
                candidate_context = ctypes.c_void_p()
                desc = _ContextDesc(
                    ctypes.sizeof(_ContextDesc),
                    0,
                    0,
                    int(validation),
                    0,
                    _runtime_library_argument(runtime_path),
                )
                result = int(
                    candidate_loaded.api.create_context(
                        ctypes.byref(desc), ctypes.byref(candidate_context)
                    )
                )
                if result == _SUCCESS and candidate_context.value:
                    loaded = candidate_loaded
                    context = candidate_context
                    break
                message = _provider_error(candidate_loaded.api)
                failures.append(f"{candidate_loaded.path}: {message}")
                if result != _OPTIX_UNAVAILABLE:
                    raise TaichiRuntimeError(message)
            if loaded is None or context is None:
                raise TaichiRuntimeError(
                    "no compatible OptiX provider adapter: " + "; ".join(failures)
                )
        self._loaded = loaded
        self._context = context
        self._runtime_prog = program
        self._runtime_generation = int(impl.runtime_generation())
        self._scenes = weakref.WeakSet()
        self._gases = weakref.WeakSet()
        self._typed_prepared = False
        self._alpha_prepared = False
        self._shared_pipeline_sbt_bytes = 0
        info = loaded.api.info
        self.identity = MappingProxyType(
            {
                "library_candidate": loaded.path,
                "vendor_library_candidate": runtime_path or "system_default",
                "provider_source": provider_source,
                "provider_candidates_attempted": tuple(attempted),
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

    def _graph_provider_memory_report(self):
        resident = not self.closed and runtime_generation_matches(self)
        return make_memory_report(
            "optix_context", "cuda",
            (
                HardwareMemoryComponent(
                    "shared_pipeline_sbt", self._shared_pipeline_sbt_bytes,
                    True, "runtime", "provider", resident=resident,
                ),
                HardwareMemoryComponent(
                    "optix_driver_context_state", None, False, "runtime", "driver",
                    resident=resident,
                ),
            ),
            lifecycle_state="closed" if self.closed else "ready" if resident else "runtime_invalid",
            ownership_scope="provider_context_shared_across_scenes_and_gases",
        )

    def triangle_scene(self, vertices, indices, *, allow_update=True):
        self._validate_lifetime()
        return OptixTriangleScene(self, vertices, indices, allow_update=allow_update)

    def triangle_gas(
        self, vertices, indices, *, allow_update=True, opacity_micromap=None
    ):
        """Create one independently retained triangle GAS for IAS reuse."""

        self._validate_lifetime()
        _require_instance_api(self._loaded.api)
        return OptixTriangleGAS(
            self,
            vertices,
            indices,
            allow_update=allow_update,
            opacity_micromap=opacity_micromap,
        )

    def instance_scene(self, instances, *, allow_update=True):
        """Create a fixed-topology IAS, retaining every referenced GAS."""

        self._validate_lifetime()
        _require_instance_api(self._loaded.api)
        return OptixInstanceScene(self, instances, allow_update=allow_update)

    def _prepare_typed(self, scene):
        self._validate_lifetime()
        if self._typed_prepared:
            return
        api = self._loaded.api
        if (
            not _api_has(api, "prepare_typed")
            or not _api_has(api, "trace_typed")
            or not int(api.info.features) & _TYPED_HITS
        ):
            raise TaichiRuntimeError(
                "OptiX adapter does not support typed hits; use a newer Forge adapter"
            )
        with hardware_failure_phase("provider_plan_failure"):
            # The existing submission gate also orders explicit scene/context
            # close against this cold adapter preparation; no work is replayed.
            scope = self._runtime_prog._begin_external_cuda_submission()
            try:
                scene._validate_lifetime()
                _invoke_checked(api, api.prepare_typed, self._context)
                memory = _SceneMemory()
                memory.struct_size = ctypes.sizeof(memory)
                _invoke_checked(
                    api,
                    scene._memory_function,
                    scene._scene,
                    ctypes.byref(memory),
                )
                self._shared_pipeline_sbt_bytes = int(memory.shared_pipeline_sbt_bytes)
            finally:
                del scope
        self._typed_prepared = True

    def _validate_lifetime(self):
        if self._context is None:
            raise TaichiRuntimeError("OptixProvider has been closed")
        validate_runtime_generation(
            self, "OptixProvider belongs to a previous Taichi runtime generation"
        )

    def _prepare_alpha(self, scene):
        self._validate_lifetime()
        if self._alpha_prepared:
            return
        api = self._loaded.api
        if not int(api.info.features) & _ALPHA_MASK or not all(
            _api_has(api, name)
            for name in ("prepare_alpha", "trace_alpha", "trace_instance_alpha")
        ):
            raise TaichiRuntimeError(
                "OptiX adapter does not support alpha masks; use a newer Forge adapter"
            )
        with hardware_failure_phase("provider_plan_failure"):
            scope = self._runtime_prog._begin_external_cuda_submission()
            try:
                scene._validate_lifetime()
                _invoke_checked(api, api.prepare_alpha, self._context)
                memory = _SceneMemory()
                memory.struct_size = ctypes.sizeof(memory)
                _invoke_checked(
                    api, scene._memory_function, scene._scene, ctypes.byref(memory)
                )
                self._shared_pipeline_sbt_bytes = int(memory.shared_pipeline_sbt_bytes)
            finally:
                del scope
        self._alpha_prepared = True

    def close(self):
        if self._context is None:
            return None
        live = tuple(scene for scene in self._scenes if not scene.closed)
        live_gases = tuple(gas for gas in self._gases if not gas.closed)
        if live or live_gases:
            raise TaichiRuntimeError(
                "OptixProvider cannot close while triangle scenes are live or "
                "shared acceleration structures are live"
            )
        context = self._context
        if runtime_generation_matches(self):
            self._runtime_prog.synchronize()
            result = int(self._loaded.api.destroy_context(context))
            if result != _SUCCESS:
                raise TaichiRuntimeError(_provider_error(self._loaded.api))
        self._context = None
        return None

    destroy = close

    def __enter__(self):
        self._validate_lifetime()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


@dataclass(frozen=True)
class OptixAlphaMask:
    """One instance's alpha rule using named UV and Texture Graph bindings.

    UVs are packed f32 pairs per GAS vertex. The texture is CUDA 2D r32f or
    rgba32f, sampled at normalized interpolated UVs with its existing sampler.
    A candidate is accepted when the selected channel is >= cutoff. No mip,
    arbitrary callback, blending or multi-layer transmission is implied.
    UV axes match Texture.sample_lod, including non-square uploads. UV values
    must be finite; GAS triangle indices must remain unchanged. Update UV or
    texel values in place using normal device ordering; replace resources by
    rebinding. None entries in the query's mask tuple accept unconditionally,
    but still participate in the filtered traversal (not the opaque fast path).
    """

    uvs: str
    texture: str
    cutoff: float = 0.5
    channel: int = 3

    def __post_init__(self):
        if any(
            not isinstance(name, str) or not name for name in (self.uvs, self.texture)
        ):
            raise ValueError("OptiX alpha bindings must be nonempty names")
        if self.uvs == self.texture:
            raise ValueError("OptiX UV and texture bindings must differ")
        if isinstance(self.cutoff, bool) or not isinstance(self.cutoff, (float, int)):
            raise TypeError("OptiX alpha cutoff must be a real number")
        if not math.isfinite(self.cutoff) or not 0 <= self.cutoff <= 1:
            raise ValueError("OptiX alpha cutoff must be finite and in [0, 1]")
        if isinstance(self.channel, bool) or not isinstance(self.channel, int):
            raise TypeError("OptiX alpha channel must be an integer")
        if not 0 <= self.channel <= 3:
            raise ValueError("OptiX alpha channel must be in [0, 3]")
        object.__setattr__(self, "cutoff", float(ctypes.c_float(self.cutoff).value))


@instrument_hardware_recording("ray.query.batch.optix")
class OptixRayQueryRecording(BackendCommandRecording):
    """One runtime-ordered OptiX launch against a fixed scene generation."""

    def __init__(
        self,
        scene,
        ray_count,
        *,
        rays="rays",
        hits="hits",
        hit_indices=None,
        alpha_masks=None,
        any_hit=False,
    ):
        if not isinstance(scene, (OptixTriangleScene, OptixInstanceScene)):
            raise TypeError("OptiX ray query recording requires an OptiX scene")
        if (
            isinstance(ray_count, bool)
            or not isinstance(ray_count, int)
            or not 1 <= ray_count <= 0xFFFFFFFF
        ):
            raise ValueError("OptiX ray count must be in [1, UINT32_MAX]")
        names = (rays, hits) if hit_indices is None else (rays, hits, hit_indices)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("OptiX ray bindings must be nonempty strings")
        if len(set(names)) != len(names):
            raise ValueError("OptiX ray bindings must be unique")
        if not isinstance(any_hit, bool):
            raise TypeError("OptiX any_hit must be a bool")
        micromap = isinstance(scene, OptixInstanceScene) and scene._has_micromaps
        if micromap:
            if hit_indices is None:
                raise ValueError("OptiX OMM scenes require typed queries")
            if alpha_masks is None:
                # OMM classification is intrinsic to this imported GAS. None
                # accepts unknown states; callers can instead supply alpha masks.
                alpha_masks = (None,) * scene.instance_count
        if alpha_masks is None:
            if any_hit:
                raise ValueError("OptiX any_hit requires an explicit alpha-mask query")
        else:
            if hit_indices is None:
                raise ValueError("OptiX alpha masks require typed hits")
            alpha_masks = tuple(alpha_masks)
            count = scene.instance_count if isinstance(scene, OptixInstanceScene) else 1
            if len(alpha_masks) != count:
                raise ValueError("OptiX alpha masks must match the instance count")
            mask_names = []
            for index, mask in enumerate(alpha_masks):
                if mask is not None:
                    if not isinstance(mask, OptixAlphaMask):
                        raise TypeError(
                            "OptiX alpha entries must be OptixAlphaMask or None"
                        )
                    if (
                        isinstance(scene, OptixInstanceScene)
                        and scene._instances[index].opaque
                    ):
                        raise ValueError(
                            "OptiX opaque instances cannot use an alpha mask"
                        )
                    mask_names.extend((mask.uvs, mask.texture))
            if set(names).intersection(mask_names):
                raise ValueError("OptiX alpha bindings must not reuse ray/hit names")
            names += tuple(dict.fromkeys(mask_names))
            if not mask_names and not any_hit and not micromap:
                # Exactly the same closest-hit semantics as the opaque route;
                # no mask pipeline/table/workspace is needed for this request.
                alpha_masks = None
        super().__init__(
            backend="cuda",
            binding_names=names,
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
        object.__setattr__(self, "hit_indices", hit_indices)
        object.__setattr__(self, "alpha_masks", alpha_masks)
        object.__setattr__(self, "any_hit", any_hit)
        object.__setattr__(self, "_micromap", micromap)
        identify_ray_recording(
            self, "trace_any" if any_hit else "trace_closest", scene._effect_name,
            ray_count=ray_count, hit_layout="legacy_float4" if hit_indices is None else "typed",
            alpha_masks=(
                tuple(None if mask is None else (mask.uvs, mask.texture, mask.cutoff, mask.channel)
                      for mask in alpha_masks)
                if alpha_masks is not None and any(mask is not None for mask in alpha_masks)
                else None
            ),
        )
        # OMM's distinct pipeline was prepared by the importing GAS adapter.
        if not micromap:
            if alpha_masks is not None:
                scene.provider._prepare_alpha(scene)
            elif hit_indices is not None:
                scene.provider._prepare_typed(scene)

    @property
    def resource_effects(self):
        effects = (
            ResourceEffect(self.rays, GraphAccess.READ),
            ResourceEffect(self.hits, GraphAccess.WRITE),
            static_resource_effect(self.scene._effect_name, GraphAccess.READ),
        )
        if self.hit_indices is not None:
            effects += (ResourceEffect(self.hit_indices, GraphAccess.WRITE),)
        if self.alpha_masks is not None:
            mask_names = dict.fromkeys(
                name
                for mask in self.alpha_masks
                if mask is not None
                for name in (mask.uvs, mask.texture)
            )
            effects += tuple(
                ResourceEffect(name, GraphAccess.READ) for name in mask_names
            )
        return effects

    def execute(self, bindings):
        return self.prepare_graph_execute(bindings)()

    def _binding_descriptions(self, bindings):
        specs = [(self.rays, (f32,), 8), (self.hits, (f32,), 4)]
        if self.hit_indices is not None:
            specs.append((self.hit_indices, (i32, u32), 4))
        descriptions = []
        for name, dtypes, width in specs:
            description, count = _ray_storage(bindings[name], width, dtypes, name)
            if count != self.ray_count:
                raise TaichiRuntimeError(
                    f"OptiX {name} binding has the wrong ray count"
                )
            descriptions.append(description)
        return descriptions

    def validate_graph_bindings(self, bindings):
        self._binding_descriptions(bindings)
        if self.alpha_masks is not None:
            self._mask_bindings(bindings)

    def _mask_bindings(self, bindings):
        from taichi_forge._lib import core

        values, descriptions, textures = [], [], []
        for index, mask in enumerate(self.alpha_masks):
            if mask is None:
                continue
            gas = (
                self.scene._instances[index].gas
                if isinstance(self.scene, OptixInstanceScene)
                else self.scene
            )
            uvs = bindings[mask.uvs]
            description, count = _ray_storage(uvs, 2, (f32,), mask.uvs, access="read")
            if count != gas.vertex_count:
                raise TaichiRuntimeError("OptiX alpha UV count must match GAS vertices")
            texture = bindings[mask.texture]
            if (
                not isinstance(texture, Texture)
                or texture.tex is None
                or texture._runtime_prog is not self.scene._runtime_prog
            ):
                raise TaichiRuntimeError(
                    "OptiX alpha texture must belong to this CUDA runtime"
                )
            if (
                texture.num_dims != 2
                or texture.mip_levels != 1
                or texture.fmt not in (core.Format.r32f, core.Format.rgba32f)
            ):
                raise TaichiRuntimeError(
                    "OptiX alpha requires a single-level 2D r32f/rgba32f texture"
                )
            if texture.fmt == core.Format.r32f and mask.channel != 0:
                raise ValueError("OptiX r32f alpha textures require channel=0")
            values.extend((uvs, gas._indices))
            descriptions.extend((description, gas._indices_description))
            textures.append(texture)
        return values, descriptions, textures

    def prepare_graph_execute(self, bindings):
        validate_exact_bindings(self, bindings, "OptiX ray query")
        self.validate_graph_lifetime()
        descriptions = self._binding_descriptions(bindings)
        if self.alpha_masks is not None:
            return self._prepare_alpha_execute(bindings, descriptions)
        values = tuple(bindings[name] for name in self.binding_names)
        storage, owners = _prepare_storage(
            self.scene, values, descriptions, (False, *([True] * (len(values) - 1)))
        )
        api = self.scene.provider._loaded.api
        if not int(api.info.features) & _WORD_ALIGNED_QUERY_STORAGE:
            if any(pointer % 16 for pointer in storage.pointers):
                raise TaichiRuntimeError(
                    "This OptiX adapter requires 16-byte aligned ray/hit storage; "
                    "use a newer Forge adapter for word-aligned dense views"
                )
        descriptor_type = _TraceDesc if self.hit_indices is None else _TypedTraceDesc
        function = self.scene._trace_function(self.hit_indices is not None)
        desc = descriptor_type(
            ctypes.sizeof(descriptor_type), self.ray_count, *storage.pointers, 0
        )
        return _PreparedOptixCall(
            self.scene,
            storage,
            partial(
                _invoke_checked, api, function, self.scene._scene, ctypes.byref(desc)
            ),
            owners,
        )

    def validate_graph_lifetime(self):
        self.scene._validate_lifetime()

    def _prepare_alpha_execute(self, bindings, descriptions):
        import numpy as np
        from taichi_forge.lang._ndarray import ScalarNdarray
        from taichi_forge.types.primitive_types import u64

        mask_values, mask_descriptions, textures = self._mask_bindings(bindings)
        # One retained allocation: 48 bytes of launch parameters, then one
        # 32-byte record per instance. Neither table nor pointers rebuild on run.
        workspace = ScalarNdarray(u64, (6 + 4 * len(self.alpha_masks),))
        workspace_description = describe_storage(workspace)
        values = (
            bindings[self.rays],
            bindings[self.hits],
            bindings[self.hit_indices],
            *mask_values,
            workspace,
        )
        descriptions = (*descriptions, *mask_descriptions, workspace_description)
        storage, owners = _prepare_storage(
            self.scene,
            values,
            descriptions,
            (False, True, True, *([False] * len(mask_values)), True),
            textures,
        )
        masks = (_AlphaMask * len(self.alpha_masks))()
        active = 0
        for index, mask in enumerate(self.alpha_masks):
            if mask is not None:
                masks[index] = _AlphaMask(
                    storage.pointers[3 + 2 * active],
                    storage.pointers[4 + 2 * active],
                    storage.texture_objects[active],
                    mask.cutoff,
                    mask.channel,
                )
                active += 1
        host = np.zeros(6 + 4 * len(self.alpha_masks), dtype=np.uint64)
        host[6:] = np.frombuffer(bytes(masks), dtype=np.uint64)
        workspace.from_numpy(host)
        api = self.scene.provider._loaded.api
        function = (
            api.trace_instance_micromap
            if self._micromap
            else (
                api.trace_instance_alpha
                if isinstance(self.scene, OptixInstanceScene)
                else api.trace_alpha
            )
        )
        desc = _AlphaTraceDesc(
            ctypes.sizeof(_AlphaTraceDesc),
            self.ray_count,
            *storage.pointers[:3],
            0,
            storage.pointers[-1] + 48,
            storage.pointers[-1],
            len(self.alpha_masks),
            int(self.any_hit),
        )
        return _PreparedOptixCall(
            self.scene,
            storage,
            partial(
                _invoke_checked, api, function, self.scene._scene, ctypes.byref(desc)
            ),
            owners,
        )

    def memory_report(self):
        report = self.scene.memory_report()
        if self.alpha_masks is None:
            return report
        return make_memory_report(
            report.provider,
            report.backend,
            (
                *report.components,
                HardwareMemoryComponent(
                    "alpha_workspace_per_prepared_binding",
                    48 + 32 * len(self.alpha_masks),
                    True,
                    "provider_generation",
                    "runtime",
                    resident=False,
                ),
            ),
            lifecycle_state=report.lifecycle_state,
            ownership_scope=report.ownership_scope,
        )

    def _as_graph_native_node(self):
        return native_recording_node(
            self,
            runtime_bindings=lambda item: tuple(
                (
                    name,
                    (
                        "texture"
                        if item.alpha_masks is not None
                        and any(
                            mask is not None and mask.texture == name
                            for mask in item.alpha_masks
                        )
                        else "ndarray"
                    ),
                )
                for name in item.binding_names
            ),
            lifetime_leases=lambda item: (item.scene, item.scene.provider),
            debug_info=lambda item: {
                "kind": item.scene._query_kind,
                "ray_count": item.ray_count,
                "provider_abi": PROVIDER_ABI_NAME,
                "hit_layout": "legacy_float4" if item.hit_indices is None else "typed",
                "alpha_masks": (
                    None
                    if item.alpha_masks is None
                    else tuple(
                        (
                            None
                            if mask is None
                            else (mask.uvs, mask.texture, mask.cutoff, mask.channel)
                        )
                        for mask in item.alpha_masks
                    )
                ),
                "any_accepted_hit": item.any_hit,
                "opaque_instances": (
                    tuple(instance.opaque for instance in item.scene._instances)
                    if isinstance(item.scene, OptixInstanceScene)
                    else ()
                ),
                "opacity_micromaps": (
                    tuple(
                        instance.gas._micromap_id for instance in item.scene._instances
                    )
                    if isinstance(item.scene, OptixInstanceScene)
                    else ()
                ),
            },
            publish_time_binding_validation_stable=True,
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
        identify_ray_recording(self, "scene_refit", scene._effect_name)

    @property
    def resource_effects(self):
        return (
            ResourceEffect(self.vertices, GraphAccess.READ),
            static_resource_effect(self.scene._effect_name, GraphAccess.WRITE),
        )

    def execute(self, bindings):
        return self.prepare_graph_execute(bindings)()

    def _binding_descriptions(self, bindings):
        description, count = _ray_storage(
            bindings[self.vertices], 3, (f32,), self.vertices
        )
        if count != self.scene.vertex_count:
            raise TaichiRuntimeError("OptiX refit must preserve the vertex count")
        return description, self.scene._indices_description

    def validate_graph_bindings(self, bindings):
        self._binding_descriptions(bindings)

    def prepare_graph_execute(self, bindings):
        validate_exact_bindings(self, bindings, "OptiX ray refit")
        self.validate_graph_lifetime()
        descriptions = self._binding_descriptions(bindings)
        values = (bindings[self.vertices], self.scene._indices)
        storage, owners = _prepare_storage(
            self.scene, values, descriptions, (False, False)
        )
        desc = _TriangleSceneDesc(
            ctypes.sizeof(_TriangleSceneDesc),
            self.scene.vertex_count,
            self.scene.triangle_count,
            1,
            *storage.pointers,
            0,
        )
        api = self.scene.provider._loaded.api
        return _PreparedOptixCall(
            self.scene,
            storage,
            partial(
                _invoke_checked,
                api,
                api.update_triangle_scene,
                self.scene._scene,
                ctypes.byref(desc),
            ),
            owners,
        )

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
            publish_time_binding_validation_stable=True,
        )


@instrument_hardware_recording("ray.as_refit.optix")
class OptixGASRefitRecording(BackendCommandRecording):
    """One fixed-topology update of an independently shared triangle GAS."""

    def __init__(self, gas, *, vertices="vertices"):
        if not isinstance(gas, OptixTriangleGAS):
            raise TypeError("OptiX GAS refit recording requires an OptixTriangleGAS")
        if not isinstance(vertices, str) or not vertices:
            raise ValueError("OptiX GAS refit binding must be a nonempty string")
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
        object.__setattr__(self, "gas", gas)
        object.__setattr__(self, "vertices", vertices)
        identify_ray_recording(self, "gas_refit", gas._effect_name)

    @property
    def resource_effects(self):
        return (
            ResourceEffect(self.vertices, GraphAccess.READ),
            static_resource_effect(self.gas._effect_name, GraphAccess.WRITE),
        )

    def _binding_descriptions(self, bindings):
        description, count = _ray_storage(
            bindings[self.vertices], 3, (f32,), self.vertices
        )
        if count != self.gas.vertex_count:
            raise TaichiRuntimeError("OptiX GAS refit must preserve the vertex count")
        return description, self.gas._indices_description

    def validate_graph_bindings(self, bindings):
        self._binding_descriptions(bindings)

    def prepare_graph_execute(self, bindings):
        validate_exact_bindings(self, bindings, "OptiX shared GAS refit")
        self.validate_graph_lifetime()
        descriptions = self._binding_descriptions(bindings)
        values = (bindings[self.vertices], self.gas._indices)
        storage, owners = _prepare_storage(
            self.gas, values, descriptions, (False, False)
        )
        desc = _TriangleSceneDesc(
            ctypes.sizeof(_TriangleSceneDesc),
            self.gas.vertex_count,
            self.gas.triangle_count,
            1,
            *storage.pointers,
            0,
        )
        api = self.gas.provider._loaded.api
        return _PreparedOptixCall(
            self.gas,
            storage,
            partial(
                _invoke_checked,
                api,
                api.update_triangle_gas,
                self.gas._gas,
                ctypes.byref(desc),
            ),
            owners,
        )

    def execute(self, bindings):
        return self.prepare_graph_execute(bindings)()

    def validate_graph_lifetime(self):
        self.gas._validate_lifetime()

    def memory_report(self):
        return self.gas.memory_report()

    def _as_graph_native_node(self):
        return native_recording_node(
            self,
            lifetime_leases=lambda item: (item.gas, item.gas.provider),
            debug_info=lambda item: {
                "kind": "optix_shared_triangle_gas_update",
                "vertex_count": item.gas.vertex_count,
                "topology_fixed": True,
            },
            publish_time_binding_validation_stable=True,
        )


def _instance_transform_storage(value, instance_count, name):
    description = describe_storage(value)
    if not description.supported or description.descriptor is None:
        raise TaichiRuntimeError(
            "OptiX instance transforms require describable dense storage: "
            f"{description.failure_reason}"
        )
    descriptor = description.descriptor
    if descriptor.scalar_type != f32:
        raise TaichiRuntimeError("OptiX instance transforms require dtype f32")
    shape = tuple(descriptor.index_shape)
    element = tuple(descriptor.element_shape)
    if not (
        (element == () and shape in ((instance_count, 3, 4), (instance_count, 12)))
        or (element == (3, 4) and shape == (instance_count,))
    ):
        raise TaichiRuntimeError(
            "OptiX instance transforms require scalar shape (N, 3, 4) or "
            "(N, 12), or AOS matrix-3x4 shape (N,), with N equal to the "
            "fixed instance count"
        )
    return description


@instrument_hardware_recording("ray.as_refit.optix")
class OptixInstanceRefitRecording(BackendCommandRecording):
    """Refit one fixed-topology IAS, optionally from device transforms."""

    def __init__(self, scene, *, transforms=None):
        if not isinstance(scene, OptixInstanceScene):
            raise TypeError(
                "OptiX instance refit recording requires an OptixInstanceScene"
            )
        if transforms is not None and (
            not isinstance(transforms, str) or not transforms
        ):
            raise ValueError(
                "OptiX instance transform binding must be a nonempty string"
            )
        super().__init__(
            backend="cuda",
            binding_names=() if transforms is None else (transforms,),
            command_count=1 if transforms is None else 2,
            queue="compute",
            stream_binding="runtime_ordered",
            barrier_policy="internal",
            workspace_ownership="provider_generation",
            replay_mode="rerecord",
            no_host_readback=True,
        )
        object.__setattr__(self, "scene", scene)
        object.__setattr__(self, "transforms", transforms)
        identify_ray_recording(
            self, "ias_refit" if transforms is None else "ias_device_refit", scene._effect_name
        )
        gas_effects = tuple(
            static_resource_effect(gas._effect_name, GraphAccess.READ)
            for gas in dict.fromkeys(scene._topology)
        )
        effects = gas_effects + (
            static_resource_effect(scene._effect_name, GraphAccess.WRITE),
        )
        if transforms is not None:
            effects = (ResourceEffect(transforms, GraphAccess.READ),) + effects
        object.__setattr__(self, "_effects", effects)

    @property
    def resource_effects(self):
        return self._effects

    def _binding_description(self, bindings):
        if self.transforms is None:
            return None
        return _instance_transform_storage(
            bindings[self.transforms], self.scene.instance_count, self.transforms
        )

    def validate_graph_bindings(self, bindings):
        self._binding_description(bindings)

    def prepare_graph_execute(self, bindings):
        validate_exact_bindings(self, bindings, "OptiX instance refit")
        self.validate_graph_lifetime()
        description = self._binding_description(bindings)
        if description is None:
            storage = self.scene._runtime_prog._prepare_external_cuda_storage((), ())
            owners = ()
            pointer = 0
        else:
            value = bindings[self.transforms]
            storage, owners = _prepare_storage(
                self.scene, (value,), (description,), (False,)
            )
            pointer = storage.pointers[0]
        desc = _InstanceUpdateDesc(
            ctypes.sizeof(_InstanceUpdateDesc),
            self.scene.instance_count,
            pointer,
            0,
        )
        api = self.scene.provider._loaded.api
        return _PreparedOptixCall(
            self.scene,
            storage,
            partial(
                _invoke_checked,
                api,
                api.update_instance_scene,
                self.scene._scene,
                ctypes.byref(desc),
            ),
            owners,
        )

    def execute(self, bindings):
        return self.prepare_graph_execute(bindings)()

    def validate_graph_lifetime(self):
        self.scene._validate_lifetime()

    def memory_report(self):
        return self.scene.memory_report()

    def _as_graph_native_node(self):
        return native_recording_node(
            self,
            lifetime_leases=lambda item: (item.scene, item.scene.provider),
            debug_info=lambda item: {
                "kind": (
                    "optix_instance_ias_device_transforms"
                    if item.transforms is not None
                    else "optix_instance_ias_bounds_update"
                ),
                "instance_count": item.scene.instance_count,
                "topology_fixed": True,
                "transform_layout": "row_major_f32_3x4",
                "instance_packing": (
                    "device" if item.transforms is not None else "retained"
                ),
            },
            publish_time_binding_validation_stable=True,
        )


class OptixTriangleGAS:
    """Independent fixed-topology triangle GAS retained by instance scenes."""

    def __init__(
        self, provider, vertices, indices, *, allow_update=True, opacity_micromap=None
    ):
        if not isinstance(provider, OptixProvider):
            raise TypeError("provider must be an OptixProvider")
        if not isinstance(allow_update, bool):
            raise TypeError("allow_update must be a bool")
        provider._validate_lifetime()
        _require_instance_api(provider._loaded.api)
        vertex_description, vertex_count = _ray_storage(vertices, 3, (f32,), "vertices")
        index_description, triangle_count = _ray_storage(indices, 3, (i32,), "indices")
        api = provider._loaded.api
        if opacity_micromap is not None:
            if not isinstance(opacity_micromap, OptixOpacityMicromap):
                raise TypeError("opacity_micromap must be an OptixOpacityMicromap")
            if opacity_micromap.triangle_count != triangle_count:
                raise ValueError(
                    "OMM triangle mapping must match the GAS triangle count"
                )
            if not int(api.info.features) & _OPACITY_MICROMAP_IMPORT or not all(
                _api_has(api, name)
                for name in ("create_triangle_gas_micromap", "trace_instance_micromap")
            ):
                raise TaichiRuntimeError(
                    "OptiX adapter does not support baked OMM import"
                )
        storage, owners = _prepare_storage(
            provider,
            (vertices, indices),
            (vertex_description, index_description),
            (False, False),
        )
        gas = ctypes.c_void_p()
        desc = _TriangleSceneDesc(
            ctypes.sizeof(_TriangleSceneDesc),
            vertex_count,
            triangle_count,
            int(allow_update),
            *storage.pointers,
            0,
        )
        micromap_memory = _MicromapMemory()
        micromap_memory.struct_size = ctypes.sizeof(micromap_memory)
        if opacity_micromap is None:
            create = partial(
                _invoke_checked,
                api,
                api.create_triangle_gas,
                provider._context,
                ctypes.byref(desc),
                ctypes.byref(gas),
            )
        else:
            imported, import_owners = opacity_micromap._native()
            create = partial(
                _invoke_checked,
                api,
                api.create_triangle_gas_micromap,
                provider._context,
                ctypes.byref(desc),
                ctypes.byref(imported),
                ctypes.byref(gas),
                ctypes.byref(micromap_memory),
            )
        with hardware_failure_phase("provider_plan_failure"):
            provider._runtime_prog._invoke_external_cuda_prepared(
                storage,
                create,
            )
            if not gas.value:
                raise TaichiRuntimeError("OptiX provider returned an empty GAS")
        self.provider = provider
        self._gas = gas
        self._runtime_prog = provider._runtime_prog
        self._runtime_generation = provider._runtime_generation
        self.vertex_count = vertex_count
        self.triangle_count = triangle_count
        self.allow_update = allow_update
        self._micromap_id = opacity_micromap.fingerprint if opacity_micromap else None
        self._micromap_memory = micromap_memory
        self._ray_retainers = weakref.WeakSet()
        self._indices = indices
        self._indices_description = index_description
        self._index_owner = indices.arr if isinstance(indices, Ndarray) else indices
        self._effect_name = RayResourceIdentity(
            "optix_triangle_gas", vertex_count=vertex_count, triangle_count=triangle_count,
            allow_update=allow_update, micromap_asset=self._micromap_id,
            adapter=_decode(api.info.build_identity),
        )
        memory = _SceneMemory()
        memory.struct_size = ctypes.sizeof(memory)
        result = int(api.get_triangle_gas_memory(gas, ctypes.byref(memory)))
        if result != _SUCCESS:
            api.destroy_triangle_gas(gas)
            self._gas = None
            raise TaichiRuntimeError(_provider_error(api))
        self._memory = memory
        provider._shared_pipeline_sbt_bytes = int(memory.shared_pipeline_sbt_bytes)
        provider._gases.add(self)

    @property
    def closed(self):
        return self._gas is None

    def record_refit(self, *, vertices="vertices"):
        self._validate_lifetime()
        if not self.allow_update:
            raise TaichiRuntimeError("OptiX shared GAS was not created for updates")
        return OptixGASRefitRecording(self, vertices=vertices)

    def refit(self, vertices):
        self.record_refit().execute({"vertices": vertices})
        return self

    def _validate_lifetime(self):
        if self._gas is None:
            raise TaichiRuntimeError("OptixTriangleGAS has been closed")
        self.provider._validate_lifetime()
        validate_runtime_generation(
            self,
            "OptixTriangleGAS belongs to a previous Taichi runtime generation",
        )

    def memory_report(self):
        """Include retained GAS storage even after its original handle closes."""
        return aggregate_ray_memory(self)

    def _local_memory_report(self):
        resident = ray_resource_resident(self)
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
                "gas_build_update_scratch",
                int(memory.build_update_scratch_bytes),
                True,
                "provider_generation",
                "provider",
                resident=resident,
            ),
        )
        return make_memory_report(
            "optix_shared_triangle_gas",
            "cuda",
            components
            + (
                ()
                if self._micromap_id is None
                else (
                    HardwareMemoryComponent(
                        "opacity_micromap_array_and_indices",
                        int(
                            self._micromap_memory.array_bytes
                            + self._micromap_memory.index_bytes
                        ),
                        True,
                        "provider_generation",
                        "provider",
                        resident=resident,
                    ),
                    HardwareMemoryComponent(
                        "opacity_micromap_import_temporary",
                        int(self._micromap_memory.build_temporary_bytes),
                        True,
                        "invocation",
                        "provider",
                        resident=False,
                        reusable=False,
                    ),
                )
            ),
            lifecycle_state="closed" if self.closed else "ready" if resident else "runtime_invalid",
            ownership_scope="gas_generation_including_retained_references",
        )

    def _graph_provider_memory_report(self):
        return self._local_memory_report()

    def _graph_provider_memory_dependencies(self):
        return (self.provider,) if ray_resource_resident(self) else ()

    def close(self):
        if self._gas is None:
            return None
        gas = self._gas
        if runtime_generation_matches(self):
            self._runtime_prog.synchronize()
            result = int(self.provider._loaded.api.destroy_triangle_gas(gas))
            if result != _SUCCESS:
                raise TaichiRuntimeError(_provider_error(self.provider._loaded.api))
        self._gas = None
        return None

    destroy = close

    def __enter__(self):
        self._validate_lifetime()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


@dataclass(frozen=True)
class OptixRayInstance:
    """One cold fixed-topology OptiX IAS instance descriptor.

    ``opaque=True`` promises that no query will alpha-filter this instance,
    allowing hardware traversal to bypass any-hit. The default keeps query-
    owned masks available. Opaque queries bypass any-hit in either case.
    """

    gas: OptixTriangleGAS
    transform: tuple = _IDENTITY_TRANSFORM_3X4
    mask: int = 0xFF
    custom_index: int = 0
    opaque: bool = False

    def __post_init__(self):
        if not isinstance(self.opaque, bool):
            raise TypeError("OptiX instance opaque must be a bool")
        if not isinstance(self.gas, OptixTriangleGAS):
            raise TypeError("OptiX ray instance gas must be an OptixTriangleGAS")
        if isinstance(self.mask, bool) or not isinstance(self.mask, int):
            raise TypeError("OptiX ray instance mask must be an integer")
        if not 0 <= self.mask <= 0xFF:
            raise ValueError("OptiX ray instance mask must be in [0, 255]")
        if isinstance(self.custom_index, bool) or not isinstance(
            self.custom_index, int
        ):
            raise TypeError("OptiX ray instance custom_index must be an integer")
        if not 0 <= self.custom_index <= 0xFFFFFF:
            raise ValueError("OptiX ray instance custom_index must be in [0, 16777215]")
        object.__setattr__(
            self, "transform", _normalize_instance_transform(self.transform)
        )


class OptixInstanceScene:
    """Fixed-topology multi-instance IAS over independently shared GASes."""

    def __init__(self, provider, instances, *, allow_update=True):
        if not isinstance(provider, OptixProvider):
            raise TypeError("provider must be an OptixProvider")
        if not isinstance(allow_update, bool):
            raise TypeError("allow_update must be a bool")
        provider._validate_lifetime()
        _require_instance_api(provider._loaded.api)
        try:
            normalized = tuple(instances)
        except TypeError as exc:
            raise TypeError("OptiX instance scene entries must be iterable") from exc
        if not normalized:
            raise ValueError("OptiX instance scene requires at least one instance")
        if len(normalized) > 0xFFFFFFFF:
            raise ValueError("OptiX instance count must fit in uint32")
        for instance in normalized:
            if not isinstance(instance, OptixRayInstance):
                raise TypeError(
                    "OptiX instance scene entries must be OptixRayInstance objects"
                )
            instance.gas._validate_lifetime()
            if instance.gas.provider is not provider:
                raise TaichiRuntimeError(
                    "OptiX instance scene and all GAS resources must share one provider"
                )
            if (
                instance.opaque
                and not int(provider._loaded.api.info.features) & _INSTANCE_OPACITY
            ):
                raise TaichiRuntimeError(
                    "OptiX adapter does not support instance opacity"
                )
            if instance.opaque and instance.gas._micromap_id is not None:
                raise ValueError("OMM instances cannot declare always-opaque geometry")
        native = (_InstanceDesc * len(normalized))()
        for index, instance in enumerate(normalized):
            native[index].struct_size = ctypes.sizeof(_InstanceDesc)
            native[index].reserved = int(instance.opaque)
            native[index].gas = instance.gas._gas
            native[index].transform[:] = instance.transform
            native[index].custom_index = instance.custom_index
            native[index].visibility_mask = instance.mask
        scene = ctypes.c_void_p()
        desc = _InstanceSceneDesc(
            ctypes.sizeof(_InstanceSceneDesc),
            len(normalized),
            int(allow_update),
            0,
            native,
            0,
        )
        api = provider._loaded.api
        storage = provider._runtime_prog._prepare_external_cuda_storage((), ())
        with hardware_failure_phase("provider_plan_failure"):
            provider._runtime_prog._invoke_external_cuda_prepared(
                storage,
                partial(
                    _invoke_checked,
                    api,
                    api.create_instance_scene,
                    provider._context,
                    ctypes.byref(desc),
                    ctypes.byref(scene),
                ),
            )
            if not scene.value:
                raise TaichiRuntimeError(
                    "OptiX provider returned an empty instance scene"
                )
        self.provider = provider
        self._scene = scene
        self._runtime_prog = provider._runtime_prog
        self._runtime_generation = provider._runtime_generation
        self._instances = normalized
        self._topology = tuple(instance.gas for instance in normalized)
        self._has_micromaps = any(gas._micromap_id is not None for gas in self._topology)
        self.allow_update = allow_update
        self._effect_name = RayResourceIdentity(
            "optix_instance_scene", children=tuple(gas._effect_name for gas in self._topology),
            opaque_instances=tuple(instance.opaque for instance in normalized),
            allow_update=allow_update, adapter=_decode(api.info.build_identity),
        )
        self._memory_function = api.get_instance_scene_memory
        self._query_kind = "optix_instance_ray_query"
        memory = _SceneMemory()
        memory.struct_size = ctypes.sizeof(memory)
        result = int(api.get_instance_scene_memory(scene, ctypes.byref(memory)))
        if result != _SUCCESS:
            api.destroy_instance_scene(scene)
            self._scene = None
            raise TaichiRuntimeError(_provider_error(api))
        self._memory = memory
        provider._shared_pipeline_sbt_bytes = int(memory.shared_pipeline_sbt_bytes)
        provider._scenes.add(self)
        for gas in dict.fromkeys(self._topology):
            gas._ray_retainers.add(self)

    @property
    def closed(self):
        return self._scene is None

    @property
    def instance_count(self):
        return len(self._topology)

    def _trace_function(self, typed):
        api = self.provider._loaded.api
        return api.trace_instance_scene_typed if typed else api.trace_instance_scene

    def record(self, ray_count, *, rays="rays", hits="hits"):
        self._validate_lifetime()
        return OptixRayQueryRecording(self, ray_count, rays=rays, hits=hits)

    def trace(self, rays, hits):
        ray_count = _ray_storage(rays, 8, (f32,), "rays")[1]
        self.record(ray_count).execute({"rays": rays, "hits": hits})
        return hits

    def record_typed(
        self,
        ray_count,
        *,
        rays="rays",
        hits="hits",
        hit_indices="hit_indices",
        alpha_masks=None,
        any_hit=False,
    ):
        """Record typed hits, optionally filtering each instance's alpha mask.

        alpha_masks is a fixed tuple of OptixAlphaMask or None per instance.
        any_hit returns the first accepted intersection, not necessarily nearest.
        """

        self._validate_lifetime()
        return OptixRayQueryRecording(
            self,
            ray_count,
            rays=rays,
            hits=hits,
            hit_indices=hit_indices,
            alpha_masks=alpha_masks,
            any_hit=any_hit,
        )

    def trace_typed(self, rays, hits, hit_indices):
        ray_count = _ray_storage(rays, 8, (f32,), "rays")[1]
        self.record_typed(ray_count).execute(
            {"rays": rays, "hits": hits, "hit_indices": hit_indices}
        )
        return hits, hit_indices

    def record_refit(self):
        """Record a bounds-only IAS update after one or more GAS updates."""

        self._validate_lifetime()
        if not self.allow_update:
            raise TaichiRuntimeError("OptiX instance scene was not created for updates")
        return OptixInstanceRefitRecording(self)

    def refit(self):
        self.record_refit().execute({})
        return self

    def record_refit_transforms(self, *, transforms="transforms"):
        """Record device packing followed by a fixed-topology IAS update.

        The binding accepts compact f32 scalar ``(N, 3, 4)`` / ``(N, 12)``
        storage or AOS matrix-3x4 ``(N,)`` storage. Producers must supply
        finite matrices with an invertible upper 3x3; Forge does not read back
        or scan per-frame values. GAS references, instance order, visibility
        masks and custom indices remain fixed in retained native descriptors.
        """

        self._validate_lifetime()
        if not self.allow_update:
            raise TaichiRuntimeError("OptiX instance scene was not created for updates")
        return OptixInstanceRefitRecording(self, transforms=transforms)

    def refit_transforms(self, transforms):
        self.record_refit_transforms().execute({"transforms": transforms})
        return self

    def _validate_lifetime(self):
        if self._scene is None:
            raise TaichiRuntimeError("OptixInstanceScene has been closed")
        self.provider._validate_lifetime()
        validate_runtime_generation(
            self,
            "OptixInstanceScene belongs to a previous Taichi runtime generation",
        )

    def memory_report(self):
        """Include unique GAS/context dependencies, not only IAS allocations."""
        return aggregate_ray_memory(self)

    def _local_memory_report(self):
        resident = self._scene is not None and runtime_generation_matches(self)
        memory = self._memory
        components = (
            HardwareMemoryComponent(
                "multi_instance_ias_storage",
                int(memory.ias_bytes),
                True,
                "provider_generation",
                "provider",
                resident=resident,
            ),
            HardwareMemoryComponent(
                "ias_build_update_scratch",
                int(memory.build_update_scratch_bytes),
                True,
                "provider_generation",
                "provider",
                resident=resident,
            ),
            HardwareMemoryComponent(
                "retained_instance_and_launch_params",
                int(memory.instance_bytes + memory.launch_params_bytes),
                True,
                "provider_generation",
                "provider",
                resident=resident,
            ),
        )
        return make_memory_report(
            "optix_instance_ray",
            "cuda",
            components,
            lifecycle_state="ready" if resident else "closed",
            ownership_scope="provider_context_and_instance_scene_generation",
        )

    def _graph_provider_memory_report(self):
        return self._local_memory_report()

    def _graph_provider_memory_dependencies(self):
        if not ray_resource_resident(self):
            return ()
        return (*dict.fromkeys(self._topology), self.provider)

    def close(self):
        if self._scene is None:
            return None
        scene = self._scene
        if runtime_generation_matches(self):
            self._runtime_prog.synchronize()
            result = int(self.provider._loaded.api.destroy_instance_scene(scene))
            if result != _SUCCESS:
                raise TaichiRuntimeError(_provider_error(self.provider._loaded.api))
        self._scene = None
        return None

    destroy = close

    def __enter__(self):
        self._validate_lifetime()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


class OptixTriangleScene:
    """Fixed-topology triangle GAS plus identity IAS owned by a provider."""

    def __init__(self, provider, vertices, indices, *, allow_update=True):
        if not isinstance(provider, OptixProvider):
            raise TypeError("provider must be an OptixProvider")
        if not isinstance(allow_update, bool):
            raise TypeError("allow_update must be a bool")
        provider._validate_lifetime()
        vertex_description, vertex_count = _ray_storage(vertices, 3, (f32,), "vertices")
        index_description, triangle_count = _ray_storage(indices, 3, (i32,), "indices")
        storage, owners = _prepare_storage(
            provider,
            (vertices, indices),
            (vertex_description, index_description),
            (False, False),
        )
        scene = ctypes.c_void_p()
        desc = _TriangleSceneDesc(
            ctypes.sizeof(_TriangleSceneDesc),
            vertex_count,
            triangle_count,
            int(allow_update),
            *storage.pointers,
            0,
        )
        api = provider._loaded.api
        with hardware_failure_phase("provider_plan_failure"):
            provider._runtime_prog._invoke_external_cuda_prepared(
                storage,
                partial(
                    _invoke_checked,
                    api,
                    api.create_triangle_scene,
                    provider._context,
                    ctypes.byref(desc),
                    ctypes.byref(scene),
                ),
            )
            if not scene.value:
                raise TaichiRuntimeError("OptiX provider returned an empty scene")
        self.provider = provider
        self._scene = scene
        self._runtime_prog = provider._runtime_prog
        self._runtime_generation = provider._runtime_generation
        self.vertex_count = vertex_count
        self.triangle_count = triangle_count
        self.allow_update = allow_update
        self._indices = indices
        self._indices_description = index_description
        self._index_owner = indices.arr if isinstance(indices, Ndarray) else indices
        self._memory_function = api.get_scene_memory
        self._query_kind = "optix_triangle_ray_query"
        self._effect_name = RayResourceIdentity(
            "optix_triangle_scene", vertex_count=vertex_count, triangle_count=triangle_count,
            allow_update=allow_update, adapter=_decode(provider._loaded.api.info.build_identity),
        )
        memory = _SceneMemory()
        memory.struct_size = ctypes.sizeof(_SceneMemory)
        result = int(provider._loaded.api.get_scene_memory(scene, ctypes.byref(memory)))
        if result != _SUCCESS:
            provider._loaded.api.destroy_triangle_scene(scene)
            self._scene = None
            raise TaichiRuntimeError(_provider_error(provider._loaded.api))
        self._memory = memory
        provider._shared_pipeline_sbt_bytes = int(memory.shared_pipeline_sbt_bytes)
        provider._scenes.add(self)

    def _trace_function(self, typed):
        api = self.provider._loaded.api
        return api.trace_typed if typed else api.trace

    @property
    def closed(self):
        return self._scene is None

    def record(self, ray_count, *, rays="rays", hits="hits"):
        self._validate_lifetime()
        return OptixRayQueryRecording(self, ray_count, rays=rays, hits=hits)

    def trace(self, rays, hits):
        ray_count = _ray_storage(rays, 8, (f32,), "rays")[1]
        self.record(ray_count).execute({"rays": rays, "hits": hits})
        return hits

    def record_typed(
        self,
        ray_count,
        *,
        rays="rays",
        hits="hits",
        hit_indices="hit_indices",
        alpha_masks=None,
        any_hit=False,
    ):
        """Record f32 (t,u,v,0) and i32/u32 (primitive,instance,custom,hit).

        Misses write (-1,0,0,0) and (-1,-1,-1,0), with UINT32_MAX for u32.
        This single-instance scene has instance ordinal and custom ID zero.
        t is the ray parameter (distance only for unit directions); triangle
        weights are (1-u-v,u,v). Scalar (N,4) and AOS vector-4 layouts are accepted.
        """
        self._validate_lifetime()
        return OptixRayQueryRecording(
            self,
            ray_count,
            rays=rays,
            hits=hits,
            hit_indices=hit_indices,
            alpha_masks=alpha_masks,
            any_hit=any_hit,
        )

    def trace_typed(self, rays, hits, hit_indices):
        self.record_typed(_ray_storage(rays, 8, (f32,), "rays")[1]).execute(
            {"rays": rays, "hits": hits, "hit_indices": hit_indices}
        )
        return hits, hit_indices

    def record_refit(self, *, vertices="vertices"):
        self._validate_lifetime()
        if not self.allow_update:
            raise TaichiRuntimeError("OptiX scene was not created for updates")
        return OptixRayRefitRecording(self, vertices=vertices)

    def refit(self, vertices):
        self.record_refit().execute({"vertices": vertices})
        return self

    def _validate_lifetime(self):
        if self._scene is None:
            raise TaichiRuntimeError("OptixTriangleScene has been closed")
        self.provider._validate_lifetime()
        validate_runtime_generation(
            self,
            "OptixTriangleScene belongs to a previous Taichi runtime generation",
        )

    def memory_report(self):
        return aggregate_ray_memory(self)

    def _local_memory_report(self):
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
        )
        return make_memory_report(
            "optix_triangle_ray",
            "cuda",
            components,
            lifecycle_state="ready" if resident else "closed",
            ownership_scope="provider_context_and_scene_generation",
        )

    def _graph_provider_memory_report(self):
        return self._local_memory_report()

    def _graph_provider_memory_dependencies(self):
        return (self.provider,) if ray_resource_resident(self) else ()

    def close(self):
        if self._scene is None:
            return None
        scene = self._scene
        if runtime_generation_matches(self):
            self._runtime_prog.synchronize()
            result = int(self.provider._loaded.api.destroy_triangle_scene(scene))
            if result != _SUCCESS:
                raise TaichiRuntimeError(_provider_error(self.provider._loaded.api))
        self._scene = None
        return None

    destroy = close

    def __enter__(self):
        self._validate_lifetime()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def load_provider(library_path=None, *, validation=False, provider_path=None):
    """Load OptiX with an optional explicit Forge adapter path.

    ``library_path`` remains the user-provided vendor runtime. ``provider_path``
    is for an explicit local Forge C-ABI adapter and is never treated as a
    vendor runtime candidate or installed into the runtime wheel.
    """

    return OptixProvider(
        library_path, validation=validation, provider_path=provider_path
    )


def is_loaded():
    return bool(passive_status()["library_loaded"])


__all__ = [
    "OptixOpacityMicromap",
    "OptixGASRefitRecording",
    "OptixInstanceRefitRecording",
    "OptixInstanceScene",
    "OptixProvider",
    "OptixRayInstance",
    "OptixRayQueryRecording",
    "OptixRayRefitRecording",
    "OptixTriangleGAS",
    "OptixTriangleScene",
    "PROVIDER_ABI_NAME",
    "PROVIDER_ABI_VERSION",
    "SUPPORTED_OPTIX_ABIS",
    "is_loaded",
    "load_provider",
    "passive_status",
    "probe_provider",
]
