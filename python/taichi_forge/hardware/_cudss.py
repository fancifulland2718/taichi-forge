"""Bundled Forge adapter for the user-managed optional cuDSS runtime."""

from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import weakref


PROVIDER_ABI_VERSION = 1
PROVIDER_ABI_NAME = "taichi-forge-cudss-provider-c-abi1"
PROVIDER_QUERY_SYMBOL = "taichi_forge_cudss_provider_query"
SUPPORTED_CUDSS_HEADER_VERSIONS = (800,)

_SUCCESS = 0
_RUNTIME_UNAVAILABLE = 3
_REQUIRED_FEATURES = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)
_loaded_plans = weakref.WeakSet()


def _provider_filename(cudss_header_version):
    stem = "taichi_forge_cudss_provider_abi1_" f"cudss{cudss_header_version // 10:03d}"
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
        for version in reversed(SUPPORTED_CUDSS_HEADER_VERSIONS):
            candidate = directory / _provider_filename(version)
            key = os.path.normcase(str(candidate))
            if key not in seen and candidate.is_file():
                candidates.append(str(candidate))
                seen.add(key)
    return tuple(candidates)


class _ProviderInfo(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("provider_abi_version", ctypes.c_uint32),
        ("cudss_header_version", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("features", ctypes.c_uint64),
        ("provider_name", ctypes.c_char_p),
        ("build_identity", ctypes.c_char_p),
    ]


class _RuntimeInfo(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version_major", ctypes.c_uint32),
        ("version_minor", ctypes.c_uint32),
        ("version_patch", ctypes.c_uint32),
        ("library_path", ctypes.c_char_p),
    ]


_ProbeRuntime = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_char_p, ctypes.POINTER(_RuntimeInfo)
)
_CreateRuntime = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(_RuntimeInfo),
)
_DestroyRuntime = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
_GetLastError = ctypes.CFUNCTYPE(
    ctypes.c_size_t, ctypes.POINTER(ctypes.c_char), ctypes.c_size_t
)


class _ProviderApi(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("provider_abi_version", ctypes.c_uint32),
        ("info", _ProviderInfo),
        ("probe_runtime", _ProbeRuntime),
        ("create_runtime", _CreateRuntime),
        ("destroy_runtime", _DestroyRuntime),
        ("create", ctypes.c_void_p),
        ("destroy", ctypes.c_void_p),
        ("set_stream", ctypes.c_void_p),
        ("config_create", ctypes.c_void_p),
        ("config_destroy", ctypes.c_void_p),
        ("data_create", ctypes.c_void_p),
        ("data_destroy", ctypes.c_void_p),
        ("matrix_create_csr", ctypes.c_void_p),
        ("matrix_create_dn", ctypes.c_void_p),
        ("matrix_destroy", ctypes.c_void_p),
        ("matrix_set_values", ctypes.c_void_p),
        ("matrix_set_csr_pointers", ctypes.c_void_p),
        ("execute", ctypes.c_void_p),
        ("get_last_error", _GetLastError),
    ]


@dataclass(frozen=True)
class _LoadedApi:
    library: object
    path: str
    api: _ProviderApi


class _ProviderRuntimeError(RuntimeError):
    def __init__(self, result, message):
        super().__init__(message)
        self.result = int(result)


def _decode(value):
    return "" if not value else value.decode("utf-8", errors="replace")


def _provider_error(api):
    required = int(api.get_last_error(None, 0))
    if required <= 1:
        return "optional cuDSS adapter call failed"
    buffer = ctypes.create_string_buffer(required)
    api.get_last_error(buffer, required)
    return buffer.value.decode("utf-8", errors="replace")


def _check_api(api):
    if api.struct_size < ctypes.sizeof(_ProviderApi):
        raise RuntimeError("cuDSS provider returned a truncated Forge API table")
    if api.provider_abi_version != PROVIDER_ABI_VERSION:
        raise RuntimeError("cuDSS provider returned a mismatched Forge ABI")
    if api.info.struct_size < ctypes.sizeof(_ProviderInfo):
        raise RuntimeError("cuDSS provider returned truncated identity facts")
    if api.info.provider_abi_version != PROVIDER_ABI_VERSION:
        raise RuntimeError("cuDSS provider identity uses a mismatched Forge ABI")
    if api.info.cudss_header_version not in SUPPORTED_CUDSS_HEADER_VERSIONS:
        raise RuntimeError("cuDSS provider header ABI is outside the bundled range")
    if int(api.info.features) & _REQUIRED_FEATURES != _REQUIRED_FEATURES:
        raise RuntimeError("cuDSS provider does not implement the solver ABI")
    for name in (
        "probe_runtime",
        "create_runtime",
        "destroy_runtime",
        "get_last_error",
    ):
        if not bool(getattr(api, name)):
            raise RuntimeError(f"cuDSS provider is missing ABI entry {name}")
    for name in (
        "create",
        "destroy",
        "set_stream",
        "config_create",
        "config_destroy",
        "data_create",
        "data_destroy",
        "matrix_create_csr",
        "matrix_create_dn",
        "matrix_destroy",
        "matrix_set_values",
        "matrix_set_csr_pointers",
        "execute",
    ):
        if not getattr(api, name):
            raise RuntimeError(f"cuDSS provider is missing ABI entry {name}")


def _load_library(path):
    return ctypes.CDLL(path)


def _query_provider(path):
    resolved = str(Path(path).expanduser().resolve())
    library = _load_library(resolved)
    try:
        query = getattr(library, PROVIDER_QUERY_SYMBOL)
    except AttributeError as exc:
        raise RuntimeError("cuDSS provider query symbol is missing") from exc
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


def _library_names():
    if sys.platform == "win32":
        return ("cudss64_0.dll",)
    return ("libcudss.so.0", "libcudss.so")


def _path_candidate(value):
    if value is None:
        return None
    path = Path(os.fspath(value)).expanduser()
    if path.is_file():
        return path.resolve()
    if path.is_dir():
        for name in _library_names():
            direct = path / name
            if direct.is_file():
                return direct.resolve()
        for relative in ("bin", "lib"):
            for name in _library_names():
                nested = path / relative / name
                if nested.is_file():
                    return nested.resolve()
    return None


def _nvidia_namespace_roots():
    spec = importlib.util.find_spec("nvidia")
    if spec is None or spec.submodule_search_locations is None:
        return ()
    return tuple(Path(item) for item in spec.submodule_search_locations)


def _cuda_driver_api_version():
    try:
        from taichi_forge._lib import core as _ti_core  # pylint: disable=C0415

        return _ti_core.cuda_driver_api_version()
    except (AttributeError, ImportError, RuntimeError):
        return None


def _relative_roots(cuda_driver_api_version):
    if cuda_driver_api_version is not None:
        major = int(cuda_driver_api_version) // 1000
        if major >= 13:
            return (
                ("cu13", "bin"),
                ("cu13", "lib"),
                ("cudss", "bin"),
                ("cudss", "lib"),
                ("cu12", "bin"),
                ("cu12", "lib"),
            )
        if major == 12:
            return (
                ("cudss", "bin"),
                ("cudss", "lib"),
                ("cu12", "bin"),
                ("cu12", "lib"),
            )
        return ()
    return (
        ("cu13", "bin"),
        ("cu13", "lib"),
        ("cudss", "bin"),
        ("cudss", "lib"),
        ("cu12", "bin"),
        ("cu12", "lib"),
    )


def resolve_cudss_library_path(library_path=None, *, cuda_driver_api_version=None):
    """Return an installed cuDSS shared library without installing anything."""

    explicit = library_path
    if explicit is None:
        explicit = os.environ.get("TI_CUDSS_LIBRARY_PATH")
    if explicit:
        candidate = _path_candidate(explicit)
        if candidate is not None:
            return str(candidate)
        # Preserve an explicit file name so the native loader can report a
        # normal provider-not-found result rather than silently changing it.
        return os.fspath(explicit)

    if cuda_driver_api_version is None:
        cuda_driver_api_version = _cuda_driver_api_version()
    relative_roots = _relative_roots(cuda_driver_api_version)
    for root in _nvidia_namespace_roots():
        for relative in relative_roots:
            candidate = _path_candidate(root.joinpath(*relative))
            if candidate is not None:
                return str(candidate)
    return ""


@lru_cache(maxsize=16)
def _resolved_library_sha256(candidate, size, mtime_ns):
    del size, mtime_ns
    digest = hashlib.sha256()
    with Path(candidate).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cudss_library_sha256(library_path):
    """Return the content identity of one resolved user-managed provider."""

    candidate = _path_candidate(library_path)
    if candidate is None:
        return None
    stat = candidate.stat()
    return _resolved_library_sha256(
        str(candidate), int(stat.st_size), int(stat.st_mtime_ns)
    )


def cudss_adapter_sha256(adapter_path=None):
    """Return the identity of the selected wheel-internal Forge adapter."""

    if adapter_path is None:
        candidates = _bundled_provider_candidates()
        if len(candidates) != 1:
            return None
        adapter_path = candidates[0]
    candidate = _path_candidate(adapter_path)
    if candidate is None:
        return None
    stat = candidate.stat()
    return _resolved_library_sha256(
        str(candidate), int(stat.st_size), int(stat.st_mtime_ns)
    )


@contextmanager
def cudss_dll_directories(library_path):
    """Keep NVIDIA wheel DLL directories active for one native load call."""

    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        yield
        return
    directories = []
    candidate = _path_candidate(library_path)
    if candidate is not None:
        directories.append(candidate.parent)
    for root in _nvidia_namespace_roots():
        for relative in (
            ("cu13", "bin"),
            ("cu12", "bin"),
            ("cudss", "bin"),
            ("cublas", "bin"),
            ("cuda_runtime", "bin"),
        ):
            directory = root.joinpath(*relative)
            if directory.is_dir():
                directories.append(directory.resolve())
    handles = []
    try:
        for directory in dict.fromkeys(directories):
            handles.append(os.add_dll_directory(str(directory)))
        yield
    finally:
        for handle in reversed(handles):
            handle.close()


def _runtime_library_argument(runtime_path):
    return None if not runtime_path else os.fsencode(runtime_path)


def _format_cudss_version(major, minor, patch):
    return f"{int(major)}.{int(minor)}.{int(patch)}"


def _probe_provider_runtime(loaded, runtime_path):
    info = _RuntimeInfo()
    info.struct_size = ctypes.sizeof(_RuntimeInfo)
    result = int(
        loaded.api.probe_runtime(
            _runtime_library_argument(runtime_path), ctypes.byref(info)
        )
    )
    if result != _SUCCESS:
        raise _ProviderRuntimeError(result, _provider_error(loaded.api))
    return {
        "version_major": int(info.version_major),
        "version_minor": int(info.version_minor),
        "version_patch": int(info.version_patch),
        "library_path": _decode(info.library_path) or runtime_path or "system_default",
    }


@dataclass(frozen=True)
class ResolvedCudssProvider:
    adapter_path: str
    adapter_binary_sha256: str
    runtime_library_path: str
    provider_version: str
    provider_header_version: int
    provider_name: str
    build_identity: str
    feature_bits: int


def resolve_cudss_provider(library_path=None, *, cuda_driver_api_version=None):
    """Resolve one bundled adapter and compatible user-managed cuDSS runtime."""

    runtime_path = resolve_cudss_library_path(
        library_path, cuda_driver_api_version=cuda_driver_api_version
    )
    candidates = _bundled_provider_candidates()
    if not candidates:
        raise RuntimeError(
            "Forge runtime wheel does not contain a cuDSS provider adapter"
        )
    failures = []
    with cudss_dll_directories(runtime_path):
        for candidate in candidates:
            try:
                loaded = _query_provider(candidate)
                runtime = _probe_provider_runtime(loaded, runtime_path)
                adapter_binary_sha256 = cudss_adapter_sha256(loaded.path)
                if adapter_binary_sha256 is None:
                    raise RuntimeError("cuDSS adapter identity is unavailable")
            except (
                AttributeError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                failures.append(f"{candidate}: {str(exc) or type(exc).__name__}")
                continue
            info = loaded.api.info
            return ResolvedCudssProvider(
                adapter_path=loaded.path,
                adapter_binary_sha256=adapter_binary_sha256,
                runtime_library_path=(
                    runtime_path
                    if runtime_path
                    else (
                        ""
                        if runtime["library_path"] == "system_default"
                        else runtime["library_path"]
                    )
                ),
                provider_version=_format_cudss_version(
                    runtime["version_major"],
                    runtime["version_minor"],
                    runtime["version_patch"],
                ),
                provider_header_version=int(info.cudss_header_version),
                provider_name=_decode(info.provider_name),
                build_identity=_decode(info.build_identity),
                feature_bits=int(info.features),
            )
    raise RuntimeError("no compatible cuDSS provider adapter: " + "; ".join(failures))


def probe_provider(path=None):
    """Probe the bundled adapter and vendor runtime without retaining either."""

    runtime_path = None
    candidates = ()
    native_facts = {
        "probe_policy": "transient_adapter_and_vendor_runtime_query",
        "provider_enablement_changed": False,
        "provider_selection_changed": False,
        "execution_qualified": False,
        "supported_cudss_header_versions": SUPPORTED_CUDSS_HEADER_VERSIONS,
    }
    result = {
        "provider_id": "cudss",
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
        runtime_path = resolve_cudss_library_path(path)
        candidates = _bundled_provider_candidates()
    except (OSError, TypeError, ValueError) as exc:
        result.update(
            discovery="incompatible",
            unavailable_reason="library_path_resolution_failed",
            last_error=str(exc) or type(exc).__name__,
            failure_scope="provider",
        )
        return result
    native_facts.update(
        provider_source="forge_runtime_wheel",
        vendor_library_candidate=runtime_path or "system_default",
        provider_candidates=tuple(candidates),
    )
    if not candidates:
        return result
    result["external_component_probed"] = True
    failures = []
    failure_results = []
    with cudss_dll_directories(runtime_path):
        for candidate in candidates:
            try:
                loaded = _query_provider(candidate)
                runtime = _probe_provider_runtime(loaded, runtime_path)
                adapter_binary_sha256 = cudss_adapter_sha256(loaded.path)
                if adapter_binary_sha256 is None:
                    raise RuntimeError("cuDSS adapter identity is unavailable")
            except (
                AttributeError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                failures.append(f"{candidate}: {str(exc) or type(exc).__name__}")
                failure_results.append(getattr(exc, "result", None))
                continue
            info = loaded.api.info
            version = _format_cudss_version(
                runtime["version_major"],
                runtime["version_minor"],
                runtime["version_patch"],
            )
            result.update(
                discovery="available",
                unavailable_reason="none",
                provider_version=version,
            )
            native_facts.update(
                library_candidate=loaded.path,
                provider_adapter_binary_sha256=adapter_binary_sha256,
                library_loaded_transiently=True,
                runtime_probe_only=True,
                plan_created=False,
                vendor_runtime_abi_compatible=True,
                vendor_library_resolved=runtime["library_path"],
                cudss_header_version=int(info.cudss_header_version),
                provider_name=_decode(info.provider_name),
                build_identity=_decode(info.build_identity),
                feature_bits=int(info.features),
            )
            if failures:
                native_facts["rejected_candidates"] = tuple(failures)
            return result
    runtime_missing = bool(failure_results) and all(
        code == _RUNTIME_UNAVAILABLE for code in failure_results
    )
    result.update(
        discovery="missing" if runtime_missing else "incompatible",
        unavailable_reason=(
            "external_library_not_found"
            if runtime_missing
            else "no_compatible_cudss_provider"
        ),
        last_error="; ".join(failures),
        failure_scope="provider",
    )
    if runtime_missing:
        native_facts["library_candidates"] = [runtime_path or "system_default"]
    native_facts["library_loaded_transiently"] = False
    return result


def _register_loaded_plan(plan):
    _loaded_plans.add(plan)


def passive_status():
    loaded = tuple(plan for plan in _loaded_plans if not plan.closed)
    native_facts = {
        "status_policy": "passive_loaded_cudss_plans",
        "external_component_probed": False,
        "provider_enablement_changed": False,
        "provider_selection_changed": False,
        "loaded_plan_count": len(loaded),
    }
    if not loaded:
        return {
            "provider_id": "cudss",
            "library_loaded": False,
            "provider_abi": PROVIDER_ABI_NAME,
            "provider_version": None,
            "native_facts": native_facts,
        }
    plan = loaded[0]
    native_facts.update(plan.provider_identity)
    return {
        "provider_id": "cudss",
        "library_loaded": True,
        "provider_abi": PROVIDER_ABI_NAME,
        "provider_version": plan.provider_identity["provider_version"],
        "native_facts": native_facts,
    }


__all__ = (
    "PROVIDER_ABI_NAME",
    "PROVIDER_ABI_VERSION",
    "ResolvedCudssProvider",
    "SUPPORTED_CUDSS_HEADER_VERSIONS",
    "cudss_dll_directories",
    "cudss_adapter_sha256",
    "cudss_library_sha256",
    "passive_status",
    "probe_provider",
    "resolve_cudss_library_path",
    "resolve_cudss_provider",
)
