"""Common cuSOLVER adapter ABI and cold CUDA storage/context binding."""

from contextlib import contextmanager
import ctypes as ct
import os

from taichi_forge.hardware._bundled_runtime_provider import (
    BundledRuntimeProviderDefinition,
    open_runtime as _open_runtime,
    passive_status as _passive_status,
    probe_provider as _probe_provider,
    resolve_library_path as _resolve_library_path,
)
from taichi_forge.lang.exception import TaichiRuntimeError

ABI = "taichi-forge-cusolverdn-provider-c-abi2"
DEFINITION = BundledRuntimeProviderDefinition(
    provider_id="cusolverdn",
    provider_name="NVIDIA cuSOLVERDn",
    adapter_stem="taichi_forge_cusolverdn_provider_abi2_generic_v1",
    query_symbol="taichi_forge_cusolverdn_provider_query",
    provider_abi_name=ABI,
    environment_variable="TI_CUSOLVERDN_LIBRARY_PATH",
    library_names=(
        ("cusolver64_12.dll", "cusolver64_11.dll")
        if os.name == "nt"
        else ("libcusolver.so.12", "libcusolver.so.11", "libcusolver.so")
    ),
    package_distributions=(
        "nvidia-cusolver",
        "nvidia-cusolver-cu13",
        "nvidia-cusolver-cu12",
    ),
    supported_version_family="generic Cholesky C API",
)


class PlanDesc(ct.Structure):
    _fields_ = [
        ("struct_size", ct.c_uint32),
        ("scalar", ct.c_uint32),
        ("rows", ct.c_int64),
        ("rhs_count", ct.c_int64),
        ("factor", ct.c_uint64),
        ("status", ct.c_uint64),
    ]


class PlanInfo(ct.Structure):
    _fields_ = [
        ("struct_size", ct.c_uint32),
        ("reserved", ct.c_uint32),
        ("device_workspace_bytes", ct.c_uint64),
        ("host_workspace_bytes", ct.c_uint64),
    ]


class ExecDesc(ct.Structure):
    _fields_ = [
        ("struct_size", ct.c_uint32),
        ("phase", ct.c_uint32),
        ("workspace", ct.c_uint64),
        ("workspace_bytes", ct.c_uint64),
        ("solution", ct.c_uint64),
        ("cuda_stream", ct.c_uint64),
    ]


class ExecutionApi(ct.Structure):
    _fields_ = [
        ("struct_size", ct.c_uint32),
        ("execution_abi_version", ct.c_uint32),
        (
            "create_plan",
            ct.CFUNCTYPE(
                ct.c_int,
                ct.c_void_p,
                ct.POINTER(PlanDesc),
                ct.POINTER(ct.c_void_p),
                ct.POINTER(PlanInfo),
            ),
        ),
        ("execute", ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.POINTER(ExecDesc))),
        ("destroy_plan", ct.CFUNCTYPE(ct.c_int, ct.c_void_p)),
    ]


def open_runtime(library_path=None):
    return _open_runtime(DEFINITION, library_path)


def resolve_library_path(library_path=None):
    return _resolve_library_path(DEFINITION, library_path)


def passive_status():
    return _passive_status(DEFINITION)


def probe_provider(library_path=None):
    return _probe_provider(DEFINITION, library_path)


def check(result, operation):
    if result:
        raise TaichiRuntimeError(f"{operation} failed with status {int(result)}")


def _function(library, name, arguments):
    function = getattr(library, name)
    function.restype = ct.c_int
    function.argtypes = arguments
    return function


class DriverBinding:
    """A context bound once to owned storage; push/pop is activation, not probing."""

    def __init__(self, pointer):
        self.library = (
            ct.WinDLL("nvcuda.dll") if os.name == "nt" else ct.CDLL("libcuda.so.1")
        )
        p = ct.c_void_p
        self.push = _function(self.library, "cuCtxPushCurrent_v2", [p])
        self.pop = _function(self.library, "cuCtxPopCurrent_v2", [ct.POINTER(p)])
        self.copy = _function(
            self.library,
            "cuMemcpyDtoDAsync_v2",
            [ct.c_uint64, ct.c_uint64, ct.c_size_t, p],
        )
        self.clear = _function(
            self.library, "cuMemsetD32Async", [ct.c_uint64, ct.c_uint, ct.c_size_t, p]
        )
        query = _function(
            self.library, "cuPointerGetAttribute", [p, ct.c_int, ct.c_uint64]
        )
        retain = _function(
            self.library, "cuDevicePrimaryCtxRetain", [ct.POINTER(p), ct.c_int]
        )
        self.release = _function(
            self.library, "cuDevicePrimaryCtxRelease_v2", [ct.c_int]
        )
        self.device = ct.c_int()
        check(query(ct.byref(self.device), 9, pointer), "CUDA storage device query")
        self.context = p()
        # Forge uses the primary context and legacy default stream. VMM-backed
        # allocations have no allocation context attribute; resolve their device
        # once and retain that primary context rather than pushing a null handle.
        check(
            retain(ct.byref(self.context), self.device), "CUDA primary context retain"
        )

    def close(self):
        if self.context:
            check(self.release(self.device), "CUDA primary context release")
            self.context = None

    @contextmanager
    def activate(self):
        check(self.push(self.context), "CUDA context activation")
        try:
            yield
        finally:
            previous = ct.c_void_p()
            check(self.pop(ct.byref(previous)), "CUDA context restoration")
