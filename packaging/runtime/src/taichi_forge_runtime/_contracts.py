"""Probe-free discovery of implementation facts from the stable C bootstrap."""

import ctypes as C
from types import MappingProxyType

from . import _native


class _Bootstrap(C.Structure):
    _fields_ = [
        ("struct_size", C.c_uint32),
        ("manifest_schema_version", C.c_uint32),
        ("native_abi_revision", C.c_int32),
        ("runtime_statistics_schema_version", C.c_uint32),
        ("feature_bitmap", C.c_uint64),
        ("compiler_abi", C.c_char * 96),
    ]


_FEATURE_BITS = {
    "cpu": 0,
    "llvm": 1,
    "cuda": 2,
    "vulkan": 3,
    "ggui": 4,
    "opengl": 5,
    "display_shared_image_reuse": 9,
    "completion_wait_concurrent": 10,
    "completion_blocking_wait": 11,
    "cuda_execution_domains": 12,
    "cuda_aot_execution_domains": 13,
    "cuda_jit_graph_execution_domains": 14,
    "cuda_external_storage_execution_domains": 15,
    "cuda_canvas_execution_domains": 16,
    "cuda_submission_parameters_u64": 17,
    "cuda_completion_gpu_timing": 18,
}


def runtime_contract_manifest():
    """Load the native library and read build facts without creating a runtime.

    This uses the stable C bootstrap, never the JIT frontend or a private C++
    object. Missing feature bits remain false for older cores; neither package
    versions nor source hashes imply support. True means the implementation is
    present, not that a device/window/provider or a particular recipe qualifies.
    """
    library = _native.load_library()
    query = library.taichi_forge_runtime_bootstrap_v1
    query.argtypes = [C.POINTER(_Bootstrap), C.c_uint32]
    query.restype = C.c_int
    native = _Bootstrap()
    if query(C.byref(native), C.sizeof(native)) != 0:
        raise RuntimeError("Native runtime rejected its stable bootstrap query")
    if native.struct_size != C.sizeof(native) or native.manifest_schema_version != 1:
        raise RuntimeError("Native runtime has an unsupported bootstrap schema")
    return MappingProxyType(
        {
            "schema_version": int(native.manifest_schema_version),
            "native_abi_revision": int(native.native_abi_revision),
            "runtime_statistics_schema": int(native.runtime_statistics_schema_version),
            "compiler_abi": bytes(native.compiler_abi).decode("ascii"),
            "features": MappingProxyType(
                {
                    name: bool(native.feature_bitmap & (1 << bit))
                    for name, bit in _FEATURE_BITS.items()
                }
            ),
        }
    )
