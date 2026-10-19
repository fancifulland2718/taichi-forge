"""Initialization-only Vulkan/CUDA association, independent of the compiler."""

import ctypes as C

from . import _c_api as abi

_POOL_COUNTERS = (
    "native_allocation_calls", "native_free_calls", "reuse_calls", "retirement_calls",
    "reserved_current_bytes", "reserved_high_bytes", "used_current_bytes", "used_high_bytes",
)


class _Config(C.Structure):
    _fields_ = [
        ("struct_size", C.c_uint32), ("schema_version", C.c_uint32),
        ("mode", C.c_uint32), ("graphics_api", C.c_uint32),
        ("device_uuid", C.c_uint8 * 16),
        ("queue_data", C.c_void_p), ("queue_data_size", C.c_uint64),
        ("owner", C.c_void_p), ("retain_owner", C.c_void_p), ("release_owner", C.c_void_p),
    ]


class _Info(C.Structure):
    _fields_ = [
        ("struct_size", C.c_uint32), ("schema_version", C.c_uint32),
        ("initialized", C.c_uint32), ("vulkan_cig", C.c_uint32),
        ("driver_entry", C.c_uint32), ("device_capability", C.c_uint32),
        ("generation", C.c_uint64), ("device_uuid", C.c_uint8 * 16),
        ("allocation_checks", C.c_uint32), ("process_lifetime", C.c_uint32),
        ("allocation_route", C.c_uint32),
        *((name, C.c_uint64) for name in _POOL_COUNTERS),
        ("fallback_reason", C.c_char * 256),
    ]


class CudaVulkanContextBinding:
    """A native external-queue descriptor supplied by a Vulkan device owner.

    The capsule is named ``taichi.cuda_vulkan_context_binding.v1`` and points
    to ``TiCudaContextConfig`` from the public C header. Its destructor owns
    the descriptor and the bytes until configuration acquires the native lease.
    Callbacks must be native code, independent of Python interpreter lifetime.
    Forge retains the queue/device lease with its process-lived CUDA context.
    """

    def __init__(self, capsule):
        get = C.pythonapi.PyCapsule_GetPointer
        get.argtypes, get.restype = [C.py_object, C.c_char_p], C.c_void_p
        address = get(capsule, b"taichi.cuda_vulkan_context_binding.v1")
        value = C.cast(address, C.POINTER(_Config)).contents
        if value.struct_size != C.sizeof(_Config) or value.schema_version != 1 or value.graphics_api != 1:
            raise ValueError("Unsupported native CUDA/Vulkan binding descriptor")
        self._capsule = capsule
        self._address = address


def _functions(library_path):
    api = abi.Api(library_path)
    try:
        configure = api.library.ti_configure_cuda_context_ext
        query = api.library.ti_get_cuda_context_info_ext
    except AttributeError:
        raise NotImplementedError("Native runtime lacks CUDA/Vulkan context selection") from None
    configure.argtypes, configure.restype = [C.POINTER(_Config)], None
    query.argtypes, query.restype = [C.POINTER(_Info)], None
    return api, configure, query


def cuda_context_info(*, library_path=None):
    """Read initialization facts without creating a context or probing a GPU."""
    api, _, query = _functions(library_path)
    info = _Info(C.sizeof(_Info), 1)
    api.call(query, C.byref(info))
    return dict(
        context_mode=("vulkan_cig" if info.vulkan_cig else "ordinary") if info.initialized else "uninitialized",
        device_uuid=bytes(info.device_uuid).hex(),
        generation=info.generation,
        driver_entry=bool(info.driver_entry),
        device_capability=bool(info.device_capability),
        allocation_route="retained_device" if info.allocation_route == 1 else "ordinary",
        retained_pool={name: getattr(info, name) for name in _POOL_COUNTERS} if info.allocation_route == 1 else None,
        allocation_checks={name: bool(info.allocation_checks & bit) for bit, name in (
            (1, "device"), (2, "managed"), (4, "async"), (8, "owned_pool"), (16, "capture")
        )},
        context_lifetime="process" if info.process_lifetime else "runtime",
        fallback_reason=info.fallback_reason.decode("utf-8", errors="replace"),
    )


def configure_cuda_context(binding=None, *, mode="required", library_path=None):
    """Select the one CUDA context before ti.init() or Runtime('cuda').

    ``required`` fails if CiG or the qualified allocation route is unavailable;
    ``auto`` can fall back only before publishing CUDA resources, and reports
    why; ``disabled`` selects the ordinary path. Invalid descriptors, late
    configuration and failed native retirement never silently fall back.
    AOT artifacts remain portable bytes; only their consuming runtime is bound.
    """
    try:
        native_mode = {"disabled": 0, "required": 1, "auto": 2}[mode]
    except (KeyError, TypeError):
        raise ValueError("mode must be required, auto or disabled") from None
    if binding is not None and not isinstance(binding, CudaVulkanContextBinding):
        raise TypeError("binding must be CudaVulkanContextBinding")
    if (mode == "required" and binding is None) or (mode == "disabled" and binding is not None):
        raise ValueError("required needs a binding; disabled does not accept one")
    config = _Config(C.sizeof(_Config), 1, native_mode, 1)
    if binding is not None:
        C.memmove(C.byref(config), binding._address, C.sizeof(config))
        config.mode = native_mode
    api, configure, _ = _functions(library_path)
    api.call(configure, C.byref(config))
    return cuda_context_info(library_path=library_path)


__all__ = ["CudaVulkanContextBinding", "configure_cuda_context", "cuda_context_info"]
