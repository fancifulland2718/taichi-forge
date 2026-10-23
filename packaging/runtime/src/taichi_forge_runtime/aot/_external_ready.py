"""Native producer readiness, checked once per complete prepared access."""

import ctypes as C


class _ReadyGate(C.Structure):
    _fields_ = [
        ("struct_size", C.c_uint32), ("schema_version", C.c_uint32),
        ("owner", C.c_void_p), ("retain_owner", C.c_void_p),
        ("release_owner", C.c_void_p), ("query", C.c_void_p),
    ]


class _NativeReadyGate:
    def __init__(self, capsule, name, backend):
        get = C.pythonapi.PyCapsule_GetPointer
        get.argtypes, get.restype = [C.py_object, C.c_char_p], C.c_void_p
        address = get(capsule, name)
        value = C.cast(address, C.POINTER(_ReadyGate)).contents
        if (value.struct_size != C.sizeof(_ReadyGate) or value.schema_version != 1
                or not all((value.owner, value.retain_owner, value.release_owner, value.query))):
            raise ValueError(f"Unsupported native {backend} external ready gate descriptor")
        self._capsule = capsule
        self._address = address


class CudaExternalReadyGate(_NativeReadyGate):
    """A producer-owned native readiness descriptor, independent of the DSL.

    The capsule ``taichi.cuda_external_ready_gate.v1`` points to the public
    ``TiCudaExternalReadyGate`` descriptor. Its callbacks must be native,
    nonblocking and independent of Python interpreter lifetime. The native
    import retains the owner; the capsule retains the descriptor until import.

    Queries identify the next ready signal by an epoch starting at 1. A false
    result makes prepared submit return None without consuming a signal or
    queuing GPU work. True preserves the ordinary semaphore acquire/release.
    """

    def __init__(self, capsule):
        super().__init__(capsule, b"taichi.cuda_external_ready_gate.v1", "CUDA")


class VulkanExternalReadyGate(_NativeReadyGate):
    """Native admission and producer lifetime for one Vulkan semaphore pair.

    The capsule ``taichi.vulkan_external_ready_gate.v1`` contains the public
    TiVulkanExternalReadyGate. Query identifies a handshake ordinal starting
    at one, independently of application payload versions. True promises that
    the matching producer signal is already enqueued; device work may remain
    pending. False consumes nothing. Callbacks must be nonblocking, native,
    independent of Python, and must not reenter Forge or submit GPU work.

    The native owner retains the producer allocation, device and synchronization
    objects through destruction of all imported objects. The capsule keeps the
    descriptor alive until import. Ready queries never replace binary waits or
    whole-buffer queue-family ownership transfers.
    """

    def __init__(self, capsule):
        super().__init__(capsule, b"taichi.vulkan_external_ready_gate.v1", "Vulkan")


__all__ = ["CudaExternalReadyGate", "VulkanExternalReadyGate"]
