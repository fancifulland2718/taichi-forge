"""Cold Vulkan import facts shared by runtime-only and DSL adapters."""

import ctypes as C
from dataclasses import dataclass
import math
import operator
import os

from . import _c_api as abi
from ._external_ready import VulkanExternalReadyGate, _ReadyGate


def _uint(value, name, bits=64, *, positive=False):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    value = operator.index(value)
    if not int(positive) <= value < 1 << bits:
        raise ValueError(f"{name} must fit {'positive ' if positive else ''}uint{bits}")
    return value


@dataclass(frozen=True, kw_only=True)
class VulkanExternalBufferDescriptor:
    """Actual creation/allocation facts for a borrowed OPAQUE_WIN32 buffer.

    Handles remain producer-owned. This description has no device ownership and
    is never serialized into an AOT artifact. Import each physical semaphore
    pair once and derive aliases from that import. The initial route requires
    the same GPU/driver, dedicated allocation, EXCLUSIVE sharing and bind offset
    zero. A finite view does not narrow the whole-buffer ownership transaction.
    """

    memory_handle: int
    ready_for_consumer: int
    ready_for_external: int
    buffer_size: int
    allocation_size: int
    bind_offset: int
    memory_type_index: int
    buffer_flags: int
    buffer_usage: int
    allocation_flags: int
    sharing_mode: int
    dedicated: bool
    device_uuid: bytes
    driver_uuid: bytes

    def __post_init__(self):
        for name in ("memory_handle", "ready_for_consumer", "ready_for_external",
                     "buffer_size", "allocation_size", "bind_offset"):
            object.__setattr__(self, name, _uint(getattr(self, name), name, positive=name != "bind_offset"))
        for name in ("memory_type_index", "buffer_flags", "buffer_usage", "allocation_flags", "sharing_mode"):
            object.__setattr__(self, name, _uint(getattr(self, name), name, 32))
        if type(self.dedicated) is not bool:
            raise TypeError("dedicated must be bool")
        if not self.dedicated or self.bind_offset or self.sharing_mode:
            raise ValueError("External Vulkan import requires dedicated EXCLUSIVE storage with bind offset zero")
        if self.buffer_size > self.allocation_size:
            raise ValueError("buffer_size exceeds allocation_size")
        if len({self.memory_handle, self.ready_for_consumer, self.ready_for_external}) != 3:
            raise ValueError("External memory and semaphore handles must be distinct")
        for name in ("device_uuid", "driver_uuid"):
            value = getattr(self, name)
            if not isinstance(value, (bytes, bytearray, memoryview)) or len(value) != 16:
                raise ValueError(f"{name} must contain exactly 16 bytes")
            value = bytes(value)
            if len(value) != 16 or not any(value):
                raise ValueError(f"{name} must be a nonzero 16-byte UUID")
            object.__setattr__(self, name, value)

    def _native_view(self, byte_offset, byte_size):
        offset = _uint(byte_offset, "byte_offset")
        size = _uint(byte_size, "byte_size", positive=True)
        if size > self.buffer_size or offset > self.buffer_size - size:
            raise ValueError("External view must be a positive range within buffer_size")
        values = {name: getattr(self, name) for name in self.__dataclass_fields__}
        for name in ("device_uuid", "driver_uuid"):
            values[name] = (C.c_uint8 * 16)(*values[name])
        return abi.VulkanExternalBufferImportInfo(
            struct_size=C.sizeof(abi.VulkanExternalBufferImportInfo), schema_version=1,
            byte_offset=offset, byte_size=size, **values,
        )


def import_buffer(runtime, descriptor, *, ready_gate, dtype, shape, element_shape=(), byte_offset=0):
    from ._runtime import NdArray, _Memory, _dtype, _shape

    if not isinstance(descriptor, VulkanExternalBufferDescriptor):
        raise TypeError("descriptor must be VulkanExternalBufferDescriptor")
    if not isinstance(ready_gate, VulkanExternalReadyGate):
        raise TypeError("ready_gate must be VulkanExternalReadyGate")
    if os.name != "nt":
        raise NotImplementedError("Managed Vulkan external buffers currently require OPAQUE_WIN32")
    code, _, width = _dtype(dtype)
    shape, element_shape = _shape(shape), _shape(element_shape)
    if any(x == 0 for x in element_shape):
        raise ValueError("Tensor element extents must be positive")
    if any(x > 0x7FFFFFFF for x in shape + element_shape):
        raise ValueError("External view extents exceed native indexing")
    size = math.prod(shape) * math.prod(element_shape) * width
    info = descriptor._native_view(byte_offset, size)
    if info.byte_offset % width:
        raise ValueError("External view offset must be scalar aligned")
    with runtime._lock:
        runtime._check()
        if runtime.arch != "vulkan":
            raise ValueError("External Vulkan import requires a Vulkan runtime")
        runtime._api.load_vulkan_external_buffer()
        handle = runtime._api.call(
            runtime._api.ti_import_vulkan_external_buffer_ext, runtime._handle,
            C.byref(info), C.cast(ready_gate._address, C.POINTER(_ReadyGate)),
        )
        memory = None
        try:
            memory = _Memory._from_import(runtime, handle, size)
            return NdArray(memory, dtype, code, shape, element_shape)
        except BaseException:
            if memory is None:
                runtime._api.call(runtime._api.ti_free_memory, runtime._handle, handle)
            else:
                memory.close()
            raise
