"""Managed Vulkan imports for fixed independent Graph execution."""

import operator
import sys

from taichi_forge._lib import core as _ti_core
from taichi_forge.lang import impl
from taichi_forge.lang._storage_view import DenseNdarrayView, StorageDescription
from taichi_forge.lang.util import cook_dtype
from taichi_forge_runtime.aot import (
    VulkanExternalBufferDescriptor,
    VulkanExternalReadyGate,
)

from ._dlpack import _current_program
from ._vulkan_cuda import _normalize_shape


class VulkanExternalAllocation:
    """One native import and one Program owner; typed views borrow that owner."""

    __slots__ = ("_native", "__weakref__")

    def __init__(self, native):
        self._native = native

    @property
    def closed(self):
        return self._native.closed

    @property
    def provider(self):
        return "vulkan"

    @property
    def allocation_bytes(self):
        return int(self._native.allocation_bytes)

    def view(self, *, dtype, shape, element_shape=(), offset_bytes=0):
        """Borrow a compact typed range relative to the imported finite range.

        Views retain this root and share its synchronization identity. Dropping
        one view leaves its siblings usable. Explicit root close retires every
        dependent prepared instance; a view has no separate close operation.
        """
        if isinstance(offset_bytes, bool):
            raise TypeError("offset_bytes must be an integer")
        offset_bytes = operator.index(offset_bytes)
        description = StorageDescription(
            self._native._describe_typed_view(
                cook_dtype(dtype),
                _normalize_shape(shape, "shape"),
                _normalize_shape(element_shape, "element_shape"),
                offset_bytes,
            )
        )
        # DenseNdarrayView has no close: its _source keeps this root alive.
        return DenseNdarrayView(self, description)

    def close(self):
        """Retire dependent work and every view at a cold boundary."""
        self._native.close()

    def __enter__(self):
        if self.closed:
            raise RuntimeError("external Vulkan allocation is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def import_vulkan_external_buffer(descriptor, *, ready_gate, byte_offset=0, byte_size=None):
    """Import one producer buffer/semaphore pair once on the current Program.

    Actual producer facts and a native gate use the same contract as the AOT
    runtime. Views are valid for fixed Graph.prepare execution domains; ordinary
    launches and external initialization cannot bypass the ownership handshake.
    Caller HANDLEs remain caller-owned on success and failure.
    """
    if not isinstance(descriptor, VulkanExternalBufferDescriptor):
        raise TypeError("descriptor must be VulkanExternalBufferDescriptor")
    if not isinstance(ready_gate, VulkanExternalReadyGate):
        raise TypeError("ready_gate must be VulkanExternalReadyGate")
    if sys.platform != "win32" or not hasattr(
        _ti_core, "_import_vulkan_external_buffer"
    ):
        raise NotImplementedError(
            "Managed Vulkan external buffers require Windows x64 with Vulkan support"
        )
    if isinstance(byte_offset, bool):
        raise TypeError("byte_offset must be an integer")
    byte_offset = operator.index(byte_offset)
    if byte_size is None:
        byte_size = descriptor.buffer_size - byte_offset
    info = descriptor._native_view(byte_offset, byte_size)
    native = _ti_core._import_vulkan_external_buffer(
        _current_program(), bytes(info), ready_gate._capsule
    )
    try:
        return VulkanExternalAllocation(native)
    except BaseException:
        native.close()
        raise


def capabilities():
    """Describe this adapter; concrete producer/device facts are checked at import."""
    arch = impl.current_cfg().arch
    return {
        "provider": "vulkan",
        "available": (
            sys.platform == "win32"
            and arch == _ti_core.Arch.vulkan
            and hasattr(_ti_core, "_import_vulkan_external_buffer")
        ),
        "backend": _ti_core.arch_name(arch),
        "handle_types": ("opaque_win32",),
        "memory": ("dedicated_exclusive_buffer",),
        "synchronization": ("binary_semaphore_pair", "native_enqueued_ready_gate"),
        "layouts": ("compact", "aos", "typed_offset_views"),
        "access": ("readwrite",),
        "execution": ("fixed_independent_domain",),
        "external_initialization": False,
        "copy_fallback": False,
    }
