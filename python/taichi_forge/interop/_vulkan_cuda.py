"""Managed external Vulkan-CUDA allocation interoperability."""

from __future__ import annotations

import operator
import sys
from collections.abc import Sequence

from taichi_forge._lib import core as _ti_core
from taichi_forge.lang import impl
from taichi_forge.lang._storage_view import StorageDescription
from taichi_forge.lang.util import cook_dtype

from ._dlpack import ExternalDenseView


def _current_program():
    program = impl.get_runtime().prog
    if program is None:
        raise RuntimeError("external Vulkan-CUDA import requires ti.init()")
    return program


def _normalize_uuid(value: bytes | bytearray | memoryview | str | Sequence[int]):
    if isinstance(value, str):
        text = value.strip().lower()
        if text.startswith("gpu-"):
            text = text[4:]
        text = text.replace("-", "").replace("{", "").replace("}", "")
        try:
            value = bytes.fromhex(text)
        except ValueError as exc:
            raise ValueError("device_uuid must contain exactly 16 UUID bytes") from exc
    else:
        try:
            value = bytes(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "device_uuid must be 16 bytes or a hexadecimal UUID string"
            ) from exc
    if len(value) != 16:
        raise ValueError(
            f"device_uuid must contain exactly 16 bytes, got {len(value)}"
        )
    return value


def _normalize_handle(value, name):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer OS handle")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer OS handle") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive OS handle")
    return int(value)


def _normalize_shape(value, name):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer sequence")
    try:
        value = (operator.index(value),)
    except TypeError:
        try:
            value = tuple(value)
        except TypeError as exc:
            raise TypeError(f"{name} must be an integer sequence") from exc
    result = []
    for extent in value:
        if isinstance(extent, bool):
            raise TypeError(f"{name} entries must be integers")
        try:
            extent = operator.index(extent)
        except TypeError as exc:
            raise TypeError(f"{name} entries must be integers") from exc
        if extent < 0:
            raise ValueError(f"{name} entries must be non-negative")
        result.append(int(extent))
    return tuple(result)


class VulkanCudaExternalAllocation:
    """Imported external allocation that produces managed dense storage views.

    The resource owns only CUDA's imported mapping and imported semaphore
    handles. The original Vulkan allocation and semaphore objects remain owned
    by the external producer.
    """

    __slots__ = ("_native",)

    def __init__(self, native):
        self._native = native

    @property
    def provider(self):
        return "vulkan_cuda"

    @property
    def allocation_bytes(self):
        return int(self._native.allocation_bytes)

    @property
    def device_uuid(self):
        return bytes(self._native.device_uuid)

    @property
    def device(self):
        return 2, int(self._native.device_id)

    @property
    def synchronized(self):
        return bool(self._native.synchronized)

    @property
    def closed(self):
        return bool(self._native.closed)

    def view(
        self,
        *,
        dtype,
        shape,
        element_shape=(),
        offset_bytes=0,
        access="readwrite",
    ):
        """Create an independently managed compact AOS view.

        Multiple views may use different scalar types and byte offsets. They
        share one synchronization-domain identity, so one Graph submission
        acquires the external allocation only once.
        """

        if self.closed:
            raise RuntimeError("external Vulkan-CUDA allocation is closed")
        shape = _normalize_shape(shape, "shape")
        element_shape = _normalize_shape(element_shape, "element_shape")
        if isinstance(offset_bytes, bool):
            raise TypeError("offset_bytes must be an integer")
        try:
            offset_bytes = operator.index(offset_bytes)
        except TypeError as exc:
            raise TypeError("offset_bytes must be an integer") from exc
        native = self._native._view(
            cook_dtype(dtype),
            shape,
            element_shape,
            int(offset_bytes),
            access,
        )
        description = StorageDescription(native.description)
        if not description.supported:
            native.close()
            raise BufferError(
                "external Vulkan-CUDA allocation cannot form an executable "
                f"dense view: {description.failure_reason}"
            )
        if not description.properties["ndarray_abi_compatible"]:
            native.close()
            raise BufferError(
                "external Vulkan-CUDA kernel bindings require compact AOS "
                "storage"
            )
        return ExternalDenseView(native, description)

    def close(self):
        self._native.close()

    def __enter__(self):
        if self.closed:
            raise RuntimeError("external Vulkan-CUDA allocation is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def current_cuda_device_uuid():
    """Return the 16-byte UUID of Forge's active CUDA device."""

    return bytes(_ti_core._current_cuda_external_device_uuid(_current_program()))


def import_vulkan_cuda_allocation(
    memory_handle,
    *,
    allocation_bytes,
    device_uuid,
    ready_for_cuda_handle=None,
    ready_for_vulkan_handle=None,
    handle_type=None,
    dedicated=True,
    allow_unsynchronized=False,
):
    """Import Vulkan-exported memory as managed CUDA dense storage.

    Handle ownership transfers to this call. A synchronized import requires a
    pair of binary semaphores: Vulkan signals ``ready_for_cuda_handle`` and
    waits on ``ready_for_vulkan_handle``. Omitting them requires the explicit
    ``allow_unsynchronized=True`` opt-in and external exclusion by the caller.
    """

    memory_handle = _normalize_handle(memory_handle, "memory_handle")
    if isinstance(allocation_bytes, bool):
        raise TypeError("allocation_bytes must be a positive integer")
    try:
        allocation_bytes = operator.index(allocation_bytes)
    except TypeError as exc:
        raise TypeError(
            "allocation_bytes must be a positive integer"
        ) from exc
    if allocation_bytes <= 0:
        raise ValueError("allocation_bytes must be a positive integer")
    device_uuid = _normalize_uuid(device_uuid)
    if (ready_for_cuda_handle is None) != (ready_for_vulkan_handle is None):
        raise ValueError(
            "ready_for_cuda_handle and ready_for_vulkan_handle must be "
            "supplied together"
        )
    if ready_for_cuda_handle is not None:
        ready_for_cuda_handle = _normalize_handle(
            ready_for_cuda_handle, "ready_for_cuda_handle"
        )
        ready_for_vulkan_handle = _normalize_handle(
            ready_for_vulkan_handle, "ready_for_vulkan_handle"
        )
        if len(
            {
                memory_handle,
                ready_for_cuda_handle,
                ready_for_vulkan_handle,
            }
        ) != 3:
            raise ValueError(
                "memory and semaphore imports require distinct OS handles"
            )
    elif not allow_unsynchronized:
        raise ValueError(
            "unsynchronized external import requires "
            "allow_unsynchronized=True"
        )
    platform_handle_type = (
        "opaque_win32" if sys.platform == "win32" else "opaque_fd"
    )
    if handle_type is None:
        handle_type = platform_handle_type
    if handle_type not in ("opaque_win32", "opaque_fd"):
        raise ValueError(
            "handle_type must be 'opaque_win32' or 'opaque_fd'"
        )
    if handle_type != platform_handle_type:
        raise ValueError(
            f"handle_type={handle_type!r} is unavailable on this platform"
        )
    if dedicated is not True:
        raise ValueError(
            "external Vulkan-CUDA import currently requires dedicated=True"
        )

    native = _ti_core._import_vulkan_cuda_allocation(
        _current_program(),
        memory_handle,
        int(allocation_bytes),
        device_uuid,
        ready_for_cuda_handle,
        ready_for_vulkan_handle,
        handle_type,
        True,
        bool(allow_unsynchronized),
    )
    return VulkanCudaExternalAllocation(native)


def capabilities():
    arch = impl.current_cfg().arch
    available = bool(_ti_core.with_cuda() and arch == _ti_core.Arch.cuda)
    return {
        "provider": "vulkan_cuda",
        "available": available,
        "backend": _ti_core.arch_name(arch),
        "handle_types": (
            ("opaque_win32",) if sys.platform == "win32" else ("opaque_fd",)
        ),
        "memory": ("dedicated_buffer",),
        "synchronization": (
            "binary_semaphore_pair",
            "explicit_unsynchronized",
        ),
        "layouts": ("compact", "aos", "typed_offset_views"),
        "access": ("readwrite",),
        "graph_access_epoch": True,
        "copy_fallback": False,
    }


__all__ = [
    "VulkanCudaExternalAllocation",
    "capabilities",
    "current_cuda_device_uuid",
    "import_vulkan_cuda_allocation",
]
