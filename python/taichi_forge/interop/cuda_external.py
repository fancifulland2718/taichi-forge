"""CUDA/Vulkan external-memory interop on Windows.

This module deliberately stays in Python and uses NVIDIA's optional
``cuda.bindings`` package.  It imports dedicated Vulkan allocations and binary
semaphores exported as ``OPAQUE_WIN32`` handles into the CUDA primary context
already selected by Taichi Forge.

The import functions *consume* the integer Win32 handle passed by the caller:
the handle is closed exactly once after the CUDA import attempt, including
error paths.  CUDA objects and mapped pointers remain owned by the returned
RAII wrappers.
"""

from __future__ import annotations

import contextlib
import ctypes
import math
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np


_ARRAY_TOKEN = object()
_CUDA_EXTERNAL_ARRAY_PROTOCOL = "taichi-forge.cuda-external-array.v1"
_MAX_U64 = (1 << 64) - 1
_SUPPORTED_DTYPES = {
    np.dtype(np.bool_),
    np.dtype(np.float16),
    np.dtype(np.float32),
    np.dtype(np.float64),
    np.dtype(np.int8),
    np.dtype(np.int16),
    np.dtype(np.int32),
    np.dtype(np.int64),
    np.dtype(np.uint8),
    np.dtype(np.uint16),
    np.dtype(np.uint32),
    np.dtype(np.uint64),
}


class CudaExternalInteropError(RuntimeError):
    """Raised when CUDA external-resource import or synchronization fails."""


def _load_cuda_driver():
    try:
        from cuda.bindings import driver
    except (ImportError, OSError) as exc:
        raise CudaExternalInteropError(
            "CUDA external interop requires the optional 'cuda-bindings>=13.2,<14' "
            "package and a compatible NVIDIA driver."
        ) from exc
    return driver


def _cuda_error_text(driver, error) -> str:
    fields = []
    for name in ("cuGetErrorName", "cuGetErrorString"):
        getter = getattr(driver, name, None)
        if getter is None:
            continue
        try:
            result = getter(error)
            if isinstance(result, tuple) and len(result) > 1 and int(result[0]) == 0:
                value = result[1]
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="replace")
                fields.append(str(value))
        except Exception:  # pragma: no cover - diagnostic best effort
            pass
    return ": ".join(fields) or f"CUDA error {int(error)}"


def _cuda_call(driver, name: str, *args):
    result = getattr(driver, name)(*args)
    if not isinstance(result, tuple):
        result = (result,)
    if not result:
        raise CudaExternalInteropError(f"{name} returned no CUDA status")
    if int(result[0]) != 0:
        raise CudaExternalInteropError(f"{name} failed: {_cuda_error_text(driver, result[0])}")
    if len(result) == 1:
        return None
    if len(result) == 2:
        return result[1]
    return result[1:]


def _normalize_device_uuid(value: bytes | bytearray | memoryview | str | Sequence[int]) -> bytes:
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
            raise TypeError("device_uuid must be 16 bytes or a hexadecimal UUID string") from exc
    if len(value) != 16:
        raise ValueError(f"device_uuid must contain exactly 16 bytes, got {len(value)}")
    return value


def _normalize_dtype(dtype) -> np.dtype:
    try:
        result = np.dtype(dtype)
    except (TypeError, ValueError):
        # Import lazily to keep this module safe during taichi_forge's import
        # cycle and to support public Taichi primitive dtypes such as ti.f32.
        from taichi_forge.lang.util import cook_dtype, to_numpy_type

        try:
            result = np.dtype(to_numpy_type(cook_dtype(dtype)))
        except Exception as exc:
            raise TypeError(f"Unsupported CUDA external array dtype: {dtype!r}") from exc
    if result not in _SUPPORTED_DTYPES or not result.isnative:
        raise TypeError(f"Unsupported CUDA external array scalar dtype: {result}")
    return result


def _normalize_shape(shape: Sequence[int] | int) -> tuple[int, ...]:
    if isinstance(shape, (int, np.integer)):
        shape = (int(shape),)
    else:
        try:
            shape = tuple(shape)
        except TypeError as exc:
            raise TypeError("shape must be an integer or a sequence of integers") from exc
    normalized = []
    for dim in shape:
        if not isinstance(dim, (int, np.integer)):
            raise TypeError(f"shape entries must be integers, got {type(dim).__name__}")
        dim = int(dim)
        if dim < 0:
            raise ValueError(f"shape entries must be non-negative, got {shape}")
        normalized.append(dim)
    return tuple(normalized)


def _close_win32_handle(handle: int) -> None:
    if sys.platform != "win32":
        raise CudaExternalInteropError("OPAQUE_WIN32 interop is only available on Windows")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # pylint: disable=no-member
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [ctypes.c_void_p]
    close_handle.restype = ctypes.c_int
    if not close_handle(ctypes.c_void_p(handle)):
        error = ctypes.get_last_error()
        raise CudaExternalInteropError(f"CloseHandle(0x{handle:x}) failed with Win32 error {error}")


class _ConsumedWin32Handle:
    def __init__(self, handle: int):
        if not isinstance(handle, int):
            raise TypeError("handle must be a raw integer Win32 HANDLE transferred by the caller")
        if handle <= 0:
            raise ValueError("handle must be a non-zero Win32 HANDLE")
        self._handle = handle

    def __enter__(self) -> int:
        return self._handle

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        handle, self._handle = self._handle, 0
        if handle:
            _close_win32_handle(handle)
        return False


def _require_taichi_cuda_context(driver):
    # Deliberately lazy: kernel_impl imports the strict array predicate below.
    from taichi_forge._lib import core as _ti_core
    from taichi_forge.lang import impl

    prog = impl.get_runtime().prog
    if prog is None:
        raise CudaExternalInteropError("Call ti.init(arch=ti.cuda) before importing CUDA external resources")
    if prog.config().arch != _ti_core.Arch.cuda:
        raise CudaExternalInteropError("CUDA external resources require a Taichi runtime initialized with arch=ti.cuda")

    context = _cuda_call(driver, "cuCtxGetCurrent")
    if int(context) == 0:
        raise CudaExternalInteropError(
            "The calling thread has no current CUDA context; import on the same thread that initialized Taichi CUDA"
        )
    device = _cuda_call(driver, "cuCtxGetDevice")
    uuid = _cuda_call(driver, "cuDeviceGetUuid", device)
    return context, device, bytes(uuid.bytes)


def _require_matching_device(driver, expected_uuid):
    context, device, actual_uuid = _require_taichi_cuda_context(driver)
    expected_uuid = _normalize_device_uuid(expected_uuid)
    if actual_uuid != expected_uuid:
        raise CudaExternalInteropError(
            "Vulkan/CUDA device UUID mismatch: " f"Vulkan={expected_uuid.hex()} CUDA={actual_uuid.hex()}"
        )
    return context, device, actual_uuid


@contextlib.contextmanager
def _using_context(driver, context):
    current = _cuda_call(driver, "cuCtxGetCurrent")
    if int(current) == int(context):
        yield
        return
    _cuda_call(driver, "cuCtxPushCurrent", context)
    try:
        yield
    finally:
        popped = _cuda_call(driver, "cuCtxPopCurrent")
        if int(popped) != int(context):
            raise CudaExternalInteropError("CUDA context stack changed while releasing an external resource")


def current_cuda_device_uuid() -> bytes:
    """Return the UUID of the device in Taichi's current CUDA context."""

    driver = _load_cuda_driver()
    return _require_taichi_cuda_context(driver)[2]


class CudaExternalArray:
    """A contiguous typed byte-range view accepted by ``ti.types.ndarray``."""

    __taichi_cuda_external_array__ = _CUDA_EXTERNAL_ARRAY_PROTOCOL
    requires_grad = False

    def __init__(
        self,
        memory: "CudaExternalMemory",
        *,
        shape: tuple[int, ...],
        dtype: np.dtype,
        offset: int,
        token,
    ):
        if token is not _ARRAY_TOKEN:
            raise TypeError("Create external CUDA arrays with CudaExternalMemory.array()")
        self._memory = memory
        self._shape = shape
        self._dtype = dtype
        self._offset = offset
        self._nbytes = math.prod(shape) * dtype.itemsize

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def offset(self) -> int:
        return self._offset

    @property
    def nbytes(self) -> int:
        return self._nbytes

    @property
    def closed(self) -> bool:
        return self._memory.closed

    @property
    def device_uuid(self) -> bytes:
        return self._memory.device_uuid

    def data_ptr(self) -> int:
        self._memory._require_open()  # pylint: disable=protected-access
        return self._memory._data_ptr + self.offset  # pylint: disable=protected-access

    def __repr__(self) -> str:
        state = "closed" if self.closed else f"ptr=0x{self.data_ptr():x}"
        return f"CudaExternalArray(shape={self.shape}, dtype={self.dtype}, " f"offset={self.offset}, {state})"


def is_cuda_external_array(value: Any) -> bool:
    """Strict predicate used by the kernel argument binder."""

    return (
        isinstance(value, CudaExternalArray) and value.__taichi_cuda_external_array__ == _CUDA_EXTERNAL_ARRAY_PROTOCOL
    )


def validate_cuda_external_array(value: Any) -> CudaExternalArray:
    if not is_cuda_external_array(value):
        raise TypeError(f"Expected CudaExternalArray, got {type(value).__name__}")
    value._memory._require_open()  # pylint: disable=protected-access
    expected = math.prod(value.shape) * value.dtype.itemsize
    if value.nbytes != expected or value.nbytes < 0:
        raise CudaExternalInteropError("Invalid CUDA external array byte extent")
    pointer = value.data_ptr()
    if pointer <= 0:
        raise CudaExternalInteropError("CUDA external array has a null device pointer")
    return value


class CudaExternalMemory:
    """Own an imported CUDA external-memory object and its full mapping."""

    def __init__(
        self,
        driver,
        context,
        device_uuid: bytes,
        external_memory,
        data_ptr,
        allocation_size: int,
    ):
        self._driver = driver
        self._context = context
        self._device_uuid = device_uuid
        self._external_memory = external_memory
        self._data_ptr = int(data_ptr)
        self._allocation_size = allocation_size

    @property
    def allocation_size(self) -> int:
        return self._allocation_size

    @property
    def device_uuid(self) -> bytes:
        return self._device_uuid

    @property
    def closed(self) -> bool:
        return self._external_memory is None

    def _require_open(self) -> None:
        if self.closed or self._data_ptr == 0:
            raise CudaExternalInteropError("CUDA external memory is closed")

    def _require_current_context(self) -> None:
        self._require_open()
        current = _cuda_call(self._driver, "cuCtxGetCurrent")
        if int(current) != int(self._context):
            raise CudaExternalInteropError(
                "CUDA external array must be launched from its importing Taichi CUDA context"
            )

    def array(self, *, shape: Sequence[int] | int, dtype, offset: int = 0) -> CudaExternalArray:
        """Create a contiguous typed view into this mapped allocation."""

        self._require_open()
        shape = _normalize_shape(shape)
        dtype = _normalize_dtype(dtype)
        if not isinstance(offset, (int, np.integer)):
            raise TypeError("offset must be an integer byte offset")
        offset = int(offset)
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if offset > _MAX_U64:
            raise ValueError("offset exceeds the CUDA 64-bit byte range")
        if offset % dtype.itemsize:
            raise ValueError(f"offset {offset} is not aligned to dtype item size {dtype.itemsize}")
        nbytes = math.prod(shape) * dtype.itemsize
        if offset > self._allocation_size or nbytes > self._allocation_size - offset:
            raise ValueError(
                f"external array byte range [{offset}, {offset + nbytes}) exceeds "
                f"allocation size {self._allocation_size}"
            )
        return CudaExternalArray(self, shape=shape, dtype=dtype, offset=offset, token=_ARRAY_TOKEN)

    def close(self) -> None:
        if self.closed:
            return
        driver = self._driver
        first_error = None
        with _using_context(driver, self._context):
            try:
                _cuda_call(driver, "cuStreamSynchronize", 0)
            except Exception as exc:
                first_error = exc
            if first_error is None and self._data_ptr:
                try:
                    _cuda_call(driver, "cuMemFree", self._data_ptr)
                    self._data_ptr = 0
                except Exception as exc:
                    first_error = first_error or exc
            if first_error is None and self._external_memory is not None:
                try:
                    _cuda_call(driver, "cuDestroyExternalMemory", self._external_memory)
                    self._external_memory = None
                except Exception as exc:
                    first_error = first_error or exc
        if first_error is not None:
            raise first_error

    def __enter__(self) -> "CudaExternalMemory":
        self._require_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class CudaExternalSemaphore:
    """Own an imported Vulkan binary semaphore."""

    def __init__(self, driver, context, device_uuid: bytes, semaphore):
        self._driver = driver
        self._context = context
        self._device_uuid = device_uuid
        self._semaphore = semaphore

    @property
    def device_uuid(self) -> bytes:
        return self._device_uuid

    @property
    def closed(self) -> bool:
        return self._semaphore is None

    def _require_open(self) -> None:
        if self.closed:
            raise CudaExternalInteropError("CUDA external semaphore is closed")

    def _enqueue(self, operation: str) -> None:
        self._require_open()
        with _using_context(self._driver, self._context):
            if operation == "wait":
                params = self._driver.CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS()
                function = "cuWaitExternalSemaphoresAsync"
            else:
                params = self._driver.CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS()
                function = "cuSignalExternalSemaphoresAsync"
            # Stream 0 is Taichi Forge's shared legacy default stream.
            _cuda_call(self._driver, function, [self._semaphore], [params], 1, 0)

    def wait(self) -> None:
        """Enqueue a binary wait on Taichi's legacy default CUDA stream."""

        self._enqueue("wait")

    def signal(self) -> None:
        """Enqueue a binary signal on Taichi's legacy default CUDA stream."""

        self._enqueue("signal")

    def close(self) -> None:
        if self.closed:
            return
        with _using_context(self._driver, self._context):
            _cuda_call(self._driver, "cuStreamSynchronize", 0)
            _cuda_call(self._driver, "cuDestroyExternalSemaphore", self._semaphore)
            self._semaphore = None

    def __enter__(self) -> "CudaExternalSemaphore":
        self._require_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def import_vulkan_memory_win32(
    handle: int,
    *,
    allocation_size: int,
    device_uuid: bytes | bytearray | memoryview | str | Sequence[int],
    dedicated: bool = True,
) -> CudaExternalMemory:
    """Import and map a dedicated Vulkan ``OPAQUE_WIN32`` allocation.

    ``handle`` ownership transfers to this call.  The Win32 duplicate is
    closed after the import attempt regardless of success or failure.
    """

    external_memory = None
    data_ptr = 0
    driver = None
    context = None
    try:
        with _ConsumedWin32Handle(handle) as raw_handle:
            if sys.platform != "win32":
                raise CudaExternalInteropError("OPAQUE_WIN32 interop is only available on Windows")
            if not isinstance(allocation_size, (int, np.integer)) or int(allocation_size) <= 0:
                raise ValueError("allocation_size must be a positive integer")
            allocation_size = int(allocation_size)
            if allocation_size > _MAX_U64:
                raise ValueError("allocation_size exceeds the CUDA 64-bit byte range")
            if dedicated is not True:
                raise ValueError("Only dedicated Vulkan OPAQUE_WIN32 allocations are supported")

            driver = _load_cuda_driver()
            context, _, actual_uuid = _require_matching_device(driver, device_uuid)
            desc = driver.CUDA_EXTERNAL_MEMORY_HANDLE_DESC()
            desc.type = driver.CUexternalMemoryHandleType.CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32
            desc.handle.win32.handle = raw_handle
            desc.handle.win32.name = 0
            desc.size = allocation_size
            desc.flags = driver.CUDA_EXTERNAL_MEMORY_DEDICATED
            external_memory = _cuda_call(driver, "cuImportExternalMemory", desc)

        buffer_desc = driver.CUDA_EXTERNAL_MEMORY_BUFFER_DESC()
        buffer_desc.offset = 0
        buffer_desc.size = allocation_size
        buffer_desc.flags = 0
        data_ptr = _cuda_call(driver, "cuExternalMemoryGetMappedBuffer", external_memory, buffer_desc)
        if int(data_ptr) == 0:
            raise CudaExternalInteropError("cuExternalMemoryGetMappedBuffer returned a null pointer")
        return CudaExternalMemory(
            driver,
            context,
            actual_uuid,
            external_memory,
            data_ptr,
            allocation_size,
        )
    except Exception:
        if driver is not None and external_memory is not None:
            with _using_context(driver, context):
                if data_ptr:
                    try:
                        _cuda_call(driver, "cuMemFree", data_ptr)
                    except Exception:
                        pass
                try:
                    _cuda_call(driver, "cuDestroyExternalMemory", external_memory)
                except Exception:
                    pass
        raise


def import_vulkan_semaphore_win32(
    handle: int,
    *,
    device_uuid: bytes | bytearray | memoryview | str | Sequence[int],
) -> CudaExternalSemaphore:
    """Import a Vulkan binary semaphore exported as ``OPAQUE_WIN32``."""

    semaphore = None
    driver = None
    context = None
    try:
        with _ConsumedWin32Handle(handle) as raw_handle:
            if sys.platform != "win32":
                raise CudaExternalInteropError("OPAQUE_WIN32 interop is only available on Windows")
            driver = _load_cuda_driver()
            context, _, actual_uuid = _require_matching_device(driver, device_uuid)
            desc = driver.CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC()
            desc.type = driver.CUexternalSemaphoreHandleType.CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32
            desc.handle.win32.handle = raw_handle
            desc.handle.win32.name = 0
            desc.flags = 0
            semaphore = _cuda_call(driver, "cuImportExternalSemaphore", desc)
        return CudaExternalSemaphore(driver, context, actual_uuid, semaphore)
    except Exception:
        if driver is not None and semaphore is not None:
            with _using_context(driver, context):
                try:
                    _cuda_call(driver, "cuDestroyExternalSemaphore", semaphore)
                except Exception:
                    pass
        raise


__all__ = [
    "CudaExternalArray",
    "CudaExternalInteropError",
    "CudaExternalMemory",
    "CudaExternalSemaphore",
    "current_cuda_device_uuid",
    "import_vulkan_memory_win32",
    "import_vulkan_semaphore_win32",
]
