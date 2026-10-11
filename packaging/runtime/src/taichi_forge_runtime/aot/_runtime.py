"""AOT resource owners and ordinary entry binding, independent of the DSL.

This consumer covers ordinary kernels/CGraphs and primitive or tensor ndarrays.
Native recordings, providers, structure arrays and external-memory imports are
not exposed here yet. Host transfers are explicit; kernel calls use device
allocations directly. Close resources explicitly or use Runtime as a context
manager. ``bind()`` reuses Python argument descriptors across submissions.
"""

from __future__ import annotations

import ctypes as C
from dataclasses import dataclass
import math
import operator
import os
from pathlib import Path
import platform
import struct
import threading

from .. import _native
from . import _c_api as abi
from ._c_api import AotError

__all__ = [
    "AotError",
    "Runtime",
    "Module",
    "NdArray",
    "Kernel",
    "Graph",
    "BoundCall",
    "Scalar",
]

_DTYPES = {
    "f16": (0, "e"),
    "f32": (1, "f"),
    "f64": (2, "d"),
    "i8": (3, "b"),
    "i16": (4, "h"),
    "i32": (5, "i"),
    "i64": (6, "q"),
    "u8": (8, "B"),
    "u16": (9, "H"),
    "u32": (10, "I"),
    "u64": (11, "Q"),
}


def _dtype(name):
    try:
        code, format = _DTYPES[name]
    except (KeyError, TypeError):
        raise ValueError(f"Unsupported AOT scalar dtype: {name!r}") from None
    return code, format, struct.calcsize("=" + format)


def _shape(value):
    try:
        values = (operator.index(value),)
    except TypeError:
        values = tuple(operator.index(x) for x in value)
    if len(values) > len(abi.Shape().dims):
        raise ValueError("Shape exceeds the C API's dimension capacity")
    if any(x < 0 or x > 0xFFFFFFFF for x in values):
        raise ValueError("Shape dimensions must be uint32 values")
    return values


def _native_shape(shape):
    return abi.Shape(len(shape), (abi.UInt * 16)(*shape))


def _name(value):
    if not isinstance(value, str) or not value or "\0" in value:
        raise ValueError("Entry and argument names must be nonempty strings without NUL")
    return value.encode("utf-8")


@dataclass(frozen=True)
class Scalar:
    dtype: str
    value: int | float

    def _argument(self):
        code, format, _ = _dtype(self.dtype)
        # The public C API encodes f16 scalar arguments as float32. Array
        # elements still use IEEE half storage; do not confuse the two ABIs.
        if self.dtype == "f16":
            format = "f"
        packed = struct.pack("=" + format, self.value)
        scalar = abi.Scalar()
        scalar.type = code
        C.memmove(C.addressof(scalar.value), packed, len(packed))
        result = abi.Argument()
        result.type, result.value.scalar = 4, scalar
        return result


class Runtime:
    """Own an AOT runtime backed by the shared Forge native library.

    ``arch`` is cpu/x64/arm64, cuda, or vulkan. Runtime creation selects the
    backend explicitly; it never probes other backends or silently falls back.
    Native driver or bundled LLVM loading may still compile code on first use.
    """

    def __init__(self, arch="cpu", *, device_index=0, library_path=None, runtime_path=None):
        self._lock = threading.RLock()
        self._handle = None
        # Keep native registrations independently of Python wrapper lifetimes.
        # Cyclic GC can finalize a runtime and its pending argument owners in
        # either order; every native child must still retire before the device.
        self._resources = set()
        self._pending = set()
        if arch == "cpu":
            arch = "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "x64"
        try:
            native_arch = {"x64": 4, "arm64": 5, "cuda": 3, "vulkan": 1}[arch]
        except (KeyError, TypeError):
            raise ValueError(f"Unsupported AOT consumer backend: {arch!r}") from None
        index = operator.index(device_index)
        if not 0 <= index <= 0xFFFFFFFF:
            raise ValueError("device_index must fit uint32")
        self.arch = arch
        self._api = abi.Api(library_path)
        if arch != "vulkan" or runtime_path is not None:
            _native.configure_runtime_dir(runtime_path)
        self._handle = self._api.call(self._api.ti_create_runtime, native_arch, index)
        if not self._handle:
            raise RuntimeError("Native AOT runtime creation returned no handle")

    def _check(self):
        if self._handle is None:
            raise RuntimeError("AOT runtime is closed")

    def load_module(self, path):
        with self._lock:
            self._check()
            path = os.fspath(Path(path).resolve(strict=True))
            if "\0" in path:
                raise ValueError("AOT module path contains NUL")
            handle = self._api.call(self._api.ti_load_aot_module, self._handle, os.fsencode(path))
            return Module(self, handle)

    def ndarray(self, dtype, shape, *, element_shape=()):
        with self._lock:
            self._check()
            code, _, width = _dtype(dtype)
            shape, element_shape = _shape(shape), _shape(element_shape)
            size = math.prod(shape) * math.prod(element_shape) * width
            if size > 0xFFFFFFFFFFFFFFFF:
                raise ValueError("Array byte size exceeds uint64")
            memory = _Memory(self, size)
            return NdArray(memory, dtype, code, shape, element_shape)

    def wait(self):
        with self._lock:
            self._check()
            self._api.call(self._api.ti_wait, self._handle)
            self._pending.clear()

    def close(self):
        with self._lock:
            if self._handle is None:
                return
            self.wait()
            # Retire code and module-local storage before external arguments.
            priority = {"prepared": 0, "module": 1, "memory": 2}
            for kind, handle in sorted(self._resources, key=lambda item: priority[item[0]]):
                self._destroy_resource(kind, handle)
            self._api.call(self._api.ti_destroy_runtime, self._handle)
            self._handle = None

    def _destroy_resource(self, kind, handle):
        if (kind, handle) not in self._resources:
            return
        if kind == "module":
            self._api.call(self._api.ti_destroy_aot_module, handle)
        elif kind == "prepared":
            self._api.call(self._api.ti_destroy_prepared_ext, self._handle, handle)
        else:
            self._api.call(self._api.ti_free_memory, self._handle, handle)
        self._resources.remove((kind, handle))

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class _Resource:
    def __init__(self, runtime, handle):
        if not handle:
            raise RuntimeError("Native AOT resource creation returned no handle")
        self._runtime, self._handle = runtime, handle
        runtime._resources.add((self._kind, handle))

    def _check(self):
        self._runtime._check()
        if self._handle is None or (self._kind, self._handle) not in self._runtime._resources:
            raise RuntimeError("AOT resource is closed")

    def close(self):
        runtime = self._runtime
        with runtime._lock:
            if self._handle is None or (self._kind, self._handle) not in runtime._resources:
                self._handle = None
                return
            runtime.wait()
            runtime._destroy_resource(self._kind, self._handle)
            self._handle = None

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class _Memory(_Resource):
    _kind = "memory"

    def __init__(self, runtime, size):
        self.size = size
        # Empty logical arrays still need a valid native binding token. Reserve
        # one physical byte; no transfer or array access uses that byte.
        info = abi.MemoryAllocateInfo(max(size, 1), False, False, False, 1)
        handle = runtime._api.call(runtime._api.ti_allocate_memory, runtime._handle, C.byref(info))
        super().__init__(runtime, handle)


class NdArray:
    """Contiguous device allocation with explicit byte-buffer host transfers."""

    def __init__(self, memory, dtype, code, shape, element_shape):
        self._memory = memory
        self._dtype, self._shape, self._element_shape = dtype, shape, element_shape
        self._code = code

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._shape

    @property
    def element_shape(self):
        return self._element_shape

    @property
    def nbytes(self):
        return self._memory.size

    def _descriptor(self, runtime):
        if self._memory._runtime is not runtime:
            raise ValueError("An ndarray belongs to a different AOT runtime")
        self._memory._check()
        return abi.NdArray(
            self._memory._handle,
            _native_shape(self.shape),
            _native_shape(self.element_shape),
            self._code,
        )

    def _transfer(self, buffer, *, to_device):
        view = memoryview(buffer)
        if not view.c_contiguous or view.nbytes != self.nbytes:
            raise ValueError("Host transfer requires a contiguous buffer of exactly nbytes")
        if not to_device and view.readonly:
            raise ValueError("Read destination must be writable")
        runtime = self._memory._runtime
        with runtime._lock:
            self._memory._check()
            device_slice = abi.MemorySlice(self._memory._handle, 0, self.nbytes)
            if view.readonly:
                # ctypes cannot borrow a readonly memoryview. Plain bytes can
                # be passed without copying; other readonly exporters need a
                # temporary host copy whose lifetime spans this synchronous IO.
                source = buffer if isinstance(buffer, bytes) else view.tobytes()
                pointer = C.c_char_p(source)
            else:
                pointer = (C.c_char * self.nbytes).from_buffer(view)
            if to_device:
                runtime._api.call(
                    runtime._api.ti_copy_memory_from_host_ext,
                    runtime._handle,
                    C.byref(device_slice),
                    pointer,
                )
            else:
                runtime._api.call(
                    runtime._api.ti_copy_memory_to_host_ext,
                    runtime._handle,
                    pointer,
                    C.byref(device_slice),
                )
            # Native transfers wait for prior submissions before accessing
            # device memory. A failure keeps pending owners until explicit wait.
            if self.nbytes:
                runtime._pending.clear()

    def write(self, buffer):
        self._transfer(buffer, to_device=True)

    def read_into(self, buffer):
        self._transfer(buffer, to_device=False)

    def read(self):
        result = bytearray(self.nbytes)
        self.read_into(result)
        return bytes(result)

    def close(self):
        self._memory.close()


class Module(_Resource):
    """Own loaded code and module-local resources; entries retain this owner."""

    _kind = "module"

    def kernel(self, name):
        with self._runtime._lock:
            self._check()
            handle = self._runtime._api.call(self._runtime._api.ti_get_aot_module_kernel, self._handle, _name(name))
            return Kernel(self, handle)

    def graph(self, name):
        with self._runtime._lock:
            self._check()
            handle = self._runtime._api.call(
                self._runtime._api.ti_get_aot_module_compute_graph,
                self._handle,
                _name(name),
            )
            return Graph(self, handle)


def _argument(runtime, value, owners):
    if isinstance(value, NdArray):
        result = abi.Argument()
        result.type, result.value.ndarray = 2, value._descriptor(runtime)
        owners.append(value._memory)
        return result
    if isinstance(value, Scalar):
        return value._argument()
    if isinstance(value, int):
        return Scalar("i32", value)._argument()
    if isinstance(value, float):
        return Scalar("f32", value)._argument()
    raise TypeError("AOT arguments must be ndarrays, Scalar, int or float")


class BoundCall:
    """Fixed Python argument descriptors submitted through the public C API.

    This is not a captured GPU Graph. Bind again to change arguments; calls
    retain modules and allocations until the runtime finishes submitted work.
    """

    def __init__(self, entry, arguments, gradients, owners):
        self._entry, self._arguments, self._gradients = entry, arguments, gradients
        self._owners = tuple(dict.fromkeys(owners))

    def run(self):
        module = self._entry._module
        runtime = module._runtime
        with runtime._lock:
            module._check()
            for owner in self._owners:
                owner._check()
            # Even a failing launch may have submitted earlier native work.
            runtime._pending.update(self._owners)
            function = (
                runtime._api.ti_launch_compute_graph_with_gradients_ext
                if isinstance(self._entry, Graph)
                else runtime._api.ti_launch_kernel_with_gradients_ext
            )
            runtime._api.call(
                function,
                runtime._handle,
                self._entry._handle,
                len(self._arguments),
                self._arguments,
                len(self._gradients),
                self._gradients,
            )

    __call__ = run

    def prepare(self):
        """Build native fixed execution without running the bound kernels."""
        runtime = self._entry._module._runtime
        with runtime._lock:
            for owner in self._owners:
                owner._check()
            graph = isinstance(self._entry, Graph)
            binding_type = abi.NamedBinding if graph else abi.Binding
            bindings = (binding_type * len(self._gradients))(
                *[
                    binding_type(
                        gradient.name if graph else gradient.argument_index,
                        0,
                        None,
                        C.pointer(gradient.gradient),
                    )
                    for gradient in self._gradients
                ]
            )
            function = runtime._api.ti_prepare_compute_graph_ext if graph else runtime._api.ti_prepare_kernel_ext
            handle = runtime._api.call(
                function,
                runtime._handle,
                self._entry._handle,
                len(self._arguments),
                self._arguments,
                len(bindings),
                bindings,
            )
            return PreparedCall(runtime, handle, self._owners)


class PreparedCall(_Resource):
    """Immutable native batch; replay resolves no names or Python arguments.

    Change array contents in place, or prepare a new object for new bindings.
    Explicitly closing a module/array invalidates its dependent executions.
    """

    _kind = "prepared"

    def __init__(self, runtime, handle, owners):
        self._owners = owners
        super().__init__(runtime, handle)

    def run(self):
        runtime = self._runtime
        with runtime._lock:
            self._check()
            runtime._pending.add(self)
            runtime._api.call(runtime._api.ti_launch_prepared_ext, runtime._handle, self._handle)

    __call__ = run

    @property
    def info(self):
        runtime = self._runtime
        with runtime._lock:
            self._check()
            native = abi.PreparedInfo()
            runtime._api.call(
                runtime._api.ti_get_prepared_execution_info_ext, runtime._handle, self._handle, C.byref(native)
            )
            return dict(
                dispatch_count=native.dispatch_count,
                execution_kind=("cpu_batch", "cuda_graph", "vulkan_commands")[native.execution_kind],
                argument_bytes=native.argument_bytes,
                launches=native.launches,
            )


class Kernel:
    def __init__(self, module, handle):
        self._module, self._handle = module, handle

    def bind(self, *arguments, gradients=None):
        runtime = self._module._runtime
        with runtime._lock:
            self._module._check()
            owners = [self._module]
            native = (abi.Argument * len(arguments))(*[_argument(runtime, value, owners) for value in arguments])
            bindings = []
            for index, gradient in (gradients or {}).items():
                index = operator.index(index)
                if not 0 <= index < len(arguments) or not isinstance(arguments[index], NdArray):
                    raise ValueError("Gradient indices must refer to ndarray arguments")
                if not isinstance(gradient, NdArray):
                    raise TypeError("A gradient must be an NdArray")
                bindings.append(abi.Gradient(index, gradient._descriptor(runtime)))
                owners.append(gradient._memory)
            return BoundCall(self, native, (abi.Gradient * len(bindings))(*bindings), owners)

    def run(self, *arguments, gradients=None):
        self.bind(*arguments, gradients=gradients).run()

    def prepare(self, *arguments, gradients=None):
        return self.bind(*arguments, gradients=gradients).prepare()

    __call__ = run


class Graph(Kernel):
    def bind(self, arguments, *, gradients=None):
        runtime = self._module._runtime
        with runtime._lock:
            self._module._check()
            owners = [self._module]
            native = (abi.NamedArgument * len(arguments))(
                *[
                    abi.NamedArgument(_name(name), _argument(runtime, value, owners))
                    for name, value in arguments.items()
                ]
            )
            bindings = []
            for name, gradient in (gradients or {}).items():
                if name not in arguments or not isinstance(arguments[name], NdArray):
                    raise ValueError("Gradient names must refer to ndarray arguments")
                if not isinstance(gradient, NdArray):
                    raise TypeError("A gradient must be an NdArray")
                bindings.append(abi.NamedGradient(_name(name), gradient._descriptor(runtime)))
                owners.append(gradient._memory)
            return BoundCall(self, native, (abi.NamedGradient * len(bindings))(*bindings), owners)

    def run(self, arguments, *, gradients=None):
        self.bind(arguments, gradients=gradients).run()

    __call__ = run
