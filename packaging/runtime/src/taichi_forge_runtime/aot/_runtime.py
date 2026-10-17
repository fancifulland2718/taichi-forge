"""AOT resource owners and ordinary entry binding, independent of the DSL.

This consumer covers ordinary kernels/CGraphs and primitive or tensor ndarrays.
Fixed recordings support loaded kernels, declared native operations and private arrays.
Explicit JIT borrowers can import scalar/tensor ndarrays with storage leases;
structure arrays and arbitrary external-memory imports are not exposed here.
Host transfers are explicit; kernel calls use device
allocations directly. Close resources explicitly or use Runtime as a context
manager. ``bind()`` reuses Python argument descriptors across submissions.
"""

from __future__ import annotations

import ctypes as C
import json
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
    "NativeResource",
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

_NATIVE_OPERATION_MODES = (
    (1, "cpu"), (2, "cuda_capture"), (4, "vulkan_inline"), (8, "cuda_synchronous"), (16, "cuda_ordered")
)


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

    def _pack(self, byte_order):
        code, format, _ = _dtype(self.dtype)
        # The public C API encodes f16 scalar arguments as float32. Array
        # elements still use IEEE half storage; do not confuse the two ABIs.
        if self.dtype == "f16":
            format = "f"
        return code, struct.pack(byte_order + format, self.value)

    def resource(self, name):
        """Describe an immutable private scalar without allocating device storage."""
        _name(name)
        _, packed = self._pack(">")
        return dict(kind="scalar", name=name, dtype=self.dtype, bits=packed.hex())

    def _argument(self):
        code, packed = self._pack("=")
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
        self._borrower = None
        self._ndarray_importer = None
        self._texture_importer = None
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

    @classmethod
    def _from_borrowed(cls, client, library_path, *, ndarray_importer=None, texture_importer=None):
        result = cls.__new__(cls)
        result._lock = result._borrower = client
        result._ndarray_importer = ndarray_importer
        result._texture_importer = texture_importer
        result._handle = None
        result._resources, result._pending = set(), set()
        try:
            result._api = abi.Api(library_path)
            result.arch = str(client.arch).split(".")[-1]
            result._handle = client.handle
            result._check()
        except BaseException:
            client.close()
            raise
        return result

    def _check(self):
        if self._handle is None or (self._borrower is not None and self._borrower.closed):
            raise RuntimeError("AOT runtime is closed")

    def device_capabilities(self):
        """Return a snapshot of this runtime's enabled device capability levels.

        Missing names have level zero. This cold query neither allocates device
        storage nor loads operation providers. Combine it with registered modes
        and provider probes before selecting a route; prepare still validates
        actual layouts and device limits. Unknown future IDs retain numeric keys.
        """
        with self._lock:
            self._check()
            count = abi.UInt()
            query = self._api.ti_get_runtime_capabilities
            self._api.call(query, self._handle, C.byref(count), None)
            entries = (abi.CapabilityLevelInfo * count.value)()
            self._api.call(query, self._handle, C.byref(count), entries)
            if count.value > len(entries):
                raise RuntimeError("Device capabilities changed during enumeration")
            return {
                abi.CAPABILITY_NAMES.get(entry.capability, entry.capability): entry.level
                for entry in entries[: count.value]
            }

    def native_operation_capabilities(self, kind, *, version=1):
        """Registered composition modes for this backend and operation version.

        This cold query does not load operation code or allocate scratch. Empty
        means unregistered. Concrete shapes, dtypes and device requirements are
        still checked at prepare time. It does not probe external providers.
        """
        return self._native_capabilities(kind, version, resource=False)

    def native_resource_capabilities(self, kind, *, version=1):
        """Registered resource-factory modes without loading dependencies.

        Resource kinds are distinct from operation kinds. A nonempty result
        identifies a compiled factory; it does not admit every device/format.
        """
        return self._native_capabilities(kind, version, resource=True)

    def _native_capabilities(self, kind, version, *, resource):
        kind = _name(kind)
        if type(version) is not int or not 1 <= version <= 0xFFFFFFFF:
            raise ValueError("Operation version must be a positive uint32")
        with self._lock:
            self._check()
            query = (
                self._api.ti_get_native_resource_capabilities_ext
                if resource
                else self._api.ti_get_native_operation_capabilities_ext
            )
            flags = self._api.call(query, self._handle, kind, version)
        return frozenset(name for bit, name in _NATIVE_OPERATION_MODES if flags & bit)

    def configure_provider(self, provider, *, adapter_path, library_path=None):
        """Set library locations for subsequent preparation on this owner.

        No code is loaded here. Existing prepared graphs retain their leases.
        ``adapter_path=None`` removes this owner's setting; ``library_path=None``
        uses the adapter's platform vendor search. Explicit paths are resolved
        now so later working-directory changes cannot redirect preparation.
        """
        provider = _name(provider)

        def path(value):
            if value is None:
                return None
            value = os.fspath(value)
            if not isinstance(value, str) or not value or "\0" in value:
                raise ValueError("Provider paths must be nonempty strings without NUL")
            return os.fsencode(Path(value).expanduser().resolve())

        adapter_path, library_path = path(adapter_path), path(library_path)
        if adapter_path is None and library_path is not None:
            raise ValueError("Removing a provider cannot specify a vendor library")
        with self._lock:
            self._check()
            self._api.call(
                self._api.ti_configure_native_provider_ext,
                self._handle,
                provider,
                adapter_path,
                library_path,
            )

    def configure_library(self, name, path):
        """Set a cold auxiliary-library location for this runtime owner.

        Only a registered factory that needs this name loads it. None removes
        the setting; existing prepared owners keep their leases. This does not
        change process search paths or embed deployment paths in an artifact.
        """
        name = _name(name)
        if path is not None:
            path = os.fspath(path)
            if not isinstance(path, str) or not path or "\0" in path:
                raise ValueError("Native library paths must be nonempty strings without NUL")
            path = os.fsencode(Path(path).expanduser().resolve())
        with self._lock:
            self._check()
            self._api.call(self._api.ti_configure_native_library_ext, self._handle, name, path)

    def probe_native_operation(self, kind, *, version=1):
        """Cold optional-dependency check; may load a library, never a plan.

        Missing providers report available=False. A successful probe does not
        qualify every device/layout; preparation validates the actual binding.
        Replay does not repeat this query. Library versions are provider-specific.
        """
        return self._probe_native(kind, version, resource=False)

    def probe_native_resource(self, kind, *, version=1):
        """Probe a resource factory's optional dependency without allocating it.

        May load configured libraries. Successful probes leave device, format,
        dimension and input-owner checks to cold resource creation.
        """
        return self._probe_native(kind, version, resource=True)

    def _probe_native(self, kind, version, *, resource):
        kind = _name(kind)
        if type(version) is not int or not 1 <= version <= 0xFFFFFFFF:
            raise ValueError("Operation version must be a positive uint32")
        with self._lock:
            self._check()
            result = abi.NativeOperationAvailability()
            query = self._api.ti_probe_native_resource_ext if resource else self._api.ti_probe_native_operation_ext
            self._api.call(query, self._handle, kind, version, C.byref(result))
        return {
            "available": bool(result.available),
            "capabilities": frozenset(name for bit, name in _NATIVE_OPERATION_MODES if result.capabilities & bit),
            "library_version": result.library_version or None,
        }

    def load_module(self, path):
        with self._lock:
            self._check()
            path = os.fspath(Path(path).resolve(strict=True))
            if "\0" in path:
                raise ValueError("AOT module path contains NUL")
            handle = self._api.call(self._api.ti_load_aot_module, self._handle, os.fsencode(path))
            module = Module(self, handle)
            module._source_path = Path(path)
            return module

    def ndarray(self, dtype, shape, *, element_shape=(), usage=("storage",)):
        """Allocate a packed array with explicit storage/uniform/vertex/index uses.

        Usage describes the native allocation, not access synchronization.
        Include ``storage`` when the same array is passed to an AOT kernel.
        """
        with self._lock:
            self._check()
            code, _, width = _dtype(dtype)
            shape, element_shape = _shape(shape), _shape(element_shape)
            size = math.prod(shape) * math.prod(element_shape) * width
            if size > 0xFFFFFFFFFFFFFFFF:
                raise ValueError("Array byte size exceeds uint64")
            memory = _Memory(self, size, usage)
            return NdArray(memory, dtype, code, shape, element_shape)

    def host_blob(self, data):
        """Copy immutable host bytes for native provider preparation.

        This uses the same owner and retirement contract as NativeResource,
        with no device allocation or transfer. Mutating or releasing the source
        after return is safe. A blob cannot serve as an ndarray or kernel AS.
        """
        payload = data if type(data) is bytes else memoryview(data).tobytes()
        with self._lock:
            self._check()
            handle = self._api.call(self._api.ti_create_host_blob_ext, self._handle, len(payload), payload)
            try:
                return NativeResource(self, handle, ())
            except BaseException:
                self._api.call(self._api.ti_destroy_native_resource_ext, self._handle, handle)
                raise

    def native_resource(self, provider, operation, *, parameters=None, bindings=None, version=1):
        """Create one resource through the common native factory catalog.

        Bindings map factory ports to same-runtime arrays or NativeResources.
        Plans may share the result; closing it retires all dependent calls and
        resources. Factories identify completed construction-only inputs;
        all other inputs remain retained. This is a cold allocation boundary.
        """
        from ._native import native_references

        values = dict(bindings or {})
        description = dict(
            kind="native",
            provider=provider,
            operation=operation,
            version=version,
            parameters={} if parameters is None else parameters,
            bindings={name: name for name in values},
        )
        native_references(description)
        encoded = json.dumps(description, allow_nan=False, separators=(",", ":")).encode("utf-8")
        with self._lock:
            self._check()
            owners, arguments = [], []
            for name, value in values.items():
                retained = []
                arguments.append(abi.NamedArgument(_name(name), _argument(self, value, retained)))
                owners.append(retained)
            flags = (abi.UInt * len(arguments))()
            handle = self._api.call(
                self._api.ti_create_native_resource_ext,
                self._handle,
                len(encoded),
                encoded,
                len(arguments),
                (abi.NamedArgument * len(arguments))(*arguments),
                flags,
            )
            try:
                retained = tuple(dict.fromkeys(owner for keep, group in zip(flags, owners) if keep for owner in group))
                return NativeResource(self, handle, retained)
            except BaseException:
                self._api.call(self._api.ti_destroy_native_resource_ext, self._handle, handle)
                raise

    def wait(self):
        with self._lock:
            self._check()
            self._api.call(self._api.ti_wait, self._handle)
            self._pending.clear()

    def import_ndarray(self, array):
        """Retain a zero-copy JIT ndarray through an explicit borrowed owner.

        This cold operation shares storage, not a Python array wrapper. The
        imported view remains live if the source wrapper is collected. Closing
        this view drops its lease and retires its prepared users; Program reset
        retires all borrowed views before backend teardown. Import ``array.grad``
        separately when binding an exported explicit gradient.
        """
        with self._lock:
            self._check()
            if self._ndarray_importer is None:
                raise RuntimeError("JIT ndarray import requires ti.aot.borrow_runtime()")
            handle, size, dtype, shape, element_shape = self._ndarray_importer(self._borrower, array)
            memory = None
            try:
                code, _, width = _dtype(dtype)
                shape, element_shape = _shape(shape), _shape(element_shape)
                if size != math.prod(shape) * math.prod(element_shape) * width:
                    raise ValueError("Imported ndarray layout disagrees with its storage extent")
                memory = _Memory._from_import(self, handle, size)
                return NdArray(memory, dtype, code, shape, element_shape)
            except BaseException:
                if memory is None:
                    self._api.call(self._api.ti_free_memory, self._handle, handle)
                else:
                    memory.close()
                raise

    def import_texture(self, texture):
        """Borrow existing JIT texture storage and sampler without copying pixels.

        The source must follow the JIT initialization contract before sampling.
        Source wrapper collection is safe; Program reset retires all borrowers.
        Closing this handle leaves the source and independent borrowers usable.
        """
        with self._lock:
            self._check()
            if self._texture_importer is None:
                raise RuntimeError("JIT Texture import requires ti.aot.borrow_runtime()")
            handle = self._texture_importer(self._borrower, texture)
            try:
                return NativeResource(self, handle, ())
            except BaseException:
                self._api.call(self._api.ti_destroy_native_resource_ext, self._handle, handle)
                raise

    def close(self):
        with self._lock:
            if self._handle is None:
                return
            if self._borrower is not None:
                try:
                    self._borrower.close()
                finally:
                    self._handle = None
                    self._resources.clear()
                    self._pending.clear()
                return
            self.wait()
            # Retire code and module-local storage before external arguments.
            priority = {"prepared": 0, "module": 1, "native_resource": 2, "memory": 3}
            for kind, handle in sorted(self._resources, key=lambda item: priority[item[0]]):
                self._destroy_resource(kind, handle)
            self._api.call(self._api.ti_destroy_runtime, self._handle)
            self._handle = None

    def _invalidate_runtime(self):
        self.close()

    def _destroy_resource(self, kind, handle):
        if (kind, handle) not in self._resources:
            return
        if kind == "module":
            self._api.call(self._api.ti_destroy_aot_module, handle)
        elif kind == "prepared":
            self._api.call(self._api.ti_destroy_prepared_ext, self._handle, handle)
        elif kind == "native_resource":
            self._api.call(self._api.ti_destroy_native_resource_ext, self._handle, handle)
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
            if runtime._borrower is not None and runtime._borrower.closed:
                runtime.close()
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


class NativeResource(_Resource):
    """One runtime-owned native object, usable by multiple execution plans."""

    _kind = "native_resource"

    def __init__(self, runtime, handle, owners):
        self._owners = owners
        super().__init__(runtime, handle)

    def close(self):
        super().close()
        self._owners = ()


def _array_usage(usage):
    flags = {"storage": 1, "uniform": 2, "vertex": 4, "index": 8}
    if not isinstance(usage, (tuple, list)) or not usage or any(
        not isinstance(item, str) or item not in flags for item in usage
    ):
        raise ValueError("Array usage must be a nonempty sequence of storage/uniform/vertex/index names")
    bits = 0
    for item in usage:
        bits |= flags[item]
    return bits


class _Memory(_Resource):
    _kind = "memory"

    def __init__(self, runtime, size, usage=("storage",)):
        self.size = size
        usage_bits = _array_usage(usage)
        # Empty logical arrays still need a valid native binding token. Reserve
        # one physical byte; no transfer or array access uses that byte.
        info = abi.MemoryAllocateInfo(max(size, 1), False, False, False, usage_bits)
        handle = runtime._api.call(runtime._api.ti_allocate_memory, runtime._handle, C.byref(info))
        super().__init__(runtime, handle)

    @classmethod
    def _from_import(cls, runtime, handle, size):
        result = cls.__new__(cls)
        result.size = size
        _Resource.__init__(result, runtime, handle)
        return result


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

    def execution_plan(self, name):
        """Read a named persistent plan from a directory or .tcm module."""
        from urllib.parse import quote
        from ._plan import ExecutionPlan

        with self._runtime._lock:
            self._check()
            _name(name)
            relative = "executions/" + quote(name, safe="-_.") + ".json"
            if self._source_path.is_dir():
                return ExecutionPlan.load(self._source_path / relative)
            from zipfile import ZipFile

            with ZipFile(self._source_path) as archive:
                return ExecutionPlan.from_json(archive.read(relative))

    def prepare_execution(self, plan, arguments, *, gradients=None):
        """Prepare a named or explicit ExecutionPlan for repeated native replay."""
        from ._plan import ExecutionPlan

        if isinstance(plan, str):
            plan = self.execution_plan(plan)
        if not isinstance(plan, ExecutionPlan):
            raise TypeError("Expected an ExecutionPlan or an exported plan name")
        return plan.prepare(self, arguments, gradients=gradients)

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
    if isinstance(value, NativeResource):
        value._check()
        if value._runtime is not runtime:
            raise ValueError("Native resource belongs to another runtime")
        result = abi.Argument()
        result.type, result.value.native_resource = 6, value._handle
        owners.append(value)
        return result
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
    raise TypeError("AOT arguments must be ndarrays, NativeResource, Scalar, int or float")


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


class _BindingFamily:
    def __init__(self, module, plan, initial_owners, persistent_owners):
        self.module, self.plan = module, plan
        self.initial_owners = initial_owners
        self.persistent_owners = persistent_owners
        self.initialized = False


class PreparedCall(_Resource):
    """Immutable native batch; replay resolves no names or Python arguments.

    Change array contents in place, or prepare a new object for new bindings.
    Explicitly closing a module/array invalidates its dependent executions.
    """

    _kind = "prepared"

    def __init__(self, runtime, handle, owners, *, replay_owners=None, family=None):
        self._owners = owners
        self._replay_owners = replay_owners
        self._family = family
        super().__init__(runtime, handle)

    def prepare_binding(self, arguments, *, gradients=None):
        """Prepare a sibling sharing the selected CUDA plan's state and executable.

        Each binding is immutable. Initialization runs once for the family;
        closing a sibling leaves the other bindings usable. Native resource
        inputs are fixed; use a new prepare_execution() for an independent family.
        """
        with self._runtime._lock:
            self._check()
            if self._family is None:
                raise ValueError("Preparing a binding requires a selected CUDA binding family")
            return self._family.plan._prepare(self._family.module, arguments, gradients=gradients, source=self)

    def initialize(self):
        """Run initialization once and wait; release initialization-only owners."""
        runtime = self._runtime
        with runtime._lock:
            self._check()
            runtime._pending.add(self)
            runtime._api.call(runtime._api.ti_initialize_prepared_ext, runtime._handle, self._handle)
            if self._family is not None:
                self._family.initialized = True
                self._family.initial_owners = ()
            if self._replay_owners is not None:
                self._owners, self._replay_owners = self._replay_owners, None

    def close(self):
        super().close()
        self._owners = ()
        self._replay_owners = None
        self._family = None

    def run(self):
        runtime = self._runtime
        with runtime._lock:
            self._check()
            if self._replay_owners is not None or (self._family is not None and not self._family.initialized):
                self.initialize()
            runtime._pending.add(self)
            runtime._api.call(runtime._api.ti_launch_prepared_ext, runtime._handle, self._handle)

    __call__ = run

    @property
    def binding_family_info(self):
        """Cold shared-allocation diagnostics, or None for ordinary executions."""
        with self._runtime._lock:
            self._check()
            native = abi.BindingFamilyInfo()
            self._runtime._api.call(
                self._runtime._api.ti_get_prepared_binding_family_info_ext,
                self._runtime._handle,
                self._handle,
                C.byref(native),
            )
            return {name: getattr(native, name) for name, _ in native._fields_} if native.identity else None

    @property
    def info(self):
        runtime = self._runtime
        with runtime._lock:
            self._check()
            native = abi.PreparedInfo()
            runtime._api.call(
                runtime._api.ti_get_prepared_execution_info_ext, runtime._handle, self._handle, C.byref(native)
            )
            memory = abi.PreparedMemoryInfo()
            runtime._api.call(
                runtime._api.ti_get_prepared_execution_memory_info_ext, runtime._handle, self._handle, C.byref(memory)
            )
            return dict(
                dispatch_count=native.dispatch_count,
                execution_kind=("cpu_batch", "cuda_graph", "vulkan_commands", "cuda_mixed", "cuda_ordered", "host_control")[
                    native.execution_kind
                ],
                argument_bytes=native.argument_bytes,
                workspace_bytes=memory.workspace_bytes,
                resource_bytes=memory.resource_bytes,
                native_resource_bytes=memory.native_resource_bytes,
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
