"""AOT resource owners and ordinary entry binding, independent of the DSL.

This consumer covers ordinary kernels/CGraphs and primitive, tensor or fixed-AoS ndarrays.
Fixed recordings support loaded kernels, declared native operations and private arrays.
Explicit JIT borrowers can import scalar/tensor ndarrays with storage leases;
Synchronized CUDA external linear buffers can be imported for selected-domain
recipes without the DSL. Structure arrays use layouts queried from loaded entries.
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
from ._completion import SubmissionTicket
from ._struct import StructLayout
from ._value import ArgPackLayout, ArgumentLayout, ReturnLayout

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
        self._dense_view_importer = None
        self._texture_importer = None
        self._acceleration_structure_importer = None
        self._handle = None
        # Keep native registrations independently of Python wrapper lifetimes.
        # Cyclic GC can finalize a runtime and its pending argument owners in
        # either order; every native child must still retire before the device.
        self._resources = set()
        self._pending = set()
        self._domain_owners = {}
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
    def _from_borrowed(cls, client, library_path, *, ndarray_importer=None, texture_importer=None,
                       dense_view_importer=None, acceleration_structure_importer=None):
        result = cls.__new__(cls)
        result._lock = result._borrower = client
        result._ndarray_importer = ndarray_importer
        result._dense_view_importer = dense_view_importer
        result._texture_importer = texture_importer
        result._acceleration_structure_importer = acceleration_structure_importer
        result._handle = None
        result._resources, result._pending = set(), set()
        result._domain_owners = {}
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

    def execution_domain(self, *, capacity=2, instance_capacity=64, priority=0, gpu_timing=False, queue="compute"):
        """Create an independent CUDA stream or additional Vulkan queue.

        Capacity bounds in-flight storage; instance_capacity budgets resident
        preparations. Vulkan requires enabled timelines and an unused physical
        queue; its priority is fixed at device creation and must be zero here.
        Vulkan queue="graphics" selects a graphics+compute family for private
        offscreen recipes. The default retains the existing compute selection.
        Queue indices are local to info["queue_family_index"].
        """
        from ._domain import create_domain

        return create_domain(self, capacity, instance_capacity, priority, gpu_timing, queue)

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
        """Allocate a packed array with explicit storage/uniform/vertex/index/indirect uses.

        Usage describes the native allocation, not access synchronization.
        Include ``storage`` when the same array is passed to an AOT kernel.
        ``dtype`` can be a scalar name or a StructLayout queried from a loaded
        entry. Structure arrays use the exported AoS stride and whole-record IO.
        """
        with self._lock:
            self._check()
            shape, element_shape = _shape(shape), _shape(element_shape)
            if isinstance(dtype, StructLayout):
                dtype._check(self)
                if element_shape:
                    raise ValueError("A structure array cannot have an element_shape")
                code, width = 13, dtype.element_stride
            else:
                code, _, width = _dtype(dtype)
            size = math.prod(shape) * math.prod(element_shape) * width
            if size > 0xFFFFFFFFFFFFFFFF:
                raise ValueError("Array byte size exceeds uint64")
            memory = _Memory(self, size, usage)
            return NdArray(memory, dtype, code, shape, element_shape)

    def dense_snode_group(self, schema, dimensions):
        """Instantiate an experimental dense schema with the native layout compiler.

        The complete root is allocated once. Qualified compact member views
        share it; closing the group drains/revokes all views. This cold API
        does not load the DSL or compile numerical definitions.
        """
        from ._dense_group import DenseSNodeGroup

        return DenseSNodeGroup.create(self, schema, dimensions)

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

    def import_dense_view(self, view):
        """Borrow a compact managed dense view without copying.

        Tree destruction drains dependent prepared users and revokes all
        aliases before freeing the root. The imported handle remains closeable.
        This cold call must run outside a borrowed runtime submission scope.
        Synchronized CUDA external storage requires a prepared execution domain;
        its semaphore pair is consumed once per complete initialization/replay.
        """
        self._check()
        if self._dense_view_importer is None:
            raise RuntimeError("Dense view import requires ti.aot.borrow_runtime()")
        # Native import acquires tree lifetime before the Program resource gate.
        # Entering self._lock here would invert that order against tree destroy.
        handle, size, dtype, shape, element_shape = self._dense_view_importer(self._borrower, view)
        with self._lock:
            self._check()
            memory = None
            try:
                code, _, width = _dtype(dtype)
                shape, element_shape = _shape(shape), _shape(element_shape)
                if size != math.prod(shape) * math.prod(element_shape) * width:
                    raise ValueError("Imported dense view layout disagrees with its extent")
                memory = _Memory._from_import(self, handle, size)
                return NdArray(memory, dtype, code, shape, element_shape)
            except BaseException:
                if memory is None:
                    self._api.call(self._api.ti_free_memory, self._handle, handle)
                else:
                    memory.close()
                raise

    def import_cuda_external_memory(
        self, *, memory_handle, ready_for_cuda, ready_for_external,
        allocation_size, device_uuid, dtype, shape, element_shape=(),
        byte_offset=0, dedicated=True, handle_type=None, ready_gate=None,
    ):
        """Import a synchronized external linear buffer for prepared recipes.

        All original OS handles remain caller-owned on success and failure.
        Keep the external allocation/device/semaphore objects alive until close.
        Each complete initialization/replay consumes one ready signal and emits
        one consumed signal. Ordinary launch/map/host-transfer paths reject it.
        Enqueue the producer signal before submitting CUDA work; it may remain
        pending on device. Enqueue the consumer wait after CUDA submit returns.
        Import each physical pair once and use plan views for aliases. The
        external producer remains responsible for draining its own queued work.

        An optional CudaExternalReadyGate retains native producer query state.
        An unready input makes prepared submit return None before any device
        work; accepted calls still perform the full semaphore protocol. Epochs
        start at 1 per imported pair. Initialization inputs must already be ready.
        """
        from ._external_ready import CudaExternalReadyGate, _ReadyGate

        if ready_gate is not None and not isinstance(ready_gate, CudaExternalReadyGate):
            raise TypeError("ready_gate must be CudaExternalReadyGate")

        def integer(value, name):
            if isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            result = operator.index(value)
            if not 0 <= result <= 0xFFFFFFFFFFFFFFFF:
                raise ValueError(f"{name} must fit uint64")
            return result

        code, _, width = _dtype(dtype)
        shape, element_shape = _shape(shape), _shape(element_shape)
        if any(x == 0 for x in element_shape):
            raise ValueError("Tensor element extents must be positive")
        if any(x > 0x7FFFFFFF for x in shape + element_shape):
            raise ValueError("External view extents exceed native indexing")
        size = math.prod(shape) * math.prod(element_shape) * width
        allocation_size = integer(allocation_size, "allocation_size")
        byte_offset = integer(byte_offset, "byte_offset")
        if not 0 < size <= allocation_size or byte_offset > allocation_size - size:
            raise ValueError("External view must be a positive range within its allocation")
        if byte_offset % width:
            raise ValueError("External view offset must be scalar aligned")
        if not isinstance(device_uuid, (bytes, bytearray)) or len(device_uuid) != 16:
            raise ValueError("device_uuid must contain exactly 16 bytes")
        if type(dedicated) is not bool:
            raise TypeError("dedicated must be bool")
        if handle_type is None:
            handle_type = "opaque_win32" if os.name == "nt" else "opaque_fd"
        if handle_type not in ("opaque_win32", "opaque_fd"):
            raise ValueError("handle_type must be opaque_win32 or opaque_fd")
        handles = tuple(integer(value, name) for value, name in (
            (memory_handle, "memory_handle"), (ready_for_cuda, "ready_for_cuda"),
            (ready_for_external, "ready_for_external"),
        ))
        if len(set(handles)) != 3 or (handle_type == "opaque_win32" and not all(handles)):
            raise ValueError("External memory and semaphore handles must be valid and distinct")
        info = abi.CudaExternalMemoryImportInfo(
            1 if handle_type == "opaque_win32" else 2, dedicated, *handles,
            allocation_size, byte_offset, size, (C.c_uint8 * 16)(*device_uuid),
        )
        with self._lock:
            self._check()
            if self.arch != "cuda":
                raise ValueError("Synchronized external import requires a CUDA runtime")
            if ready_gate is None:
                self._api.load_cuda_external_memory()
                handle = self._api.call(self._api.ti_import_cuda_external_memory_ext, self._handle, C.byref(info))
            else:
                self._api.load_cuda_external_ready_gate()
                handle = self._api.call(
                    self._api.ti_import_cuda_external_memory_with_ready_gate_ext,
                    self._handle, C.byref(info), C.cast(ready_gate._address, C.POINTER(_ReadyGate)),
                )
            memory = None
            try:
                memory = _Memory._from_import(self, handle, size)
                return NdArray(memory, dtype, code, shape, element_shape)
            except BaseException:
                if memory is None:
                    self._api.call(self._api.ti_free_memory, self._handle, handle)
                else:
                    memory.close()
                raise

    def import_vulkan_external_buffer(
        self, descriptor, *, ready_gate, dtype, shape, element_shape=(), byte_offset=0,
    ):
        """Import a managed Win32 buffer for an independent Vulkan recipe.

        Descriptor and native gate are frozen at cold import. Each accepted
        complete replay consumes one ready signal and publishes one consumed
        signal on the selected queue. Ordinary launch/map/host-transfer and
        external initialization are rejected. Close retires dependent Forge
        work; the producer remains responsible for its own queued work.
        """
        from ._external_vulkan import import_buffer

        return import_buffer(self, descriptor, ready_gate=ready_gate, dtype=dtype,
                             shape=shape, element_shape=element_shape, byte_offset=byte_offset)

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

    def import_acceleration_structure(self, resource):
        """Borrow an original Vulkan TriangleScene, TriangleBLAS or InstanceTLAS.

        This cold operation preserves storage, topology and access identity.
        It does not rebuild or initialize the AS. Prepare acquires the required
        access loan; importing alone does not authorize concurrent mutation.
        Source close leaves retained imports alive. Closing an import still used
        by an instance closes that domain through native-resource retirement.
        Close instances first to preserve their domains. Private input handles
        retire with the import; Program reset retires all borrowers.
        """
        with self._lock:
            self._check()
            if self._acceleration_structure_importer is None:
                raise RuntimeError("JIT AS import requires ti.aot.borrow_runtime()")
            handle = self._acceleration_structure_importer(self._borrower, resource)
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
                    self._domain_owners.clear()
                return
            self.wait()
            # Retire code and module-local storage before external arguments.
            priority = {"prepared": 0, "module": 1, "native_resource": 2, "memory": 3}
            for kind, handle in sorted(self._resources, key=lambda item: priority[item[0]]):
                self._destroy_resource(kind, handle)
            self._api.call(self._api.ti_destroy_runtime, self._handle)
            self._handle = None
            self._domain_owners.clear()

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
    flags = {"storage": 1, "uniform": 2, "vertex": 4, "index": 8, "indirect": 16}
    if not isinstance(usage, (tuple, list)) or not usage or any(
        not isinstance(item, str) or item not in flags for item in usage
    ):
        raise ValueError("Array usage must be a nonempty sequence of storage/uniform/vertex/index/indirect names")
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
        if isinstance(self.dtype, StructLayout):
            self.dtype._check(runtime)
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

    def component_entry_contract(self, name):
        """Read an experimental compiler-produced entry capture projection.

        This is an entry contract, not a complete component-root/instance API.
        Native entry argument validation remains authoritative at preparation.
        """
        from urllib.parse import quote
        from zipfile import ZipFile

        with self._runtime._lock:
            self._check()
            _name(name)
            relative = "component_entries/" + quote(name, safe="-_.") + ".json"
            if self._source_path.is_dir():
                value = json.loads((self._source_path / relative).read_text(encoding="utf8"))
            else:
                with ZipFile(self._source_path) as archive:
                    value = json.loads(archive.read(relative))
            if value.get("version") != 1 or value.get("entry") != name:
                raise ValueError("Unknown or mismatched component entry contract")
            return value

    def ndarray_layout(self, name):
        """Query a named fixed-AoS layout without loading executable entries."""
        runtime = self._runtime
        key = _name(name)
        with runtime._lock:
            self._check()
            layouts = getattr(self, "_ndarray_layouts", None)
            if layouts is None:
                layouts = self._ndarray_layouts = {}
            if key not in layouts:
                handle = runtime._api.call(
                    runtime._api.ti_get_aot_module_ndarray_layout_ext,
                    runtime._handle, self._handle, key,
                )
                layouts[key] = StructLayout._query(self, handle)
            return layouts[key]

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

    def prepare_execution(self, plan, arguments, *, gradients=None, execution_domain=None, mutable_parameters=()):
        """Prepare a named or explicit ExecutionPlan for repeated native replay."""
        from ._plan import ExecutionPlan

        if isinstance(plan, str):
            plan = self.execution_plan(plan)
        if not isinstance(plan, ExecutionPlan):
            raise TypeError("Expected an ExecutionPlan or an exported plan name")
        return plan.prepare(self, arguments, gradients=gradients, execution_domain=execution_domain,
                            mutable_parameters=mutable_parameters)

    def prepare_workspace_pool(self, plan, capacity, *, arguments=None, arrays=None, gradients=None):
        """Share code while preparing independent resources and result leases.

        ``arguments`` are shared external bindings. ``arrays`` maps each private
        binding name to Runtime.ndarray keyword arguments; ``gradients`` maps
        primal binding names to gradient binding names. Private resources in the
        plan are also rebuilt for each slot. No additional device streams are
        created and shared external providers retain their existing contracts.
        """
        from ._workspace import WorkspacePool

        return WorkspacePool(self, plan, capacity, arguments=arguments, arrays=arrays, gradients=gradients)

    def kernel(self, name):
        with self._runtime._lock:
            self._check()
            handle = self._runtime._api.call(self._runtime._api.ti_get_aot_module_kernel, self._handle, _name(name))
            return Kernel(self, handle, name=name)

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
        if isinstance(value.dtype, StructLayout):
            owners.append(value.dtype._module)
        return result
    if isinstance(value, Scalar):
        return value._argument()
    if isinstance(value, int):
        return Scalar("i32", value)._argument()
    if isinstance(value, float):
        return Scalar("f32", value)._argument()
    raise TypeError("AOT arguments must be ndarrays, NativeResource, Scalar, int or float")


def _layout_binding(key, value, *, named=False):
    if isinstance(value, NdArray) and isinstance(value.dtype, StructLayout):
        kind = abi.NamedBinding if named else abi.Binding
        return kind(_name(key) if named else key, value.dtype.version, value.dtype._handle, None)
    return None


def _gradient_binding(runtime, key, primal, gradient, owners, *, named=False):
    if not isinstance(primal, NdArray) or not isinstance(gradient, NdArray):
        raise TypeError("A gradient and its primal must be NdArrays")
    if isinstance(primal.dtype, StructLayout) or isinstance(gradient.dtype, StructLayout):
        raise ValueError("Structure ndarray gradients are not supported")
    kind = abi.NamedBinding if named else abi.Binding
    descriptor = gradient._descriptor(runtime)
    owners.append(gradient._memory)
    return kind(_name(key) if named else key, 0, None, C.pointer(descriptor))


class BoundCall:
    """Fixed Python argument descriptors submitted through the public C API.

    This is not a captured GPU Graph. Bind again to change arguments; calls
    retain modules and allocations until the runtime finishes submitted work.
    """

    def __init__(self, entry, arguments, bindings, owners, value_buffers=()):
        self._entry, self._arguments, self._bindings = entry, arguments, bindings
        self._owners = tuple(dict.fromkeys(owners))
        # Native prepare copies these bytes into its context. Ordinary bound
        # launches keep the immutable payload alive independently of user values.
        self._value_buffers = tuple(value_buffers)
        layout = entry._return_layout
        self._result = C.create_string_buffer(layout.size) if layout is not None and layout.size else None

    def run(self):
        module = self._entry._module
        runtime = module._runtime
        with runtime._lock:
            module._check()
            for owner in self._owners:
                owner._check()
            # Even a failing launch may have submitted earlier native work.
            runtime._pending.update(self._owners)
            if self._result is not None:
                runtime._api.call(
                    runtime._api.ti_launch_kernel_with_result_ext,
                    runtime._handle, self._entry._handle,
                    len(self._arguments), self._arguments,
                    len(self._bindings), self._bindings,
                    self._entry._return_layout.size, self._result,
                )
                # A successful nonempty host result completes prior ordinary
                # work, just like wait(); keep owners on every failure path.
                runtime._pending.clear()
                return self._entry._return_layout.decode(self._result)
            function = (
                runtime._api.ti_launch_compute_graph_with_bindings_ext
                if isinstance(self._entry, Graph)
                else runtime._api.ti_launch_kernel_with_bindings_ext
            )
            runtime._api.call(
                function,
                runtime._handle,
                self._entry._handle,
                len(self._arguments),
                self._arguments,
                len(self._bindings),
                self._bindings,
            )
            if self._entry._return_layout is not None:
                return self._entry._return_layout.decode(b"")

    __call__ = run

    def prepare(self):
        """Build native fixed execution without running the bound kernels."""
        runtime = self._entry._module._runtime
        with runtime._lock:
            for owner in self._owners:
                owner._check()
            graph = isinstance(self._entry, Graph)
            function = runtime._api.ti_prepare_compute_graph_ext if graph else runtime._api.ti_prepare_kernel_ext
            handle = runtime._api.call(
                function,
                runtime._handle,
                self._entry._handle,
                len(self._arguments),
                self._arguments,
                len(self._bindings),
                self._bindings,
            )
            return PreparedCall(runtime, handle, self._owners, return_layout=self._entry._return_layout)


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

    def __init__(self, runtime, handle, owners, *, replay_owners=None, family=None, execution_domain=None,
                 return_layout=None, submission_parameters=None):
        self._owners = owners
        self._replay_owners = replay_owners
        self._family = family
        self._execution_domain = execution_domain
        self._closing = False
        self._close_lock = threading.RLock()
        self._domain_owner_key = None
        # Imports created exclusively by a JIT adapter for this instance.
        # User-owned AOT modules/arrays must never be included here.
        self._instance_imports = ()
        self._return_layout = return_layout
        from ._parameters import SubmissionParameters
        self._submission_parameters = submission_parameters or SubmissionParameters()
        super().__init__(runtime, handle)
        if execution_domain is not None:
            execution_domain._clients.add(runtime)
            self._domain_owner_key = execution_domain._native_handle()
            runtime._domain_owners.setdefault(self._domain_owner_key, {})[handle] = owners

    def _check(self):
        super()._check()
        if self._closing:
            raise RuntimeError("Prepared execution is retiring")

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
            if self._execution_domain is None:
                runtime._pending.add(self)
            runtime._api.call(runtime._api.ti_initialize_prepared_ext, runtime._handle, self._handle)
            if self._execution_domain is not None:
                return
            if self._family is not None:
                self._family.initialized = True
                self._family.initial_owners = ()
            if self._replay_owners is not None:
                self._owners, self._replay_owners = self._replay_owners, None

    def close(self):
        retirement_error = None
        if self._execution_domain is None:
            super().close()
        else:
            runtime = self._runtime
            with self._close_lock:
                with runtime._lock:
                    if runtime._borrower is not None and runtime._borrower.closed:
                        runtime.close()
                    handle = self._handle
                    live = handle is not None and (self._kind, handle) in runtime._resources
                    retirement = None
                    if live:
                        retirement = runtime._api.call(
                            self._execution_domain._functions["instance"], runtime._handle, handle
                        )
                        self._closing = True
                if retirement is not None:
                    # The native token pins only domain/instance state. Waiting
                    # must not hold Runtime or borrowed Program submission locks.
                    try:
                        runtime._api.call(self._execution_domain._functions["retire_instance"], retirement)
                    except Exception as error:
                        # A completed device result may carry a logical error.
                        # The native finish below is the proof that retirement
                        # nevertheless succeeded; backend/cleanup failures do
                        # not pass it and must retain all resource owners.
                        retirement_error = error
                    finally:
                        runtime._api.call(self._execution_domain._functions["release_instance"], retirement)
                with runtime._lock:
                    if runtime._borrower is not None and runtime._borrower.closed:
                        runtime.close()
                    if live and (self._kind, handle) in runtime._resources:
                        try:
                            runtime._api.call(
                                runtime._api.ti_finish_prepared_instance_retirement_ext, runtime._handle, handle
                            )
                        except Exception:
                            if retirement_error is not None:
                                raise retirement_error from None
                            raise
                        runtime._resources.remove((self._kind, handle))
                    # Native retirement and edge removal already prove these
                    # private import handles idle. Ordinary _Resource.close()
                    # would unnecessarily drain every domain again.
                    for resource in self._instance_imports:
                        runtime._destroy_resource(resource._kind, resource._handle)
                        resource._handle = None
                    self._instance_imports = ()
                    owners = runtime._domain_owners.get(self._domain_owner_key)
                    if owners is not None:
                        owners.pop(handle, None)
                        if not owners:
                            runtime._domain_owners.pop(self._domain_owner_key, None)
                    self._handle = None
        self._owners = ()
        self._replay_owners = None
        self._family = None

        if retirement_error is not None:
            raise retirement_error

    def release(self):
        """Drop an independent host handle; retain its budget until domain close.

        This path performs no device wait and is used by garbage collection.
        Use close() at a cold boundary to retire and reclaim one instance.
        """
        if self._execution_domain is None:
            raise ValueError("Host-only release requires an execution-domain instance")
        runtime = self._runtime
        with self._close_lock, runtime._lock:
            if runtime._borrower is not None and runtime._borrower.closed:
                runtime.close()
            if self._handle is not None and (self._kind, self._handle) in runtime._resources:
                runtime._destroy_resource(self._kind, self._handle)
            self._handle = None
            self._instance_imports = ()
        self._owners = ()
        self._replay_owners = None
        self._family = None

    def __del__(self):
        try:
            if self._execution_domain is None:
                self.close()
            else:
                self.release()
        except Exception:
            pass

    def run(self):
        if self._execution_domain is not None:
            completion = self.submit()
            if completion is None:
                raise RuntimeError("Execution domain is full")
            completion.close()
            return
        runtime = self._runtime
        with runtime._lock:
            self._check()
            if self._replay_owners is not None or (self._family is not None and not self._family.initialized):
                self.initialize()
            runtime._pending.add(self)
            runtime._api.call(runtime._api.ti_launch_prepared_ext, runtime._handle, self._handle)

    def submit(self, *, wait_for=(), ready_only=False, values=None, sample_gpu=False):
        """Try one replay, returning a ticket or None if full or input unready.

        Initialize explicitly in the cold phase. Dependencies enqueue device
        waits on their exact prefixes; this method never waits on the host.
        With ready_only=True, a pending dependency also returns None before
        device submission. External native ready gates are always checked.
        """
        from ._domain import Completion

        if type(ready_only) is not bool:
            raise TypeError("ready_only must be bool")
        if type(sample_gpu) is not bool:
            raise TypeError("sample_gpu must be bool")
        parameters = self._submission_parameters.snapshot(values)
        predecessors = tuple(wait_for)
        if self._execution_domain is None:
            if predecessors or ready_only or sample_gpu:
                raise ValueError("wait_for/ready_only/sample_gpu requires an execution-domain binding")
            return self._submit_ordinary()
        runtime = self._runtime
        if any(not isinstance(ticket, Completion) for ticket in predecessors):
            raise TypeError("wait_for must contain Completion tickets")
        if any(ticket._api.library._handle != runtime._api.library._handle for ticket in predecessors):
            raise ValueError("Completion belongs to another native runtime library")
        dependencies = (abi.Handle * len(predecessors))(*[ticket._native_handle() for ticket in predecessors])
        with runtime._lock:
            self._check()
            handle = abi.Handle()
            function = runtime._api.ti_try_submit_prepared_ext
            suffix = ()
            if sample_gpu:
                runtime._api.load_completion_timing()
                function = runtime._api.ti_try_submit_prepared_sampled_ext
                suffix = (ready_only, True, len(parameters), self._submission_parameters.bind(parameters))
            elif self._submission_parameters.names:
                function = runtime._api.ti_try_submit_prepared_parameters_ext
                suffix = (ready_only, len(parameters), self._submission_parameters.bind(parameters))
            elif ready_only:
                runtime._api.load_ready_submit()
                function = runtime._api.ti_try_submit_prepared_ready_ext
            accepted = runtime._api.call(
                function,
                runtime._handle, self._handle, len(dependencies), dependencies, C.byref(handle),
                *suffix,
            )
        if not accepted:
            return None
        try:
            return Completion(runtime._api, handle.value)
        except BaseException:
            if handle.value:
                runtime._api.call(runtime._api.ti_release_completion_ext, handle.value)
            raise

    __call__ = run

    def result(self):
        """Wait and decode this direct kernel binding's latest submitted result.

        Repeated submissions overwrite one frame. Completion tickets do not
        snapshot results. Separate preparations own independent frames; the
        decoded Python value is independent of both frame and module lifetime.
        """
        runtime = self._runtime
        with runtime._lock:
            self._check()
            layout = self._return_layout
            if layout is None:
                raise ValueError("This prepared call has no kernel return value")
            data = C.create_string_buffer(layout.size)
            runtime._api.call(runtime._api.ti_read_prepared_kernel_result_ext,
                              runtime._handle, self._handle, layout.size, data)
            runtime._pending.clear()
            return layout.decode(data)

    def _submit_ordinary(self):
        """Run and return a completion ticket; output storage remains caller-owned."""
        runtime = self._runtime
        with runtime._lock:
            self._check()
            if self._replay_owners is not None or (self._family is not None and not self._family.initialized):
                self.initialize()
            runtime._pending.add(self)
            handle = runtime._api.call(runtime._api.ti_submit_prepared_ext, runtime._handle, self._handle)
            try:
                return SubmissionTicket(runtime._api, handle, (self,))
            except BaseException:
                runtime._api.call(runtime._api.ti_destroy_submission_ext, handle)
                raise

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
    def __init__(self, module, handle, *, name=None):
        self._module, self._handle = module, handle
        self._name = name
        self._layouts = {}
        layout = None if isinstance(self, Graph) else ReturnLayout(module, handle)
        self._return_layout = layout if layout is not None and layout.has_values else None
        self._argument_layout = None if isinstance(self, Graph) else ArgumentLayout(module, handle)
        self._argpack_layouts = None

    def argpack(self, argument, values, *, gradients=None):
        """Create an immutable pack from nested dictionaries and member resources.

        ``argument`` is the logical kernel argument index. Gradient keys are
        tuples of member names, for example ``{("child", "values"): dx}``.
        Reuse this NativeResource across compatible calls; create a new pack to
        change values or resource bindings. Array contents may change in place.
        """
        index = operator.index(argument)
        if self._argument_layout is None or index not in self._argument_layout.pack_indices:
            raise ValueError("Argument is not an exported ArgPack")
        runtime = self._module._runtime
        with runtime._lock:
            self._module._check()
            if self._argpack_layouts is None:
                self._argpack_layouts = {}
            if index not in self._argpack_layouts:
                self._argpack_layouts[index] = ArgPackLayout(self._module, self._handle, index)
            leaves, buffers = self._argpack_layouts[index].encode(values)
            gradient_members = {}
            for path, gradient in (gradients or {}).items():
                key = (path,) if isinstance(path, str) else tuple(path)
                if not key or not all(isinstance(name, str) for name in key) or key in gradient_members:
                    raise ValueError("ArgPack gradients require unique member-name paths")
                gradient_members[key] = gradient
            owners, native = [], []
            for path, names, is_value, value in leaves:
                indices = (abi.UInt * len(path))(*path)
                buffers.append(indices)
                member = abi.ArgPackMember()
                member.path_count, member.path = len(path), indices
                member.argument = value if is_value else _argument(runtime, value, owners)
                binding = None if is_value else _layout_binding(0, value)
                if binding is not None:
                    member.layout_version, member.layout = binding.layout_version, binding.layout
                if names in gradient_members:
                    gradient = gradient_members.pop(names)
                    binding = _gradient_binding(runtime, 0, value, gradient, owners)
                    member.gradient = binding.gradient
                    buffers.append(binding)
                native.append(member)
            if gradient_members:
                raise ValueError("Unknown ArgPack gradient member")
            members = (abi.ArgPackMember * len(native))(*native)
            handle = runtime._api.call(runtime._api.ti_create_kernel_argpack_ext,
                                       runtime._handle, self._handle, index, len(native), members)
            try:
                return NativeResource(runtime, handle, tuple(dict.fromkeys(owners)))
            except BaseException:
                runtime._api.call(runtime._api.ti_destroy_native_resource_ext, runtime._handle, handle)
                raise

    def ndarray_layout(self, argument):
        """Query and cache an exported structure argument's immutable AoS ABI.

        Pass a kernel argument index, or a Graph argument name. Use the result
        as Runtime.ndarray's dtype and pack records using its member offsets.
        """
        runtime = self._module._runtime
        named = isinstance(self, Graph)
        key = _name(argument) if named else operator.index(argument)
        if not named and not 0 <= key <= 0xFFFFFFFF:
            raise ValueError("Argument index must fit uint32")
        with runtime._lock:
            self._module._check()
            if key not in self._layouts:
                query = (
                    runtime._api.ti_get_compute_graph_ndarray_layout_ext if named
                    else runtime._api.ti_get_kernel_ndarray_layout_ext
                )
                handle = runtime._api.call(query, runtime._handle, self._handle, key)
                self._layouts[key] = StructLayout._query(self._module, handle)
            return self._layouts[key]

    def bind(self, *arguments, gradients=None):
        runtime = self._module._runtime
        with runtime._lock:
            self._module._check()
            if len(arguments) != self._argument_layout.count:
                raise ValueError("Argument count does not match the exported kernel ABI")
            owners = [self._module]
            buffers = []
            native = (abi.Argument * len(arguments))(*[
                self._argument_layout.encode(index, value, buffers)
                if index in self._argument_layout.encoders else _argument(runtime, value, owners)
                for index, value in enumerate(arguments)
            ])
            bindings = [
                binding for index, value in enumerate(arguments)
                if (binding := _layout_binding(index, value)) is not None
            ]
            for index, gradient in (gradients or {}).items():
                index = operator.index(index)
                if not 0 <= index < len(arguments) or not isinstance(arguments[index], NdArray):
                    raise ValueError("Gradient indices must refer to ndarray arguments")
                bindings.append(_gradient_binding(runtime, index, arguments[index], gradient, owners))
            return BoundCall(self, native, (abi.Binding * len(bindings))(*bindings), owners, buffers)

    def run(self, *arguments, gradients=None):
        """Run and return Python scalars, tuples or named dictionaries.

        Nonempty host results wait for completion. Void calls remain asynchronous.
        """
        return self.bind(*arguments, gradients=gradients).run()

    def prepare(self, *arguments, gradients=None):
        return self.bind(*arguments, gradients=gradients).prepare()

    __call__ = run


class Graph(Kernel):
    def argpack(self, argument, values, *, gradients=None):
        """Create a named Graph pack using its compiled kernel contract.

        Values and member gradients follow Kernel.argpack(). All uses are
        validated by bind/prepare, including dispatches at other root positions.
        """
        runtime = self._module._runtime
        with runtime._lock:
            self._module._check()
            index = abi.UInt()
            handle = runtime._api.call(runtime._api.ti_get_compute_graph_argpack_kernel_ext,
                                       runtime._handle, self._handle, _name(argument), C.byref(index))
            return Kernel(self._module, handle).argpack(index.value, values, gradients=gradients)

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
            bindings = [
                binding for name, value in arguments.items()
                if (binding := _layout_binding(name, value, named=True)) is not None
            ]
            for name, gradient in (gradients or {}).items():
                if name not in arguments or not isinstance(arguments[name], NdArray):
                    raise ValueError("Gradient names must refer to ndarray arguments")
                bindings.append(_gradient_binding(runtime, name, arguments[name], gradient, owners, named=True))
            return BoundCall(self, native, (abi.NamedBinding * len(bindings))(*bindings), owners)

    def run(self, arguments, *, gradients=None):
        self.bind(arguments, gradients=gradients).run()

    __call__ = run
