from taichi_forge._lib import core as _ti_core
from taichi_forge.lang import impl
from taichi_forge.lang._storage_view import (
    DenseNdarrayView,
    StorageDescription,
)


_DLCPU = 1
_DLCUDA = 2
_DLCUDA_HOST = 3
_DLCUDA_MANAGED = 13


class ExternalDenseView(DenseNdarrayView):
    """Non-owning dense storage imported from an external provider."""

    __slots__ = ()

    @property
    def provider(self):
        return self._source.provider

    @property
    def device(self):
        return self._source.device_type, self._source.device_id

    @property
    def allocation_bytes(self):
        return self._source.allocation_bytes

    @property
    def closed(self):
        return self._source.closed

    def close(self):
        self._source.close()

    def __enter__(self):
        if self.closed:
            raise RuntimeError("external dense view is already closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def _current_program():
    program = impl.get_runtime().prog
    if program is None:
        raise RuntimeError("external storage import requires ti.init()")
    return program


def _normalize_dlpack_device(source):
    device_query = getattr(source, "__dlpack_device__", None)
    if device_query is None:
        raise TypeError("source does not implement __dlpack_device__")
    device = tuple(device_query())
    if len(device) != 2:
        raise BufferError("__dlpack_device__() must return (device_type, device_id)")
    try:
        return int(device[0]), int(device[1])
    except (TypeError, ValueError) as exc:
        raise BufferError("DLPack device values must be integers") from exc


def _validate_device_for_program(device_type):
    arch = impl.current_cfg().arch
    if arch in (_ti_core.Arch.x64, _ti_core.Arch.arm64):
        if device_type not in (_DLCPU, _DLCUDA_HOST):
            raise BufferError("DLPack device is incompatible with the CPU backend")
        return
    if arch == _ti_core.Arch.cuda:
        if device_type not in (_DLCUDA, _DLCUDA_MANAGED):
            raise BufferError("DLPack device is incompatible with the CUDA backend")
        return
    raise BufferError(
        "the current backend cannot import DLPack storage without copying"
    )


def _request_capsule(source, device_type):
    exporter = getattr(source, "__dlpack__", None)
    if exporter is None:
        raise TypeError("source does not implement __dlpack__")

    stream = 1 if device_type in (_DLCUDA, _DLCUDA_MANAGED) else None
    modern_kwargs = {
        "max_version": (1, 0),
        "copy": False,
    }
    if stream is not None:
        modern_kwargs["stream"] = stream
    try:
        return exporter(**modern_kwargs)
    except TypeError:
        pass

    if stream is not None:
        try:
            return exporter(stream=stream)
        except TypeError:
            pass
    return exporter()


def _from_dlpack(source, *, element_shape=(), layout="aos", access="readwrite"):
    device_type, _ = _normalize_dlpack_device(source)
    _validate_device_for_program(device_type)
    capsule = _request_capsule(source, device_type)
    native = _ti_core._import_dlpack_capsule(
        _current_program(),
        capsule,
        tuple(int(extent) for extent in element_shape),
        layout,
        access,
    )
    description = StorageDescription(native.description)
    if not description.supported:
        native.close()
        raise BufferError(
            "DLPack storage cannot form an executable dense view: "
            f"{description.failure_reason}"
        )
    return ExternalDenseView(native, description)
    if not description.properties["ndarray_abi_compatible"]:
        native.close()
        raise BufferError(
            "DLPack kernel bindings currently require compact AOS storage"
        )


def from_dlpack(
    source,
    *,
    element_shape=(),
    access="readwrite",
    copy=False,
):
    """Import a DLPack producer as a strict zero-copy dense storage view.

    The current executable contract accepts CPU storage on CPU and CUDA storage
    on CUDA. Cross-device imports and materialization are intentionally
    rejected. The returned view keeps the producer allocation alive until
    ``close()`` or runtime finalization.
    """

    if copy is not False:
        raise ValueError("Forge from_dlpack() currently requires copy=False")
    view = _from_dlpack(
        source,
        element_shape=tuple(int(extent) for extent in element_shape),
        access=access,
    )
    return view


def _legacy_external_view(source, *, element_shape=(), layout="aos"):
    """Best-effort adapter used behind the historical external-array API."""

    if layout != "aos":
        return None
    if not hasattr(source, "__dlpack__") or not hasattr(
        source, "__dlpack_device__"
    ):
        return None
    try:
        return _from_dlpack(
            source,
            element_shape=element_shape,
            layout=layout,
            access="readwrite",
        )
    except (BufferError, TypeError, ValueError, RuntimeError):
        return None


def capabilities():
    """Return the strict provider capabilities for the active runtime."""

    arch = impl.current_cfg().arch
    if arch in (_ti_core.Arch.x64, _ti_core.Arch.arm64):
        devices = ("cpu", "cuda_host")
    elif arch == _ti_core.Arch.cuda:
        devices = ("cuda", "cuda_managed")
    else:
        devices = ()
    return {
        "schema_version": 1,
        "provider": "dlpack",
        "backend": _ti_core.arch_name(arch),
        "zero_copy": bool(devices),
        "devices": devices,
        "copy_fallback": False,
        "layouts": ("compact", "aos"),
        "access": ("readwrite",),
    }
