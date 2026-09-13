"""Cold import of external baked OMM data; no baker or device owner."""

import ctypes
from dataclasses import dataclass, field
import hashlib
import struct


class _MicromapEntry(ctypes.Structure):
    _fields_ = [
        ("byte_offset", ctypes.c_uint32),
        ("subdivision_level", ctypes.c_uint16),
        ("format", ctypes.c_uint16),
    ]


class _MicromapDesc(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("micromap_count", ctypes.c_uint32),
        ("data_size", ctypes.c_uint64),
        ("data", ctypes.c_void_p),
        ("entries", ctypes.POINTER(_MicromapEntry)),
        ("triangle_count", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("triangle_indices", ctypes.POINTER(ctypes.c_int32)),
    ]


class _MicromapMemory(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("array_bytes", ctypes.c_uint64),
        ("index_bytes", ctypes.c_uint64),
        ("build_temporary_bytes", ctypes.c_uint64),
    ]


def _packed(value, layout, name):
    try:
        view = memoryview(value)
        if (
            layout == "<i"
            and view.itemsize != 1
            and (
                view.itemsize != 4
                or view.format not in ("i", "l", "<i", "<l", "=i", "=l")
            )
        ):
            raise ValueError(
                "OMM typed indices must use little-endian int32; convert before import"
            )
        return view.tobytes()
    except TypeError:
        pass
    try:
        if layout == "<i":
            return b"".join(struct.pack(layout, item) for item in value)
        return b"".join(struct.pack(layout, *item) for item in value)
    except (TypeError, ValueError, struct.error) as exc:
        raise ValueError(f"Invalid packed OMM {name}") from exc


@dataclass(frozen=True)
class OptixOpacityMicromap:
    """Baked OMM input for ``provider.triangle_gas(..., opacity_micromap=...)``.

    ``data`` is a bytes-like bitstream in native OptiX microtriangle order.
    ``descriptors`` is packed little-endian ``(u32 offset, u16 level, u16 format)``
    or an iterable of those triples; format 1 means two-state, 2 means four-state.
    ``triangle_indices`` is optional packed little-endian int32 data or an iterable
    of indices. None means one micromap per triangle. -1 .. -4 are native uniform
    transparent, opaque, unknown-transparent and unknown-opaque special indices.

    Inputs are copied into immutable host bytes. The GAS imports them once and
    owns device storage. The caller/baker owns classification accuracy and its
    correspondence with geometry, UVs, filtering and cutoff. No texture scan or
    classification validation is performed by Forge.
    """

    data: bytes = field(repr=False)
    descriptors: object = field(repr=False)
    triangle_indices: object = field(default=None, repr=False)
    fingerprint: str = field(init=False)

    def __post_init__(self):
        data = memoryview(self.data).tobytes()
        descriptors = _packed(self.descriptors, "<IHH", "descriptors")
        indices = (
            None
            if self.triangle_indices is None
            else _packed(self.triangle_indices, "<i", "triangle indices")
        )
        if bool(data) != bool(descriptors) or len(descriptors) % 8:
            raise ValueError("OMM requires data and complete 8-byte descriptors")
        if indices is not None and (not indices or len(indices) % 4):
            raise ValueError("OMM triangle indices must be packed int32")
        if not descriptors and indices is None:
            raise ValueError("An empty OMM array requires predefined triangle indices")
        count = len(descriptors) // 8 if indices is None else len(indices) // 4
        if max(count, len(descriptors) // 8) > 0xFFFFFFFF:
            raise ValueError("OMM counts must fit uint32")
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "descriptors", descriptors)
        object.__setattr__(self, "triangle_indices", indices)
        digest = hashlib.sha256(b"forge-optix-baked-omm1")
        digest.update(
            struct.pack("<QQQ", len(data), len(descriptors), len(indices or b""))
        )
        for value in (data, descriptors, indices or b""):
            digest.update(value)
        object.__setattr__(self, "fingerprint", digest.hexdigest())

    @property
    def triangle_count(self):
        return (
            len(self.descriptors) // 8
            if self.triangle_indices is None
            else len(self.triangle_indices) // 4
        )

    def _native(self):
        # ctypes pointers retain the immutable bytes for the synchronous import.
        data = ctypes.c_char_p(self.data)
        entries = ctypes.c_char_p(self.descriptors)
        indices = ctypes.c_char_p(self.triangle_indices)
        desc = _MicromapDesc(
            ctypes.sizeof(_MicromapDesc),
            len(self.descriptors) // 8,
            len(self.data),
            ctypes.cast(data, ctypes.c_void_p),
            ctypes.cast(entries, ctypes.POINTER(_MicromapEntry)),
            self.triangle_count,
            0,
            ctypes.cast(indices, ctypes.POINTER(ctypes.c_int32)),
        )
        return desc, (self, data, entries, indices)
