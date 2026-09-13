"""Cold import of external baked OMM data; no baker or device owner."""

import ctypes
from dataclasses import dataclass
from taichi_forge.hardware._micromap import BakedOpacityMicromap


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


@dataclass(frozen=True)
class OptixOpacityMicromap(BakedOpacityMicromap):
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

    _identity_tag = b"forge-optix-baked-omm1"

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
