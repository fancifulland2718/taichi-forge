"""Explicit C-compatible launch/SBT data layouts, packed only at preparation.

Resource addresses are supplied by an owner-retaining backend resolver, never
guessed from Python objects. This module performs no GPU allocation or upload.
"""

from dataclasses import asdict, dataclass, field
import struct

from taichi_forge.hardware._shader_artifact import _digest

_SCALARS = {"i32": ("i", 4), "u32": ("I", 4), "i64": ("q", 8), "u64": ("Q", 8), "f32": ("f", 4), "f64": ("d", 8)}
_RESOURCES = frozenset(("buffer", "texture", "scene"))


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class OptixParameterField:
    """One scalar/array or managed resource in an explicit byte layout.

    Resource fields have dtype u64 and count 1. ``access`` describes the full
    shader use, including indirect/SBT references; it is not inferred from PTX.
    """

    name: str
    offset: int
    dtype: str = "u64"
    count: int = 1
    kind: str = "value"
    access: str = "read"
    alignment: int | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("parameter name must be nonempty")
        _integer(self.offset, "offset")
        _integer(self.count, "count", 1)
        if self.dtype not in _SCALARS:
            raise ValueError("unsupported parameter scalar dtype")
        natural = _SCALARS[self.dtype][1]
        alignment = natural if self.alignment is None else _integer(self.alignment, "alignment", 1)
        if alignment < natural or alignment & (alignment - 1) or self.offset % alignment:
            raise ValueError("parameter offset/alignment must respect the scalar ABI")
        if self.kind != "value" and self.kind not in _RESOURCES:
            raise ValueError("unsupported parameter resource kind")
        if self.kind in _RESOURCES and (self.dtype != "u64" or self.count != 1):
            raise ValueError("resource parameters require one u64 ABI slot")
        if self.access not in ("read", "write", "read_write"):
            raise ValueError("parameter access must be read, write or read_write")
        if self.kind != "buffer" and self.access != "read":
            raise ValueError("only buffer parameters support write access")
        object.__setattr__(self, "alignment", alignment)

    @property
    def size(self):
        return _SCALARS[self.dtype][1] * self.count


@dataclass(frozen=True)
class OptixParameterLayout:
    """Frozen little-endian layout with explicit size and zero-filled padding.

    It can describe launch parameters or a hit-record payload. Native SBT
    headers/record stride belong to the provider and are not included here.
    """

    size: int
    fields: tuple[OptixParameterField, ...] = ()
    alignment: int = 8
    layout_id: str = field(init=False)

    def __post_init__(self):
        _integer(self.size, "layout size")
        _integer(self.alignment, "layout alignment", 1)
        if self.alignment & (self.alignment - 1) or self.size % self.alignment:
            raise ValueError("layout alignment must be power-of-two and divide size")
        fields = tuple(self.fields)
        if any(not isinstance(item, OptixParameterField) for item in fields):
            raise TypeError("fields must be OptixParameterField values")
        fields = tuple(sorted(fields, key=lambda item: item.offset))
        end, names = 0, set()
        for item in fields:
            if item.name in names or item.offset < end or item.offset + item.size > self.size:
                raise ValueError("parameter fields overlap, repeat names or exceed layout size")
            if self.alignment < item.alignment:
                raise ValueError("layout alignment is smaller than a field alignment")
            names.add(item.name)
            end = item.offset + item.size
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "layout_id", "optix-params:" + _digest(self.to_dict()))

    def to_dict(self):
        return {"size": self.size, "alignment": self.alignment, "fields": tuple(asdict(item) for item in self.fields)}

    def _pack(self, values, resolve_resource):
        if set(values) != {item.name for item in self.fields}:
            raise ValueError("parameter bindings must exactly match the frozen layout")
        result = bytearray(self.size)
        for item in self.fields:
            value = values[item.name]
            if item.kind in _RESOURCES:
                value = resolve_resource(item, value)
            values_to_pack = (value,) if item.count == 1 else tuple(value)
            if len(values_to_pack) != item.count:
                raise ValueError(f"parameter {item.name!r} has incorrect element count")
            try:
                struct.pack_into("<" + _SCALARS[item.dtype][0] * item.count, result, item.offset, *values_to_pack)
            except (struct.error, TypeError, OverflowError) as exc:
                raise ValueError(f"parameter {item.name!r} does not fit its declared ABI") from exc
        return bytes(result)
