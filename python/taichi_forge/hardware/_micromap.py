"""Immutable host input shared by backend-specific baked micromap adapters."""

from dataclasses import dataclass, field
import hashlib
import struct


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
class BakedOpacityMicromap:
    """Backend-native packed input; interpretation belongs to the backend adapter."""

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
        digest = hashlib.sha256(self._identity_tag)
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
