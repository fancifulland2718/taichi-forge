"""Immutable snapshots of module-owned fixed-AoS layout handles."""

import ctypes as C
from dataclasses import dataclass, field

from . import _c_api as abi


@dataclass(frozen=True)
class StructMember:
    path: str
    dtype: str
    element_shape: tuple
    offset: int
    size: int
    alignment: int


@dataclass(frozen=True)
class StructLayout:
    """Query from Kernel/Graph.ndarray_layout; valid for its owning module.

    Member paths and byte offsets describe the exported device ABI, independent
    of host compiler packing. Metadata is copied once; replay never queries it.
    """

    _module: object = field(repr=False, compare=False)
    _handle: int = field(repr=False)
    version: int
    element_stride: int
    alignment: int
    members: tuple

    def _check(self, runtime):
        if self._module._runtime is not runtime:
            raise ValueError("A structure layout belongs to a different AOT runtime")
        self._module._check()

    def member(self, path):
        for member in self.members:
            if member.path == path:
                return member
        raise KeyError(path)

    @classmethod
    def _query(cls, module, handle):
        from ._runtime import _DTYPES

        runtime = module._runtime
        info = abi.NdArrayLayoutInfo()
        runtime._api.call(runtime._api.ti_get_ndarray_layout_info_ext, runtime._handle, handle, C.byref(info))
        types = {code: name for name, (code, _) in _DTYPES.items()}
        types[7] = "u1"
        members = []
        for index in range(info.member_count):
            item = abi.NdArrayLayoutMember()
            runtime._api.call(
                runtime._api.ti_get_ndarray_layout_member_ext, runtime._handle, handle, index, C.byref(item)
            )
            members.append(
                StructMember(
                    item.path.decode("utf-8"),
                    types[item.scalar_type],
                    tuple(item.element_shape.dims[: item.element_shape.dim_count]),
                    item.offset,
                    item.size,
                    item.alignment,
                )
            )
        return cls(module, handle, info.version, info.element_stride, info.alignment, tuple(members))
