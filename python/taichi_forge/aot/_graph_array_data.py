"""Cold array metadata shared by private snapshots and provider reconstruction."""

from dataclasses import dataclass
from math import prod


def array_resource_fields(*, shape, dtype=None, element_shape=(), layout=None):
    if (dtype is None) == (layout is None) or (layout is not None and element_shape):
        raise ValueError("Array data requires either a named structure layout or a scalar/tensor dtype")
    fields = dict(shape=list(shape))
    if layout is not None:
        fields["layout"] = layout
    else:
        fields.update(dtype=dtype, element_shape=list(element_shape))
    return fields


@dataclass(frozen=True)
class ArrayDataLayout:
    shape: tuple
    element_bytes: int
    dtype: str | None = None
    element_shape: tuple = ()
    layout: str | None = None

    @property
    def byte_size(self):
        return prod(self.shape) * self.element_bytes

    def resource_fields(self):
        return array_resource_fields(
            shape=self.shape, dtype=self.dtype, element_shape=self.element_shape, layout=self.layout)


@dataclass(frozen=True)
class ArraySnapshot:
    owner: object
    name: str
    layout: ArrayDataLayout
