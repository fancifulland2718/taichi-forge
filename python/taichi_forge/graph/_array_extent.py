"""Declared array shape dependencies for ordinary Graph-private storage."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ArrayExtent:
    """Use one logical index extent of a normal Field or whole Ndarray.

    Ordinary Graphs specialize storage to this source's current shape. Component
    capture keeps the declared resource reference, so consumer preparation uses
    the bound array's extent instead of the producer's integer. Element axes of
    vectors/matrices are not index axes.
    """

    array: object
    axis: int = 0

    def __post_init__(self):
        from taichi_forge.lang.field import Field
        from taichi_forge.lang._ndarray import Ndarray

        if not isinstance(self.array, (Field, Ndarray)):
            raise TypeError("ArrayExtent requires a normal Field or whole Ndarray")
        if type(self.axis) is not int or not 0 <= self.axis < len(self.array.shape):
            raise ValueError("ArrayExtent axis is outside the logical index shape")


def private_array_shape(shape, capture=None):
    if isinstance(shape, (int, ArrayExtent)):
        shape = (shape,)
    concrete, bindings = [], []
    for value in shape:
        binding = None
        if isinstance(value, ArrayExtent):
            if capture is not None:
                binding = (capture.array_extent(value.array), value.axis)
            value = value.array.shape[value.axis]
        concrete.append(int(value))
        bindings.append(binding)
    return tuple(concrete), tuple(bindings) if any(bindings) else ()
