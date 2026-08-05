"""Small recordable kernels used by LinearOperator compositions."""

from taichi_forge.lang.kernel_impl import kernel
from taichi_forge.types import f32, i32, ndarray_type


@kernel
def scale_f32(
    values: ndarray_type.ndarray(dtype=f32, ndim=1),
    scale: f32,
    size: i32,
):
    for index in range(size):
        values[index] *= scale


@kernel
def add_f32(
    addend: ndarray_type.ndarray(dtype=f32, ndim=1),
    output: ndarray_type.ndarray(dtype=f32, ndim=1),
    size: i32,
):
    for index in range(size):
        output[index] += addend[index]
