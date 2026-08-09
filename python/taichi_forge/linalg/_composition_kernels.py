"""Small recordable kernels used by LinearOperator compositions."""

from taichi_forge.lang.kernel_impl import kernel
from taichi_forge.types import f32, i32, ndarray_type


@kernel
def identity_f32(
    active_size: i32,
    operator_data: ndarray_type.ndarray(dtype=i32, ndim=1),
    input: ndarray_type.ndarray(dtype=f32, ndim=1),
    output: ndarray_type.ndarray(dtype=f32, ndim=1),
):
    for index in range(active_size):
        output[index] = input[index]


@kernel
def scale_f32(
    values: ndarray_type.ndarray(dtype=f32, ndim=1),
    scale: f32,
    size: i32,
):
    for index in range(size):
        values[index] *= scale


@kernel
def axpby_f32(
    addend: ndarray_type.ndarray(dtype=f32, ndim=1),
    output: ndarray_type.ndarray(dtype=f32, ndim=1),
    output_scale: f32,
    addend_scale: f32,
    size: i32,
):
    for index in range(size):
        output[index] = (
            output_scale * output[index] + addend_scale * addend[index]
        )
