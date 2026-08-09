"""Recordable kernels for trusted scalar and small-block inverse actions."""

from taichi_forge.lang.kernel_impl import kernel
from taichi_forge.lang.impl import static
from taichi_forge.types import f32, i32, ndarray_type


@kernel
def apply_inverse_blocks_f32(
    active_size: i32,
    topology: ndarray_type.ndarray(dtype=i32, ndim=1),
    inverse_blocks: ndarray_type.ndarray(dtype=f32, ndim=1),
    input: ndarray_type.ndarray(dtype=f32, ndim=1),
    output: ndarray_type.ndarray(dtype=f32, ndim=1),
):
    block_size = topology[2 * active_size]
    for index in range(active_size):
        matrix_offset = topology[2 * index]
        vector_offset = topology[2 * index + 1]
        value = 0.0
        for column in static(range(4)):
            if column < block_size:
                value += (
                    inverse_blocks[matrix_offset + column]
                    * input[vector_offset + column]
                )
        output[index] = value
