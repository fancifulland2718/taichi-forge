"""Recordable kernels for trusted scalar and small-block inverse actions."""

from taichi_forge.lang.kernel_impl import kernel
from taichi_forge.types import f32, i32, ndarray_type


@kernel
def apply_inverse_blocks_1_f32(
    active_size: i32,
    topology: ndarray_type.ndarray(dtype=i32, ndim=1),
    inverse_blocks: ndarray_type.ndarray(dtype=f32, ndim=1),
    input: ndarray_type.ndarray(dtype=f32, ndim=1),
    output: ndarray_type.ndarray(dtype=f32, ndim=1),
):
    for index in range(active_size):
        output[index] = inverse_blocks[index] * input[index]


@kernel
def apply_inverse_blocks_2_f32(
    active_size: i32,
    topology: ndarray_type.ndarray(dtype=i32, ndim=1),
    inverse_blocks: ndarray_type.ndarray(dtype=f32, ndim=1),
    input: ndarray_type.ndarray(dtype=f32, ndim=1),
    output: ndarray_type.ndarray(dtype=f32, ndim=1),
):
    for index in range(active_size):
        block = index // 2
        row = index % 2
        matrix_offset = block * 4 + row * 2
        vector_offset = block * 2
        output[index] = (
            inverse_blocks[matrix_offset] * input[vector_offset]
            + inverse_blocks[matrix_offset + 1] * input[vector_offset + 1]
        )


@kernel
def apply_inverse_blocks_3_f32(
    active_size: i32,
    topology: ndarray_type.ndarray(dtype=i32, ndim=1),
    inverse_blocks: ndarray_type.ndarray(dtype=f32, ndim=1),
    input: ndarray_type.ndarray(dtype=f32, ndim=1),
    output: ndarray_type.ndarray(dtype=f32, ndim=1),
):
    for index in range(active_size):
        block = index // 3
        row = index % 3
        matrix_offset = block * 9 + row * 3
        vector_offset = block * 3
        output[index] = (
            inverse_blocks[matrix_offset] * input[vector_offset]
            + inverse_blocks[matrix_offset + 1] * input[vector_offset + 1]
            + inverse_blocks[matrix_offset + 2] * input[vector_offset + 2]
        )


@kernel
def apply_inverse_blocks_4_f32(
    active_size: i32,
    topology: ndarray_type.ndarray(dtype=i32, ndim=1),
    inverse_blocks: ndarray_type.ndarray(dtype=f32, ndim=1),
    input: ndarray_type.ndarray(dtype=f32, ndim=1),
    output: ndarray_type.ndarray(dtype=f32, ndim=1),
):
    for index in range(active_size):
        block = index // 4
        row = index % 4
        matrix_offset = block * 16 + row * 4
        vector_offset = block * 4
        output[index] = (
            inverse_blocks[matrix_offset] * input[vector_offset]
            + inverse_blocks[matrix_offset + 1] * input[vector_offset + 1]
            + inverse_blocks[matrix_offset + 2] * input[vector_offset + 2]
            + inverse_blocks[matrix_offset + 3] * input[vector_offset + 3]
        )
