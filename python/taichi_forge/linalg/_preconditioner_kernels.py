"""Recordable kernels for trusted scalar and small-block inverse actions."""

from taichi_forge.lang._ndrange import ndrange
from taichi_forge.lang.impl import static
from taichi_forge.lang.kernel_impl import func, kernel
from taichi_forge.lang.matrix import Matrix
from taichi_forge.math.mathimpl import isinf, isnan
from taichi_forge.types import f32, i32, ndarray_type
from taichi_forge.types.annotations import template


@func
def _build_inverse_block_f32(
    block_size: template(),
    block: i32,
    blocks: template(),
    inverse_blocks: template(),
    status: template(),
    regularization: f32,
    pivot_tolerance: f32,
):
    augmented = Matrix.zero(f32, block_size, block_size * 2)
    source_offset = block * static(block_size * block_size)
    state = 0
    matrix_scale = 0.0
    for row, column in static(ndrange(block_size, block_size)):
        value = blocks[source_offset + row * block_size + column]
        if row == column:
            value += regularization
        if isnan(value) or isinf(value):
            state = 1
        matrix_scale = max(matrix_scale, abs(value))
        augmented[row, column] = value
        augmented[row, block_size + column] = 1.0 if row == column else 0.0

    if matrix_scale == 0.0 and state == 0:
        state = 2
    pivot_threshold = pivot_tolerance * matrix_scale

    for pivot in static(range(block_size)):
        selected = pivot
        magnitude = abs(augmented[pivot, pivot])
        for candidate in static(range(pivot + 1, block_size)):
            candidate_magnitude = abs(augmented[candidate, pivot])
            if candidate_magnitude > magnitude:
                magnitude = candidate_magnitude
                selected = candidate
        if isnan(magnitude) or isinf(magnitude):
            state = 1
        elif magnitude <= pivot_threshold and state == 0:
            state = 2
        if state == 0:
            for column in static(range(block_size * 2)):
                swap = augmented[pivot, column]
                augmented[pivot, column] = augmented[selected, column]
                augmented[selected, column] = swap
            pivot_value = augmented[pivot, pivot]
            for column in static(range(block_size * 2)):
                augmented[pivot, column] /= pivot_value
            for row in static(range(block_size)):
                if row != pivot:
                    factor = augmented[row, pivot]
                    for column in static(range(block_size * 2)):
                        augmented[row, column] -= factor * augmented[pivot, column]

    status[block] = state
    for row, column in static(ndrange(block_size, block_size)):
        value = augmented[row, block_size + column] if state == 0 else 0.0
        if isnan(value) or isinf(value):
            value = 0.0
            status[block] = 1
        inverse_blocks[source_offset + row * block_size + column] = value


@kernel
def build_inverse_blocks_1_f32(
    block_count: i32,
    regularization: f32,
    pivot_tolerance: f32,
    blocks: ndarray_type.ndarray(dtype=f32, ndim=1),
    inverse_blocks: ndarray_type.ndarray(dtype=f32, ndim=1),
    status: ndarray_type.ndarray(dtype=i32, ndim=1),
):
    for block in range(block_count):
        _build_inverse_block_f32(
            1,
            block,
            blocks,
            inverse_blocks,
            status,
            regularization,
            pivot_tolerance,
        )


@kernel
def build_inverse_blocks_2_f32(
    block_count: i32,
    regularization: f32,
    pivot_tolerance: f32,
    blocks: ndarray_type.ndarray(dtype=f32, ndim=1),
    inverse_blocks: ndarray_type.ndarray(dtype=f32, ndim=1),
    status: ndarray_type.ndarray(dtype=i32, ndim=1),
):
    for block in range(block_count):
        _build_inverse_block_f32(
            2,
            block,
            blocks,
            inverse_blocks,
            status,
            regularization,
            pivot_tolerance,
        )


@kernel
def build_inverse_blocks_3_f32(
    block_count: i32,
    regularization: f32,
    pivot_tolerance: f32,
    blocks: ndarray_type.ndarray(dtype=f32, ndim=1),
    inverse_blocks: ndarray_type.ndarray(dtype=f32, ndim=1),
    status: ndarray_type.ndarray(dtype=i32, ndim=1),
):
    for block in range(block_count):
        _build_inverse_block_f32(
            3,
            block,
            blocks,
            inverse_blocks,
            status,
            regularization,
            pivot_tolerance,
        )


@kernel
def build_inverse_blocks_4_f32(
    block_count: i32,
    regularization: f32,
    pivot_tolerance: f32,
    blocks: ndarray_type.ndarray(dtype=f32, ndim=1),
    inverse_blocks: ndarray_type.ndarray(dtype=f32, ndim=1),
    status: ndarray_type.ndarray(dtype=i32, ndim=1),
):
    for block in range(block_count):
        _build_inverse_block_f32(
            4,
            block,
            blocks,
            inverse_blocks,
            status,
            regularization,
            pivot_tolerance,
        )


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
