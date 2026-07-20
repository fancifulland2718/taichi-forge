from taichi_forge.lang import ops
from taichi_forge.lang.kernel_impl import func
from taichi_forge.types.annotations import template
from taichi_forge.types.primitive_types import f32, i32


@func
def insert_triplet_i32_storage(storage: template(), row, column, value):
    slot = ops.atomic_add(storage[0], 1)
    capacity = storage[1]
    if slot < capacity:
        base = 2 + slot * 3
        storage[base] = ops.cast(row, i32)
        storage[base + 1] = ops.cast(column, i32)
        storage[base + 2] = ops.bit_cast(ops.cast(value, f32), i32)
    else:
        # Preserve a bounded overflow sentinel even when many invocations race.
        ops.atomic_min(storage[0], capacity + 1)
