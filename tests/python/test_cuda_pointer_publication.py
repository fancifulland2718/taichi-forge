import numpy as np

import taichi_forge as ti
from tests import test_utils


@test_utils.test(
    arch=ti.cuda,
    require=ti.extension.sparse,
    offline_cache=False,
    cuda_sparse_pool_auto_size=True,
    cuda_sparse_per_snode_pool=True,
    cuda_pointer_deterministic_slot=True,
)
def test_cuda_pointer_publication_with_divergent_writers():
    blocks = 16
    block_size = 8
    size = blocks * block_size
    value = ti.field(ti.i32)
    pointer = ti.root.pointer(ti.ijk, (blocks, blocks, blocks))
    pointer.dense(ti.ijk, (block_size, block_size, block_size)).place(value)

    @ti.kernel
    def fill(epoch: ti.i32, inactive_class: ti.i32):
        for i, j, k in ti.ndrange(size, size, size):
            bi, bj, bk = i // block_size, j // block_size, k // block_size
            if (bi * 13 + bj * 7 + bk) % 11 != inactive_class:
                linear = (i * size + j) * size + k
                state = ti.cast(linear + 1 + epoch * 97, ti.u32)
                rounds = 1 + ((linear * 13 ^ (linear >> 4)) & 15)
                # Different lanes reach adjacent pointer slots at different
                # times. Every round contributes to the exact output oracle.
                for step in range(rounds):
                    state = state ^ (state << 13)
                    state = state ^ (state >> 17)
                    state = state ^ (state << 5)
                value[i, j, k] = ti.cast(state & ti.u32(0x3FFFFFFF), ti.i32) + 1

    @ti.kernel
    def read_activity(output: ti.types.ndarray(dtype=ti.i32, ndim=3)):
        for i, j, k in ti.ndrange(blocks, blocks, blocks):
            output[i, j, k] = ti.is_active(pointer, [i, j, k])

    block_indices = np.indices((blocks, blocks, blocks), dtype=np.int32)
    classes = (block_indices[0] * 13 + block_indices[1] * 7 + block_indices[2]) % 11
    linear = np.arange(size**3, dtype=np.uint32)
    rounds = 1 + ((linear * np.uint32(13) ^ (linear >> np.uint32(4))) & np.uint32(15))
    activity = np.empty((blocks, blocks, blocks), dtype=np.int32)

    def expected_values(epoch, inactive_class):
        state = linear + np.uint32(1 + epoch * 97)
        for step in range(16):
            updated = state ^ (state << np.uint32(13))
            updated ^= updated >> np.uint32(17)
            updated ^= updated << np.uint32(5)
            state = np.where(rounds > step, updated, state)
        result = ((state & np.uint32(0x3FFFFFFF)) + np.uint32(1)).astype(np.int32)
        active = classes != inactive_class
        leaf_active = active.repeat(block_size, 0).repeat(block_size, 1).repeat(block_size, 2)
        return np.where(leaf_active, result.reshape((size, size, size)), 0)

    def check_inactive():
        read_activity(activity)
        np.testing.assert_array_equal(activity, np.zeros_like(activity))
        # A dense-shaped read must return ambient zero without activating slots.
        assert not np.any(value.to_numpy())
        read_activity(activity)
        np.testing.assert_array_equal(activity, np.zeros_like(activity))

    def check_fill(epoch, inactive_class):
        fill(epoch, inactive_class)
        np.testing.assert_array_equal(value.to_numpy(), expected_values(epoch, inactive_class))
        read_activity(activity)
        np.testing.assert_array_equal(activity, (classes != inactive_class).astype(np.int32))

    check_inactive()
    check_fill(1, 0)  # Cold publication across many neighboring pointer slots.
    check_fill(2, 0)  # Published slots take the topology-stable activation path.
    pointer.deactivate_all()
    check_inactive()
    # Change the inactive set so reset also clears previously written children.
    check_fill(3, 1)
