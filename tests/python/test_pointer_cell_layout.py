import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("allocation", ["allocated", "deterministic", "fast_reset"])
@pytest.mark.parametrize("dynamic_index", [False, True])
@test_utils.test(arch=[ti.cpu, ti.cuda], require=ti.extension.sparse, offline_cache=False)
def test_pointer_cell_contains_all_children(allocation, dynamic_index):
    ti.cfg.cuda_pointer_deterministic_slot = allocation != "allocated"
    ti.cfg.cuda_pointer_fast_reset = allocation == "fast_reset"
    rows = 32
    vector = ti.Vector.field(3, ti.i32)
    pointer = ti.root.pointer(ti.i, 4)
    for component in range(3):
        pointer.dense(ti.i, 8).place(vector.get_scalar_field(component))

    @ti.kernel
    def fill(offset: ti.i32):
        for i in range(rows):
            if ti.static(dynamic_index):
                for j in range(3):
                    vector[i][j] = offset + 3 * i + j
            else:
                for j in ti.static(range(3)):
                    vector[i][j] = offset + 3 * i + j

    @ti.kernel
    def activate_first_child():
        for i in range(rows):
            vector[i][0] = 7

    @ti.kernel
    def get_addresses(output: ti.types.ndarray()):
        for i in range(rows):
            for j in ti.static(range(3)):
                output[i, j] = ti.get_addr(vector.get_scalar_field(j), i)

    addresses = ti.ndarray(ti.u64, shape=(rows, 3))
    for offset in (1, 101):
        fill(offset)
        np.testing.assert_array_equal(vector.to_numpy(), offset + np.arange(rows * 3).reshape(rows, 3))
        get_addresses(addresses)
        assert np.unique(addresses.to_numpy()).size == rows * 3
        pointer.deactivate_all()
        activate_first_child()
        expected = np.zeros((rows, 3), dtype=np.int32)
        expected[:, 0] = 7
        np.testing.assert_array_equal(vector.to_numpy(), expected)
        pointer.deactivate_all()
