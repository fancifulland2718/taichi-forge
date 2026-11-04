"""Kernel metadata snapshots must not freeze array contents or later calls."""

import numpy as np
import taichi_forge as ti

from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_aliased_payload_and_changed_argument_metadata():
    @ti.kernel
    def update(
        a: ti.types.ndarray(ti.i32, 2), b: ti.types.ndarray(ti.i32, 2), repeats: ti.i32
    ):
        for i, j in a:
            for _ in range(repeats):
                a[i, j] += 1
                b[i, j] += a[i, j]

    for shape, repeats in (((3, 5), 1), ((7, 2), 3), ((3, 5), 2), ((2, 4), 0)):
        data = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
        array = ti.ndarray(ti.i32, shape=shape)
        array.from_numpy(data)
        # These arguments intentionally alias: only ABI metadata is immutable.
        update(array, array, repeats)
        expected = data * (2**repeats) + 2 * (2**repeats - 1)
        np.testing.assert_array_equal(array.to_numpy(), expected)
    assert len(update._primal.compiled_kernels) == 1
