"""Dynamic loop bounds preserve device-snapshot semantics."""

import numpy as np
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_dynamic_start_shape_end_and_device_snapshot():
    out = ti.ndarray(ti.i32, 37)
    bounds = ti.ndarray(ti.i32, 2)

    @ti.kernel
    def from_shape(start: ti.i32, out: ti.types.ndarray(ti.i32, 1)):
        for i in range(start, out.shape[0]):
            out[i] = i

    @ti.kernel
    def shape_start(out: ti.types.ndarray(ti.i32, 1)):
        for i in range(out.shape[0] - 7, out.shape[0]):
            out[i] = i

    @ti.kernel
    def snapshot(
        n: ti.i32, out: ti.types.ndarray(ti.i32, 1), bounds: ti.types.ndarray(ti.i32, 1)
    ):
        bounds[0] = 3
        bounds[1] = n
        for i in range(bounds[0], bounds[1]):
            out[i] = i
            if i == 3:
                bounds[1] = 3  # The already selected range must not shrink.

    for start in (7, 37, 39):
        out.fill(-91)
        from_shape(start, out)
        expected = np.full(37, -91, np.int32)
        expected[start:] = np.arange(start, 37)
        np.testing.assert_array_equal(out.to_numpy(), expected)
    out.fill(-91)
    shape_start(out)
    expected = np.full(37, -91, np.int32)
    expected[30:] = np.arange(30, 37)
    np.testing.assert_array_equal(out.to_numpy(), expected)
    for end in (37, 9, 3):
        out.fill(-91)
        snapshot(end, out, bounds)
        expected = np.full(37, -91, np.int32)
        expected[3:end] = np.arange(3, end)
        np.testing.assert_array_equal(out.to_numpy(), expected)
        assert bounds[1] == 3
