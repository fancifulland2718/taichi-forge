import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("dtype,numpy_dtype,bits", [(ti.u32, np.uint32, 32), (ti.u64, np.uint64, 64)])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_unsigned_negation_and_subtraction(dtype, numpy_dtype, bits):
    mask = (1 << bits) - 1
    host = np.asarray([0, 1, (1 << (bits - 1)) - 1, 1 << (bits - 1), mask], numpy_dtype)
    source = ti.ndarray(dtype, len(host))
    output = ti.ndarray(dtype, (len(host), 3))
    source.from_numpy(host)

    @ti.kernel
    def negate(values: ti.types.ndarray(dtype, 1), result: ti.types.ndarray(dtype, 2)):
        for i in range(values.shape[0]):
            result[i, 0] = -values[i]
            result[i, 1] = ti.cast(3, dtype) - values[i]
            result[i, 2] = -ti.Vector([values[i], values[i]])[0]

    negate(source, output)
    expected = np.asarray([[(-int(v)) & mask, (3 - int(v)) & mask, (-int(v)) & mask] for v in host], numpy_dtype)
    np.testing.assert_array_equal(output.to_numpy(), expected)
