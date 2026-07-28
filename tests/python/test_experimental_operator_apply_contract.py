import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


def _array(dtype, values):
    numpy_dtype = np.float64 if dtype == ti.f64 else np.float32
    values = np.asarray(values, dtype=numpy_dtype)
    result = ti.ndarray(dtype, shape=values.size)
    result.from_numpy(values)
    return result


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_generalized_apply_cpu_contract_and_aliasing():
    for dtype in (ti.f32, ti.f64):
        operator = ti.linalg.identity(4, dtype=dtype)
        input_values = np.asarray([1.0, -2.0, 0.5, 3.0])
        addend_values = np.asarray([-0.5, 1.0, 4.0, -2.0])
        input_array = _array(dtype, input_values)
        addend = _array(dtype, addend_values)

        result = operator.apply(
            input_array, alpha=1.75, beta=-0.25, addend=addend
        )
        np.testing.assert_allclose(
            result.to_numpy(),
            1.75 * input_values - 0.25 * addend_values,
            rtol=2e-6 if dtype == ti.f32 else 1e-13,
            atol=2e-6 if dtype == ti.f32 else 1e-13,
        )

        accumulator = _array(dtype, addend_values)
        returned = operator.apply(
            input_array,
            out=accumulator,
            alpha=2.0,
            beta=0.5,
            addend=accumulator,
        )
        assert returned is accumulator
        np.testing.assert_allclose(
            accumulator.to_numpy(),
            2.0 * input_values + 0.5 * addend_values,
            rtol=2e-6 if dtype == ti.f32 else 1e-13,
            atol=2e-6 if dtype == ti.f32 else 1e-13,
        )

        poison = _array(dtype, [np.nan, np.nan, np.nan, np.nan])
        np.testing.assert_allclose(
            operator.apply(
                input_array, alpha=-0.75, beta=0.0, addend=poison
            ).to_numpy(),
            -0.75 * input_values,
            rtol=2e-6 if dtype == ti.f32 else 1e-13,
            atol=2e-6 if dtype == ti.f32 else 1e-13,
        )

        with pytest.raises(RuntimeError, match="requires addend"):
            operator.apply(input_array, beta=1.0)
        with pytest.raises(RuntimeError, match="input/output aliasing"):
            operator.apply(input_array, out=input_array, alpha=0.0)
        with pytest.raises(RuntimeError, match="finite"):
            operator.apply(input_array, alpha=np.nan)

        stats = operator.statistics()
        assert stats["generalized_lowerings"] == 3
        assert stats["scratch_builds"] == 1
        assert stats["scratch_reuses"] == 2


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_generalized_apply_gpu_fails_closed_without_host_fallback():
    size = 4
    topology = ti.ndarray(ti.i32, shape=size)
    topology.from_numpy(np.arange(size, dtype=np.int32))

    @ti.kernel
    def identity_kernel(
        active_size: ti.i32,
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(active_size):
            y[index] = x[topology_data[index]]

    operator = ti.linalg.LinearOperator.from_kernel(
        identity_kernel, size, topology
    )
    values = _array(ti.f32, [1.0, -2.0, 0.5, 3.0])
    poison = _array(ti.f32, [np.nan, np.nan, np.nan, np.nan])
    np.testing.assert_allclose(
        operator.apply(values, beta=0.0, addend=poison).to_numpy(),
        values.to_numpy(),
    )
    with pytest.raises(RuntimeError, match="unavailable on this GPU"):
        operator.apply(values, alpha=0.5)
    with pytest.raises(RuntimeError, match="unavailable on this GPU"):
        operator.apply(values, beta=1.0, addend=values)
    assert operator.statistics()["generalized_lowerings"] == 0
