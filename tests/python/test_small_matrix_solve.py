import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("n", [1, 5, 9, 16, 32])
@pytest.mark.parametrize("dtype", [ti.f32, ti.f64])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, fast_math=False)
def test_pivoted_solve_and_inverse(n, dtype):
    if dtype == ti.f64 and not ti._lib.core.is_extension_supported(ti.cfg.arch, ti.extension.data64):
        pytest.skip("Backend lacks f64 support")
    np_dtype = np.float32 if dtype == ti.f32 else np.float64
    rng = np.random.default_rng(318)
    values = rng.normal(size=(n, n)) * 0.2 + np.eye(n) * (n + 1)
    values = np.roll(values, 1, axis=0).astype(np_dtype)
    # Require a row exchange rather than relying on nonzero diagonal entries.
    if n > 1:
        values[0, 0] = 0
    rhs = rng.normal(size=n).astype(np_dtype)
    solution = np.zeros(n, dtype=np_dtype)
    inv = np.zeros((n, n), dtype=np_dtype)

    @ti.kernel
    def run(
        a: ti.types.ndarray(dtype=dtype, ndim=2),
        b: ti.types.ndarray(dtype=dtype, ndim=1),
        x: ti.types.ndarray(dtype=dtype, ndim=1),
        result: ti.types.ndarray(dtype=dtype, ndim=2),
    ):
        local_a = ti.Matrix([[a[i, j] for j in ti.static(range(n))] for i in ti.static(range(n))])
        local_b = ti.Vector([b[i] for i in ti.static(range(n))])
        local_x = ti.solve(local_a, local_b, dt=dtype)
        local_inv = local_a.inverse()
        for i in range(n):
            x[i] = local_x[i]
            for j in range(n):
                result[i, j] = local_inv[i, j]

    run(values, rhs, solution, inv)
    tolerance = 3e-6 if dtype == ti.f32 else 2e-12
    np.testing.assert_allclose(solution, np.linalg.solve(values, rhs), rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(inv, np.linalg.inv(values), rtol=tolerance, atol=tolerance)
    residual = np.linalg.norm(values @ solution - rhs, np.inf)
    scale = np.linalg.norm(values, np.inf) * np.linalg.norm(solution, np.inf) + np.linalg.norm(rhs, np.inf)
    assert residual / scale < tolerance
    np.testing.assert_allclose(values @ inv, np.eye(n), rtol=tolerance, atol=tolerance)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_batched_solve_is_local_and_four_by_four_boundary():
    output = ti.ndarray(ti.f32, shape=(8, 4))

    @ti.kernel
    def run(output: ti.types.ndarray(dtype=ti.f32, ndim=2)):
        for batch in range(8):
            a = ti.Matrix([[0.0, 2.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0], [0.0, 0.0, 4.0, 0.0], [0.0, 0.0, 0.0, 5.0]])
            b = ti.Vector([2.0, 3.0, 4.0, 5.0]) * (batch + 1)
            x = ti.solve(a, b)
            for i in range(4):
                output[batch, i] = x[i]

    run(output)
    np.testing.assert_allclose(output.to_numpy(), np.repeat(np.arange(1, 9)[:, None], 4, axis=1))


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, debug=False, fast_math=False)
def test_singular_large_inverse_returns_nan():
    @ti.kernel
    def run() -> ti.f32:
        a = ti.Matrix.zero(ti.f32, 5, 5)
        return a.inverse()[0, 0]

    assert np.isnan(run())


@test_utils.test(arch=ti.cpu, offline_cache=False, debug=True)
def test_singular_large_solve_debug_assertion():
    @ti.kernel
    def run() -> ti.f32:
        a = ti.Matrix.zero(ti.f32, 9, 9)
        b = ti.Vector.zero(ti.f32, 9)
        return ti.solve(a, b)[0]

    with pytest.raises(ti.TaichiAssertionError, match="Matrix is singular"):
        run()


@test_utils.test(arch=ti.cpu)
def test_host_inverse_preserves_input_and_pivots():
    values = np.roll(np.diag(np.arange(1, 10, dtype=np.float64)), 1, axis=0)
    matrix = ti.Matrix(values)
    result = matrix.inverse().to_numpy()
    np.testing.assert_allclose(result, np.linalg.inv(values))
    np.testing.assert_array_equal(matrix.to_numpy(), values)


@test_utils.test(arch=ti.cpu)
def test_large_solve_rejects_nonsquare_and_rhs_shape():
    @ti.kernel
    def nonsquare():
        a = ti.Matrix.zero(ti.f32, 4, 5)
        b = ti.Vector.zero(ti.f32, 4)
        print(ti.solve(a, b))

    @ti.kernel
    def wrong_rhs():
        a = ti.Matrix.identity(ti.f32, 5)
        b = ti.Matrix.zero(ti.f32, 5, 2)
        print(ti.solve(a, b))

    with pytest.raises(ti.TaichiCompilationError, match="Only square matrix"):
        nonsquare()
    with pytest.raises(ti.TaichiCompilationError, match="must be a vector"):
        wrong_rhs()
