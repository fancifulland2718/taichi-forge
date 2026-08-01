import numpy as np

import taichi_forge as ti
from tests import test_utils


@test_utils.test(default_fp=ti.f32, fast_math=False)
def test_fixed_tensor_tuple_helper_outer_product_accumulation():
    n = 32
    source = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n)
    output = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n)
    determinant = ti.field(ti.f32, shape=n)

    source_np = np.empty((n, 3, 3), dtype=np.float32)
    for i in range(n):
        scale = np.float32(1.0 + 0.01 * i)
        source_np[i] = np.array(
            [
                [scale, 0.125, -0.25],
                [0.5, scale + 0.25, 0.375],
                [-0.125, 0.75, scale + 0.5],
            ],
            dtype=np.float32,
        )
    source.from_numpy(source_np)

    @ti.func
    def helper(matrix):
        row = ti.Vector([matrix[0, 0], matrix[0, 1], matrix[0, 2]])
        scaled = 0.75 * row
        return matrix.determinant(), scaled

    @ti.kernel
    def run():
        for i in range(n):
            matrix = source[i]
            det, vector = helper(matrix)
            outer = vector.outer_product(vector)
            accumulated = ti.Matrix.zero(ti.f32, 3, 3)
            for lane in ti.static(range(3)):
                accumulated += outer * ti.cast(lane + 1, ti.f32)
            output[i] = accumulated + matrix.transpose() @ matrix
            determinant[i] = det

    run()

    vector_np = source_np[:, 0, :] * np.float32(0.75)
    expected = np.empty_like(source_np)
    for i in range(n):
        expected[i] = (
            np.outer(vector_np[i], vector_np[i]) * np.float32(6.0)
            + source_np[i].T @ source_np[i]
        )
    np.testing.assert_allclose(output.to_numpy(), expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(
        determinant.to_numpy(), np.linalg.det(source_np), rtol=2e-5, atol=2e-5
    )
