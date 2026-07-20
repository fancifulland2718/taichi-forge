import numpy as np

import taichi_forge as ti
from taichi_forge.examples.simulation.implicit_mass_spring import Cloth
from tests import test_utils


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_pinned_pressure_laplacian_is_valid_for_llt():
    resolution = 8
    size = resolution * resolution
    builder = ti.linalg.SparseMatrixBuilder(
        size,
        size,
        max_num_triplets=size * 5,
        dtype=ti.f32,
    )

    @ti.kernel
    def assemble(matrix: ti.types.sparse_matrix_builder()):
        for i, j in ti.ndrange(resolution, resolution):
            row = i * resolution + j
            if row == 0:
                matrix[row, row] += 1.0
            else:
                center = 0.0
                if j != 0:
                    center += 1.0
                    if row - 1 != 0:
                        matrix[row, row - 1] += -1.0
                if j != resolution - 1:
                    center += 1.0
                    matrix[row, row + 1] += -1.0
                if i != 0:
                    center += 1.0
                    if row - resolution != 0:
                        matrix[row, row - resolution] += -1.0
                if i != resolution - 1:
                    center += 1.0
                    matrix[row, row + resolution] += -1.0
                matrix[row, row] += center

    assemble(builder)
    matrix = builder.build()
    rhs = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    rhs[0] = 0.0

    solver = ti.linalg.SparseSolver(dtype=ti.f32, solver_type="LLT")
    solver.analyze_pattern(matrix)
    solver.factorize(matrix)
    assert solver.info()
    solution = solver.solve(rhs)
    assert solver.info()

    applied = matrix @ solution.astype(np.float32, copy=False)
    np.testing.assert_allclose(applied, rhs, rtol=2e-4, atol=2e-5)
    assert solution[0] == test_utils.approx(0.0, abs=2e-6)


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_implicit_mass_spring_reuses_fixed_symbolic_pattern():
    cloth = Cloth(N=2)
    initial_positions = cloth.pos.to_numpy()

    cloth.update(0.01)
    cloth.update(0.01)

    positions = cloth.pos.to_numpy()
    assert cloth.solver_pattern_analyzed
    assert np.all(np.isfinite(positions))
    assert np.linalg.norm(positions - initial_positions) > 0.0
