import numpy as np

import taichi_forge as ti
from taichi_forge.examples.simulation.implicit_mass_spring import Cloth
from taichi_forge.examples.simulation.implicit_linear_operator import (
    ImplicitSpringChain,
)
from taichi_forge.lang import impl
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


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_implicit_linear_operator_rebinds_coefficients_and_reuses_solve_plan():
    simulation = ImplicitSpringChain(node_count=24)
    initial_displacement = simulation.displacement.to_numpy()
    results = [simulation.step() for _ in range(4)]

    assert all(result.converged for result in results)
    assert all(0 < result.iterations <= 96 for result in results)
    assert simulation.operator_numeric_version == 4
    assert simulation.preconditioner_numeric_version == 4
    assert np.all(simulation.inverse_status.to_numpy() == 0)
    expected_operator, expected_blocks, _ = simulation._coefficients(3)
    np.testing.assert_allclose(
        simulation.operator_candidate.to_numpy(),
        expected_operator,
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        simulation.inverse_blocks.to_numpy(),
        np.linalg.inv(expected_blocks.reshape(-1, 2, 2)).reshape(-1),
        rtol=2e-5,
        atol=2e-5,
    )
    displacement = simulation.displacement.to_numpy()
    assert np.all(np.isfinite(displacement))
    assert np.linalg.norm(displacement - initial_displacement) > 0.0

    lifecycle = simulation.preconditioner.statistics()
    assert lifecycle["setup_calls"] == 1
    assert lifecycle["update_successes"] == 3
    assert lifecycle["stale_rejections"] == 0
    stats = simulation.solve_plan.statistics()
    if impl.current_cfg().arch in (ti.cuda, ti.vulkan):
        assert stats["submission"]["execution_path"] == "cached_graph_submission"
        assert stats["submission"]["graphs_materialized"] == 1
        assert stats["vector_io"]["pack_calls"] == 0
        assert stats["vector_io"]["unpack_calls"] == 0
    else:
        assert stats["operations"]["operator_plan_invalidations"] == 0
        assert stats["operations"]["workspace_builds"] == 1
        assert stats["operations"]["workspace_reuses"] == 3
