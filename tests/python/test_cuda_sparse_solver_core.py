"""Legacy CUDA factor state and reused solve storage through public entry points."""

import gc
import os

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils


def _matrix(dense):
    dense = np.asarray(dense, dtype=np.float32)
    n = dense.shape[0]
    builder = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=n * n)

    @ti.kernel
    def fill(builder: ti.types.sparse_matrix_builder(), values: ti.types.ndarray()):
        for i, j in ti.ndrange(n, n):
            if values[i, j] != 0:
                builder[i, j] += values[i, j]

    fill(builder, dense)
    # No application-provided external CUDA context or submission scope.
    return builder.build()


@pytest.mark.parametrize("method", ["LLT", "LU"])
@pytest.mark.parametrize("ordering", ["AMD", "COLAMD"])
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_sparse_core_reuses_storage_and_preserves_factor_state(method, ordering):
    dense = np.asarray([[4, 1, 0], [1 if method == "LLT" else 2, 3, 1], [0, 1, 2]], dtype=np.float32)
    matrix = _matrix(dense)
    solver = ti.linalg.SparseSolver(solver_type=method, ordering=ordering, provider="cusolver_sp")
    solver.analyze_pattern(matrix)
    assert not solver.info()
    solver.factorize(matrix)
    assert solver.info()
    assert solver.selected_provider == "cusolver_sp"
    initial = solver.solver.runtime_statistics()
    outputs = []
    for rhs_values in ([1, 2, 3], [3, -1, 0.5], [-2, 0, 4]):
        rhs = ti.ndarray(ti.f32, 3)
        rhs.from_numpy(np.asarray(rhs_values, dtype=np.float32))
        output = ti.ndarray(ti.f32, 3)
        output.fill(float("nan"))
        solver.solver.solve_rf(impl.get_runtime().prog, matrix.matrix, rhs.arr, output.arr)
        outputs.append((rhs, output, np.linalg.solve(dense, rhs_values)))
    after = solver.solver.runtime_statistics()
    assert after["workspace_allocations"] == initial["workspace_allocations"]
    assert after["descriptor_creations"] == initial["descriptor_creations"]
    assert after["solve_calls"] - initial["solve_calls"] == 3
    assert after["rhs_readback_bytes"] - initial["rhs_readback_bytes"] == (36 if method == "LU" else 0)
    for _, output, expected in outputs:
        np.testing.assert_allclose(output.to_numpy(), expected, rtol=1e-5, atol=1e-6)

    # Same stored pattern, different values and owner. Numeric failure clears
    # both native info and public state and never reuses the previous factor.
    bad = _matrix([[2, 1, 0], [1, 1, 1], [0, 1, 2]] if method == "LU" else [[4, 1, 0], [1, -3, 1], [0, 1, 2]])
    solver.factorize(bad)
    assert not solver.info()
    assert not solver.provider_status()["reuse"]["factorization_active"]
    with pytest.raises(RuntimeError, match="no successful numeric factorization"):
        solver.solve(outputs[0][0])
    solver.factorize(matrix)
    assert solver.info()
    recovered = solver.solver.runtime_statistics()
    assert recovered["workspace_allocations"] == initial["workspace_allocations"]
    np.testing.assert_allclose(solver.solve(outputs[0][0]).to_numpy(), outputs[0][2], rtol=1e-5, atol=1e-6)

    # Rejected shape/alias must not write output or destroy the valid factor.
    output = ti.ndarray(ti.f32, 3)
    output.fill(-17)
    for rhs, message in (
        (ti.ndarray(ti.f64, 3), "factorization dtype"),
        (ti.ndarray(ti.f32, 2), "RHS and"),
        (ti.Vector.ndarray(2, ti.f32, 3), "one-dimensional scalar"),
    ):
        with pytest.raises(RuntimeError, match=message):
            solver.solver.solve_rf(impl.get_runtime().prog, matrix.matrix, rhs.arr, output.arr)
    with pytest.raises(RuntimeError, match="overlap"):
        solver.solver.solve_rf(impl.get_runtime().prog, matrix.matrix, output.arr, output.arr)
    assert solver.info()
    np.testing.assert_array_equal(output.to_numpy(), [-17, -17, -17])

    # Destroy the owner after enqueue, before reading the output.
    rhs = outputs[1][0]
    solver.solver.solve_rf(impl.get_runtime().prog, matrix.matrix, rhs.arr, output.arr)
    del solver
    gc.collect()
    np.testing.assert_allclose(output.to_numpy(), outputs[1][2], rtol=1e-5, atol=1e-6)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_legacy_ldlt_is_not_silently_cholesky():
    matrix = _matrix([[2, 3, 0], [3, 2, 0], [0, 0, -2]])
    solver = ti.linalg.SparseSolver(solver_type="LDLT", provider="cusolver_sp")
    with pytest.raises(RuntimeError, match="does not implement LDLT"):
        solver.compute(matrix)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_public_ldlt_uses_real_symmetric_indefinite_provider():
    library = os.environ.get("TI_CUDSS_TEST_LIBRARY")
    if not library:
        pytest.skip("set TI_CUDSS_TEST_LIBRARY to the user-managed cuDSS 0.8.x library")
    assert ti.hardware.linalg.cudss_is_available(library_path=library)
    dense = np.asarray([[2, 3, 0], [3, 2, 0], [0, 0, -2]], dtype=np.float32)
    matrix = _matrix(dense)
    solver = ti.linalg.SparseSolver(solver_type="LDLT", provider="cudss", library_path=library)
    assert not solver.info()
    solver.compute(matrix)
    assert solver.selected_provider == "cudss"
    assert solver.info()
    assert solver.solver.matrix_type == "symmetric"
    for values in ([1, 2, 3], [-1, 3, 0.5]):
        rhs = ti.ndarray(ti.f32, 3)
        rhs.from_numpy(np.asarray(values, dtype=np.float32))
        solution = solver.solve(rhs).to_numpy()
        np.testing.assert_allclose(solution, np.linalg.solve(dense, values), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(dense @ solution, values, rtol=1e-5, atol=1e-6)
    solver.solver.close()
