import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils

"""
A_psd used in the tests is a random positive definite matrix with a given number of rows and columns:
A_psd = A * A^T
Reference: https://stackoverflow.com/questions/619335/a-simple-algorithm-for-generating-positive-semidefinite-matrices
2023.5.31 qbao: It's observed that the matrix generated above is semi-definite, and it fails about 5% of the tests.
Therefore, A_psd is modified from A * A^T to A * A^T + np.eye(n) to improve stability.
"""


def _build_direct_solver_matrix(values, storage_format="col_major"):
    values = np.asarray(values, dtype=np.float32)
    rows, cols = values.shape
    builder = ti.linalg.SparseMatrixBuilder(
        rows,
        cols,
        max_num_triplets=rows * cols,
        dtype=ti.f32,
        storage_format=storage_format,
    )

    @ti.kernel
    def fill(
        matrix_builder: ti.types.sparse_matrix_builder(),
        matrix_values: ti.types.ndarray(),
    ):
        for i, j in ti.ndrange(rows, cols):
            if matrix_values[i, j] != 0.0:
                matrix_builder[i, j] += matrix_values[i, j]

    fill(builder, values)
    return builder.build()


@pytest.mark.parametrize("storage_format", ["col_major", "row_major"])
@test_utils.test(arch=ti.cpu)
def test_sparse_solver_reuses_exact_symbolic_pattern(storage_format):
    dense_a = np.array(
        [[4.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 3.0]],
        dtype=np.float32,
    )
    dense_b = np.array(
        [[6.0, -2.0, 0.0], [-2.0, 5.0, -1.0], [0.0, -1.0, 4.0]],
        dtype=np.float32,
    )
    dense_changed_pattern = dense_b.copy()
    dense_changed_pattern[0, 2] = 0.25
    a = _build_direct_solver_matrix(dense_a, storage_format)
    b_matrix = _build_direct_solver_matrix(dense_b, storage_format)
    changed_pattern = _build_direct_solver_matrix(
        dense_changed_pattern, storage_format
    )
    rhs = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    solver = ti.linalg.SparseSolver(dtype=ti.f32, solver_type="LLT")
    solver.analyze_pattern(a)
    solver.factorize(a)
    assert np.allclose(solver.solve(rhs), np.linalg.solve(dense_a, rhs))

    solver.factorize(b_matrix)
    expected_b = np.linalg.solve(dense_b, rhs)
    assert np.allclose(solver.solve(rhs), expected_b)

    with pytest.raises(
        RuntimeError, match="requires the same sparse pattern"
    ):
        solver.factorize(changed_pattern)
    assert np.allclose(solver.solve(rhs), expected_b)

    b_matrix[0, 0] = 7.0
    with pytest.raises(RuntimeError, match="factorization is stale"):
        solver.solve(rhs)
    dense_b[0, 0] = 7.0
    solver.factorize(b_matrix)
    assert np.allclose(solver.solve(rhs), np.linalg.solve(dense_b, rhs))


@test_utils.test(arch=ti.cpu)
def test_sparse_solver_factorize_requires_analysis():
    matrix = _build_direct_solver_matrix(np.eye(2, dtype=np.float32))
    solver = ti.linalg.SparseSolver(dtype=ti.f32, solver_type="LLT")
    with pytest.raises(RuntimeError, match="analyze_pattern.*first"):
        solver.factorize(matrix)


@test_utils.test(arch=ti.cuda)
def test_cuda_sparse_solver_refreshes_values_for_reused_pattern():
    dense_a = np.array(
        [[4.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 3.0]],
        dtype=np.float32,
    )
    dense_b = np.array(
        [[6.0, -2.0, 0.0], [-2.0, 5.0, -1.0], [0.0, -1.0, 4.0]],
        dtype=np.float32,
    )
    dense_changed_pattern = dense_b.copy()
    dense_changed_pattern[0, 2] = 0.25
    a = _build_direct_solver_matrix(dense_a)
    b_matrix = _build_direct_solver_matrix(dense_b)
    changed_pattern = _build_direct_solver_matrix(dense_changed_pattern)
    rhs = ti.ndarray(dtype=ti.f32, shape=3)
    rhs.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))

    solver = ti.linalg.SparseSolver(dtype=ti.f32, solver_type="LLT")
    solver.analyze_pattern(a)
    solver.factorize(a)
    assert np.allclose(
        solver.solve(rhs).to_numpy(),
        np.linalg.solve(dense_a, rhs.to_numpy()),
        rtol=5.0e-3,
    )

    solver.factorize(b_matrix)
    expected_b = np.linalg.solve(dense_b, rhs.to_numpy())
    assert np.allclose(
        solver.solve(rhs).to_numpy(), expected_b, rtol=5.0e-3
    )

    with pytest.raises(
        RuntimeError, match="requires the same sparse pattern"
    ):
        solver.factorize(changed_pattern)
    assert np.allclose(
        solver.solve(rhs).to_numpy(), expected_b, rtol=5.0e-3
    )


@pytest.mark.parametrize("dtype", [ti.f32, ti.f64])
@pytest.mark.parametrize("solver_type", ["LLT", "LDLT", "LU"])
@pytest.mark.parametrize("ordering", ["AMD", "COLAMD"])
@test_utils.test(arch=ti.x64)
def test_sparse_LLT_solver(dtype, solver_type, ordering):
    np_dtype = ti.lang.util.to_numpy_type(dtype)
    n = 10
    A = np.random.rand(n, n)
    A_psd = (np.dot(A, A.transpose()) + np.eye(n)).astype(np_dtype)
    Abuilder = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100, dtype=dtype)
    b = ti.field(dtype=dtype, shape=n)

    @ti.kernel
    def fill(
        Abuilder: ti.types.sparse_matrix_builder(),
        InputArray: ti.types.ndarray(),
        b: ti.template(),
    ):
        for i, j in ti.ndrange(n, n):
            Abuilder[i, j] += InputArray[i, j]
        for i in range(n):
            b[i] = i + 1

    fill(Abuilder, A_psd, b)
    A = Abuilder.build()
    solver = ti.linalg.SparseSolver(dtype=dtype, solver_type=solver_type, ordering=ordering)
    solver.analyze_pattern(A)
    solver.factorize(A)
    x = solver.solve(b)

    res = np.linalg.solve(A_psd, b.to_numpy())
    for i in range(n):
        assert x[i] == test_utils.approx(res[i], rel=1.0)


@pytest.mark.parametrize("dtype", [ti.f32])
@pytest.mark.parametrize("solver_type", ["LLT", "LDLT", "LU"])
@pytest.mark.parametrize("ordering", ["AMD", "COLAMD"])
@test_utils.test(arch=ti.cpu)
def test_sparse_solver_ndarray_vector(dtype, solver_type, ordering):
    np_dtype = ti.lang.util.to_numpy_type(dtype)
    n = 10
    A = np.random.rand(n, n)
    A_psd = (np.dot(A, A.transpose()) + np.eye(n)).astype(np_dtype)
    Abuilder = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=300, dtype=dtype)
    b = ti.ndarray(ti.f32, shape=n)

    @ti.kernel
    def fill(
        Abuilder: ti.types.sparse_matrix_builder(),
        InputArray: ti.types.ndarray(),
        b: ti.types.ndarray(),
    ):
        for i, j in ti.ndrange(n, n):
            Abuilder[i, j] += InputArray[i, j]
        for i in range(n):
            b[i] = i + 1

    fill(Abuilder, A_psd, b)
    A = Abuilder.build()
    solver = ti.linalg.SparseSolver(dtype=dtype, solver_type=solver_type, ordering=ordering)
    solver.analyze_pattern(A)
    solver.factorize(A)
    x = solver.solve(b)

    res = np.linalg.solve(A_psd, b.to_numpy())
    for i in range(n):
        assert x[i] == test_utils.approx(res[i], rel=1.0)


@test_utils.test(arch=ti.cuda)
def test_gpu_sparse_solver():
    from scipy.sparse import coo_matrix

    @ti.kernel
    def init_b(b: ti.types.ndarray(), nrows: ti.i32):
        for i in range(nrows):
            b[i] = 1.0 + i / nrows

    n = 10
    A = np.random.rand(n, n)
    A_psd = (np.dot(A, A.transpose()) + np.eye(n)).astype(np.float32)

    A_raw_coo = coo_matrix(A_psd)
    nrows, ncols = A_raw_coo.shape
    nnz = A_raw_coo.nnz

    A_csr = A_raw_coo.tocsr()
    b = ti.ndarray(shape=nrows, dtype=ti.f32)
    init_b(b, nrows)

    # solve Ax = b using cusolver
    A_coo = A_csr.tocoo()
    A_builder = ti.linalg.SparseMatrixBuilder(num_rows=nrows, num_cols=ncols, dtype=ti.f32, max_num_triplets=nnz)

    @ti.kernel
    def fill(
        A_builder: ti.types.sparse_matrix_builder(),
        row_coo: ti.types.ndarray(),
        col_coo: ti.types.ndarray(),
        val_coo: ti.types.ndarray(),
    ):
        for i in range(nnz):
            A_builder[row_coo[i], col_coo[i]] += val_coo[i]

    fill(A_builder, A_coo.row, A_coo.col, A_coo.data)
    A_ti = A_builder.build()
    x_ti = ti.ndarray(shape=ncols, dtype=ti.float32)

    # solve Ax=b using numpy
    b_np = b.to_numpy()
    x_np = np.linalg.solve(A_psd, b_np)

    # solve Ax=b using cusolver refectorization
    solver = ti.linalg.SparseSolver(dtype=ti.f32)
    solver.analyze_pattern(A_ti)
    solver.factorize(A_ti)
    x_ti = solver.solve(b)
    ti.sync()
    assert np.allclose(x_ti.to_numpy(), x_np, rtol=5.0e-3)

    # solve Ax = b using compute function
    solver = ti.linalg.SparseSolver(dtype=ti.f32)
    solver.compute(A_ti)
    x_cti = solver.solve(b)
    ti.sync()
    assert np.allclose(x_cti.to_numpy(), x_np, rtol=5.0e-3)


@pytest.mark.parametrize("dtype", [ti.f32])
@pytest.mark.parametrize("solver_type", ["LLT", "LU"])
@test_utils.test(arch=ti.cuda)
def test_gpu_sparse_solver2(dtype, solver_type):
    np_dtype = ti.lang.util.to_numpy_type(dtype)
    n = 10
    A = np.random.rand(n, n)
    A_psd = (np.dot(A, A.transpose()) + np.eye(n)).astype(np_dtype)
    Abuilder = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=300, dtype=dtype)
    b = ti.ndarray(dtype, shape=n)

    @ti.kernel
    def fill(
        Abuilder: ti.types.sparse_matrix_builder(),
        InputArray: ti.types.ndarray(),
        b: ti.types.ndarray(),
    ):
        for i, j in ti.ndrange(n, n):
            Abuilder[i, j] += InputArray[i, j]
        for i in range(n):
            b[i] = i + 1

    fill(Abuilder, A_psd, b)
    A = Abuilder.build()
    solver = ti.linalg.SparseSolver(dtype=dtype, solver_type=solver_type)
    solver.analyze_pattern(A)
    solver.factorize(A)
    x = solver.solve(b)

    res = np.linalg.solve(A_psd, b.to_numpy())
    for i in range(n):
        assert x[i] == test_utils.approx(res[i], rel=1.0)
