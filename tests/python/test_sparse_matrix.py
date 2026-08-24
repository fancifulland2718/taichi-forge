import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.lang.exception import TaichiRuntimeError
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_sparse_matrix_builder_rejects_unimplemented_format_without_building():
    n = 4
    builder = ti.linalg.SparseMatrixBuilder(
        n, n, max_num_triplets=n, dtype=ti.f32
    )

    with pytest.raises(TaichiRuntimeError, match="supports CSR only"):
        builder.build(_format="BSR")

    @ti.kernel
    def fill(matrix: ti.types.sparse_matrix_builder()):
        for i in range(n):
            matrix[i, i] += i + 1

    fill(builder)
    matrix = builder.build(_format="csr")
    assert matrix._num_nonzero() == n
    contract = matrix._get_format_contract()
    assert contract["schema_version"] == 1
    assert contract["identity"]["storage_format"] in ("csr", "csc")
    assert contract["identity"]["index_dtype"] == "i32"
    assert contract["pattern"]["empty_supported"]
    assert contract["operations"]["ndarray_spmv"]
    assert contract["constraints"]["public_builder_available"]
    assert not contract["constraints"]["public_bsr_available"]
    assert not contract["constraints"]["silent_format_fallback"]


@pytest.mark.parametrize("dtype", [ti.f32, ti.f64])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_cpu_sparse_builder_rejects_oversized_count_and_remains_reusable(dtype):
    empty_builder = ti.linalg.SparseMatrixBuilder(1, 1, dtype=dtype)
    assert empty_builder.build()._num_nonzero() == 0

    builder = ti.linalg.SparseMatrixBuilder(
        2, 2, max_num_triplets=1, dtype=dtype
    )

    @ti.kernel
    def overflow(matrix: ti.types.sparse_matrix_builder()):
        matrix[0, 0] += 1.0
        matrix[1, 1] += 2.0

    overflow(builder)

    with pytest.raises(RuntimeError, match="triplet count 2 exceeds capacity"):
        builder.build()

    @ti.kernel
    def fill(matrix: ti.types.sparse_matrix_builder()):
        matrix[1, 1] += 4.0

    fill(builder)
    matrix = builder.build()
    assert matrix._num_nonzero() == 1
    assert matrix[1, 1] == pytest.approx(4.0)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_sparse_builder_bounds_insertion_and_remains_reusable():
    builder = ti.linalg.SparseMatrixBuilder(
        2, 2, max_num_triplets=1, dtype=ti.f32
    )

    @ti.kernel
    def overflow(matrix: ti.types.sparse_matrix_builder()):
        matrix[0, 0] += 1.0
        matrix[1, 1] += 2.0

    overflow(builder)
    with pytest.raises(RuntimeError, match="triplet count 2 exceeds capacity"):
        builder.build()

    @ti.kernel
    def fill(matrix: ti.types.sparse_matrix_builder()):
        matrix[1, 1] += 4.0

    fill(builder)
    matrix = builder.build()
    assert matrix._num_nonzero() == 1
    assert matrix[1, 1] == pytest.approx(4.0)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_public_builder_uses_active_count_and_publishes_exact_csr():
    builder = ti.linalg.SparseMatrixBuilder(
        3, 3, max_num_triplets=8, dtype=ti.f32
    )

    @ti.kernel
    def fill(matrix: ti.types.sparse_matrix_builder()):
        matrix[0, 0] += 1.25
        matrix[2, 1] += -3.0
        matrix[0, 0] += 2.75

    fill(builder)
    matrix = builder.build()
    assert matrix._num_nonzero() == 2
    assert matrix[0, 0] == pytest.approx(4.0)
    assert matrix[2, 1] == pytest.approx(-3.0)
    stats = matrix._debug_runtime_stats()
    assert stats["resources"]["pattern_reserved_bytes"] == 24
    assert stats["resources"]["values_reserved_bytes"] == 8
    assert stats["transfers"]["device_to_host_bytes"] == 0
    assert stats["transfers"]["host_to_device_bytes"] == 0
    assert stats["transfers"]["device_to_device_bytes"] == 32

    empty = builder.build()
    assert empty._num_nonzero() == 0
    assert matrix[0, 0] == pytest.approx(4.0)


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_sparse_builder_bounds_insertion_and_remains_reusable():
    builder = ti.linalg.SparseMatrixBuilder(
        2, 2, max_num_triplets=1, dtype=ti.f32
    )

    @ti.kernel
    def overflow(matrix: ti.types.sparse_matrix_builder()):
        matrix[0, 0] += 1.0
        matrix[1, 1] += 2.0

    overflow(builder)
    with pytest.raises(RuntimeError, match="triplet count 2 exceeds capacity"):
        builder.build()

    @ti.kernel
    def fill(matrix: ti.types.sparse_matrix_builder()):
        matrix[1, 1] += 4.0

    fill(builder)
    matrix = builder.build()
    x = ti.ndarray(dtype=ti.f32, shape=2)
    x.from_numpy(np.asarray([2.0, 3.0], dtype=np.float32))
    y = matrix @ x
    np.testing.assert_allclose(
        y.to_numpy(), [0.0, 12.0], rtol=1e-6, atol=1e-6
    )


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_public_builder_uses_active_count_and_publishes_exact_csr():
    builder = ti.linalg.SparseMatrixBuilder(
        3, 3, max_num_triplets=8, dtype=ti.f32
    )

    @ti.kernel
    def fill(matrix: ti.types.sparse_matrix_builder()):
        matrix[0, 0] += 1.25
        matrix[2, 1] += -3.0
        matrix[0, 0] += 2.75

    fill(builder)
    matrix = builder.build()
    assert matrix._num_nonzero() == 2
    x = ti.ndarray(dtype=ti.f32, shape=3)
    x.from_numpy(np.asarray([2.0, 5.0, 7.0], dtype=np.float32))
    y = matrix @ x
    np.testing.assert_allclose(
        y.to_numpy(), np.asarray([8.0, 0.0, -15.0], dtype=np.float32)
    )
    stats = matrix._debug_runtime_stats()
    assert stats["resources"]["pattern_reserved_bytes"] == 24
    assert stats["resources"]["values_reserved_bytes"] == 8
    assert stats["transfers"]["device_to_host_bytes"] == 0
    assert stats["transfers"]["host_to_device_bytes"] == 0
    assert stats["transfers"]["device_to_device_bytes"] == 32

    empty = builder.build()
    assert empty._num_nonzero() == 0
    empty_y = empty @ x
    np.testing.assert_allclose(
        empty_y.to_numpy(), np.zeros(3, dtype=np.float32)
    )
    old_y = matrix @ x
    np.testing.assert_allclose(
        old_y.to_numpy(), np.asarray([8.0, 0.0, -15.0], dtype=np.float32)
    )


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_cpu_sparse_builder_is_validated_deterministic_and_transactional():
    builder = ti.linalg.SparseMatrixBuilder(
        2, 2, max_num_triplets=3, dtype=ti.f32
    )

    @ti.kernel
    def fill_invalid_index(matrix: ti.types.sparse_matrix_builder()):
        matrix[2, 0] += 1.0

    fill_invalid_index(builder)
    with pytest.raises(RuntimeError, match="outside matrix dimensions"):
        builder.build()

    @ti.kernel
    def fill_nonfinite(matrix: ti.types.sparse_matrix_builder()):
        matrix[0, 0] += ti.cast(float("inf"), ti.f32)

    fill_nonfinite(builder)
    with pytest.raises(RuntimeError, match="contains a non-finite value"):
        builder.build()

    @ti.kernel
    def fill_duplicate_overflow(matrix: ti.types.sparse_matrix_builder()):
        matrix[0, 0] += ti.cast(3.0e38, ti.f32)
        matrix[0, 0] += ti.cast(3.0e38, ti.f32)

    fill_duplicate_overflow(builder)
    with pytest.raises(RuntimeError, match="duplicate sum.*non-finite"):
        builder.build()

    @ti.kernel
    def fill_order_sensitive_duplicates(
        matrix: ti.types.sparse_matrix_builder(),
    ):
        matrix[0, 0] += ti.cast(1.0e20, ti.f32)
        matrix[0, 0] += ti.cast(-1.0e20, ti.f32)
        matrix[0, 0] += 3.0

    fill_order_sensitive_duplicates(builder)
    matrix = builder.build()
    assert matrix._num_nonzero() == 1
    assert matrix[0, 0] == pytest.approx(3.0)


@pytest.mark.parametrize(
    "dtype, storage_format",
    [
        (ti.f32, "col_major"),
        (ti.f32, "row_major"),
        (ti.f64, "col_major"),
        (ti.f64, "row_major"),
    ],
)
@test_utils.test(arch=ti.cpu)
def test_sparse_matrix_builder_deprecated_anno(dtype, storage_format):
    n = 8
    Abuilder = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100, dtype=dtype, storage_format=storage_format)

    @ti.kernel
    def fill(Abuilder: ti.types.sparse_matrix_builder()):
        for i, j in ti.ndrange(n, n):
            Abuilder[i, j] += i + j

    fill(Abuilder)
    A = Abuilder.build()
    for i in range(n):
        for j in range(n):
            assert A[i, j] == i + j


@pytest.mark.parametrize(
    "dtype, storage_format",
    [
        (ti.f32, "col_major"),
        (ti.f32, "row_major"),
        (ti.f64, "col_major"),
        (ti.f64, "row_major"),
    ],
)
@test_utils.test(arch=ti.cpu)
def test_sparse_matrix_builder(dtype, storage_format):
    n = 8
    Abuilder = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100, dtype=dtype, storage_format=storage_format)

    @ti.kernel
    def fill(Abuilder: ti.types.sparse_matrix_builder()):
        for i, j in ti.ndrange(n, n):
            Abuilder[i, j] += i + j

    fill(Abuilder)
    A = Abuilder.build()
    for i in range(n):
        for j in range(n):
            assert A[i, j] == i + j


@pytest.mark.parametrize(
    "dtype, storage_format",
    [
        (ti.f32, "col_major"),
        (ti.f32, "row_major"),
        (ti.f64, "col_major"),
        (ti.f64, "row_major"),
    ],
)
@test_utils.test(arch=ti.cpu)
def test_build_sparse_matrix_frome_ndarray(dtype, storage_format):
    n = 8
    triplets = ti.Vector.ndarray(n=3, dtype=ti.f32, shape=n)
    A = ti.linalg.SparseMatrix(n=10, m=10, dtype=ti.f32, storage_format=storage_format)

    @ti.kernel
    def fill(triplets: ti.types.ndarray()):
        for i in range(n):
            triplet = ti.Vector([i, i, i], dt=ti.f32)
            triplets[i] = triplet

    fill(triplets)
    A.build_from_ndarray(triplets)

    for i in range(n):
        assert A[i, i] == i


@pytest.mark.parametrize(
    "dtype, storage_format",
    [
        (ti.f32, "col_major"),
        (ti.f32, "row_major"),
        (ti.f64, "col_major"),
        (ti.f64, "row_major"),
    ],
)
@test_utils.test(arch=ti.cpu)
def test_sparse_matrix_shape(dtype, storage_format):
    n, m = 8, 9
    Abuilder = ti.linalg.SparseMatrixBuilder(n, m, max_num_triplets=100, dtype=dtype, storage_format=storage_format)

    @ti.kernel
    def fill(Abuilder: ti.types.sparse_matrix_builder()):
        for i, j in ti.ndrange(n, m):
            Abuilder[i, j] += i + j

    fill(Abuilder)
    A = Abuilder.build()
    assert A.shape == (n, m)


@pytest.mark.parametrize(
    "dtype, storage_format",
    [
        (ti.f32, "col_major"),
        (ti.f32, "row_major"),
        (ti.f64, "col_major"),
        (ti.f64, "row_major"),
    ],
)
@test_utils.test(arch=ti.cpu)
def test_sparse_matrix_element_access(dtype, storage_format):
    n = 8
    Abuilder = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100, dtype=dtype, storage_format=storage_format)

    @ti.kernel
    def fill(Abuilder: ti.types.sparse_matrix_builder()):
        for i in range(n):
            Abuilder[i, i] += i

    fill(Abuilder)
    A = Abuilder.build()
    for i in range(n):
        assert A[i, i] == i


@pytest.mark.parametrize(
    "dtype, storage_format",
    [
        (ti.f32, "col_major"),
        (ti.f32, "row_major"),
        (ti.f64, "col_major"),
        (ti.f64, "row_major"),
    ],
)
@test_utils.test(arch=ti.cpu)
def test_sparse_matrix_element_modify(dtype, storage_format):
    n = 8
    Abuilder = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100, dtype=dtype, storage_format=storage_format)

    @ti.kernel
    def fill(Abuilder: ti.types.sparse_matrix_builder()):
        for i in range(n):
            Abuilder[i, i] += i

    fill(Abuilder)
    A = Abuilder.build()
    A[0, 0] = 1024.0
    assert A[0, 0] == 1024.0


@pytest.mark.parametrize(
    "dtype, storage_format",
    [
        (ti.f32, "col_major"),
        (ti.f32, "row_major"),
        (ti.f64, "col_major"),
        (ti.f64, "row_major"),
    ],
)
@test_utils.test(arch=ti.cpu)
def test_sparse_matrix_value_only_update(dtype, storage_format):
    n = 8
    builder = ti.linalg.SparseMatrixBuilder(
        n, n, max_num_triplets=n, dtype=dtype, storage_format=storage_format
    )

    @ti.kernel
    def fill_pattern(A: ti.types.sparse_matrix_builder()):
        for i in range(n):
            A[i, i] += i + 1

    fill_pattern(builder)
    A = builder.build()
    values = ti.ndarray(dtype=dtype, shape=n)

    @ti.kernel
    def fill_values(v: ti.types.ndarray()):
        for i in range(n):
            v[i] = 2 * (i + 1)

    fill_values(values)
    assert A._num_nonzero() == n
    A._update_values(values)
    for i in range(n):
        assert A[i, i] == 2 * (i + 1)

    wrong_size = ti.ndarray(dtype=dtype, shape=n - 1)
    with pytest.raises(Exception, match="expects exactly"):
        A._update_values(wrong_size)
    for i in range(n):
        assert A[i, i] == 2 * (i + 1)


@pytest.mark.parametrize(
    "dtype, storage_format",
    [
        (ti.f32, "col_major"),
        (ti.f32, "row_major"),
        (ti.f64, "col_major"),
        (ti.f64, "row_major"),
    ],
)
@test_utils.test(arch=ti.cpu)
def test_sparse_matrix_addition(dtype, storage_format):
    n = 8
    Abuilder = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100, dtype=dtype, storage_format=storage_format)
    Bbuilder = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100, dtype=dtype, storage_format=storage_format)

    @ti.kernel
    def fill(
        Abuilder: ti.types.sparse_matrix_builder(),
        Bbuilder: ti.types.sparse_matrix_builder(),
    ):
        for i, j in ti.ndrange(n, n):
            Abuilder[i, j] += i + j
            Bbuilder[i, j] += i - j

    fill(Abuilder, Bbuilder)
    A = Abuilder.build()
    B = Bbuilder.build()
    C = A + B
    assert C.dtype == dtype
    for i in range(n):
        for j in range(n):
            assert C[i, j] == 2 * i


@pytest.mark.parametrize(
    "dtype, storage_format",
    [
        (ti.f32, "col_major"),
        (ti.f32, "row_major"),
        (ti.f64, "col_major"),
        (ti.f64, "row_major"),
    ],
)
@test_utils.test(arch=ti.cpu)
def test_sparse_matrix_subtraction(dtype, storage_format):
    n = 8
    Abuilder = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100, dtype=dtype, storage_format=storage_format)
    Bbuilder = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100, dtype=dtype, storage_format=storage_format)

    @ti.kernel
    def fill(
        Abuilder: ti.types.sparse_matrix_builder(),
        Bbuilder: ti.types.sparse_matrix_builder(),
    ):
        for i, j in ti.ndrange(n, n):
            Abuilder[i, j] += i + j
            Bbuilder[i, j] += i - j

    fill(Abuilder, Bbuilder)
    A = Abuilder.build()
    B = Bbuilder.build()
    C = A - B
    assert C.dtype == dtype
    for i in range(n):
        for j in range(n):
            assert C[i, j] == 2 * j


@pytest.mark.parametrize(
    "dtype, storage_format",
    [
        (ti.f32, "col_major"),
        (ti.f32, "row_major"),
        (ti.f64, "col_major"),
        (ti.f64, "row_major"),
    ],
)
@test_utils.test(arch=ti.cpu)
def test_sparse_matrix_scalar_multiplication(dtype, storage_format):
    n = 8
    Abuilder = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100, dtype=dtype, storage_format=storage_format)

    @ti.kernel
    def fill(Abuilder: ti.types.sparse_matrix_builder()):
        for i, j in ti.ndrange(n, n):
            Abuilder[i, j] += i + j

    fill(Abuilder)
    A = Abuilder.build()
    B = A * 3.0
    assert B.dtype == dtype
    for i in range(n):
        for j in range(n):
            assert B[i, j] == 3 * (i + j)


@pytest.mark.parametrize(
    "dtype, storage_format",
    [
        (ti.f32, "col_major"),
        (ti.f32, "row_major"),
        (ti.f64, "col_major"),
        (ti.f64, "row_major"),
    ],
)
@test_utils.test(arch=ti.cpu)
def test_sparse_matrix_transpose(dtype, storage_format):
    n = 8
    Abuilder = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100, dtype=dtype, storage_format=storage_format)

    @ti.kernel
    def fill(Abuilder: ti.types.sparse_matrix_builder()):
        for i, j in ti.ndrange(n, n):
            Abuilder[i, j] += i + j

    fill(Abuilder)
    A = Abuilder.build()
    B = A.transpose()
    assert B.dtype == dtype
    for i in range(n):
        for j in range(n):
            assert B[i, j] == A[j, i]


@pytest.mark.parametrize(
    "dtype, storage_format",
    [
        (ti.f32, "col_major"),
        (ti.f32, "row_major"),
        (ti.f64, "col_major"),
        (ti.f64, "row_major"),
    ],
)
@test_utils.test(arch=ti.cpu)
def test_sparse_matrix_elementwise_multiplication(dtype, storage_format):
    n = 8
    Abuilder = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100, dtype=dtype, storage_format=storage_format)
    Bbuilder = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100, dtype=dtype, storage_format=storage_format)

    @ti.kernel
    def fill(
        Abuilder: ti.types.sparse_matrix_builder(),
        Bbuilder: ti.types.sparse_matrix_builder(),
    ):
        for i, j in ti.ndrange(n, n):
            Abuilder[i, j] += i + j
            Bbuilder[i, j] += i - j

    fill(Abuilder, Bbuilder)
    A = Abuilder.build()
    B = Bbuilder.build()
    C = A * B
    for i in range(n):
        for j in range(n):
            assert C[i, j] == (i + j) * (i - j)


@pytest.mark.parametrize(
    "dtype, storage_format",
    [
        (ti.f32, "col_major"),
        (ti.f32, "row_major"),
        (ti.f64, "col_major"),
        (ti.f64, "row_major"),
    ],
)
@test_utils.test(arch=ti.cpu)
def test_sparse_matrix_multiplication(dtype, storage_format):
    n = 2
    Abuilder = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100, dtype=dtype, storage_format=storage_format)
    Bbuilder = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100, dtype=dtype, storage_format=storage_format)

    @ti.kernel
    def fill(
        Abuilder: ti.types.sparse_matrix_builder(),
        Bbuilder: ti.types.sparse_matrix_builder(),
    ):
        for i, j in ti.ndrange(n, n):
            Abuilder[i, j] += i + j
            Bbuilder[i, j] += i - j

    fill(Abuilder, Bbuilder)
    A = Abuilder.build()
    B = Bbuilder.build()
    C = A @ B
    assert C[0, 0] == 1.0
    assert C[0, 1] == 0.0
    assert C[1, 0] == 2.0
    assert C[1, 1] == -1.0


@pytest.mark.parametrize(
    "dtype, storage_format",
    [
        (ti.f32, "col_major"),
        (ti.f32, "row_major"),
        (ti.f64, "col_major"),
        (ti.f64, "row_major"),
    ],
)
@test_utils.test(arch=ti.cpu)
def test_sparse_matrix_nonsymmetric_multiplication(dtype, storage_format):
    n, k, m = 2, 3, 4
    Abuilder = ti.linalg.SparseMatrixBuilder(n, k, max_num_triplets=100, dtype=dtype, storage_format=storage_format)
    Bbuilder = ti.linalg.SparseMatrixBuilder(k, m, max_num_triplets=100, dtype=dtype, storage_format=storage_format)

    @ti.kernel
    def fill(
        Abuilder: ti.types.sparse_matrix_builder(),
        Bbuilder: ti.types.sparse_matrix_builder(),
    ):
        for i, j in ti.ndrange(n, k):
            Abuilder[i, j] += i + j
        for i, j in ti.ndrange(k, m):
            Bbuilder[i, j] -= i + j

    fill(Abuilder, Bbuilder)
    A = Abuilder.build()
    B = Bbuilder.build()
    C = A @ B
    GT = [[-5, -8, -11, -14], [-8, -14, -20, -26]]
    for i in range(n):
        for j in range(m):
            assert C[i, j] == GT[i][j]


@pytest.mark.parametrize(
    "dtype, storage_format",
    [
        (ti.f32, "col_major"),
        (ti.f32, "row_major"),
        (ti.f64, "col_major"),
        (ti.f64, "row_major"),
    ],
)
@test_utils.test(arch=ti.cpu)
def test_sparse_matrix_ndarray_vector_multiplication(dtype, storage_format):
    n = 2
    Abuilder = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100, dtype=dtype, storage_format=storage_format)
    x = ti.ndarray(dtype, n)

    @ti.kernel
    def fill(Abuilder: ti.types.sparse_matrix_builder()):
        for i, j in ti.ndrange(n, n):
            Abuilder[i, j] += i + j

    fill(Abuilder)
    x.fill(1.0)
    A = Abuilder.build()
    res = A @ x
    res_n = res.to_numpy()
    assert res_n[0] == 1.0
    assert res_n[1] == 3.0


@test_utils.test(arch=ti.cuda)
def test_gpu_sparse_matrix():
    import numpy as np

    num_triplets, num_rows, num_cols = 9, 4, 4
    np_idx_dtype, np_val_dtype = np.int32, np.float32
    coo_row = np.asarray([0, 0, 0, 1, 2, 2, 2, 3, 3], dtype=np_idx_dtype)
    coo_col = np.asarray([0, 2, 3, 1, 0, 2, 3, 1, 3], dtype=np_idx_dtype)
    coo_val = np.asarray([i + 1.0 for i in range(num_triplets)], dtype=np_val_dtype)
    h_X = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np_val_dtype)
    h_Y = np.asarray([19.0, 8.0, 51.0, 52.0], dtype=np_val_dtype)

    ti_dtype = ti.f32
    X = ti.ndarray(shape=num_cols, dtype=ti_dtype)
    Y = ti.ndarray(shape=num_rows, dtype=ti_dtype)

    X.from_numpy(h_X)
    Y.fill(0.0)

    A_builder = ti.linalg.SparseMatrixBuilder(num_rows=4, num_cols=4, dtype=ti_dtype, max_num_triplets=50)

    @ti.kernel
    def fill(
        A: ti.types.sparse_matrix_builder(),
        coo_row: ti.types.ndarray(),
        coo_col: ti.types.ndarray(),
        coo_val: ti.types.ndarray(),
    ):
        for i in range(num_triplets):
            A[coo_row[i], coo_col[i]] += coo_val[i]

    fill(A_builder, coo_row, coo_col, coo_val)
    A = A_builder.build()

    # Compute Y = A @ X
    Y = A @ X
    for i in range(4):
        assert Y[i] == h_Y[i]

    # Reuse the same matrix plan with different input and output addresses.
    X2 = ti.ndarray(shape=num_cols, dtype=ti_dtype)
    X2.from_numpy(2.0 * h_X)
    Y2 = A @ X2
    for i in range(4):
        assert Y2[i] == 2.0 * h_Y[i]

    Y3 = A @ X
    for i in range(4):
        assert Y3[i] == h_Y[i]

    # Update CSR values in place. The row/column pattern and warm cuSPARSE
    # descriptor/workspace remain valid.
    values2 = ti.ndarray(shape=num_triplets, dtype=ti_dtype)
    values2.from_numpy(2.0 * coo_val)
    assert A._num_nonzero() == num_triplets
    A._update_values(values2)
    Y4 = A @ X
    for i in range(4):
        assert Y4[i] == 2.0 * h_Y[i]

    wrong_size = ti.ndarray(shape=num_triplets - 1, dtype=ti_dtype)
    with pytest.raises(Exception, match="expects exactly"):
        A._update_values(wrong_size)
    Y5 = A @ X
    for i in range(4):
        assert Y5[i] == 2.0 * h_Y[i]


@pytest.mark.parametrize("N", [5])
@test_utils.test(arch=ti.cuda)
def test_gpu_sparse_matrix_ops(N):
    import numpy as np
    from numpy.random import default_rng
    from scipy import stats
    from scipy.sparse import coo_matrix, random

    @ti.kernel
    def fill(
        A: ti.types.sparse_matrix_builder(),
        coo_row: ti.types.ndarray(),
        coo_col: ti.types.ndarray(),
        coo_val: ti.types.ndarray(),
        nnz: ti.i32,
    ):
        for i in range(nnz):
            A[coo_row[i], coo_col[i]] += coo_val[i]

    seed = 2
    np.random.seed(seed)
    rng = default_rng(seed)
    rvs = stats.poisson(3, loc=1).rvs
    np_dtype = np.float32
    val_dt = ti.float32

    n_rows = N
    n_cols = N - 1

    S1 = random(n_rows, n_cols, density=0.5, random_state=rng, data_rvs=rvs).astype(np_dtype).tocoo()
    S2 = random(n_rows, n_cols, density=0.5, random_state=rng, data_rvs=rvs).astype(np_dtype).tocoo()

    nnz_A = S1.nnz
    nnz_B = S2.nnz

    A_builder = ti.linalg.SparseMatrixBuilder(num_rows=n_rows, num_cols=n_cols, dtype=val_dt, max_num_triplets=nnz_A)
    B_builder = ti.linalg.SparseMatrixBuilder(num_rows=n_rows, num_cols=n_cols, dtype=val_dt, max_num_triplets=nnz_B)
    fill(A_builder, S1.row, S1.col, S1.data, nnz_A)
    fill(B_builder, S2.row, S2.col, S2.data, nnz_B)
    A = A_builder.build()
    B = B_builder.build()

    def verify(scipy_spm, taichi_spm):
        scipy_spm = scipy_spm.tocoo()
        for i, j, v in zip(scipy_spm.row, scipy_spm.col, scipy_spm.data):
            assert v == test_utils.approx(taichi_spm[i, j], rel=1e-5)

    C = A + B
    S3 = S1 + S2
    verify(S3, C)

    D = C - A
    S4 = S3 - S1
    verify(S4, D)

    E = A * 2.5
    S5 = S1 * 2.5
    verify(S5, E)

    F = A * 2.5
    S6 = S1 * 2.5
    verify(S6, F)

    G = A.transpose()
    S7 = S1.T
    verify(S7, G)

    H = A @ B.transpose()
    S8 = S1 @ S2.T
    verify(S8, H)


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
)
def test_public_bsr_pattern_matrix_spmv_update_and_sharing():
    row_offsets_host = np.asarray([0, 1, 2], dtype=np.int32)
    column_indices_host = np.asarray([0, 1], dtype=np.int32)
    row_offsets = ti.ndarray(dtype=ti.i32, shape=3)
    column_indices = ti.ndarray(dtype=ti.i32, shape=2)
    row_offsets.from_numpy(row_offsets_host)
    column_indices.from_numpy(column_indices_host)
    pattern = ti.linalg.SparsePattern.bsr(
        block_rows=2,
        block_cols=2,
        block_size=2,
        row_offsets=row_offsets,
        column_indices=column_indices,
    )
    assert pattern.shape == (4, 4)
    assert pattern.block_shape == (2, 2)
    assert pattern.block_size == 2
    assert pattern.num_blocks == 2
    assert pattern.storage_format == "bsr"

    identity_values = np.tile(np.eye(2, dtype=np.float32).reshape(-1), 2)
    values_a = ti.ndarray(dtype=ti.f32, shape=8)
    values_b = ti.ndarray(dtype=ti.f32, shape=8)
    values_a.from_numpy(identity_values)
    values_b.from_numpy(2.0 * identity_values)
    try:
        matrix_a = pattern.matrix(values_a)
        matrix_b = ti.linalg.SparseMatrix.from_pattern(pattern, values_b)
    except RuntimeError as exc:
        if ti.lang.impl.current_cfg().arch == ti.cuda and "does not support generic BSR SpMV" in str(exc):
            pytest.skip("loaded cuSPARSE provider lacks generic BSR SpMV")
        raise

    vector_host = np.asarray([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
    vector = ti.ndarray(dtype=ti.f32, shape=4)
    vector.from_numpy(vector_host)
    np.testing.assert_allclose((matrix_a @ vector).to_numpy(), vector_host)
    np.testing.assert_allclose((matrix_b @ vector).to_numpy(), 2.0 * vector_host)

    matrix_a.update_values(values_b)
    np.testing.assert_allclose((matrix_a @ vector).to_numpy(), 2.0 * vector_host)
    if ti.lang.impl.current_cfg().arch == ti.cpu:
        exact = np.asarray([0.5, -1.0, 1.5, -2.0], dtype=np.float32)
        rhs = ti.ndarray(dtype=ti.f32, shape=4)
        solution = ti.ndarray(dtype=ti.f32, shape=4)
        rhs.from_numpy(2.0 * exact)
        solution.fill(0.0)
        program = ti.lang.impl.get_runtime().prog
        solver = ti._lib.core._make_cpu_operator_cg_solver(
            program, matrix_a.matrix, 8, 1e-6, 0.0
        )
        solver.solve(program, solution.arr, rhs.arr)
        assert solver.is_success()
        np.testing.assert_allclose(solution.to_numpy(), exact, rtol=2e-5, atol=2e-5)
    elif ti.lang.impl.current_cfg().arch == ti.cuda:
        exact = np.asarray([0.5, -1.0, 1.5, -2.0], dtype=np.float32)
        rhs = ti.ndarray(dtype=ti.f32, shape=4)
        solution = ti.ndarray(dtype=ti.f32, shape=4)
        rhs.from_numpy(2.0 * exact)
        solution.fill(0.0)
        cg = ti.linalg.SparseCG(
            matrix_a,
            rhs,
            solution,
            max_iter=8,
            atol=1e-6,
            preconditioner="block_jacobi",
        )
        solved, converged = cg.solve()
        assert converged
        np.testing.assert_allclose(solved.to_numpy(), exact, rtol=2e-5, atol=2e-5)
        solution.fill(0.0)
        solved_again, converged_again = cg.solve()
        assert converged_again
        np.testing.assert_allclose(
            solved_again.to_numpy(), exact, rtol=2e-5, atol=2e-5
        )
        plan_stats = cg._debug_runtime_stats()
        assert plan_stats["operations"]["workspace_builds"] == 1
        assert plan_stats["operations"]["workspace_reuses"] == 1
        assert plan_stats["operations"]["preconditioner_apply_calls"] > 0
        assert plan_stats["identity"]["operator_action_provider"] == "cusparse"
        assert (
            plan_stats["identity"]["preconditioner_action_provider"]
            == "cuda_block_jacobi"
        )
        assert plan_stats["identity"]["operator_asynchronous_submit"]
        assert plan_stats["identity"]["preconditioner_asynchronous_submit"]
        assert plan_stats["identity"]["preconditioner_behavior"] == "fixed_linear"
        assert plan_stats["operations"]["operator_generation_pins"] == 3
        assert plan_stats["operations"]["preconditioner_generation_pins"] == 3
        assert plan_stats["operations"]["preconditioner_setup_calls"] == 1
        assert plan_stats["operations"]["preconditioner_update_calls"] == 2
        assert plan_stats["operations"]["operator_plan_invalidations"] == 0
        assert (
            plan_stats["operations"]["preconditioner_plan_invalidations"]
            == 0
        )
    elif ti.lang.impl.current_cfg().arch == ti.vulkan:
        exact = np.asarray([0.5, -1.0, 1.5, -2.0], dtype=np.float32)
        rhs = ti.ndarray(dtype=ti.f32, shape=4)
        solution = ti.ndarray(dtype=ti.f32, shape=4)
        rhs.from_numpy(2.0 * exact)
        solution.fill(0.0)
        program = ti.lang.impl.get_runtime().prog
        preconditioner = (
            ti._lib.core._make_sparse_block_jacobi_preconditioner_plan(
                program, matrix_a.matrix
            )
        )
        plan = (
            ti._lib.core._make_vulkan_block_jacobi_pcg_convergence_plan(
                program, matrix_a.matrix, preconditioner, 8, 1e-6
            )
        )
        plan.solve(program, solution.arr, rhs.arr)
        assert plan.is_success()
        np.testing.assert_allclose(
            solution.to_numpy(), exact, rtol=2e-5, atol=2e-5
        )
        solution.fill(0.0)
        plan.solve(program, solution.arr, rhs.arr)
        assert plan.is_success()
        plan_stats = plan._debug_runtime_stats()
        assert plan_stats["operations"]["workspace_builds"] == 1
        assert plan_stats["operations"]["workspace_reuses"] == 1
        assert plan_stats["operations"]["preconditioner_apply_calls"] > 0
        assert (
            plan_stats["identity"]["operator_action_provider"]
            == "forge_vulkan_native"
        )
        assert (
            plan_stats["identity"]["preconditioner_action_provider"]
            == "vulkan_block_jacobi"
        )
        assert plan_stats["identity"]["operator_asynchronous_submit"]
        assert plan_stats["identity"]["preconditioner_asynchronous_submit"]
        assert plan_stats["identity"]["preconditioner_behavior"] == "fixed_linear"
        assert plan_stats["operations"]["operator_generation_pins"] == 3
        assert plan_stats["operations"]["preconditioner_generation_pins"] == 3
        assert plan_stats["operations"]["preconditioner_setup_calls"] == 1
        assert plan_stats["operations"]["preconditioner_update_calls"] == 2
        assert plan_stats["operations"]["operator_plan_invalidations"] == 0
        assert (
            plan_stats["operations"]["preconditioner_plan_invalidations"]
            == 0
        )
    contract = matrix_a._get_format_contract()
    supports_public_cg = ti.lang.impl.current_cfg().arch in (ti.cpu, ti.cuda)
    assert contract["pattern"]["ownership"] == "shared_immutable"
    assert contract["operations"]["public_cg"] == supports_public_cg
    assert not contract["operations"]["public_direct_solver"]
    assert not contract["operations"]["public_jacobi_selection"]
    assert (
        contract["operations"]["public_block_jacobi_selection"]
        == supports_public_cg
    )
    assert contract["constraints"]["public_bsr_available"]
    assert not contract["constraints"]["public_builder_available"]
    assert not contract["constraints"]["silent_format_fallback"]
    pattern_stats = pattern._debug_runtime_stats()
    stats_a = matrix_a._debug_runtime_stats()
    stats_b = matrix_b._debug_runtime_stats()
    assert pattern_stats["lifecycle"]["operator_references"] == 2
    assert stats_a["identity"]["pattern_id"] == pattern_stats["identity"]["pattern_id"]
    assert stats_b["identity"]["pattern_id"] == pattern_stats["identity"]["pattern_id"]
    assert stats_a["identity"]["numeric_version"] == 2
    assert stats_b["identity"]["numeric_version"] == 1

    with pytest.raises(TaichiRuntimeError, match="no NumPy or host fallback"):
        pattern.matrix(identity_values)
    rank_two_values = ti.ndarray(dtype=ti.f32, shape=(2, 4))
    with pytest.raises(TaichiRuntimeError, match="one-dimensional"):
        pattern.matrix(rank_two_values)


@test_utils.test(arch=[ti.cpu], offline_cache=False)
def test_public_cpu_bsr_pattern_supports_f64_values():
    row_offsets = ti.ndarray(dtype=ti.i32, shape=2)
    column_indices = ti.ndarray(dtype=ti.i32, shape=1)
    row_offsets.from_numpy(np.asarray([0, 1], dtype=np.int32))
    column_indices.from_numpy(np.asarray([0], dtype=np.int32))
    pattern = ti.linalg.SparsePattern.bsr(1, 1, 2, row_offsets, column_indices)
    values = ti.ndarray(dtype=ti.f64, shape=4)
    values.from_numpy(np.eye(2, dtype=np.float64).reshape(-1))
    matrix = pattern.matrix(values)
    vector_host = np.asarray([1.25, -2.5], dtype=np.float64)
    vector = ti.ndarray(dtype=ti.f64, shape=2)
    vector.from_numpy(vector_host)
    np.testing.assert_allclose((matrix @ vector).to_numpy(), vector_host)
    assert matrix.dtype == ti.f64


@test_utils.test(arch=[ti.cpu], offline_cache=False)
def test_fixed_sparse_pattern_topology_fingerprint_is_stable_and_exact():
    def make_pattern(columns):
        row_offsets = ti.ndarray(dtype=ti.i32, shape=3)
        column_indices = ti.ndarray(dtype=ti.i32, shape=2)
        row_offsets.from_numpy(np.asarray([0, 1, 2], dtype=np.int32))
        column_indices.from_numpy(np.asarray(columns, dtype=np.int32))
        return ti.linalg.SparsePattern.csr(
            2, 2, row_offsets, column_indices
        )

    diagonal = make_pattern([0, 1])
    diagonal_copy = make_pattern([0, 1])
    off_diagonal = make_pattern([1, 0])
    diagonal_fingerprint = diagonal._debug_runtime_stats()["identity"][
        "topology_fingerprint"
    ]
    assert diagonal_fingerprint.startswith("tf-sp-v1:")
    assert (
        diagonal_copy._debug_runtime_stats()["identity"][
            "topology_fingerprint"
        ]
        == diagonal_fingerprint
    )
    assert (
        off_diagonal._debug_runtime_stats()["identity"][
            "topology_fingerprint"
        ]
        != diagonal_fingerprint
    )

    values = ti.ndarray(dtype=ti.f32, shape=2)
    values.fill(1.0)
    matrix = diagonal.matrix(values)
    assert (
        matrix._debug_runtime_stats()["identity"]["topology_fingerprint"]
        == diagonal_fingerprint
    )
    assert (
        matrix._get_format_contract()["identity"]["topology_fingerprint"]
        == diagonal_fingerprint
    )


@test_utils.test(arch=[ti.cpu], offline_cache=False)
def test_public_cpu_rectangular_bsr_keeps_solver_operations_disabled():
    row_offsets = ti.ndarray(dtype=ti.i32, shape=3)
    column_indices = ti.ndarray(dtype=ti.i32, shape=2)
    row_offsets.from_numpy(np.asarray([0, 1, 2], dtype=np.int32))
    column_indices.from_numpy(np.asarray([0, 2], dtype=np.int32))
    pattern = ti.linalg.SparsePattern.bsr(
        2, 3, 2, row_offsets, column_indices
    )
    values = ti.ndarray(dtype=ti.f32, shape=8)
    values.from_numpy(np.tile(np.eye(2, dtype=np.float32).reshape(-1), 2))
    matrix = pattern.matrix(values)
    contract = matrix._get_format_contract()
    assert matrix.shape == (4, 6)
    assert not contract["operations"]["public_cg"]
    assert not contract["operations"]["public_block_jacobi_selection"]
    assert not contract["operations"]["public_direct_solver"]
    with pytest.raises(
        TaichiRuntimeError,
        match="operation 'public_cg'.*no fallback was performed",
    ):
        ti.linalg.SparseCG(matrix, np.ones(4, dtype=np.float32))


@test_utils.test(arch=[ti.cpu], offline_cache=False)
def test_public_bsr_pattern_rejects_numpy_indices_without_fallback():
    with pytest.raises(TaichiRuntimeError, match="cannot be constructed directly"):
        ti.linalg.SparsePattern()

    with pytest.raises(TaichiRuntimeError, match="no NumPy or host fallback"):
        ti.linalg.SparsePattern.bsr(
            1,
            1,
            2,
            np.asarray([0, 1], dtype=np.int32),
            np.asarray([0], dtype=np.int32),
        )

    row_offsets = ti.ndarray(dtype=ti.i32, shape=2)
    column_indices = ti.ndarray(dtype=ti.i32, shape=1)
    with pytest.raises(TaichiRuntimeError, match="block_size must be one of"):
        ti.linalg.SparsePattern.bsr(1, 1, 4, row_offsets, column_indices)
    with pytest.raises(TaichiRuntimeError, match="must be an integer"):
        ti.linalg.SparsePattern.bsr(1.5, 1, 2, row_offsets, column_indices)
