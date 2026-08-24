import pytest

import taichi_forge as ti
from tests import test_utils


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
def test_sparse_matrix_vector_multiplication1(dtype, storage_format):
    n = 8
    Abuilder = ti.linalg.SparseMatrixBuilder(
        n, n, max_num_triplets=100, dtype=dtype, storage_format=storage_format
    )
    b = ti.field(ti.f32, shape=n)

    @ti.kernel
    def fill(Abuilder: ti.types.sparse_matrix_builder(), b: ti.template()):
        for i, j in ti.ndrange(n, n):
            Abuilder[i, j] += i

        for i in range(n):
            b[i] = 1.0

    fill(Abuilder, b)
    A = Abuilder.build()
    x = A @ b
    for i in range(n):
        assert x[i] == 8 * i


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
def test_sparse_matrix_vector_multiplication2(dtype, storage_format):
    n = 8
    Abuilder = ti.linalg.SparseMatrixBuilder(
        n, n, max_num_triplets=100, dtype=dtype, storage_format=storage_format
    )
    b = ti.field(ti.f32, shape=n)

    @ti.kernel
    def fill(Abuilder: ti.types.sparse_matrix_builder(), b: ti.template()):
        for i, j in ti.ndrange(n, n):
            Abuilder[i, j] += i - j

        for i in range(n):
            b[i] = 1.0

    fill(Abuilder, b)
    A = Abuilder.build()

    x = A @ b
    import numpy as np

    res = np.array([-28, -20, -12, -4, 4, 12, 20, 28])
    for i in range(n):
        assert x[i] == res[i]


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
def test_sparse_matrix_vector_multiplication3(dtype, storage_format):
    n = 8
    Abuilder = ti.linalg.SparseMatrixBuilder(
        n, n, max_num_triplets=100, dtype=dtype, storage_format=storage_format
    )
    b = ti.field(ti.f32, shape=n)

    @ti.kernel
    def fill(Abuilder: ti.types.sparse_matrix_builder(), b: ti.template()):
        for i, j in ti.ndrange(n, n):
            Abuilder[i, j] += i + j

        for i in range(n):
            b[i] = 1.0

    fill(Abuilder, b)
    A = Abuilder.build()

    x = A @ b
    import numpy as np

    res = np.array([28, 36, 44, 52, 60, 68, 76, 84])
    for i in range(n):
        assert x[i] == res[i]


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_sparse_matrix_operator_runtime_statistics():
    n = 4
    builder = ti.linalg.SparseMatrixBuilder(
        n, n, max_num_triplets=n, dtype=ti.f32, storage_format="row_major"
    )

    @ti.kernel
    def fill(matrix: ti.types.sparse_matrix_builder()):
        for i in range(n):
            matrix[i, i] += i + 1

    fill(builder)
    matrix = builder.build()
    before = matrix._debug_runtime_stats()
    identity = before["identity"]
    resources = before["resources"]
    operations = before["operations"]
    assert before["schema_version"] == 1
    assert identity["backend_family"] in ("cpu", "cuda")
    assert identity["rows"] == n
    assert identity["cols"] == n
    assert identity["nnz"] == n
    assert identity["pattern_version"] == 1
    assert identity["numeric_version"] == 1
    assert operations["pattern_builds"] == 1
    assert resources["pattern_reserved_bytes"] > 0
    assert resources["values_reserved_bytes"] >= n * 4
    assert resources["operator_owned_reserved_bytes"] == (
        resources["pattern_reserved_bytes"]
        + resources["values_reserved_bytes"]
        + resources["spmv_workspace_reserved_bytes"]
    )

    x = ti.ndarray(dtype=ti.f32, shape=n)
    y = ti.ndarray(dtype=ti.f32, shape=n)
    x.fill(1)
    prog = ti.lang.impl.get_runtime().prog
    matrix.matrix.spmv(prog, x.arr, y.arr)
    matrix.matrix.spmv(prog, x.arr, y.arr)
    ti.sync()

    after_spmv = matrix._debug_runtime_stats()
    operations = after_spmv["operations"]
    resources = after_spmv["resources"]
    provider = after_spmv["provider"]
    assert operations["spmv_calls"] == 2
    if identity["backend_family"] == "cpu":
        assert operations["spmv_plan_builds"] == 0
        assert operations["spmv_plan_reuses"] == 0
        assert resources["spmv_workspace_reserved_bytes"] == 0
        assert resources["matrix_descriptor_count"] == 0
    else:
        assert operations["spmv_plan_builds"] == 1
        assert operations["spmv_plan_reuses"] == 1
        assert operations["spmv_handle_creations"] == 1
        assert operations["dense_vector_descriptor_creations"] == 2
        assert operations["dense_vector_descriptor_rebinds"] == 0
        assert resources["matrix_descriptor_count"] == 1
        assert resources["dense_vector_descriptor_count"] == 2
        assert resources["spmv_handle_count"] == 1
        # CUDA transactional assembly keeps builder payloads device-resident.
        assert after_spmv["transfers"]["host_to_device_bytes"] == 0
        assert after_spmv["transfers"]["device_to_host_bytes"] == 0
        assert after_spmv["transfers"]["device_to_device_bytes"] > 0
        if provider["spmv_preprocess_available"]:
            assert provider["spmv_preprocess_active"]
            assert provider["spmv_preprocess_last_error"] == 0
            assert operations["spmv_preprocess_builds"] == 1
            assert operations["spmv_preprocess_reuses"] == 1
            assert operations["spmv_preprocess_fallbacks"] == 0
        else:
            assert not provider["spmv_preprocess_active"]
            assert operations["spmv_preprocess_builds"] == 0
            assert operations["spmv_preprocess_reuses"] == 0
            assert operations["spmv_preprocess_fallbacks"] == 2

    values = ti.ndarray(dtype=ti.f32, shape=n)
    values.fill(2)
    matrix._update_values(values)
    matrix.matrix.spmv(prog, x.arr, y.arr)
    ti.sync()

    import numpy as np

    np.testing.assert_allclose(y.to_numpy(), np.full(n, 2.0, dtype=np.float32))
    updated = matrix._debug_runtime_stats()
    assert updated["identity"]["pattern_version"] == 1
    assert updated["identity"]["numeric_version"] == 2
    assert updated["operations"]["numeric_updates"] == 1
    assert updated["operations"]["numeric_update_bytes"] == n * 4
    assert updated["operations"]["spmv_calls"] == 3
    if identity["backend_family"] == "cuda":
        if provider["spmv_preprocess_available"]:
            assert updated["operations"]["spmv_preprocess_builds"] == 1
            assert updated["operations"]["spmv_preprocess_reuses"] == 2
            assert updated["operations"]["spmv_preprocess_fallbacks"] == 0
        else:
            assert updated["operations"]["spmv_preprocess_builds"] == 0
            assert updated["operations"]["spmv_preprocess_reuses"] == 0
            assert updated["operations"]["spmv_preprocess_fallbacks"] == 3
        assert updated["transfers"]["device_to_device_bytes"] >= n * 4


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_spmv_preprocess_runtime_disable(monkeypatch):
    monkeypatch.setenv("TI_CUDA_CUSPARSE_SPMV_PREPROCESS", "0")
    n = 4
    builder = ti.linalg.SparseMatrixBuilder(
        n, n, max_num_triplets=n, dtype=ti.f32, storage_format="row_major"
    )

    @ti.kernel
    def fill(matrix: ti.types.sparse_matrix_builder()):
        for i in range(n):
            matrix[i, i] += i + 1

    fill(builder)
    matrix = builder.build()
    x = ti.ndarray(dtype=ti.f32, shape=n)
    y = ti.ndarray(dtype=ti.f32, shape=n)
    x.fill(1)
    prog = ti.lang.impl.get_runtime().prog
    matrix.matrix.spmv(prog, x.arr, y.arr)
    matrix.matrix.spmv(prog, x.arr, y.arr)
    ti.sync()

    stats = matrix._debug_runtime_stats()
    provider = stats["provider"]
    operations = stats["operations"]
    assert not provider["spmv_preprocess_active"]
    assert provider["spmv_preprocess_last_error"] == 0
    assert operations["spmv_preprocess_builds"] == 0
    assert operations["spmv_preprocess_reuses"] == 0
    assert operations["spmv_preprocess_fallbacks"] == 2


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_sparse_domain_auto_spmv_requires_scoped_provider_evidence():
    import numpy as np

    from taichi_forge.hardware import ProviderAdmissionEvidence
    from taichi_forge.hardware._admission import (
        _current_cuda_device_scope,
        _current_runtime_scope,
    )

    n = 8
    row_offsets_host = np.empty(n + 1, dtype=np.int32)
    columns_host = []
    values_host = []
    row_offsets_host[0] = 0
    for row in range(n):
        columns_host.append(row)
        values_host.append(row + 1)
        if row + 1 < n:
            columns_host.append(row + 1)
            values_host.append(0.5)
        row_offsets_host[row + 1] = len(columns_host)
    columns_host = np.asarray(columns_host, dtype=np.int32)
    values_host = np.asarray(values_host, dtype=np.float32)
    row_offsets = ti.ndarray(dtype=ti.i32, shape=n + 1)
    columns = ti.ndarray(dtype=ti.i32, shape=len(columns_host))
    values = ti.ndarray(dtype=ti.f32, shape=len(values_host))
    row_offsets.from_numpy(row_offsets_host)
    columns.from_numpy(columns_host)
    values.from_numpy(values_host)
    matrix = ti.linalg.SparsePattern.csr(n, n, row_offsets, columns).matrix(values)
    vector_values = np.arange(1, n + 1, dtype=np.float32)
    vector = ti.ndarray(dtype=ti.f32, shape=n)
    vector.from_numpy(vector_values)
    expected = np.arange(1, n + 1, dtype=np.float32) * vector_values
    expected[:-1] += 0.5 * vector_values[1:]

    fallback = matrix @ vector
    ti.sync()
    np.testing.assert_allclose(fallback.to_numpy(), expected, rtol=1e-6)
    stats = matrix._debug_runtime_stats()
    assert stats["auto_provider"]["candidates"] == 1
    assert stats["auto_provider"]["admitted"] == 0
    assert stats["auto_provider"]["kernel_fallbacks"] == 1
    assert stats["auto_provider"]["rejection_reasons"] == {
        "missing_admission_evidence": 1
    }
    assert stats["operations"]["spmv_plan_builds"] == 0

    def profile(topology_fingerprint):
        native = matrix._debug_runtime_stats()
        identity = native["identity"]
        provider = native["provider"]
        return ProviderAdmissionEvidence._from_record(
            {
                "schema": "taichi_forge.provider_admission.v1",
                "schema_version": 1,
                "operation_id": "linalg.spmv.cusparse",
                "provider_id": "cusparse",
                "baseline_id": "cuda_driver_kernel",
                "backend": "cuda",
                "device_scope": _current_cuda_device_scope(),
                "provider_scope": {
                    "provider_abi": "cusparse-dynamic-symbols-v1",
                    "provider_version": provider["library_version"],
                },
                "workload_scope": {
                    "rows": identity["rows"],
                    "cols": identity["cols"],
                    "nnz": identity["nnz"],
                    "storage_format": identity["storage_format"],
                    "block_size": identity["block_size"],
                    "topology_fingerprint": topology_fingerprint,
                },
                "runtime_scope": _current_runtime_scope(),
                "performance": {
                    "expected_reuse": 16,
                    "provider_median_ns": 50.0,
                    "baseline_median_ns": 100.0,
                    "provider_first_use_overhead_ns": 0.0,
                    "baseline_first_use_overhead_ns": 0.0,
                    "transfer_ns": 0.0,
                    "conversion_ns": 0.0,
                    "provider_samples": 48,
                    "baseline_samples": 48,
                    "provider_cv": 0.02,
                    "baseline_cv": 0.02,
                    "order_drift": 0.01,
                    "minimum_block_ms": 100.0,
                    "minimum_margin": 0.05,
                    "paired_p05": 1.8,
                    "fresh_processes": 8,
                    "order_processes": {"ab": 4, "ba": 4},
                },
                "qualification": {
                    "correctness_and_route_qualified": True,
                    "stable": True,
                    "minimum_block_qualified": True,
                },
            },
            source_schema=("taichi_forge.hardware_acceleration_qualification.v4"),
            source_digest="test-artifact",
        )

    matrix.set_provider_profile(profile("tf-sp-v1:wrong-topology"))
    rejected = matrix @ vector
    ti.sync()
    np.testing.assert_allclose(rejected.to_numpy(), expected, rtol=1e-6)
    stats = matrix._debug_runtime_stats()
    assert stats["auto_provider"]["rejection_reasons"]["workload_scope_mismatch"] == 1
    assert stats["operations"]["spmv_plan_builds"] == 0

    fingerprint = matrix._debug_runtime_stats()["identity"]["topology_fingerprint"]
    matrix.set_provider_profile(profile(fingerprint))
    admitted = matrix @ vector
    ti.sync()
    np.testing.assert_allclose(admitted.to_numpy(), expected, rtol=1e-6)
    stats = matrix._debug_runtime_stats()
    assert stats["auto_provider"]["candidates"] == 3
    assert stats["auto_provider"]["admitted"] == 1
    assert stats["auto_provider"]["last_decision"]["route"] == "provider"
    assert stats["auto_provider"]["last_decision"]["reason"] == (
        "qualified_cost_advantage"
    )
    assert stats["operations"]["spmv_plan_builds"] == 1
    assert not hasattr(matrix, "set_spmv_auto_cost_evidence")

    matrix.set_provider_profile(None)
    explicit = matrix.spmv(vector, method="provider")
    ti.sync()
    np.testing.assert_allclose(explicit.to_numpy(), expected, rtol=1e-6)
    stats = matrix._debug_runtime_stats()
    assert stats["auto_provider"]["candidates"] == 3
    assert not stats["auto_provider"]["provider_profile_present"]
