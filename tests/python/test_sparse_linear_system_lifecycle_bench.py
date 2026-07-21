import importlib.util
from pathlib import Path

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils


_REPO_ROOT = Path(__file__).resolve().parents[2]
_BENCH_PATH = (
    _REPO_ROOT / "benchmarks" / "sparse_linear_system_lifecycle_bench.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "sparse_linear_system_lifecycle_bench", _BENCH_PATH
)
sparse_linear_system_lifecycle_bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sparse_linear_system_lifecycle_bench)


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_repeated_poisson_solve_reuses_fixed_pattern_resources():
    report = sparse_linear_system_lifecycle_bench.run_initialized(
        ti, n=16, max_iter=64, atol=1e-5
    )

    assert report["schema"] == (
        "taichi_forge.sparse_linear_system_lifecycle.v1"
    )
    assert report["schema_version"] == 1
    assert report["correct"]
    assert report["supported"]
    assert report["phase_order"] == list(
        sparse_linear_system_lifecycle_bench.PHASES
    )
    assert set(report["phases"]) == set(report["phase_order"])
    assert not report["performance_valid"]
    assert report["config"]["rhs_count"] == 3
    assert report["operator_final"]["identity"]["pattern_version"] == 1
    assert report["operator_final"]["identity"]["numeric_version"] == 2
    assert report["operator_final"]["operations"]["numeric_updates"] == 1
    assert report["plan_final"]["operations"]["solve_calls"] == 3
    assert report["checks"][
        "operator_resources_stable_across_numeric_update"
    ]
    assert report["checks"]["plan_resources_stable_after_first_solve"]
    assert report["checks"][
        "numeric_update_marks_plan_stale_until_next_solve"
    ]

    for name in (
        "first_rhs_solve",
        "second_rhs_solve",
        "updated_values_solve",
    ):
        phase = report["phases"][name]
        assert phase["converged"]
        assert phase["solution_error_linf"] <= 2e-4
        assert phase["reference_residual_norm"] <= 4e-5

    if report["arch"] == "cpu":
        assert not report["plan_final"]["resources"][
            "solver_state_rebuilt_each_solve"
        ]
        assert report["plan_final"]["operations"]["operator_apply_calls"] is None
        assert report["plan_final"]["operations"]["workspace_builds"] == 2
        assert report["plan_final"]["operations"]["workspace_reuses"] == 1
    else:
        operations = report["plan_final"]["operations"]
        assert operations["workspace_builds"] == 1
        assert operations["workspace_reuses"] == 2
        assert operations["host_scalar_reductions"] > 0
        assembly = report["assembly_probe"]
        assert assembly["supported"]
        assert assembly["correct"]
        assert assembly["unique_nnz"] == 8
        assert assembly["duplicate_triplets"] == 2
        assert assembly["contains_empty_row"]
        assert assembly["first_spmv_error_linf"] <= 1e-6
        assert assembly[
            "retained_matrix_error_after_failed_build_linf"
        ] <= 1e-6
        assembly_final = assembly["plan_final"]
        assert assembly_final["identity"]["backend_family"] == "cuda"
        assert assembly_final["operations"]["build_calls"] == 2
        assert assembly_final["operations"]["successful_builds"] == 1
        assert assembly_final["operations"]["failed_builds"] == 1
        assert assembly_final["operations"]["workspace_reuses"] == 1
        assert assembly_final["operations"]["host_synchronizations"] == 2
        assert assembly_final["operations"]["host_control_readbacks"] == 2
        assert assembly_final["transfers"]["device_to_host_bytes"] == 16
        assert (
            assembly_final["transfers"]["device_payload_readback_bytes"] == 0
        )
        assert assembly_final["contract"]["transactional_publish"]
        assert assembly_final["contract"]["exact_sized_published_csr"]
        assert not assembly_final["contract"]["cuda_toolkit_required"]


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_sparse_linear_system_capability_is_explicit():
    report = sparse_linear_system_lifecycle_bench.run_initialized(ti, n=16)

    assert report["correct"]
    assert not report["supported"]
    assert report["phase_order"] == []
    assert report["phases"] == {}
    assert "fixed_pattern_csr_spmv" in report["capability"][
        "available_primitives"
    ]
    assert "csr_or_bsr_spmv" not in report["capability"][
        "missing_primitives"
    ]
    assert "f32_axpy" in report["capability"]["available_primitives"]
    assert "f32_dot" in report["capability"]["available_primitives"]
    assert "f32_norm" in report["capability"]["available_primitives"]
    assert "device_scalar_fixed_iteration_cg_plan" in report[
        "capability"
    ]["available_primitives"]
    assert "device_convergence_bounded_cg_plan" in report[
        "capability"
    ]["available_primitives"]
    assert "device_resident_bounded_triplet_to_csr" in report[
        "capability"
    ]["available_primitives"]
    assert "transactional_exact_sized_csr_publish" in report[
        "capability"
    ]["available_primitives"]
    assert "conditional_dispatch_exit" in report["capability"][
        "missing_primitives"
    ]
    assert "public_sparse_builder" in report["capability"][
        "missing_primitives"
    ]
    assert "public_sparse_cg" in report["capability"][
        "missing_primitives"
    ]
    probe = report["operator_probe"]
    assert probe["supported"]
    assert probe["correct"]
    assert probe["first_spmv_error_linf"] <= 2e-5
    assert probe["second_spmv_error_linf"] <= 2e-5
    assert probe["updated_spmv_error_linf"] <= 3e-5
    assert probe["resources_stable_across_numeric_update"]
    operator = probe["operator_final"]
    assert operator["identity"]["backend_family"] == "vulkan"
    assert operator["identity"]["storage_format"] == "csr"
    assert operator["identity"]["pattern_version"] == 1
    assert operator["identity"]["numeric_version"] == 2
    assert operator["operations"]["spmv_calls"] == 4
    assert operator["operations"]["spmv_plan_builds"] == 1
    assert operator["operations"]["spmv_plan_reuses"] == 3
    assert operator["provider"]["name"] == "forge_vulkan_native"
    iteration = probe["minimal_iteration"]
    assert iteration["correct"]
    assert iteration["residual_dot_relative_error"] <= 2e-5
    assert iteration["direction_dot_relative_error"] <= 2e-5
    assert iteration["alpha_relative_error"] <= 3e-5
    assert iteration["iterate_error_linf"] <= 3e-5
    assert iteration["residual_error_linf"] <= 5e-5
    assert iteration["norm_error_abs"] <= 5e-5
    assert iteration["workspace_stable"]
    assert (
        iteration["sparse_workspace_bytes"]
        == iteration["expected_partial_bytes"]
    )
    plan = report["iteration_plan_probe"]
    assert plan["supported"]
    assert plan["fixed_iteration_only"]
    assert plan["correct"]
    assert plan["resources_stable_across_numeric_update"]
    assert plan["stale_after_value_update"]["identity"][
        "operator_numeric_changed_since_last_solve"
    ]
    plan_final = plan["plan_final"]
    assert plan_final["identity"]["method"] == "cg_fixed_iteration_probe"
    assert plan_final["identity"]["operator_pattern_version"] == 1
    assert plan_final["identity"]["operator_numeric_version"] == 3
    assert plan_final["operations"]["solve_calls"] == 3
    assert plan_final["operations"]["total_iterations"] == 12
    assert plan_final["operations"]["workspace_builds"] == 1
    assert plan_final["operations"]["workspace_reuses"] == 2
    assert plan_final["operations"]["operator_apply_calls"] == 15
    assert plan_final["operations"]["host_scalar_reductions"] == 0
    assert plan_final["operations"]["host_scalar_readbacks"] == 9
    assert plan_final["operations"]["host_synchronizations"] == 3
    assert plan_final["operations"]["device_scalar_operations"] == 42
    assert plan_final["resources"]["persistent_vector_count"] == 3
    assert plan_final["resources"]["persistent_scalar_count"] == 9
    adaptive = report["adaptive_plan_probe"]
    assert adaptive["supported"]
    assert adaptive["bounded_masked_execution"]
    assert adaptive["correct"]
    assert adaptive["first"]["success"]
    assert adaptive["first"]["status"] == 2
    assert adaptive["initially_converged"]["iterations"] == 0
    adaptive_final = adaptive["plan_final"]
    assert adaptive_final["identity"]["method"] == "cg_bounded_masked_probe"
    assert adaptive_final["operations"]["solve_calls"] == 2
    assert adaptive_final["operations"]["host_synchronizations"] == 2
    assert adaptive_final["operations"]["host_scalar_readbacks"] == 8
    assert adaptive_final["operations"]["bounded_masked_execution"]
    assert adaptive_final["resources"]["persistent_scalar_count"] == 11
    assembly = report["assembly_probe"]
    assert assembly["supported"]
    assert assembly["correct"]
    assert assembly["input_triplets"] == 10
    assert assembly["unique_nnz"] == 8
    assert assembly["duplicate_triplets"] == 2
    assert assembly["contains_empty_row"]
    assert assembly["first_spmv_error_linf"] <= 1e-6
    assert assembly[
        "retained_matrix_error_after_failed_build_linf"
    ] <= 1e-6
    assert assembly["failure_status"] == 1
    assembly_final = assembly["plan_final"]
    assert assembly_final["operations"]["build_calls"] == 2
    assert assembly_final["operations"]["successful_builds"] == 1
    assert assembly_final["operations"]["failed_builds"] == 1
    assert assembly_final["operations"]["workspace_reuses"] == 1
    assert assembly_final["operations"]["host_synchronizations"] == 2
    assert assembly_final["operations"]["host_control_readbacks"] == 2
    assert assembly_final["transfers"]["device_to_host_bytes"] == 16
    assert assembly_final["transfers"]["device_payload_readback_bytes"] == 0
    assert assembly_final["contract"]["transactional_publish"]


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_vulkan_sparse_assembly_plan_rejects_non_vulkan_backend():
    prog = ti.lang.impl.get_runtime().prog

    assert not prog._vulkan_sparse_assembly_available()
    with pytest.raises(RuntimeError, match="active Vulkan Program"):
        ti._lib.core._make_vulkan_sparse_assembly_plan(prog, 4, 4, 8)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_cuda_sparse_assembly_plan_rejects_non_cuda_backend():
    prog = ti.lang.impl.get_runtime().prog

    assert not prog._cuda_sparse_assembly_available()
    with pytest.raises(RuntimeError, match="active CUDA Program"):
        ti._lib.core._make_cuda_sparse_assembly_plan(prog, 4, 4, 8)


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_device_sparse_assembly_is_bounded_reusable_and_transactional():
    rows = 5
    cols = 5
    triplet_rows_host = np.asarray(
        [3, 0, 1, 1, 0, 3, 2, 1, 0, 2], dtype=np.int32
    )
    triplet_columns_host = np.asarray(
        [4, 0, 2, 2, 3, 1, 1, 0, 0, 4], dtype=np.int32
    )
    triplet_values_host = np.asarray(
        [2.0, 1.0, 1.25, 2.75, -1.0, 3.0, 4.0, 5.0, 0.5, -2.0],
        dtype=np.float32,
    )
    capacity = triplet_rows_host.size
    triplet_rows = ti.ndarray(dtype=ti.i32, shape=capacity)
    triplet_columns = ti.ndarray(dtype=ti.i32, shape=capacity)
    triplet_values = ti.ndarray(dtype=ti.f32, shape=capacity)
    triplet_rows.from_numpy(triplet_rows_host)
    triplet_columns.from_numpy(triplet_columns_host)
    triplet_values.from_numpy(triplet_values_host)
    prog = ti.lang.impl.get_runtime().prog
    if ti.lang.impl.current_cfg().arch == ti.cuda:
        assert prog._cuda_sparse_assembly_available()
        factory = ti._lib.core._make_cuda_sparse_assembly_plan
    else:
        assert prog._vulkan_sparse_assembly_available()
        factory = ti._lib.core._make_vulkan_sparse_assembly_plan
    plan = factory(prog, rows, cols, capacity)

    def dense_from_triplets(row_data, column_data, value_data):
        dense = np.zeros((rows, cols), dtype=np.float32)
        for row, column, value in zip(row_data, column_data, value_data):
            dense[row, column] += value
        return dense

    x_host = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    x = ti.ndarray(dtype=ti.f32, shape=cols)
    y = ti.ndarray(dtype=ti.f32, shape=rows)
    x.from_numpy(x_host)

    first = plan.build(
        prog, triplet_rows.arr, triplet_columns.arr, triplet_values.arr
    )
    first.spmv(prog, x.arr, y.arr)
    np.testing.assert_allclose(
        y.to_numpy(),
        dense_from_triplets(
            triplet_rows_host, triplet_columns_host, triplet_values_host
        )
        @ x_host,
        rtol=1e-6,
        atol=1e-6,
    )
    assert first.num_nonzero() == 8
    first_matrix_stats = first._debug_runtime_stats()
    assert first_matrix_stats["resources"]["pattern_reserved_bytes"] == 56
    assert first_matrix_stats["resources"]["values_reserved_bytes"] == 32
    assert first_matrix_stats["transfers"]["device_to_host_bytes"] == 0
    assert first_matrix_stats["transfers"]["device_to_device_bytes"] == 88

    first_stats = plan._debug_runtime_stats()
    assert first_stats["identity"]["method"] == (
        "radix_sort_segment_reduce_csr"
    )
    assert first_stats["status"]["last_status"] == 0
    assert first_stats["status"]["last_input_triplets"] == capacity
    assert first_stats["status"]["last_unique_nnz"] == 8
    assert first_stats["status"]["last_duplicate_triplets"] == 2
    assert first_stats["operations"]["build_calls"] == 1
    assert first_stats["operations"]["successful_builds"] == 1
    assert first_stats["operations"]["workspace_builds"] == 1
    assert first_stats["operations"]["workspace_reuses"] == 0
    assert first_stats["operations"]["host_synchronizations"] == 1
    assert first_stats["operations"]["host_control_readbacks"] == 1
    assert first_stats["operations"]["host_scalar_readbacks"] == 2
    expected_staging_bytes = 400
    assert first_stats["resources"][
        "persistent_workspace_reserved_bytes"
    ] == expected_staging_bytes
    assert first_stats["resources"][
        "shared_radix_sort_workspace_reserved_bytes"
    ] > 0
    scan_workspace_bytes = first_stats["resources"][
        "shared_scan_workspace_reserved_bytes"
    ]
    if ti.lang.impl.current_cfg().arch == ti.cuda:
        assert scan_workspace_bytes == 0
    else:
        assert scan_workspace_bytes > 0
    assert first_stats["transfers"]["host_to_device_bytes"] == 0
    assert first_stats["transfers"]["device_to_host_bytes"] == 8
    assert first_stats["transfers"]["device_payload_readback_bytes"] == 0
    assert first_stats["transfers"]["device_to_device_bytes"] == 88
    assert first_stats["contract"]["transactional_publish"]
    assert first_stats["contract"]["exact_sized_published_csr"]

    second_rows_host = np.asarray(
        [4, 0, 2, 0, 2, 2, 1, 1, 3, 4], dtype=np.int32
    )
    second_columns_host = np.asarray(
        [4, 1, 0, 1, 0, 3, 2, 4, 1, 0], dtype=np.int32
    )
    second_values_host = np.asarray(
        [1.0, 2.0, 3.0, -0.5, 1.0, 4.0, -2.0, 5.0, 6.0, 7.0],
        dtype=np.float32,
    )
    triplet_rows.from_numpy(second_rows_host)
    triplet_columns.from_numpy(second_columns_host)
    triplet_values.from_numpy(second_values_host)
    second = plan.build(
        prog, triplet_rows.arr, triplet_columns.arr, triplet_values.arr
    )
    second.spmv(prog, x.arr, y.arr)
    np.testing.assert_allclose(
        y.to_numpy(),
        dense_from_triplets(
            second_rows_host, second_columns_host, second_values_host
        )
        @ x_host,
        rtol=1e-6,
        atol=1e-6,
    )
    assert second.num_nonzero() == 8

    # Reusing plan staging must not mutate a matrix already published from it.
    first.spmv(prog, x.arr, y.arr)
    np.testing.assert_allclose(
        y.to_numpy(),
        dense_from_triplets(
            triplet_rows_host, triplet_columns_host, triplet_values_host
        )
        @ x_host,
        rtol=1e-6,
        atol=1e-6,
    )

    invalid_rows_host = second_rows_host.copy()
    invalid_rows_host[3] = rows
    triplet_rows.from_numpy(invalid_rows_host)
    with pytest.raises(RuntimeError, match="index outside"):
        plan.build(
            prog, triplet_rows.arr, triplet_columns.arr, triplet_values.arr
        )

    # A failed transaction cannot corrupt the last successfully published CSR.
    second.spmv(prog, x.arr, y.arr)
    np.testing.assert_allclose(
        y.to_numpy(),
        dense_from_triplets(
            second_rows_host, second_columns_host, second_values_host
        )
        @ x_host,
        rtol=1e-6,
        atol=1e-6,
    )
    final_stats = plan._debug_runtime_stats()
    assert final_stats["status"]["last_status"] == 1
    assert final_stats["operations"]["build_calls"] == 3
    assert final_stats["operations"]["successful_builds"] == 2
    assert final_stats["operations"]["failed_builds"] == 1
    assert final_stats["operations"]["workspace_builds"] == 1
    assert final_stats["operations"]["workspace_reuses"] == 2
    assert final_stats["operations"]["host_synchronizations"] == 3
    assert final_stats["operations"]["host_control_readbacks"] == 3
    assert final_stats["operations"]["host_scalar_readbacks"] == 6
    assert final_stats["operations"][
        "workspace_growth_synchronizations"
    ] <= 1
    assert final_stats["transfers"]["device_to_host_bytes"] == 24
    assert final_stats["transfers"]["device_to_device_bytes"] == 176


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_device_sparse_assembly_reports_nonfinite_failures_before_publish():
    triplet_rows = ti.ndarray(dtype=ti.i32, shape=2)
    triplet_columns = ti.ndarray(dtype=ti.i32, shape=2)
    triplet_values = ti.ndarray(dtype=ti.f32, shape=2)
    triplet_rows.from_numpy(np.zeros(2, dtype=np.int32))
    triplet_columns.from_numpy(np.zeros(2, dtype=np.int32))
    prog = ti.lang.impl.get_runtime().prog
    if ti.lang.impl.current_cfg().arch == ti.cuda:
        factory = ti._lib.core._make_cuda_sparse_assembly_plan
    else:
        factory = ti._lib.core._make_vulkan_sparse_assembly_plan
    plan = factory(prog, 1, 1, 2)

    triplet_values.from_numpy(
        np.asarray([np.inf, 1.0], dtype=np.float32)
    )
    with pytest.raises(RuntimeError, match="non-finite input"):
        plan.build(
            prog, triplet_rows.arr, triplet_columns.arr, triplet_values.arr
        )

    triplet_values.from_numpy(
        np.asarray([3.0e38, 3.0e38], dtype=np.float32)
    )
    with pytest.raises(RuntimeError, match="non-finite duplicate sum"):
        plan.build(
            prog, triplet_rows.arr, triplet_columns.arr, triplet_values.arr
        )

    triplet_values.from_numpy(np.asarray([1.0, 2.0], dtype=np.float32))
    matrix = plan.build(
        prog, triplet_rows.arr, triplet_columns.arr, triplet_values.arr
    )
    x = ti.ndarray(dtype=ti.f32, shape=1)
    y = ti.ndarray(dtype=ti.f32, shape=1)
    x.from_numpy(np.asarray([4.0], dtype=np.float32))
    matrix.spmv(prog, x.arr, y.arr)
    np.testing.assert_allclose(
        y.to_numpy(), np.asarray([12.0], dtype=np.float32)
    )
    stats = plan._debug_runtime_stats()
    assert stats["status"]["last_status"] == 0
    assert stats["status"]["last_unique_nnz"] == 1
    assert stats["operations"]["build_calls"] == 3
    assert stats["operations"]["successful_builds"] == 1
    assert stats["operations"]["failed_builds"] == 2
    assert stats["transfers"]["device_to_host_bytes"] == 24
    assert stats["transfers"]["device_to_device_bytes"] == 16


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_fixed_csr_rejects_duplicate_columns_before_ownership():
    row_offsets_host = np.asarray([0, 2, 2], dtype=np.int32)
    column_indices_host = np.asarray([0, 0], dtype=np.int32)
    values_host = np.arange(2, dtype=np.float32)
    row_offsets = ti.ndarray(dtype=ti.i32, shape=3)
    column_indices = ti.ndarray(dtype=ti.i32, shape=2)
    values = ti.ndarray(dtype=ti.f32, shape=2)
    row_offsets.from_numpy(row_offsets_host)
    column_indices.from_numpy(column_indices_host)
    values.from_numpy(values_host)
    prog = ti.lang.impl.get_runtime().prog

    with pytest.raises(RuntimeError, match="strictly increasing and unique"):
        prog._create_vulkan_csr_matrix(
            2,
            2,
            row_offsets.arr,
            column_indices.arr,
            values.arr,
        )

    np.testing.assert_array_equal(row_offsets.to_numpy(), row_offsets_host)
    np.testing.assert_array_equal(
        column_indices.to_numpy(), column_indices_host
    )
    np.testing.assert_array_equal(values.to_numpy(), values_host)


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_sparse_axpy_rejects_alias_without_mutation():
    values_host = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
    values = ti.ndarray(dtype=ti.f32, shape=values_host.size)
    values.from_numpy(values_host)
    prog = ti.lang.impl.get_runtime().prog

    with pytest.raises(RuntimeError, match="aliased x/y"):
        prog._vulkan_sparse_axpy(
            values.arr,
            values.arr,
            values_host.size,
            0.5,
        )

    np.testing.assert_array_equal(values.to_numpy(), values_host)


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_sparse_vector_algebra_multigroup():
    n = 2053
    x_host = np.linspace(0.1, 1.1, n, dtype=np.float32)
    y_host = np.linspace(-0.3, 0.7, n, dtype=np.float32)
    dst_host = np.linspace(1.0, -1.0, n, dtype=np.float32)
    x = ti.ndarray(dtype=ti.f32, shape=n)
    y = ti.ndarray(dtype=ti.f32, shape=n)
    dst = ti.ndarray(dtype=ti.f32, shape=n)
    dot_output = ti.ndarray(dtype=ti.f32, shape=1)
    norm_output = ti.ndarray(dtype=ti.f32, shape=1)
    x.from_numpy(x_host)
    y.from_numpy(y_host)
    dst.from_numpy(dst_host)
    prog = ti.lang.impl.get_runtime().prog

    dot_workspace = prog._vulkan_sparse_dot(
        x.arr, y.arr, dot_output.arr, n
    )
    norm_workspace = prog._vulkan_sparse_norm(
        x.arr, norm_output.arr, n
    )
    prog._vulkan_sparse_axpy(x.arr, dst.arr, n, 0.25)
    ti.sync()

    np.testing.assert_allclose(
        dot_output.to_numpy()[0],
        np.dot(x_host, y_host),
        rtol=3e-5,
        atol=3e-4,
    )
    np.testing.assert_allclose(
        norm_output.to_numpy()[0],
        np.linalg.norm(x_host),
        rtol=3e-5,
        atol=3e-4,
    )
    np.testing.assert_allclose(
        dst.to_numpy(),
        dst_host + np.float32(0.25) * x_host,
        rtol=2e-6,
        atol=2e-6,
    )
    expected_partial_bytes = ((n + 1023) // 1024) * 4
    sparse_workspace = prog._vulkan_sparse_algebra_workspace_bytes()
    reduce_workspace = prog.vulkan_reduce_workspace_bytes()
    assert sparse_workspace == expected_partial_bytes
    assert dot_workspace == sparse_workspace + reduce_workspace
    assert norm_workspace == dot_workspace


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_fixed_iteration_plan_reports_non_spd_breakdown():
    row_offsets_host = np.asarray([0, 1, 2], dtype=np.int32)
    column_indices_host = np.asarray([0, 1], dtype=np.int32)
    values_host = np.asarray([-1.0, -1.0], dtype=np.float32)
    row_offsets = ti.ndarray(dtype=ti.i32, shape=3)
    column_indices = ti.ndarray(dtype=ti.i32, shape=2)
    values = ti.ndarray(dtype=ti.f32, shape=2)
    solution = ti.ndarray(dtype=ti.f32, shape=2)
    rhs = ti.ndarray(dtype=ti.f32, shape=2)
    row_offsets.from_numpy(row_offsets_host)
    column_indices.from_numpy(column_indices_host)
    values.from_numpy(values_host)
    solution.from_numpy(np.zeros(2, dtype=np.float32))
    rhs.from_numpy(np.ones(2, dtype=np.float32))
    prog = ti.lang.impl.get_runtime().prog
    matrix = prog._create_vulkan_csr_matrix(
        2,
        2,
        row_offsets.arr,
        column_indices.arr,
        values.arr,
    )
    plan = ti._lib.core._make_vulkan_cg_convergence_plan(
        prog, matrix, 2, 1e-6
    )
    not_run = dict(plan._get_last_result())
    assert not_run["status_code"] == -1
    assert not_run["termination_reason"] == "not_run"

    plan.solve(prog, solution.arr, rhs.arr)

    assert not plan.is_success()
    assert plan.get_status() == 1
    assert plan.get_iterations() == 0
    result = dict(plan._get_last_result())
    assert result["status_code"] == 1
    assert result["termination_reason"] == "breakdown"
    assert result["breakdown"]
    assert not result["converged"]
    assert not result["reached_max_iterations"]
    np.testing.assert_array_equal(
        solution.to_numpy(), np.zeros(2, dtype=np.float32)
    )
    stats = plan._debug_runtime_stats()
    assert stats["operations"]["solve_calls"] == 1
    assert stats["operations"]["host_synchronizations"] == 1
    assert stats["operations"]["host_scalar_readbacks"] == 4
    assert stats["operations"]["bounded_masked_execution"]


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_bounded_convergence_plan_states():
    n = 16
    row_offsets_host, column_indices_host, values_host = (
        sparse_linear_system_lifecycle_bench._poisson_csr_pattern(n)
    )
    row_offsets = ti.ndarray(dtype=ti.i32, shape=n + 1)
    column_indices = ti.ndarray(
        dtype=ti.i32, shape=column_indices_host.size
    )
    values = ti.ndarray(dtype=ti.f32, shape=values_host.size)
    solution = ti.ndarray(dtype=ti.f32, shape=n)
    rhs = ti.ndarray(dtype=ti.f32, shape=n)
    row_offsets.from_numpy(row_offsets_host)
    column_indices.from_numpy(column_indices_host)
    values.from_numpy(values_host)
    prog = ti.lang.impl.get_runtime().prog
    matrix = prog._create_vulkan_csr_matrix(
        n,
        n,
        row_offsets.arr,
        column_indices.arr,
        values.arr,
    )
    max_iterations = 16
    tolerance = 1e-4
    plan = ti._lib.core._make_vulkan_cg_convergence_plan(
        prog, matrix, max_iterations, tolerance
    )
    exact = np.sin(
        np.linspace(0.15, 2.5, n, dtype=np.float32)
    ).astype(np.float32)
    rhs_host = sparse_linear_system_lifecycle_bench._poisson_apply(
        exact
    )
    solution.from_numpy(np.zeros(n, dtype=np.float32))
    rhs.from_numpy(rhs_host)

    plan.solve(prog, solution.arr, rhs.arr)

    assert plan.is_success()
    assert plan.get_status() == 2
    result = dict(plan._get_last_result())
    assert result["termination_reason"] == "converged"
    assert result["converged"]
    assert 0 < plan.get_iterations() <= max_iterations
    assert plan.get_residual_norm() <= tolerance
    np.testing.assert_allclose(
        solution.to_numpy(), exact, rtol=2e-3, atol=2e-3
    )
    first_iterations = plan.get_iterations()
    first = plan._debug_runtime_stats()
    assert first["operations"]["fixed_iteration_only"] is False
    assert first["operations"]["bounded_masked_execution"]
    assert first["operations"]["operator_apply_calls"] == 17
    assert first["operations"]["total_iterations"] == first_iterations
    assert first["operations"]["device_scalar_operations"] == 80
    assert first["operations"]["host_synchronizations"] == 1
    assert first["operations"]["host_scalar_readbacks"] == 4
    assert first["resources"]["persistent_scalar_count"] == 11
    assert first["resources"]["persistent_scalar_reserved_bytes"] == 44

    solution.from_numpy(np.zeros(n, dtype=np.float32))
    rhs.from_numpy(np.zeros(n, dtype=np.float32))
    plan.solve(prog, solution.arr, rhs.arr)

    assert plan.is_success()
    assert plan.get_status() == 2
    assert plan.get_iterations() == 0
    assert plan.get_residual_norm() == 0.0
    np.testing.assert_array_equal(
        solution.to_numpy(), np.zeros(n, dtype=np.float32)
    )
    second = plan._debug_runtime_stats()
    assert second["operations"]["solve_calls"] == 2
    assert second["operations"]["workspace_builds"] == 1
    assert second["operations"]["workspace_reuses"] == 1
    assert second["operations"]["total_iterations"] == first_iterations

    exhausted = ti._lib.core._make_vulkan_cg_convergence_plan(
        prog, matrix, 1, 1e-12
    )
    solution.from_numpy(np.zeros(n, dtype=np.float32))
    rhs.from_numpy(rhs_host)
    exhausted.solve(prog, solution.arr, rhs.arr)
    assert not exhausted.is_success()
    assert exhausted.get_status() == 0
    exhausted_result = dict(exhausted._get_last_result())
    assert exhausted_result["termination_reason"] == "max_iterations"
    assert exhausted_result["reached_max_iterations"]
    assert not exhausted_result["converged"]
    assert exhausted.get_iterations() == 1
    assert exhausted.get_residual_norm() > 1e-12
