"""Repeated sparse linear-system lifecycle diagnostic.

The workload uses a row-major 1D Dirichlet Poisson operator with known
solutions. One SparseMatrix and one SparseCG plan solve two right-hand sides
with fixed values, then the matrix values are scaled in compressed storage
order and the same plan solves a third right-hand side. The report attributes
operator versions, persistent resources, SpMV plan reuse, solve work, scalar
reductions, and direct transfers. Timings are diagnostic unless explicitly
requested by the CLI.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpu_idle_guard import (  # noqa: E402
    finalize_performance_measurement,
    prepare_performance_measurement,
)


SCHEMA = "taichi_forge.sparse_linear_system_lifecycle.v1"
PHASES = (
    "assemble",
    "first_rhs_solve",
    "second_rhs_solve",
    "numeric_update",
    "updated_values_solve",
)


def _arch_name(ti):
    arch = ti.lang.impl.current_cfg().arch
    if arch == ti.cpu:
        return "cpu"
    if arch == ti.cuda:
        return "cuda"
    if arch == ti.vulkan:
        return "vulkan"
    return str(arch)


def _numeric_delta(before, after):
    result = {}
    for key in sorted(before.keys() & after.keys()):
        left = before[key]
        right = after[key]
        if (
            isinstance(left, (int, float))
            and not isinstance(left, bool)
            and isinstance(right, (int, float))
            and not isinstance(right, bool)
        ):
            result[key] = right - left
        else:
            result[key] = None
    return result


def _stats_delta(before, after):
    return {
        section: _numeric_delta(before[section], after[section])
        for section in ("identity", "operations", "resources", "transfers")
    }


def _poisson_apply(values, scale=1.0):
    result = 2.0 * values.copy()
    result[1:] -= values[:-1]
    result[:-1] -= values[1:]
    return (scale * result).astype(np.float32)


def _poisson_storage_values(n, scale=1.0):
    values = []
    for row in range(n):
        if row > 0:
            values.append(-scale)
        values.append(2.0 * scale)
        if row + 1 < n:
            values.append(-scale)
    return np.asarray(values, dtype=np.float32)


def _as_numpy(value):
    return value.to_numpy() if hasattr(value, "to_numpy") else np.asarray(value)


def _poisson_csr_pattern(n, scale=1.0):
    row_offsets = [0]
    column_indices = []
    values = []
    for row in range(n):
        if row > 0:
            column_indices.append(row - 1)
            values.append(-scale)
        column_indices.append(row)
        values.append(2.0 * scale)
        if row + 1 < n:
            column_indices.append(row + 1)
            values.append(-scale)
        row_offsets.append(len(column_indices))
    return (
        np.asarray(row_offsets, dtype=np.int32),
        np.asarray(column_indices, dtype=np.int32),
        np.asarray(values, dtype=np.float32),
    )


def _fixed_cg_reference(rhs, *, scale, iterations):
    x = np.zeros_like(rhs, dtype=np.float32)
    residual = rhs.astype(np.float32, copy=True)
    direction = residual.copy()
    initial_norm = float(
        np.linalg.norm(residual.astype(np.float64))
    )
    rr = float(np.dot(residual, residual))
    for iteration in range(iterations):
        applied = _poisson_apply(direction, scale=scale)
        denominator = float(np.dot(direction, applied))
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise RuntimeError(
                f"fixed CG reference breakdown: pAp={denominator}"
            )
        alpha = np.float32(rr / denominator)
        x = (x + alpha * direction).astype(np.float32)
        residual = (residual - alpha * applied).astype(np.float32)
        next_rr = float(np.dot(residual, residual))
        if iteration + 1 < iterations:
            beta = np.float32(next_rr / rr)
            direction = (
                residual + beta * direction
            ).astype(np.float32)
        rr = next_rr
    return {
        "x": x,
        "initial_residual_norm": initial_norm,
        "residual_norm": float(
            np.linalg.norm(residual.astype(np.float64))
        ),
    }


def _device_assembly_probe(ti, backend):
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
    x = ti.ndarray(dtype=ti.f32, shape=cols)
    y = ti.ndarray(dtype=ti.f32, shape=rows)
    triplet_rows.from_numpy(triplet_rows_host)
    triplet_columns.from_numpy(triplet_columns_host)
    triplet_values.from_numpy(triplet_values_host)
    x_host = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    x.from_numpy(x_host)
    expected = np.zeros((rows, cols), dtype=np.float32)
    for row, column, value in zip(
        triplet_rows_host, triplet_columns_host, triplet_values_host
    ):
        expected[row, column] += value

    prog = ti.lang.impl.get_runtime().prog
    if backend == "vulkan":
        available = prog._vulkan_sparse_assembly_available()
        factory = ti._lib.core._make_vulkan_sparse_assembly_plan
        unavailable_reason = (
            "Vulkan shaderInt64 sparse assembly unavailable"
        )
    elif backend == "cuda":
        available = prog._cuda_sparse_assembly_available()
        factory = ti._lib.core._make_cuda_sparse_assembly_plan
        unavailable_reason = (
            "CUDA Driver hierarchical sparse assembly unavailable"
        )
    else:
        raise ValueError(f"unsupported device assembly backend: {backend}")
    if not available:
        return {
            "supported": False,
            "correct": False,
            "reason": unavailable_reason,
        }
    plan = factory(prog, rows, cols, capacity)
    matrix = plan.build(
        prog, triplet_rows.arr, triplet_columns.arr, triplet_values.arr
    )
    matrix.spmv(prog, x.arr, y.arr)
    ti.sync()
    first_result = y.to_numpy()
    first_error = float(np.max(np.abs(first_result - expected @ x_host)))
    matrix_stats = matrix._debug_runtime_stats()
    after_success = plan._debug_runtime_stats()

    invalid_rows = triplet_rows_host.copy()
    invalid_rows[0] = rows
    triplet_rows.from_numpy(invalid_rows)
    failure = None
    try:
        plan.build(
            prog, triplet_rows.arr, triplet_columns.arr, triplet_values.arr
        )
    except RuntimeError as exc:
        failure = str(exc)
    matrix.spmv(prog, x.arr, y.arr)
    ti.sync()
    retained_error = float(
        np.max(np.abs(y.to_numpy() - expected @ x_host))
    )
    final = plan._debug_runtime_stats()
    correct = (
        matrix.num_nonzero() == 8
        and first_error <= 1e-6
        and retained_error <= 1e-6
        and failure is not None
        and "index outside" in failure
        and after_success["status"]["last_unique_nnz"] == 8
        and after_success["status"]["last_duplicate_triplets"] == 2
        and after_success["transfers"]["device_payload_readback_bytes"] == 0
        and matrix_stats["transfers"]["device_to_host_bytes"] == 0
        and final["status"]["last_status"] == 1
        and final["operations"]["build_calls"] == 2
        and final["operations"]["successful_builds"] == 1
        and final["operations"]["failed_builds"] == 1
        and final["operations"]["workspace_builds"] == 1
        and final["operations"]["workspace_reuses"] == 1
        and final["operations"]["host_synchronizations"] == 2
        and final["operations"]["host_control_readbacks"] == 2
        and final["transfers"]["device_to_host_bytes"] == 16
        and final["transfers"]["device_to_device_bytes"] == 88
        and final["contract"]["transactional_publish"]
        and final["contract"]["exact_sized_published_csr"]
    )
    if not correct:
        raise RuntimeError(
            f"{backend} bounded sparse assembly probe mismatch: "
            f"first_error={first_error}, retained_error={retained_error}, "
            f"failure={failure}, after_success={after_success}, "
            f"matrix={matrix_stats}, final={final}"
        )
    return {
        "supported": True,
        "correct": True,
        "input_triplets": int(capacity),
        "unique_nnz": 8,
        "duplicate_triplets": 2,
        "contains_empty_row": True,
        "first_spmv_error_linf": first_error,
        "retained_matrix_error_after_failed_build_linf": retained_error,
        "failure_status": final["status"]["last_status"],
        "matrix": matrix_stats,
        "plan_after_success": after_success,
        "plan_final": final,
    }


def _vulkan_operator_only_report(ti, n):
    assembly_probe = _device_assembly_probe(ti, "vulkan")
    row_offsets_host, column_indices_host, values_host = (
        _poisson_csr_pattern(n)
    )
    nnz = column_indices_host.size
    row_offsets = ti.ndarray(dtype=ti.i32, shape=n + 1)
    column_indices = ti.ndarray(dtype=ti.i32, shape=nnz)
    values = ti.ndarray(dtype=ti.f32, shape=nnz)
    updated_values = ti.ndarray(dtype=ti.f32, shape=nnz)
    x = ti.ndarray(dtype=ti.f32, shape=n)
    y = ti.ndarray(dtype=ti.f32, shape=n)
    iterate_x = ti.ndarray(dtype=ti.f32, shape=n)
    residual = ti.ndarray(dtype=ti.f32, shape=n)
    direction = ti.ndarray(dtype=ti.f32, shape=n)
    applied_direction = ti.ndarray(dtype=ti.f32, shape=n)
    residual_dot = ti.ndarray(dtype=ti.f32, shape=1)
    direction_dot = ti.ndarray(dtype=ti.f32, shape=1)
    residual_norm = ti.ndarray(dtype=ti.f32, shape=1)
    row_offsets.from_numpy(row_offsets_host)
    column_indices.from_numpy(column_indices_host)
    values.from_numpy(values_host)
    updated_values.from_numpy(
        _poisson_storage_values(n, scale=1.5)
    )
    prog = ti.lang.impl.get_runtime().prog
    core = prog._create_vulkan_csr_matrix(
        n,
        n,
        row_offsets.arr,
        column_indices.arr,
        values.arr,
    )
    operator = ti.linalg.SparseMatrix(sm=core)

    def apply(vector, scale):
        x.from_numpy(vector)
        operator.matrix.spmv(prog, x.arr, y.arr)
        ti.sync()
        result = y.to_numpy()
        expected = _poisson_apply(vector, scale=scale)
        return {
            "error_linf": float(np.max(np.abs(result - expected))),
            "operator_after": operator._debug_runtime_stats(),
        }

    first = apply(
        np.linspace(0.25, 1.0, n, dtype=np.float32),
        1.0,
    )
    second = apply(
        np.linspace(1.0, -0.5, n, dtype=np.float32),
        1.0,
    )
    before_update = operator._debug_runtime_stats()
    operator._update_values(updated_values)
    after_update = operator._debug_runtime_stats()
    third = apply(
        np.sin(np.linspace(0.0, np.pi, n, dtype=np.float32)).astype(
            np.float32
        ),
        1.5,
    )

    exact = (
        np.sin(np.linspace(0.2, 2.6, n, dtype=np.float32))
        + np.linspace(0.0, 0.15, n, dtype=np.float32)
    ).astype(np.float32)
    rhs = _poisson_apply(exact, scale=1.5)
    zeros = np.zeros(n, dtype=np.float32)
    iterate_x.from_numpy(zeros)
    residual.from_numpy(rhs)
    direction.from_numpy(rhs)
    operator.matrix.spmv(
        prog, direction.arr, applied_direction.arr
    )
    first_dot_workspace = prog._vulkan_sparse_dot(
        residual.arr, residual.arr, residual_dot.arr, n
    )
    second_dot_workspace = prog._vulkan_sparse_dot(
        direction.arr,
        applied_direction.arr,
        direction_dot.arr,
        n,
    )
    ti.sync()
    residual_dot_device = float(residual_dot.to_numpy()[0])
    direction_dot_device = float(direction_dot.to_numpy()[0])
    if direction_dot_device <= 0.0:
        raise RuntimeError(
            "Vulkan minimal iteration produced a non-positive pAp: "
            f"{direction_dot_device}"
        )
    alpha = residual_dot_device / direction_dot_device
    prog._vulkan_sparse_axpy(
        direction.arr, iterate_x.arr, n, alpha
    )
    prog._vulkan_sparse_axpy(
        applied_direction.arr, residual.arr, n, -alpha
    )
    norm_workspace = prog._vulkan_sparse_norm(
        residual.arr, residual_norm.arr, n
    )
    ti.sync()
    iterate_device = iterate_x.to_numpy()
    residual_device = residual.to_numpy()
    norm_device = float(residual_norm.to_numpy()[0])

    applied_host = _poisson_apply(rhs, scale=1.5)
    residual_dot_host = float(
        np.dot(rhs.astype(np.float64), rhs.astype(np.float64))
    )
    direction_dot_host = float(
        np.dot(
            rhs.astype(np.float64),
            applied_host.astype(np.float64),
        )
    )
    alpha_host = residual_dot_host / direction_dot_host
    iterate_expected = (alpha * rhs).astype(np.float32)
    residual_expected = (
        rhs - np.float32(alpha) * applied_host
    ).astype(np.float32)
    norm_host = float(
        np.linalg.norm(residual_expected.astype(np.float64))
    )
    sparse_workspace_bytes = (
        prog._vulkan_sparse_algebra_workspace_bytes()
    )
    reduce_workspace_bytes = prog.vulkan_reduce_workspace_bytes()
    expected_partial_bytes = (
        min(65535, (n + 1023) // 1024) * np.dtype(np.float32).itemsize
    )

    def relative_error(actual, expected):
        return abs(actual - expected) / max(1.0, abs(expected))

    iteration_probe = {
        "residual_dot_relative_error": relative_error(
            residual_dot_device, residual_dot_host
        ),
        "direction_dot_relative_error": relative_error(
            direction_dot_device, direction_dot_host
        ),
        "alpha_relative_error": relative_error(alpha, alpha_host),
        "iterate_error_linf": float(
            np.max(np.abs(iterate_device - iterate_expected))
        ),
        "residual_error_linf": float(
            np.max(np.abs(residual_device - residual_expected))
        ),
        "norm_error_abs": abs(norm_device - norm_host),
        "sparse_workspace_bytes": int(sparse_workspace_bytes),
        "reduce_workspace_bytes": int(reduce_workspace_bytes),
        "expected_partial_bytes": int(expected_partial_bytes),
        "workspace_stable": (
            first_dot_workspace == second_dot_workspace
            and second_dot_workspace == norm_workspace
            and sparse_workspace_bytes == expected_partial_bytes
            and norm_workspace
            == sparse_workspace_bytes + reduce_workspace_bytes
        ),
    }
    iteration_probe["correct"] = (
        iteration_probe["residual_dot_relative_error"] <= 2e-5
        and iteration_probe["direction_dot_relative_error"] <= 2e-5
        and iteration_probe["alpha_relative_error"] <= 3e-5
        and iteration_probe["iterate_error_linf"] <= 3e-5
        and iteration_probe["residual_error_linf"] <= 5e-5
        and iteration_probe["norm_error_abs"] <= 5e-5
        and iteration_probe["workspace_stable"]
    )
    final = operator._debug_runtime_stats()

    fixed_iterations = 4
    plan_x = ti.ndarray(dtype=ti.f32, shape=n)
    plan_rhs = ti.ndarray(dtype=ti.f32, shape=n)
    plan = ti._lib.core._make_vulkan_cg_iteration_plan(
        prog, operator.matrix, fixed_iterations
    )

    def run_plan_case(exact_solution, scale):
        rhs_host = _poisson_apply(exact_solution, scale=scale)
        plan_x.from_numpy(np.zeros(n, dtype=np.float32))
        plan_rhs.from_numpy(rhs_host)
        plan.solve(prog, plan_x.arr, plan_rhs.arr)
        result = plan_x.to_numpy()
        reference = _fixed_cg_reference(
            rhs_host,
            scale=scale,
            iterations=fixed_iterations,
        )
        return {
            "success": bool(plan.is_success()),
            "status": int(plan.get_status()),
            "iterations": int(plan.get_iterations()),
            "initial_residual_norm": float(
                plan.get_initial_residual_norm()
            ),
            "residual_norm": float(plan.get_residual_norm()),
            "initial_norm_error_abs": abs(
                plan.get_initial_residual_norm()
                - reference["initial_residual_norm"]
            ),
            "residual_norm_error_abs": abs(
                plan.get_residual_norm()
                - reference["residual_norm"]
            ),
            "solution_error_linf": float(
                np.max(np.abs(result - reference["x"]))
            ),
            "stats": plan._debug_runtime_stats(),
        }

    first_plan = run_plan_case(
        np.sin(np.linspace(0.15, 2.4, n, dtype=np.float32)),
        1.5,
    )
    second_plan = run_plan_case(
        np.linspace(-0.4, 0.9, n, dtype=np.float32),
        1.5,
    )
    before_plan_value_update = plan._debug_runtime_stats()
    values.from_numpy(_poisson_storage_values(n, scale=0.75))
    operator._update_values(values)
    stale_plan = plan._debug_runtime_stats()
    third_plan = run_plan_case(
        (
            np.cos(np.linspace(0.1, 2.2, n, dtype=np.float32))
            + np.linspace(0.0, 0.1, n, dtype=np.float32)
        ).astype(np.float32),
        0.75,
    )
    plan_final = plan._debug_runtime_stats()
    expected_plan_d2d = 3 * (2 * n * 4 + 8)
    expected_plan_d2h = 3 * 12
    plan_resources_stable = (
        before_plan_value_update["resources"]
        == plan_final["resources"]
    )
    plan_probe = {
        "supported": True,
        "fixed_iteration_only": True,
        "first": first_plan,
        "second": second_plan,
        "stale_after_value_update": stale_plan,
        "third": third_plan,
        "resources_stable_across_numeric_update": plan_resources_stable,
        "plan_final": plan_final,
    }
    plan_probe["correct"] = (
        all(
            case["success"]
            and case["status"] == 0
            and case["iterations"] == fixed_iterations
            and case["initial_norm_error_abs"] <= 3e-5
            and case["residual_norm_error_abs"] <= 3e-4
            and case["solution_error_linf"] <= 3e-4
            and case["residual_norm"]
            < case["initial_residual_norm"]
            for case in (first_plan, second_plan, third_plan)
        )
        and before_plan_value_update["operations"]["solve_calls"] == 2
        and before_plan_value_update["operations"]["workspace_builds"]
        == 1
        and before_plan_value_update["operations"]["workspace_reuses"]
        == 1
        and stale_plan["identity"][
            "operator_numeric_changed_since_last_solve"
        ]
        and stale_plan["identity"]["operator_numeric_version"] == 3
        and stale_plan["identity"]["last_solve_numeric_version"] == 2
        and plan_final["identity"]["operator_pattern_version"] == 1
        and plan_final["identity"]["operator_numeric_version"] == 3
        and plan_final["identity"]["last_solve_numeric_version"] == 3
        and not plan_final["identity"][
            "operator_numeric_changed_since_last_solve"
        ]
        and plan_final["operations"]["solve_calls"] == 3
        and plan_final["operations"]["total_iterations"] == 12
        and plan_final["operations"]["workspace_builds"] == 1
        and plan_final["operations"]["workspace_reuses"] == 2
        and plan_final["operations"]["operator_apply_calls"] == 15
        and plan_final["operations"]["host_scalar_reductions"] == 0
        and plan_final["operations"]["host_scalar_readbacks"] == 9
        and plan_final["operations"]["host_synchronizations"] == 3
        and plan_final["operations"]["device_scalar_operations"] == 42
        and plan_final["operations"]["fixed_iteration_only"]
        and plan_final["resources"]["persistent_vector_count"] == 3
        and plan_final["resources"][
            "persistent_vector_reserved_bytes"
        ]
        == 3 * n * 4
        and plan_final["resources"]["persistent_scalar_count"] == 9
        and plan_final["resources"][
            "persistent_scalar_reserved_bytes"
        ]
        == 36
        and plan_final["transfers"]["device_to_device_bytes"]
        == expected_plan_d2d
        and plan_final["transfers"]["device_to_host_bytes"]
        == expected_plan_d2h
        and plan_final["transfers"]["host_to_device_bytes"] == 4
        and plan_resources_stable
    )

    adaptive_max_iterations = 4
    adaptive_tolerance = 1e-4
    adaptive_plan = ti._lib.core._make_vulkan_cg_convergence_plan(
        prog,
        operator.matrix,
        adaptive_max_iterations,
        adaptive_tolerance,
    )
    eigenmode = np.sin(
        np.pi
        * (np.arange(n, dtype=np.float32) + 1.0)
        / np.float32(n + 1)
    ).astype(np.float32)
    adaptive_rhs_host = _poisson_apply(eigenmode, scale=0.75)
    plan_x.from_numpy(np.zeros(n, dtype=np.float32))
    plan_rhs.from_numpy(adaptive_rhs_host)
    adaptive_plan.solve(prog, plan_x.arr, plan_rhs.arr)
    adaptive_solution = plan_x.to_numpy()
    adaptive_first = {
        "success": bool(adaptive_plan.is_success()),
        "status": int(adaptive_plan.get_status()),
        "iterations": int(adaptive_plan.get_iterations()),
        "residual_norm": float(adaptive_plan.get_residual_norm()),
        "solution_error_linf": float(
            np.max(np.abs(adaptive_solution - eigenmode))
        ),
        "stats": adaptive_plan._debug_runtime_stats(),
    }
    plan_x.from_numpy(np.zeros(n, dtype=np.float32))
    plan_rhs.from_numpy(np.zeros(n, dtype=np.float32))
    adaptive_plan.solve(prog, plan_x.arr, plan_rhs.arr)
    adaptive_zero = {
        "success": bool(adaptive_plan.is_success()),
        "status": int(adaptive_plan.get_status()),
        "iterations": int(adaptive_plan.get_iterations()),
        "residual_norm": float(adaptive_plan.get_residual_norm()),
        "solution_error_linf": float(
            np.max(np.abs(plan_x.to_numpy()))
        ),
        "stats": adaptive_plan._debug_runtime_stats(),
    }
    adaptive_final = adaptive_plan._debug_runtime_stats()
    adaptive_probe = {
        "supported": True,
        "bounded_masked_execution": True,
        "first": adaptive_first,
        "initially_converged": adaptive_zero,
        "plan_final": adaptive_final,
    }
    adaptive_probe["correct"] = (
        adaptive_first["success"]
        and adaptive_first["status"] == 2
        and 0 < adaptive_first["iterations"]
        <= adaptive_max_iterations
        and adaptive_first["residual_norm"] <= adaptive_tolerance
        and adaptive_first["solution_error_linf"] <= 2e-3
        and adaptive_zero["success"]
        and adaptive_zero["status"] == 2
        and adaptive_zero["iterations"] == 0
        and adaptive_zero["residual_norm"] == 0.0
        and adaptive_zero["solution_error_linf"] == 0.0
        and adaptive_final["identity"]["method"]
        == "cg_bounded_masked_probe"
        and adaptive_final["operations"]["solve_calls"] == 2
        and adaptive_final["operations"]["workspace_builds"] == 1
        and adaptive_final["operations"]["workspace_reuses"] == 1
        and adaptive_final["operations"]["operator_apply_calls"] == 10
        and adaptive_final["operations"]["host_synchronizations"] == 2
        and adaptive_final["operations"]["host_scalar_readbacks"] == 8
        and adaptive_final["operations"]["host_scalar_reductions"] == 0
        and adaptive_final["operations"]["device_scalar_operations"]
        == 40
        and adaptive_final["operations"]["bounded_masked_execution"]
        and not adaptive_final["operations"]["fixed_iteration_only"]
        and adaptive_final["resources"]["persistent_scalar_count"] == 11
        and adaptive_final["resources"][
            "persistent_scalar_reserved_bytes"
        ]
        == 44
    )
    expected_pattern_bytes = (n + 1 + nnz) * 4
    expected_value_bytes = nnz * 4
    resources_stable = (
        before_update["resources"] == after_update["resources"]
    )
    correct = (
        assembly_probe["correct"]
        and operator._num_nonzero() == nnz
        and first["error_linf"] <= 2e-5
        and second["error_linf"] <= 2e-5
        and third["error_linf"] <= 3e-5
        and iteration_probe["correct"]
        and plan_probe["correct"]
        and adaptive_probe["correct"]
        and final["identity"]["backend_family"] == "vulkan"
        and final["identity"]["storage_format"] == "csr"
        and final["identity"]["pattern_version"] == 1
        and final["identity"]["numeric_version"] == 2
        and final["operations"]["numeric_updates"] == 1
        and final["operations"]["spmv_calls"] == 4
        and final["operations"]["spmv_plan_builds"] == 1
        and final["operations"]["spmv_plan_reuses"] == 3
        and final["resources"]["pattern_reserved_bytes"]
        == expected_pattern_bytes
        and final["resources"]["values_reserved_bytes"]
        == expected_value_bytes
        and resources_stable
    )
    if not correct:
        raise RuntimeError(
            "Vulkan fixed CSR operator probe mismatch: "
            f"first={first}, second={second}, third={third}, "
            f"iteration={iteration_probe}, "
            f"plan={plan_probe}, "
            f"adaptive={adaptive_probe}, "
            f"before_update={before_update}, "
            f"after_update={after_update}, final={final}"
        )
    return {
        "schema": SCHEMA,
        "schema_version": 1,
        "arch": "vulkan",
        "correct": True,
        "supported": False,
        "config": {
            "rows": n,
            "nnz": int(nnz),
            "operator": "dirichlet_poisson_1d",
        },
        "capability": {
            "reason": (
                "fixed-pattern CSR plus device-scalar fixed and bounded "
                "adaptive probes and bounded device triplet-to-CSR assembly "
                "are available internally; public SparseCG/builder and "
                "conditional dispatch exit remain unsupported"
            ),
            "available_primitives": [
                "device_resident_bounded_triplet_to_csr",
                "transactional_exact_sized_csr_publish",
                "fixed_pattern_csr_spmv",
                "csr_value_only_update",
                "f32_axpy",
                "f32_dot",
                "f32_norm",
                "device_scalar_fixed_iteration_cg_plan",
                "device_convergence_bounded_cg_plan",
            ],
            "missing_primitives": [
                "conditional_dispatch_exit",
                "public_sparse_builder",
                "public_sparse_cg",
                "preconditioner",
            ],
        },
        "phase_order": [],
        "phases": {},
        "operator_probe": {
            "supported": True,
            "correct": True,
            "first_spmv_error_linf": first["error_linf"],
            "second_spmv_error_linf": second["error_linf"],
            "updated_spmv_error_linf": third["error_linf"],
            "resources_stable_across_numeric_update": resources_stable,
            "minimal_iteration": iteration_probe,
            "operator_final": final,
        },
        "assembly_probe": assembly_probe,
        "iteration_plan_probe": plan_probe,
        "adaptive_plan_probe": adaptive_probe,
        "performance_valid": False,
    }


def run_initialized(ti, *, n=32, max_iter=None, atol=1e-5):
    """Run the repeated Poisson solve in an initialized Program."""
    if n < 2:
        raise ValueError("n must be at least 2")
    arch = _arch_name(ti)
    if arch == "vulkan":
        return _vulkan_operator_only_report(ti, n)
    if arch not in ("cpu", "cuda"):
        raise RuntimeError(f"unsupported sparse linear-system arch: {arch}")
    assembly_probe = (
        _device_assembly_probe(ti, "cuda")
        if arch == "cuda"
        else {
            "supported": False,
            "correct": False,
            "reason": "device-resident bounded assembly is GPU-only",
        }
    )
    max_iter = max_iter if max_iter is not None else 4 * n
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if atol <= 0:
        raise ValueError("atol must be positive")

    phases = {}

    def measure(name, callback):
        started = time.perf_counter_ns()
        result = callback()
        ti.sync()
        phases[name] = {
            "duration_ns": time.perf_counter_ns() - started,
            **(result or {}),
        }

    holder = {}

    def assemble():
        builder = ti.linalg.SparseMatrixBuilder(
            n,
            n,
            max_num_triplets=3 * n - 2,
            dtype=ti.f32,
            storage_format="row_major",
        )

        @ti.kernel
        def fill(matrix: ti.types.sparse_matrix_builder()):
            for row in range(n):
                matrix[row, row] += 2.0
                if row > 0:
                    matrix[row, row - 1] += -1.0
                if row + 1 < n:
                    matrix[row, row + 1] += -1.0

        fill(builder)
        matrix = builder.build()
        holder["matrix"] = matrix
        return {"operator_after": matrix._debug_runtime_stats()}

    measure("assemble", assemble)
    matrix = holder["matrix"]
    rhs = ti.ndarray(dtype=ti.f32, shape=n)
    initial = ti.ndarray(dtype=ti.f32, shape=n)
    compressed_values = ti.ndarray(dtype=ti.f32, shape=3 * n - 2)
    cg = ti.linalg.SparseCG(
        matrix, rhs, initial, max_iter=max_iter, atol=atol
    )

    def solve_phase(name, expected, initial_value, scale):
        rhs_value = _poisson_apply(expected, scale=scale)
        rhs.from_numpy(rhs_value)
        initial.from_numpy(initial_value)
        operator_before = matrix._debug_runtime_stats()
        plan_before = cg._debug_runtime_stats()

        def solve():
            solution, converged = cg.solve()
            solution_host = _as_numpy(solution)
            residual = _poisson_apply(solution_host, scale=scale) - rhs_value
            return {
                "converged": bool(converged),
                "iterations": int(cg._last_solve_info.iterations),
                "initial_residual_norm": float(
                    cg._last_solve_info.initial_residual_norm
                ),
                "reported_residual_norm": float(
                    cg._last_solve_info.residual_norm
                ),
                "solution_error_linf": float(
                    np.max(np.abs(solution_host - expected))
                ),
                "reference_residual_norm": float(np.linalg.norm(residual)),
            }

        measure(name, solve)
        operator_after = matrix._debug_runtime_stats()
        plan_after = cg._debug_runtime_stats()
        phases[name]["operator_delta"] = _stats_delta(
            operator_before, operator_after
        )
        phases[name]["plan_delta"] = _stats_delta(plan_before, plan_after)
        phases[name]["operator_after"] = operator_after
        phases[name]["plan_after"] = plan_after

    first_expected = np.linspace(0.25, 1.0, n, dtype=np.float32)
    second_expected = np.linspace(1.0, -0.5, n, dtype=np.float32)
    third_expected = np.sin(
        np.linspace(0.0, np.pi, n, dtype=np.float32)
    ).astype(np.float32)
    solve_phase(
        "first_rhs_solve",
        first_expected,
        np.zeros(n, dtype=np.float32),
        1.0,
    )
    solve_phase(
        "second_rhs_solve",
        second_expected,
        np.full(n, 0.125, dtype=np.float32),
        1.0,
    )

    scale = 1.5
    compressed_values.from_numpy(_poisson_storage_values(n, scale=scale))
    operator_before_update = matrix._debug_runtime_stats()
    plan_before_update = cg._debug_runtime_stats()

    def numeric_update():
        matrix._update_values(compressed_values)

    measure("numeric_update", numeric_update)
    operator_after_update = matrix._debug_runtime_stats()
    plan_after_update = cg._debug_runtime_stats()
    phases["numeric_update"]["operator_delta"] = _stats_delta(
        operator_before_update, operator_after_update
    )
    phases["numeric_update"]["plan_delta"] = _stats_delta(
        plan_before_update, plan_after_update
    )
    phases["numeric_update"]["operator_after"] = operator_after_update
    phases["numeric_update"]["plan_after"] = plan_after_update

    solve_phase(
        "updated_values_solve",
        third_expected,
        np.full(n, -0.125, dtype=np.float32),
        scale,
    )

    final_operator = matrix._debug_runtime_stats()
    final_plan = cg._debug_runtime_stats()
    solution_tolerance = 2e-4
    residual_tolerance = max(4 * atol, 2e-5)
    solves = [
        phases[name]
        for name in (
            "first_rhs_solve",
            "second_rhs_solve",
            "updated_values_solve",
        )
    ]
    resources_stable_across_update = (
        operator_before_update["resources"]
        == operator_after_update["resources"]
    )
    plan_resources_stable_after_first_solve = (
        phases["first_rhs_solve"]["plan_after"]["resources"]
        == final_plan["resources"]
    )
    correct = (
        all(phase["converged"] for phase in solves)
        and all(
            phase["solution_error_linf"] <= solution_tolerance
            for phase in solves
        )
        and all(
            phase["reference_residual_norm"] <= residual_tolerance
            for phase in solves
        )
        and final_operator["identity"]["pattern_version"] == 1
        and final_operator["identity"]["numeric_version"] == 2
        and final_operator["operations"]["numeric_updates"] == 1
        and final_plan["operations"]["solve_calls"] == 3
        and not final_plan["identity"][
            "operator_numeric_changed_since_last_solve"
        ]
        and plan_after_update["identity"][
            "operator_numeric_changed_since_last_solve"
        ]
        and resources_stable_across_update
        and plan_resources_stable_after_first_solve
    )
    if arch == "cuda":
        correct = (
            correct
            and assembly_probe["correct"]
            and final_plan["operations"]["workspace_builds"] == 1
            and final_plan["operations"]["workspace_reuses"] == 2
            and final_plan["operations"]["host_scalar_reductions"] > 0
        )
    else:
        correct = (
            correct
            and final_plan["operations"]["workspace_builds"] == 2
            and final_plan["operations"]["workspace_reuses"] == 1
            and not final_plan["resources"][
                "solver_state_rebuilt_each_solve"
            ]
        )
    if not correct:
        raise RuntimeError(
            "sparse linear-system lifecycle mismatch: "
            f"operator={final_operator}, plan={final_plan}, phases={phases}"
        )

    return {
        "schema": SCHEMA,
        "schema_version": 1,
        "arch": arch,
        "correct": True,
        "supported": True,
        "config": {
            "rows": n,
            "nnz": 3 * n - 2,
            "dtype": "f32",
            "storage_format": "row_major_or_cuda_csr",
            "operator": "dirichlet_poisson_1d",
            "rhs_count": 3,
            "numeric_scale": scale,
            "max_iterations": max_iter,
            "absolute_tolerance": atol,
        },
        "telemetry_contract": {
            "timing": "diagnostic_only",
            "fixed_pattern_lifecycles": [
                "fixed_values_multiple_rhs",
                "value_only_update_then_rhs",
            ],
            "pattern_change": "requires_new_operator_and_solve_plan",
            "value_order": "compressed_storage_order",
            "operator_resources_exclude": "rhs_solution_and_solver_vectors",
            "plan_resources_exclude": "operator_rhs_solution_and_caller_vectors",
        },
        "phase_order": list(PHASES),
        "phases": phases,
        "operator_final": final_operator,
        "plan_final": final_plan,
        "assembly_probe": assembly_probe,
        "checks": {
            "operator_resources_stable_across_numeric_update": (
                resources_stable_across_update
            ),
            "plan_resources_stable_after_first_solve": (
                plan_resources_stable_after_first_solve
            ),
            "numeric_update_marks_plan_stale_until_next_solve": bool(
                plan_after_update["identity"][
                    "operator_numeric_changed_since_last_solve"
                ]
            ),
        },
        "performance_valid": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch", choices=("cpu", "cuda", "vulkan"), default="cpu"
    )
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--max-iter", type=int)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--performance", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    measurement = prepare_performance_measurement(
        args.arch, requested=args.performance
    )

    import taichi_forge as ti

    arch = {"cpu": ti.cpu, "cuda": ti.cuda, "vulkan": ti.vulkan}[args.arch]
    ti.init(arch=arch, enable_fallback=False, offline_cache=False)
    try:
        result = run_initialized(
            ti, n=args.n, max_iter=args.max_iter, atol=args.atol
        )
        result.update(
            finalize_performance_measurement(
                measurement, correct=result["correct"]
            )
        )
    finally:
        ti.reset()
    result["program_reset_completed"] = True
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
