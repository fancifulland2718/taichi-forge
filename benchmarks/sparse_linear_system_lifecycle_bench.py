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


def _unsupported_vulkan_report(n):
    return {
        "schema": SCHEMA,
        "schema_version": 1,
        "arch": "vulkan",
        "correct": True,
        "supported": False,
        "config": {"rows": n, "operator": "dirichlet_poisson_1d"},
        "capability": {
            "reason": "SparseMatrix and SparseCG support CPU/CUDA only",
            "missing_primitives": [
                "bounded_triplet_or_pattern_build",
                "csr_or_bsr_spmv",
                "axpy",
                "dot",
                "norm",
                "persistent_cg_plan",
            ],
        },
        "phase_order": [],
        "phases": {},
        "performance_valid": False,
    }


def run_initialized(ti, *, n=32, max_iter=None, atol=1e-5):
    """Run the repeated Poisson solve in an initialized Program."""
    if n < 2:
        raise ValueError("n must be at least 2")
    arch = _arch_name(ti)
    if arch == "vulkan":
        return _unsupported_vulkan_report(n)
    if arch not in ("cpu", "cuda"):
        raise RuntimeError(f"unsupported sparse linear-system arch: {arch}")
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
