"""Provider-neutral qualification evidence for experimental solve plans."""

import copy
import json
import math
import platform
import time
from typing import Mapping

import numpy as np

from taichi_forge._lib import core as _ti_core
from taichi_forge.lang._ndarray import ScalarNdarray
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.linalg._runtime import (
    BatchedSolvePlan,
    LinearOperator,
    PreconditionerPlan,
    SolvePlan,
    _current_program,
    _qualification_check,
    _qualification_error,
    _qualification_non_negative_integer,
    _qualification_tolerance,
    _require_current_scalar_ndarray,
    _readonly_copy,
)
from taichi_forge.types import f32


class SolveQualificationReport:
    """Immutable, JSON-serializable solve-plan qualification evidence."""

    SCHEMA = "taichi_forge.linalg.solve_qualification.v1"

    def __init__(self, record):
        self._record = copy.deepcopy(record)

    @property
    def passed(self):
        return bool(self._record["passed"])

    @property
    def record(self):
        return _readonly_copy(self._record)

    def to_dict(self):
        """Returns a detached mutable copy suitable for persistence."""
        return copy.deepcopy(self._record)

    def to_json(self, *, indent=2):
        """Serializes the qualification record without writing a file."""
        return json.dumps(self._record, indent=indent, sort_keys=True)


def _json_copy(value):
    if isinstance(value, Mapping):
        return {str(name): _json_copy(item) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_copy(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return copy.deepcopy(value)


def _summary(values):
    if not values:
        return None
    milliseconds = [value / 1e6 for value in values]
    return {
        "minimum": min(milliseconds),
        "median": float(np.median(milliseconds)),
        "maximum": max(milliseconds),
    }


def _memory_pool_snapshot():
    try:
        return dict(_ti_core.get_device_memory_pool_stats())
    except Exception as exc:  # pragma: no cover - backend-dependent failure
        return {"unavailable_reason": str(exc)}


def _counter_delta(before, after):
    result = {}
    for name, final in after.items():
        initial = before.get(name)
        if (
            isinstance(final, (int, float))
            and not isinstance(final, bool)
            and isinstance(initial, (int, float))
            and not isinstance(initial, bool)
        ):
            result[name] = final - initial
    return result


def _provider_record(operator):
    if not isinstance(operator, LinearOperator):
        return None
    return {
        "provider": operator.provider,
        "provider_kind": operator._provider_kind,
        "execution_kind": operator.execution_kind,
        "shape": list(operator.shape),
        "dtype": operator._metadata_snapshot["dtype"],
        "resource_stamp": _json_copy(
            operator._metadata_snapshot["resource_stamp"]
        ),
    }


def _preconditioner_record(plan):
    preconditioner = plan.preconditioner
    if preconditioner is None:
        return {"kind": "identity", "provider": None}
    if isinstance(preconditioner, str):
        return {"kind": "built_in", "provider": preconditioner}
    if isinstance(preconditioner, LinearOperator):
        result = _provider_record(preconditioner)
        result["kind"] = "linear_operator"
        return result
    if isinstance(preconditioner, PreconditionerPlan):
        actions = [
            _provider_record(action) for action in preconditioner.actions
        ]
        result = dict(actions[0])
        if any(
            action["provider"] != actions[0]["provider"]
            for action in actions[1:]
        ):
            result["provider"] = "mixed_action_table"
        result.update(
            {
                "kind": "preconditioner_plan",
                "method": preconditioner.method,
                "behavior": preconditioner.behavior,
                "selection": preconditioner.selection,
                "action_count": len(actions),
                "actions": actions,
                "metadata": _json_copy(preconditioner.metadata),
            }
        )
        return result
    return {"kind": type(preconditioner).__name__, "provider": None}


def _normalize_expected_termination(expected, count):
    if isinstance(expected, str):
        values = (expected.casefold(),) * count
    else:
        try:
            values = tuple(str(value).casefold() for value in expected)
        except TypeError as exc:
            raise TaichiRuntimeError(
                "expected_termination must be a string or sequence"
            ) from exc
        if len(values) != count:
            raise TaichiRuntimeError(
                "expected_termination must have one value per system"
            )
    allowed = {"converged", "breakdown", "max_iterations"}
    if any(value not in allowed for value in values):
        raise TaichiRuntimeError(
            "expected_termination values must be converged, breakdown, or "
            "max_iterations"
        )
    return values


def _terminal_snapshot(result, batched):
    if batched:
        return {
            "termination_reasons": list(result.termination_reasons),
            "iterations": list(result.iterations),
            "initial_residual_norms": list(result.initial_residual_norms),
            "residual_norms": list(result.residual_norms),
            "effective_tolerances": list(result.effective_tolerances),
            "all_converged": result.all_converged,
        }
    return {
        "termination_reasons": [result.termination_reason],
        "iterations": [result.iterations],
        "initial_residual_norms": [result.initial_residual_norm],
        "residual_norms": [result.residual_norm],
        "effective_tolerances": [result.effective_tolerance],
        "all_converged": result.converged,
    }


def _true_residual(plan, rhs_host, solution):
    applied = plan.operator.apply(solution).to_numpy()
    residual = rhs_host - applied
    if isinstance(plan, BatchedSolvePlan):
        reshaped = residual.reshape(plan.batch_size, plan.system_size)
        return [float(np.linalg.norm(row)) for row in reshaped]
    return [float(np.linalg.norm(residual))]


def _run_once(plan, rhs, initial_guess, out, pacer, lane, on_saturation):
    submission = plan.statistics().get("submission", {})
    asynchronous = bool(submission.get("qualified", False))
    start_ns = time.perf_counter_ns()
    submit_ns = None
    if asynchronous:
        submit_start_ns = time.perf_counter_ns()
        ticket = plan.submit(
            rhs,
            initial_guess=initial_guess,
            out=out,
            pacer=pacer,
            lane=lane,
            on_saturation=on_saturation,
        )
        submit_ns = time.perf_counter_ns() - submit_start_ns
        result = ticket.result()
    else:
        if pacer is not None or lane is not None or on_saturation != "wait":
            raise TaichiRuntimeError(
                "pacer/lane/on_saturation require a qualified asynchronous "
                "BatchedSolvePlan"
            )
        result = plan.solve(rhs, initial_guess=initial_guess, out=out)
    return result, time.perf_counter_ns() - start_ns, submit_ns


def qualify_solve_plan(
    plan_or_factory,
    rhs,
    *,
    reference=None,
    initial_guess=None,
    out=None,
    use_plan_default_output=False,
    expected_termination="converged",
    warmup=1,
    repetitions=5,
    atol=None,
    rtol=None,
    pacer=None,
    lane=None,
    on_saturation="wait",
    metadata=None,
):
    """Qualifies one public single-system or independent-batch solve plan.

    ``plan_or_factory`` may be an existing plan or a zero-argument factory.
    A factory lets the report measure plan construction separately. Timing is
    synchronous wall time unless a qualified fixed-budget batch exposes its
    public asynchronous submission boundary. No device timestamp is inferred
    from wall time. By default the qualification helper allocates and reuses
    one output array when ``out`` is omitted. Set
    ``use_plan_default_output=True`` to benchmark the plan's native
    ``out=None`` return path instead.
    """
    warmup = _qualification_non_negative_integer(warmup, "warmup")
    repetitions = _qualification_non_negative_integer(
        repetitions, "repetitions"
    )
    if repetitions == 0:
        raise TaichiRuntimeError("repetitions must be positive")
    metadata = {} if metadata is None else metadata
    if not isinstance(metadata, Mapping) or any(
        not isinstance(name, str) for name in metadata
    ):
        raise TaichiRuntimeError(
            "metadata must be a mapping with string keys"
        )
    custom_metadata = _json_copy(metadata)
    try:
        json.dumps(custom_metadata)
    except (TypeError, ValueError) as exc:
        raise TaichiRuntimeError("metadata must be JSON-serializable") from exc

    plan_types = (SolvePlan, BatchedSolvePlan)
    build_ns = None
    build_pool_before = None
    build_pool_after = None
    if isinstance(plan_or_factory, plan_types):
        plan = plan_or_factory
    elif callable(plan_or_factory):
        build_pool_before = _memory_pool_snapshot()
        start_ns = time.perf_counter_ns()
        plan = plan_or_factory()
        build_ns = time.perf_counter_ns() - start_ns
        build_pool_after = _memory_pool_snapshot()
        if not isinstance(plan, plan_types):
            raise TypeError(
                "plan factory must return SolvePlan or BatchedSolvePlan"
            )
    else:
        raise TypeError(
            "plan_or_factory must be SolvePlan, BatchedSolvePlan, or a "
            "zero-argument factory"
        )

    batched = isinstance(plan, BatchedSolvePlan)
    system_count = plan.batch_size if batched else 1
    total_size = plan.total_size if batched else plan.operator.shape[0]
    numpy_dtype = np.float32 if plan.operator.dtype == f32 else np.float64
    atol = _qualification_tolerance(
        atol, 2e-5 if numpy_dtype == np.float32 else 1e-12, "atol"
    )
    rtol = _qualification_tolerance(
        rtol, 2e-5 if numpy_dtype == np.float32 else 1e-12, "rtol"
    )
    expected_termination = _normalize_expected_termination(
        expected_termination, system_count
    )
    if not isinstance(use_plan_default_output, bool):
        raise TaichiRuntimeError("use_plan_default_output must be bool")
    if use_plan_default_output and out is not None:
        raise TaichiRuntimeError(
            "use_plan_default_output=True requires out=None"
        )
    if out is None and not use_plan_default_output:
        out = ScalarNdarray(plan.operator.dtype, (total_size,))

    rhs = _require_current_scalar_ndarray(
        rhs, "solve qualification RHS", total_size, plan.operator.dtype
    )
    rhs_host = rhs.to_numpy().astype(numpy_dtype, copy=False)
    if reference is None:
        expected_solution = None
    elif callable(reference):
        expected_solution = np.asarray(
            reference(rhs_host.copy()), dtype=numpy_dtype
        )
    else:
        expected_solution = np.asarray(reference, dtype=numpy_dtype)
    if expected_solution is not None and expected_solution.shape != (
        total_size,
    ):
        raise TaichiRuntimeError(
            f"reference must have shape ({total_size},), got "
            f"{expected_solution.shape}"
        )

    initial_statistics = _json_copy(plan.statistics())
    solve_pool_before = _memory_pool_snapshot()
    pacer_before = _json_copy(pacer.statistics()) if pacer else None
    _, first_ns, first_submit_ns = _run_once(
        plan, rhs, initial_guess, out, pacer, lane, on_saturation
    )
    first_statistics = _json_copy(plan.statistics())
    for _ in range(warmup):
        _run_once(plan, rhs, initial_guess, out, pacer, lane, on_saturation)
    warm_baseline = _json_copy(plan.statistics())
    warm_pool_before = _memory_pool_snapshot()
    pacer_warm_baseline = _json_copy(pacer.statistics()) if pacer else None

    solve_ns = []
    submit_ns = []
    completion_ns = []
    results = []
    iteration_trace = []
    provider_system_iterations = 0
    for repetition in range(repetitions):
        result, elapsed_ns, submission_ns = _run_once(
            plan, rhs, initial_guess, out, pacer, lane, on_saturation
        )
        results.append(result)
        solve_ns.append(elapsed_ns)
        if submission_ns is not None:
            submit_ns.append(submission_ns)
            completion_ns.append(max(elapsed_ns - submission_ns, 0))
        terminal_snapshot = _terminal_snapshot(result, batched)
        per_solve_statistics = _json_copy(plan.statistics())
        per_solve_operations = _json_copy(
            per_solve_statistics.get("operations", {})
        )
        if batched:
            iteration_trace.append(
                {
                    "repetition": repetition,
                    "stop_reasons": terminal_snapshot[
                        "termination_reasons"
                    ],
                    "logical_iterations": terminal_snapshot["iterations"],
                    "issued_iterations": per_solve_operations.get(
                        "last_issued_iterations"
                    ),
                    "executed_system_iterations": (
                        per_solve_operations.get(
                            "last_executed_system_iterations"
                        )
                    ),
                    "provider_system_iterations": (
                        per_solve_operations.get(
                            "last_provider_system_iterations"
                        )
                    ),
                    "convergence_observation_boundaries": None,
                    "observation_boundaries_unavailable_reason": (
                        "batched_plan_does_not_export_per_system_boundaries"
                    ),
                }
            )
        else:
            logical = int(result.iterations)
            boundaries = list(
                per_solve_operations.get(
                    "last_convergence_observation_boundaries", []
                )
            )
            reported_executed = per_solve_operations.get(
                "last_executed_iterations"
            )
            backend_family = per_solve_statistics.get("identity", {}).get(
                "backend_family"
            )
            if reported_executed is None or (
                int(reported_executed) < logical
                and backend_family in ("cpu", "x64", "arm64")
            ):
                executed = logical
                executed_source = "logical_fallback_cpu_unreported"
            else:
                executed = int(reported_executed)
                executed_source = "backend_last_executed_iterations"
            iteration_trace.append(
                {
                    "repetition": repetition,
                    "stop_reason": result.termination_reason,
                    "logical_stop_iteration": logical,
                    "executed_through_iteration": executed,
                    "executed_through_source": executed_source,
                    "convergence_observation_boundaries": boundaries,
                }
            )
        if batched:
            provider_system_iterations += per_solve_operations[
                "last_provider_system_iterations"
            ]
    final_statistics = _json_copy(plan.statistics())
    pacer_after = _json_copy(pacer.statistics()) if pacer else None
    solve_pool_after = _memory_pool_snapshot()

    final_result = results[-1]
    terminal = _terminal_snapshot(final_result, batched)
    solution_host = final_result.solution.to_numpy()
    true_residual_norms = _true_residual(
        plan, rhs_host, final_result.solution
    )
    checks = []
    checks.append(
        _qualification_check(
            "finite_solution",
            bool(np.all(np.isfinite(solution_host))),
            {
                "nonfinite_values": int(
                    np.count_nonzero(~np.isfinite(solution_host))
                )
            },
            {"nonfinite_values": 0},
        )
    )
    observed_terminations = [
        _terminal_snapshot(result, batched)["termination_reasons"]
        for result in results
    ]
    checks.append(
        _qualification_check(
            "termination",
            all(
                tuple(observed) == expected_termination
                for observed in observed_terminations
            ),
            {"observed": observed_terminations},
            {"expected": list(expected_termination)},
        )
    )
    if expected_solution is None:
        checks.append(
            {
                "name": "solution_reference",
                "status": "not_requested",
                "metrics": {},
                "tolerance": {"atol": atol, "rtol": rtol},
                "details": "No solution reference was supplied.",
            }
        )
    else:
        absolute, relative = _qualification_error(
            solution_host, expected_solution
        )
        checks.append(
            _qualification_check(
                "solution_reference",
                absolute <= atol or relative <= rtol,
                {
                    "max_absolute_error": absolute,
                    "max_relative_error": relative,
                },
                {"atol": atol, "rtol": rtol},
            )
        )

    residual_passed = True
    residual_limits = []
    for reason, true_norm, effective, rhs_partition in zip(
        terminal["termination_reasons"],
        true_residual_norms,
        terminal["effective_tolerances"],
        np.split(rhs_host, system_count),
    ):
        rounding = 16.0 * np.finfo(numpy_dtype).eps * max(
            float(np.linalg.norm(rhs_partition)), 1.0
        )
        limit = float(max(float(effective), rounding))
        residual_limits.append(limit)
        if reason == "converged" and (
            not math.isfinite(true_norm) or true_norm > limit
        ):
            residual_passed = False
    checks.append(
        _qualification_check(
            "true_residual",
            residual_passed,
            {"norms": true_residual_norms},
            {"converged_limits": residual_limits},
            "Independent b - A(x) check for converged systems.",
        )
    )

    operations_before = warm_baseline.get("operations", {})
    operations_after = final_statistics.get("operations", {})
    operation_delta = _counter_delta(operations_before, operations_after)
    transfer_delta = _counter_delta(
        warm_baseline.get("transfers", {}),
        final_statistics.get("transfers", {}),
    )
    logical_iterations = sum(
        sum(result.iterations) if batched else result.iterations
        for result in results
    )
    if batched:
        executed_iterations = int(
            operation_delta.get("executed_system_iterations", 0)
        )
        provider_iterations = int(provider_system_iterations)
    else:
        executed_iterations = int(
            operation_delta.get("executed_iterations", logical_iterations)
        )
        provider_iterations = executed_iterations
    wasted_iterations = max(provider_iterations - logical_iterations, 0)
    active_efficiency = (
        float(executed_iterations) / provider_iterations
        if provider_iterations
        else 1.0
    )
    telemetry_passed = (
        executed_iterations >= logical_iterations
        and provider_iterations >= executed_iterations
    )
    checks.append(
        _qualification_check(
            "telemetry_invariants",
            telemetry_passed,
            {
                "logical_iterations": logical_iterations,
                "executed_iterations": executed_iterations,
                "provider_iterations": provider_iterations,
                "wasted_provider_iterations": wasted_iterations,
                "active_efficiency": active_efficiency,
            },
            {
                "executed_gte_logical": True,
                "provider_gte_executed": True,
            },
        )
    )
    if batched:
        trace_passed = all(
            entry["issued_iterations"] is not None
            and entry["executed_system_iterations"] is not None
            and entry["provider_system_iterations"] is not None
            and sum(entry["logical_iterations"])
            <= entry["executed_system_iterations"]
            <= entry["provider_system_iterations"]
            for entry in iteration_trace
        )
    else:
        trace_passed = True
        for entry in iteration_trace:
            logical = entry["logical_stop_iteration"]
            executed = entry["executed_through_iteration"]
            boundaries = entry["convergence_observation_boundaries"]
            monotonic = all(
                left < right
                for left, right in zip(boundaries, boundaries[1:])
            )
            endpoints_valid = not boundaries or (
                boundaries[0] == 0 and boundaries[-1] == executed
            )
            trace_passed = trace_passed and (
                0 <= logical <= executed <= plan.max_iterations
                and monotonic
                and endpoints_valid
            )
    checks.append(
        _qualification_check(
            "iteration_stop_trace",
            trace_passed,
            {"per_solve": iteration_trace},
            {
                "logical_lte_executed_lte_budget": True,
                "observation_boundaries_strictly_increasing": True,
                "last_observation_equals_executed": True,
            },
            "Per-solve stopping telemetry is sampled outside the timed solve.",
        )
    )

    program = _current_program()
    backend = _ti_core.arch_name(program.config().arch)
    cuda_runtime = _ti_core.cuda_version() if backend == "cuda" else None
    plan_record = {
        "kind": "batched" if batched else "single",
        "method": plan.method,
        "dtype": plan.operator._metadata_snapshot["dtype"],
        "size": plan.system_size if batched else plan.operator.shape[0],
        "batch_size": plan.batch_size if batched else 1,
        "total_size": total_size,
        "max_iterations": plan.max_iterations,
        "execution_policy": plan.execution_policy,
        "check_interval": plan.check_interval,
        "operator": _provider_record(plan.operator),
        "preconditioner": _preconditioner_record(plan),
        "execution_capabilities": _json_copy(
            plan.execution_capabilities()
        ),
    }
    memory_pool = {
        "scope": "process_global_device_pool",
        "qualification_before": solve_pool_before,
        "qualification_after": solve_pool_after,
        "qualification_delta": _counter_delta(
            solve_pool_before, solve_pool_after
        ),
        "warm_before": warm_pool_before,
        "warm_delta": _counter_delta(warm_pool_before, solve_pool_after),
    }
    if build_pool_before is not None:
        memory_pool.update(
            {
                "build_before": build_pool_before,
                "build_after": build_pool_after,
                "build_delta": _counter_delta(
                    build_pool_before, build_pool_after
                ),
            }
        )
    pacing = None
    if pacer is not None:
        pacing = {
            "qualification_before": pacer_before,
            "after": pacer_after,
            "qualification_delta": _counter_delta(
                pacer_before, pacer_after
            ),
            "warm_before": pacer_warm_baseline,
            "warm_delta": _counter_delta(
                pacer_warm_baseline, pacer_after
            ),
        }
    record = {
        "schema": SolveQualificationReport.SCHEMA,
        "schema_version": 1,
        "passed": not any(check["status"] == "failed" for check in checks),
        "environment": {
            "taichi_version": _ti_core.get_version_string(),
            "taichi_commit": _ti_core.get_commit_hash(),
            "backend": backend,
            "device": None,
            "driver": None,
            "cuda_runtime": cuda_runtime,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device_identity_unavailable_reason": (
                "No safe backend-neutral runtime query is available."
            ),
        },
        "plan": plan_record,
        "configuration": {
            "warmup": warmup,
            "repetitions": repetitions,
            "qualification_atol": atol,
            "qualification_rtol": rtol,
            "expected_termination": list(expected_termination),
        },
        "checks": checks,
        "terminal": terminal,
        "timing": {
            "plan_build_ms": None if build_ns is None else build_ns / 1e6,
            "plan_build_available": build_ns is not None,
            "first_solve_ms": first_ns / 1e6,
            "first_host_submit_ms": (
                None if first_submit_ns is None else first_submit_ns / 1e6
            ),
            "warm_solve_ms": _summary(solve_ns),
            "warm_host_submit_ms": _summary(submit_ns),
            "warm_completion_wait_ms": _summary(completion_ns),
            "host_submit_available": bool(submit_ns),
            "wall_clock_boundary": "synchronous_result_materialization",
            "device_span_ms": None,
            "device_span_unavailable_reason": (
                "The public plan exposes no backend timestamp span; use a "
                "profiler sidecar such as Nsight for device timing."
            ),
        },
        "metrics": {
            "logical_iterations": logical_iterations,
            "executed_iterations": executed_iterations,
            "provider_iterations": provider_iterations,
            "wasted_provider_iterations": wasted_iterations,
            "active_efficiency": active_efficiency,
            "iteration_trace": iteration_trace,
            "true_residual_norms": true_residual_norms,
            "operation_delta": operation_delta,
            "transfer_delta": transfer_delta,
            "memory_pool": memory_pool,
            "pacing": pacing,
        },
        "statistics": {
            "initial": initial_statistics,
            "after_first": first_statistics,
            "warm_baseline": warm_baseline,
            "final": final_statistics,
        },
        "metadata": custom_metadata,
    }
    record = _json_copy(record)
    json.dumps(record)
    return SolveQualificationReport(record)


def summarize_solve_qualifications(reports):
    """Builds a deterministic backend/provider solve support matrix."""
    rows = []
    passed = 0
    failed = 0
    for report in tuple(reports):
        if not isinstance(report, SolveQualificationReport):
            raise TypeError(
                "reports must contain SolveQualificationReport values"
            )
        record = report.to_dict()
        plan = record["plan"]
        checks = {
            check["name"]: check["status"] for check in record["checks"]
        }
        row = {
            "backend": record["environment"]["backend"],
            "kind": plan["kind"],
            "method": plan["method"],
            "dtype": plan["dtype"],
            "size": plan["size"],
            "batch_size": plan["batch_size"],
            "execution_policy": plan["execution_policy"],
            "check_interval": plan["check_interval"],
            "operator_provider": plan["operator"]["provider"],
            "preconditioner_provider": plan["preconditioner"].get(
                "provider"
            ),
            "passed": bool(record["passed"]),
            "checks": checks,
            "timing": copy.deepcopy(record["timing"]),
            "metrics": copy.deepcopy(record["metrics"]),
            "unsupported_checks": sorted(
                name
                for name, status in checks.items()
                if status == "unsupported"
            ),
        }
        rows.append(row)
        if row["passed"]:
            passed += 1
        else:
            failed += 1
    rows.sort(
        key=lambda row: (
            row["backend"],
            row["kind"],
            row["method"],
            row["dtype"],
            row["size"],
            row["batch_size"],
            row["execution_policy"],
            row["check_interval"],
            row["operator_provider"],
            row["preconditioner_provider"] or "",
        )
    )
    return {
        "schema": "taichi_forge.linalg.solve_qualification_matrix.v1",
        "schema_version": 1,
        "summary": {
            "reports": len(rows),
            "passed": passed,
            "failed": failed,
        },
        "rows": rows,
    }


__all__ = [
    "SolveQualificationReport",
    "qualify_solve_plan",
    "summarize_solve_qualifications",
]
