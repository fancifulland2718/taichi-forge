"""Measure fixed overhead and device work for f32 Graph Krylov solves.

Run GPU measurements only while the target device is otherwise idle. Steady
solve latency is the primary comparison; plan construction and first solve are
reported separately because compiler and cache state make them order-sensitive.
"""

import argparse
import json
import statistics
import time

import numpy as np

import taichi_forge as ti


def _arch(name):
    return {
        "cpu": ti.cpu,
        "cuda": ti.cuda,
        "vulkan": ti.vulkan,
    }[name]


def _percentile(values, fraction):
    values = sorted(values)
    position = round((len(values) - 1) * fraction)
    return values[position]


def _diagonal_operator(size, provider):
    diagonal_host = np.linspace(1.0, 4.0, size, dtype=np.float32)
    topology = ti.ndarray(ti.i32, shape=size)
    numeric = ti.ndarray(ti.f32, shape=size)
    topology.from_numpy(np.arange(size, dtype=np.int32))
    numeric.from_numpy(diagonal_host)

    @ti.kernel
    def apply_diagonal(
        active_size: ti.i32,
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        numeric_data: ti.types.ndarray(dtype=ti.f32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(active_size):
            source = topology_data[index]
            y[index] = numeric_data[index] * x[source]

    operator = ti.linalg.LinearOperator.from_kernel(
        apply_diagonal,
        size,
        topology,
        numeric=numeric,
        traits=ti.linalg.OperatorTraits.spd(),
    )
    if provider == "composition_sum":
        operator = 0.5 * operator + 0.5 * operator
    return operator, diagonal_host


def _new_plan(operator, args, execution_policy):
    options = {
        "method": "cg",
        "max_iterations": args.max_iterations,
        "atol": args.atol,
        "rtol": args.rtol,
    }
    if execution_policy is not None:
        options["execution_policy"] = execution_policy
        options["check_interval"] = args.check_interval
    start = time.perf_counter_ns()
    plan = ti.linalg.experimental.SolvePlan(operator, **options)
    build_ms = (time.perf_counter_ns() - start) / 1.0e6
    return plan, build_ms


def _kernel_time_per_solve(repeats):
    seconds = ti.profiler.get_kernel_profiler_total_time()
    if seconds <= 0.0:
        return None
    return seconds * 1.0e6 / repeats


def _measure_plan(
    name,
    plan,
    build_ms,
    rhs,
    exact,
    diagonal,
    args,
):
    output = ti.ndarray(ti.f32, shape=args.size)

    first_start = time.perf_counter_ns()
    first_result = plan.solve(rhs, out=output)
    first_us = (time.perf_counter_ns() - first_start) / 1.0e3
    if not first_result.converged:
        raise RuntimeError(
            f"{name} failed during first solve: "
            f"{first_result.termination_reason}"
        )

    for _ in range(args.warmups):
        result = plan.solve(rhs, out=output)
        if not result.converged:
            raise RuntimeError(
                f"{name} failed during warmup: {result.termination_reason}"
            )

    if args.kernel_profiler:
        ti.profiler.clear_kernel_profiler_info()
    wall_us = []
    result = first_result
    for _ in range(args.repeats):
        start = time.perf_counter_ns()
        result = plan.solve(rhs, out=output)
        wall_us.append((time.perf_counter_ns() - start) / 1.0e3)
        if not result.converged:
            raise RuntimeError(
                f"{name} failed during measurement: "
                f"{result.termination_reason}"
            )

    kernel_us = None
    if args.kernel_profiler:
        kernel_us = _kernel_time_per_solve(args.repeats)

    solution = output.to_numpy()
    residual = diagonal * solution - rhs.to_numpy()
    max_error = float(np.max(np.abs(solution - exact)))
    residual_norm = float(np.linalg.norm(residual.astype(np.float64)))
    if max_error > 5.0e-3:
        raise RuntimeError(
            f"{name} exceeded the f32 correctness gate: {max_error}"
        )

    statistics_snapshot = plan.statistics()
    identity = statistics_snapshot["identity"]
    operations = statistics_snapshot["operations"]
    median_us = statistics.median(wall_us)
    return {
        "name": name,
        "requested_execution_policy": identity.get(
            "requested_solver_execution_policy"
        ),
        "selected_execution_policy": identity.get(
            "solver_execution_policy"
        ),
        "control_path": identity.get(
            "solver_control_path",
            identity.get("bounded_control_path"),
        ),
        "plan_build_ms": build_ms,
        "first_solve_completion_us": first_us,
        "steady_solve_completion_us": {
            "median": median_us,
            "p10": _percentile(wall_us, 0.1),
            "p90": _percentile(wall_us, 0.9),
        },
        "device_kernel_us_per_solve": kernel_us,
        "kernel_profiler_visible": kernel_us is not None,
        "estimated_non_kernel_or_opaque_us_per_solve": (
            None if kernel_us is None else median_us - kernel_us
        ),
        "terminal": {
            "iterations": result.iterations,
            "residual_norm": result.residual_norm,
            "true_residual_norm": residual_norm,
            "effective_tolerance": result.effective_tolerance,
            "max_abs_error": max_error,
        },
        "statistics_after_measurement": {
            key: operations.get(key)
            for key in (
                "last_logical_iterations",
                "last_executed_iterations",
                "host_scalar_readbacks",
                "host_readback_batches",
                "host_synchronizations",
                "convergence_observations",
                "solver_chunk_direct_submissions",
                "wasted_iterations",
            )
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch", choices=("cpu", "cuda", "vulkan"), required=True
    )
    parser.add_argument(
        "--provider",
        choices=("compiled_kernel", "composition_sum"),
        default="compiled_kernel",
    )
    parser.add_argument("--size", type=int, default=262144)
    parser.add_argument("--max-iterations", type=int, default=32)
    parser.add_argument("--atol", type=float, default=1.0e-4)
    parser.add_argument("--rtol", type=float, default=1.0e-5)
    parser.add_argument("--check-interval", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--kernel-profiler", action="store_true")
    parser.add_argument("--output")
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=(
            "automatic",
            "host_check_every_k",
            "device_convergent",
        ),
    )
    args = parser.parse_args()
    if args.size <= 0 or args.repeats <= 0 or args.warmups < 0:
        parser.error("size/repeats must be positive and warmups non-negative")

    ti.init(
        arch=_arch(args.arch),
        offline_cache=False,
        kernel_profiler=args.kernel_profiler,
    )
    operator, diagonal = _diagonal_operator(args.size, args.provider)
    exact = np.sin(
        np.linspace(0.0, 8.0, args.size, dtype=np.float32)
    )
    rhs_host = diagonal * exact
    rhs = ti.ndarray(ti.f32, shape=args.size)
    rhs.from_numpy(rhs_host)

    results = []
    automatic, automatic_build_ms = _new_plan(operator, args, None)
    capabilities = automatic.execution_capabilities()
    policies = args.policies
    if policies is None:
        policies = (
            ("automatic",)
            if args.arch == "cpu"
            else (
                ("automatic", "host_check_every_k", "device_convergent")
                if capabilities["device_convergent"]["supported"]
                else ("automatic", "host_check_every_k")
            )
        )
    if "host_check_every_k" in policies and args.arch == "cpu":
        parser.error("host_check_every_k is unavailable on CPU")

    if "automatic" in policies:
        results.append(
            _measure_plan(
                "automatic",
                automatic,
                automatic_build_ms,
                rhs,
                exact,
                diagonal,
                args,
            )
        )

    if "host_check_every_k" in policies:
        host_checked, host_checked_build_ms = _new_plan(
            operator, args, "host_check_every_k"
        )
        results.append(
            _measure_plan(
                "host_check_every_k",
                host_checked,
                host_checked_build_ms,
                rhs,
                exact,
                diagonal,
                args,
            )
        )

    device_capability = capabilities["device_convergent"]
    if "device_convergent" in policies:
        if not device_capability["supported"]:
            parser.error(
                "device_convergent is unavailable: "
                f"{device_capability['unsupported_reason']}"
            )
        device, device_build_ms = _new_plan(
            operator, args, "device_convergent"
        )
        results.append(
            _measure_plan(
                "device_convergent",
                device,
                device_build_ms,
                rhs,
                exact,
                diagonal,
                args,
            )
        )

    report = {
        "schema": "taichi_forge.linalg.graph_krylov_benchmark.v1",
        "arch": args.arch,
        "provider": args.provider,
        "size": args.size,
        "max_iterations": args.max_iterations,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "policies": list(policies),
        "contract": {
            "dtype": "f32",
            "provider": args.provider,
            "method": "cg",
            "timed_boundary": "synchronous_SolvePlan.solve_completion",
            "output_zero_fill_is_included": True,
            "terminal_scalar_observation_is_included": True,
            "solution_vector_readback_is_excluded": True,
            "operation_counters_include_first_warmup_and_repeats": True,
            "plan_build_and_first_solve_are_order_sensitive": True,
            "zero_or_opaque_kernel_profiler_time_is_null": True,
        },
        "device_convergent_capability": device_capability,
        "results": results,
    }
    ti.reset()
    encoded = json.dumps(report, indent=2)
    print(encoded)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(encoded + "\n")


if __name__ == "__main__":
    main()
