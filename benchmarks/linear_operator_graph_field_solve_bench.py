"""Compare Graph Krylov ndarray, staged Field, and direct Field boundaries.

Run GPU measurements only while the target device is otherwise idle.  The
three modes share one SolvePlan, operator, convergence policy, right-hand side,
and solution contract. Measurement order rotates by batch, and the first solve
after each binding-mode switch is discarded to reduce replay and drift bias.
"""

import argparse
import json
import statistics
import time

import numpy as np

import taichi_forge as ti


def _arch(name):
    return {"cuda": ti.cuda, "vulkan": ti.vulkan}[name]


def _percentile(values, fraction):
    values = sorted(values)
    return values[round((len(values) - 1) * fraction)]


def _diagonal_operator(size, provider):
    diagonal_host = np.linspace(1.0, 4.0, size, dtype=np.float32)
    topology = ti.ndarray(ti.i32, shape=size)
    diagonal = ti.ndarray(ti.f32, shape=size)
    topology.from_numpy(np.arange(size, dtype=np.int32))
    diagonal.from_numpy(diagonal_host)

    @ti.kernel
    def apply_diagonal(
        active_size: ti.i32,
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        diagonal_data: ti.types.ndarray(dtype=ti.f32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(active_size):
            source = topology_data[index]
            y[index] = diagonal_data[index] * x[source]

    operator = ti.linalg.LinearOperator.from_kernel(
        apply_diagonal,
        size,
        topology,
        numeric=diagonal,
        traits=ti.linalg.OperatorTraits.spd(),
    )
    if provider == "composition_sum":
        operator = 0.5 * operator + 0.5 * operator
    return operator, diagonal_host


def _new_plan(operator, args):
    return ti.linalg.experimental.SolvePlan(
        operator,
        method="cg",
        max_iterations=args.max_iterations,
        atol=args.atol,
        rtol=args.rtol,
        execution_policy="device_convergent",
    )


def _solve_mode(plan, mode, rhs, output):
    # Benchmark-only A/B control; this is deliberately not public API.
    plan._graph_krylov_direct_field_enabled = mode != "forced_staged_field"
    return plan.solve(rhs, out=output)


def _new_storage(mode, size, rhs_host):
    if mode == "ndarray":
        rhs = ti.ndarray(ti.f32, shape=size)
        output = ti.ndarray(ti.f32, shape=size)
    else:
        rhs = ti.field(ti.f32, shape=size)
        output = ti.field(ti.f32, shape=size)
    rhs.from_numpy(rhs_host)
    return rhs, output


def _summary(mode, samples, result, telemetry, rhs, output, diagonal, exact):
    solution_host = output.to_numpy()
    rhs_host = rhs.to_numpy()
    residual = diagonal * solution_host - rhs_host
    max_error = float(np.max(np.abs(solution_host - exact)))
    residual_norm = float(np.linalg.norm(residual.astype(np.float64)))
    if max_error > 5.0e-3:
        raise RuntimeError(f"{mode} exceeded the f32 correctness gate: {max_error}")

    stats = telemetry
    vector_io = stats["vector_io"]
    identity = stats["identity"]
    resources = stats["resources"]
    return {
        "mode": mode,
        "steady_solve_completion_us": {
            "median": statistics.median(samples),
            "p10": _percentile(samples, 0.1),
            "p90": _percentile(samples, 0.9),
        },
        "terminal": {
            "iterations": result.iterations,
            "residual_norm": result.residual_norm,
            "true_residual_norm": residual_norm,
            "max_abs_error": max_error,
        },
        "execution": {
            "policy": identity.get("solver_execution_policy"),
            "control_path": identity.get("solver_control_path"),
            "direct_dense_field_selected": stats["execution_capabilities"][
                "direct_dense_field_solve"
            ]["selected"],
        },
        "vector_io": {
            key: vector_io[key]
            for key in (
                "staging_buffer_builds",
                "staging_reserved_bytes",
                "pack_calls",
                "unpack_calls",
                "packed_logical_bytes",
                "unpacked_logical_bytes",
                "transfer_graph_submissions",
                "transfer_native_submissions",
                "completion_syncs",
                "direct_graph_solve_submissions",
                "direct_graph_solve_full_boundary_submissions",
                "direct_graph_solve_field_bindings",
                "direct_dense_field_submissions",
            )
        },
        "memory": {
            "solver_persistent_workspace_bytes": resources[
                "persistent_workspace_payload_bytes"
            ],
            "direct_solution_workspace_bytes": resources[
                "direct_solution_workspace_bytes"
            ],
            "boundary_staging_reserved_bytes": vector_io["staging_reserved_bytes"],
            "solver_plus_boundary_reserved_bytes": resources[
                "persistent_workspace_payload_bytes"
            ]
            + vector_io["staging_reserved_bytes"],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=("cuda", "vulkan"), required=True)
    parser.add_argument(
        "--provider",
        choices=("compiled_kernel", "composition_sum"),
        default="composition_sum",
    )
    parser.add_argument("--size", type=int, default=262144)
    parser.add_argument("--max-iterations", type=int, default=32)
    parser.add_argument("--atol", type=float, default=1.0e-4)
    parser.add_argument("--rtol", type=float, default=1.0e-5)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.size <= 0 or args.repeats <= 0 or args.warmups < 0 or args.batch_size <= 0:
        parser.error(
            "size/repeats/batch-size must be positive and warmups non-negative"
        )

    ti.init(arch=_arch(args.arch), offline_cache=False)
    operator, diagonal = _diagonal_operator(args.size, args.provider)
    exact = np.sin(np.linspace(0.0, 8.0, args.size, dtype=np.float32))
    rhs_host = diagonal * exact

    modes = ("ndarray", "forced_staged_field", "direct_field")
    performance_plan = _new_plan(operator, args)
    cases = {}
    for mode in modes:
        rhs, output = _new_storage(mode, args.size, rhs_host)
        first = _solve_mode(performance_plan, mode, rhs, output)
        if not first.converged:
            raise RuntimeError(
                f"{mode} failed its first solve: {first.termination_reason}"
            )
        cases[mode] = {
            "rhs": rhs,
            "output": output,
            "result": first,
            "samples": [],
        }

    for mode in modes:
        case = cases[mode]
        for _ in range(args.warmups):
            case["result"] = _solve_mode(
                performance_plan,
                mode,
                case["rhs"],
                case["output"],
            )

    measured = {mode: 0 for mode in modes}
    batch_index = 0
    while any(count < args.repeats for count in measured.values()):
        offset = batch_index % len(modes)
        order = modes[offset:] + modes[:offset]
        for mode in order:
            case = cases[mode]
            remaining = args.repeats - measured[mode]
            if remaining <= 0:
                continue
            # Do not time the first solve after a runtime-binding mode switch.
            # It may populate or select a different replay instance.
            case["result"] = _solve_mode(
                performance_plan,
                mode,
                case["rhs"],
                case["output"],
            )
            for _ in range(min(args.batch_size, remaining)):
                start = time.perf_counter_ns()
                result = _solve_mode(
                    performance_plan,
                    mode,
                    case["rhs"],
                    case["output"],
                )
                elapsed_us = (time.perf_counter_ns() - start) / 1.0e3
                if not result.converged:
                    raise RuntimeError(
                        f"{mode} failed during measurement: "
                        f"{result.termination_reason}"
                    )
                case["result"] = result
                case["samples"].append(elapsed_us)
                measured[mode] += 1
        batch_index += 1

    telemetry = {}
    for mode in modes:
        telemetry_plan = _new_plan(operator, args)
        case = cases[mode]
        telemetry_result = _solve_mode(
            telemetry_plan,
            mode,
            case["rhs"],
            case["output"],
        )
        if not telemetry_result.converged:
            raise RuntimeError(
                f"{mode} failed its telemetry solve: "
                f"{telemetry_result.termination_reason}"
            )
        telemetry[mode] = telemetry_plan.statistics()

    results = [
        _summary(
            mode,
            cases[mode]["samples"],
            cases[mode]["result"],
            telemetry[mode],
            cases[mode]["rhs"],
            cases[mode]["output"],
            diagonal,
            exact,
        )
        for mode in modes
    ]
    medians = {
        item["mode"]: item["steady_solve_completion_us"]["median"] for item in results
    }
    direct = medians["direct_field"]
    comparisons = {
        "direct_vs_forced_staged_speedup": medians["forced_staged_field"] / direct,
        "direct_vs_forced_staged_latency_reduction_percent": 100.0
        * (1.0 - direct / medians["forced_staged_field"]),
        "direct_vs_ndarray_latency_overhead_percent": 100.0
        * (direct / medians["ndarray"] - 1.0),
    }
    report = {
        "schema": "taichi_forge.linear_operator_graph_field_solve_bench.v1",
        "arch": args.arch,
        "provider": args.provider,
        "size": args.size,
        "max_iterations": args.max_iterations,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "batch_size": args.batch_size,
        "measurement_order": "rotating_batches_discard_mode_transition",
        "results": results,
        "comparisons": comparisons,
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output_file:
            output_file.write(payload)
            output_file.write("\n")


if __name__ == "__main__":
    main()
