"""Measure queued SolvePlan Graph submissions with one or two workspaces.

The second-submit host latency is the primary comparison. Workspace lanes do
not create backend streams, so pair completion time is reported as a guardrail
rather than an expected throughput speedup. Run GPU measurements only while
the target device is otherwise idle.
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


def _summarize(values):
    return {
        "median": statistics.median(values),
        "p10": _percentile(values, 0.1),
        "p90": _percentile(values, 0.9),
    }


def _operator(size):
    diagonal_host = np.linspace(1.0, 4.0, size, dtype=np.float32)
    topology = ti.ndarray(ti.i32, shape=size)
    diagonal = ti.ndarray(ti.f32, shape=size)
    topology.from_numpy(np.arange(size, dtype=np.int32))
    diagonal.from_numpy(diagonal_host)

    @ti.kernel
    def apply_diagonal(
        active_size: ti.i32,
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        values: ti.types.ndarray(dtype=ti.f32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(active_size):
            y[index] = values[index] * x[topology_data[index]]

    return (
        ti.linalg.LinearOperator.from_kernel(
            apply_diagonal,
            size,
            topology,
            numeric=diagonal,
            traits=ti.linalg.OperatorTraits.spd(),
        ),
        diagonal_host,
    )


def _build_graph(action, lanes):
    builder = ti.graph.GraphBuilder()
    builder.append_native(action)
    return builder.compile(
        workspace_lanes=lanes,
        workspace_saturation="wait" if lanes == 1 else "raise",
    )


def _submit_pair(graph, action, rhs, outputs):
    packets = (action.allocate_terminal(), action.allocate_terminal())
    started = time.perf_counter_ns()
    first = graph.submit({"rhs": rhs, "output": outputs[0], **packets[0].arguments})
    first_submitted = time.perf_counter_ns()
    second = graph.submit({"rhs": rhs, "output": outputs[1], **packets[1].arguments})
    second_submitted = time.perf_counter_ns()
    first.wait()
    second.wait()
    completed = time.perf_counter_ns()
    terminals = (packets[0].snapshot(), packets[1].snapshot())
    if not all(result.converged for result in terminals):
        raise RuntimeError("workspace-lane benchmark solve did not converge")
    return {
        "first_submit_us": (first_submitted - started) / 1.0e3,
        "second_submit_us": (second_submitted - first_submitted) / 1.0e3,
        "pair_completion_us": (completed - started) / 1.0e3,
        "workspace_lanes": (first.workspace_lane, second.workspace_lane),
        "iterations": tuple(result.iterations for result in terminals),
    }


def _measure(graph, action, rhs, outputs, warmups, repeats):
    for _ in range(warmups):
        _submit_pair(graph, action, rhs, outputs)
    samples = [_submit_pair(graph, action, rhs, outputs) for _ in range(repeats)]
    memory = graph.execution_stats().memory
    return {
        "first_submit_us": _summarize(
            [sample["first_submit_us"] for sample in samples]
        ),
        "second_submit_us": _summarize(
            [sample["second_submit_us"] for sample in samples]
        ),
        "pair_completion_us": _summarize(
            [sample["pair_completion_us"] for sample in samples]
        ),
        "selected_workspace_lanes": samples[-1]["workspace_lanes"],
        "terminal_iterations": samples[-1]["iterations"],
        "memory": {
            "workspace_lane_capacity": memory.workspace_lane_capacity,
            "workspace_lanes_materialized": memory.workspace_lanes_materialized,
            "workspace_lane_waits": memory.workspace_lane_waits,
            "internal_storage_waits": memory.internal_storage_waits,
            "persistent_internal_storage_bytes": (
                memory.persistent_internal_storage_bytes
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=("cuda", "vulkan"), required=True)
    parser.add_argument("--size", type=int, default=262144)
    parser.add_argument("--max-iterations", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--case-order",
        choices=("single-first", "dual-first"),
        default="single-first",
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.size <= 0 or args.max_iterations <= 0:
        parser.error("--size and --max-iterations must be positive")
    if args.warmups < 0 or args.repeats <= 0:
        parser.error("--warmups must be nonnegative and --repeats positive")

    ti.init(arch=_arch(args.arch), offline_cache=False)
    operator, diagonal = _operator(args.size)
    plan = ti.linalg.experimental.SolvePlan(
        operator,
        method="cg",
        max_iterations=args.max_iterations,
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    rhs_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "rhs", ti.f32, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1)
    action = plan.graph_action(rhs_arg, output_arg, name="workspace_lane_cg")
    lane_order = (1, 2) if args.case_order == "single-first" else (2, 1)
    graphs = {str(lanes): _build_graph(action, lanes) for lanes in lane_order}

    rhs_host = np.linspace(-1.0, 2.0, args.size, dtype=np.float32)
    expected = rhs_host / diagonal
    rhs = ti.ndarray(ti.f32, shape=args.size)
    rhs.from_numpy(rhs_host)
    outputs = (
        ti.ndarray(ti.f32, shape=args.size),
        ti.ndarray(ti.f32, shape=args.size),
    )
    cases = {
        lanes: _measure(
            graph,
            action,
            rhs,
            outputs,
            args.warmups,
            args.repeats,
        )
        for lanes, graph in graphs.items()
    }
    max_abs_error = max(
        float(np.max(np.abs(output.to_numpy() - expected))) for output in outputs
    )
    if max_abs_error > 5.0e-3:
        raise RuntimeError(f"solution error exceeded gate: {max_abs_error}")

    single_second = cases["1"]["second_submit_us"]["median"]
    dual_second = cases["2"]["second_submit_us"]["median"]
    single_pair = cases["1"]["pair_completion_us"]["median"]
    dual_pair = cases["2"]["pair_completion_us"]["median"]
    result = {
        "arch": args.arch,
        "size": args.size,
        "max_iterations": args.max_iterations,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "case_order": args.case_order,
        "cases": cases,
        "comparison": {
            "second_submit_speedup": single_second / dual_second,
            "pair_completion_ratio_dual_over_single": dual_pair / single_pair,
            "scope": (
                "host workspace-fence avoidance; no backend-stream overlap claim"
            ),
        },
        "correctness": {"max_abs_error": max_abs_error},
    }
    serialized = json.dumps(result, indent=2, sort_keys=True)
    print(serialized)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output_file:
            output_file.write(serialized)
            output_file.write("\n")


if __name__ == "__main__":
    main()
