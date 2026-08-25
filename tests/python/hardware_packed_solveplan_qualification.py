"""Fresh-process crossover qualification for packed Field SolvePlan boundaries."""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

import numpy as np

import hardware_acceleration_qualification as qualification
import taichi_forge as ti


CASES = (
    "cuda-packed-vector-solve",
    "cuda-packed-matrix-solve",
    "vulkan-packed-vector-solve",
    "vulkan-packed-matrix-solve",
)


def _packed_solveplan_case(case, order, args):
    backend = "cuda" if case.startswith("cuda-") else "vulkan"
    field_kind = "vector" if "-vector-" in case else "matrix"
    if backend == "cuda":
        qualification._init_cuda()
    else:
        qualification._init_vulkan()

    nodes = args.nodes
    lanes = 3 if field_kind == "vector" else 9
    size = nodes * lanes
    values = np.linspace(0.25, 1.25, size, dtype=np.float32)
    if field_kind == "vector":
        host_values = values.reshape(nodes, 3)
        rhs = ti.Vector.field(3, ti.f32, shape=nodes)
        direct_output = ti.Vector.field(3, ti.f32, shape=nodes)
        staged_output = ti.Vector.field(3, ti.f32, shape=nodes)
    else:
        host_values = values.reshape(nodes, 3, 3)
        rhs = ti.Matrix.field(3, 3, ti.f32, shape=nodes)
        direct_output = ti.Matrix.field(3, 3, ti.f32, shape=nodes)
        staged_output = ti.Matrix.field(3, 3, ti.f32, shape=nodes)
    rhs.from_numpy(host_values)

    input_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "input", ti.f32, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1)
    topology_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "topology", ti.i32, ndim=1)
    topology = ti.ndarray(ti.i32, shape=size)
    topology.from_numpy(np.arange(size, dtype=np.int32))

    @ti.kernel
    def identity(
        topology_values: ti.types.ndarray(dtype=ti.i32, ndim=1),
        input_values: ti.types.ndarray(dtype=ti.f32, ndim=1),
        output_values: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(size):
            output_values[index] = input_values[topology_values[index]]

    operator_builder = ti.graph.GraphBuilder()
    operator_builder.dispatch(identity, topology_arg, input_arg, output_arg)
    operator = ti.linalg.LinearOperator.from_graph(
        operator_builder.compile(),
        size,
        topology={"topology": topology},
        traits=ti.linalg.OperatorTraits.spd(),
    )
    direct_plan = ti.linalg.experimental.SolvePlan(
        operator,
        method="cg",
        max_iterations=8,
        atol=1e-6,
        execution_policy="device_convergent",
    )
    staged_plan = ti.linalg.experimental.SolvePlan(
        operator,
        method="cg",
        max_iterations=8,
        atol=1e-6,
        execution_policy="device_convergent",
    )
    staged_rhs = ti.ndarray(ti.f32, shape=size)
    staged_solution = ti.ndarray(ti.f32, shape=size)

    if field_kind == "vector":

        @ti.kernel
        def pack(
            source: ti.template(),
            destination: ti.types.ndarray(dtype=ti.f32, ndim=1),
        ):
            for index in range(nodes):
                for lane in ti.static(range(3)):
                    destination[index * 3 + lane] = source[index][lane]

        @ti.kernel
        def unpack(
            source: ti.types.ndarray(dtype=ti.f32, ndim=1),
            destination: ti.template(),
        ):
            for index in range(nodes):
                for lane in ti.static(range(3)):
                    destination[index][lane] = source[index * 3 + lane]

    else:

        @ti.kernel
        def pack(
            source: ti.template(),
            destination: ti.types.ndarray(dtype=ti.f32, ndim=1),
        ):
            for index in range(nodes):
                for row, column in ti.static(ti.ndrange(3, 3)):
                    destination[index * 9 + row * 3 + column] = source[index][
                        row, column
                    ]

        @ti.kernel
        def unpack(
            source: ti.types.ndarray(dtype=ti.f32, ndim=1),
            destination: ti.template(),
        ):
            for index in range(nodes):
                for row, column in ti.static(ti.ndrange(3, 3)):
                    destination[index][row, column] = source[
                        index * 9 + row * 3 + column
                    ]

    def hardware():
        direct_plan.solve(rhs, out=direct_output)

    def baseline():
        pack(rhs, staged_rhs)
        staged_plan.solve(staged_rhs, out=staged_solution)
        unpack(staged_solution, staged_output)

    timing = qualification._measure_pair(
        hardware,
        baseline,
        order,
        args.warmup,
        args.rounds,
        args.repetitions,
        args.minimum_block_ms,
        args.maximum_repetitions,
    )
    hardware()
    baseline()
    ti.sync()
    direct_error = qualification._error(direct_output.to_numpy(), host_values)
    staged_error = qualification._error(staged_output.to_numpy(), host_values)
    direct_stats = direct_plan.statistics()["vector_io"]
    staged_stats = staged_plan.statistics()["vector_io"]
    capability = direct_plan.execution_capabilities()["direct_dense_field_solve"]
    passed = bool(
        direct_error[1] <= 1e-6
        and staged_error[1] <= 1e-6
        and capability["selected"]
        and direct_stats["pack_calls"] == 0
        and direct_stats["unpack_calls"] == 0
        and direct_stats["direct_graph_solve_full_boundary_submissions"] > 0
        and staged_stats["direct_graph_solve_full_boundary_submissions"] == 0
    )
    result = qualification._provenance(case, order)
    result.update(
        {
            "status": "passed" if passed else "failed",
            "workload": {
                "equation": ("identity SPD CG with packed physics degrees of freedom"),
                "field_kind": field_kind,
                "nodes": nodes,
                "lanes_per_node": lanes,
                "scalar_extent": size,
                "iterations": 8,
                "timed_scope": (
                    "pack/fused-boundary+device-convergent CG+unpack+"
                    "terminal synchronization"
                ),
                "hardware": ("packed Field direct graph-fused SolvePlan boundary"),
                "baseline": ("explicit pack+scalar ndarray SolvePlan+explicit unpack"),
                "host_readback_included": False,
            },
            "timing": timing,
            "correctness": {
                "direct_max_abs": direct_error[0],
                "direct_max_rel": direct_error[1],
                "staged_max_abs": staged_error[0],
                "staged_max_rel": staged_error[1],
            },
            "route": {
                "selection": ("eligible" if capability["selected"] else "ineligible"),
                "direct_dense_field_solve": capability,
                "direct_vector_io": direct_stats,
                "staged_vector_io": staged_stats,
            },
        }
    )
    ti.reset()
    return result


def _worker(args):
    try:
        result = _packed_solveplan_case(args.case, args.order, args)
    except Exception as exc:  # worker failures must remain machine-readable
        result = {
            "schema": qualification.SCHEMA,
            "case": args.case,
            "order": args.order,
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "pid": os.getpid(),
            "timestamp_ns": time.time_ns(),
        }
    with open(args.worker_output, "w", encoding="utf-8") as output:
        json.dump(result, output, sort_keys=True)
    return 0 if result["status"] in ("passed", "skipped") else 1


def _parent(args):
    cases = tuple(item.strip() for item in args.cases.split(",") if item.strip())
    unknown = sorted(set(cases).difference(CASES))
    if not cases or unknown:
        raise ValueError(f"unknown or empty cases: {unknown}")
    script = pathlib.Path(__file__).resolve()
    reports = []
    with tempfile.TemporaryDirectory(
        prefix="forge-packed-solveplan-qualification-"
    ) as temp:
        temp_path = pathlib.Path(temp)
        for case in cases:
            workers = []
            schedule = qualification._balanced_worker_schedule(args.workers_per_order)
            for launch_index, (order, worker_index) in enumerate(schedule):
                worker_output = temp_path / f"{case}-{order}-{worker_index}.json"
                command = [
                    sys.executable,
                    str(script),
                    "--worker",
                    "--case",
                    case,
                    "--order",
                    order,
                    "--worker-output",
                    str(worker_output),
                    "--nodes",
                    str(args.nodes),
                    "--warmup",
                    str(args.warmup),
                    "--rounds",
                    str(args.rounds),
                    "--repetitions",
                    str(args.repetitions),
                    "--minimum-block-ms",
                    str(args.minimum_block_ms),
                    "--maximum-repetitions",
                    str(args.maximum_repetitions),
                ]
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    env=os.environ.copy(),
                )
                if worker_output.exists():
                    with open(worker_output, "r", encoding="utf-8") as source:
                        worker = json.load(source)
                else:
                    worker = {
                        "schema": qualification.SCHEMA,
                        "case": case,
                        "order": order,
                        "status": "error",
                        "error_type": "WorkerProcessError",
                        "error": (
                            f"exit={completed.returncode}; "
                            f"stdout={completed.stdout[-1000:]!r}; "
                            f"stderr={completed.stderr[-1000:]!r}"
                        ),
                    }
                worker["worker_exit_code"] = completed.returncode
                worker["launch_index"] = launch_index
                worker["worker_index"] = worker_index
                workers.append(worker)
            reports.append(
                qualification._aggregate(
                    case,
                    tuple(workers),
                    args.cv_limit,
                    args.drift_limit,
                )
            )
    source_root = script.parents[2]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_root,
        check=False,
        capture_output=True,
        text=True,
    )
    source_status = subprocess.run(
        ["git", "status", "--short"],
        cwd=source_root,
        check=False,
        capture_output=True,
        text=True,
    )
    report = {
        "schema": qualification.SCHEMA,
        "qualification": "packed_solveplan_crossover",
        "generated_at_ns": time.time_ns(),
        "source_revision": (
            revision.stdout.strip() if revision.returncode == 0 else None
        ),
        "source_status": (
            tuple(source_status.stdout.splitlines())
            if source_status.returncode == 0
            else None
        ),
        "policy": {
            "workers_per_order": args.workers_per_order,
            "warmup": args.warmup,
            "rounds": args.rounds,
            "repetitions": args.repetitions,
            "minimum_block_ms": args.minimum_block_ms,
            "maximum_repetitions": args.maximum_repetitions,
            "cv_limit": args.cv_limit,
            "order_drift_limit": args.drift_limit,
            "nodes": args.nodes,
        },
        "cases": reports,
    }
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(report, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if all(case["status"] == "passed" for case in reports) else 1


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--order", choices=("ab", "ba"))
    parser.add_argument("--worker-output")
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--output", default="packed-solveplan-qualification.json")
    parser.add_argument("--nodes", type=int, default=4096)
    parser.add_argument("--workers-per-order", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=25)
    parser.add_argument("--minimum-block-ms", type=float, default=100.0)
    parser.add_argument("--maximum-repetitions", type=int, default=1048576)
    parser.add_argument("--cv-limit", type=float, default=0.05)
    parser.add_argument("--drift-limit", type=float, default=0.05)
    args = parser.parse_args()
    if args.worker and (not args.case or not args.order or not args.worker_output):
        parser.error("worker mode requires --case, --order, and --worker-output")
    if (
        args.nodes <= 0
        or args.workers_per_order <= 0
        or args.warmup < 0
        or args.rounds < 5
        or args.repetitions <= 0
        or args.minimum_block_ms <= 0.0
        or args.maximum_repetitions < args.repetitions
        or not 0.0 < args.cv_limit < 1.0
        or not 0.0 < args.drift_limit < 1.0
    ):
        parser.error("invalid qualification bounds")
    return args


def main():
    args = _parse_args()
    return _worker(args) if args.worker else _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
