"""Fresh-process crossover qualification for representative physics workloads.

This qualification-only tool reuses the existing hardware workload runners
while varying one physically meaningful size axis per workload family.  It is
not imported by the runtime package and adds no wheel dependency.
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

import hardware_acceleration_qualification as qualification


FAMILIES = (
    "cuda-fft-poisson-batch",
    "cuda-spmv-krylov-grid",
    "cuda-spmv-krylov-stencil-radius",
    "cuda-cudss-tet-fem-grid",
)


def _positive_ints(value, *, minimum=1):
    try:
        points = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("points must be comma-separated integers") from exc
    if not points or any(point < minimum for point in points):
        raise argparse.ArgumentTypeError(f"points must be integers >= {minimum}")
    if tuple(sorted(set(points))) != points:
        raise argparse.ArgumentTypeError("points must be unique and strictly increasing")
    return points


def _family_points(args):
    return {
        "cuda-fft-poisson-batch": tuple(
            {
                "label": f"length-{args.poisson_length}-batch-{batch}",
                "work_units": args.poisson_length * batch,
                "worker_arguments": (
                    "--poisson-length",
                    str(args.poisson_length),
                    "--poisson-batch",
                    str(batch),
                ),
                "parameters": {"length": args.poisson_length, "batch": batch},
            }
            for batch in args.poisson_batches
        ),
        "cuda-spmv-krylov-grid": tuple(
            {
                "label": f"grid-{grid}-radius-{args.krylov_stencil_radius}",
                "work_units": grid * grid,
                "worker_arguments": (
                    "--krylov-grid",
                    str(grid),
                    "--krylov-iterations",
                    str(args.krylov_iterations),
                    "--krylov-stencil-radius",
                    str(args.krylov_stencil_radius),
                    "--krylov-baseline",
                    "taichi",
                ),
                "parameters": {
                    "grid": grid,
                    "iterations": args.krylov_iterations,
                    "stencil_radius": args.krylov_stencil_radius,
                },
            }
            for grid in args.krylov_grids
        ),
        "cuda-spmv-krylov-stencil-radius": tuple(
            {
                "label": f"grid-{args.krylov_radius_grid}-radius-{radius}",
                "work_units": args.krylov_radius_grid
                * args.krylov_radius_grid
                * (2 * radius + 1) ** 2,
                "worker_arguments": (
                    "--krylov-grid",
                    str(args.krylov_radius_grid),
                    "--krylov-iterations",
                    str(args.krylov_iterations),
                    "--krylov-stencil-radius",
                    str(radius),
                    "--krylov-baseline",
                    "taichi",
                ),
                "parameters": {
                    "grid": args.krylov_radius_grid,
                    "iterations": args.krylov_iterations,
                    "stencil_radius": radius,
                    "maximum_stencil_entries": (2 * radius + 1) ** 2,
                },
            }
            for radius in args.krylov_stencil_radii
        ),
        "cuda-cudss-tet-fem-grid": tuple(
            {
                "label": f"grid-{grid}",
                "work_units": 3 * grid * grid * grid,
                "worker_arguments": ("--fem-grid", str(grid)),
                "parameters": {"grid": grid, "degrees_of_freedom": 3 * grid**3},
            }
            for grid in args.fem_grids
        ),
    }


def _base_case(family):
    return {
        "cuda-fft-poisson-batch": "cuda-fft-poisson",
        "cuda-spmv-krylov-grid": "cuda-spmv-krylov",
        "cuda-spmv-krylov-stencil-radius": "cuda-spmv-krylov",
        "cuda-cudss-tet-fem-grid": "cuda-cudss-tet-fem",
    }[family]


def _point_qualifies(report, tier):
    if report.get("status") != "passed" or not report.get("correctness_and_route_qualified"):
        return False
    if tier == "retain":
        return bool(report.get("retention_eligible"))
    if tier == "auto_select":
        return bool(report.get("auto_admission", {}).get("eligible"))
    if tier == "public_claim":
        return bool(report.get("performance_claim_eligible"))
    raise ValueError(f"unknown crossover qualification tier {tier!r}")


def _tier_crossover_summary(point_reports, tier):
    qualified_indices = [
        index for index, report in enumerate(point_reports) if _point_qualifies(report, tier)
    ]
    first_qualified = point_reports[qualified_indices[0]]["point"] if qualified_indices else None
    reversals = []
    if qualified_indices:
        for report in point_reports[qualified_indices[0] + 1 :]:
            if not _point_qualifies(report, tier):
                reversals.append(report["point"]["label"])
    return {
        "first_qualified_point": first_qualified,
        "qualified_points": tuple(
            report["point"]["label"]
            for report in point_reports
            if _point_qualifies(report, tier)
        ),
        "reversals_after_first_qualified": tuple(reversals),
        "monotonic_after_first_qualified": bool(first_qualified is not None and not reversals),
    }


def _crossover_summary(point_reports):
    measured = [report for report in point_reports if report.get("status") != "skipped"]
    tiers = {
        tier: _tier_crossover_summary(point_reports, tier)
        for tier in ("retain", "auto_select", "public_claim")
    }
    first_retained = tiers["retain"]["first_qualified_point"]
    return {
        "status": (
            "not_measured"
            if not measured
            else (
                "retention_crossover_observed"
                if first_retained is not None
                else "no_retention_qualified_point"
            )
        ),
        "tiers": tiers,
        "correctness_and_route_complete": bool(
            measured and all(report.get("correctness_and_route_qualified", False) for report in measured)
        ),
        "all_points_performance_qualified": bool(
            measured and all(report.get("performance_evidence", {}).get("qualified", False) for report in measured)
        ),
        "point_count": len(point_reports),
        "measured_point_count": len(measured),
    }


def _load_worker_result(path, completed, *, family, point, order):
    if path.exists():
        with open(path, "r", encoding="utf-8") as source:
            return json.load(source)
    return {
        "schema": qualification.SCHEMA,
        "case": _base_case(family),
        "order": order,
        "status": "error",
        "error_type": "WorkerProcessError",
        "error": (
            f"point={point['label']}; exit={completed.returncode}; "
            f"stdout={completed.stdout[-1000:]!r}; stderr={completed.stderr[-1000:]!r}"
        ),
    }


def _run_point(args, *, script, temp_path, family, point):
    workers = []
    schedule = qualification._balanced_worker_schedule(args.workers_per_order)
    for launch_index, (order, worker_index) in enumerate(schedule):
        worker_output = temp_path / f"{family}-{point['label']}-{order}-{worker_index}.json"
        command = [
            sys.executable,
            str(script),
            "--worker",
            "--case",
            _base_case(family),
            "--order",
            order,
            "--worker-output",
            str(worker_output),
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
            *point["worker_arguments"],
        ]
        if args.cudss_library:
            command.extend(("--cudss-library", args.cudss_library))
        counter_before = (
            qualification._windows_performance_counter_snapshot() if args.windows_performance_counters else None
        )
        completed = subprocess.run(command, check=False, capture_output=True, text=True, env=os.environ.copy())
        counter_after = (
            qualification._windows_performance_counter_snapshot() if args.windows_performance_counters else None
        )
        worker = _load_worker_result(
            worker_output,
            completed,
            family=family,
            point=point,
            order=order,
        )
        worker["worker_exit_code"] = completed.returncode
        worker["launch_index"] = launch_index
        worker["worker_index"] = worker_index
        if counter_before is not None and counter_after is not None:
            worker["performance_environment"] = qualification._performance_environment_record(
                counter_before, counter_after
            )
        workers.append(worker)
    report = qualification._aggregate(family, tuple(workers), args.cv_limit, args.drift_limit)
    report["base_case"] = _base_case(family)
    report["point"] = {key: point[key] for key in ("label", "work_units", "parameters")}
    return report


def _parent(args):
    families = tuple(item.strip() for item in args.families.split(",") if item.strip())
    unknown = sorted(set(families).difference(FAMILIES))
    if not families or unknown:
        raise ValueError(f"unknown or empty families: {unknown}")
    worker_script = pathlib.Path(__file__).with_name("hardware_acceleration_qualification.py").resolve()
    points_by_family = _family_points(args)
    reports = []
    with tempfile.TemporaryDirectory(prefix="forge-physics-crossover-") as temp:
        temp_path = pathlib.Path(temp)
        for family in families:
            point_reports = [
                _run_point(
                    args,
                    script=worker_script,
                    temp_path=temp_path,
                    family=family,
                    point=point,
                )
                for point in points_by_family[family]
            ]
            reports.append({"family": family, "points": point_reports})

    source_provenance = qualification._source_checkout_provenance(worker_script.parents[2])
    build_artifacts = qualification._local_build_artifact_provenance()
    all_point_reports = [point for report in reports for point in report["points"]]
    build_provenance = qualification._apply_build_provenance_gate(
        all_point_reports, source_provenance["source_revision"]
    )
    for report in reports:
        report["crossover"] = _crossover_summary(report["points"])
    output = {
        "schema": qualification.SCHEMA,
        "qualification": "physics_hardware_crossover",
        "generated_at_ns": time.time_ns(),
        **source_provenance,
        **build_artifacts,
        "build_provenance": build_provenance,
        "policy": {
            "workers_per_order": args.workers_per_order,
            "warmup": args.warmup,
            "rounds": args.rounds,
            "repetitions": args.repetitions,
            "minimum_block_ms": args.minimum_block_ms,
            "maximum_repetitions": args.maximum_repetitions,
            "cv_limit": args.cv_limit,
            "order_drift_limit": args.drift_limit,
            "windows_performance_counters": args.windows_performance_counters,
            "points": {
                family: [
                    {key: point[key] for key in ("label", "work_units", "parameters")}
                    for point in points_by_family[family]
                ]
                for family in families
            },
        },
        "families": reports,
    }
    with open(args.output, "w", encoding="utf-8") as destination:
        json.dump(output, destination, indent=2, sort_keys=True)
        destination.write("\n")
    print(json.dumps(output, sort_keys=True))
    return 0 if all(point["status"] in ("passed", "skipped") for point in all_point_reports) else 1


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--output", default="hardware-physics-crossover.json")
    parser.add_argument("--poisson-length", type=int, default=4096)
    parser.add_argument("--poisson-batches", default="1,4,16")
    parser.add_argument("--krylov-grids", default="64,128,256")
    parser.add_argument("--krylov-iterations", type=int, default=48)
    parser.add_argument("--krylov-stencil-radius", type=int, default=1)
    parser.add_argument("--krylov-radius-grid", type=int, default=128)
    parser.add_argument("--krylov-stencil-radii", default="1,2,3")
    parser.add_argument("--fem-grids", default="4,6,8")
    parser.add_argument("--cudss-library")
    parser.add_argument("--workers-per-order", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--repetitions", type=int, default=25)
    parser.add_argument("--minimum-block-ms", type=float, default=100.0)
    parser.add_argument("--maximum-repetitions", type=int, default=1048576)
    parser.add_argument("--cv-limit", type=float, default=0.05)
    parser.add_argument("--drift-limit", type=float, default=0.05)
    parser.add_argument("--windows-performance-counters", action="store_true")
    args = parser.parse_args()
    try:
        args.poisson_batches = _positive_ints(args.poisson_batches)
        args.krylov_grids = _positive_ints(args.krylov_grids, minimum=2)
        args.krylov_stencil_radii = _positive_ints(args.krylov_stencil_radii)
        args.fem_grids = _positive_ints(args.fem_grids, minimum=3)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if (
        args.poisson_length < 2
        or args.poisson_length & (args.poisson_length - 1)
        or args.krylov_iterations <= 0
        or args.krylov_stencil_radius <= 0
        or args.krylov_radius_grid < 2
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
    return _parent(_parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
