"""Opt-in Graph pipeline telemetry overhead and memory qualification."""

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpu_idle_guard import (  # noqa: E402
    finalize_performance_measurement,
    prepare_performance_measurement,
)


SCHEMA = "taichi_forge.graph_pipeline_telemetry.v1"


def _summary(samples):
    ordered = sorted(int(value) for value in samples)
    mean = statistics.fmean(ordered)
    return {
        "samples": len(ordered),
        "median_ns": int(statistics.median(ordered)),
        "p95_ns": ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)],
        "min_ns": ordered[0],
        "max_ns": ordered[-1],
        "cv_pct": statistics.pstdev(ordered) / mean * 100.0 if mean else 0.0,
    }


def run_initialized(ti, *, capacity=65536, count=4097, samples=11, repeats=20):
    if not 0 < count <= capacity:
        raise ValueError("count must be positive and inside capacity")
    if min(capacity, samples, repeats) <= 0:
        raise ValueError("capacity, samples, and repeats must be positive")

    @ti.kernel
    def consume(
        extent: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in range(capacity):
            if i < ti.device_extent_count(extent):
                output[i] = i + 3

    extent_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "extent", ti.i32, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch_bounded(
        consume,
        extent_arg,
        output_arg,
        extent=extent_arg,
        capacity=capacity,
        block_dim=128,
        label="phase=payload/sweep=0",
    )
    graph = builder.compile()
    extent = ti.DeviceExtent(capacity)
    output = ti.ndarray(ti.i32, shape=capacity)
    extent.set(count)
    args = {"extent": extent, "output": output}

    graph.run(args)
    graph.submit(args).wait()
    graph.submit(args, telemetry=True).pipeline_report()
    ti.sync()

    def measure(callback):
        durations = []
        for _ in range(samples):
            started = time.perf_counter_ns()
            for _ in range(repeats):
                callback()
            durations.append((time.perf_counter_ns() - started) // repeats)
        return _summary(durations)

    run_summary = measure(lambda: (graph.run(args), ti.sync()))
    submit_summary = measure(lambda: graph.submit(args).wait())
    telemetry_summary = measure(lambda: graph.submit(args, telemetry=True).pipeline_report())
    result = output.to_numpy()
    correct = int(result[0]) == 3 and int(result[count - 1]) == count + 2
    pipeline = graph.submit(args, telemetry=True).pipeline_report()
    bounded = pipeline.stages[0].bounded_dispatches[0]
    stats = graph._instance.structured_telemetry_arena_stats
    report_correct = (
        pipeline.schema_version == 2
        and bounded.label == "phase=payload/sweep=0"
        and bounded.source_count == count
        and bounded.useful_count == count
        and bounded.capacity == capacity
        and bounded.snapshot_status == "ticket_device_snapshot"
        and stats["reserved_bytes"] == 8
    )
    return {
        "schema": SCHEMA,
        "arch": (
            "cpu"
            if ti.lang.impl.current_cfg().arch == ti.cpu
            else ("cuda" if ti.lang.impl.current_cfg().arch == ti.cuda else "vulkan")
        ),
        "configuration": {
            "capacity": capacity,
            "count": count,
            "samples": samples,
            "repeats": repeats,
        },
        "timing": {
            "graph_run_and_sync": run_summary,
            "submit_wait_default": submit_summary,
            "submit_pipeline_report": telemetry_summary,
            "telemetry_minus_default_median_ns": (telemetry_summary["median_ns"] - submit_summary["median_ns"]),
            "telemetry_to_default_median_ratio": (
                telemetry_summary["median_ns"] / submit_summary["median_ns"] if submit_summary["median_ns"] else None
            ),
        },
        "memory": stats,
        "pipeline": {
            "task_count": pipeline.task_count,
            "bounded_dispatch_count": pipeline.bounded_dispatch_count,
            "selected_route": bounded.selected_route,
            "physical_launch_kind": bounded.physical_launch_kind,
            "encoded_lanes": bounded.encoded_lanes,
        },
        "correctness": {"payload": correct, "report": report_correct},
        "correct": correct and report_correct,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=("cpu", "cuda", "vulkan"), default="cpu")
    parser.add_argument("--capacity", type=int, default=65536)
    parser.add_argument("--count", type=int, default=4097)
    parser.add_argument("--samples", type=int, default=11)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--performance", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    measurement = prepare_performance_measurement(args.arch, requested=args.performance)

    import taichi_forge as ti

    arch = {"cpu": ti.cpu, "cuda": ti.cuda, "vulkan": ti.vulkan}[args.arch]
    ti.init(arch=arch, enable_fallback=False, offline_cache=False)
    try:
        result = run_initialized(
            ti,
            capacity=args.capacity,
            count=args.count,
            samples=args.samples,
            repeats=args.repeats,
        )
        result.update(finalize_performance_measurement(measurement, correct=result["correct"]))
    finally:
        ti.reset()
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    return 0 if result["correct"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
