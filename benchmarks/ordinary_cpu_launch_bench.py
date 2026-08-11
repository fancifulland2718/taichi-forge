"""Measure steady-state CPU launch cost and native stage attribution.

The benchmark keeps payloads intentionally tiny so fixed launch work remains
visible.  It reports distributions rather than a single best sample and never
enables attribution inside the timed window unless explicitly requested.
"""

import argparse
import dataclasses
import json
import os
import statistics
import time


def _percentile(samples, fraction):
    ordered = sorted(samples)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def _measure(call, *, batches, calls_per_batch, warmup):
    for _ in range(warmup):
        call()
    samples = []
    for _ in range(batches):
        start = time.perf_counter_ns()
        for _ in range(calls_per_batch):
            call()
        samples.append((time.perf_counter_ns() - start) / calls_per_batch)
    return {
        "samples": len(samples),
        "median_ns": statistics.median(samples),
        "p10_ns": _percentile(samples, 0.10),
        "p90_ns": _percentile(samples, 0.90),
        "mean_ns": statistics.fmean(samples),
        "cv": statistics.pstdev(samples) / statistics.fmean(samples),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=40)
    parser.add_argument("--calls-per-batch", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--attribution", action="store_true")
    parser.add_argument("--package", choices=("forge", "vanilla"), default="forge")
    args = parser.parse_args()

    if args.attribution:
        os.environ["TI_DEBUG_ORDINARY_LAUNCH_ATTRIBUTION"] = "1"
    os.environ.setdefault("TI_SKIP_VERSION_CHECK", "ON")

    if args.package == "forge":
        import taichi_forge as ti
    else:
        import taichi as ti

    ti.init(
        arch=ti.cpu,
        cpu_max_num_threads=args.threads,
        offline_cache=False,
    )

    @ti.kernel
    def scalar_only(value: ti.i32) -> ti.i32:
        return value + 1

    @ti.kernel
    def one_resource(value: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        value[0] += 1

    @ti.kernel
    def two_resources(
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        destination: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        destination[0] = source[0] + 1

    @ti.kernel
    def four_resources(
        first: ti.types.ndarray(dtype=ti.i32, ndim=1),
        second: ti.types.ndarray(dtype=ti.i32, ndim=1),
        third: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        output[0] = first[0] + second[0] + third[0]

    @ti.kernel
    def range_fill(
        output: ti.types.ndarray(dtype=ti.f32, ndim=1), value: ti.f32
    ):
        for i in output:
            output[i] = value

    arrays = [ti.ndarray(ti.i32, shape=1) for _ in range(4)]
    for index, array in enumerate(arrays):
        array.fill(index + 1)
    wide = ti.ndarray(ti.f32, shape=65_536)
    range_fill_manifest = None
    if args.package == "forge" and hasattr(range_fill, "task_manifest"):
        range_fill_manifest = tuple(
            dataclasses.asdict(task)
            for task in range_fill.task_manifest(wide, 1.25)
        )

    cases = {
        "zero_resource": lambda: scalar_only(3),
        "one_resource": lambda: one_resource(arrays[0]),
        "two_resources": lambda: two_resources(arrays[0], arrays[1]),
        "four_resources": lambda: four_resources(*arrays),
        "range_fill_65k": lambda: range_fill(wide, 1.25),
    }
    program = ti.lang.impl.get_runtime().prog
    results = {}
    for name, call in cases.items():
        call()
        ti.sync()
        if args.attribution and args.package == "forge":
            program._debug_reset_ordinary_launch_attribution()
        timing = _measure(
            call,
            batches=args.batches,
            calls_per_batch=args.calls_per_batch,
            warmup=args.warmup,
        )
        ti.sync()
        results[name] = {
            "timing": timing,
            "attribution": (
                dict(program._debug_ordinary_launch_attribution())
                if args.attribution and args.package == "forge"
                else None
            ),
        }

    print(
        json.dumps(
            {
                "schema_version": 1,
                "backend": "cpu",
                "package": args.package,
                "threads": args.threads,
                "batches": args.batches,
                "calls_per_batch": args.calls_per_batch,
                "attribution_enabled": args.attribution,
                "range_fill_task_manifest": range_fill_manifest,
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
    )
    ti.reset()


if __name__ == "__main__":
    main()
