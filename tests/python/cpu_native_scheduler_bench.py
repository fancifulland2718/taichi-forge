"""Manual CPU native-primitive scheduler benchmark.

Run this in a fresh process after deploying the local runtime:

    python tests/python/cpu_native_scheduler_bench.py --items 1048576

Two independent native primitive pipelines share one CPU Program. The script
compares serial execution with two GIL-released Python callers and verifies
transform, reduce, unique scatter, and scatter-add results after every sample.
"""

import argparse
import json
import statistics
import threading
import time

import numpy as np
import taichi_forge as ti
from taichi_forge.lang import impl


def _percentile(values, fraction):
    values = sorted(values)
    return values[min(len(values) - 1, int(np.ceil(len(values) * fraction)) - 1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=int, default=1 << 20)
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.items, args.runs, args.samples, args.threads) <= 0:
        parser.error("items, runs, samples, and threads must be positive")

    ti.init(arch=ti.cpu, cpu_max_num_threads=args.threads)
    runtime = impl.get_runtime()
    runtime.materialize()
    n = args.items
    source_data = np.ones(n, dtype=np.int32)
    reverse_indices = np.arange(n - 1, -1, -1, dtype=np.int32)

    def make_pipeline():
        src = ti.ndarray(ti.i32, shape=n)
        transformed = ti.ndarray(ti.i32, shape=n)
        copied = ti.ndarray(ti.i32, shape=n)
        scatter_sum = ti.ndarray(ti.i32, shape=n)
        indices = ti.ndarray(ti.i32, shape=n)
        reduced = ti.ndarray(ti.i32, shape=1)
        src.from_numpy(source_data)
        indices.from_numpy(reverse_indices)
        workspaces = (
            ti.algorithms.TransformWorkspace(max_items=n),
            ti.algorithms.ReduceWorkspace(max_items=n),
            ti.algorithms.IndexedCopyWorkspace(max_items=n),
            ti.algorithms.ScatterAddWorkspace(max_items=n),
        )

        def reset():
            transformed.fill(0)
            copied.fill(0)
            scatter_sum.fill(0)
            reduced.fill(0)

        def run():
            transform_ws, reduce_ws, copy_ws, scatter_ws = workspaces
            for _ in range(args.runs):
                ti.algorithms.experimental_transform(
                    src, transformed, scale=3, bias=2, method="cpu_native",
                    workspace=transform_ws,
                )
                ti.algorithms.experimental_reduce(
                    transformed, reduced, op="sum", method="cpu_native",
                    workspace=reduce_ws,
                )
                ti.algorithms.experimental_scatter(
                    transformed, indices, copied, method="cpu_native",
                    workspace=copy_ws,
                )
                ti.algorithms.experimental_scatter_add(
                    transformed, indices, scatter_sum, method="cpu_native",
                    workspace=scatter_ws,
                )

        def verify():
            expected = np.full(n, 5, dtype=np.int32)
            np.testing.assert_array_equal(transformed.to_numpy(), expected)
            np.testing.assert_array_equal(copied.to_numpy(), expected)
            np.testing.assert_array_equal(
                scatter_sum.to_numpy(), expected * args.runs
            )
            assert reduced.to_numpy()[0] == n * 5

        return reset, run, verify

    first = make_pipeline()
    second = make_pipeline()
    for pipeline in (first, second):
        pipeline[0]()
        pipeline[1]()
        pipeline[2]()

    serial_ms = []
    concurrent_ms = []
    for _ in range(args.samples):
        for pipeline in (first, second):
            pipeline[0]()
        started = time.perf_counter()
        first[1]()
        second[1]()
        serial_ms.append((time.perf_counter() - started) * 1e3)
        first[2]()
        second[2]()

        for pipeline in (first, second):
            pipeline[0]()
        start = threading.Barrier(2)
        failures = []

        def run_concurrently(pipeline):
            try:
                start.wait(timeout=10)
                pipeline[1]()
            except BaseException as exc:
                failures.append(exc)

        workers = [threading.Thread(target=run_concurrently, args=(pipeline,))
                   for pipeline in (first, second)]
        started = time.perf_counter()
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=120)
        if any(worker.is_alive() for worker in workers):
            raise RuntimeError("concurrent native primitive workers deadlocked")
        if failures:
            raise failures[0]
        concurrent_ms.append((time.perf_counter() - started) * 1e3)
        first[2]()
        second[2]()

    serial_median = statistics.median(serial_ms)
    concurrent_median = statistics.median(concurrent_ms)
    print(json.dumps({
        "items": n,
        "runs_per_pipeline": args.runs,
        "samples": args.samples,
        "cpu_max_num_threads": args.threads,
        "serial_median_ms": round(serial_median, 4),
        "serial_p95_ms": round(_percentile(serial_ms, 0.95), 4),
        "concurrent_median_ms": round(concurrent_median, 4),
        "concurrent_p95_ms": round(_percentile(concurrent_ms, 0.95), 4),
        "concurrent_speedup": round(serial_median / concurrent_median, 4),
    }, sort_keys=True))
    ti.reset()


if __name__ == "__main__":
    main()
