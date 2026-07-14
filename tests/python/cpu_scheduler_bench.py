"""CPU dense-Field Graph scheduler benchmark for 1/2/4/8 host callers.

Each caller owns an independent Graph and dense Field, while all kernels use
one Program-owned bounded worker pool. The benchmark compares serial and
GIL-released concurrent submission, verifies every result, and reports caller
fairness plus process RSS. It intentionally does not create one worker pool per
Graph or Python thread.

Example:

    python tests/python/cpu_scheduler_bench.py --items 1048576 \
        --runs 12 --samples 5 --threads 4 --caller-counts 1 2 4 8
"""

import argparse
import json
import statistics
import threading
import time
from pathlib import Path

import numpy as np

import taichi_forge as ti


def _percentile(values, fraction):
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(np.ceil(fraction * len(ordered))) - 1)
    return ordered[index]


def _rss_mb():
    try:
        import psutil  # pylint: disable=import-outside-toplevel

        return psutil.Process().memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        return None


def _summary(values):
    return {
        "median_ms": statistics.median(values),
        "p95_ms": _percentile(values, 0.95),
        "samples_ms": values,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=int, default=1 << 20)
    parser.add_argument("--runs", type=int, default=12)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--caller-counts", nargs="+", type=int, default=[1, 2, 4, 8]
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(
        args.items,
        args.runs,
        args.samples,
        args.threads,
        *args.caller_counts,
    ) <= 0:
        parser.error("all numeric arguments must be positive")

    ti.init(
        arch=ti.cpu,
        cpu_max_num_threads=args.threads,
        enable_fallback=False,
        offline_cache=False,
    )

    @ti.kernel
    def step(values: ti.template()):
        for i in values:
            values[i] += 1

    max_callers = max(args.caller_counts)
    fields = [ti.field(ti.i32, shape=args.items) for _ in range(max_callers)]
    graphs = []
    for values in fields:
        builder = ti.graph.GraphBuilder()
        builder.dispatch(step, template_args={"values": values})
        graphs.append(builder.compile())

    # Compile every specialization and initialize every graph cache before
    # timing scheduler work.
    for graph in graphs:
        graph.run({})
    for values in fields:
        values.fill(0)

    rss_before_mb = _rss_mb()
    results = []
    for caller_count in args.caller_counts:
        selected_graphs = graphs[:caller_count]
        selected_fields = fields[:caller_count]
        serial_ms = []
        concurrent_ms = []
        fairness = []

        def check(expected):
            for values in selected_fields:
                actual = values.to_numpy()
                if not np.all(actual == expected):
                    raise RuntimeError(
                        f"scheduler result mismatch: expected {expected}"
                    )

        for _ in range(args.samples):
            for values in selected_fields:
                values.fill(0)
            started = time.perf_counter()
            # Match the cross-Field working-set rotation produced by FIFO host
            # callers. Grouping all runs of one Field would keep 16 MiB hot in
            # LLC and incorrectly charge cache-locality loss to the scheduler.
            for _ in range(args.runs):
                for graph in selected_graphs:
                    graph.run({})
            ti.sync()
            serial_ms.append((time.perf_counter() - started) * 1e3)
            check(args.runs)

            for values in selected_fields:
                values.fill(0)
            start = threading.Barrier(caller_count)
            failures = []
            failure_lock = threading.Lock()
            worker_ms = [0.0] * caller_count

            def run_graph(index, graph):
                try:
                    start.wait(timeout=10.0)
                    worker_start = time.perf_counter()
                    for _ in range(args.runs):
                        graph.run({})
                    worker_ms[index] = (
                        time.perf_counter() - worker_start
                    ) * 1e3
                except BaseException as exc:
                    with failure_lock:
                        failures.append(exc)

            threads = [
                threading.Thread(target=run_graph, args=(index, graph))
                for index, graph in enumerate(selected_graphs)
            ]
            started = time.perf_counter()
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=120.0)
            ti.sync()
            if any(thread.is_alive() for thread in threads):
                raise RuntimeError("concurrent scheduler workers deadlocked")
            if failures:
                raise failures[0]
            concurrent_ms.append((time.perf_counter() - started) * 1e3)
            positive_worker_ms = [value for value in worker_ms if value > 0]
            fairness.append(
                min(positive_worker_ms) / max(positive_worker_ms)
            )
            check(args.runs)

        serial = _summary(serial_ms)
        concurrent = _summary(concurrent_ms)
        results.append(
            {
                "callers": caller_count,
                "serial": serial,
                "concurrent": concurrent,
                "concurrent_speedup": (
                    serial["median_ms"] / concurrent["median_ms"]
                ),
                "concurrent_invocations_per_second": (
                    caller_count
                    * args.runs
                    / (concurrent["median_ms"] / 1000.0)
                ),
                "caller_fairness_min_over_max_median": statistics.median(
                    fairness
                ),
                "caller_fairness_samples": fairness,
            }
        )

    report = {
        "schema": "taichi_forge.cpu_dense_field_graph_callers.v1",
        "items_per_caller": args.items,
        "runs_per_caller": args.runs,
        "samples": args.samples,
        "cpu_max_num_threads": args.threads,
        "rss_before_mb": rss_before_mb,
        "rss_after_mb": _rss_mb(),
        "results": results,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")

    del graphs, fields
    ti.reset()


if __name__ == "__main__":
    main()
