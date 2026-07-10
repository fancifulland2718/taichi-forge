"""Manual CPU scheduler baseline for independent graph submissions.

Run in a fresh process after deploying the local runtime:

    python tests/python/cpu_scheduler_bench.py --items 4194304 --runs 12

Two separately compiled graph executables share one Program but own different
graph caches. Each contains one CPU range-for kernel. The benchmark compares
the total time of running both workloads serially with two GIL-released host
callers running them concurrently, and checks both outputs.
"""

import argparse
import json
import statistics
import threading
import time

import numpy as np

import taichi_forge as ti


def _percentile(values, fraction):
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(np.ceil(fraction * len(ordered))) - 1)
    return ordered[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=int, default=1 << 22)
    parser.add_argument("--runs", type=int, default=12)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.items, args.runs, args.samples, args.threads) <= 0:
        parser.error("items, runs, samples, and threads must be positive")

    ti.init(arch=ti.cpu, cpu_max_num_threads=args.threads)

    @ti.kernel
    def step(values: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in values:
            values[i] += 1

    symbolic_values = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "values", ti.i32, ndim=1)

    def make_graph():
        builder = ti.graph.GraphBuilder()
        builder.dispatch(step, symbolic_values)
        return builder.compile()

    first_graph = make_graph()
    second_graph = make_graph()
    first = ti.ndarray(ti.i32, shape=args.items)
    second = ti.ndarray(ti.i32, shape=args.items)

    # Compile and initialize both graph caches before timing scheduler work.
    first.fill(0)
    second.fill(0)
    first_graph.run({"values": first})
    second_graph.run({"values": second})
    ti.sync()

    serial_ms = []
    concurrent_ms = []
    for _ in range(args.samples):
        first.fill(0)
        second.fill(0)
        started = time.perf_counter()
        for _ in range(args.runs):
            first_graph.run({"values": first})
        for _ in range(args.runs):
            second_graph.run({"values": second})
        ti.sync()
        serial_ms.append((time.perf_counter() - started) * 1e3)
        if not np.all(first.to_numpy() == args.runs) or not np.all(
            second.to_numpy() == args.runs
        ):
            raise RuntimeError("serial scheduler result mismatch")

        first.fill(0)
        second.fill(0)
        start = threading.Barrier(2)
        failures = []
        failure_lock = threading.Lock()

        def run_graph(graph, values):
            try:
                start.wait(timeout=10)
                for _ in range(args.runs):
                    graph.run({"values": values})
            except BaseException as exc:
                with failure_lock:
                    failures.append(exc)

        first_thread = threading.Thread(target=run_graph, args=(first_graph, first))
        second_thread = threading.Thread(target=run_graph, args=(second_graph, second))
        started = time.perf_counter()
        first_thread.start()
        second_thread.start()
        first_thread.join(timeout=120)
        second_thread.join(timeout=120)
        ti.sync()
        if first_thread.is_alive() or second_thread.is_alive():
            raise RuntimeError("concurrent scheduler workers deadlocked")
        if failures:
            raise failures[0]
        concurrent_ms.append((time.perf_counter() - started) * 1e3)
        if not np.all(first.to_numpy() == args.runs) or not np.all(
            second.to_numpy() == args.runs
        ):
            raise RuntimeError("concurrent scheduler result mismatch")

    serial_median = statistics.median(serial_ms)
    concurrent_median = statistics.median(concurrent_ms)
    print(
        json.dumps(
            {
                "items": args.items,
                "runs_per_graph": args.runs,
                "samples": args.samples,
                "cpu_max_num_threads": args.threads,
                "serial_median_ms": round(serial_median, 4),
                "serial_p95_ms": round(_percentile(serial_ms, 0.95), 4),
                "concurrent_median_ms": round(concurrent_median, 4),
                "concurrent_p95_ms": round(_percentile(concurrent_ms, 0.95), 4),
                "concurrent_speedup": round(serial_median / concurrent_median, 4),
            },
            sort_keys=True,
        )
    )

    del first_graph, second_graph, first, second
    ti.reset()


if __name__ == "__main__":
    main()
