"""Manual CUDA graph replay baseline for backend-runtime changes.

Run this in a fresh process after deploying the local runtime:

    python tests/python/cuda_graph_runtime_bench.py --items 1048576

The first run includes graph compilation/capture. Every measured replay is
synchronized so median and p95 represent completion latency rather than queued
submission throughput. The final reset is intentional: it also checks that
capture-owned buffers do not outlive the CUDA program.
"""

import argparse
import json
import shutil
import statistics
import subprocess
import time

import numpy as np

import taichi_forge as ti


def _gpu_memory_mib():
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return None
    result = subprocess.run(
        [nvidia_smi, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    try:
        return int(result.stdout.splitlines()[0].strip())
    except (IndexError, ValueError):
        return None


def _percentile(values, fraction):
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(np.ceil(fraction * len(ordered))) - 1)
    return ordered[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=int, default=1 << 20)
    parser.add_argument("--warm-runs", type=int, default=20)
    parser.add_argument("--measure-runs", type=int, default=80)
    args = parser.parse_args()
    if args.items <= 0 or args.warm_runs < 0 or args.measure_runs < 5:
        parser.error("items must be positive, warm-runs non-negative, measure-runs >= 5")

    ti.init(arch=ti.cuda, enable_fallback=False)

    @ti.kernel
    def increment(values: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in values:
            values[i] += 1

    symbolic_values = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "values", ti.i32, ndim=1)
    builder = ti.graph.GraphBuilder()
    sequence = builder.create_sequential()
    sequence.dispatch(increment, symbolic_values)
    for _ in range(4):
        builder.append(sequence)
    graph = builder.compile()
    values = ti.ndarray(ti.i32, shape=args.items)
    values.fill(0)

    memory_before_mib = _gpu_memory_mib()
    started = time.perf_counter()
    graph.run({"values": values})
    ti.sync()
    cold_ms = (time.perf_counter() - started) * 1e3

    for _ in range(args.warm_runs):
        graph.run({"values": values})
    ti.sync()

    replay_ms = []
    for _ in range(args.measure_runs):
        started = time.perf_counter()
        graph.run({"values": values})
        ti.sync()
        replay_ms.append((time.perf_counter() - started) * 1e3)

    expected = 4 * (1 + args.warm_runs + args.measure_runs)
    actual = values.to_numpy()
    if not np.all(actual == expected):
        raise RuntimeError(f"graph result mismatch: expected {expected}, got {actual[0]}")

    memory_after_mib = _gpu_memory_mib()
    print(
        json.dumps(
            {
                "items": args.items,
                "warm_runs": args.warm_runs,
                "measure_runs": args.measure_runs,
                "cold_ms": round(cold_ms, 4),
                "replay_median_ms": round(statistics.median(replay_ms), 4),
                "replay_p95_ms": round(_percentile(replay_ms, 0.95), 4),
                "gpu_memory_before_mib": memory_before_mib,
                "gpu_memory_after_mib": memory_after_mib,
                "result": int(actual[0]),
            },
            sort_keys=True,
        )
    )

    del graph
    del values
    ti.reset()


if __name__ == "__main__":
    main()
