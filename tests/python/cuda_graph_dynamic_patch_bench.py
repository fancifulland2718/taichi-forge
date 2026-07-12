"""A/B latency and memory probe for CUDA graph dynamic argument patching.

The patch path alternates same-structure ndarray bindings and scalar values on
one captured graph. The baseline explicitly clears the executable before every
run, reproducing the previous synchronize + recapture behavior. Each sample is
synchronized so median/p95 measure completed work rather than enqueue rate.
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


def _build_graph():
    @ti.kernel
    def transform(
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
        bias: ti.i32,
    ):
        for i in output:
            output[i] = source[i] + bias

    sym_source = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "source", ti.i32, ndim=1
    )
    sym_output = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1
    )
    sym_bias = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "bias", ti.i32)
    builder = ti.graph.GraphBuilder()
    sequence = builder.create_sequential()
    sequence.dispatch(transform, sym_source, sym_output, sym_bias)
    for _ in range(4):
        builder.append(sequence)
    return builder.compile()


def _cache(graph):
    return graph._instance._backend_executable._jit_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=int, default=1 << 12)
    parser.add_argument("--warm-runs", type=int, default=20)
    parser.add_argument("--measure-runs", type=int, default=120)
    parser.add_argument("--batch-runs", type=int, default=1000)
    args = parser.parse_args()
    if (
        args.items <= 0
        or args.warm_runs < 1
        or args.measure_runs < 10
        or args.batch_runs < 10
    ):
        parser.error(
            "items must be positive and warm/measure/batch runs must be valid"
        )

    ti.init(arch=ti.cuda, enable_fallback=False)
    source_np = [
        np.arange(args.items, dtype=np.int32),
        np.arange(args.items, dtype=np.int32)[::-1].copy(),
    ]
    sources = [ti.ndarray(ti.i32, shape=args.items) for _ in range(2)]
    outputs = [ti.ndarray(ti.i32, shape=args.items) for _ in range(2)]
    for source, values in zip(sources, source_np):
        source.from_numpy(values)

    patch_graph = _build_graph()
    recapture_graph = _build_graph()

    def arguments(index):
        slot = index & 1
        return {
            "source": sources[slot],
            "output": outputs[slot],
            "bias": index,
        }

    for index in range(args.warm_runs):
        patch_graph.run(arguments(index))
    ti.sync()
    recapture_graph.run(arguments(0))
    ti.sync()

    memory_before_mib = _gpu_memory_mib()

    def measure(graph, force_recapture):
        samples = []
        cache = _cache(graph)
        for index in range(args.measure_runs):
            started = time.perf_counter()
            if force_recapture:
                cache.clear_runtime_state()
            graph.run(arguments(index))
            ti.sync()
            samples.append((time.perf_counter() - started) * 1e3)
        return samples

    recapture_ms = measure(recapture_graph, force_recapture=True)
    patch_ms = measure(patch_graph, force_recapture=False)

    def measure_batch(graph, force_recapture):
        cache = _cache(graph)
        started = time.perf_counter()
        for index in range(args.batch_runs):
            if force_recapture:
                cache.clear_runtime_state()
            graph.run(arguments(index))
        ti.sync()
        return time.perf_counter() - started

    recapture_batch_seconds = measure_batch(
        recapture_graph, force_recapture=True
    )
    patch_batch_seconds = measure_batch(patch_graph, force_recapture=False)

    final_index = args.batch_runs - 1
    final_slot = final_index & 1
    np.testing.assert_array_equal(
        outputs[final_slot].to_numpy(),
        source_np[final_slot] + final_index,
    )
    memory_after_mib = _gpu_memory_mib()

    recapture_median = statistics.median(recapture_ms)
    patch_median = statistics.median(patch_ms)
    print(
        json.dumps(
            {
                "items": args.items,
                "dispatches": 4,
                "warm_runs": args.warm_runs,
                "measure_runs": args.measure_runs,
                "batch_runs": args.batch_runs,
                "patch_median_ms": round(patch_median, 4),
                "patch_p95_ms": round(_percentile(patch_ms, 0.95), 4),
                "recapture_median_ms": round(recapture_median, 4),
                "recapture_p95_ms": round(
                    _percentile(recapture_ms, 0.95), 4
                ),
                "median_speedup": round(recapture_median / patch_median, 3),
                "patch_batch_submissions_per_second": round(
                    args.batch_runs / patch_batch_seconds, 2
                ),
                "recapture_batch_submissions_per_second": round(
                    args.batch_runs / recapture_batch_seconds, 2
                ),
                "batch_speedup": round(
                    recapture_batch_seconds / patch_batch_seconds, 3
                ),
                "gpu_memory_before_mib": memory_before_mib,
                "gpu_memory_after_mib": memory_after_mib,
                "result": "pass",
            },
            sort_keys=True,
        )
    )

    del patch_graph
    del recapture_graph
    del sources
    del outputs
    ti.reset()


if __name__ == "__main__":
    main()
