"""Measure CUDA host-lock contention and allocator routing under submitters.

Run this in a fresh process after deploying the local runtime:

    python tests/python/cuda_driver_telemetry_stress.py --iterations 1024

The same precompiled CUDA kernel is submitted from 1, 2, and 4 Python
threads. The kernel atomically updates one scalar, so the result checks both
submission completion and the absence of an application-level data race. The
reported counters are sampled every 64 host lock acquisitions; they are an
indicator for deciding whether a later lock-topology change is justified, not
an exact count of every Driver or context call.
"""

import argparse
import json
import threading
import time

import taichi_forge as ti
from taichi_forge._lib import core as ti_core

_DIAGNOSTIC_KEYS = (
    "cuda_driver_lock_sampled_acquisitions",
    "cuda_driver_lock_contended_acquisitions",
    "cuda_context_lock_sampled_acquisitions",
    "cuda_context_lock_contended_acquisitions",
    "cuda_async_allocation_calls",
    "cuda_sync_allocation_fallback_calls",
    "cuda_async_free_calls",
    "cuda_sync_free_fallback_calls",
)


def _snapshot():
    return {key: int(ti_core.query_int64(key)) for key in _DIAGNOSTIC_KEYS}


def _delta(before, after):
    return {key: after[key] - before[key] for key in _DIAGNOSTIC_KEYS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1024)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    args = parser.parse_args()
    if args.iterations <= 0 or not args.threads or any(
            count <= 0 for count in args.threads):
        parser.error("iterations and every thread count must be positive")

    ti.init(arch=ti.cuda, enable_fallback=False)
    counter = ti.field(dtype=ti.i32, shape=())

    @ti.kernel
    def step():
        ti.atomic_add(counter[None], 1)

    try:
        # Compile and materialize before collecting samples, so first-use JIT
        # and runtime initialization do not dominate the submission result.
        step()
        ti.sync()
        counter[None] = 0

        allocator_before = _snapshot()
        temporary = ti.ndarray(dtype=ti.i32, shape=4096)
        temporary.fill(1)
        del temporary
        ti.sync()
        allocator_after = _snapshot()

        samples = []
        for thread_count in args.threads:
            counter[None] = 0
            ti.sync()
            before = _snapshot()
            barrier = threading.Barrier(thread_count + 1)

            def submitter():
                barrier.wait()
                for _ in range(args.iterations):
                    step()

            workers = [
                threading.Thread(target=submitter) for _ in range(thread_count)
            ]
            for worker in workers:
                worker.start()
            started = time.perf_counter()
            barrier.wait()
            for worker in workers:
                worker.join()
            ti.sync()
            elapsed_ms = (time.perf_counter() - started) * 1e3
            after_submissions = _snapshot()
            result = counter[None]
            expected = thread_count * args.iterations
            if result != expected:
                raise RuntimeError(f"expected {expected}, got {result}")
            after = _snapshot()
            samples.append({
                "threads":
                thread_count,
                "submissions":
                expected,
                "elapsed_ms":
                round(elapsed_ms, 4),
                "submissions_per_second":
                round(expected * 1e3 / elapsed_ms, 2),
                "submission_diagnostic_delta":
                _delta(before, after_submissions),
                "result_read_diagnostic_delta":
                _delta(after_submissions, after),
            })

        print(
            json.dumps(
                {
                    "allocator_probe_delta":
                    _delta(allocator_before, allocator_after),
                    "iterations_per_submitter":
                    args.iterations,
                    "samples":
                    samples,
                },
                sort_keys=True,
            ))
    finally:
        ti.reset()


if __name__ == "__main__":
    main()
