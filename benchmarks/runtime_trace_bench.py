"""Paired host-submission benchmark for the bounded runtime trace.

Run one backend per process after confirming that no unrelated Python/GPU
compute workload is active. Trace allocation/start/stop and backend sync are
outside the timed submission loop.
"""

import argparse
import json
import statistics
import time

import taichi_forge as ti


def percentile(samples, fraction):
    ordered = sorted(samples)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=("cpu", "cuda", "vulkan"), required=True)
    parser.add_argument("--iterations", type=int, default=4096)
    parser.add_argument("--rounds", type=int, default=6)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.iterations <= 0 or args.iterations + 8 > (1 << 20):
        raise ValueError("iterations must be in [1, 1048568]")
    if args.rounds <= 0:
        raise ValueError("rounds must be positive")

    arch = {"cpu": ti.cpu, "cuda": ti.cuda, "vulkan": ti.vulkan}[args.arch]
    ti.init(arch=arch, offline_cache=False)
    value = ti.ndarray(ti.i32, shape=1)

    @ti.kernel
    def step(x: ti.types.ndarray()):
        x[0] += 1

    step(value)
    ti.sync()
    for _ in range(128):
        step(value)
    ti.sync()
    prog = ti.lang.impl.get_runtime().prog

    def measure(enabled):
        ti.sync()
        if enabled:
            prog._runtime_trace_start(
                max_threads=1, events_per_thread=args.iterations + 8
            )
        started = time.perf_counter_ns()
        for _ in range(args.iterations):
            step(value)
        elapsed_ns = time.perf_counter_ns() - started
        trace = prog._runtime_trace_stop() if enabled else None
        ti.sync()
        if trace is not None:
            assert trace["recorded_events"] == args.iterations, trace
            assert trace["dropped_events"] == 0, trace
        return elapsed_ns / args.iterations, trace

    trace_off = []
    trace_on = []
    last_trace = None
    for round_index in range(args.rounds):
        order = (False, True) if round_index % 2 == 0 else (True, False)
        for enabled in order:
            sample, trace = measure(enabled)
            (trace_on if enabled else trace_off).append(sample)
            if trace is not None:
                last_trace = trace

    expected = 1 + 128 + args.rounds * 2 * args.iterations
    actual = int(value.to_numpy()[0])
    if actual != expected:
        raise AssertionError(f"result mismatch: expected {expected}, got {actual}")

    off_median = statistics.median(trace_off)
    on_median = statistics.median(trace_on)
    result = {
        "arch": args.arch,
        "iterations_per_sample": args.iterations,
        "rounds": args.rounds,
        "trace_off_ns_per_submission_median": off_median,
        "trace_off_ns_per_submission_p95": percentile(trace_off, 0.95),
        "trace_on_ns_per_submission_median": on_median,
        "trace_on_ns_per_submission_p95": percentile(trace_on, 0.95),
        "trace_on_median_overhead_percent": (on_median / off_median - 1.0)
        * 100.0,
        "trace_buffer_bytes": last_trace["allocated_bytes"],
        "trace_capacity": last_trace["event_capacity"],
        "trace_recorded": last_trace["recorded_events"],
        "trace_dropped": last_trace["dropped_events"],
        "result": actual,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
