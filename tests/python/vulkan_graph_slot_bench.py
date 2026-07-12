"""Vulkan graph replay slot-cap throughput and memory probe.

Run in a fresh process. The production policy keeps a fixed eight-slot ring;
the probe records saturation fallbacks, throughput, RSS, and device memory.
"""

import argparse
import json
import time

import numpy as np

import taichi_forge as ti
from taichi_forge._lib import core as ti_core

from vulkan_graph_retirement_stress import _gpu_memory_mib, _rss_mib


_TELEMETRY_KEYS = (
    "vulkan_graph_replay_slot_saturation_fallbacks",
)


def _telemetry():
    return {key: int(ti_core.query_int64(key)) for key in _TELEMETRY_KEYS}


def _counter_delta(before, after):
    result = {}
    for key in _TELEMETRY_KEYS:
        result[key] = after[key] - before[key]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=int, default=1 << 20)
    parser.add_argument("--iterations", type=int, default=512)
    parser.add_argument("--dispatches", type=int, default=2)
    parser.add_argument("--work", type=int, default=1)
    args = parser.parse_args()
    if (
        args.items < 1
        or args.iterations < 1
        or args.dispatches < 2
        or not 1 <= args.work <= 64
    ):
        parser.error(
            "items/iterations must be positive, dispatches >= 2, "
            "and work in [1, 64]"
        )

    ti.init(arch=ti.vulkan, enable_fallback=False)

    @ti.kernel
    def increment(values: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in values:
            value = ti.cast(values[i], ti.u32)
            for _ in ti.static(range(args.work)):
                value = value * ti.u32(1664525) + ti.u32(1013904223)
            values[i] = ti.cast(value, ti.i32)

    sym_values = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "values", ti.i32, ndim=1
    )
    builder = ti.graph.GraphBuilder()
    for _ in range(args.dispatches):
        builder.dispatch(increment, sym_values)
    graph = builder.compile()
    values = ti.ndarray(ti.i32, shape=args.items)
    values.fill(0)

    # Compile pipelines and exclude one-time registration from the sample.
    graph.run({"values": values})
    ti.sync()
    cache = graph._instance._backend_executable._jit_cache
    cache.clear_runtime_state()
    values.fill(0)
    ti.sync()
    telemetry_before = _telemetry()

    rss_before_mib = _rss_mib()
    gpu_before_mib = _gpu_memory_mib()
    wall_begin = time.perf_counter()
    submit_begin = wall_begin
    for _ in range(args.iterations):
        graph.run({"values": values})
    submit_seconds = time.perf_counter() - submit_begin
    telemetry_before_sync = _telemetry()
    ti.sync()
    wall_seconds = time.perf_counter() - wall_begin

    expected = 0
    for _ in range(args.dispatches * args.iterations * args.work):
        expected = (expected * 1664525 + 1013904223) & 0xFFFFFFFF
    if expected >= 1 << 31:
        expected -= 1 << 32
    np.testing.assert_array_equal(
        values.to_numpy(),
        np.full(args.items, expected, dtype=np.int32),
    )
    rss_after_mib = _rss_mib()
    gpu_after_mib = _gpu_memory_mib()
    telemetry_after_sync = _telemetry()
    report = {
        "max_slots": 8,
        "items": args.items,
        "iterations": args.iterations,
        "dispatches": args.dispatches,
        "work": args.work,
        "submit_seconds": round(submit_seconds, 6),
        "wall_seconds": round(wall_seconds, 6),
        "submissions_per_second": round(args.iterations / submit_seconds, 2),
        "completed_graphs_per_second": round(args.iterations / wall_seconds, 2),
        "rss_before_mib": rss_before_mib,
        "rss_after_mib": rss_after_mib,
        "rss_delta_mib": (
            None
            if rss_before_mib is None or rss_after_mib is None
            else round(rss_after_mib - rss_before_mib, 3)
        ),
        "gpu_memory_before_mib": gpu_before_mib,
        "gpu_memory_after_mib": gpu_after_mib,
        "telemetry_before_sync": _counter_delta(
            telemetry_before, telemetry_before_sync
        ),
        "telemetry_after_sync": _counter_delta(
            telemetry_before, telemetry_after_sync
        ),
        "result": "pass",
    }
    print(json.dumps(report, sort_keys=True))
    ti.reset()


if __name__ == "__main__":
    main()
