"""Compare fixed masking with a device-count strided worker kernel.

This is an internal qualification probe, not a public API example.  Every
variant keeps the count device-owned and uses one end-of-sample ``ti.sync()``.
The worker route launches a bounded fixed worker grid and lets each worker
consume a strided slice of the useful prefix.  It tests whether a portable
kernel contract can reduce sparse masked-lane overhead without host readback.
"""

import argparse
import json
import random
import statistics
import time

import numpy as np

import taichi_forge as ti
from taichi_forge._lib import core as ti_core
from taichi_forge.lang import impl


def _measure(call, rounds):
    start = time.perf_counter_ns()
    for _ in range(rounds):
        call()
    ti.sync()
    return (time.perf_counter_ns() - start) / rounds / 1.0e3


def _paired_samples(variants, rounds, samples, seed):
    names = tuple(variants)
    rng = random.Random(seed)
    result = {name: [] for name in names}
    for _ in range(samples):
        order = list(names)
        rng.shuffle(order)
        for name in order:
            result[name].append(_measure(variants[name], rounds))
    return result


def _summary(samples):
    return {
        name: {
            "median_us": statistics.median(values),
            "min_us": min(values),
            "max_us": max(values),
        }
        for name, values in samples.items()
    }


def _pool_snapshot():
    return {
        "runtime": impl.get_runtime().prog._runtime_statistics_snapshot()["memory"],
        "host": dict(ti_core.get_host_memory_pool_stats()),
        "device": dict(ti_core.get_device_memory_pool_stats()),
    }


def _stable(before, after):
    runtime_keys = (
        "live_resources",
        "retiring_resources",
        "host_requested_live_bytes",
        "host_raw_bytes",
        "device_requested_live_bytes",
        "device_raw_bytes",
        "device_cached_bytes",
    )
    pool_keys = (
        "raw_chunks",
        "raw_bytes",
        "requested_live_bytes",
        "reserved_bytes",
        "committed_bytes",
        "capacity_bytes",
        "used_bytes",
    )
    runtime_ok = all(
        before["runtime"].get(key) is None
        or after["runtime"].get(key) is None
        or after["runtime"][key] <= before["runtime"][key]
        for key in runtime_keys
    )
    pools_ok = all(
        before[pool].get(key) is None
        or after[pool].get(key) is None
        or after[pool][key] == before[pool][key]
        for pool in ("host", "device")
        for key in pool_keys
    )
    return runtime_ok and pools_ok


def run(args):
    capacity = args.capacity
    workers = min(args.workers, capacity)
    block_dim = args.block_dim
    extent = ti.DeviceExtent(capacity)
    source = ti.ndarray(ti.f32, shape=capacity)
    source.from_numpy(
        np.arange(capacity, dtype=np.float32) * np.float32(0.000001)
        + np.float32(0.25)
    )
    outputs = {
        name: ti.ndarray(ti.f32, shape=capacity)
        for name in ("fixed", "bounded", "workers")
    }

    @ti.kernel
    def publish(requested: ti.i32, state: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        ti.device_extent_publish(state, capacity, requested)

    @ti.func
    def evaluate(value: ti.f32, index: ti.i32):
        result = value
        for _ in ti.static(range(args.payload_work)):
            result = result * 0.99991 + ti.sin(result + index * 0.000001)
        return result

    @ti.kernel
    def fixed_payload(
        state: ti.types.ndarray(dtype=ti.i32, ndim=1),
        values: ti.types.ndarray(dtype=ti.f32, ndim=1),
        output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        ti.loop_config(block_dim=block_dim)
        for i in range(capacity):
            if i < ti.device_extent_count(state):
                output[i] = evaluate(values[i], i)

    @ti.kernel
    def worker_payload(
        state: ti.types.ndarray(dtype=ti.i32, ndim=1),
        values: ti.types.ndarray(dtype=ti.f32, ndim=1),
        output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        ti.loop_config(block_dim=block_dim)
        for worker in range(workers):
            count = ti.device_extent_count(state)
            index = worker
            while index < count:
                output[index] = evaluate(values[index], index)
                index += workers

    requested_arg = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "requested", ti.i32)
    extent_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "extent", ti.i32, ndim=1
    )
    source_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "source", ti.f32, ndim=1
    )
    output_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1
    )

    fixed_builder = ti.graph.GraphBuilder()
    fixed_builder.dispatch(publish, requested_arg, extent_arg)
    fixed_builder.dispatch(fixed_payload, extent_arg, source_arg, output_arg)
    fixed_graph = fixed_builder.compile()

    bounded_builder = ti.graph.GraphBuilder()
    bounded_builder.dispatch(publish, requested_arg, extent_arg)
    bounded_handle = bounded_builder.dispatch_bounded(
        fixed_payload,
        extent_arg,
        source_arg,
        output_arg,
        extent=extent_arg,
        capacity=capacity,
        block_dim=block_dim,
    )
    bounded_graph = bounded_builder.compile()

    worker_builder = ti.graph.GraphBuilder()
    worker_builder.dispatch(publish, requested_arg, extent_arg)
    worker_builder.dispatch(worker_payload, extent_arg, source_arg, output_arg)
    worker_graph = worker_builder.compile()

    def variants(count):
        common = {"requested": count, "extent": extent, "source": source}
        return {
            "fixed": lambda: fixed_graph.run(
                {**common, "extent": extent.state, "output": outputs["fixed"]}
            ),
            "bounded": lambda: bounded_graph.run(
                {**common, "output": outputs["bounded"]}
            ),
            "workers": lambda: worker_graph.run(
                {
                    **common,
                    "extent": extent.state,
                    "output": outputs["workers"],
                }
            ),
        }

    counts = tuple(
        sorted(
            {
                0,
                1,
                *(capacity * percent // 100 for percent in args.active_percent),
                capacity,
            }
        )
    )
    for count in counts:
        for call in variants(count).values():
            for _ in range(args.warmups):
                call()
    ti.sync()
    fixed_graph.execution_stats()
    bounded_graph.execution_stats()
    worker_graph.execution_stats()
    for count in counts:
        for call in variants(count).values():
            call()
    ti.sync()

    memory_before = _pool_snapshot()
    cases = []
    correct = True
    for case, count in enumerate(counts):
        samples = _paired_samples(
            variants(count), args.rounds, args.samples, args.seed + case
        )
        report = _summary(samples)
        fixed_us = report["fixed"]["median_us"]
        bounded_us = report["bounded"]["median_us"]
        worker_us = report["workers"]["median_us"]
        cases.append(
            {
                "count": count,
                "active_ratio": count / capacity,
                "samples": report,
                "workers_speedup_over_fixed": fixed_us / worker_us,
                "workers_speedup_over_bounded": bounded_us / worker_us,
            }
        )

        for output in outputs.values():
            output.fill(-1)
        for call in variants(count).values():
            call()
        ti.sync()
        host = {name: output.to_numpy() for name, output in outputs.items()}
        correct = correct and all(
            np.allclose(host[name][:count], host["fixed"][:count])
            for name in ("bounded", "workers")
        )
        if count < capacity:
            correct = correct and all(
                float(values[count]) == -1.0 for values in host.values()
            )

    memory_after = _pool_snapshot()
    return {
        "schema_version": 1,
        "arch": ti_core.arch_name(impl.current_cfg().arch),
        "capacity": capacity,
        "worker_count": workers,
        "block_dim": block_dim,
        "payload_work": args.payload_work,
        "bounded_capabilities": bounded_handle.capabilities.__dict__,
        "cases": cases,
        "correct": correct,
        "memory_stable": _stable(memory_before, memory_after),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=("cpu", "cuda", "vulkan"), required=True)
    parser.add_argument("--capacity", type=int, default=1 << 20)
    parser.add_argument("--workers", type=int, default=8192)
    parser.add_argument("--block-dim", type=int, default=128)
    parser.add_argument("--payload-work", type=int, default=16)
    parser.add_argument(
        "--active-percent", type=int, nargs="+", default=(1, 5, 10, 25, 50)
    )
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    if min(
        args.capacity,
        args.workers,
        args.block_dim,
        args.payload_work,
        args.rounds,
        args.samples,
    ) <= 0:
        parser.error("capacity, workers, block size, work, rounds, and samples must be positive")
    if any(percent < 0 or percent > 100 for percent in args.active_percent):
        parser.error("--active-percent values must be in [0, 100]")

    arch = {"cpu": ti.cpu, "cuda": ti.cuda, "vulkan": ti.vulkan}[args.arch]
    ti.init(arch=arch, offline_cache=False)
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
