"""Measure device-prefix composition and bounded Graph replay.

The benchmark keeps end-of-batch ``ti.sync()`` in every asynchronous variant.
The prefix comparison differs only by the explicit count observation that was
previously needed between primitives.  The Graph comparison keeps the same
producer and payload across direct, fixed-Graph, and bounded-Graph routes.
"""

import argparse
import json
import os
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
    orders = [list(names) for _ in range(samples)]
    rng = random.Random(seed)
    for order in orders:
        rng.shuffle(order)
    result = {name: [] for name in names}
    for order in orders:
        for name in order:
            result[name].append(_measure(variants[name], rounds))
    return result


def _summarize(samples):
    return {
        name: {
            "median_us": statistics.median(values),
            "min_us": min(values),
            "max_us": max(values),
        }
        for name, values in samples.items()
    }


def _memory_non_growth(before, after):
    keys = (
        "live_resources",
        "retiring_resources",
        "host_requested_live_bytes",
        "host_raw_bytes",
        "device_requested_live_bytes",
        "device_raw_bytes",
        "device_cached_bytes",
    )
    return all(
        before.get(key) is None
        or after.get(key) is None
        or after[key] <= before[key]
        for key in keys
    )


def _pool_ownership_stable(before, after):
    keys = (
        "raw_chunks",
        "raw_bytes",
        "requested_live_bytes",
        "reserved_bytes",
        "committed_bytes",
        "capacity_bytes",
        "used_bytes",
    )
    return all(
        before.get(key) is None
        or after.get(key) is None
        or after[key] == before[key]
        for key in keys
    )


def _prefix_benchmark(args):
    capacity = args.prefix_capacity
    count = capacity * args.sparse_percent // 100
    values = ti.ndarray(ti.i32, shape=capacity)
    flags = ti.ndarray(ti.i32, shape=capacity)
    compacted = ti.ndarray(ti.i32, shape=capacity)
    extent = ti.DeviceExtent(capacity)
    output_extent = ti.DeviceExtent(capacity)
    workspace = ti.algorithms.DevicePrefixWorkspace(capacity)

    @ti.kernel
    def initialize(
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        mask: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in range(capacity):
            source[i] = i % 17 + 1
            mask[i] = i % 4 != 0

    initialize(values, flags)
    extent.set(count)
    source = ti.algorithms.device_prefix(values, extent, workspace=workspace)
    compacted_prefix = ti.algorithms.device_prefix(
        compacted, output_extent, workspace=workspace
    )

    def device_resident():
        source.compact(flags, compacted, output_extent)
        compacted_prefix.scan()

    def host_observed():
        source.compact(flags, compacted, output_extent)
        output_extent.snapshot()
        compacted_prefix.scan()

    for _ in range(args.warmups):
        device_resident()
        host_observed()
    ti.sync()
    program = impl.get_runtime().prog
    before = program._runtime_statistics_snapshot()["memory"]
    host_pool_before = dict(ti_core.get_host_memory_pool_stats())
    device_pool_before = dict(ti_core.get_device_memory_pool_stats())
    samples = _paired_samples(
        {
            "device_resident": device_resident,
            "host_observed": host_observed,
        },
        args.prefix_rounds,
        args.samples,
        args.seed,
    )
    after = program._runtime_statistics_snapshot()["memory"]
    host_pool_stable = _pool_ownership_stable(
        host_pool_before, dict(ti_core.get_host_memory_pool_stats())
    )
    device_pool_stable = _pool_ownership_stable(
        device_pool_before, dict(ti_core.get_device_memory_pool_stats())
    )

    device_us = statistics.median(samples["device_resident"])
    host_us = statistics.median(samples["host_observed"])
    expected_values = np.arange(count, dtype=np.int64)
    expected_values = expected_values[expected_values % 4 != 0] % 17 + 1
    expected = np.cumsum(expected_values, dtype=np.int64).astype(np.int32)
    result_count = output_extent.snapshot().count
    actual = compacted.to_numpy()[:result_count]
    return {
        "capacity": capacity,
        "useful_count": count,
        "compacted_count": result_count,
        "samples": _summarize(samples),
        "speedup_over_host_observation": host_us / device_us,
        "host_observations_per_sample": args.prefix_rounds,
        "correct": np.array_equal(actual, expected),
        "workspace_bytes": workspace.workspace_bytes_current,
        "workspace_peak_bytes": workspace.workspace_bytes_peak,
        "workspace_allocations": workspace.allocation_count,
        "runtime_memory_non_growth": _memory_non_growth(before, after),
        "host_pool_stable": host_pool_stable,
        "device_pool_stable": device_pool_stable,
    }


def _graph_benchmark(args):
    capacity = args.graph_capacity
    block_dim = args.block_dim
    extent = ti.DeviceExtent(capacity)
    input_values = ti.ndarray(ti.f32, shape=capacity)
    outputs = {
        name: ti.ndarray(ti.f32, shape=capacity)
        for name in ("direct", "fixed_graph", "bounded_graph")
    }
    input_values.from_numpy(
        (np.arange(capacity, dtype=np.float32) % 251) * np.float32(0.001)
        + np.float32(0.25)
    )

    @ti.kernel
    def publish(
        requested: ti.i32,
        state: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        ti.device_extent_publish(state, capacity, requested)

    @ti.kernel
    def payload(
        state: ti.types.ndarray(dtype=ti.i32, ndim=1),
        source: ti.types.ndarray(dtype=ti.f32, ndim=1),
        output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        ti.loop_config(block_dim=block_dim)
        for i in range(capacity):
            if i < ti.device_extent_count(state):
                value = source[i]
                for _ in ti.static(range(args.payload_work)):
                    value = value * 0.99991 + ti.sin(value + i * 0.000001)
                output[i] = value

    requested_arg = ti.graph.Arg(
        ti.graph.ArgKind.SCALAR, "requested", ti.i32
    )
    extent_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "extent", ti.i32, ndim=1
    )
    input_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "input", ti.f32, ndim=1
    )
    output_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1
    )

    fixed_builder = ti.graph.GraphBuilder()
    fixed_builder.dispatch(publish, requested_arg, extent_arg)
    fixed_builder.dispatch(payload, extent_arg, input_arg, output_arg)
    fixed_graph = fixed_builder.compile()

    bounded_builder = ti.graph.GraphBuilder()
    bounded_builder.dispatch(publish, requested_arg, extent_arg)
    handle = bounded_builder.dispatch_bounded(
        payload,
        extent_arg,
        input_arg,
        output_arg,
        extent=extent_arg,
        capacity=capacity,
        block_dim=block_dim,
    )
    bounded_graph = bounded_builder.compile()

    def variants_for(count):
        fixed_args = {
            "requested": count,
            "extent": extent.state,
            "input": input_values,
            "output": outputs["fixed_graph"],
        }
        bounded_args = {
            "requested": count,
            "extent": extent,
            "input": input_values,
            "output": outputs["bounded_graph"],
        }
        return {
            "direct": lambda: (
                publish(count, extent.state),
                payload(extent.state, input_values, outputs["direct"]),
            ),
            "fixed_graph": lambda: fixed_graph.run(fixed_args),
            "bounded_graph": lambda: bounded_graph.run(bounded_args),
        }

    counts = {
        "zero": 0,
        "sparse": capacity * args.sparse_percent // 100,
        "full": capacity,
    }
    for count in counts.values():
        for call in variants_for(count).values():
            for _ in range(args.warmups):
                call()
    ti.sync()
    fixed_graph.execution_stats()
    bounded_graph.execution_stats()
    for count in counts.values():
        for call in variants_for(count).values():
            for _ in range(args.warmups):
                call()
    ti.sync()

    program = impl.get_runtime().prog
    before = program._runtime_statistics_snapshot()["memory"]
    host_pool_before = dict(ti_core.get_host_memory_pool_stats())
    device_pool_before = dict(ti_core.get_device_memory_pool_stats())
    reports = {}
    for index, (case, count) in enumerate(counts.items()):
        samples = _paired_samples(
            variants_for(count),
            args.graph_rounds,
            args.samples,
            args.seed + index,
        )
        summary = _summarize(samples)
        fixed_us = summary["fixed_graph"]["median_us"]
        bounded_us = summary["bounded_graph"]["median_us"]
        summary["bounded_speedup_over_fixed"] = fixed_us / bounded_us
        summary["bounded_speedup_over_direct"] = (
            summary["direct"]["median_us"] / bounded_us
        )
        reports[case] = {"count": count, **summary}
    after = program._runtime_statistics_snapshot()["memory"]
    host_pool_stable = _pool_ownership_stable(
        host_pool_before, dict(ti_core.get_host_memory_pool_stats())
    )
    device_pool_stable = _pool_ownership_stable(
        device_pool_before, dict(ti_core.get_device_memory_pool_stats())
    )

    correct = True
    for name, count in counts.items():
        for output in outputs.values():
            output.fill(-1)
        variants = variants_for(count)
        for call in variants.values():
            call()
        ti.sync()
        snapshot = handle.snapshot(extent)
        if snapshot.useful_count != count:
            raise RuntimeError(f"bounded count mismatch for {name}")
        host_outputs = {key: value.to_numpy() for key, value in outputs.items()}
        if count:
            reference = host_outputs["direct"][:count]
            correct = correct and np.allclose(
                host_outputs["fixed_graph"][:count], reference
            )
            correct = correct and np.allclose(
                host_outputs["bounded_graph"][:count], reference
            )
        if count < capacity:
            correct = correct and all(
                float(output[count]) == -1.0 for output in host_outputs.values()
            )
    return {
        "capacity": capacity,
        "block_dim": block_dim,
        "payload_work": args.payload_work,
        "capabilities": handle.capabilities.__dict__,
        "cases": reports,
        "workspace_bytes": handle.workspace_bytes,
        "workspace_allocations": handle.workspace_allocation_count,
        "correct": correct,
        "runtime_memory_non_growth": _memory_non_growth(before, after),
        "host_pool_stable": host_pool_stable,
        "device_pool_stable": device_pool_stable,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=("cpu", "cuda", "vulkan"), required=True)
    parser.add_argument("--suite", choices=("all", "prefix", "graph"), default="all")
    parser.add_argument("--prefix-capacity", type=int, default=1 << 18)
    parser.add_argument("--graph-capacity", type=int, default=1 << 20)
    parser.add_argument("--sparse-percent", type=int, default=10)
    parser.add_argument("--block-dim", type=int, default=128)
    parser.add_argument("--payload-work", type=int, default=16)
    parser.add_argument("--prefix-rounds", type=int, default=20)
    parser.add_argument("--graph-rounds", type=int, default=100)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument(
        "--cuda-route",
        choices=("auto", "device_update", "masked_capacity"),
        default="auto",
    )
    parser.add_argument(
        "--cpu-route",
        choices=("auto", "exact_scheduler", "masked_capacity"),
        default="auto",
    )
    args = parser.parse_args()
    if not 0 <= args.sparse_percent <= 100:
        parser.error("--sparse-percent must be in [0, 100]")
    if min(
        args.prefix_capacity,
        args.graph_capacity,
        args.block_dim,
        args.payload_work,
        args.prefix_rounds,
        args.graph_rounds,
        args.samples,
    ) <= 0:
        parser.error("capacities, block size, rounds, and samples must be positive")

    if args.arch == "cuda":
        os.environ["TI_CUDA_BOUNDED_DISPATCH_MODE"] = args.cuda_route
    elif args.arch == "cpu":
        os.environ["TI_CPU_BOUNDED_DISPATCH_MODE"] = args.cpu_route
    arch = {"cpu": ti.cpu, "cuda": ti.cuda, "vulkan": ti.vulkan}[args.arch]
    ti.init(arch=arch, offline_cache=False)
    requested_route = (
        args.cuda_route
        if args.arch == "cuda"
        else args.cpu_route if args.arch == "cpu" else "not_applicable"
    )
    bounded_capabilities = ti.graph.bounded_dispatch_capabilities()
    expected_selected = {
        "device_update": "exact_device_grid_update",
        "exact_scheduler": "exact_cpu_scheduler",
        "masked_capacity": "masked_capacity",
    }.get(requested_route)
    route_identity_valid = requested_route in ("auto", "not_applicable") or (
        bounded_capabilities["selected_route"] == expected_selected
    )
    if not route_identity_valid:
        raise RuntimeError(
            "bounded benchmark route mismatch: "
            f"requested={requested_route}, "
            f"selected={bounded_capabilities['selected_route']}"
        )
    result = {
        "schema_version": 2,
        "arch": ti_core.arch_name(impl.current_cfg().arch),
        "requested_route": requested_route,
        "selected_route": bounded_capabilities["selected_route"],
        "route_identity_valid": route_identity_valid,
        "bounded_dispatch_capabilities": bounded_capabilities,
    }
    if args.suite in ("all", "prefix"):
        result["device_prefix"] = _prefix_benchmark(args)
    if args.suite in ("all", "graph"):
        result["bounded_graph"] = _graph_benchmark(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
