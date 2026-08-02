"""Qualify producer-owned bounded launch state and ticket observations.

Run one backend per process.  Every timed batch has an explicit final
``ti.sync()``; Observation samples additionally materialize every ticket so
the reported cost includes completion and host visibility.
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


def _paired(variants, rounds, samples, seed):
    rng = random.Random(seed)
    values = {name: [] for name in variants}
    for _ in range(samples):
        order = list(variants)
        rng.shuffle(order)
        for name in order:
            values[name].append(_measure(variants[name], rounds))
    return {
        name: {
            "median_us": statistics.median(samples_),
            "min_us": min(samples_),
            "max_us": max(samples_),
        }
        for name, samples_ in values.items()
    }


def _memory_snapshot():
    program = impl.get_runtime().prog
    return {
        "runtime": dict(program._runtime_statistics_snapshot()["memory"]),
        "host_pool": dict(ti_core.get_host_memory_pool_stats()),
        "device_pool": dict(ti_core.get_device_memory_pool_stats()),
    }


def _non_growth(before, after):
    runtime_keys = (
        "live_resources",
        "retiring_resources",
        "host_requested_live_bytes",
        "host_raw_bytes",
        "device_requested_live_bytes",
        "device_raw_bytes",
        "device_cached_bytes",
    )
    runtime_ok = all(
        before["runtime"].get(key) is None
        or after["runtime"].get(key) is None
        or after["runtime"][key] <= before["runtime"][key]
        for key in runtime_keys
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
    pools_ok = all(
        before[pool].get(key) is None
        or after[pool].get(key) is None
        or after[pool][key] == before[pool][key]
        for pool in ("host_pool", "device_pool")
        for key in pool_keys
    )
    return runtime_ok and pools_ok


def _producer_launch_benchmark(args):
    capacity = args.capacity
    block_dim = args.block_dim
    values = ti.ndarray(ti.i32, shape=capacity)
    flags = ti.ndarray(ti.i32, shape=capacity)
    input_extent = ti.DeviceExtent(capacity)
    values_host = np.arange(capacity, dtype=np.int32) + 1
    flags_host = ((np.arange(capacity) % 4) != 0).astype(np.int32)
    values.from_numpy(values_host)
    flags.from_numpy(flags_host)
    input_extent.set(capacity * args.active_percent // 100)

    @ti.kernel
    def consume(
        compacted: ti.types.ndarray(dtype=ti.i32, ndim=1),
        extent: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        ti.loop_config(block_dim=block_dim)
        for i in range(capacity):
            if i < ti.device_extent_count(extent):
                value = compacted[i]
                for _ in ti.static(range(args.payload_work)):
                    value = (value * 17 + i) ^ (value >> 3)
                output[i] = value

    values_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "values", ti.i32, ndim=1
    )
    input_extent_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "input_extent", ti.i32, ndim=1
    )
    flags_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "flags", ti.i32, ndim=1
    )
    compacted_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "compacted", ti.i32, ndim=1
    )
    output_extent_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "output_extent", ti.i32, ndim=1
    )
    output_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1
    )

    def build(producer_owned):
        compacted = ti.ndarray(ti.i32, shape=capacity)
        output = ti.ndarray(ti.i32, shape=capacity)
        output_extent = ti.DeviceExtent(capacity)
        launch_state = (
            output_extent.dispatch_state(block_dim) if producer_owned else None
        )
        sequence = ti.algorithms.DevicePrefixSequence(capacity)
        sequence.input(values_arg, input_extent_arg).compact(
            flags_arg,
            compacted_arg,
            output_extent_arg,
            dispatch_state=launch_state,
        )
        builder = ti.graph.GraphBuilder()
        builder.append_native(sequence)
        handle = builder.dispatch_bounded(
            consume,
            compacted_arg,
            output_extent_arg,
            output_arg,
            extent=output_extent_arg,
            capacity=capacity,
            block_dim=block_dim,
            launch_state=launch_state,
        )
        graph = builder.compile()
        runtime_args = {
            "values": values,
            "input_extent": input_extent,
            "flags": flags,
            "compacted": compacted,
            "output_extent": output_extent,
            "output": output,
        }
        return graph, handle, output_extent, output, runtime_args, launch_state

    legacy = build(False)
    producer = build(True)
    variants = {
        "consumer_prepared_packet": lambda: legacy[0].run(legacy[4]),
        "producer_owned_packet": lambda: producer[0].run(producer[4]),
    }
    for call in variants.values():
        for _ in range(args.warmups):
            call()
    ti.sync()
    before = _memory_snapshot()
    samples = _paired(variants, args.rounds, args.samples, args.seed)
    after = _memory_snapshot()

    legacy_us = samples["consumer_prepared_packet"]["median_us"]
    producer_us = samples["producer_owned_packet"]["median_us"]
    legacy[0].run(legacy[4])
    producer[0].run(producer[4])
    ti.sync()
    expected_count = int(
        np.count_nonzero(
            flags_host[: input_extent.snapshot().count]
        )
    )
    correct = (
        legacy[2].snapshot().count == expected_count
        and producer[2].snapshot().count == expected_count
        and np.array_equal(
            legacy[3].to_numpy()[:expected_count],
            producer[3].to_numpy()[:expected_count],
        )
    )
    return {
        "capacity": capacity,
        "active_percent": args.active_percent,
        "compacted_count": expected_count,
        "samples": samples,
        "producer_owned_speedup": legacy_us / producer_us,
        "legacy_preparation_dispatches": (
            legacy[1].capabilities.preparation_dispatches
        ),
        "producer_preparation_dispatches": (
            producer[1].capabilities.preparation_dispatches
        ),
        "producer_state_bytes": (
            0 if producer[5] is None else producer[5].workspace_bytes
        ),
        "correct": correct,
        "memory_stable": _non_growth(before, after),
    }


def _observation_benchmark(args):
    value_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "value", ti.i32, ndim=0
    )

    @ti.kernel
    def advance(value: ti.types.ndarray(dtype=ti.i32, ndim=0)):
        value[None] += 1

    def build(completion_attached):
        os.environ["TI_GRAPH_COMPLETION_ATTACHED_OBSERVATION"] = (
            "1" if completion_attached else "0"
        )
        builder = ti.graph.GraphBuilder()
        builder.dispatch(advance, value_arg)
        builder.observe(value_arg, name="tail")
        graph = builder.compile()
        value = ti.ndarray(ti.i32, shape=())
        value.fill(0)
        runtime_args = {"value": value}
        return graph, value, runtime_args

    legacy = build(False)
    attached = build(True)

    def run(item):
        ticket = item[0].submit(item[2])
        return ticket.observations()["tail"]["value"]

    variants = {
        "deferred_device_copy": lambda: run(legacy),
        "completion_attached": lambda: run(attached),
    }
    for call in variants.values():
        for _ in range(args.warmups):
            call()
    ti.sync()
    program = impl.get_runtime().prog
    transfer_before = dict(program._graph_observation_staging_stats())
    before = _memory_snapshot()
    samples = _paired(
        variants, args.observation_rounds, args.samples, args.seed + 1
    )
    after = _memory_snapshot()
    transfer_after = dict(program._graph_observation_staging_stats())
    legacy_us = samples["deferred_device_copy"]["median_us"]
    attached_us = samples["completion_attached"]["median_us"]
    ti.sync()
    correct = all(
        run(item) == int(item[1].to_numpy()[()])
        for item in (legacy, attached)
    )
    return {
        "samples": samples,
        "completion_attached_speedup": legacy_us / attached_us,
        "correct": correct,
        "memory_stable": _non_growth(before, after),
        "legacy_memory": legacy[0].execution_stats().memory.__dict__,
        "completion_attached_memory": attached[0].execution_stats().memory.__dict__,
        "staging_counter_delta": {
            key: int(transfer_after[key]) - int(transfer_before[key])
            for key in transfer_after
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=("cpu", "cuda", "vulkan"), required=True)
    parser.add_argument("--suite", choices=("all", "producer", "observation"), default="all")
    parser.add_argument("--capacity", type=int, default=1 << 18)
    parser.add_argument("--active-percent", type=int, default=10)
    parser.add_argument("--block-dim", type=int, default=128)
    parser.add_argument("--payload-work", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--observation-rounds", type=int, default=100)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()
    if not 0 <= args.active_percent <= 100:
        parser.error("--active-percent must be in [0, 100]")
    if min(
        args.capacity,
        args.block_dim,
        args.payload_work,
        args.rounds,
        args.observation_rounds,
        args.samples,
        args.warmups,
    ) <= 0:
        parser.error("capacities, work, rounds, samples, and warmups must be positive")

    arch = {"cpu": ti.cpu, "cuda": ti.cuda, "vulkan": ti.vulkan}[args.arch]
    ti.init(arch=arch, offline_cache=False)
    result = {
        "schema_version": 1,
        "arch": ti_core.arch_name(impl.current_cfg().arch),
        "capabilities": ti.graph.dynamic_work_capabilities(),
    }
    if args.suite in ("all", "producer"):
        result["producer_owned_launch"] = _producer_launch_benchmark(args)
    if args.suite in ("all", "observation"):
        result["completion_attached_observation"] = _observation_benchmark(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
