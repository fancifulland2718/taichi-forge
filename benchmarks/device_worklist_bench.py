"""Qualify fixed-capacity worklists and deterministic keyed claims.

Run one backend per process.  Timed batches end with ``ti.sync()``.  The
atomic suite compares host-readback, fixed masked Graph, and bounded Graph
pipelines, including the legacy launch-state adapter as a paired ownership
control. The conflict suite compares a device-resident stable claim with a full
host round trip and reports build/first/warm costs separately.
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
    return {name: _summarize(samples_) for name, samples_ in values.items()}, values


def _summarize(samples):
    mean = statistics.fmean(samples)
    return {
        "median_us": statistics.median(samples),
        "p95_us": float(np.percentile(samples, 95)),
        "p99_us": float(np.percentile(samples, 99)),
        "mean_us": mean,
        "cv": statistics.pstdev(samples) / mean if mean else 0.0,
        "min_us": min(samples),
        "max_us": max(samples),
    }


def _paired_speedup(numerator, denominator):
    if len(numerator) != len(denominator):
        raise RuntimeError("paired benchmark sample counts do not match")
    ratios = [left / right for left, right in zip(numerator, denominator)]
    mean = statistics.fmean(ratios)
    return {
        "median": statistics.median(ratios),
        "p05": float(np.percentile(ratios, 5)),
        "p95": float(np.percentile(ratios, 95)),
        "mean": mean,
        "cv": statistics.pstdev(ratios) / mean if mean else 0.0,
        "min": min(ratios),
        "max": max(ratios),
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


def _build_us(build):
    start = time.perf_counter_ns()
    result = build()
    return result, (time.perf_counter_ns() - start) / 1.0e3


def _atomic_pipeline(args):
    capacity = args.capacity
    produced = capacity * args.active_percent // 100
    block_dim = args.block_dim

    @ti.kernel
    def produce(
        values: ti.types.ndarray(dtype=ti.i32, ndim=1),
        extent_state: ti.types.ndarray(dtype=ti.i32, ndim=1),
        generated: ti.types.ndarray(dtype=ti.i32, ndim=0),
        overflow: ti.types.ndarray(dtype=ti.i32, ndim=0),
        limit: ti.i32,
        count: ti.i32,
    ):
        for i in range(count):
            ti.algorithms.device_worklist_append(
                values,
                extent_state,
                generated,
                overflow,
                limit,
                i + 1,
            )

    @ti.kernel
    def consume_masked(
        values: ti.types.ndarray(dtype=ti.i32, ndim=1),
        extent_state: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        ti.loop_config(block_dim=block_dim)
        for i in range(capacity):
            if i < ti.device_extent_count(extent_state):
                value = values[i]
                for _ in ti.static(range(args.payload_work)):
                    value = value * 7 + 3
                output[i] = value

    @ti.kernel
    def consume_exact(
        values: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
        count: ti.i32,
    ):
        ti.loop_config(block_dim=block_dim)
        for i in range(count):
            value = values[i]
            for _ in ti.static(range(args.payload_work)):
                value = value * 7 + 3
            output[i] = value

    def build_graph(
        name,
        bounded,
        *,
        compatibility_state=False,
    ):
        worklist = ti.algorithms.DeviceWorklist(capacity, ti.i32)
        graph_args = worklist.graph_args(name)
        count_arg = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "count", ti.i32)
        output_args = tuple(
            ti.graph.Arg(
                ti.graph.ArgKind.NDARRAY,
                f"output_{index}",
                ti.i32,
                ndim=1,
            )
            for index in range(args.consumers)
        )
        reset = ti.algorithms.DeviceWorklistSequence(graph_args).prepare_next()
        launch_state = (
            worklist.next_extent.dispatch_state(block_dim)
            if bounded and compatibility_state
            else None
        )
        finalize = ti.algorithms.DeviceWorklistSequence(
            graph_args
        ).finalize_next(dispatch_state=launch_state)
        builder = ti.graph.GraphBuilder()
        builder.append_native(reset)
        builder.dispatch(produce, *graph_args.append_arguments(), count_arg)
        builder.append_native(finalize)
        handle = None
        if bounded:
            handles = tuple(
                builder.dispatch_bounded(
                    consume_masked,
                    graph_args.next_values,
                    graph_args.next_extent,
                    output_arg,
                    extent=graph_args.next_extent,
                    capacity=capacity,
                    block_dim=block_dim,
                    launch_state=launch_state,
                )
                for output_arg in output_args
            )
            handle = handles[0]
        else:
            for output_arg in output_args:
                builder.dispatch(
                    consume_masked,
                    graph_args.next_values,
                    graph_args.next_extent,
                    output_arg,
                )
        graph = builder.compile()
        outputs = tuple(
            ti.ndarray(ti.i32, shape=capacity)
            for _ in range(args.consumers)
        )
        runtime_args = worklist.runtime_arguments(name, include_capacity=True)
        runtime_args.update(count=produced)
        runtime_args.update(
            (f"output_{index}", output)
            for index, output in enumerate(outputs)
        )
        return worklist, graph, outputs, runtime_args, handle, launch_state

    fixed, fixed_build_us = _build_us(lambda: build_graph("fixed", False))
    bounded, bounded_build_us = _build_us(
        lambda: build_graph("bounded", True)
    )
    compatibility, compatibility_build_us = _build_us(
        lambda: build_graph(
            "compatibility", True, compatibility_state=True
        )
    )
    host_worklist = ti.algorithms.DeviceWorklist(capacity, ti.i32)
    host_outputs = tuple(
        ti.ndarray(ti.i32, shape=capacity)
        for _ in range(args.consumers)
    )

    def host_roundtrip():
        host_worklist.prepare_next()
        produce(*host_worklist.append_arguments(), produced)
        host_worklist.commit_next()
        count = host_worklist.extent.check()
        for output in host_outputs:
            consume_exact(host_worklist.values, output, count)

    def fixed_graph():
        fixed[1].run(fixed[3])

    def bounded_graph():
        bounded[1].run(bounded[3])

    def compatibility_state_graph():
        compatibility[1].run(compatibility[3])

    variants = {
        "host_roundtrip": host_roundtrip,
        "fixed_graph": fixed_graph,
        "bounded_graph": bounded_graph,
        "compatibility_state_graph": compatibility_state_graph,
    }
    first = {
        "host_roundtrip_us": _measure(host_roundtrip, 1),
        "fixed_graph_us": _measure(fixed_graph, 1),
        "bounded_graph_us": _measure(bounded_graph, 1),
        "compatibility_state_graph_us": _measure(
            compatibility_state_graph, 1
        ),
    }
    for call in variants.values():
        for _ in range(args.warmups):
            call()
    ti.sync()
    before = _memory_snapshot()
    samples, raw_samples = _paired(
        variants, args.rounds, args.samples, args.seed
    )
    after = _memory_snapshot()

    expected = np.arange(1, produced + 1, dtype=np.int32)
    for _ in range(args.payload_work):
        expected = expected * 7 + 3
    fixed_graph()
    bounded_graph()
    compatibility_state_graph()
    ti.sync()
    correct = (
        fixed[0].next_extent.snapshot().count == produced
        and bounded[0].next_extent.snapshot().count == produced
        and compatibility[0].next_extent.snapshot().count == produced
        and all(
            np.array_equal(
                np.sort(output.to_numpy()[:produced]), np.sort(expected)
            )
            for output in (*fixed[2], *bounded[2], *compatibility[2])
        )
    )
    # Re-snapshot after every first-use and host materialization above. Each
    # qualification window measures one replayable ownership route so
    # allocation growth cannot be hidden by a different variant's pool state.
    stability_variants = {
        "bounded_graph": bounded_graph,
        "compatibility_state_graph": compatibility_state_graph,
    }
    stability_memory = {}
    for name, call in stability_variants.items():
        call()
        ti.sync()
        stability_before = _memory_snapshot()
        for _ in range(args.stability_rounds):
            call()
        ti.sync()
        stability_after = _memory_snapshot()
        stability_memory[name] = _non_growth(stability_before, stability_after)
    bounded_us = samples["bounded_graph"]["median_us"]

    def update_stats(graph):
        segments = tuple(
            item
            for item in graph.execution_stats().segments
            if item.kind == "cgraph"
        )
        return {
            "paths": [segment.last_path for segment in segments],
            "driver_error": max(
                (segment.last_driver_error for segment in segments),
                default=0,
            ),
            "groups": sum(
                segment.bounded_update_groups for segment in segments
            ),
            "updater_dispatches": sum(
                segment.bounded_updater_dispatches for segment in segments
            ),
            "grouped_payloads": sum(
                segment.bounded_grouped_payloads for segment in segments
            ),
            "producer_fused_groups": sum(
                segment.bounded_producer_fused_groups for segment in segments
            ),
            "persistent_control_bytes": sum(
                segment.persistent_bounded_control_bytes
                for segment in segments
            ),
        }

    return {
        "capacity": capacity,
        "consumers": args.consumers,
        "produced": produced,
        "active_percent": args.active_percent,
        "ownership_routes": {
            "bounded_graph": "graph_owned",
            "compatibility_state_graph": "legacy_launch_state_adapter",
        },
        "build_us": {
            "fixed_graph": fixed_build_us,
            "bounded_graph": bounded_build_us,
            "compatibility_state_graph": compatibility_build_us,
        },
        "first": first,
        "warm": samples,
        "speedup_vs_host_roundtrip": (
            samples["host_roundtrip"]["median_us"] / bounded_us
        ),
        "speedup_vs_fixed_graph": (
            samples["fixed_graph"]["median_us"] / bounded_us
        ),
        "bounded_route": bounded[4].capabilities.execution_semantics,
        "bounded_preparation_dispatches": (
            bounded[4].capabilities.preparation_dispatches
        ),
        "compatibility_preparation_dispatches": (
            compatibility[4].capabilities.preparation_dispatches
        ),
        "compatibility_state_speedup": (
            bounded_us
            / samples["compatibility_state_graph"]["median_us"]
        ),
        "paired_speedups": {
            "bounded_over_fixed": _paired_speedup(
                raw_samples["fixed_graph"], raw_samples["bounded_graph"]
            ),
            "bounded_over_host_roundtrip": _paired_speedup(
                raw_samples["host_roundtrip"],
                raw_samples["bounded_graph"],
            ),
            "compatibility_state_over_graph_owned": _paired_speedup(
                raw_samples["bounded_graph"],
                raw_samples["compatibility_state_graph"],
            ),
        },
        "bounded_graph_stats": update_stats(bounded[1]),
        "compatibility_state_graph_stats": update_stats(compatibility[1]),
        "overflow_free_slot_reservation_atomics_per_item": 1,
        "correct": correct,
        "timed_window_memory_stable": _non_growth(before, after),
        "stability_replays": args.stability_rounds,
        "stability_memory": stability_memory,
        "memory_stable": all(stability_memory.values()),
        "worklist_memory": bounded[0].memory_report(),
    }


def _conflict_pipeline(args):
    capacity = args.conflict_capacity
    active = capacity * args.active_percent // 100
    active = max(1, active)
    key_count = args.conflict_key_count or max(1, active // 4)
    worklist = ti.algorithms.DeviceWorklist(capacity, ti.i32)
    values_host = np.arange(capacity, dtype=np.int32) * 11 + 7
    keys_host = (np.arange(capacity, dtype=np.int32) * 17 + 3) % key_count
    priorities_host = (np.arange(capacity, dtype=np.int32) * 13 + 5) % 97
    worklist.values.from_numpy(values_host)
    worklist.extent.set(active)
    keys = ti.ndarray(ti.i32, shape=capacity)
    priorities = ti.ndarray(ti.i32, shape=capacity)
    output_keys = ti.ndarray(ti.i32, shape=capacity)
    output_priorities = ti.ndarray(ti.i32, shape=capacity)
    output_ordinals = ti.ndarray(ti.i32, shape=capacity)
    keys.from_numpy(keys_host)
    priorities.from_numpy(priorities_host)
    graph_args = worklist.graph_args("claims")
    keys_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "keys", ti.i32, ndim=1)
    priorities_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "priorities", ti.i32, ndim=1
    )
    output_keys_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "output_keys", ti.i32, ndim=1
    )
    output_priorities_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "output_priorities", ti.i32, ndim=1
    )
    output_ordinals_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "output_ordinals", ti.i32, ndim=1
    )

    def build():
        sequence = ti.algorithms.DeviceWorklistSequence(graph_args)
        sequence.resolve_conflicts(
            keys_arg,
            output_keys_arg,
            output_priorities_arg,
            output_ordinals_arg,
            priorities=priorities_arg,
            policy="min_priority",
        )
        builder = ti.graph.GraphBuilder()
        builder.append_native(sequence)
        return builder.compile(), sequence

    built, build_us = _build_us(build)
    graph, sequence = built
    runtime_args = worklist.runtime_arguments("claims")
    runtime_args.update(
        keys=keys,
        priorities=priorities,
        output_keys=output_keys,
        output_priorities=output_priorities,
        output_ordinals=output_ordinals,
    )

    def device_claim():
        graph.run(runtime_args)

    def host_roundtrip():
        host_keys = keys.to_numpy()[:active]
        host_priorities = priorities.to_numpy()[:active]
        host_values = worklist.values.to_numpy()[:active]
        ordinals = np.arange(active, dtype=np.int32)
        order = np.lexsort((ordinals, host_priorities, host_keys))
        sorted_keys = host_keys[order]
        winners = np.empty(active, dtype=bool)
        winners[0] = True
        winners[1:] = sorted_keys[1:] != sorted_keys[:-1]
        # Materialize the same payload boundary as the device path.
        _ = host_values[order][winners]

    first = {
        "device_claim_us": _measure(device_claim, 1),
        "host_roundtrip_us": _measure(host_roundtrip, 1),
    }
    for call in (device_claim, host_roundtrip):
        for _ in range(args.warmups):
            call()
    ti.sync()
    before = _memory_snapshot()
    samples, _ = _paired(
        {"device_claim": device_claim, "host_roundtrip": host_roundtrip},
        args.conflict_rounds,
        args.samples,
        args.seed + 1,
    )
    after = _memory_snapshot()

    device_claim()
    ti.sync()
    ordinals = np.arange(active, dtype=np.int32)
    order = np.lexsort((ordinals, priorities_host[:active], keys_host[:active]))
    sorted_keys = keys_host[:active][order]
    winner_mask = np.empty(active, dtype=bool)
    winner_mask[0] = True
    winner_mask[1:] = sorted_keys[1:] != sorted_keys[:-1]
    winner_order = order[winner_mask]
    winner_count = int(winner_order.size)
    expected_keys = keys_host[:active][winner_order]
    expected_values = values_host[:active][winner_order]
    expected_ordinals = winner_order.astype(np.int32, copy=False)
    correct = (
        worklist.next_extent.snapshot().count == winner_count
        and np.array_equal(
            output_keys.to_numpy()[:winner_count],
            expected_keys,
        )
        and np.array_equal(
            worklist.next_values.to_numpy()[:winner_count],
            expected_values,
        )
        and np.array_equal(output_ordinals.to_numpy()[:winner_count], expected_ordinals)
    )
    deterministic = correct
    for _ in range(args.determinism_rounds):
        device_claim()
        ti.sync()
        deterministic = deterministic and np.array_equal(
            output_keys.to_numpy()[:winner_count], expected_keys
        )
        deterministic = deterministic and np.array_equal(
            worklist.next_values.to_numpy()[:winner_count], expected_values
        )
        deterministic = deterministic and np.array_equal(
            output_ordinals.to_numpy()[:winner_count], expected_ordinals
        )

    # Host observations in the determinism check can lazily create staging.
    # Measure allocation stability only after those one-time paths are warm.
    device_claim()
    ti.sync()
    stability_before = _memory_snapshot()
    for _ in range(args.stability_rounds):
        device_claim()
    ti.sync()
    stability_after = _memory_snapshot()
    device_us = samples["device_claim"]["median_us"]
    return {
        "capacity": capacity,
        "active": active,
        "key_count": key_count,
        "winner_count": winner_count,
        "build_us": build_us,
        "first": first,
        "warm": samples,
        "speedup_vs_host_roundtrip": (
            samples["host_roundtrip"]["median_us"] / device_us
        ),
        "correct": correct,
        "deterministic_replays": args.determinism_rounds,
        "deterministic": deterministic,
        "deterministic_order": "key_priority_ordinal",
        "timed_window_memory_stable": _non_growth(before, after),
        "stability_replays": args.stability_rounds,
        "memory_stable": _non_growth(stability_before, stability_after),
        "worklist_memory": worklist.memory_report(),
        "sequence_memory": sequence.memory_report(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=("cpu", "cuda", "vulkan"), required=True)
    parser.add_argument("--suite", choices=("all", "atomic", "conflict"), default="all")
    parser.add_argument("--capacity", type=int, default=1 << 18)
    parser.add_argument("--conflict-capacity", type=int, default=1 << 14)
    parser.add_argument("--conflict-key-count", type=int, default=0)
    parser.add_argument("--active-percent", type=int, default=10)
    parser.add_argument("--block-dim", type=int, default=128)
    parser.add_argument("--payload-work", type=int, default=4)
    parser.add_argument("--consumers", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--conflict-rounds", type=int, default=10)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--stability-rounds", type=int, default=1000)
    parser.add_argument("--determinism-rounds", type=int, default=32)
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
    if not 0 <= args.active_percent <= 100:
        parser.error("--active-percent must be in [0, 100]")
    if args.conflict_key_count < 0:
        parser.error("--conflict-key-count must be non-negative")
    if (
        min(
            args.capacity,
            args.conflict_capacity,
            args.block_dim,
            args.payload_work,
            args.consumers,
            args.rounds,
            args.conflict_rounds,
            args.samples,
            args.warmups,
            args.stability_rounds,
            args.determinism_rounds,
        )
        <= 0
    ):
        parser.error("capacity, work, rounds, samples, and warmups must be positive")

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
        "device_update": "adaptive_device_grid_update",
        "exact_scheduler": "exact_cpu_scheduler",
        "masked_capacity": "masked_capacity",
    }.get(requested_route)
    route_identity_valid = requested_route in ("auto", "not_applicable") or (
        bounded_capabilities["selected_route"] == expected_selected
    )
    if not route_identity_valid:
        raise RuntimeError(
            "worklist benchmark route mismatch: "
            f"requested={requested_route}, "
            f"selected={bounded_capabilities['selected_route']}"
        )
    result = {
        "schema_version": 2,
        "arch": args.arch,
        "requested_route": requested_route,
        "selected_route": bounded_capabilities["selected_route"],
        "route_identity_valid": route_identity_valid,
        "capabilities": ti.graph.dynamic_work_capabilities(),
    }
    if args.suite in ("all", "atomic"):
        result["atomic_worklist"] = _atomic_pipeline(args)
    if args.suite in ("all", "conflict"):
        result["deterministic_conflict"] = _conflict_pipeline(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
