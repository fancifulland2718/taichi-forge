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
    result = {}
    for name, values in samples.items():
        mean = statistics.fmean(values)
        result[name] = {
            "median_us": statistics.median(values),
            "p95_us": float(np.percentile(values, 95)),
            "mean_us": mean,
            "cv": statistics.pstdev(values) / mean if mean else 0.0,
            "min_us": min(values),
            "max_us": max(values),
        }
    return result


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


def _positive_byte_growth(before, after, *keys):
    values = tuple((before.get(key), after.get(key)) for key in keys)
    if any(old is None or new is None for old, new in values):
        return None
    return sum(max(0, new - old) for old, new in values)


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
    graph_node_count = args.graph_node_count
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
    for _ in range(graph_node_count):
        fixed_builder.dispatch(payload, extent_arg, input_arg, output_arg)
    fixed_graph = fixed_builder.compile()

    bounded_builder = ti.graph.GraphBuilder()
    bounded_builder.dispatch(publish, requested_arg, extent_arg)
    handles = tuple(
        bounded_builder.dispatch_bounded(
            payload,
            extent_arg,
            input_arg,
            output_arg,
            extent=extent_arg,
            capacity=capacity,
            block_dim=block_dim,
        )
        for _ in range(graph_node_count)
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

        def direct():
            publish(count, extent.state)
            for _ in range(graph_node_count):
                payload(extent.state, input_values, outputs["direct"])

        return {
            "direct": direct,
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

    stress = None
    if args.stress_replays:
        stress_args = {
            "requested": 0,
            "extent": extent,
            "input": input_values,
            "output": outputs["bounded_graph"],
        }
        stress_counts = (
            0,
            capacity,
            1,
            max(0, capacity - 1),
        )
        ti.sync()
        stress_before = program._runtime_statistics_snapshot()["memory"]
        stress_host_before = dict(ti_core.get_host_memory_pool_stats())
        stress_device_before = dict(ti_core.get_device_memory_pool_stats())
        stress_start = time.perf_counter_ns()
        for replay in range(args.stress_replays):
            stress_args["requested"] = stress_counts[replay % len(stress_counts)]
            bounded_graph.run(stress_args)
            # Vulkan Graph replay owns a bounded in-flight slot set. Keep the
            # stress test within that documented ownership contract while
            # still exercising long-lived slot reuse and memory stability.
            if args.arch == "vulkan" and (replay + 1) % 8 == 0:
                ti.sync()
        ti.sync()
        stress_elapsed = time.perf_counter_ns() - stress_start
        stress_after = program._runtime_statistics_snapshot()["memory"]
        stress_host_after = dict(ti_core.get_host_memory_pool_stats())
        stress_device_after = dict(ti_core.get_device_memory_pool_stats())
        final_count = stress_counts[(args.stress_replays - 1) % len(stress_counts)]
        final_snapshot = handles[0].snapshot(extent)
        stress = {
            "replays": args.stress_replays,
            "mean_us": stress_elapsed / args.stress_replays / 1.0e3,
            "final_count": final_snapshot.useful_count,
            "expected_final_count": final_count,
            "runtime_memory_non_growth": _memory_non_growth(
                stress_before, stress_after
            ),
            "host_pool_stable": _pool_ownership_stable(
                stress_host_before, stress_host_after
            ),
            "device_pool_stable": _pool_ownership_stable(
                stress_device_before, stress_device_after
            ),
        }

    node_memory = None
    if args.memory_node_count:
        node_before = program._runtime_statistics_snapshot()["memory"]
        node_host_before = dict(ti_core.get_host_memory_pool_stats())
        node_device_before = dict(ti_core.get_device_memory_pool_stats())
        node_builder = ti.graph.GraphBuilder()
        node_builder.dispatch(publish, requested_arg, extent_arg)
        node_handles = []
        for node in range(args.memory_node_count):
            node_handles.append(
                node_builder.dispatch_bounded(
                    payload,
                    extent_arg,
                    input_arg,
                    output_arg,
                    extent=extent_arg,
                    capacity=capacity,
                    block_dim=block_dim,
                )
            )
        node_graph = node_builder.compile()
        node_args = {
            "requested": 0,
            "extent": extent,
            "input": input_values,
            "output": outputs["bounded_graph"],
        }
        # Opt in before execution so the report can distinguish capture from
        # replay, then sample only after both paths have completed.
        node_graph.execution_stats()
        node_graph.run(node_args)
        node_graph.run(node_args)
        ti.sync()
        node_report = node_graph.execution_stats()
        node_after = program._runtime_statistics_snapshot()["memory"]
        node_host_after = dict(ti_core.get_host_memory_pool_stats())
        node_device_after = dict(ti_core.get_device_memory_pool_stats())
        node_memory = {
            "nodes": args.memory_node_count,
            "persistent_argument_bytes": (
                node_report.memory.persistent_argument_bytes
            ),
            "persistent_bounded_control_bytes": (
                node_report.memory.persistent_bounded_control_bytes
            ),
            "execution_path": node_report.execution_path,
            "fallback_reason": node_report.fallback_reason,
            "segment_argument_bytes": [
                segment.persistent_argument_bytes
                for segment in node_report.segments
            ],
            "segment_bounded_control_bytes": [
                segment.persistent_bounded_control_bytes
                for segment in node_report.segments
            ],
            "workspace_bytes": sum(item.workspace_bytes for item in node_handles),
            "runtime_live_byte_growth": _positive_byte_growth(
                node_before,
                node_after,
                "host_requested_live_bytes",
                "device_requested_live_bytes",
            ),
            "host_pool_byte_growth": _positive_byte_growth(
                node_host_before,
                node_host_after,
                "requested_live_bytes",
            ),
            "device_pool_byte_growth": _positive_byte_growth(
                node_device_before,
                node_device_after,
                "requested_live_bytes",
            ),
        }
    after = program._runtime_statistics_snapshot()["memory"]
    host_pool_stable = _pool_ownership_stable(
        host_pool_before, dict(ti_core.get_host_memory_pool_stats())
    )
    device_pool_stable = _pool_ownership_stable(
        device_pool_before, dict(ti_core.get_device_memory_pool_stats())
    )

    correct = True
    physical_counts = {}
    for name, count in counts.items():
        for output in outputs.values():
            output.fill(-1)
        variants = variants_for(count)
        for call in variants.values():
            call()
        ti.sync()
        snapshot = handles[0].snapshot(extent)
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
        physical_counts[name] = {
            "useful_count": snapshot.useful_count,
            "executed_count": snapshot.executed_count,
            "encoded_lanes": snapshot.encoded_lanes,
            "skipped_count": snapshot.skipped_count,
        }
    return {
        "capacity": capacity,
        "block_dim": block_dim,
        "graph_node_count": graph_node_count,
        "payload_work": args.payload_work,
        "capabilities": handles[0].capabilities.__dict__,
        "cases": reports,
        "physical_counts": physical_counts,
        "stress": stress,
        "node_memory": node_memory,
        "workspace_bytes": sum(handle.workspace_bytes for handle in handles),
        "workspace_allocations": sum(
            handle.workspace_allocation_count for handle in handles
        ),
        "correct": correct,
        "runtime_memory_non_growth": _memory_non_growth(before, after),
        "host_pool_stable": host_pool_stable,
        "device_pool_stable": device_pool_stable,
    }


def _cuda_route_comparison(args):
    """Compare CUDA bounded routes in one runtime to limit process noise."""

    capacity = args.graph_capacity
    block_dim = args.block_dim
    graph_node_count = args.graph_node_count
    extent = ti.DeviceExtent(capacity)
    input_values = ti.ndarray(ti.f32, shape=capacity)
    input_values.from_numpy(
        (np.arange(capacity, dtype=np.float32) % 251) * np.float32(0.001)
        + np.float32(0.25)
    )

    route_modes = ["masked_capacity", "auto"]
    probe = dict(ti_core.cuda_bounded_dispatch_probe())
    if probe["exact_device_grid_available"]:
        route_modes.append("device_update")
    variant_names = {
        "masked_capacity": "bounded_masked",
        "auto": "bounded_exact",
        "device_update": "bounded_adaptive",
    }
    outputs = {
        name: ti.ndarray(ti.f32, shape=capacity)
        for name in ("direct", "fixed_graph")
        + tuple(variant_names[mode] for mode in route_modes)
    }

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

    requested_arg = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "requested", ti.i32)
    extent_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "extent", ti.i32, ndim=1)
    input_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "input", ti.f32, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1)

    fixed_builder = ti.graph.GraphBuilder()
    fixed_builder.dispatch(publish, requested_arg, extent_arg)
    for _ in range(graph_node_count):
        fixed_builder.dispatch(payload, extent_arg, input_arg, output_arg)
    fixed_graph = fixed_builder.compile()

    route_graphs = {}
    route_handles = {}
    route_capabilities = {}
    for mode in route_modes:
        os.environ["TI_CUDA_BOUNDED_DISPATCH_MODE"] = mode
        builder = ti.graph.GraphBuilder()
        builder.dispatch(publish, requested_arg, extent_arg)
        handles = tuple(
            builder.dispatch_bounded(
                payload,
                extent_arg,
                input_arg,
                output_arg,
                extent=extent_arg,
                capacity=capacity,
                block_dim=block_dim,
            )
            for _ in range(graph_node_count)
        )
        route_graphs[mode] = builder.compile()
        route_handles[mode] = handles
        route_capabilities[mode] = handles[0].capabilities.__dict__

    def variants_for(count):
        fixed_args = {
            "requested": count,
            "extent": extent.state,
            "input": input_values,
            "output": outputs["fixed_graph"],
        }

        def direct():
            publish(count, extent.state)
            for _ in range(graph_node_count):
                payload(extent.state, input_values, outputs["direct"])

        variants = {
            "direct": direct,
            "fixed_graph": lambda: fixed_graph.run(fixed_args),
        }
        for mode in route_modes:
            name = variant_names[mode]
            runtime_args = {
                "requested": count,
                "extent": extent,
                "input": input_values,
                "output": outputs[name],
            }
            variants[name] = (
                lambda graph=route_graphs[mode], bound=runtime_args: graph.run(bound)
            )
        return variants

    counts = {
        "zero": 0,
        "sparse": capacity * args.sparse_percent // 100,
        "full": capacity,
    }
    for graph in route_graphs.values():
        graph.execution_stats()
    for count in counts.values():
        for call in variants_for(count).values():
            for _ in range(args.warmups):
                call()
    ti.sync()

    program = impl.get_runtime().prog
    before = program._runtime_statistics_snapshot()["memory"]
    host_pool_before = dict(ti_core.get_host_memory_pool_stats())
    device_pool_before = dict(ti_core.get_device_memory_pool_stats())
    cases = {}
    for index, (case, count) in enumerate(counts.items()):
        samples = _paired_samples(
            variants_for(count),
            args.graph_rounds,
            args.samples,
            args.seed + index,
        )
        summary = _summarize(samples)
        fixed_us = summary["fixed_graph"]["median_us"]
        masked_us = summary["bounded_masked"]["median_us"]
        comparisons = {}
        for mode in route_modes:
            name = variant_names[mode]
            route_us = summary[name]["median_us"]
            comparisons[name] = {
                "speedup_over_fixed": fixed_us / route_us,
                "speedup_over_masked": masked_us / route_us,
                "speedup_over_direct": summary["direct"]["median_us"] / route_us,
            }
        cases[case] = {
            "count": count,
            "samples": summary,
            "comparisons": comparisons,
        }

    correct = True
    physical_counts = {}
    for case, count in counts.items():
        for output in outputs.values():
            output.fill(-1)
        for call in variants_for(count).values():
            call()
        ti.sync()
        host_outputs = {key: value.to_numpy() for key, value in outputs.items()}
        reference = host_outputs["direct"]
        for output in host_outputs.values():
            correct = correct and np.allclose(output[:count], reference[:count])
            if count < capacity:
                correct = correct and float(output[count]) == -1.0
        physical_counts[case] = {}
        for mode in route_modes:
            snapshot = route_handles[mode][0].snapshot(extent)
            physical_counts[case][variant_names[mode]] = {
                "useful_count": snapshot.useful_count,
                "executed_count": snapshot.executed_count,
                "encoded_lanes": snapshot.encoded_lanes,
                "skipped_count": snapshot.skipped_count,
            }

    stress = {}
    if args.stress_replays:
        stress_counts = (0, capacity, 1, max(0, capacity - 1))
        for mode in route_modes:
            name = variant_names[mode]
            runtime_args = {
                "requested": 0,
                "extent": extent,
                "input": input_values,
                "output": outputs[name],
            }
            ti.sync()
            memory_before = program._runtime_statistics_snapshot()["memory"]
            start = time.perf_counter_ns()
            for replay in range(args.stress_replays):
                runtime_args["requested"] = stress_counts[replay % len(stress_counts)]
                route_graphs[mode].run(runtime_args)
            ti.sync()
            memory_after = program._runtime_statistics_snapshot()["memory"]
            stress[name] = {
                "replays": args.stress_replays,
                "mean_us": (time.perf_counter_ns() - start)
                / args.stress_replays
                / 1.0e3,
                "runtime_memory_non_growth": _memory_non_growth(
                    memory_before, memory_after
                ),
            }

    memory = {}
    for mode in route_modes:
        report = route_graphs[mode].execution_stats()
        memory[variant_names[mode]] = {
            "persistent_argument_bytes": report.memory.persistent_argument_bytes,
            "persistent_bounded_control_bytes": (
                report.memory.persistent_bounded_control_bytes
            ),
            "workspace_bytes": sum(
                handle.workspace_bytes for handle in route_handles[mode]
            ),
        }
    after = program._runtime_statistics_snapshot()["memory"]
    return {
        "capacity": capacity,
        "block_dim": block_dim,
        "graph_node_count": graph_node_count,
        "payload_work": args.payload_work,
        "routes": route_capabilities,
        "cases": cases,
        "physical_counts": physical_counts,
        "stress": stress,
        "memory": memory,
        "correct": correct,
        "runtime_memory_non_growth": _memory_non_growth(before, after),
        "host_pool_stable": _pool_ownership_stable(
            host_pool_before, dict(ti_core.get_host_memory_pool_stats())
        ),
        "device_pool_stable": _pool_ownership_stable(
            device_pool_before, dict(ti_core.get_device_memory_pool_stats())
        ),
    }


def _dynamic_count_sequences(capacity, block_dim, sparse_percent, length, seed):
    sparse = capacity * sparse_percent // 100
    sparse = min(capacity, max(0, sparse))
    second_sparse = min(capacity, max(sparse + block_dim, capacity // 2))
    boundary_values = tuple(
        min(capacity, max(0, value))
        for value in (
            0,
            1,
            block_dim - 1,
            block_dim,
            block_dim + 1,
            2 * block_dim - 1,
            2 * block_dim,
            sparse,
            capacity,
        )
    )
    rng = random.Random(seed)
    random_values = tuple(rng.randrange(capacity + 1) for _ in range(length))

    def repeat(pattern):
        return tuple(pattern[index % len(pattern)] for index in range(length))

    burst_pattern = tuple(
        value for value in (0, sparse, capacity, second_sparse) for _ in range(8)
    )
    return {
        "constant_zero": repeat((0,)),
        "constant_sparse": repeat((sparse,)),
        "constant_full": repeat((capacity,)),
        "alternating_zero_full": repeat((0, capacity)),
        "alternating_sparse_full": repeat((sparse, capacity)),
        "bursty": repeat(burst_pattern),
        "sawtooth": repeat(boundary_values),
        "random": random_values,
    }


def _planned_state_transitions(sequence, capacity, block_dim):
    # Each timed batch is primed with the last value, so the first replay has
    # the same predecessor as a continuously repeated sequence.
    def state(count):
        clamped = min(capacity, max(0, count))
        return (clamped != 0, (clamped + block_dim - 1) // block_dim)

    previous = state(sequence[-1])
    changes = 0
    for count in sequence:
        current = state(count)
        if current != previous:
            changes += 1
        previous = current
    return {
        "replays": len(sequence),
        "state_changes": changes,
        "cache_hits": len(sequence) - changes,
        "state_change_rate": changes / len(sequence),
    }


def _paired_dynamic_samples(variants, sequence, samples, seed):
    names = tuple(variants)
    orders = [list(names) for _ in range(samples)]
    rng = random.Random(seed)
    for order in orders:
        rng.shuffle(order)
    result = {name: [] for name in names}
    for order in orders:
        for name in order:
            invoke = variants[name]
            start = time.perf_counter_ns()
            for count in sequence:
                invoke(count)
            ti.sync()
            result[name].append(
                (time.perf_counter_ns() - start) / len(sequence) / 1.0e3
            )
    return result


def _cuda_update_policy_comparison(args):
    """Compare CUDA bounded updater policies inside one initialized runtime."""

    probe = dict(ti_core.cuda_bounded_dispatch_probe())
    if not probe["exact_device_grid_available"]:
        raise RuntimeError(
            "CUDA updater-policy comparison requires device_update: "
            + probe["unavailable_reason"]
        )

    capacity = args.graph_capacity
    block_dim = args.block_dim
    graph_node_count = args.graph_node_count
    extent = ti.DeviceExtent(capacity)
    input_values = ti.ndarray(ti.f32, shape=capacity)
    input_values.from_numpy(
        (np.arange(capacity, dtype=np.float32) % 251) * np.float32(0.001)
        + np.float32(0.25)
    )
    policy_variants = [
        ("grouped_stateful", "grouped_stateful"),
        ("per_node", "per_node"),
    ]
    if args.cuda_updater_telemetry == "on":
        policy_variants.append(("grouped_telemetry", "grouped_stateful"))
    outputs = {
        name: ti.ndarray(ti.f32, shape=capacity)
        for name in ("fixed_graph",) + tuple(name for name, _ in policy_variants)
    }

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

    requested_arg = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "requested", ti.i32)
    extent_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "extent", ti.i32, ndim=1)
    input_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "input", ti.f32, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1)

    fixed_builder = ti.graph.GraphBuilder()
    fixed_builder.dispatch(publish, requested_arg, extent_arg)
    for _ in range(graph_node_count):
        fixed_builder.dispatch(payload, extent_arg, input_arg, output_arg)
    fixed_graph = fixed_builder.compile()

    graphs = {}
    handles = {}
    for name, policy in policy_variants:
        os.environ["TI_GRAPH_CUDA_BOUNDED_UPDATE_POLICY"] = policy
        builder = ti.graph.GraphBuilder()
        builder.dispatch(publish, requested_arg, extent_arg)
        handles[name] = tuple(
            builder.dispatch_bounded(
                payload,
                extent_arg,
                input_arg,
                output_arg,
                extent=extent_arg,
                capacity=capacity,
                block_dim=block_dim,
            )
            for _ in range(graph_node_count)
        )
        graphs[name] = builder.compile()

    fixed_args = {
        "requested": 0,
        "extent": extent.state,
        "input": input_values,
        "output": outputs["fixed_graph"],
    }
    policy_args = {
        name: {
            "requested": 0,
            "extent": extent,
            "input": input_values,
            "output": outputs[name],
        }
        for name in graphs
    }

    def invoke_fixed(count):
        fixed_args["requested"] = count
        fixed_graph.run(fixed_args)

    def invoke_policy(name, count):
        policy_args[name]["requested"] = count
        graphs[name].run(policy_args[name])

    variants = {
        "fixed_graph": invoke_fixed,
        "grouped_stateful": lambda count: invoke_policy(
            "grouped_stateful", count
        ),
        "per_node": lambda count: invoke_policy("per_node", count),
    }
    if args.cuda_updater_telemetry == "on":
        variants["grouped_telemetry"] = lambda count: invoke_policy(
            "grouped_telemetry", count
        )
        graphs["grouped_telemetry"].execution_stats()
    sequences = _dynamic_count_sequences(
        capacity,
        block_dim,
        args.sparse_percent,
        args.dynamic_sequence_rounds,
        args.seed,
    )
    if args.dynamic_sequence != "all":
        sequences = {args.dynamic_sequence: sequences[args.dynamic_sequence]}
    for sequence in sequences.values():
        for invoke in variants.values():
            for count in sequence[: args.warmups]:
                invoke(count)
    ti.sync()

    program = impl.get_runtime().prog
    before = program._runtime_statistics_snapshot()["memory"]
    host_pool_before = dict(ti_core.get_host_memory_pool_stats())
    device_pool_before = dict(ti_core.get_device_memory_pool_stats())
    cases = {}
    for index, (name, sequence) in enumerate(sequences.items()):
        for invoke in variants.values():
            invoke(sequence[-1])
        ti.sync()
        telemetry_before = (
            graphs["grouped_telemetry"].execution_stats().segments[0]
            if args.cuda_updater_telemetry == "on"
            else None
        )
        samples = _paired_dynamic_samples(
            variants, sequence, args.samples, args.seed + index
        )
        telemetry_after = (
            graphs["grouped_telemetry"].execution_stats().segments[0]
            if args.cuda_updater_telemetry == "on"
            else None
        )
        summary = _summarize(samples)
        grouped_us = summary["grouped_stateful"]["median_us"]
        per_node_us = summary["per_node"]["median_us"]
        fixed_us = summary["fixed_graph"]["median_us"]
        case_result = {
            "planned": _planned_state_transitions(
                sequence, capacity, block_dim
            ),
            "first_counts": list(sequence[: min(16, len(sequence))]),
            "samples": summary,
            "grouped_speedup_over_per_node": per_node_us / grouped_us,
            "grouped_speedup_over_fixed": fixed_us / grouped_us,
            "per_node_speedup_over_fixed": fixed_us / per_node_us,
            "telemetry": (
                {
                    field: getattr(telemetry_after, field)
                    - getattr(telemetry_before, field)
                    for field in (
                        "bounded_update_replays",
                        "bounded_update_state_changes",
                        "bounded_update_cache_hits",
                        "bounded_node_api_calls",
                    )
                }
                if telemetry_before is not None
                else None
            ),
        }
        if args.cuda_updater_telemetry == "on":
            telemetry_us = summary["grouped_telemetry"]["median_us"]
            case_result["telemetry_overhead_ratio"] = (
                telemetry_us / grouped_us
            )
        cases[name] = case_result

    correct = True
    final_counts = {}
    for name, sequence in sequences.items():
        final_count = sequence[-1]
        for output in outputs.values():
            output.fill(-1)
        for invoke in variants.values():
            invoke(final_count)
        ti.sync()
        observed = {key: value.to_numpy() for key, value in outputs.items()}
        reference = observed["fixed_graph"]
        for output in observed.values():
            correct = correct and np.allclose(
                output[:final_count], reference[:final_count]
            )
            if final_count < capacity:
                correct = correct and float(output[final_count]) == -1.0
        final_counts[name] = {
            name: handles[name][0].snapshot(extent).useful_count
            for name in graphs
        }

    reports = {}
    for name, graph in graphs.items():
        report = graph.execution_stats()
        segment = report.segments[0]
        reports[name] = {
            "execution_path": segment.last_path,
            "fallback_reason": segment.fallback_reason,
            "bounded_update_groups": segment.bounded_update_groups,
            "bounded_updater_dispatches": segment.bounded_updater_dispatches,
            "bounded_grouped_payloads": segment.bounded_grouped_payloads,
            "bounded_max_group_size": segment.bounded_max_group_size,
            "bounded_update_replays": segment.bounded_update_replays,
            "bounded_update_state_changes": (
                segment.bounded_update_state_changes
            ),
            "bounded_update_cache_hits": segment.bounded_update_cache_hits,
            "bounded_node_api_calls": segment.bounded_node_api_calls,
            "persistent_argument_bytes": report.memory.persistent_argument_bytes,
            "persistent_bounded_control_bytes": (
                report.memory.persistent_bounded_control_bytes
            ),
        }
    after = program._runtime_statistics_snapshot()["memory"]
    return {
        "capacity": capacity,
        "block_dim": block_dim,
        "graph_node_count": graph_node_count,
        "payload_work": args.payload_work,
        "sequence_rounds": args.dynamic_sequence_rounds,
        "cases": cases,
        "reports": reports,
        "final_counts": final_counts,
        "correct": correct,
        "runtime_memory_non_growth": _memory_non_growth(before, after),
        "host_pool_stable": _pool_ownership_stable(
            host_pool_before, dict(ti_core.get_host_memory_pool_stats())
        ),
        "device_pool_stable": _pool_ownership_stable(
            device_pool_before, dict(ti_core.get_device_memory_pool_stats())
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=("cpu", "cuda", "vulkan"), required=True)
    parser.add_argument("--suite", choices=("all", "prefix", "graph"), default="all")
    parser.add_argument("--prefix-capacity", type=int, default=1 << 18)
    parser.add_argument("--graph-capacity", type=int, default=1 << 20)
    parser.add_argument("--graph-node-count", type=int, default=1)
    parser.add_argument("--sparse-percent", type=int, default=10)
    parser.add_argument("--block-dim", type=int, default=128)
    parser.add_argument("--payload-work", type=int, default=16)
    parser.add_argument("--prefix-rounds", type=int, default=20)
    parser.add_argument("--graph-rounds", type=int, default=100)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--stress-replays", type=int, default=0)
    parser.add_argument("--memory-node-count", type=int, default=0)
    parser.add_argument("--dynamic-sequence-rounds", type=int, default=200)
    parser.add_argument(
        "--dynamic-sequence",
        choices=(
            "all",
            "constant_zero",
            "constant_sparse",
            "constant_full",
            "alternating_zero_full",
            "alternating_sparse_full",
            "bursty",
            "sawtooth",
            "random",
        ),
        default="all",
    )
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument(
        "--cuda-route",
        choices=("auto", "device_update", "masked_capacity", "compare"),
        default="auto",
    )
    parser.add_argument(
        "--cpu-route",
        choices=("auto", "exact_scheduler", "masked_capacity"),
        default="auto",
    )
    parser.add_argument(
        "--vulkan-packet-policy",
        choices=("auto", "reuse_consecutive", "per_consumer"),
        default="auto",
    )
    parser.add_argument(
        "--cuda-update-policy",
        choices=("auto", "grouped_stateful", "per_node", "compare"),
        default="auto",
    )
    parser.add_argument(
        "--cuda-updater-telemetry",
        choices=("off", "on"),
        default="off",
    )
    args = parser.parse_args()
    if not 0 <= args.sparse_percent <= 100:
        parser.error("--sparse-percent must be in [0, 100]")
    if (
        min(
            args.prefix_capacity,
            args.graph_capacity,
            args.graph_node_count,
            args.block_dim,
            args.payload_work,
            args.prefix_rounds,
            args.graph_rounds,
            args.samples,
            args.dynamic_sequence_rounds,
        )
        <= 0
    ):
        parser.error("capacities, block size, rounds, and samples must be positive")
    if args.stress_replays < 0 or args.memory_node_count < 0:
        parser.error("stress replay and memory node counts must be non-negative")

    if args.arch == "cuda":
        os.environ["TI_CUDA_BOUNDED_DISPATCH_MODE"] = (
            "device_update"
            if args.cuda_update_policy == "compare"
            else "auto" if args.cuda_route == "compare" else args.cuda_route
        )
        if args.cuda_update_policy != "compare":
            os.environ["TI_GRAPH_CUDA_BOUNDED_UPDATE_POLICY"] = (
                args.cuda_update_policy
            )
    elif args.arch == "cpu":
        os.environ["TI_CPU_BOUNDED_DISPATCH_MODE"] = args.cpu_route
    elif args.arch == "vulkan":
        os.environ["TI_GRAPH_VULKAN_BOUNDED_PACKET_POLICY"] = args.vulkan_packet_policy
    arch = {"cpu": ti.cpu, "cuda": ti.cuda, "vulkan": ti.vulkan}[args.arch]
    ti.init(arch=arch, offline_cache=False)
    requested_route = (
        "device_update"
        if args.arch == "cuda" and args.cuda_update_policy == "compare"
        else args.cuda_route
        if args.arch == "cuda"
        else args.cpu_route if args.arch == "cpu" else "not_applicable"
    )
    bounded_capabilities = ti.graph.bounded_dispatch_capabilities()
    expected_selected = {
        "device_update": "adaptive_device_grid_update",
        "exact_scheduler": "exact_cpu_scheduler",
        "masked_capacity": "masked_capacity",
    }.get(requested_route)
    route_identity_valid = requested_route in (
        "auto",
        "compare",
        "not_applicable",
    ) or (bounded_capabilities["selected_route"] == expected_selected)
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
        "vulkan_packet_policy": (
            args.vulkan_packet_policy if args.arch == "vulkan" else "not_applicable"
        ),
        "cuda_update_policy": (
            args.cuda_update_policy if args.arch == "cuda" else "not_applicable"
        ),
        "cuda_updater_telemetry": (
            args.cuda_updater_telemetry
            if args.arch == "cuda"
            else "not_applicable"
        ),
    }
    if args.suite in ("all", "prefix"):
        result["device_prefix"] = _prefix_benchmark(args)
    if args.suite in ("all", "graph"):
        result["bounded_graph"] = (
            _cuda_update_policy_comparison(args)
            if args.arch == "cuda" and args.cuda_update_policy == "compare"
            else (
                _cuda_route_comparison(args)
                if args.arch == "cuda" and args.cuda_route == "compare"
                else _graph_benchmark(args)
            )
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
