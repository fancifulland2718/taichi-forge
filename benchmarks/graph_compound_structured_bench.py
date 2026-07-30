"""Measure multi-region structured Graph transaction overhead.

Run GPU cases only while the selected device is otherwise idle. ``submit``
reports the host enqueue call separately from completion wait and total
latency. Vulkan strategy selection can be fixed with ``--strategy`` to compare
compact tails against the automatic coarse-conditional tail planner.
"""

import argparse
import json
import os
import statistics
import time

import numpy as np

import taichi_forge as ti


def _arch(name):
    return {"cpu": ti.cpu, "cuda": ti.cuda, "vulkan": ti.vulkan}[name]


def _percentile(values, fraction):
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _summary(values):
    return {
        "median": statistics.median(values),
        "p10": _percentile(values, 0.1),
        "p90": _percentile(values, 0.9),
    }


def _build_graph(
    *,
    size,
    regions,
    budget,
    chunk_size,
    independent_actions,
    vulkan_first_chunk_strategy,
    lowering_mode,
):
    @ti.kernel
    def initialize_state(
        values: ti.types.ndarray(dtype=ti.f32, ndim=1),
        state: ti.types.ndarray(dtype=ti.i32, ndim=0),
    ):
        state[None] = 0
        for index in range(size):
            values[index] = 0.0

    @ti.kernel
    def initialize_control(
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
    ):
        predicate[None] = 0
        counter[None] = 0

    @ti.kernel
    def initialize_values(values: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for index in range(size):
            values[index] = 0.0

    @ti.kernel
    def evaluate_condition(
        state: ti.types.ndarray(dtype=ti.i32, ndim=0),
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        target: ti.i32,
    ):
        predicate[None] = int(state[None] < target)

    @ti.kernel
    def step(
        values: ti.types.ndarray(dtype=ti.f32, ndim=1),
        state: ti.types.ndarray(dtype=ti.i32, ndim=0),
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
    ):
        for index in range(size):
            if predicate[None] != 0:
                values[index] += 1.0
        if predicate[None] != 0:
            state[None] += 1
            counter[None] += 1

    @ti.kernel
    def independent_step(
        values: ti.types.ndarray(dtype=ti.f32, ndim=1),
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
    ):
        for index in range(size):
            if predicate[None] != 0:
                values[index] += 1.0

    scalar = lambda name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.i32, ndim=0)
    values = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "values", ti.f32, ndim=1)
    action_values = tuple(
        ti.graph.Arg(
            ti.graph.ArgKind.NDARRAY,
            f"action_values_{index}",
            ti.f32,
            ndim=1,
        )
        for index in range(independent_actions)
    )
    state = scalar("state")
    predicates = tuple(scalar(f"predicate_{index}") for index in range(regions))
    counters = tuple(scalar(f"counter_{index}") for index in range(regions))
    targets = tuple(
        ti.graph.Arg(ti.graph.ArgKind.SCALAR, f"target_{index}", ti.i32)
        for index in range(regions)
    )

    builder = ti.graph.GraphBuilder()
    builder.dispatch(initialize_state, values, state)
    for action_value in action_values:
        builder.dispatch(initialize_values, action_value)
    for predicate, counter in zip(predicates, counters):
        builder.dispatch(initialize_control, predicate, counter)
    for index, (predicate, counter, target) in enumerate(
        zip(predicates, counters, targets)
    ):
        condition = builder.create_sequential()
        condition.dispatch(evaluate_condition, state, predicate, target)
        body = builder.create_sequential()
        for action_value in action_values:
            body.dispatch(independent_step, action_value, predicate)
        body.dispatch(step, values, state, predicate, counter)
        builder.while_loop(
            condition,
            body,
            predicate=predicate,
            control_inputs=(state, target),
            carried_state=(values, state, *action_values),
            counter=counter,
            max_iterations=budget,
            chunk_size=chunk_size,
            vulkan_first_chunk_strategy=vulkan_first_chunk_strategy,
            lowering_mode=lowering_mode,
            name=f"stage_{index}",
        )
    builder.observe(state, *counters, name="terminal")
    return builder.compile()


def _arguments(
    *,
    size,
    regions,
    active_regions,
    logical_iterations,
    independent_actions,
):
    args = {
        "values": ti.ndarray(ti.f32, shape=size),
        "state": ti.ndarray(ti.i32, shape=()),
    }
    for index in range(regions):
        args[f"predicate_{index}"] = ti.ndarray(ti.i32, shape=())
        args[f"counter_{index}"] = ti.ndarray(ti.i32, shape=())
        args[f"target_{index}"] = min(index + 1, active_regions) * logical_iterations
    for index in range(independent_actions):
        args[f"action_values_{index}"] = ti.ndarray(ti.f32, shape=size)
    return args


def _measure(graph, args, *, mode, warmups, repeats, telemetry):
    # Enable detailed backend counters before recording the first replay.
    graph.execution_stats()
    for _ in range(warmups):
        if mode == "submit":
            ticket = graph.submit(args, telemetry=telemetry)
            ticket.telemetry() if telemetry else ticket.wait()
        else:
            graph.run(args)
    ti.sync()

    before = ti.runtime.stats()
    queue_before = (
        ti.lang.impl.get_runtime().prog._debug_vulkan_queue_submission_stats()
    )
    call_us = []
    total_us = []
    wait_us = []
    telemetry_us = []
    instrumented_total_us = []
    last_telemetry = None
    for _ in range(repeats):
        start = time.perf_counter_ns()
        if mode == "submit":
            ticket = graph.submit(args, telemetry=telemetry)
            submitted = time.perf_counter_ns()
            ticket.wait()
            completed = time.perf_counter_ns()
            if telemetry:
                last_telemetry = ticket.telemetry()
            finished = time.perf_counter_ns()
            call_us.append((submitted - start) / 1.0e3)
            wait_us.append((completed - submitted) / 1.0e3)
            total_us.append((completed - start) / 1.0e3)
            telemetry_us.append((finished - completed) / 1.0e3)
            instrumented_total_us.append((finished - start) / 1.0e3)
        else:
            graph.run(args)
            finished = time.perf_counter_ns()
            elapsed = (finished - start) / 1.0e3
            call_us.append(elapsed)
            wait_us.append(0.0)
            total_us.append(elapsed)
            telemetry_us.append(0.0)
            instrumented_total_us.append(elapsed)
    ti.sync()
    after = ti.runtime.stats()
    queue_after = ti.lang.impl.get_runtime().prog._debug_vulkan_queue_submission_stats()
    memory = graph.execution_stats().memory
    result = {
        "mode": mode,
        "host_call_us": _summary(call_us),
        "completion_wait_us": _summary(wait_us),
        "end_to_end_us": _summary(total_us),
        "telemetry_materialize_us": _summary(telemetry_us),
        "instrumented_end_to_end_us": _summary(instrumented_total_us),
        "last_telemetry": (
            None
            if last_telemetry is None
            else {
                "device_snapshot_bytes": last_telemetry.device_snapshot_bytes,
                "host_readback_bytes": last_telemetry.host_readback_bytes,
                "logical_iterations": [
                    region.logical_iterations for region in last_telemetry.regions
                ],
                "masked_iterations": [
                    region.masked_iterations for region in last_telemetry.regions
                ],
                "coarse_skipped_chunks": [
                    region.coarse_skipped_chunk_count
                    for region in last_telemetry.regions
                ],
                "queue_submit_calls": (
                    last_telemetry.queue.queue_submit_calls
                    if last_telemetry.queue.available
                    else None
                ),
                "gpu_timestamp_status": (last_telemetry.gpu_timestamp_status),
            }
        ),
        "graph_memory": {
            "persistent_bytes": memory.persistent_bytes,
            "persistent_argument_bytes": memory.persistent_argument_bytes,
            "persistent_observation_bytes": memory.persistent_observation_bytes,
            "persistent_telemetry_bytes": memory.persistent_telemetry_bytes,
            "opaque_driver_bytes": memory.opaque_driver_bytes,
        },
        "runtime_delta": {
            "graph_submissions": (
                after.submission.graph_submissions - before.submission.graph_submissions
            ),
            "graph_backend_submissions": (
                after.submission.graph_backend_submissions
                - before.submission.graph_backend_submissions
            ),
            "backend_waits": (
                None
                if after.synchronization.backend_waits is None
                else (
                    after.synchronization.backend_waits
                    - before.synchronization.backend_waits
                )
            ),
            "backend_wait_ns": (
                None
                if after.synchronization.backend_wait_ns is None
                else (
                    after.synchronization.backend_wait_ns
                    - before.synchronization.backend_wait_ns
                )
            ),
            "native_queue_submit_calls": (
                queue_after["queue_submit_calls"] - queue_before["queue_submit_calls"]
                if queue_after["supported"]
                else None
            ),
            "native_submitted_command_buffers": (
                queue_after["submitted_command_buffers"]
                - queue_before["submitted_command_buffers"]
                if queue_after["supported"]
                else None
            ),
            "native_batched_queue_submit_calls": (
                queue_after["batched_queue_submit_calls"]
                - queue_before["batched_queue_submit_calls"]
                if queue_after["supported"]
                else None
            ),
            "native_batched_command_buffers": (
                queue_after["batched_command_buffers"]
                - queue_before["batched_command_buffers"]
                if queue_after["supported"]
                else None
            ),
        },
    }
    structured_stats = [
        stats
        for stats in graph._graph_stats
        if isinstance(stats, dict) and "dependency_barriers" in stats
    ]
    result["structured_effect_counters"] = {
        name: sum(int(stats[name]) for stats in structured_stats)
        for name in (
            "effect_reads",
            "effect_writes",
            "dependency_barriers",
            "exit_barriers",
            "barrier_deferrals",
            "rar_elisions",
        )
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=("cpu", "cuda", "vulkan"), required=True)
    parser.add_argument("--mode", choices=("run", "submit"), required=True)
    parser.add_argument("--strategy", default="auto")
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--regions", type=int, default=16)
    parser.add_argument("--active-regions", type=int)
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument(
        "--independent-actions",
        type=int,
        default=0,
        help=(
            "Extra independent body dispatches used to qualify barrier "
            "planning"
        ),
    )
    parser.add_argument(
        "--first-chunk-strategy",
        choices=("auto", "compact", "coarse_conditional"),
        default="auto",
    )
    parser.add_argument("--logical-iterations", type=int, default=12)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--telemetry", action="store_true")
    parser.add_argument(
        "--compound-preparation",
        choices=("legacy", "single"),
        default="single",
        help=(
            "Vulkan qualification switch for per-chunk versus per-region "
            "preparation"
        ),
    )
    parser.add_argument(
        "--barrier-policy",
        choices=("eager", "effect_planned"),
        default="effect_planned",
        help="Vulkan qualification switch for structured replay barriers",
    )
    args = parser.parse_args()
    active_regions = (
        args.regions if args.active_regions is None else args.active_regions
    )
    if active_regions < 0 or active_regions > args.regions:
        parser.error("--active-regions must be between zero and --regions")
    if args.independent_actions < 0:
        parser.error("--independent-actions must be nonnegative")

    if args.strategy:
        os.environ["TI_GRAPH_VULKAN_STRUCTURED_STRATEGY"] = args.strategy
    os.environ["TI_VULKAN_COMPOUND_SINGLE_PREPARATION"] = (
        "1" if args.compound_preparation == "single" else "0"
    )
    os.environ["TI_VULKAN_STRUCTURED_HAZARD_PLANNER"] = (
        "1" if args.barrier_policy == "effect_planned" else "0"
    )
    lowering_mode = "portable" if args.arch == "cpu" else "native_required"
    ti.init(arch=_arch(args.arch), offline_cache=False)
    capabilities = ti.graph.structured_control_capabilities()
    if (
        args.mode == "submit"
        and not capabilities["device_control"]["compound_structured_submit"]
    ):
        raise RuntimeError(capabilities["device_control"]["structured_submit_reason"])

    build_start = time.perf_counter_ns()
    graph = _build_graph(
        size=args.size,
        regions=args.regions,
        budget=args.budget,
        chunk_size=args.chunk_size,
        independent_actions=args.independent_actions,
        vulkan_first_chunk_strategy=args.first_chunk_strategy,
        lowering_mode=lowering_mode,
    )
    build_ms = (time.perf_counter_ns() - build_start) / 1.0e6
    runtime_args = _arguments(
        size=args.size,
        regions=args.regions,
        active_regions=active_regions,
        logical_iterations=args.logical_iterations,
        independent_actions=args.independent_actions,
    )
    result = _measure(
        graph,
        runtime_args,
        mode=args.mode,
        warmups=args.warmups,
        repeats=args.repeats,
        telemetry=args.telemetry,
    )

    expected_state = active_regions * args.logical_iterations
    assert runtime_args["state"].to_numpy()[()] == expected_state
    np.testing.assert_array_equal(
        runtime_args["values"].to_numpy(),
        np.full(args.size, expected_state, dtype=np.float32),
    )
    for index in range(args.independent_actions):
        np.testing.assert_array_equal(
            runtime_args[f"action_values_{index}"].to_numpy(),
            np.full(args.size, expected_state, dtype=np.float32),
        )
    for index in range(args.regions):
        assert runtime_args[f"counter_{index}"].to_numpy()[()] == (
            args.logical_iterations if index < active_regions else 0
        )

    print(
        json.dumps(
            {
                "arch": args.arch,
                "strategy": args.strategy,
                "compound_preparation": args.compound_preparation,
                "barrier_policy": args.barrier_policy,
                "size": args.size,
                "regions": args.regions,
                "active_regions": active_regions,
                "budget": args.budget,
                "chunk_size": args.chunk_size,
                "independent_actions": args.independent_actions,
                "first_chunk_strategy": args.first_chunk_strategy,
                "logical_iterations": args.logical_iterations,
                "warmups": args.warmups,
                "repeats": args.repeats,
                "telemetry": args.telemetry,
                "build_ms": build_ms,
                "capabilities": capabilities["device_control"],
                "result": result,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
