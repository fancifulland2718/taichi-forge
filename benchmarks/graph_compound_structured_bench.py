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


def _build_graph(*, size, regions, budget, lowering_mode):
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

    scalar = lambda name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.i32, ndim=0)
    values = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "values", ti.f32, ndim=1)
    state = scalar("state")
    predicates = tuple(scalar(f"predicate_{index}") for index in range(regions))
    counters = tuple(scalar(f"counter_{index}") for index in range(regions))
    targets = tuple(
        ti.graph.Arg(ti.graph.ArgKind.SCALAR, f"target_{index}", ti.i32)
        for index in range(regions)
    )

    builder = ti.graph.GraphBuilder()
    builder.dispatch(initialize_state, values, state)
    for predicate, counter in zip(predicates, counters):
        builder.dispatch(initialize_control, predicate, counter)
    for index, (predicate, counter, target) in enumerate(
        zip(predicates, counters, targets)
    ):
        condition = builder.create_sequential()
        condition.dispatch(evaluate_condition, state, predicate, target)
        body = builder.create_sequential()
        body.dispatch(step, values, state, predicate, counter)
        builder.while_loop(
            condition,
            body,
            predicate=predicate,
            control_inputs=(state, target),
            carried_state=(values, state),
            counter=counter,
            max_iterations=budget,
            chunk_size=64,
            lowering_mode=lowering_mode,
            name=f"stage_{index}",
        )
    builder.observe(state, *counters, name="terminal")
    return builder.compile()


def _arguments(*, size, regions, logical_iterations):
    args = {
        "values": ti.ndarray(ti.f32, shape=size),
        "state": ti.ndarray(ti.i32, shape=()),
    }
    for index in range(regions):
        args[f"predicate_{index}"] = ti.ndarray(ti.i32, shape=())
        args[f"counter_{index}"] = ti.ndarray(ti.i32, shape=())
        args[f"target_{index}"] = (index + 1) * logical_iterations
    return args


def _measure(graph, args, *, mode, warmups, repeats):
    for _ in range(warmups):
        if mode == "submit":
            graph.submit(args).wait()
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
    for _ in range(repeats):
        start = time.perf_counter_ns()
        if mode == "submit":
            ticket = graph.submit(args)
            submitted = time.perf_counter_ns()
            ticket.wait()
            finished = time.perf_counter_ns()
            call_us.append((submitted - start) / 1.0e3)
            wait_us.append((finished - submitted) / 1.0e3)
            total_us.append((finished - start) / 1.0e3)
        else:
            graph.run(args)
            finished = time.perf_counter_ns()
            elapsed = (finished - start) / 1.0e3
            call_us.append(elapsed)
            wait_us.append(0.0)
            total_us.append(elapsed)
    ti.sync()
    after = ti.runtime.stats()
    queue_after = ti.lang.impl.get_runtime().prog._debug_vulkan_queue_submission_stats()
    return {
        "mode": mode,
        "host_call_us": _summary(call_us),
        "completion_wait_us": _summary(wait_us),
        "end_to_end_us": _summary(total_us),
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=("cpu", "cuda", "vulkan"), required=True)
    parser.add_argument("--mode", choices=("run", "submit"), required=True)
    parser.add_argument("--strategy", default="auto")
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--regions", type=int, default=16)
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--logical-iterations", type=int, default=12)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()

    if args.strategy:
        os.environ["TI_GRAPH_VULKAN_STRUCTURED_STRATEGY"] = args.strategy
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
        lowering_mode=lowering_mode,
    )
    build_ms = (time.perf_counter_ns() - build_start) / 1.0e6
    runtime_args = _arguments(
        size=args.size,
        regions=args.regions,
        logical_iterations=args.logical_iterations,
    )
    result = _measure(
        graph,
        runtime_args,
        mode=args.mode,
        warmups=args.warmups,
        repeats=args.repeats,
    )

    expected_state = args.regions * args.logical_iterations
    assert runtime_args["state"].to_numpy()[()] == expected_state
    np.testing.assert_array_equal(
        runtime_args["values"].to_numpy(),
        np.full(args.size, expected_state, dtype=np.float32),
    )
    for index in range(args.regions):
        assert (
            runtime_args[f"counter_{index}"].to_numpy()[()] == args.logical_iterations
        )

    print(
        json.dumps(
            {
                "arch": args.arch,
                "strategy": args.strategy,
                "size": args.size,
                "regions": args.regions,
                "budget": args.budget,
                "logical_iterations": args.logical_iterations,
                "warmups": args.warmups,
                "repeats": args.repeats,
                "build_ms": build_ms,
                "capabilities": capabilities["device_control"],
                "result": result,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
