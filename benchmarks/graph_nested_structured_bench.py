"""Measure depth-2 structured Graph control against bounded outer expansion.

The nested case always uses ``Graph.run()`` because depth-2 native submission
is intentionally unavailable. ``Graph.run()`` may still select the qualified
single-replay Vulkan lowering. The two oracle cases can use either ``run`` or
``submit``:

* ``static_expanded`` records ``outer_budget`` root-level inner while regions
  and gates inactive outer slots.
* ``compact_oracle`` records only ``active_outer`` root-level inner while
  regions. It is an upper-bound oracle, not a generally reusable graph.

All cases execute the same payload and write device-owned ``stop_positions``.
The array records the actual inner counter at every active outer invocation,
which makes early termination independently observable from host timing.
"""

import argparse
import json
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
        "p95": _percentile(values, 0.95),
    }


def _scalar(name):
    return ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.i32, ndim=0)


def _build_nested_graph(*, size, outer_budget, inner_budget, chunk_size):
    @ti.kernel
    def initialize(
        values: ti.types.ndarray(dtype=ti.f32, ndim=1),
        stop_positions: ti.types.ndarray(dtype=ti.i32, ndim=1),
        outer_predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        outer_counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
        inner_predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        inner_counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
    ):
        outer_predicate[None] = 0
        outer_counter[None] = 0
        inner_predicate[None] = 0
        inner_counter[None] = 0
        for index in range(size):
            values[index] = 0.0
        for index in range(outer_budget):
            stop_positions[index] = -1

    @ti.kernel
    def outer_condition(
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
        active_outer: ti.i32,
    ):
        predicate[None] = int(counter[None] < active_outer)

    @ti.kernel
    def reset_inner(
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
    ):
        predicate[None] = 0
        counter[None] = 0

    @ti.kernel
    def inner_condition(
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
        outer_predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        outer_counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
        inner_base: ti.i32,
        inner_variation: ti.i32,
    ):
        target = inner_base + outer_counter[None] % inner_variation
        predicate[None] = int(
            outer_predicate[None] != 0 and counter[None] < target
        )

    @ti.kernel
    def inner_step(
        values: ti.types.ndarray(dtype=ti.f32, ndim=1),
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
    ):
        for index in range(size):
            if predicate[None] != 0:
                values[index] += 1.0
        if predicate[None] != 0:
            counter[None] += 1

    @ti.kernel
    def record_stop(
        stop_positions: ti.types.ndarray(dtype=ti.i32, ndim=1),
        outer_predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        outer_counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
        inner_counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
    ):
        if outer_predicate[None] != 0:
            stop_positions[outer_counter[None]] = inner_counter[None]

    @ti.kernel
    def outer_step(
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
    ):
        if predicate[None] != 0:
            counter[None] += 1

    values = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "values", ti.f32, ndim=1
    )
    stop_positions = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "stop_positions", ti.i32, ndim=1
    )
    outer_predicate = _scalar("outer_predicate")
    outer_counter = _scalar("outer_counter")
    inner_predicate = _scalar("inner_predicate")
    inner_counter = _scalar("inner_counter")
    active_outer = ti.graph.Arg(
        ti.graph.ArgKind.SCALAR, "active_outer", ti.i32
    )
    inner_base = ti.graph.Arg(
        ti.graph.ArgKind.SCALAR, "inner_base", ti.i32
    )
    inner_variation = ti.graph.Arg(
        ti.graph.ArgKind.SCALAR, "inner_variation", ti.i32
    )

    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        initialize,
        values,
        stop_positions,
        outer_predicate,
        outer_counter,
        inner_predicate,
        inner_counter,
    )
    outer_condition_region = builder.create_sequential()
    outer_condition_region.dispatch(
        outer_condition,
        outer_predicate,
        outer_counter,
        active_outer,
    )
    inner_condition_region = builder.create_sequential()
    inner_condition_region.dispatch(
        inner_condition,
        inner_predicate,
        inner_counter,
        outer_predicate,
        outer_counter,
        inner_base,
        inner_variation,
    )
    inner_body = builder.create_sequential()
    inner_body.dispatch(
        inner_step,
        values,
        inner_predicate,
        inner_counter,
    )
    outer_body = builder.create_sequential()
    outer_body.dispatch(reset_inner, inner_predicate, inner_counter)
    outer_body.while_loop(
        inner_condition_region,
        inner_body,
        predicate=inner_predicate,
        control_inputs=(
            inner_counter,
            outer_predicate,
            outer_counter,
            inner_base,
            inner_variation,
        ),
        carried_state=(values,),
        counter=inner_counter,
        max_iterations=inner_budget,
        chunk_size=chunk_size,
        lowering_mode="auto",
        name="inner",
    )
    outer_body.dispatch(
        record_stop,
        stop_positions,
        outer_predicate,
        outer_counter,
        inner_counter,
    )
    outer_body.dispatch(outer_step, outer_predicate, outer_counter)
    builder.while_loop(
        outer_condition_region,
        outer_body,
        predicate=outer_predicate,
        control_inputs=(outer_counter, active_outer),
        carried_state=(values, stop_positions, inner_counter),
        counter=outer_counter,
        max_iterations=outer_budget,
        lowering_mode="auto",
        name="outer",
    )
    return builder.compile()


def _build_expanded_graph(
    *,
    size,
    outer_budget,
    stage_count,
    inner_budget,
    chunk_size,
    lowering_mode,
):
    @ti.kernel
    def initialize(
        values: ti.types.ndarray(dtype=ti.f32, ndim=1),
        stop_positions: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for index in range(size):
            values[index] = 0.0
        for index in range(outer_budget):
            stop_positions[index] = -1

    @ti.kernel
    def reset_inner(
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
    ):
        predicate[None] = 0
        counter[None] = 0

    @ti.kernel
    def inner_condition(
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
        enabled: ti.i32,
        target: ti.i32,
    ):
        predicate[None] = int(enabled != 0 and counter[None] < target)

    @ti.kernel
    def inner_step(
        values: ti.types.ndarray(dtype=ti.f32, ndim=1),
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
    ):
        for index in range(size):
            if predicate[None] != 0:
                values[index] += 1.0
        if predicate[None] != 0:
            counter[None] += 1

    @ti.kernel
    def record_stop(
        stop_positions: ti.types.ndarray(dtype=ti.i32, ndim=1),
        counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
        enabled: ti.i32,
        slot: ti.i32,
    ):
        stop_positions[slot] = counter[None] if enabled != 0 else -1

    values = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "values", ti.f32, ndim=1
    )
    stop_positions = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "stop_positions", ti.i32, ndim=1
    )
    predicates = tuple(_scalar(f"predicate_{index}") for index in range(stage_count))
    counters = tuple(_scalar(f"counter_{index}") for index in range(stage_count))
    enabled = tuple(
        ti.graph.Arg(
            ti.graph.ArgKind.SCALAR,
            f"enabled_{index}",
            ti.i32,
        )
        for index in range(stage_count)
    )
    targets = tuple(
        ti.graph.Arg(
            ti.graph.ArgKind.SCALAR,
            f"target_{index}",
            ti.i32,
        )
        for index in range(stage_count)
    )
    slots = tuple(
        ti.graph.Arg(
            ti.graph.ArgKind.SCALAR,
            f"slot_{index}",
            ti.i32,
        )
        for index in range(stage_count)
    )

    builder = ti.graph.GraphBuilder()
    builder.dispatch(initialize, values, stop_positions)
    for index in range(stage_count):
        builder.dispatch(reset_inner, predicates[index], counters[index])
        condition = builder.create_sequential()
        condition.dispatch(
            inner_condition,
            predicates[index],
            counters[index],
            enabled[index],
            targets[index],
        )
        body = builder.create_sequential()
        body.dispatch(
            inner_step,
            values,
            predicates[index],
            counters[index],
        )
        builder.while_loop(
            condition,
            body,
            predicate=predicates[index],
            control_inputs=(
                counters[index],
                enabled[index],
                targets[index],
            ),
            carried_state=(values,),
            counter=counters[index],
            max_iterations=inner_budget,
            chunk_size=chunk_size,
            masked_execution=True,
            lowering_mode=lowering_mode,
            name=f"inner_{index}",
        )
        builder.dispatch(
            record_stop,
            stop_positions,
            counters[index],
            enabled[index],
            slots[index],
        )
    return builder.compile()


def _nested_arguments(*, size, outer_budget, active_outer, inner_base, variation):
    return {
        "values": ti.ndarray(ti.f32, shape=size),
        "stop_positions": ti.ndarray(ti.i32, shape=outer_budget),
        "outer_predicate": ti.ndarray(ti.i32, shape=()),
        "outer_counter": ti.ndarray(ti.i32, shape=()),
        "inner_predicate": ti.ndarray(ti.i32, shape=()),
        "inner_counter": ti.ndarray(ti.i32, shape=()),
        "active_outer": active_outer,
        "inner_base": inner_base,
        "inner_variation": variation,
    }


def _expanded_arguments(
    *,
    size,
    outer_budget,
    stage_count,
    active_outer,
    inner_base,
    variation,
):
    args = {
        "values": ti.ndarray(ti.f32, shape=size),
        "stop_positions": ti.ndarray(ti.i32, shape=outer_budget),
    }
    for index in range(stage_count):
        args[f"predicate_{index}"] = ti.ndarray(ti.i32, shape=())
        args[f"counter_{index}"] = ti.ndarray(ti.i32, shape=())
        args[f"enabled_{index}"] = int(index < active_outer)
        args[f"target_{index}"] = inner_base + index % variation
        args[f"slot_{index}"] = index
    return args


def _queue_submission_stats():
    try:
        return dict(
            ti.lang.impl.get_runtime().prog._debug_vulkan_queue_submission_stats()
        )
    except (AttributeError, RuntimeError):
        return {"supported": False}


def _runtime_delta(before, after, queue_before, queue_after):
    return {
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
            if queue_after.get("supported", False)
            else None
        ),
        "native_submitted_command_buffers": (
            queue_after["submitted_command_buffers"]
            - queue_before["submitted_command_buffers"]
            if queue_after.get("supported", False)
            else None
        ),
        "native_batched_queue_submit_calls": (
            queue_after["batched_queue_submit_calls"]
            - queue_before["batched_queue_submit_calls"]
            if queue_after.get("supported", False)
            else None
        ),
        "native_batched_command_buffers": (
            queue_after["batched_command_buffers"]
            - queue_before["batched_command_buffers"]
            if queue_after.get("supported", False)
            else None
        ),
    }


def _flatten_dicts(value):
    if isinstance(value, dict):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _flatten_dicts(item)


def _backend_counters(graph):
    records = tuple(_flatten_dicts(graph._graph_stats))
    names = (
        "attempts",
        "captures",
        "records",
        "replays",
        "exact_replays",
        "patched_replays",
        "ordinary_fallbacks",
        "dependency_barriers",
        "exit_barriers",
        "barrier_deferrals",
    )
    return {
        name: sum(int(record.get(name, 0)) for record in records)
        for name in names
    }


def _measure(graph, runtime_args, *, mode, warmups, repeats):
    graph.execution_stats()

    def invoke():
        start = time.perf_counter_ns()
        if mode == "submit":
            ticket = graph.submit(runtime_args)
            submitted = time.perf_counter_ns()
            ticket.wait()
            completed = time.perf_counter_ns()
            return (
                (submitted - start) / 1.0e3,
                (completed - submitted) / 1.0e3,
                (completed - start) / 1.0e3,
            )
        graph.run(runtime_args)
        completed = time.perf_counter_ns()
        elapsed = (completed - start) / 1.0e3
        return elapsed, 0.0, elapsed

    for _ in range(warmups):
        invoke()
    ti.sync()
    before = ti.runtime.stats()
    queue_before = _queue_submission_stats()
    host_call_us = []
    completion_wait_us = []
    end_to_end_us = []
    for _ in range(repeats):
        host_call, completion_wait, end_to_end = invoke()
        host_call_us.append(host_call)
        completion_wait_us.append(completion_wait)
        end_to_end_us.append(end_to_end)
    ti.sync()
    after = ti.runtime.stats()
    queue_after = _queue_submission_stats()
    memory = graph.execution_stats().memory
    reports = None
    if mode == "run":
        reports = [
            {
                "name": report.name,
                "region_path": report.region_path,
                "structured_depth": report.structured_depth,
                "kind": getattr(report, "kind", "while"),
                "lowering": report.lowering,
                "native_upgrade_reason": report.native_upgrade_reason,
                "logical_iterations": getattr(report, "logical_iterations", None),
                "executed_iterations": getattr(report, "executed_iterations", None),
                "nested_region_path": getattr(report, "nested_region_path", ""),
                "nested_logical_iterations": list(
                    getattr(report, "nested_logical_iterations", ())
                ),
                "nested_encoded_iterations": list(
                    getattr(report, "nested_encoded_iterations", ())
                ),
                "control_arena_bytes": getattr(
                    report, "control_arena_bytes", 0
                ),
                "device_to_host_bytes": getattr(
                    report, "device_to_host_bytes", 0
                ),
                "indirect_dispatch_count": getattr(
                    report, "indirect_dispatch_count", 0
                ),
                "controller_dispatch_count": getattr(
                    report, "controller_dispatch_count", 0
                ),
                "controller_invocation_count": getattr(
                    report, "controller_invocation_count", 0
                ),
                "zero_dispatch_count": getattr(
                    report, "zero_dispatch_count", 0
                ),
            }
            for report in graph.control_flow_stats()
        ]
    return {
        "mode": mode,
        "host_call_us": _summary(host_call_us),
        "completion_wait_us": _summary(completion_wait_us),
        "end_to_end_us": _summary(end_to_end_us),
        "control_flow_reports": reports,
        "control_flow_report_status": (
            "synchronous_latest_run"
            if mode == "run"
            else "unavailable_after_async_submit"
        ),
        "graph_memory": {
            "persistent_bytes": memory.persistent_bytes,
            "persistent_argument_bytes": memory.persistent_argument_bytes,
            "persistent_observation_bytes": memory.persistent_observation_bytes,
            "persistent_temporary_bytes": memory.persistent_temporary_bytes,
            "persistent_telemetry_bytes": memory.persistent_telemetry_bytes,
            "opaque_driver_bytes": memory.opaque_driver_bytes,
        },
        "submission_delta": _runtime_delta(
            before,
            after,
            queue_before,
            queue_after,
        ),
        "backend_counters": _backend_counters(graph),
    }


def _validate_outputs(
    runtime_args,
    *,
    size,
    outer_budget,
    expected_stops,
):
    expected_total = sum(expected_stops)
    np.testing.assert_array_equal(
        runtime_args["values"].to_numpy(),
        np.full(size, expected_total, dtype=np.float32),
    )
    expected = np.full(outer_budget, -1, dtype=np.int32)
    expected[: len(expected_stops)] = np.asarray(expected_stops, dtype=np.int32)
    actual = runtime_args["stop_positions"].to_numpy()
    np.testing.assert_array_equal(actual, expected)
    return {
        "expected_total_inner_iterations": expected_total,
        "stop_positions": actual.tolist(),
        "active_stop_positions": actual[: len(expected_stops)].tolist(),
        "inactive_slots_remain_negative": bool(
            np.all(actual[len(expected_stops) :] == -1)
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=("cpu", "cuda", "vulkan"), required=True)
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--outer-budget", type=int, default=16)
    parser.add_argument("--active-outer", type=int, default=4)
    parser.add_argument("--inner-budget", type=int, default=32)
    parser.add_argument("--inner-base", type=int, default=8)
    parser.add_argument("--inner-variation", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--oracle-mode",
        choices=("run", "submit"),
        default="run",
        help="Execution mode for static_expanded and compact_oracle only",
    )
    args = parser.parse_args()

    if args.size <= 0:
        parser.error("--size must be positive")
    if args.outer_budget <= 0:
        parser.error("--outer-budget must be positive")
    if args.active_outer < 0 or args.active_outer > args.outer_budget:
        parser.error("--active-outer must be between zero and --outer-budget")
    if args.inner_budget <= 0:
        parser.error("--inner-budget must be positive")
    if args.inner_base < 0:
        parser.error("--inner-base must be nonnegative")
    if args.inner_variation <= 0:
        parser.error("--inner-variation must be positive")
    if args.inner_base + args.inner_variation - 1 > args.inner_budget:
        parser.error("inner stop targets must not exceed --inner-budget")
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    if args.warmups < 0 or args.repeats <= 0:
        parser.error("--warmups must be nonnegative and --repeats positive")

    ti.init(arch=_arch(args.arch), offline_cache=False)
    capabilities = ti.graph.structured_control_capabilities()
    if args.oracle_mode == "submit" and not capabilities["device_control"][
        "structured_submit"
    ]:
        parser.error(
            "--oracle-mode=submit requires qualified root structured "
            f"submission: {capabilities['device_control']['structured_submit_reason']}"
        )

    expected_stops = [
        args.inner_base + index % args.inner_variation
        for index in range(args.active_outer)
    ]
    oracle_lowering = (
        "native_required" if args.oracle_mode == "submit" else "auto"
    )
    cases = []

    build_start = time.perf_counter_ns()
    nested = _build_nested_graph(
        size=args.size,
        outer_budget=args.outer_budget,
        inner_budget=args.inner_budget,
        chunk_size=args.chunk_size,
    )
    nested_build_ms = (time.perf_counter_ns() - build_start) / 1.0e6
    nested_args = _nested_arguments(
        size=args.size,
        outer_budget=args.outer_budget,
        active_outer=args.active_outer,
        inner_base=args.inner_base,
        variation=args.inner_variation,
    )
    nested_result = _measure(
        nested,
        nested_args,
        mode="run",
        warmups=args.warmups,
        repeats=args.repeats,
    )
    nested_result["correctness"] = _validate_outputs(
        nested_args,
        size=args.size,
        outer_budget=args.outer_budget,
        expected_stops=expected_stops,
    )
    cases.append(
        {
            "case": "nested",
            "graph_shape": "outer_while_contains_inner_while",
            "execution_contract": "Graph.run automatic depth-2 control",
            "recorded_outer_regions": 1,
            "build_ms": nested_build_ms,
            "result": nested_result,
        }
    )

    for case_name, stage_count in (
        ("static_expanded", args.outer_budget),
        ("compact_oracle", args.active_outer),
    ):
        build_start = time.perf_counter_ns()
        graph = _build_expanded_graph(
            size=args.size,
            outer_budget=args.outer_budget,
            stage_count=stage_count,
            inner_budget=args.inner_budget,
            chunk_size=args.chunk_size,
            lowering_mode=oracle_lowering,
        )
        build_ms = (time.perf_counter_ns() - build_start) / 1.0e6
        runtime_args = _expanded_arguments(
            size=args.size,
            outer_budget=args.outer_budget,
            stage_count=stage_count,
            active_outer=args.active_outer,
            inner_base=args.inner_base,
            variation=args.inner_variation,
        )
        result = _measure(
            graph,
            runtime_args,
            mode=args.oracle_mode,
            warmups=args.warmups,
            repeats=args.repeats,
        )
        result["correctness"] = _validate_outputs(
            runtime_args,
            size=args.size,
            outer_budget=args.outer_budget,
            expected_stops=expected_stops,
        )
        cases.append(
            {
                "case": case_name,
                "graph_shape": (
                    "fixed_outer_budget_root_inner_regions"
                    if case_name == "static_expanded"
                    else "active_outer_root_inner_regions"
                ),
                "execution_contract": (
                    f"Graph.{args.oracle_mode} root-level structured control"
                ),
                "recorded_outer_regions": stage_count,
                "build_ms": build_ms,
                "result": result,
            }
        )

    print(
        json.dumps(
            {
                "arch": args.arch,
                "size": args.size,
                "outer_budget": args.outer_budget,
                "active_outer": args.active_outer,
                "inner_budget": args.inner_budget,
                "inner_base": args.inner_base,
                "inner_variation": args.inner_variation,
                "expected_stop_positions": expected_stops,
                "chunk_size": args.chunk_size,
                "warmups": args.warmups,
                "repeats": args.repeats,
                "oracle_mode": args.oracle_mode,
                "capabilities": {
                    key: capabilities["device_control"][key]
                    for key in (
                        "nested_structured_control",
                        "max_structured_depth",
                        "nested_exact_portable",
                        "nested_native_lowering",
                        "nested_leaf_native_upgrade",
                        "nested_leaf_native_kinds",
                        "native_max_structured_depth",
                        "nested_async_submit",
                        "structured_submit",
                        "structured_submit_reason",
                    )
                },
                "measurement_contract": {
                    "nested_always_uses_graph_run": True,
                    "oracle_mode_applies_only_to_non_nested_cases": True,
                    "stop_positions_written_on_device": True,
                    "host_readback_excluded_from_timing": True,
                    "cases_share_one_process_and_jit_cache": True,
                    "compact_oracle_requires_graph_rebuild_for_active_outer": True,
                },
                "cases": cases,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
