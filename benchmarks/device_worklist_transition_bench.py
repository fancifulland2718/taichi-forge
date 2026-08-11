"""Measure staged versus finalize-free DeviceWorklist transitions."""

import argparse
import json
import random
import statistics
import time
from collections.abc import Mapping

import numpy as np

import taichi_forge as ti


def _json_safe(value):
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _summary(samples):
    mean = statistics.fmean(samples)
    return {
        "median_us": statistics.median(samples),
        "mean_us": mean,
        "cv": statistics.pstdev(samples) / mean if mean else 0.0,
        "samples_us": samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=("cpu", "cuda", "vulkan"), required=True)
    parser.add_argument("--capacity", type=int, default=65_536)
    parser.add_argument("--count", type=int, default=16_384)
    parser.add_argument("--launches", type=int, default=20)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    if not 0 <= args.count <= args.capacity:
        raise ValueError("count must be in [0, capacity]")

    arch = {"cpu": ti.cpu, "cuda": ti.cuda, "vulkan": ti.vulkan}[args.arch]
    ti.init(arch=arch, offline_cache=False)
    capacity = args.capacity

    @ti.kernel
    def produce_staged(
        values: ti.types.ndarray(dtype=ti.i32, ndim=1),
        extent: ti.types.ndarray(dtype=ti.i32, ndim=1),
        generated: ti.types.ndarray(dtype=ti.i32, ndim=0),
        overflow: ti.types.ndarray(dtype=ti.i32, ndim=0),
        limit: ti.i32,
        count: ti.i32,
    ):
        for i in range(count):
            ti.algorithms.device_worklist_append(
                values, extent, generated, overflow, limit, i + 1
            )

    @ti.kernel
    def produce_direct(
        values: ti.types.ndarray(dtype=ti.i32, ndim=1),
        extent: ti.types.ndarray(dtype=ti.i32, ndim=1),
        overflow: ti.types.ndarray(dtype=ti.i32, ndim=0),
        limit: ti.i32,
        count: ti.i32,
    ):
        for i in range(count):
            ti.algorithms.device_worklist_append_direct(
                values, extent, overflow, limit, i + 1
            )

    @ti.kernel
    def consume(
        values: ti.types.ndarray(dtype=ti.i32, ndim=1),
        extent: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in range(capacity):
            if i < ti.device_extent_count(extent):
                output[i] = values[i] * 3 + 1

    def build(mode):
        worklist = ti.algorithms.DeviceWorklist(
            capacity,
            ti.i32,
            telemetry=False,
            transition_mode=mode,
        )
        symbolic = worklist.graph_args(mode)
        count_arg = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "count", ti.i32)
        output_arg = ti.graph.Arg(
            ti.graph.ArgKind.NDARRAY, f"{mode}_output", ti.i32, ndim=1
        )
        builder = ti.graph.GraphBuilder()
        builder.append_native(
            ti.algorithms.DeviceWorklistSequence(symbolic).prepare_next(),
            admission="auto",
        )
        builder.dispatch(
            produce_direct if mode == "direct" else produce_staged,
            *symbolic.append_arguments(),
            count_arg,
        )
        if mode == "staged":
            builder.append_native(
                ti.algorithms.DeviceWorklistSequence(symbolic).finalize_next(),
                admission="auto",
            )
        builder.dispatch(
            consume, symbolic.next_values, symbolic.next_extent, output_arg
        )
        graph = builder.compile()
        output = ti.ndarray(ti.i32, shape=capacity)
        runtime_args = worklist.runtime_arguments(mode, include_capacity=True)
        runtime_args.update(count=args.count, **{f"{mode}_output": output})
        return worklist, graph, output, runtime_args

    routes = {mode: build(mode) for mode in ("staged", "direct")}

    def run(route, launches):
        start = time.perf_counter_ns()
        for _ in range(launches):
            route[1].run(route[3])
        ti.sync()
        return (time.perf_counter_ns() - start) / launches / 1.0e3

    for route in routes.values():
        run(route, 3)
    samples = {mode: [] for mode in routes}
    order_rng = random.Random(args.seed)
    for _ in range(args.samples):
        order = list(routes)
        order_rng.shuffle(order)
        for mode in order:
            samples[mode].append(run(routes[mode], args.launches))

    expected = np.arange(1, args.count + 1, dtype=np.int32) * 3 + 1
    for mode, route in routes.items():
        run(route, 1)
        actual = np.sort(route[2].to_numpy()[: args.count])
        np.testing.assert_array_equal(actual, expected)
        if route[0].next_extent.snapshot().count != args.count:
            raise RuntimeError(f"{mode} extent mismatch")

    summary = {mode: _summary(values) for mode, values in samples.items()}
    staged = summary["staged"]["median_us"]
    direct = summary["direct"]["median_us"]
    result = {
        "schema_version": 1,
        "arch": args.arch,
        "capacity": capacity,
        "count": args.count,
        "launches_per_sample": args.launches,
        "correct": True,
        "summary": summary,
        "direct_speedup": staged / direct,
        "direct_percent_change": 100.0 * (direct / staged - 1.0),
        "physical_plan": {
            mode: _json_safe(route[1].physical_plan())
            for mode, route in routes.items()
        },
        "memory": {mode: route[0].memory_report() for mode, route in routes.items()},
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
