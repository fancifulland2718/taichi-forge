"""Measure structured-control preparation, replay, and observation overhead.

Run each GPU backend only while the target device is otherwise idle. Compare
``auto`` and ``portable`` in the same fresh CUDA process; preparation numbers
remain order/cache sensitive and steady replay is the primary result.
"""

import argparse
import json
import statistics
import time

import numpy as np

import taichi_forge as ti


RUNNING = 0
CONVERGED = 1


def _arch(name):
    return {
        "cpu": ti.cpu,
        "cuda": ti.cuda,
        "vulkan": ti.vulkan,
    }[name]


def _build_graph(size, iterations, lowering_mode):
    @ti.kernel
    def initialize(
        values: ti.types.ndarray(dtype=ti.f32, ndim=1),
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        status: ti.types.ndarray(dtype=ti.i32, ndim=0),
        counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
    ):
        predicate[None] = 0
        status[None] = RUNNING
        counter[None] = 0
        for index in range(size):
            values[index] = 0.0

    @ti.kernel
    def condition(
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        status: ti.types.ndarray(dtype=ti.i32, ndim=0),
        counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
        target: ti.i32,
    ):
        if status[None] == RUNNING and counter[None] >= target:
            status[None] = CONVERGED
        predicate[None] = int(status[None] == RUNNING)

    @ti.kernel
    def step(
        values: ti.types.ndarray(dtype=ti.f32, ndim=1),
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
    ):
        for index in range(size):
            if predicate[None] != 0:
                values[index] += 1.0
        if predicate[None] != 0:
            counter[None] += 1

    values_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "values", ti.f32, ndim=1
    )
    predicate_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "predicate", ti.i32, ndim=0
    )
    status_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "status", ti.i32, ndim=0
    )
    counter_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "counter", ti.i32, ndim=0
    )
    target_arg = ti.graph.Arg(
        ti.graph.ArgKind.SCALAR, "target", ti.i32
    )
    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        initialize, values_arg, predicate_arg, status_arg, counter_arg
    )
    condition_region = builder.create_sequential()
    condition_region.dispatch(
        condition,
        predicate_arg,
        status_arg,
        counter_arg,
        target_arg,
    )
    body = builder.create_sequential()
    body.dispatch(step, values_arg, predicate_arg, counter_arg)
    builder.while_loop(
        condition_region,
        body,
        predicate=predicate_arg,
        status=status_arg,
        control_inputs=(counter_arg, target_arg),
        carried_state=(values_arg,),
        counter=counter_arg,
        max_iterations=iterations + 4,
        lowering_mode=lowering_mode,
        name=f"control_{lowering_mode}",
    )
    return builder.compile()


def _percentile(values, fraction):
    values = sorted(values)
    position = round((len(values) - 1) * fraction)
    return values[position]


def _measure(mode, size, iterations, warmups, repeats, kernel_profiler):
    build_start = time.perf_counter_ns()
    graph = _build_graph(size, iterations, mode)
    build_ms = (time.perf_counter_ns() - build_start) / 1.0e6
    args = {
        "values": ti.ndarray(ti.f32, shape=size),
        "predicate": ti.ndarray(ti.i32, shape=()),
        "status": ti.ndarray(ti.i32, shape=()),
        "counter": ti.ndarray(ti.i32, shape=()),
        "target": iterations,
    }
    first_start = time.perf_counter_ns()
    graph.run(args)
    ti.sync()
    first_ms = (time.perf_counter_ns() - first_start) / 1.0e6
    for _ in range(warmups):
        graph.run(args)
    ti.sync()

    wall_us = []
    if kernel_profiler:
        ti.profiler.clear_kernel_profiler_info()
    for _ in range(repeats):
        start = time.perf_counter_ns()
        graph.run(args)
        wall_us.append((time.perf_counter_ns() - start) / 1.0e3)
    ti.sync()
    kernel_us = None
    if kernel_profiler:
        profile_seconds = ti.profiler.get_kernel_profiler_total_time()
        if profile_seconds > 0.0:
            kernel_us = profile_seconds * 1.0e6 / repeats
    report = graph.control_flow_stats()[0]
    np.testing.assert_array_equal(
        args["values"].to_numpy(),
        np.full(size, iterations, dtype=np.float32),
    )
    assert report.final_status == CONVERGED
    assert report.logical_iterations == iterations
    median_us = statistics.median(wall_us)
    return {
        "mode": mode,
        "build_ms": build_ms,
        "first_run_ms": first_ms,
        "steady_wall_us": {
            "median": median_us,
            "p10": _percentile(wall_us, 0.1),
            "p90": _percentile(wall_us, 0.9),
        },
        "device_kernel_us_per_run": kernel_us,
        "kernel_profiler_visible": kernel_us is not None,
        "estimated_non_kernel_us_per_run": (
            None if kernel_us is None else median_us - kernel_us
        ),
        "lowering": report.lowering,
        "logical_iterations": report.logical_iterations,
        "executed_iterations": report.executed_iterations,
        "observation_batches": report.observation_batches,
        "device_to_host_bytes": report.device_to_host_bytes,
        "native_upgrade_reason": report.native_upgrade_reason,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=("cpu", "cuda", "vulkan"), required=True)
    parser.add_argument("--size", type=int, default=262144)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--kernel-profiler", action="store_true")
    args = parser.parse_args()
    ti.init(
        arch=_arch(args.arch),
        offline_cache=False,
        kernel_profiler=args.kernel_profiler,
    )
    modes = ("auto", "portable") if args.arch == "cuda" else ("auto",)
    result = {
        "arch": args.arch,
        "size": args.size,
        "iterations": args.iterations,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "contract": {
            "preparation_is_order_and_cache_sensitive": True,
            "steady_wall_time_excludes_result_readback": True,
            "zero_or_invisible_kernel_profiler_time_is_null": True,
        },
        "results": [
            _measure(
                mode,
                args.size,
                args.iterations,
                args.warmups,
                args.repeats,
                args.kernel_profiler,
            )
            for mode in modes
        ],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
