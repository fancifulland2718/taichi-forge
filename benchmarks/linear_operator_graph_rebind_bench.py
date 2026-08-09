"""Compare numeric-generation Graph rebinding with per-update rebuilding."""

import argparse
import json
import statistics
import time

import numpy as np

import taichi_forge as ti


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=("cpu", "cuda", "vulkan"), required=True)
    parser.add_argument("--size", type=int, default=262144)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=20)
    return parser.parse_args()


def _array(values, dtype):
    values = np.asarray(values)
    result = ti.ndarray(dtype, shape=values.size)
    result.from_numpy(values)
    return result


def main():
    args = _arguments()
    if args.size <= 0 or args.warmup < 0 or args.repeats <= 0:
        raise ValueError("size/repeats must be positive and warmup non-negative")
    ti.init(arch=getattr(ti, args.arch), offline_cache=False)
    size = args.size
    topology = _array(np.arange(size, dtype=np.int32), ti.i32)
    numeric_a = _array(np.full(size, 2.0, np.float32), ti.f32)
    numeric_b = _array(np.full(size, 3.0, np.float32), ti.f32)

    @ti.kernel
    def diagonal_apply(
        active_size: ti.i32,
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        numeric_data: ti.types.ndarray(dtype=ti.f32, ndim=1),
        input: ti.types.ndarray(dtype=ti.f32, ndim=1),
        output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(active_size):
            output[index] = numeric_data[index] * input[topology_data[index]]

    def make_operator():
        return ti.linalg.LinearOperator.from_kernel(
            diagonal_apply,
            size,
            topology,
            numeric=numeric_a,
            traits=ti.linalg.OperatorTraits.spd(),
        )

    input_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "input", ti.f32, ndim=1
    )
    output_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1
    )

    def compile_graph(operator):
        builder = ti.graph.GraphBuilder()
        builder.append_native(operator.graph_action(input_arg, output_arg))
        return builder.compile()

    rebound_operator = make_operator()
    rebuilt_operator = make_operator()
    rebound_graph = compile_graph(rebound_operator)
    source = _array(np.linspace(-1.0, 1.0, size, dtype=np.float32), ti.f32)
    rebound_output = ti.ndarray(ti.f32, shape=size)
    rebuilt_output = ti.ndarray(ti.f32, shape=size)
    rebound_args = {"input": source, "output": rebound_output}
    rebuilt_args = {"input": source, "output": rebuilt_output}
    versions = {"rebind": 1, "rebuild": 1}

    def update_rebind(numeric):
        version = versions["rebind"]
        rebound_operator.update_numeric(
            numeric,
            expected_topology_version=1,
            expected_numeric_version=version,
        )
        versions["rebind"] = version + 1
        rebound_graph.submit(rebound_args).wait()

    def update_rebuild(numeric):
        version = versions["rebuild"]
        rebuilt_operator.update_numeric(
            numeric,
            expected_topology_version=1,
            expected_numeric_version=version,
        )
        versions["rebuild"] = version + 1
        graph = compile_graph(rebuilt_operator)
        graph.submit(rebuilt_args).wait()

    samples = {"numeric_rebind": [], "graph_rebuild": []}
    total_rounds = args.warmup + args.repeats
    for index in range(total_rounds):
        numeric = numeric_a if index % 2 else numeric_b
        methods = (
            (("numeric_rebind", update_rebind), ("graph_rebuild", update_rebuild))
            if index % 2
            else (("graph_rebuild", update_rebuild), ("numeric_rebind", update_rebind))
        )
        for name, method in methods:
            start = time.perf_counter_ns()
            method(numeric)
            elapsed_ms = (time.perf_counter_ns() - start) / 1.0e6
            if index >= args.warmup:
                samples[name].append(elapsed_ms)

    expected_scale = 2.0 if (total_rounds - 1) % 2 else 3.0
    expected = expected_scale * source.to_numpy()
    rebound_error = float(
        np.max(np.abs(rebound_output.to_numpy() - expected), initial=0.0)
    )
    rebuilt_error = float(
        np.max(np.abs(rebuilt_output.to_numpy() - expected), initial=0.0)
    )
    rebind_median = statistics.median(samples["numeric_rebind"])
    rebuild_median = statistics.median(samples["graph_rebuild"])
    provider_stats = rebound_operator._provider_core._debug_runtime_stats()
    print(
        json.dumps(
            {
                "schema_version": 1,
                "backend": args.arch,
                "size": size,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "median_ms": {
                    key: statistics.median(value)
                    for key, value in samples.items()
                },
                "rebuild_over_rebind": rebuild_median / rebind_median,
                "correctness": {
                    "rebind_max_abs_error": rebound_error,
                    "rebuild_max_abs_error": rebuilt_error,
                },
                "rebind_provider_generations": {
                    "operations": provider_stats["operations"],
                    "resources": provider_stats["resources"],
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
