"""Compare fused shifted Graph lowering with an explicit identity sum."""

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
    parser.add_argument("--shift", type=float, default=1.5)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=50)
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
    diagonal = _array(np.full(size, 2.0, np.float32), ti.f32)
    ones = _array(np.ones(size, dtype=np.float32), ti.f32)

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

    traits = ti.linalg.OperatorTraits.spd()
    base = ti.linalg.LinearOperator.from_kernel(
        diagonal_apply, size, topology, numeric=diagonal, traits=traits
    )
    identity = ti.linalg.LinearOperator.from_kernel(
        diagonal_apply, size, topology, numeric=ones, traits=traits
    )
    operators = {
        "shifted": base.shifted(args.shift),
        "explicit_identity_sum": base + args.shift * identity,
    }
    input_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "input", ti.f32, ndim=1
    )
    output_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1
    )
    graphs = {}
    outputs = {}
    for name, operator in operators.items():
        builder = ti.graph.GraphBuilder()
        builder.append_native(operator.graph_action(input_arg, output_arg))
        graphs[name] = builder.compile()
        outputs[name] = ti.ndarray(ti.f32, shape=size)

    source_host = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    source = _array(source_host, ti.f32)
    samples = {name: [] for name in operators}
    total_rounds = args.warmup + args.repeats
    for index in range(total_rounds):
        order = tuple(operators) if index % 2 else tuple(reversed(operators))
        for name in order:
            start = time.perf_counter_ns()
            graphs[name].submit(
                {"input": source, "output": outputs[name]}
            ).wait()
            elapsed_ms = (time.perf_counter_ns() - start) / 1.0e6
            if index >= args.warmup:
                samples[name].append(elapsed_ms)

    expected = (2.0 + args.shift) * source_host
    medians = {
        name: statistics.median(values) for name, values in samples.items()
    }
    print(
        json.dumps(
            {
                "schema_version": 1,
                "backend": args.arch,
                "size": size,
                "shift": args.shift,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "median_ms": medians,
                "explicit_over_shifted": (
                    medians["explicit_identity_sum"] / medians["shifted"]
                ),
                "correctness": {
                    name: float(
                        np.max(
                            np.abs(output.to_numpy() - expected), initial=0.0
                        )
                    )
                    for name, output in outputs.items()
                },
                "graph_nodes": {
                    name: graph._debug_info["nodes"]
                    for name, graph in graphs.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
