"""Benchmark recordable LinearOperator composition and direct Field binding.

Run GPU measurements only while the target device is otherwise idle. The
equivalent paths deliberately execute the same five mathematical dispatches;
this keeps Graph submission overhead separate from algebraic simplification.
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


def _measure_interleaved(methods, warmups, samples, batch):
    first_us = {}
    for name, function in methods:
        start = time.perf_counter_ns()
        function()
        first_us[name] = (time.perf_counter_ns() - start) / 1.0e3

    for warmup in range(warmups):
        offset = warmup % len(methods)
        for _, function in methods[offset:] + methods[:offset]:
            function()

    timings = {name: [] for name, _ in methods}
    for sample in range(samples):
        offset = sample % len(methods)
        for name, function in methods[offset:] + methods[:offset]:
            start = time.perf_counter_ns()
            for _ in range(batch):
                function()
            timings[name].append((time.perf_counter_ns() - start) / 1.0e3 / batch)

    return {
        name: {
            "first_completion_us": first_us[name],
            "steady_completion_us": {
                "median": statistics.median(timings[name]),
                "p10": _percentile(timings[name], 0.1),
                "p90": _percentile(timings[name], 0.9),
            },
        }
        for name, _ in methods
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=("cpu", "cuda", "vulkan"), required=True)
    parser.add_argument("--size", type=int, default=262144)
    parser.add_argument("--warmups", type=int, default=8)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.size <= 0 or args.warmups < 0 or args.samples <= 0 or args.batch <= 0:
        parser.error("size/samples/batch must be positive and warmups non-negative")

    ti.init(arch=_arch(args.arch), offline_cache=False)
    size = args.size
    topology = ti.ndarray(ti.i32, shape=size)
    diagonal = ti.ndarray(ti.f32, shape=size)
    topology.from_numpy(np.arange(size, dtype=np.int32))
    diagonal_host = np.linspace(1.25, 2.75, size, dtype=np.float32)
    diagonal.from_numpy(diagonal_host)

    @ti.kernel
    def diagonal_apply(
        active_size: ti.i32,
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        numeric_data: ti.types.ndarray(dtype=ti.f32, ndim=1),
        source: ti.types.ndarray(dtype=ti.f32, ndim=1),
        output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(active_size):
            output[index] = numeric_data[index] * source[topology_data[index]]

    @ti.kernel
    def scale_two(values: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for index in range(size):
            values[index] *= 2.0

    @ti.kernel
    def add(
        addend: ti.types.ndarray(dtype=ti.f32, ndim=1),
        output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(size):
            output[index] += addend[index]

    base = ti.linalg.LinearOperator.from_kernel(
        diagonal_apply,
        size,
        topology,
        numeric=diagonal,
        traits=ti.linalg.OperatorTraits.spd(),
    )
    composed = (2.0 * base + base).compose(base)

    input_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "input", ti.f32, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1)

    def build_automatic_graph():
        builder = ti.graph.GraphBuilder()
        builder.append_native(composed.graph_action(input_arg, output_arg))
        return builder.compile()

    automatic_graph = build_automatic_graph()
    direct_field_graph = build_automatic_graph()
    staged_field_graph = build_automatic_graph()

    first_scratch_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "first_scratch", ti.f32, ndim=1)
    second_scratch_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "second_scratch", ti.f32, ndim=1)
    explicit_builder = ti.graph.GraphBuilder()
    explicit_builder.append_native(base.graph_action(input_arg, first_scratch_arg))
    explicit_builder.append_native(base.graph_action(first_scratch_arg, output_arg))
    explicit_builder.dispatch(scale_two, output_arg)
    explicit_builder.append_native(base.graph_action(first_scratch_arg, second_scratch_arg))
    explicit_builder.dispatch(add, second_scratch_arg, output_arg)
    explicit_graph = explicit_builder.compile()

    source_host = np.sin(np.linspace(0.0, 16.0, size, dtype=np.float32))
    expected = 3.0 * diagonal_host * diagonal_host * source_host
    source = ti.ndarray(ti.f32, shape=size)
    source.from_numpy(source_host)
    output = ti.ndarray(ti.f32, shape=size)
    first_scratch = ti.ndarray(ti.f32, shape=size)
    second_scratch = ti.ndarray(ti.f32, shape=size)

    automatic_args = {"input": source, "output": output}
    explicit_args = {
        "input": source,
        "output": output,
        "first_scratch": first_scratch,
        "second_scratch": second_scratch,
    }

    def automatic_graph_step():
        automatic_graph.submit(automatic_args).wait()

    def explicit_graph_step():
        explicit_graph.submit(explicit_args).wait()

    def standalone_composition_step():
        composed.apply(source, out=output)
        ti.sync()

    def no_graph_step():
        base.apply(source, out=first_scratch)
        base.apply(first_scratch, out=output)
        scale_two(output)
        base.apply(first_scratch, out=second_scratch)
        add(second_scratch, output)
        ti.sync()

    scalar_source = ti.field(ti.f32, shape=size)
    scalar_output = ti.field(ti.f32, shape=size)
    scalar_source.from_numpy(source_host)
    field_args = {"input": scalar_source, "output": scalar_output}
    staged_input = ti.ndarray(ti.f32, shape=size)
    staged_output = ti.ndarray(ti.f32, shape=size)

    @ti.kernel
    def pack_field(output_array: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for index in range(size):
            output_array[index] = scalar_source[index]

    @ti.kernel
    def unpack_field(input_array: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for index in range(size):
            scalar_output[index] = input_array[index]

    staged_args = {"input": staged_input, "output": staged_output}

    def direct_field_graph_step():
        direct_field_graph.submit(field_args).wait()

    def staged_field_sequence_step():
        pack_field(staged_input)
        staged_field_graph.submit(staged_args).wait()
        unpack_field(staged_output)
        ti.sync()

    methods = [
        ("no_graph_sequence", no_graph_step),
        ("standalone_composition", standalone_composition_step),
        ("explicit_graph", explicit_graph_step),
        ("automatic_composition_graph", automatic_graph_step),
        ("direct_field_graph", direct_field_graph_step),
        ("staged_field_sequence", staged_field_sequence_step),
    ]
    measurements = _measure_interleaved(methods, args.warmups, args.samples, args.batch)

    automatic_graph_step()
    max_error = float(np.max(np.abs(output.to_numpy() - expected)))
    direct_field_graph_step()
    field_max_error = float(np.max(np.abs(scalar_output.to_numpy() - expected)))
    tolerance = 2.0e-5 * max(1.0, float(np.max(np.abs(expected))))
    if max_error > tolerance or field_max_error > tolerance:
        raise RuntimeError(
            "composition correctness gate failed: "
            f"ndarray={max_error}, field={field_max_error}, tolerance={tolerance}"
        )

    median = {name: value["steady_completion_us"]["median"] for name, value in measurements.items()}
    memory = automatic_graph.execution_stats().memory
    report = {
        "schema_version": 1,
        "arch": args.arch,
        "size": size,
        "warmups": args.warmups,
        "samples": args.samples,
        "batch": args.batch,
        "measurements": measurements,
        "ratios": {
            "automatic_graph_speedup_vs_no_graph": (
                median["no_graph_sequence"] / median["automatic_composition_graph"]
            ),
            "automatic_graph_speedup_vs_standalone": (
                median["standalone_composition"] / median["automatic_composition_graph"]
            ),
            "automatic_vs_explicit_graph": (median["automatic_composition_graph"] / median["explicit_graph"]),
            "direct_field_speedup_vs_staged": (median["staged_field_sequence"] / median["direct_field_graph"]),
        },
        "correctness": {
            "ndarray_max_abs_error": max_error,
            "field_max_abs_error": field_max_error,
            "tolerance": tolerance,
        },
        "automatic_graph_memory": {
            "planned_temporary_bytes": memory.planned_temporary_bytes,
            "persistent_temporary_bytes": memory.persistent_temporary_bytes,
            "temporary_arena_slots": memory.temporary_arena_slots,
            "temporary_arena_allocations": memory.temporary_arena_allocations,
            "temporary_arena_reuses": memory.temporary_arena_reuses,
            "opaque_driver_bytes": memory.opaque_driver_bytes,
        },
        "automatic_graph_execution": {
            "dispatch_count": automatic_graph.execution_stats().dispatch_count,
            "runtime_arg_count": automatic_graph.execution_stats().runtime_arg_count,
            "execution_path": automatic_graph.execution_stats().execution_path,
        },
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(encoded + "\n")


if __name__ == "__main__":
    main()
