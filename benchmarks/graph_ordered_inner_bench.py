"""Compare one ordered nested ticket with host-separated inner Graphs.

The split case is an optimistic host-known outer loop, but it still has to
complete each adaptive inner Graph before advancing. The combined case keeps
the outer loop and both ordered inner loops in one backend submission.
"""

import argparse
import json
import statistics
import time

import numpy as np

import taichi_forge as ti


def _arch(name):
    return {"cuda": ti.cuda, "vulkan": ti.vulkan}[name]


def _scalar(name):
    return ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.i32, ndim=0)


def _percentile(values, fraction):
    values = sorted(values)
    return values[round((len(values) - 1) * fraction)]


def _summary(values):
    return {
        "median": statistics.median(values),
        "p10": _percentile(values, 0.1),
        "p90": _percentile(values, 0.9),
    }


def _build(size, outer_budget, a_budget, b_budget):
    @ti.kernel
    def initialize(
        values: ti.types.ndarray(dtype=ti.f32, ndim=1),
        outer_state: ti.types.ndarray(dtype=ti.i32, ndim=0),
        outer_predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        outer_counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
        a_state: ti.types.ndarray(dtype=ti.i32, ndim=0),
        a_predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        a_counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
        b_state: ti.types.ndarray(dtype=ti.i32, ndim=0),
        b_predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        b_counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
    ):
        outer_state[None] = 0
        outer_predicate[None] = 0
        outer_counter[None] = 0
        a_state[None] = 0
        a_predicate[None] = 0
        a_counter[None] = 0
        b_state[None] = 0
        b_predicate[None] = 0
        b_counter[None] = 0
        for index in range(size):
            values[index] = 0.0

    @ti.kernel
    def reset_inner(
        a_state: ti.types.ndarray(dtype=ti.i32, ndim=0),
        a_counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
        b_state: ti.types.ndarray(dtype=ti.i32, ndim=0),
        b_counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
    ):
        a_state[None] = 0
        a_counter[None] = 0
        b_state[None] = 0
        b_counter[None] = 0

    @ti.kernel
    def evaluate_outer(
        state: ti.types.ndarray(dtype=ti.i32, ndim=0),
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        limit: ti.i32,
    ):
        predicate[None] = int(state[None] < limit)

    @ti.kernel
    def evaluate_a_nested(
        state: ti.types.ndarray(dtype=ti.i32, ndim=0),
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        outer_state: ti.types.ndarray(dtype=ti.i32, ndim=0),
        base: ti.i32,
    ):
        predicate[None] = int(state[None] < base + outer_state[None])

    @ti.kernel
    def evaluate_a_split(
        state: ti.types.ndarray(dtype=ti.i32, ndim=0),
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        target: ti.i32,
    ):
        predicate[None] = int(state[None] < target)

    @ti.kernel
    def evaluate_b(
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
    def finish_outer(
        state: ti.types.ndarray(dtype=ti.i32, ndim=0),
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
    ):
        if predicate[None] != 0:
            state[None] += 1
            counter[None] += 1

    names = (
        "outer_state",
        "outer_predicate",
        "outer_counter",
        "a_state",
        "a_predicate",
        "a_counter",
        "b_state",
        "b_predicate",
        "b_counter",
    )
    controls = {name: _scalar(name) for name in names}
    values = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "values", ti.f32, ndim=1)
    outer_limit = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "outer_limit", ti.i32)
    a_base = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "a_base", ti.i32)
    a_target = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "a_target", ti.i32)
    b_target = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "b_target", ti.i32)

    def append_inner(factory, region, condition_kernel, prefix, target_arg):
        condition = factory.create_sequential()
        condition_args = [
            controls[f"{prefix}_state"],
            controls[f"{prefix}_predicate"],
        ]
        if condition_kernel is evaluate_a_nested:
            condition_args.append(controls["outer_state"])
        condition_args.append(target_arg)
        condition.dispatch(condition_kernel, *condition_args)
        body = factory.create_sequential()
        body.dispatch(
            step,
            values,
            controls[f"{prefix}_state"],
            controls[f"{prefix}_predicate"],
            controls[f"{prefix}_counter"],
        )
        region.while_loop(
            condition,
            body,
            predicate=controls[f"{prefix}_predicate"],
            counter=controls[f"{prefix}_counter"],
            max_iterations=a_budget if prefix == "a" else b_budget,
            chunk_size=4,
            masked_execution=True,
            name=f"inner_{prefix}",
        )

    combined_builder = ti.graph.GraphBuilder()
    combined_builder.dispatch(initialize, values, *(controls[name] for name in names))
    outer_condition = combined_builder.create_sequential()
    outer_condition.dispatch(
        evaluate_outer,
        controls["outer_state"],
        controls["outer_predicate"],
        outer_limit,
    )
    outer_body = combined_builder.create_sequential()
    outer_body.dispatch(
        reset_inner,
        controls["a_state"],
        controls["a_counter"],
        controls["b_state"],
        controls["b_counter"],
    )
    append_inner(combined_builder, outer_body, evaluate_a_nested, "a", a_base)
    append_inner(combined_builder, outer_body, evaluate_b, "b", b_target)
    outer_body.dispatch(
        finish_outer,
        controls["outer_state"],
        controls["outer_predicate"],
        controls["outer_counter"],
    )
    combined_builder.while_loop(
        outer_condition,
        outer_body,
        predicate=controls["outer_predicate"],
        counter=controls["outer_counter"],
        max_iterations=outer_budget,
        chunk_size=4,
        masked_execution=True,
        name="outer",
    )

    split_graphs = []
    for condition_kernel, prefix, target_arg in (
        (evaluate_a_split, "a", a_target),
        (evaluate_b, "b", b_target),
    ):
        builder = ti.graph.GraphBuilder()
        append_inner(builder, builder, condition_kernel, prefix, target_arg)
        split_graphs.append(builder.compile())

    kernels = {
        "initialize": initialize,
        "reset_inner": reset_inner,
        "finish_outer": finish_outer,
    }
    return combined_builder.compile(), tuple(split_graphs), kernels, names


def _arguments(size, names, outer_limit, a_base, b_target):
    result = {name: ti.ndarray(ti.i32, shape=()) for name in names}
    result["values"] = ti.ndarray(ti.f32, shape=size)
    result.update(
        {
            "outer_limit": outer_limit,
            "a_base": a_base,
            "a_target": a_base,
            "b_target": b_target,
        }
    )
    return result


def _run_combined(graph, args):
    runtime_args = {name: value for name, value in args.items() if name != "a_target"}
    ticket = graph.submit(runtime_args)
    ticket.wait()


def _run_split(graphs, kernels, args):
    names = (
        "outer_state",
        "outer_predicate",
        "outer_counter",
        "a_state",
        "a_predicate",
        "a_counter",
        "b_state",
        "b_predicate",
        "b_counter",
    )
    kernels["initialize"](args["values"], *(args[name] for name in names))
    for outer_index in range(args["outer_limit"]):
        kernels["reset_inner"](
            args["a_state"],
            args["a_counter"],
            args["b_state"],
            args["b_counter"],
        )
        a_args = {
            "values": args["values"],
            "a_state": args["a_state"],
            "a_predicate": args["a_predicate"],
            "a_counter": args["a_counter"],
            "a_target": args["a_base"] + outer_index,
        }
        b_args = {
            "values": args["values"],
            "b_state": args["b_state"],
            "b_predicate": args["b_predicate"],
            "b_counter": args["b_counter"],
            "b_target": args["b_target"],
        }
        graphs[0].submit(a_args).wait()
        graphs[1].submit(b_args).wait()
        kernels["finish_outer"](
            args["outer_state"],
            args["outer_predicate"],
            args["outer_counter"],
        )
    ti.sync()


def _measure(callback, warmups, repeats):
    for _ in range(warmups):
        callback()
    values = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        callback()
        values.append((time.perf_counter_ns() - started) / 1.0e3)
    return _summary(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=("cuda", "vulkan"), required=True)
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--outer-limit", type=int, default=4)
    parser.add_argument("--outer-budget", type=int, default=8)
    parser.add_argument("--a-base", type=int, default=6)
    parser.add_argument("--a-budget", type=int, default=16)
    parser.add_argument("--b-target", type=int, default=2)
    parser.add_argument("--b-budget", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output")
    args = parser.parse_args()
    if min(args.size, args.outer_limit, args.a_base, args.b_target) <= 0:
        parser.error("size, active loop counts, and targets must be positive")
    if args.outer_limit > args.outer_budget:
        parser.error("--outer-limit must not exceed --outer-budget")

    ti.init(arch=_arch(args.arch), offline_cache=False)
    combined, split, kernels, names = _build(
        args.size, args.outer_budget, args.a_budget, args.b_budget
    )
    combined_args = _arguments(
        args.size, names, args.outer_limit, args.a_base, args.b_target
    )
    split_args = _arguments(
        args.size, names, args.outer_limit, args.a_base, args.b_target
    )
    combined_result = _measure(
        lambda: _run_combined(combined, combined_args),
        args.warmups,
        args.repeats,
    )
    split_result = _measure(
        lambda: _run_split(split, kernels, split_args),
        args.warmups,
        args.repeats,
    )
    expected = sum(args.a_base + index for index in range(args.outer_limit))
    expected += args.outer_limit * args.b_target
    combined_values = combined_args["values"].to_numpy()
    split_values = split_args["values"].to_numpy()
    if not np.all(combined_values == expected) or not np.all(split_values == expected):
        raise RuntimeError("ordered-inner benchmark correctness check failed")

    result = {
        "arch": args.arch,
        "size": args.size,
        "outer_limit": args.outer_limit,
        "inner_targets": {
            "a": [args.a_base + index for index in range(args.outer_limit)],
            "b": args.b_target,
        },
        "warmups": args.warmups,
        "repeats": args.repeats,
        "single_ticket_us": combined_result,
        "host_separated_us": split_result,
        "median_speedup": split_result["median"] / combined_result["median"],
        "correctness": {"expected_value": expected},
    }
    serialized = json.dumps(result, indent=2, sort_keys=True)
    print(serialized)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output_file:
            output_file.write(serialized)
            output_file.write("\n")


if __name__ == "__main__":
    main()
