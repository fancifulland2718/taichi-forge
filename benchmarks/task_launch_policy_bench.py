"""Paired auto-vs-policy benchmark for direct JIT range kernels.

Example:
  python benchmarks/task_launch_policy_bench.py --arch cuda --block 256 \
      --count 16777216 --batch 300 --pairs 41
"""

import argparse
import json
import random
import statistics
import time

import taichi_forge as ti


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=("cpu", "cuda", "vulkan"), required=True)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--count", type=int, default=1 << 22)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--pairs", type=int, default=21)
    parser.add_argument("--seed", type=int, default=20260802)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.count <= 0 or args.batch <= 0 or args.pairs <= 0:
        raise ValueError("count, batch, and pairs must be positive")

    arch = {"cpu": ti.cpu, "cuda": ti.cuda, "vulkan": ti.vulkan}[args.arch]
    ti.init(arch=arch, offline_cache=False)
    lhs = ti.ndarray(ti.f32, shape=args.count)
    rhs = ti.ndarray(ti.f32, shape=args.count)
    result = ti.ndarray(ti.f32, shape=args.count)
    sample = ti.field(ti.f32, shape=2)

    @ti.kernel
    def initialize(
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in range(args.count):
            x[i] = 0.25
            y[i] = 0.75

    @ti.kernel
    def update(
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
        out: ti.types.ndarray(dtype=ti.f32, ndim=1),
        count: ti.i32,
    ):
        for i in range(count):
            a = x[i]
            b = y[i]
            for _ in ti.static(range(16)):
                a = a * 1.000013 + b * 0.00017
                b = b * 0.999991 + a * 0.00011
            out[i] = a + b

    @ti.kernel
    def capture(index: ti.i32, out: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        sample[index] = out[0]

    initialize(lhs, rhs)
    mode = "hint" if args.arch == "cpu" else "require"
    tuned = update.with_launch_policy(ti.TaskLaunchPolicy.block(args.block, mode=mode))
    report = tuned.report(lhs, rhs, result, args.count)
    range_task = next(task for task in report.tasks if task.task_type == "range_for")
    if args.arch != "cpu" and range_task.selected_block_size != args.block:
        raise RuntimeError("backend did not select the requested block")

    update(lhs, rhs, result, args.count)
    capture(0, result)
    tuned(lhs, rhs, result, args.count)
    capture(1, result)
    ti.sync()
    if sample[0] != sample[1]:
        raise RuntimeError("auto and policy results differ")

    for _ in range(12):
        update(lhs, rhs, result, args.count)
        tuned(lhs, rhs, result, args.count)
    ti.sync()
    program = ti.lang.impl.get_runtime().prog
    memory_before = program._runtime_statistics_snapshot()["memory"]

    def measure(call):
        start = time.perf_counter_ns()
        for _ in range(args.batch):
            call(lhs, rhs, result, args.count)
        ti.sync()
        return (time.perf_counter_ns() - start) / args.batch / 1e6

    samples = {"auto": [], "policy": []}
    orders = [["auto", "policy"], ["policy", "auto"]] * ((args.pairs + 1) // 2)
    random.Random(args.seed).shuffle(orders)
    variants = {"auto": update, "policy": tuned}
    for order in orders[: args.pairs]:
        for name in order:
            samples[name].append(measure(variants[name]))

    memory_after = program._runtime_statistics_snapshot()["memory"]
    if memory_after != memory_before:
        raise RuntimeError("warm policy measurement changed runtime memory ownership")
    auto = statistics.median(samples["auto"])
    policy = statistics.median(samples["policy"])
    payload = {
        "arch": args.arch,
        "block": args.block,
        "selected_block": range_task.selected_block_size,
        "status": report.status,
        "count": args.count,
        "batch": args.batch,
        "pairs": args.pairs,
        "auto_median_ms": auto,
        "policy_median_ms": policy,
        "speedup_percent": (auto / policy - 1.0) * 100.0,
        "auto_min_ms": min(samples["auto"]),
        "auto_max_ms": max(samples["auto"]),
        "policy_min_ms": min(samples["policy"]),
        "policy_max_ms": max(samples["policy"]),
        "runtime_memory": memory_after,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
