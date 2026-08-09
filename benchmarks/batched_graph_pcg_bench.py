"""Measure batched PCG control policies with one recordable A/M pair."""

import argparse
import json
import statistics
import time

import numpy as np

import taichi_forge as ti
from taichi_forge._lib import core as ti_core
from taichi_forge.lang import impl


def _array(values, dtype=ti.f32):
    values = np.asarray(values)
    result = ti.ndarray(dtype, shape=values.size)
    result.from_numpy(values)
    return result


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=("cuda", "vulkan"), required=True)
    parser.add_argument("--size", type=int, default=262144)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-iterations", type=int, default=32)
    parser.add_argument(
        "--preconditioner",
        choices=("exact", "sqrt", "identity"),
        default="sqrt",
    )
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=(
            "host_each_iteration",
            "host_check_every_k",
            "fixed_budget_masked",
            "device_convergent",
        ),
        default=(
            "host_each_iteration",
            "host_check_every_k",
            "fixed_budget_masked",
            "device_convergent",
        ),
    )
    return parser.parse_args()


def main():
    args = _arguments()
    if args.size <= 0 or args.batch_size <= 0:
        raise ValueError("size and batch-size must be positive")
    if args.size % args.batch_size:
        raise ValueError("size must be divisible by batch-size")
    ti.init(arch=getattr(ti, args.arch), offline_cache=False)

    size = args.size
    topology = _array(np.arange(size, dtype=np.int32), ti.i32)
    system_size = size // args.batch_size
    system_diagonal = np.geomspace(
        1.0, 1.0e4, system_size, dtype=np.float32
    )
    diagonal_host = np.tile(system_diagonal, args.batch_size)

    @ti.kernel
    def diagonal_apply(
        active_size: ti.i32,
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        numeric_data: ti.types.ndarray(dtype=ti.f32, ndim=1),
        input: ti.types.ndarray(dtype=ti.f32, ndim=1),
        output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(active_size):
            output[index] = (
                numeric_data[index] * input[topology_data[index]]
            )

    operator = ti.linalg.LinearOperator.from_kernel(
        diagonal_apply,
        size,
        topology,
        numeric=_array(diagonal_host),
        traits=ti.linalg.OperatorTraits.spd(),
    )
    inverse_host = {
        "exact": 1.0 / diagonal_host,
        "sqrt": 1.0 / np.sqrt(diagonal_host),
        "identity": np.ones_like(diagonal_host),
    }[args.preconditioner]
    preconditioner = ti.linalg.inverse_block_diagonal(
        _array(inverse_host), 1, assume_spd=True
    )
    exact = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    rhs = _array(diagonal_host * exact)

    policies = tuple(dict.fromkeys(args.policies))
    plans = {}
    unsupported = {}
    for policy in policies:
        options = {}
        if policy == "host_check_every_k":
            options["check_interval"] = 4
        try:
            plans[policy] = ti.linalg.experimental.BatchedSolvePlan(
                operator,
                args.batch_size,
                independent_systems=True,
                method="pcg",
                preconditioner=preconditioner,
                max_iterations=args.max_iterations,
                atol=1e-5,
                execution_policy=policy,
                **options,
            )
        except RuntimeError as exc:
            unsupported[policy] = str(exc)

    samples = {policy: [] for policy in plans}
    last_results = {}
    schedule = tuple(plans)
    runtime_memory_before = None
    host_pool_before = None
    device_pool_before = None
    for round_index in range(args.warmup + args.repeats):
        if round_index == args.warmup:
            ti.sync()
            runtime_stats = (
                impl.get_runtime().prog._runtime_statistics_snapshot()
            )
            runtime_memory_before = runtime_stats["memory"]
            host_pool_before = dict(ti_core.get_host_memory_pool_stats())
            device_pool_before = dict(ti_core.get_device_memory_pool_stats())
        ordered = schedule if round_index % 2 == 0 else tuple(reversed(schedule))
        for policy in ordered:
            start = time.perf_counter_ns()
            result = plans[policy].solve(rhs)
            elapsed = time.perf_counter_ns() - start
            if not result.all_converged:
                raise RuntimeError(f"{policy} did not converge")
            last_results[policy] = result
            if round_index >= args.warmup:
                samples[policy].append(elapsed / 1.0e6)

    ti.sync()
    runtime_memory_after = impl.get_runtime().prog._runtime_statistics_snapshot()[
        "memory"
    ]
    host_pool_after = dict(ti_core.get_host_memory_pool_stats())
    device_pool_after = dict(ti_core.get_device_memory_pool_stats())

    medians = {
        policy: statistics.median(values)
        for policy, values in samples.items()
    }
    device = medians.get("device_convergent")
    records = {}
    for policy, plan in plans.items():
        stats = plan.statistics()
        record = {
            "median_ms": medians[policy],
            "minimum_ms": min(samples[policy]),
            "iteration_min": min(last_results[policy].iterations),
            "iteration_max": max(last_results[policy].iterations),
            "speedup_vs_device_convergent": (
                None if device is None else medians[policy] / device
            ),
        }
        if not args.compact:
            record.update(
                iterations=last_results[policy].iterations,
                operations=stats["operations"],
                device_convergent_replay=stats[
                    "device_convergent_replay"
                ],
                recurrence_replay=stats["recurrence_replay"],
                resources=stats["resources"],
            )
        records[policy] = record
    print(
        json.dumps(
            {
                "schema_version": 1,
                "backend": args.arch,
                "size": size,
                "batch_size": args.batch_size,
                "max_iterations": args.max_iterations,
                "preconditioner": args.preconditioner,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "steady_memory": {
                    "runtime_memory_stable": (
                        runtime_memory_after == runtime_memory_before
                    ),
                    "host_pool_stable": host_pool_after == host_pool_before,
                    "device_pool_stable": (
                        device_pool_after == device_pool_before
                    ),
                },
                "records": records,
                "unsupported": unsupported,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
