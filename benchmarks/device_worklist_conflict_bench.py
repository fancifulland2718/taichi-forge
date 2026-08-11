"""Compare deterministic DeviceWorklist conflict strategies.

Both routes consume byte-identical values, keys, priorities, and ordinals.
Samples use shuffled paired order and synchronize only after each launch batch.
The JSON result includes raw samples, variability, correctness parity, and
workspace accounting so published comparisons remain independently auditable.
"""

import argparse
import json
import random
import statistics
import time

import numpy as np

import taichi_forge as ti


def _arch(name):
    return {"cpu": ti.cpu, "cuda": ti.cuda, "vulkan": ti.vulkan}[name]


def _summary(values):
    mean = statistics.fmean(values)
    return {
        "median_us": statistics.median(values),
        "p95_us": float(np.percentile(values, 95)),
        "mean_us": mean,
        "cv": statistics.pstdev(values) / mean if mean else 0.0,
        "min_us": min(values),
        "max_us": max(values),
    }


def _make_route(
    capacity,
    key_capacity,
    count,
    strategy,
    values_host,
    keys_host,
    priorities_host,
):
    worklist = ti.algorithms.DeviceWorklist(capacity, ti.i32)
    worklist.values.from_numpy(values_host)
    worklist.extent.set(count)
    keys = ti.ndarray(ti.i32, shape=capacity)
    priorities = ti.ndarray(ti.i32, shape=capacity)
    keys.from_numpy(keys_host)
    priorities.from_numpy(priorities_host)
    output_keys = ti.ndarray(ti.i32, shape=capacity)
    output_priorities = ti.ndarray(ti.i32, shape=capacity)
    output_ordinals = ti.ndarray(ti.i32, shape=capacity)

    prefix = f"claims_{strategy}"
    args = worklist.graph_args(prefix)
    keys_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, f"keys_{strategy}", ti.i32, ndim=1
    )
    priorities_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY,
        f"priorities_{strategy}",
        ti.i32,
        ndim=1,
    )
    output_keys_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY,
        f"output_keys_{strategy}",
        ti.i32,
        ndim=1,
    )
    output_priorities_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY,
        f"output_priorities_{strategy}",
        ti.i32,
        ndim=1,
    )
    output_ordinals_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY,
        f"output_ordinals_{strategy}",
        ti.i32,
        ndim=1,
    )
    sequence = ti.algorithms.DeviceWorklistSequence(args)
    sequence.resolve_conflicts(
        keys_arg,
        output_keys_arg,
        output_priorities_arg,
        output_ordinals_arg,
        priorities=priorities_arg,
        policy="min_priority",
        strategy=strategy,
        key_capacity=key_capacity,
    )
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    graph = builder.compile()
    runtime_args = worklist.runtime_arguments(prefix)
    runtime_args.update(
        {
            f"keys_{strategy}": keys,
            f"priorities_{strategy}": priorities,
            f"output_keys_{strategy}": output_keys,
            f"output_priorities_{strategy}": output_priorities,
            f"output_ordinals_{strategy}": output_ordinals,
        }
    )
    return {
        "graph": graph,
        "runtime_args": runtime_args,
        "worklist": worklist,
        "output_keys": output_keys,
        "output_priorities": output_priorities,
        "output_ordinals": output_ordinals,
        "memory": sequence.memory_report(),
    }


def _measure(route, launches):
    start = time.perf_counter_ns()
    for _ in range(launches):
        route["graph"].run(route["runtime_args"])
    ti.sync()
    return (time.perf_counter_ns() - start) / launches / 1.0e3


def _snapshot(route):
    count = route["worklist"].next_extent.snapshot().count
    return {
        "count": count,
        "values": route["worklist"].next_values.to_numpy()[:count].copy(),
        "keys": route["output_keys"].to_numpy()[:count].copy(),
        "priorities": route["output_priorities"].to_numpy()[:count].copy(),
        "ordinals": route["output_ordinals"].to_numpy()[:count].copy(),
    }


def _assert_parity(lhs, rhs):
    if lhs["count"] != rhs["count"]:
        raise RuntimeError(
            f"winner count mismatch: {lhs['count']} != {rhs['count']}"
        )
    for name in ("values", "keys", "priorities", "ordinals"):
        np.testing.assert_array_equal(lhs[name], rhs[name])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=("cpu", "cuda", "vulkan"), required=True)
    parser.add_argument("--capacity", type=int, default=262_144)
    parser.add_argument("--key-capacity", type=int, default=4_096)
    parser.add_argument("--count", type=int, default=262_144)
    parser.add_argument("--launches", type=int, default=20)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--seed", type=int, default=1729)
    ns = parser.parse_args()
    if not 0 <= ns.count <= ns.capacity:
        raise ValueError("count must be in [0, capacity]")
    if not 0 < ns.key_capacity <= ns.capacity:
        raise ValueError("key-capacity must be in [1, capacity]")

    ti.init(arch=_arch(ns.arch), offline_cache=False)
    rng = np.random.default_rng(ns.seed)
    values_host = rng.integers(0, 1 << 20, ns.capacity, dtype=np.int32)
    keys_host = rng.integers(0, ns.key_capacity, ns.capacity, dtype=np.int32)
    priorities_host = rng.integers(
        -(1 << 20), 1 << 20, ns.capacity, dtype=np.int32
    )
    routes = {
        strategy: _make_route(
            ns.capacity,
            ns.key_capacity,
            ns.count,
            strategy,
            values_host,
            keys_host,
            priorities_host,
        )
        for strategy in ("dense_atomic", "radix_grouped")
    }
    for route in routes.values():
        _measure(route, 2)
    snapshots = {}
    for name, route in routes.items():
        _measure(route, 1)
        snapshots[name] = _snapshot(route)
    _assert_parity(snapshots["dense_atomic"], snapshots["radix_grouped"])

    samples = {name: [] for name in routes}
    order_rng = random.Random(ns.seed ^ 0x5EED)
    for _ in range(ns.samples):
        order = list(routes)
        order_rng.shuffle(order)
        for name in order:
            samples[name].append(_measure(routes[name], ns.launches))
    summaries = {name: _summary(values) for name, values in samples.items()}
    dense = summaries["dense_atomic"]["median_us"]
    radix = summaries["radix_grouped"]["median_us"]
    result = {
        "schema_version": 1,
        "arch": ns.arch,
        "capacity": ns.capacity,
        "key_capacity": ns.key_capacity,
        "count": ns.count,
        "seed": ns.seed,
        "launches_per_sample": ns.launches,
        "correctness_parity": True,
        "winner_count": snapshots["dense_atomic"]["count"],
        "samples_us_per_launch": samples,
        "summary": summaries,
        "dense_vs_radix_percent": 100.0 * (dense / radix - 1.0),
        "memory": {name: route["memory"] for name, route in routes.items()},
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
