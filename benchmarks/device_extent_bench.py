"""Measure DeviceExtent publication cost without hidden observation.

The paired Graph comparison keeps dispatch count and payload identical.  The
only semantic difference is a raw count store versus DeviceExtent's bounded
publish.  A separate direct comparison reports the extra launch required when
an existing raw producer needs ``normalize()``; native producers should fuse
the publish contract instead.
"""

import argparse
import json
import statistics
import time

import taichi_forge as ti
from taichi_forge._lib import core as ti_core
from taichi_forge.lang import impl


def _arch(value):
    try:
        return {"cpu": ti.cpu, "cuda": ti.cuda, "vulkan": ti.vulkan}[value]
    except KeyError as exc:
        raise argparse.ArgumentTypeError("arch must be cpu, cuda, or vulkan") from exc


def _measure(call, rounds, samples):
    result = []
    for _ in range(samples):
        start = time.perf_counter()
        for _ in range(rounds):
            call()
        ti.sync()
        result.append((time.perf_counter() - start) * 1e6 / rounds)
    return result


def _median_ratio(baseline, candidate):
    base = statistics.median(baseline)
    value = statistics.median(candidate)
    return base, value, (value / base - 1.0) * 100.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=_arch, default=ti.cuda)
    parser.add_argument("--capacity", type=int, default=1 << 20)
    parser.add_argument("--count", type=int, default=(1 << 20) * 3 // 5)
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--samples", type=int, default=21)
    args = parser.parse_args()
    if args.capacity <= 0 or not 0 <= args.count <= args.capacity:
        parser.error("require capacity > 0 and 0 <= count <= capacity")

    ti.init(arch=args.arch, offline_cache=False)
    raw_state = ti.ndarray(ti.i32, shape=2)
    extent = ti.DeviceExtent(args.capacity)
    raw_output = ti.ndarray(ti.i32, shape=args.capacity)
    extent_output = ti.ndarray(ti.i32, shape=args.capacity)

    @ti.kernel
    def raw_publish(state: ti.types.ndarray(dtype=ti.i32, ndim=1), count: ti.i32):
        state[0] = count
        state[1] = 0

    @ti.kernel
    def bounded_publish(
        state: ti.types.ndarray(dtype=ti.i32, ndim=1), count: ti.i32
    ):
        ti.device_extent_publish(state, args.capacity, count)

    @ti.kernel
    def payload(
        state: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in range(args.capacity):
            if i < state[0]:
                value = i
                for _ in ti.static(range(4)):
                    value = (value * 1664525 + 1013904223) & 0x7FFFFFFF
                output[i] = value
            else:
                output[i] = 0

    state_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "state", ti.i32, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1)
    count_arg = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "count", ti.i32)

    raw_builder = ti.graph.GraphBuilder()
    raw_builder.dispatch(raw_publish, state_arg, count_arg)
    raw_builder.dispatch(payload, state_arg, output_arg)
    raw_graph = raw_builder.compile()

    extent_builder = ti.graph.GraphBuilder()
    extent_builder.dispatch(bounded_publish, state_arg, count_arg)
    extent_builder.dispatch(payload, state_arg, output_arg)
    extent_graph = extent_builder.compile()

    raw_args = {"state": raw_state, "output": raw_output, "count": args.count}
    extent_args = {
        "state": extent.state,
        "output": extent_output,
        "count": args.count,
    }

    for _ in range(64):
        raw_graph.run(raw_args)
        extent_graph.run(extent_args)
    for _ in range(64):
        raw_publish(extent.state, args.count)
        payload(extent.state, extent_output)
        raw_publish(extent.state, args.count)
        extent.normalize()
        payload(extent.state, extent_output)
    ti.sync()

    program = impl.get_runtime().prog
    memory_before = program._runtime_statistics_snapshot()["memory"]
    host_before = dict(ti_core.get_host_memory_pool_stats())
    device_before = dict(ti_core.get_device_memory_pool_stats())

    raw_samples = []
    extent_samples = []
    for sample in range(args.samples):
        first, second = (
            ((raw_graph, raw_args), (extent_graph, extent_args))
            if sample % 2 == 0
            else ((extent_graph, extent_args), (raw_graph, raw_args))
        )
        measured = {}
        for graph, runtime_args in (first, second):
            start = time.perf_counter()
            for _ in range(args.rounds):
                graph.run(runtime_args)
            ti.sync()
            measured[id(graph)] = (time.perf_counter() - start) * 1e6 / args.rounds
        raw_samples.append(measured[id(raw_graph)])
        extent_samples.append(measured[id(extent_graph)])

    raw_direct = _measure(
        lambda: (raw_publish(extent.state, args.count), payload(extent.state, extent_output)),
        args.rounds,
        args.samples,
    )
    normalize_direct = _measure(
        lambda: (
            raw_publish(extent.state, args.count),
            extent.normalize(),
            payload(extent.state, extent_output),
        ),
        args.rounds,
        args.samples,
    )

    graph_base, graph_extent, graph_delta = _median_ratio(raw_samples, extent_samples)
    direct_base, direct_normalize, normalize_delta = _median_ratio(
        raw_direct, normalize_direct
    )
    result = {
        "arch": ti_core.arch_name(impl.current_cfg().arch),
        "capacity": args.capacity,
        "count": args.count,
        "rounds": args.rounds,
        "samples": args.samples,
        "graph_raw_us": graph_base,
        "graph_extent_publish_us": graph_extent,
        "graph_extent_delta_percent": graph_delta,
        "direct_raw_us": direct_base,
        "direct_raw_plus_normalize_us": direct_normalize,
        "normalize_extra_launch_delta_percent": normalize_delta,
        "binding_stable": extent.binding.allocation_identity
        == extent.state._runtime_allocation_identity,
        "runtime_memory_stable": program._runtime_statistics_snapshot()["memory"]
        == memory_before,
        "host_pool_stable": dict(ti_core.get_host_memory_pool_stats()) == host_before,
        "device_pool_stable": dict(ti_core.get_device_memory_pool_stats())
        == device_before,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
