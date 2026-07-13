"""Benchmark zero-copy versus Python-copied mixed-Graph segment arguments.

The compiled-Graph binding already filters its Python dict by the current
segment declaration while constructing the C++ IValue map. 'backend-filter'
is the production path; 'python-copy' quantifies the rejected alternative of
materializing another node-local dict in Python. Run modes in fresh processes
so JIT/replay caches and allocator state do not cross-contaminate.
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import time
import types

import numpy as np

import taichi_forge as ti
import taichi_forge.algorithms._algorithms as alg_impl
from taichi_forge.graph._graph import _CompiledCGraphNode


RESULT_PREFIX = "MIXED_GRAPH_SEGMENT_RESULT "


def _rss_mb():
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (
            1024.0 * 1024.0
        )
    except Exception:
        return None


def _gpu_process_mb(pid):
    if platform.system() == "Windows":
        command = (
            "$p="
            + str(pid)
            + ";$s=0;(Get-Counter "
            + "'\\GPU Process Memory(*)\\Dedicated Usage').CounterSamples|"
            + "? InstanceName -like ('pid_'+$p+'_*')|"
            + "%{$s+=$_.CookedValue};"
            + "[Console]::WriteLine([math]::Round($s/1MB,3))"
        )
        argv = ["powershell", "-NoProfile", "-Command", command]
    else:
        argv = [
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    try:
        output = subprocess.check_output(
            argv, text=True, stderr=subprocess.DEVNULL, timeout=3.0
        )
        if platform.system() == "Windows":
            return float(output.strip())
        total = 0.0
        found = False
        for line in output.splitlines():
            process_id, used = [
                part.strip() for part in line.split(",", 1)
            ]
            if int(process_id) == pid:
                total += float(used)
                found = True
        return total if found else None
    except Exception:
        return None


def _median_p95(values):
    ordered = sorted(values)
    p95_index = min(
        len(ordered) - 1, max(0, int(len(ordered) * 0.95 + 0.999999) - 1)
    )
    return {
        "median_ms": statistics.median(values),
        "p95_ms": ordered[p95_index],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "samples": len(values),
    }


def _install_python_copy_argument_mode(graph):
    def run_with_python_copy(self, context):
        flattened = context.flattened_args()
        node_args = {
            name: flattened[name] for name in self.runtime_arg_names
        }
        self.compiled_graph.jit_run_cached(
            context.compile_config(),
            node_args,
            self._jit_cache,
        )

    for node in graph._spec.nodes:
        if isinstance(node, _CompiledCGraphNode):
            node.run = types.MethodType(run_with_python_copy, node)


def _build_graph(size):
    src = ti.ndarray(ti.i32, shape=size)
    prepared = ti.ndarray(ti.i32, shape=size)
    transformed = ti.ndarray(ti.i32, shape=size)
    output = ti.ndarray(ti.i32, shape=size)
    src.from_numpy(np.arange(size, dtype=np.int32))

    @ti.kernel
    def prepare(
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        destination: ti.types.ndarray(dtype=ti.i32, ndim=1),
        scale: ti.i32,
    ):
        for i in source:
            destination[i] = source[i] * scale + 1

    @ti.kernel
    def finalize(
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        destination: ti.types.ndarray(dtype=ti.i32, ndim=1),
        bias: ti.i32,
    ):
        for i in source:
            destination[i] = source[i] + bias

    src_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "src", ti.i32, ndim=1
    )
    prepared_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "prepared", ti.i32, ndim=1
    )
    prepare_scale_arg = ti.graph.Arg(
        ti.graph.ArgKind.SCALAR, "prepare_scale", ti.i32
    )
    transformed_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "transformed", ti.i32, ndim=1
    )
    output_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1
    )
    bias_arg = ti.graph.Arg(
        ti.graph.ArgKind.SCALAR, "bias", ti.i32
    )

    sequence = alg_impl.primitive_sequence()
    sequence.transform(prepared, transformed, scale=2, bias=3)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(prepare, src_arg, prepared_arg, prepare_scale_arg)
    builder.dispatch(prepare, src_arg, prepared_arg, prepare_scale_arg)
    builder.append_native(sequence)
    builder.dispatch(finalize, transformed_arg, output_arg, bias_arg)
    builder.dispatch(finalize, transformed_arg, output_arg, bias_arg)
    graph = builder.compile()
    runtime_args = {
        "src": src,
        "prepared": prepared,
        "prepare_scale": 2,
        "transformed": transformed,
        "output": output,
        "bias": 5,
    }
    return graph, runtime_args, output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch", choices=("cpu", "cuda", "vulkan"), required=True
    )
    parser.add_argument(
        "--argument-mode",
        choices=("backend-filter", "python-copy"),
        required=True,
    )
    parser.add_argument("--size", type=int, default=65536)
    parser.add_argument("--warmups", type=int, default=12)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--sample-gpu-memory", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--vary-scalars", action="store_true")
    args = parser.parse_args()

    arch = {"cpu": ti.cpu, "cuda": ti.cuda, "vulkan": ti.vulkan}[
        args.arch
    ]
    rss_before = _rss_mb()
    gpu_before = (
        _gpu_process_mb(os.getpid())
        if args.sample_gpu_memory and args.arch != "cpu"
        else None
    )
    ti.init(arch=arch, offline_cache=False)
    build_start = time.perf_counter()
    graph, runtime_args, output = _build_graph(args.size)
    if args.argument_mode == "python-copy":
        _install_python_copy_argument_mode(graph)
    build_ms = (time.perf_counter() - build_start) * 1000.0
    diagnostics_before = graph._graph_stats if args.diagnostics else None

    first_start = time.perf_counter()
    graph.run(runtime_args)
    ti.sync()
    first_ms = (time.perf_counter() - first_start) * 1000.0
    for index in range(args.warmups):
        if args.vary_scalars:
            runtime_args["prepare_scale"] = index % 3 + 1
            runtime_args["bias"] = index - 3
        graph.run(runtime_args)
    ti.sync()

    samples = []
    for sample in range(args.samples):
        start = time.perf_counter()
        for batch_index in range(args.batch):
            if args.vary_scalars:
                runtime_args["prepare_scale"] = (
                    sample + batch_index
                ) % 3 + 1
                runtime_args["bias"] = sample - batch_index
            graph.run(runtime_args)
        ti.sync()
        samples.append(
            (time.perf_counter() - start) * 1000.0 / args.batch
        )

    rss_after = _rss_mb()
    gpu_after = (
        _gpu_process_mb(os.getpid())
        if args.sample_gpu_memory and args.arch != "cpu"
        else None
    )
    result = {
        "arch": args.arch,
        "argument_mode": args.argument_mode,
        "size": args.size,
        "vary_scalars": args.vary_scalars,
        "build_ms": build_ms,
        "first_ms": first_ms,
        "steady": _median_p95(samples),
        "rss_before_mb": rss_before,
        "rss_after_mb": rss_after,
        "rss_delta_mb": (
            None
            if rss_before is None or rss_after is None
            else rss_after - rss_before
        ),
        "gpu_before_mb": gpu_before,
        "gpu_after_mb": gpu_after,
        "gpu_delta_mb": (
            None
            if gpu_before is None or gpu_after is None
            else gpu_after - gpu_before
        ),
        "diagnostics_before": diagnostics_before,
        "graph_stats": graph._graph_stats,
        "node_runtime_arg_names": [
            sorted(node.runtime_arg_names) for node in graph._spec.nodes
        ],
        "checksum": int(output.to_numpy().sum(dtype=np.int64)),
    }
    print(RESULT_PREFIX + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
