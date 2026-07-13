"""SNodeTree/Graph lifecycle churn benchmark for dense Field bindings.

Each iteration creates a dense FieldsBuilder tree, compiles and runs a Graph,
destroys the tree, and then allows the tree id to be reused.  This measures the
cold lifecycle transaction separately from steady Graph replay benchmarks.
"""

import argparse
import gc
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _rss_mb():
    try:
        import psutil  # pylint: disable=import-outside-toplevel

        return psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        return None


def _gpu_process_mb():
    if platform.system() == "Windows":
        command = (
            f"$p={os.getpid()};$s=0;"
            "(Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage')"
            ".CounterSamples|? InstanceName -like ('pid_'+$p+'_*')|"
            "%{$s+=$_.CookedValue};"
            "[Console]::WriteLine([math]::Round($s/1MB,3))"
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
            process_id, used = [part.strip() for part in line.split(",", 1)]
            if int(process_id) == os.getpid():
                total += float(used)
                found = True
        return total if found else None
    except Exception:
        return None


def _summary_ms(values):
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(len(ordered) * 0.95))
    return {
        "median_ms": statistics.median(values),
        "p95_ms": ordered[p95_index],
        "max_ms": max(values),
    }


def _memory_summary(samples, name):
    values = [sample[name] for sample in samples if sample[name] is not None]
    if not values:
        return {
            "start_mb": None,
            "end_mb": None,
            "peak_mb": None,
            "delta_mb": None,
            "tail_delta_mb": None,
        }
    midpoint = samples[len(samples) // 2][name]
    return {
        "start_mb": values[0],
        "end_mb": values[-1],
        "peak_mb": max(values),
        "delta_mb": values[-1] - values[0],
        "tail_delta_mb": (
            None if midpoint is None else values[-1] - midpoint
        ),
    }


def _arch(ti, name):
    return {"cpu": ti.cpu, "cuda": ti.cuda, "vulkan": ti.vulkan}[name]


def run(args):
    import taichi_forge as ti  # pylint: disable=import-outside-toplevel
    from taichi_forge.lang.exception import (  # pylint: disable=import-outside-toplevel
        TaichiRuntimeError,
    )

    init_start = time.perf_counter()
    ti.init(
        arch=_arch(ti, args.arch),
        enable_fallback=False,
        offline_cache=False,
    )
    init_ms = (time.perf_counter() - init_start) * 1000.0

    @ti.kernel
    def write_values(field: ti.template(), value: ti.i32):
        for i in field:
            field[i] = value + i

    sym_value = ti.graph.Arg(
        ti.graph.ArgKind.SCALAR, "value", ti.i32
    )
    create_times = []
    build_times = []
    run_times = []
    destroy_times = []
    iteration_times = []
    identities = []
    stale_rejected = False
    sample_period = max(1, args.iterations // args.memory_samples)
    memory = [
        {
            "iteration": 0,
            "rss_mb": _rss_mb(),
            "gpu_mb": _gpu_process_mb() if args.arch != "cpu" else None,
        }
    ]

    total_start = time.perf_counter()
    for iteration in range(args.iterations):
        iteration_start = time.perf_counter()
        create_start = time.perf_counter()
        values = ti.field(dtype=ti.i32)
        fields_builder = ti.FieldsBuilder()
        fields_builder.dense(ti.i, args.size).place(values)
        tree = fields_builder.finalize()
        create_times.append((time.perf_counter() - create_start) * 1000.0)
        identity = (tree.id, tree.generation)
        identities.append(identity)

        build_start = time.perf_counter()
        graph_builder = ti.graph.GraphBuilder()
        graph_builder.dispatch(
            write_values,
            sym_value,
            template_args={"field": values},
        )
        graph = graph_builder.compile()
        build_times.append((time.perf_counter() - build_start) * 1000.0)

        run_start = time.perf_counter()
        graph.run({"value": iteration})
        ti.sync()
        run_times.append((time.perf_counter() - run_start) * 1000.0)
        if iteration in (0, args.iterations - 1):
            actual = values.to_numpy()
            expected = np.arange(args.size, dtype=np.int32) + iteration
            if not np.array_equal(actual, expected):
                raise RuntimeError(f"result mismatch at iteration {iteration}")

        destroy_start = time.perf_counter()
        tree.destroy()
        destroy_times.append((time.perf_counter() - destroy_start) * 1000.0)
        if iteration == 0:
            try:
                graph.run({"value": iteration})
            except TaichiRuntimeError:
                stale_rejected = True
            else:
                raise RuntimeError("destroyed-tree Graph was not rejected")

        del graph, graph_builder, tree, fields_builder, values
        if (iteration + 1) % sample_period == 0 or iteration + 1 == args.iterations:
            gc.collect()
            memory.append(
                {
                    "iteration": iteration + 1,
                    "rss_mb": _rss_mb(),
                    "gpu_mb": (
                        _gpu_process_mb() if args.arch != "cpu" else None
                    ),
                }
            )
        iteration_times.append(
            (time.perf_counter() - iteration_start) * 1000.0
        )

    total_ms = (time.perf_counter() - total_start) * 1000.0
    generations = [generation for _, generation in identities]
    tree_ids = [tree_id for tree_id, _ in identities]
    result = {
        "schema": "taichi_forge.graph_dense_field_lifecycle.v1",
        "arch": args.arch,
        "actual_arch": str(ti.lang.impl.current_cfg().arch),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "iterations": args.iterations,
        "field_size": args.size,
        "init_ms": init_ms,
        "total_ms": total_ms,
        "iterations_per_second": args.iterations / (total_ms / 1000.0),
        "create": _summary_ms(create_times),
        "graph_build": _summary_ms(build_times),
        "run_and_sync": _summary_ms(run_times),
        "destroy": _summary_ms(destroy_times),
        "iteration": _summary_ms(iteration_times),
        "rss": _memory_summary(memory, "rss_mb"),
        "gpu": _memory_summary(memory, "gpu_mb"),
        "memory_samples": memory,
        "one_tree_id_reused": len(set(tree_ids)) == 1,
        "generation_strictly_increasing": all(
            left < right for left, right in zip(generations, generations[1:])
        ),
        "first_identity": identities[0],
        "last_identity": identities[-1],
        "stale_graph_rejected": stale_rejected,
    }
    ti.reset()
    gc.collect()
    result["post_reset"] = {
        "rss_mb": _rss_mb(),
        "gpu_mb": _gpu_process_mb() if args.arch != "cpu" else None,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch", choices=["cpu", "cuda", "vulkan"], default="cpu"
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--memory-samples", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.iterations <= 0 or args.size <= 0 or args.memory_samples <= 0:
        parser.error("iterations, size, and memory-samples must be positive")

    result = run(args)
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
