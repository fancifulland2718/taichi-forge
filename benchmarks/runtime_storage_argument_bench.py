"""Measure dense runtime-argument overhead without per-sample helper processes.

The benchmark separates three quantities:

* synchronized end-to-end wall time per invocation;
* host submission interval while a batch is enqueued before one synchronization;
* backend kernel time reported by Taichi's kernel profiler.

The remaining batched wall time is reported as non-kernel time.  It contains
argument preparation, host submission, device scheduling gaps, and the single
amortized synchronization, so it must not be presented as pure Python overhead.
GPU performance runs fail closed unless ``gpu_idle_guard`` verifies that no
other Python GPU compute process is active.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))

from gpu_idle_guard import (  # pylint: disable=wrong-import-position
    finalize_performance_measurement,
    prepare_performance_measurement,
)

import taichi_forge as ti  # pylint: disable=wrong-import-position


def _arch(name):
    return {"cpu": ti.cpu, "cuda": ti.cuda, "vulkan": ti.vulkan}[name]


def _summary(values):
    return {
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "samples": len(values),
    }


def _kernel_profiler_us_per_invocation(body, batch_size):
    ti.profiler.clear_kernel_profiler_info()
    for _ in range(batch_size):
        body()
    ti.sync()
    seconds = ti.profiler.get_kernel_profiler_total_time()
    return seconds * 1.0e6 / batch_size


def _measure_case(name, body, check, *, warmups, repeats, batch_size):
    for _ in range(warmups):
        body()
    ti.sync()

    synchronized_us = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        body()
        ti.sync()
        synchronized_us.append((time.perf_counter_ns() - start) / 1.0e3)

    submit_us = []
    batch_wall_us = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        for _ in range(batch_size):
            body()
        submitted = time.perf_counter_ns()
        ti.sync()
        completed = time.perf_counter_ns()
        submit_us.append((submitted - start) / 1.0e3 / batch_size)
        batch_wall_us.append((completed - start) / 1.0e3 / batch_size)

    kernel_us = _kernel_profiler_us_per_invocation(body, batch_size)
    correct, max_abs_error = check()
    batch_median = statistics.median(batch_wall_us)
    return {
        "name": name,
        "correct": bool(correct),
        "max_abs_error": float(max_abs_error),
        "dispatches_per_invocation": 2,
        "synchronized_wall_us_per_invocation": _summary(synchronized_us),
        "host_submit_us_per_invocation": _summary(submit_us),
        "batch_wall_us_per_invocation": _summary(batch_wall_us),
        "kernel_us_per_invocation": kernel_us,
        "kernel_us_per_dispatch": kernel_us / 2.0,
        "batch_non_kernel_us_per_invocation": max(0.0, batch_median - kernel_us),
    }


def run(args):
    measurement = prepare_performance_measurement(
        args.arch, requested=args.performance
    )
    ti.init(
        arch=_arch(args.arch),
        offline_cache=False,
        kernel_profiler=True,
    )
    actual_arch = ti.lang.impl.current_cfg().arch
    if actual_arch != _arch(args.arch):
        result = {
            **measurement,
            "arch": args.arch,
            "actual_arch": str(actual_arch),
            "skipped": True,
            "reason": "requested backend is unavailable",
            "cases": [],
        }
        result.update(
            finalize_performance_measurement(
                measurement,
                skipped=True,
                reason="requested backend is unavailable",
            )
        )
        return result

    n = args.n
    source_np = ((np.arange(n, dtype=np.float32) * 0.25) % 31.0).astype(
        np.float32
    )
    scale = np.float32(1.75)
    bias = np.float32(-0.5)
    expected = source_np * scale + bias + np.float32(1.0)

    @ti.kernel
    def stage0(
        source: ti.types.ndarray(dtype=ti.f32, ndim=1),
        temporary: ti.types.ndarray(dtype=ti.f32, ndim=1),
        factor: ti.f32,
        offset: ti.f32,
    ):
        for i in temporary:
            temporary[i] = source[i] * factor + offset

    @ti.kernel
    def stage1(
        temporary: ti.types.ndarray(dtype=ti.f32, ndim=1),
        output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in output:
            output[i] = temporary[i] + 1.0

    def make_check(output):
        def check():
            actual = output.to_numpy()
            delta = np.abs(actual - expected)
            return np.allclose(actual, expected, rtol=1.0e-6, atol=1.0e-6), np.max(
                delta
            )

        return check

    array_source = ti.ndarray(ti.f32, shape=n)
    array_temporary = ti.ndarray(ti.f32, shape=n)
    array_output = ti.ndarray(ti.f32, shape=n)
    array_source.from_numpy(source_np)

    def run_array_kernels():
        stage0(array_source, array_temporary, scale, bias)
        stage1(array_temporary, array_output)

    field_source = ti.field(ti.f32, shape=n)
    field_temporary = ti.field(ti.f32, shape=n)
    field_output = ti.field(ti.f32, shape=n)
    field_source.from_numpy(source_np)
    field_source_view = ti.experimental.ndarray_view(field_source)
    field_temporary_view = ti.experimental.ndarray_view(field_temporary)
    field_output_view = ti.experimental.ndarray_view(field_output)

    def run_field_view_kernels():
        stage0(field_source_view, field_temporary_view, scale, bias)
        stage1(field_temporary_view, field_output_view)

    symbolic_source = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "source", ti.f32, ndim=1
    )
    symbolic_temporary = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "temporary", ti.f32, ndim=1
    )
    symbolic_output = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1
    )
    symbolic_scale = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "scale", ti.f32)
    symbolic_bias = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "bias", ti.f32)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        stage0,
        symbolic_source,
        symbolic_temporary,
        symbolic_scale,
        symbolic_bias,
    )
    builder.dispatch(stage1, symbolic_temporary, symbolic_output)
    graph = builder.compile()
    graph_args = {
        "source": array_source,
        "temporary": array_temporary,
        "output": array_output,
        "scale": float(scale),
        "bias": float(bias),
    }

    def run_array_graph():
        graph.run(graph_args)

    field_graph_args = {
        "source": field_source,
        "temporary": field_temporary,
        "output": field_output,
        "scale": float(scale),
        "bias": float(bias),
    }
    field_view_graph_args = {
        "source": field_source_view,
        "temporary": field_temporary_view,
        "output": field_output_view,
        "scale": float(scale),
        "bias": float(bias),
    }

    def run_field_graph():
        graph.run(field_graph_args)

    def run_field_view_graph():
        graph.run(field_view_graph_args)

    available_cases = {
        "ordinary_ndarray": (run_array_kernels, make_check(array_output)),
        "ordinary_dense_field_view": (
            run_field_view_kernels,
            make_check(field_output),
        ),
        "graph_ndarray": (run_array_graph, make_check(array_output)),
        "graph_dense_field": (run_field_graph, make_check(field_output)),
        "graph_dense_field_view": (
            run_field_view_graph,
            make_check(field_output),
        ),
    }
    requested_cases = (
        list(available_cases)
        if args.cases == "all"
        else [name.strip() for name in args.cases.split(",") if name.strip()]
    )
    unknown_cases = set(requested_cases) - set(available_cases)
    if unknown_cases:
        raise ValueError(
            "unknown benchmark cases: " + ", ".join(sorted(unknown_cases))
        )
    cases = []
    for name in requested_cases:
        body, check = available_cases[name]
        cases.append(
            _measure_case(
                name,
                body,
                check,
                warmups=args.warmups,
                repeats=args.repeats,
                batch_size=args.batch_size,
            )
        )
    correct = all(case["correct"] for case in cases)
    result = {
        "arch": args.arch,
        "actual_arch": str(actual_arch),
        "n": n,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "batch_size": args.batch_size,
        "skipped": False,
        "cases": cases,
    }
    result.update(
        finalize_performance_measurement(
            measurement,
            correct=correct,
            reason=None if correct else "result consistency check failed",
        )
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=("cpu", "cuda", "vulkan"), required=True)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--cases",
        default="all",
        help=(
            "comma-separated subset of ordinary_ndarray, "
            "ordinary_dense_field_view, graph_ndarray, graph_dense_field, "
            "graph_dense_field_view"
        ),
    )
    parser.add_argument("--performance", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args)
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return 0 if result.get("skipped") or all(
        case["correct"] for case in result["cases"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
