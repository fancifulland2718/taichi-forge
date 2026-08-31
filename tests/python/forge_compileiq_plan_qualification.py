"""Fresh-process qualification for Forge task and Graph plan search.

The parent launches one AB or BA pair per fresh process.  Each route block is
calibrated independently to the requested minimum duration.  Compile/build
time is retained as diagnostic evidence and never participates in admission.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np


_SCOPES = (
    "kernel_two_stage_1m",
    "graph_dispatch_small_4k",
    "graph_bandwidth_large_1m",
    "graph_compute_64k",
)
_RESULT_PREFIX = "FORGE_QUALIFICATION_RESULT="


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timed_block(invoke, repetitions, minimum_block_ms):
    repetitions = max(1, int(repetitions))
    while True:
        start = time.perf_counter_ns()
        for _ in range(repetitions):
            invoke()
        import taichi_forge as ti

        ti.sync()
        elapsed_ns = time.perf_counter_ns() - start
        elapsed_ms = elapsed_ns / 1.0e6
        if elapsed_ms >= minimum_block_ms:
            return {
                "repetitions": repetitions,
                "elapsed_ms": elapsed_ms,
                "ns_per_replay": elapsed_ns / repetitions,
                "minimum_block_ms": minimum_block_ms,
                "satisfied": True,
            }
        scale = max(2, math.ceil(minimum_block_ms / max(elapsed_ms, 0.001) * 1.08))
        repetitions = min(2_000_000, repetitions * scale)
        if repetitions == 2_000_000 and elapsed_ms < minimum_block_ms:
            raise RuntimeError("timing block could not reach its minimum duration")


def _calibrate(invoke, minimum_block_ms):
    return _timed_block(invoke, 1, minimum_block_ms)["repetitions"]


def _measure_pair(routes, order, minimum_block_ms):
    for invoke in routes.values():
        for _ in range(8):
            invoke()
    import taichi_forge as ti

    ti.sync()
    calibration = {name: _calibrate(invoke, minimum_block_ms) for name, invoke in routes.items()}
    blocks = {}
    for name in order:
        blocks[name] = _timed_block(routes[name], calibration[name], minimum_block_ms)
    return blocks


def _graph_worker(scope, order, minimum_block_ms):
    import taichi_forge as ti
    from taichi_forge._lib import core as ti_core

    count, compute_rounds = {
        "graph_dispatch_small_4k": (1 << 12, 0),
        "graph_bandwidth_large_1m": (1 << 20, 0),
        "graph_compute_64k": (1 << 16, 24),
    }[scope]
    modulus = 1_000_003
    stage_constants = ((17, 3), (29, 5), (43, 7), (61, 11))

    @ti.func
    def work(value, multiplier, increment):
        value = (value * multiplier + increment) % modulus
        for iteration in ti.static(range(compute_rounds)):
            value = (value * (97 + iteration % 13) + increment + iteration) % modulus
        return value

    @ti.kernel
    def stage_one(
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in source:
            output[i] = work(source[i], 17, 3)

    @ti.kernel
    def stage_two(
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        input_values: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in source:
            output[i] = work(input_values[i], 29, 5)

    @ti.kernel
    def stage_three(
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        input_values: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in source:
            output[i] = work(input_values[i], 43, 7)

    @ti.kernel
    def stage_four(
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        input_values: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in source:
            output[i] = work(input_values[i], 61, 11)

    symbolic = {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.i32, ndim=1)
        for name in ("source", "temporary_a", "temporary_b", "temporary_c", "output")
    }

    def build(recipe):
        previous = os.environ.get("TAICHI_FORGE_INTERNAL_MAP_FUSION")
        os.environ["TAICHI_FORGE_INTERNAL_MAP_FUSION"] = recipe
        started = time.perf_counter_ns()
        try:
            builder = ti.graph.GraphBuilder()
            builder.dispatch(stage_one, symbolic["source"], symbolic["temporary_a"])
            builder.dispatch(
                stage_two,
                symbolic["source"],
                symbolic["temporary_a"],
                symbolic["temporary_b"],
            )
            builder.dispatch(
                stage_three,
                symbolic["source"],
                symbolic["temporary_b"],
                symbolic["temporary_c"],
            )
            builder.dispatch(
                stage_four,
                symbolic["source"],
                symbolic["temporary_c"],
                symbolic["output"],
            )
            graph = builder.compile()
        finally:
            if previous is None:
                os.environ.pop("TAICHI_FORGE_INTERNAL_MAP_FUSION", None)
            else:
                os.environ["TAICHI_FORGE_INTERNAL_MAP_FUSION"] = previous
        return graph, (time.perf_counter_ns() - started) / 1.0e6

    baseline_graph, baseline_build_ms = build("baseline")
    candidate_graph, candidate_build_ms = build("exact-v1:0,1,2,3")
    physical = {
        "baseline": baseline_graph.physical_plan()["physical_dispatch_count"],
        "candidate": candidate_graph.physical_plan()["physical_dispatch_count"],
    }
    if physical != {"baseline": 4, "candidate": 1}:
        raise RuntimeError(f"exact Graph topology did not materialize: {physical}")

    source_np = np.arange(count, dtype=np.int32) % 1009
    expected = source_np.astype(np.int64)
    for multiplier, increment in stage_constants:
        expected = (expected * multiplier + increment) % modulus
        for iteration in range(compute_rounds):
            expected = (expected * (97 + iteration % 13) + increment + iteration) % modulus
    expected = expected.astype(np.int32)

    arguments = {}
    routes = {}
    graphs = {"baseline": baseline_graph, "candidate": candidate_graph}
    for name, graph in graphs.items():
        arrays = {
            item: ti.ndarray(ti.i32, shape=count)
            for item in ("source", "temporary_a", "temporary_b", "temporary_c", "output")
        }
        arrays["source"].from_numpy(source_np)
        arguments[name] = arrays
        routes[name] = lambda graph=graph, arrays=arrays: graph.run(arrays)

    for name in routes:
        routes[name]()
    ti.sync()
    correctness = {name: bool(np.array_equal(arguments[name]["output"].to_numpy(), expected)) for name in routes}
    memory_after_warmup = dict(ti_core.get_device_memory_pool_stats())
    blocks = _measure_pair(routes, order, minimum_block_ms)
    memory_after_timing = dict(ti_core.get_device_memory_pool_stats())
    for name in routes:
        routes[name]()
    ti.sync()
    correctness.update(
        {
            name: correctness[name] and bool(np.array_equal(arguments[name]["output"].to_numpy(), expected))
            for name in routes
        }
    )
    return {
        "scope": scope,
        "kind": "graph_partition_plan",
        "count": count,
        "compute_rounds_per_stage": compute_rounds,
        "order": order,
        "blocks": blocks,
        "correctness": correctness,
        "memory_stable": memory_after_warmup == memory_after_timing,
        "memory_after_warmup": memory_after_warmup,
        "memory_after_timing": memory_after_timing,
        "physical_dispatches": physical,
        "compile_build_ms_diagnostic": {
            "baseline": baseline_build_ms,
            "candidate": candidate_build_ms,
        },
        "candidate_materialization": "exact-v1:0,1,2,3",
    }


def _kernel_worker(order, minimum_block_ms):
    import taichi_forge as ti
    from taichi_forge._lib import core as ti_core
    from taichi_forge.lang._offload_execution_plan import (
        _OffloadExecutionPlan,
        _bind_offload_execution_plan,
    )

    count = 1 << 20

    @ti.kernel
    def two_stage(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(count):
            out[i] = i * 2
        for i in range(count):
            out[i] = out[i] * 3 + 1

    probe = ti.ndarray(ti.i32, shape=count)
    started = time.perf_counter_ns()
    baseline_plan = _OffloadExecutionPlan.from_task_manifests(two_stage.task_manifest(probe))
    baseline_compile_ms = (time.perf_counter_ns() - started) / 1.0e6
    ranges = tuple(task for task in baseline_plan.tasks if task.task_kind == "range_for")
    if len(ranges) != 2:
        raise RuntimeError("qualification kernel did not retain two physical tasks")
    candidate_plan = baseline_plan.replace_task(
        ranges[0].task_index,
        workgroup_size=64,
        range_work_per_thread_target=2,
    ).replace_task(
        ranges[1].task_index,
        workgroup_size=256,
        range_work_per_thread_target=4,
    )
    baseline = _bind_offload_execution_plan(two_stage, baseline_plan)
    candidate = _bind_offload_execution_plan(two_stage, candidate_plan)
    started = time.perf_counter_ns()
    candidate_report = candidate.report(probe)
    candidate_compile_ms = (time.perf_counter_ns() - started) / 1.0e6

    outputs = {
        "baseline": ti.ndarray(ti.i32, shape=count),
        "candidate": ti.ndarray(ti.i32, shape=count),
    }
    routes = {
        "baseline": lambda: baseline(outputs["baseline"]),
        "candidate": lambda: candidate(outputs["candidate"]),
    }
    expected = np.arange(count, dtype=np.int32) * 6 + 1
    for invoke in routes.values():
        invoke()
    ti.sync()
    correctness = {name: bool(np.array_equal(output.to_numpy(), expected)) for name, output in outputs.items()}
    memory_after_warmup = dict(ti_core.get_device_memory_pool_stats())
    blocks = _measure_pair(routes, order, minimum_block_ms)
    memory_after_timing = dict(ti_core.get_device_memory_pool_stats())
    for invoke in routes.values():
        invoke()
    ti.sync()
    correctness.update(
        {
            name: correctness[name] and bool(np.array_equal(output.to_numpy(), expected))
            for name, output in outputs.items()
        }
    )
    return {
        "scope": "kernel_two_stage_1m",
        "kind": "kernel_offload_execution_plan",
        "count": count,
        "order": order,
        "blocks": blocks,
        "correctness": correctness,
        "memory_stable": memory_after_warmup == memory_after_timing,
        "memory_after_warmup": memory_after_warmup,
        "memory_after_timing": memory_after_timing,
        "compile_build_ms_diagnostic": {
            "baseline": baseline_compile_ms,
            "candidate": candidate_compile_ms,
        },
        "baseline_plan_identity": baseline_plan.identity,
        "candidate_plan_identity": candidate_plan.identity,
        "candidate_compilation_identity": candidate_plan.compilation_identity,
        "candidate_tasks": tuple(
            {
                "task_index": task.task_index,
                "selected_block_size": task.selected_block_size,
                "requested_range_work_per_thread_target": (task.requested_range_work_per_thread_target),
            }
            for task in candidate_report.tasks
            if task.task_type == "range_for"
        ),
    }


def _worker(args):
    import taichi_forge as ti
    from taichi_forge._compileiq_opaque import _validated_compileiq_capability
    from taichi_forge._lib import core as ti_core

    ti.init(arch=ti.cuda, offline_cache=False)
    capability, _, _, source_lock = _validated_compileiq_capability()
    order = tuple(args.order.split(","))
    if order not in (("baseline", "candidate"), ("candidate", "baseline")):
        raise ValueError("worker order must be baseline,candidate or candidate,baseline")
    result = (
        _kernel_worker(order, args.minimum_block_ms)
        if args.scope == "kernel_two_stage_1m"
        else _graph_worker(args.scope, order, args.minimum_block_ms)
    )
    result.update(
        {
            "pid": os.getpid(),
            "timestamp_ns": time.time_ns(),
            "runtime_commit": str(ti_core.get_commit_hash()).lower(),
            "compileiq_capability": dict(capability),
            "compileiq_python_source_lock": source_lock,
        }
    )
    print(_RESULT_PREFIX + json.dumps(result, sort_keys=True))


def _cv(values):
    mean = statistics.fmean(values)
    return 0.0 if mean == 0.0 else statistics.pstdev(values) / mean


def _aggregate_scope(scope, workers, minimum_block_ms):
    ratios = tuple(
        worker["blocks"]["candidate"]["ns_per_replay"] / worker["blocks"]["baseline"]["ns_per_replay"]
        for worker in workers
    )
    by_order = {}
    for label, expected in (
        ("ab", ["baseline", "candidate"]),
        ("ba", ["candidate", "baseline"]),
    ):
        selected = [ratio for ratio, worker in zip(ratios, workers) if worker["order"] == expected]
        by_order[label] = statistics.median(selected)
    order_drift = abs(by_order["ab"] - by_order["ba"])
    all_blocks = tuple(block for worker in workers for block in worker["blocks"].values())
    correctness = all(all(worker["correctness"].values()) for worker in workers)
    memory_stable = all(worker["memory_stable"] for worker in workers)
    worst_positive = max(ratios) < 1.0
    status = "qualified_positive" if correctness and memory_stable and worst_positive else "negative_retained"
    return {
        "scope": scope,
        "status": status,
        "candidate_over_baseline_process_ratios": ratios,
        "median_candidate_over_baseline": statistics.median(ratios),
        "worst_candidate_over_baseline": max(ratios),
        "best_candidate_over_baseline": min(ratios),
        "ratio_cv": _cv(ratios),
        "order_medians": by_order,
        "order_drift": order_drift,
        "correctness": correctness,
        "memory_stable": memory_stable,
        "worst_positive": worst_positive,
        "minimum_observed_block_ms": min(block["elapsed_ms"] for block in all_blocks),
        "minimum_block_satisfied": all(
            block["satisfied"] and block["elapsed_ms"] >= minimum_block_ms for block in all_blocks
        ),
        "processes": len(workers),
        "worker_evidence": workers,
    }


def _parent(args):
    script = Path(__file__).resolve()
    all_workers = {}
    for scope in _SCOPES:
        workers = []
        for process_index in range(args.processes):
            order = "baseline,candidate" if process_index % 2 == 0 else "candidate,baseline"
            command = [
                sys.executable,
                str(script),
                "--worker",
                "--scope",
                scope,
                "--order",
                order,
                "--minimum-block-ms",
                str(args.minimum_block_ms),
            ]
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                env=os.environ.copy(),
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"worker failed for {scope} process {process_index}:\n" + completed.stdout + completed.stderr
                )
            encoded = next(
                (
                    line[len(_RESULT_PREFIX) :]
                    for line in reversed(completed.stdout.splitlines())
                    if line.startswith(_RESULT_PREFIX)
                ),
                None,
            )
            if encoded is None:
                raise RuntimeError("worker did not emit machine-readable evidence")
            workers.append(json.loads(encoded))
            print(
                f"{scope}: {process_index + 1}/{args.processes}",
                flush=True,
            )
        all_workers[scope] = workers

    summaries = {
        scope: _aggregate_scope(
            scope,
            workers,
            args.minimum_block_ms,
        )
        for scope, workers in all_workers.items()
    }
    negative_scopes = tuple(scope for scope, summary in summaries.items() if summary["status"] == "negative_retained")
    runtime_commits = sorted({worker["runtime_commit"] for workers in all_workers.values() for worker in workers})
    capability_ids = sorted(
        {worker["compileiq_capability"]["capability_id"] for workers in all_workers.values() for worker in workers}
    )
    report = {
        "schema": "taichi_forge.compileiq-plan-qualification.v1",
        "evidence_class": "independent_fresh_process_ab_ba",
        "generated_at_unix_ns": time.time_ns(),
        "policy": {
            "fresh_processes_per_scope": args.processes,
            "orders": ("ab", "ba"),
            "minimum_block_ms": args.minimum_block_ms,
            "correctness_required": True,
            "memory_stable_required": True,
            "worst_positive_required_for_admission": True,
            "compile_time": "diagnostic_only_not_a_gate",
        },
        "provenance": {
            "runtime_commits": runtime_commits,
            "compileiq_capability_ids": capability_ids,
            "qualification_script": str(script),
            "qualification_script_sha256": _sha256(script),
        },
        "scopes": summaries,
        "negative_cluster_review": {
            "negative_scopes": negative_scopes,
            "negative_count": len(negative_scopes),
            "status": ("review_required" if len(negative_scopes) >= 3 else "no_negative_cluster"),
        },
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "summaries": summaries}, indent=2))


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--scope", choices=_SCOPES)
    parser.add_argument("--order", default="baseline,candidate")
    parser.add_argument("--processes", type=int, default=10)
    parser.add_argument("--minimum-block-ms", type=float, default=250.0)
    parser.add_argument(
        "--output",
        default=(".agent/experiments/forge-compileiq-r11-r12/qualification.json"),
    )
    args = parser.parse_args()
    if args.worker and args.scope is None:
        parser.error("--worker requires --scope")
    if args.processes < 8 or args.processes % 2:
        parser.error("--processes must be an even integer >= 8")
    if args.minimum_block_ms < 250.0:
        parser.error("--minimum-block-ms must be at least 250")
    return args


if __name__ == "__main__":
    parsed = _parse_args()
    if parsed.worker:
        _worker(parsed)
    else:
        _parent(parsed)
