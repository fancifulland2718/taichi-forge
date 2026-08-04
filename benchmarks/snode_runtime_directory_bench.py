"""LLVM SNode runtime-directory scaling and lifecycle qualification.

This benchmark compares the same dense lookup at a low and a greater-than-
4096 tree-local SNode index, reports the exact tree/runtime-directory memory,
then churns one reusable tree slot. It is a current-implementation scaling
check, not a historical pre-refactor binary comparison.
"""

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpu_idle_guard import (  # noqa: E402
    finalize_performance_measurement,
    prepare_performance_measurement,
)


SCHEMA = "taichi_forge.snode_runtime_directory.v1"


def _summary(samples):
    ordered = sorted(int(value) for value in samples)
    mean = statistics.fmean(ordered)
    p95 = ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]
    return {
        "samples": len(ordered),
        "median_ns": int(statistics.median(ordered)),
        "p95_ns": int(p95),
        "min_ns": ordered[0],
        "max_ns": ordered[-1],
        "cv_pct": statistics.pstdev(ordered) / mean * 100.0 if mean else 0.0,
    }


def _build_dense_tree(ti, place_count, *, collect_stats=True):
    started = time.perf_counter_ns()
    builder = ti.FieldsBuilder()
    dense = builder.dense(ti.i, 1)
    fields = []
    for _ in range(place_count):
        field = ti.field(ti.i32)
        dense.place(field)
        fields.append(field)
    assembled_ns = time.perf_counter_ns() - started
    started = time.perf_counter_ns()
    tree = builder.finalize()
    if collect_stats:
        ti.sync()
    finalized_ns = time.perf_counter_ns() - started
    stats = dict(ti.lang.impl.get_runtime().prog._debug_sparse_snode_tree_stats(tree.id)) if collect_stats else None
    return {
        "tree": tree,
        "fields": fields,
        "assembled_ns": assembled_ns,
        "finalized_ns": finalized_ns,
        "memory": None if stats is None else dict(stats["memory"]),
        "generation": None if stats is None else int(stats["generation"]),
    }


def run_initialized(
    ti,
    *,
    wide_places=4097,
    work_items=262144,
    samples=9,
    repeats=20,
    churn=129,
    concurrent_trees=513,
):
    if wide_places < 4095:
        raise ValueError("wide_places must exercise a tree-local id above 4096")
    if min(work_items, samples, repeats) <= 0 or min(churn, concurrent_trees) < 0:
        raise ValueError("work_items, samples, and repeats must be positive")

    runtime = ti.lang.impl.get_runtime()
    runtime.materialize()
    prog = runtime.prog
    directory_before = dict(prog._debug_snode_runtime_directory_stats())
    concurrent = []
    growth_started = time.perf_counter_ns()
    for _ in range(concurrent_trees):
        current = _build_dense_tree(ti, 1, collect_stats=False)
        concurrent.append(current)
    growth_duration_ns = time.perf_counter_ns() - growth_started
    directory_expanded = dict(prog._debug_snode_runtime_directory_stats())
    if concurrent:
        concurrent[0]["fields"][0][0] = 31
        concurrent[-1]["fields"][0][0] = 47
        concurrent_lookup_correct = concurrent[0]["fields"][0][0] == 31 and concurrent[-1]["fields"][0][0] == 47
    else:
        concurrent_lookup_correct = True
    for current in reversed(concurrent):
        current["tree"].destroy()
    directory_after_growth_retire = dict(prog._debug_snode_runtime_directory_stats())
    small = _build_dense_tree(ti, 1)
    wide = _build_dense_tree(ti, wide_places)
    directory_live = dict(prog._debug_snode_runtime_directory_stats())
    output = ti.ndarray(ti.i32, shape=work_items)

    @ti.kernel
    def read_field(
        source: ti.template(),
        destination: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in destination:
            destination[i] = source[0] + i

    small_field = small["fields"][-1]
    wide_field = wide["fields"][-1]
    small_field[0] = 17
    wide_field[0] = 29
    read_field(small_field, output)
    read_field(wide_field, output)
    ti.sync()

    def sample(source):
        values = []
        for _ in range(samples):
            started = time.perf_counter_ns()
            for _ in range(repeats):
                read_field(source, output)
            ti.sync()
            values.append((time.perf_counter_ns() - started) // repeats)
        return values

    small_samples = sample(small_field)
    wide_samples = sample(wide_field)
    wide_output = output.to_numpy()
    lookup_correct = int(wide_output[0]) == 29 and int(wide_output[-1]) == 29 + work_items - 1
    small_summary = _summary(small_samples)
    wide_summary = _summary(wide_samples)

    small_tree_id = small["tree"].id
    wide_tree_id = wide["tree"].id
    small["tree"].destroy()
    wide["tree"].destroy()
    directory_retired = dict(prog._debug_snode_runtime_directory_stats())

    churn_samples = []
    churn_ids = []
    churn_generations = []
    for _ in range(churn):
        started = time.perf_counter_ns()
        current = _build_dense_tree(ti, 1)
        churn_ids.append(current["tree"].id)
        churn_generations.append(current["generation"])
        current["tree"].destroy()
        ti.sync()
        churn_samples.append(time.perf_counter_ns() - started)
    directory_final = dict(prog._debug_snode_runtime_directory_stats())

    runtime_state_small = small["memory"]["runtime_state_reserved_bytes"]
    runtime_state_wide = wide["memory"]["runtime_state_reserved_bytes"]
    lifecycle_correct = (
        directory_live["available"]
        and directory_live["active_tree_count"] == directory_before["active_tree_count"] + 2
        and directory_retired["active_tree_count"] == directory_before["active_tree_count"]
        and directory_final["active_tree_count"] == directory_before["active_tree_count"]
        and directory_final["capacity"] == directory_live["capacity"]
        and concurrent_lookup_correct
        and directory_expanded["active_tree_count"] == directory_before["active_tree_count"] + concurrent_trees
        and directory_after_growth_retire["active_tree_count"] == directory_before["active_tree_count"]
        and directory_after_growth_retire["capacity"] == directory_expanded["capacity"]
        and len(set(churn_ids)) <= 1
        and all(left < right for left, right in zip(churn_generations, churn_generations[1:]))
    )
    memory_correct = (
        runtime_state_small == 40 + 3 * 48
        and runtime_state_wide == 40 + (wide_places + 2) * 48
        and directory_live["reserved_bytes"] == directory_live["capacity"] * 8
    )
    return {
        "schema": SCHEMA,
        "arch": "cpu" if ti.lang.impl.current_cfg().arch == ti.cpu else "cuda",
        "configuration": {
            "wide_places": wide_places,
            "wide_runtime_nodes": wide_places + 2,
            "work_items": work_items,
            "samples": samples,
            "repeats": repeats,
            "churn": churn,
            "concurrent_trees": concurrent_trees,
        },
        "lookup": {
            "small": small_summary,
            "wide": wide_summary,
            "wide_to_small_median_ratio": (
                wide_summary["median_ns"] / small_summary["median_ns"] if small_summary["median_ns"] else None
            ),
            "interpretation": (
                "same current runtime, low versus greater-than-4096 tree-local id; "
                "this detects lookup scaling but is not a historical A/B"
            ),
        },
        "materialization": {
            "small_assemble_ns": small["assembled_ns"],
            "small_finalize_ns": small["finalized_ns"],
            "wide_assemble_ns": wide["assembled_ns"],
            "wide_finalize_ns": wide["finalized_ns"],
            "directory_growth_ns": growth_duration_ns,
        },
        "memory": {
            "small_tree": small["memory"],
            "wide_tree": wide["memory"],
            "directory_before": directory_before,
            "directory_expanded": directory_expanded,
            "directory_after_growth_retire": directory_after_growth_retire,
            "directory_live": directory_live,
            "directory_retired": directory_retired,
            "directory_final": directory_final,
        },
        "lifecycle": {
            "retired_tree_ids": [small_tree_id, wide_tree_id],
            "churn_tree_ids": churn_ids,
            "churn_generations": churn_generations,
            "churn_duration": _summary(churn_samples) if churn_samples else None,
            "tree_slot_reused": len(set(churn_ids)) <= 1,
            "generation_strictly_increasing": all(
                left < right for left, right in zip(churn_generations, churn_generations[1:])
            ),
        },
        "correctness": {
            "lookup": lookup_correct,
            "lifecycle": lifecycle_correct,
            "memory_accounting": memory_correct,
        },
        "correct": lookup_correct and lifecycle_correct and memory_correct,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--wide-places", type=int, default=4097)
    parser.add_argument("--work-items", type=int, default=262144)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--churn", type=int, default=129)
    parser.add_argument("--concurrent-trees", type=int, default=513)
    parser.add_argument("--performance", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    measurement = prepare_performance_measurement(args.arch, requested=args.performance)

    import taichi_forge as ti

    ti.init(
        arch=ti.cpu if args.arch == "cpu" else ti.cuda,
        enable_fallback=False,
        offline_cache=False,
    )
    try:
        result = run_initialized(
            ti,
            wide_places=args.wide_places,
            work_items=args.work_items,
            samples=args.samples,
            repeats=args.repeats,
            churn=args.churn,
            concurrent_trees=args.concurrent_trees,
        )
        result.update(finalize_performance_measurement(measurement, correct=result["correct"]))
    finally:
        ti.reset()
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    return 0 if result["correct"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
