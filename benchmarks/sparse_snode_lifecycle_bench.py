"""Sparse SNodeTree lifecycle and phase-telemetry baseline.

The benchmark keeps one Program alive while repeatedly creating, using,
deactivating, and destroying a pointer+dense SNodeTree.  It deliberately
reports unavailable tree-local counters as gaps instead of attributing global
allocator or process memory to one tree.
"""

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpu_idle_guard import (  # noqa: E402
    finalize_performance_measurement,
    prepare_performance_measurement,
)


SCHEMA = "taichi_forge.sparse_snode_lifecycle.v2"
PHASES = (
    "create",
    "materialize",
    "compile_empty_workload",
    "first_activate",
    "cold_struct_for",
    "warm_struct_for",
    "body_direct_lookup",
    "clear",
    "deactivate_gc",
    "post_deactivate_verify",
    "destroy",
)
_MEMORY_FIELDS = (
    "live_resources",
    "retiring_resources",
    "inflight_resources",
    "host_requested_live_bytes",
    "host_raw_bytes",
    "host_capacity_bytes",
    "device_requested_live_bytes",
    "device_raw_bytes",
    "device_cached_bytes",
    "cuda_mempool_reserved_bytes",
    "cuda_mempool_used_bytes",
)
_RUNTIME_COUNTERS = {
    "kernel_submissions": ("submission", "kernel_submissions"),
    "failed_submissions": ("submission", "failed_submissions"),
    "program_syncs": ("synchronization", "program_syncs"),
    "program_sync_wait_ns": ("synchronization", "program_sync_wait_ns"),
    "backend_waits": ("synchronization", "backend_waits"),
    "backend_wait_ns": ("synchronization", "backend_wait_ns"),
    "host_to_device_bytes": ("transfer", "host_to_device_bytes"),
    "device_to_host_bytes": ("transfer", "device_to_host_bytes"),
    "device_to_device_bytes": ("transfer", "device_to_device_bytes"),
}


def _arch_name(ti) -> str:
    arch = ti.lang.impl.current_cfg().arch
    if arch == ti.cpu:
        return "cpu"
    if arch == ti.cuda:
        return "cuda"
    if arch == ti.vulkan:
        return "vulkan"
    return str(arch)


def _runtime_snapshot(ti) -> dict:
    runtime = asdict(ti.runtime.stats())
    pools = ti.tools.memory_pool_stats()
    prog = ti.lang.impl.get_runtime().prog
    active_tree_ids = list(prog.get_active_snode_tree_ids())
    tree_stats = {}
    for tree_id in active_tree_ids:
        snapshot = dict(prog._debug_sparse_snode_tree_stats(tree_id))
        snapshot["memory"] = dict(snapshot["memory"])
        snapshot["listgen"] = dict(snapshot["listgen"])
        snapshot["listgen"]["totals"] = dict(
            snapshot["listgen"]["totals"]
        )
        snapshot["listgen"]["nodes"] = [
            dict(node) for node in snapshot["listgen"]["nodes"]
        ]
        tree_stats[str(tree_id)] = snapshot
    return {
        "runtime_statistics_schema_version": runtime["schema_version"],
        "runtime_counters": {
            name: runtime[section][field]
            for name, (section, field) in _RUNTIME_COUNTERS.items()
        },
        "memory": {
            name: runtime["memory"][name] for name in _MEMORY_FIELDS
        },
        "allocator": {
            "host": {
                key: value
                for key, value in pools["host"].items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            },
            "device": {
                key: value
                for key, value in pools["device"].items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            },
        },
        "lifecycle": {
            "active_snode_tree_ids": active_tree_ids,
            "snode_trees": tree_stats,
            "snode_field_mappings": int(
                prog._debug_snode_field_mapping_count()
            ),
            "kernel_definitions": int(prog._debug_kernel_definition_count()),
            "kernel_registrations": int(
                prog._debug_kernel_registration_count()
            ),
        },
    }


def _numeric_delta(before: dict, after: dict) -> dict:
    result = {}
    for key in sorted(before.keys() & after.keys()):
        left = before[key]
        right = after[key]
        if (
            isinstance(left, (int, float))
            and not isinstance(left, bool)
            and isinstance(right, (int, float))
            and not isinstance(right, bool)
        ):
            result[key] = right - left
        else:
            result[key] = None
    return result


def _phase_sample(ti, callback) -> dict:
    before = _runtime_snapshot(ti)
    started = time.perf_counter_ns()
    callback()
    ti.sync()
    duration_ns = time.perf_counter_ns() - started
    after = _runtime_snapshot(ti)
    before_trees = before["lifecycle"]["snode_trees"]
    after_trees = after["lifecycle"]["snode_trees"]
    return {
        "duration_ns": duration_ns,
        "runtime_counter_delta": _numeric_delta(
            before["runtime_counters"], after["runtime_counters"]
        ),
        "memory_before": before["memory"],
        "memory_after": after["memory"],
        "memory_delta": _numeric_delta(
            before["memory"], after["memory"]
        ),
        "allocator_delta": {
            side: _numeric_delta(
                before["allocator"][side], after["allocator"][side]
            )
            for side in ("host", "device")
        },
        "lifecycle_before": before["lifecycle"],
        "lifecycle_after": after["lifecycle"],
        "tree_memory_delta": {
            tree_id: _numeric_delta(
                before_trees[tree_id]["memory"],
                after_trees[tree_id]["memory"],
            )
            for tree_id in sorted(before_trees.keys() & after_trees.keys())
        },
        "tree_listgen_delta": {
            tree_id: _numeric_delta(
                before_trees[tree_id]["listgen"]["totals"],
                after_trees[tree_id]["listgen"]["totals"],
            )
            for tree_id in sorted(before_trees.keys() & after_trees.keys())
        },
    }


def _summary_ns(samples: list[int]) -> dict:
    ordered = sorted(samples)
    mean = statistics.fmean(ordered)
    p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "samples": len(ordered),
        "median_ns": int(statistics.median(ordered)),
        "p95_ns": int(ordered[p95_index]),
        "min_ns": int(ordered[0]),
        "max_ns": int(ordered[-1]),
        "cv_pct": (
            float(statistics.pstdev(ordered) / mean * 100.0)
            if mean
            else 0.0
        ),
    }


def _snode_delta(baseline: dict, sample: dict) -> dict:
    return {
        "active_snode_trees": (
            len(sample["active_snode_tree_ids"])
            - len(baseline["active_snode_tree_ids"])
        ),
        "snode_field_mappings": (
            sample["snode_field_mappings"]
            - baseline["snode_field_mappings"]
        ),
    }


def _executable_delta(baseline: dict, sample: dict) -> dict:
    return {
        "kernel_definitions": (
            sample["kernel_definitions"] - baseline["kernel_definitions"]
        ),
        "kernel_registrations": (
            sample["kernel_registrations"]
            - baseline["kernel_registrations"]
        ),
    }


def _monotonic_growth(values: list[int]) -> bool:
    return (
        len(values) > 1
        and all(right >= left for left, right in zip(values, values[1:]))
        and any(right > left for left, right in zip(values, values[1:]))
    )


def run_initialized(
    ti,
    *,
    iterations: int = 100,
    root_blocks: int = 64,
    block_size: int = 8,
    active_blocks: int = 8,
) -> dict:
    """Run the lifecycle workload in the already initialized Program."""
    if min(iterations, root_blocks, block_size, active_blocks) <= 0:
        raise ValueError("iterations and sparse dimensions must be positive")
    if active_blocks > root_blocks:
        raise ValueError("active_blocks cannot exceed root_blocks")

    accumulator = ti.field(dtype=ti.i64, shape=())

    @ti.kernel
    def clear_accumulator():
        accumulator[None] = 0

    @ti.kernel
    def activate(
        field: ti.template(), cycle: ti.i32, enabled: ti.i32
    ):
        for block in range(active_blocks):
            for offset in ti.static(range(block_size)):
                if enabled:
                    index = block * block_size + offset
                    field[index] = cycle + index + 1

    @ti.kernel
    def struct_reduce(field: ti.template()):
        for index in field:
            ti.atomic_add(accumulator[None], ti.cast(field[index], ti.i64))

    @ti.kernel
    def direct_lookup_body(field: ti.template()):
        for index in range(active_blocks * block_size):
            ti.atomic_add(accumulator[None], ti.cast(field[index], ti.i64))

    @ti.kernel
    def clear_values(field: ti.template()):
        for index in field:
            field[index] = 0

    # Finalize the persistent scalar and kernel definitions before taking the
    # lifecycle baseline. Per-tree template specializations are exercised in
    # compile_empty_workload and retired by SNodeTree.destroy().
    ti.lang.impl.get_runtime().materialize()
    clear_accumulator()
    ti.sync()

    ti.lang.impl.get_runtime().prog._debug_reset_sparse_listgen_stats()
    baseline = _runtime_snapshot(ti)
    phase_samples = {phase: [] for phase in PHASES}
    iteration_records = []
    after_destroy_snapshots = []
    tree_identities = []

    for cycle in range(iterations):
        holder = {}

        def create():
            field = ti.field(dtype=ti.i32)
            builder = ti.FieldsBuilder()
            pointer_kwargs = (
                {"vk_max_active": root_blocks}
                if _arch_name(ti) in ("cuda", "vulkan")
                else {}
            )
            pointer = builder.pointer(ti.i, root_blocks, **pointer_kwargs)
            pointer.dense(ti.i, block_size).place(field)
            holder.update(
                field=field,
                builder=builder,
                pointer=pointer,
                tree=builder.finalize(),
            )

        record = {"index": cycle, "phases": {}, "checks": {}}

        def measure(name, callback):
            sample = _phase_sample(ti, callback)
            record["phases"][name] = sample
            phase_samples[name].append(sample)

        measure("create", create)
        field = holder["field"]
        pointer = holder["pointer"]
        tree = holder["tree"]
        identity = [int(tree.id), int(tree.generation)]
        record["tree_identity"] = identity
        tree_identities.append(identity)

        measure("materialize", ti.lang.impl.get_runtime().materialize)

        def compile_empty_workload():
            activate(field, cycle, 0)
            clear_accumulator()
            struct_reduce(field)
            clear_accumulator()
            direct_lookup_body(field)
            clear_values(field)

        measure("compile_empty_workload", compile_empty_workload)
        measure("first_activate", lambda: activate(field, cycle, 1))

        def traverse():
            clear_accumulator()
            struct_reduce(field)

        expected = (
            active_blocks * block_size * (cycle + 1)
            + (active_blocks * block_size)
            * (active_blocks * block_size - 1)
            // 2
        )
        measure("cold_struct_for", traverse)
        record["checks"]["cold_struct_for"] = int(accumulator[None])
        measure("warm_struct_for", traverse)
        record["checks"]["warm_struct_for"] = int(accumulator[None])

        def body():
            clear_accumulator()
            direct_lookup_body(field)

        measure("body_direct_lookup", body)
        record["checks"]["body_direct_lookup"] = int(accumulator[None])
        if any(value != expected for value in record["checks"].values()):
            raise RuntimeError(
                f"sparse lifecycle result mismatch at cycle {cycle}: "
                f"{record['checks']} != {expected}"
            )

        measure("clear", lambda: clear_values(field))
        measure("deactivate_gc", pointer.deactivate_all)
        measure("post_deactivate_verify", traverse)
        record["checks"]["post_deactivate_verify"] = int(accumulator[None])
        if record["checks"]["post_deactivate_verify"] != 0:
            raise RuntimeError(
                f"deactivation left active values at cycle {cycle}"
            )

        measure("destroy", tree.destroy)
        after_destroy = _runtime_snapshot(ti)
        after_destroy_snapshots.append(after_destroy)
        record["after_destroy"] = after_destroy
        iteration_records.append(record)

        del field, pointer, tree
        holder.clear()

    final = _runtime_snapshot(ti)
    snode_deltas = [
        _snode_delta(baseline["lifecycle"], item["lifecycle"])
        for item in after_destroy_snapshots
    ]
    executable_deltas = [
        _executable_delta(baseline["lifecycle"], item["lifecycle"])
        for item in after_destroy_snapshots
    ]
    snode_growth_values = [
        sum(max(0, value) for value in item.values())
        for item in snode_deltas
    ]
    executable_growth_values = [
        sum(max(0, value) for value in item.values())
        for item in executable_deltas
    ]
    arch_name = _arch_name(ti)
    live_memory_field = {
        "cpu": "host_requested_live_bytes",
        "cuda": "device_requested_live_bytes",
    }.get(arch_name)
    baseline_live_memory = (
        None
        if live_memory_field is None
        else baseline["memory"][live_memory_field]
    )
    live_memory_deltas = [
        (
            None
            if live_memory_field is None
            or baseline_live_memory is None
            or item["memory"][live_memory_field] is None
            else (
                item["memory"][live_memory_field] - baseline_live_memory
            )
        )
        for item in after_destroy_snapshots
    ]
    exact_live_memory_deltas = [
        value for value in live_memory_deltas if value is not None
    ]
    lifecycle_warmup_cycles = 1 if len(exact_live_memory_deltas) > 1 else 0
    steady_live_memory_deltas = exact_live_memory_deltas[
        lifecycle_warmup_cycles:
    ]
    snode_recovered = all(
        all(value == 0 for value in item.values())
        for item in snode_deltas
    )
    live_memory_recovered = (
        None
        if not exact_live_memory_deltas
        else all(value == 0 for value in exact_live_memory_deltas)
    )
    generations = [identity[1] for identity in tree_identities]
    tree_ids = [identity[0] for identity in tree_identities]

    return {
        "schema": SCHEMA,
        "schema_version": 2,
        "arch": arch_name,
        "correct": True,
        "config": {
            "iterations": iterations,
            "layout": "pointer_dense_1d",
            "root_blocks": root_blocks,
            "block_size": block_size,
            "active_blocks": active_blocks,
            "active_cells": active_blocks * block_size,
        },
        "telemetry_contract": {
            "runtime_statistics_schema_version": (
                baseline["runtime_statistics_schema_version"]
            ),
            "memory_scope": "program_aggregate_plus_tree_inventory",
            "program_memory_scope": "program_aggregate",
            "tree_memory_scope": (
                "tree_owned allocations plus backend-available logical "
                "runtime resources"
            ),
            "phase_scope": "callback_plus_backend_sync",
            "cold_struct_for_scope": (
                "precompiled struct-for listgen plus body plus sync after "
                "first activation"
            ),
            "warm_struct_for_scope": (
                "same precompiled struct-for on unchanged topology plus sync"
            ),
            "body_direct_lookup_scope": (
                "precompiled direct lookup range body plus sync"
            ),
            "per_tree_memory_available": True,
            "per_tree_listgen_decisions_available": True,
            "per_tree_memory_contract": {
                "all_backends_exact": [
                    "tree_owned_reserved_bytes",
                    "root_reserved_bytes",
                    "sparse_pool_reserved_bytes",
                ],
                "llvm_exact": [
                    "runtime_metadata_requested_bytes",
                    "direct_ambient_requested_bytes",
                    "allocator_payload_reserved_bytes",
                    "allocator_payload_used_bytes",
                    "allocator_bookkeeping_reserved_bytes",
                    "active_list_reserved_bytes",
                    "active_list_used_bytes",
                    "allocator_in_use_elements",
                    "allocator_free_elements",
                    "allocator_recycled_elements",
                ],
                "program_shared": [
                    "shared_listgen_workspace_reserved_bytes",
                ],
                "overlap_rule": (
                    "LLVM logical runtime resources may be backed by the "
                    "reported tree-owned CUDA pool or Program CPU reuse "
                    "pool and must not be added to tree_owned_reserved_bytes"
                ),
            },
            "unavailable": [
                "per_tree_payload_committed_peak_bytes",
                "per_tree_active_list_peak_bytes",
                "per_tree_reclaimable_or_releasable_bytes",
                "llvm_candidate_slots_dispatched",
                "gfx_tree_local_metadata_payload_active_list_split",
            ],
            "listgen_contract": {
                "all_backends_exact": [
                    "requests",
                    "rebuilds",
                    "reuse_hits",
                    "invalidations",
                    "last_rebuild_reason",
                ],
                "gfx_exact": [
                    "candidate_slots_dispatched",
                    "resident_evictions",
                ],
                "cpu_exact": [
                    "scanned_elements",
                    "emitted_elements",
                    "serial_rebuilds",
                    "parallel_rebuilds",
                ],
                "cpu_work_units": {
                    "scanned_elements": (
                        "activity candidates plus hash buckets inspected"
                    ),
                    "emitted_elements": "active-list Element records appended",
                },
                "cpu_parallel_gate": (
                    "nonroot generic listgen only; at least 64 parent-list "
                    "entries, 65536 candidate slots, two CPU threads, and "
                    "at most 64 MiB of shared prefix offsets"
                ),
                "decision_identity": "requests == rebuilds + reuse_hits",
                "counter_scope": "cumulative_since_debug_reset",
                "normal_path_overhead": (
                    "disabled until the private debug reset is called"
                ),
            },
        },
        "phase_order": list(PHASES),
        "phase_summary": {
            phase: _summary_ns(
                [sample["duration_ns"] for sample in samples]
            )
            for phase, samples in phase_samples.items()
        },
        "iterations": iteration_records,
        "lifecycle": {
            "baseline": baseline,
            "final": final,
            "snode_deltas_after_destroy": snode_deltas,
            "frontend_executable_deltas_after_destroy": executable_deltas,
            "requested_live_memory_field": live_memory_field,
            "requested_live_memory_deltas_after_destroy": live_memory_deltas,
            "requested_live_memory_warmup_cycles": lifecycle_warmup_cycles,
            "snode_state_recovered_each_cycle": snode_recovered,
            "requested_live_memory_recovered_each_cycle": (
                live_memory_recovered
            ),
            "no_monotonic_snode_growth": not _monotonic_growth(
                snode_growth_values
            ),
            "no_monotonic_frontend_executable_growth": (
                not _monotonic_growth(executable_growth_values)
            ),
            "no_monotonic_requested_live_memory_growth_raw": (
                None
                if not exact_live_memory_deltas
                else not _monotonic_growth(exact_live_memory_deltas)
            ),
            "no_monotonic_requested_live_memory_growth_after_warmup": (
                None
                if not steady_live_memory_deltas
                else not _monotonic_growth(steady_live_memory_deltas)
            ),
            "tree_id_reused": len(set(tree_ids)) == 1,
            "generation_strictly_increasing": all(
                left < right
                for left, right in zip(generations, generations[1:])
            ),
            "memory_recovery_evidence": (
                f"exact_{live_memory_field}"
                if exact_live_memory_deltas
                else "unavailable_for_backend"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch", choices=("cpu", "cuda", "vulkan"), default="cpu"
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--root-blocks", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--active-blocks", type=int, default=8)
    parser.add_argument(
        "--performance",
        action="store_true",
        help=(
            "Mark timings as performance data. GPU runs then require the "
            "automatic idle admission check."
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    measurement = prepare_performance_measurement(
        args.arch, requested=args.performance
    )

    import taichi_forge as ti

    arch = {"cpu": ti.cpu, "cuda": ti.cuda, "vulkan": ti.vulkan}[args.arch]
    init_started = time.perf_counter_ns()
    ti.init(
        arch=arch,
        enable_fallback=False,
        offline_cache=False,
        vulkan_sparse_experimental=True,
        cuda_sparse_pool_auto_size=True,
        cuda_sparse_per_snode_pool=True,
    )
    init_duration_ns = time.perf_counter_ns() - init_started
    try:
        result = run_initialized(
            ti,
            iterations=args.iterations,
            root_blocks=args.root_blocks,
            block_size=args.block_size,
            active_blocks=args.active_blocks,
        )
        result["init_duration_ns"] = init_duration_ns
        result.update(
            finalize_performance_measurement(
                measurement, correct=result["correct"]
            )
        )
    finally:
        ti.reset()
    result["program_reset_completed"] = True

    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        output = args.output
        if not output.is_absolute():
            output = ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
