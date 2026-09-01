"""Strict local S4 qualification for the private GraphMemory recipe.

This driver compares two explicitly reconstructed Graph recipes for the same
CUDA stencil: the direct offload plan and ``shared_staged_1d``.  Its primary
timing path binds each Graph once and replays an immutable GraphBindingSet.
Mutable raw dictionaries are measured only as an ineligible compatibility
diagnostic.

The parent process launches ten fresh workers per scope, balanced between AB
and BA order.  Every worker uses one common replay count for both routes and
five paired blocks of at least 250 ms.  Compilation, binding publication,
submit/pacer, A-B-A switching, and raw-dict measurements are outside the main
timing blocks.

This is evidence for a private recipe only.  It neither exposes the recipe nor
admits it to CompileIQ.
"""

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Callable, Sequence
import uuid

try:
    from .runtime_common import (
        command_output,
        git_metadata,
        gpu_compute_processes,
        gpu_conflicting_processes,
        gpu_snapshot,
        host_metadata,
        process_gpu_memory_mib,
        runtime_device_identity,
        runtime_memory_observation,
        sha256_file,
        working_set_bytes,
        write_json,
    )
except ImportError:  # Direct script execution.
    from runtime_common import (
        command_output,
        git_metadata,
        gpu_compute_processes,
        gpu_conflicting_processes,
        gpu_snapshot,
        host_metadata,
        process_gpu_memory_mib,
        runtime_device_identity,
        runtime_memory_observation,
        sha256_file,
        working_set_bytes,
        write_json,
    )


SCHEMA = "taichi_forge.graph-memory-s4-qualification.v1"
RESULT_PREFIX = "GRAPH_MEMORY_S4_RESULT="
ROUTES = ("direct", "staged")
SCOPES = ("radius1", "radius4")
PRIMARY_BINDING_MODE = "stable_graph_binding_set"
RAW_BINDING_MODE = "raw_dict_compatibility"
REQUIRED_FRESH_PROCESSES = 10
REQUIRED_BLOCKS = 5
MINIMUM_BLOCK_MS = 250.0
MINIMUM_STABILITY_REPLAYS = 10_000
MAX_COMMON_REPLAYS = 2_000_000
MEMORY_RUNTIME_KEYS = (
    "host_requested_live_bytes",
    "host_raw_bytes",
    "device_requested_live_bytes",
    "device_raw_bytes",
    "device_cached_bytes",
)
MEMORY_POOL_KEYS = (
    "raw_chunks",
    "requested_live_bytes",
    "raw_bytes",
    "reserved_bytes",
    "committed_bytes",
    "used_bytes",
    "cached_blocks",
    "cached_bytes",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _coefficient_of_variation(values: Sequence[float]) -> float:
    samples = tuple(float(value) for value in values)
    if not samples:
        raise ValueError("coefficient of variation requires at least one value")
    mean = statistics.fmean(samples)
    return 0.0 if mean == 0.0 else statistics.pstdev(samples) / mean


def _balanced_orders(processes: int) -> tuple[tuple[str, str], ...]:
    if processes != REQUIRED_FRESH_PROCESSES:
        raise ValueError(
            f"formal qualification requires exactly {REQUIRED_FRESH_PROCESSES} "
            "fresh processes per scope"
        )
    return tuple(
        ROUTES if index % 2 == 0 else tuple(reversed(ROUTES))
        for index in range(processes)
    )


def qualification_policy_errors(args: Any) -> list[str]:
    errors: list[str] = []
    if int(args.processes) != REQUIRED_FRESH_PROCESSES:
        errors.append(f"processes must equal {REQUIRED_FRESH_PROCESSES} for formal S4")
    if int(args.blocks) != REQUIRED_BLOCKS:
        errors.append(f"blocks must equal {REQUIRED_BLOCKS} for formal S4")
    if float(args.minimum_block_ms) < MINIMUM_BLOCK_MS:
        errors.append(f"minimum_block_ms must be at least {MINIMUM_BLOCK_MS:g}")
    if int(args.stability_replays) < MINIMUM_STABILITY_REPLAYS:
        errors.append(
            "stability_replays must be at least " f"{MINIMUM_STABILITY_REPLAYS}"
        )
    if int(args.count) <= 2 * max(_scope_radius(scope) for scope in SCOPES):
        errors.append("count is too small for every qualified stencil radius")
    if int(args.raw_diagnostic_replays) < 1:
        errors.append("raw_diagnostic_replays must be positive")
    return errors


def _scope_radius(scope: str) -> int:
    try:
        return {"radius1": 1, "radius4": 4}[scope]
    except KeyError as exc:
        raise ValueError(f"unknown GraphMemory S4 scope: {scope!r}") from exc


def _nested_value(value: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = value
    for item in path:
        if not isinstance(current, Mapping) or item not in current:
            return None
        current = current[item]
    return current


def _plateau_comparison(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare allocator high-water state without imposing an engine cap."""

    comparisons: list[dict[str, Any]] = []
    paths = [
        (("runtime", "memory", key), f"runtime.memory.{key}")
        for key in MEMORY_RUNTIME_KEYS
    ]
    paths.extend(
        (("pools", pool, key), f"pools.{pool}.{key}")
        for pool in ("host", "device")
        for key in MEMORY_POOL_KEYS
    )
    for path, label in paths:
        before = _nested_value(first, path)
        after = _nested_value(second, path)
        if (
            before is None
            or after is None
            or isinstance(before, bool)
            or isinstance(after, bool)
            or not isinstance(before, (int, float))
            or not isinstance(after, (int, float))
        ):
            continue
        comparisons.append(
            {
                "field": label,
                "first": before,
                "second": after,
                "delta": after - before,
                "nonincreasing": after <= before,
            }
        )
    available = bool(first.get("available") and second.get("available"))
    passed = (
        available
        and bool(comparisons)
        and all(item["nonincreasing"] for item in comparisons)
    )
    return {
        "available": available,
        "comparable_fields": len(comparisons),
        "passed": passed,
        "comparison": comparisons,
        "policy": "second_wave_must_not_exceed_first_wave; no fixed byte cap",
    }


def _worker_policy_errors(
    worker: Mapping[str, Any],
    minimum_block_ms: float = MINIMUM_BLOCK_MS,
    blocks: int = REQUIRED_BLOCKS,
) -> list[str]:
    errors: list[str] = []
    prefix = f"worker[{worker.get('worker_index', '?')}]"
    if worker.get("schema") != SCHEMA:
        errors.append(f"{prefix}: schema mismatch")
    if worker.get("scope") not in SCOPES:
        errors.append(f"{prefix}: unknown scope")
    order = tuple(worker.get("order", ()))
    if order not in (ROUTES, tuple(reversed(ROUTES))):
        errors.append(f"{prefix}: invalid AB/BA order")
    if worker.get("primary_binding_mode") != PRIMARY_BINDING_MODE:
        errors.append(f"{prefix}: primary timing did not use GraphBindingSet")

    common_replays = worker.get("common_replays")
    if (
        isinstance(common_replays, bool)
        or not isinstance(common_replays, int)
        or common_replays < 1
    ):
        errors.append(f"{prefix}: invalid common replay count")
    timing_blocks = tuple(worker.get("timing_blocks", ()))
    if len(timing_blocks) != blocks:
        errors.append(f"{prefix}: expected exactly {blocks} paired timing blocks")
    for block_index, block in enumerate(timing_blocks):
        measurements = block.get("routes", {}) if isinstance(block, Mapping) else {}
        if set(measurements) != set(ROUTES):
            errors.append(f"{prefix}: block {block_index} does not contain both routes")
            continue
        for route in ROUTES:
            measurement = measurements[route]
            if measurement.get("replays") != common_replays:
                errors.append(
                    f"{prefix}: block {block_index} {route} did not use common replays"
                )
            if (
                not measurement.get("minimum_satisfied")
                or float(measurement.get("elapsed_ms", 0.0)) < minimum_block_ms
            ):
                errors.append(
                    f"{prefix}: block {block_index} {route} is shorter than minimum"
                )

    correctness = worker.get("correctness", {})
    if not correctness or not all(value is True for value in correctness.values()):
        errors.append(f"{prefix}: exact correctness gate failed")
    if worker.get("route_evidence", {}).get("passed") is not True:
        errors.append(f"{prefix}: exact route/materialization gate failed")
    if worker.get("provenance", {}).get("passed") is not True:
        errors.append(f"{prefix}: source/native provenance gate failed")
    if worker.get("noise", {}).get("passed") is not True:
        errors.append(f"{prefix}: external GPU noise gate failed")

    forbidden = worker.get("forbidden_calls", {})
    if forbidden.get("publish_instrumented") is not True:
        errors.append(f"{prefix}: forbidden-call instrumentation was not exercised")
    stable_delta = forbidden.get("stable_path_delta", {})
    if set(stable_delta) != {
        "describe_storage",
        "validate_storage_owner",
        "analyze_storage_alias",
    } or any(value != 0 for value in stable_delta.values()):
        errors.append(f"{prefix}: stable replay repeated forbidden validation")

    memory = worker.get("memory_plateau", {})
    if memory.get("passed") is not True:
        errors.append(f"{prefix}: memory did not plateau across stability waves")
    if int(memory.get("replays_per_wave", 0)) < MINIMUM_STABILITY_REPLAYS:
        errors.append(f"{prefix}: memory stability wave is too short")
    smokes = worker.get("smoke", {})
    if set(smokes) != {"submit", "paced_submit", "a_b_a"} or not all(
        value is True for value in smokes.values()
    ):
        errors.append(f"{prefix}: submit/paced/A-B-A smoke failed")

    raw = worker.get("raw_dict_diagnostic", {})
    if raw.get("binding_mode") != RAW_BINDING_MODE:
        errors.append(f"{prefix}: raw-dict diagnostic identity is missing")
    if raw.get("admission_eligible") is not False:
        errors.append(f"{prefix}: raw dictionaries must never be admission eligible")
    return errors


def _aggregate_scope(
    scope: str,
    workers: Sequence[Mapping[str, Any]],
    minimum_block_ms: float = MINIMUM_BLOCK_MS,
    blocks: int = REQUIRED_BLOCKS,
) -> dict[str, Any]:
    worker_rows = tuple(workers)
    errors: list[str] = []
    if len(worker_rows) != REQUIRED_FRESH_PROCESSES:
        errors.append(
            f"scope {scope}: expected {REQUIRED_FRESH_PROCESSES} fresh processes"
        )
    instance_ids = [worker.get("process_instance_id") for worker in worker_rows]
    if len(instance_ids) != len(set(instance_ids)) or any(
        not value for value in instance_ids
    ):
        errors.append(f"scope {scope}: process instance identities are not unique")
    orders = [tuple(worker.get("order", ())) for worker in worker_rows]
    if (
        orders.count(ROUTES) != REQUIRED_FRESH_PROCESSES // 2
        or orders.count(tuple(reversed(ROUTES))) != REQUIRED_FRESH_PROCESSES // 2
    ):
        errors.append(f"scope {scope}: process orders are not balanced AB/BA")
    for worker in worker_rows:
        errors.extend(_worker_policy_errors(worker, minimum_block_ms, blocks))

    ratios = tuple(
        float(worker.get("process_ratio", math.inf)) for worker in worker_rows
    )
    finite_positive_ratios = bool(ratios) and all(
        math.isfinite(value) and value > 0.0 for value in ratios
    )
    if not finite_positive_ratios:
        errors.append(f"scope {scope}: invalid staged/direct process ratio")
    structural_passed = not errors
    worst_positive = structural_passed and max(ratios) < 1.0
    status = (
        "qualified_positive"
        if worst_positive
        else "negative_retained" if structural_passed else "invalid_evidence"
    )
    by_order: dict[str, float | None] = {}
    for label, expected in (("ab", ROUTES), ("ba", tuple(reversed(ROUTES)))):
        selected = [
            ratio
            for ratio, worker in zip(ratios, worker_rows)
            if tuple(worker.get("order", ())) == expected
        ]
        by_order[label] = statistics.median(selected) if selected else None
    return {
        "scope": scope,
        "status": status,
        "structural_gates_passed": structural_passed,
        "policy_errors": errors,
        "processes": len(worker_rows),
        "orders_balanced": not any("balanced AB/BA" in error for error in errors),
        "candidate_over_baseline_process_ratios": list(ratios),
        "median_staged_over_direct": (
            statistics.median(ratios) if finite_positive_ratios else None
        ),
        "best_staged_over_direct": min(ratios) if finite_positive_ratios else None,
        "worst_staged_over_direct": max(ratios) if finite_positive_ratios else None,
        "ratio_cv": (
            _coefficient_of_variation(ratios) if finite_positive_ratios else None
        ),
        "order_medians": by_order,
        "strict_worst_positive": worst_positive,
        "worker_evidence": list(worker_rows),
    }


def _report_policy_errors(report: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("schema") != SCHEMA:
        errors.append("report schema mismatch")
    policy = report.get("policy", {})
    if policy.get("primary_binding_mode") != PRIMARY_BINDING_MODE:
        errors.append("report primary binding mode is not stable GraphBindingSet")
    if policy.get("raw_dict_admission_eligible") is not False:
        errors.append("report permits raw-dict admission")
    if report.get("provenance", {}).get("passed") is not True:
        errors.append("report source/native provenance failed")
    if report.get("noise", {}).get("passed") is not True:
        errors.append("report external GPU noise gate failed")
    scopes = report.get("scopes", {})
    if set(scopes) != set(SCOPES):
        errors.append("report does not contain the exact qualified scopes")
    elif not all(value.get("structural_gates_passed") for value in scopes.values()):
        errors.append("one or more scopes failed structural gates")
    return errors


def _timed_route(
    ti: Any, invoke: Callable[[], None], replays: int, minimum_block_ms: float
) -> dict[str, Any]:
    started = time.perf_counter_ns()
    for _ in range(replays):
        invoke()
    ti.sync()
    elapsed_ns = time.perf_counter_ns() - started
    elapsed_ms = elapsed_ns / 1.0e6
    return {
        "replays": replays,
        "elapsed_ms": elapsed_ms,
        "ns_per_replay": elapsed_ns / replays,
        "minimum_block_ms": minimum_block_ms,
        "minimum_satisfied": elapsed_ms >= minimum_block_ms,
    }


def _calibrate_common_replays(
    ti: Any,
    routes: Mapping[str, Callable[[], None]],
    minimum_block_ms: float,
) -> tuple[int, dict[str, Any]]:
    replays = 32
    while True:
        observations = {
            route: _timed_route(ti, routes[route], replays, minimum_block_ms)
            for route in ROUTES
        }
        shortest = min(item["elapsed_ms"] for item in observations.values())
        if shortest >= minimum_block_ms:
            return replays, observations
        if replays >= MAX_COMMON_REPLAYS:
            raise RuntimeError("common timing replay count could not reach 250 ms")
        scale = max(
            2,
            math.ceil(minimum_block_ms / max(shortest, 0.001) * 1.08),
        )
        replays = min(MAX_COMMON_REPLAYS, replays * scale)


def _measure_paired_blocks(
    ti: Any,
    routes: Mapping[str, Callable[[], None]],
    process_order: tuple[str, str],
    replays: int,
    minimum_block_ms: float,
    blocks: int,
) -> list[dict[str, Any]]:
    evidence = []
    for block_index in range(blocks):
        order = (
            process_order if block_index % 2 == 0 else tuple(reversed(process_order))
        )
        measurements = {
            route: _timed_route(ti, routes[route], replays, minimum_block_ms)
            for route in order
        }
        evidence.append(
            {
                "block_index": block_index,
                "order": list(order),
                "routes": measurements,
                "staged_over_direct": (
                    measurements["staged"]["ns_per_replay"]
                    / measurements["direct"]["ns_per_replay"]
                ),
            }
        )
    return evidence


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _normalize_allow_dirty_paths(
    repo_root: Path, values: Sequence[str | Path]
) -> tuple[str, ...]:
    normalized = []
    for value in values:
        raw = Path(value)
        candidate = raw.resolve() if raw.is_absolute() else (repo_root / raw).resolve()
        try:
            relative = candidate.relative_to(repo_root.resolve())
        except ValueError as exc:
            raise ValueError(
                f"allowed dirty path is outside the repository: {value}"
            ) from exc
        text = relative.as_posix()
        if not text or text == ".":
            raise ValueError(f"invalid allowed dirty path: {value}")
        if text not in normalized:
            normalized.append(text)
    return tuple(normalized)


def _status_entries(status_lines: Sequence[str]) -> tuple[dict[str, str], ...]:
    entries = []
    for line in status_lines:
        if len(line) < 4:
            entries.append({"status": line[:2], "path": "", "raw": line})
            continue
        entries.append(
            {
                "status": line[:2],
                "path": line[3:].replace("\\", "/"),
                "raw": line,
            }
        )
    return tuple(entries)


def _git_status_entries(repo_root: Path) -> tuple[dict[str, str], ...]:
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repo_root.as_posix()}",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "-z",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    records = completed.stdout.split("\0")
    entries: list[dict[str, str]] = []
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        status = record[:2]
        path = record[3:].replace("\\", "/") if len(record) >= 4 else ""
        if "R" in status or "C" in status:
            second = records[index] if index < len(records) else ""
            index += 1
            path = f"{path} -> {second.replace(chr(92), '/')}"
        entries.append(
            {
                "status": status,
                "path": path,
                "raw": f"{status} {path}",
            }
        )
    return tuple(entries)


def _eol_only_diff(repo_root: Path, relative_path: str) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repo_root.as_posix()}",
            "diff",
            "HEAD",
            "--ignore-space-at-eol",
            "--exit-code",
            "--",
            relative_path,
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "path": relative_path,
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _counter_delta(
    after: Mapping[str, int], before: Mapping[str, int]
) -> dict[str, int]:
    return {name: int(after[name]) - int(before[name]) for name in before}


def _memory_observation(ti: Any) -> dict[str, Any]:
    observation = runtime_memory_observation(ti)
    observation.update(
        {
            "working_set_bytes_diagnostic": working_set_bytes(),
            "process_gpu_memory_mib_diagnostic": process_gpu_memory_mib(os.getpid()),
        }
    )
    return observation


def _exact_output(array: Any, expected: Any, np: Any) -> bool:
    actual = array.to_numpy()
    passed = bool(np.array_equal(actual, expected))
    del actual
    return passed


def _manifest_row(manifest: Any) -> dict[str, Any]:
    return _jsonable(vars(manifest))


def _worker_provenance(
    ti: Any,
    repo_root: Path,
    expected_head: str,
) -> dict[str, Any]:
    from taichi_forge._contracts import validate_runtime_contract

    contract = validate_runtime_contract(require_native_manifest=True)
    contract_row = _jsonable(contract)
    package_path = Path(ti.__file__).resolve()
    core_path = Path(ti._lib.core.__file__).resolve()
    runtime_source_id = str(contract["runtime"]["source_id"]).lower()
    shim_source_id = str(contract["shim"]["source_id"]).lower()
    core_commit = str(ti._lib.core.get_commit_hash()).lower()
    expected = str(expected_head).lower()
    device = runtime_device_identity(ti, "cuda")
    checks = {
        "expected_head_is_full_sha": len(expected) == 40
        and all(character in "0123456789abcdef" for character in expected),
        "runtime_source_matches_head": runtime_source_id == expected,
        "shim_source_matches_head": shim_source_id == expected,
        "native_commit_matches_head": core_commit == expected,
        "package_inside_repository": _path_within(package_path, repo_root),
        "native_core_inside_repository": _path_within(core_path, repo_root),
        "native_manifest_present": not bool(contract["runtime"]["legacy_runtime"]),
        "runtime_contract_compatible": (
            contract["native_abi_revision"] == contract["required_native_abi_revision"]
        ),
        "physical_device_binding_verified": bool(device["binding_verified"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "expected_head": expected,
        "package_path": str(package_path),
        "native_core_path": str(core_path),
        "native_core_sha256": sha256_file(core_path),
        "native_commit": core_commit,
        "runtime_contract": contract_row,
        "device_identity": device,
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
    }


def _run_stability_wave(
    ti: Any,
    routes: Mapping[str, Callable[[], None]],
    replays: int,
) -> None:
    for _ in range(replays):
        routes["direct"]()
        routes["staged"]()
    ti.sync()


def _graph_memory_worker(args: Any) -> dict[str, Any]:
    # Imports stay inside the worker so --help and policy tests never touch CUDA.
    import numpy as np
    import taichi_forge as ti
    from taichi_forge.graph import _graph as graph_impl
    from taichi_forge.lang._offload_execution_plan import (
        _OffloadExecutionPlan,
        _bind_offload_execution_plan,
    )

    repo_root = Path(args.repo_root).resolve()
    radius = _scope_radius(args.scope)
    count = int(args.count)
    process_order = tuple(args.order.split(","))
    if process_order not in (ROUTES, tuple(reversed(ROUTES))):
        raise ValueError("worker order must be direct,staged or staged,direct")

    noise_rows_before = gpu_compute_processes()
    noise_conflicts_before = gpu_conflicting_processes(
        noise_rows_before, ignored_pids=(os.getpid(), os.getppid())
    )
    gpu_before = gpu_snapshot()
    ti.init(
        arch=ti.cuda,
        enable_fallback=False,
        offline_cache=False,
    )
    provenance = _worker_provenance(ti, repo_root, args.expected_head)

    @ti.kernel
    def stencil(
        source: ti.types.ndarray(dtype=ti.f32, ndim=1),
        output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in range(radius, count - radius):
            if ti.static(radius == 1):
                output[i] = source[i - 1] + source[i] + source[i + 1]
            else:
                output[i] = (
                    source[i - 4]
                    + source[i - 3]
                    + source[i - 2]
                    + source[i - 1]
                    + source[i]
                    + source[i + 1]
                    + source[i + 2]
                    + source[i + 3]
                    + source[i + 4]
                )

    source = ti.ndarray(ti.f32, shape=count)
    direct_output = ti.ndarray(ti.f32, shape=count)
    staged_output = ti.ndarray(ti.f32, shape=count)
    alternate_source = ti.ndarray(ti.f32, shape=count)
    alternate_output = ti.ndarray(ti.f32, shape=count)
    source.fill(1.0)
    alternate_source.fill(2.0)
    direct_output.fill(0.0)
    staged_output.fill(0.0)
    alternate_output.fill(0.0)

    baseline_plan = _OffloadExecutionPlan.from_task_manifests(
        stencil.task_manifest(source, direct_output)
    )
    ranges = tuple(
        task for task in baseline_plan.tasks if task.task_kind == "range_for"
    )
    if len(ranges) != 1:
        raise RuntimeError("GraphMemory qualification requires one range_for task")
    staged_plan = baseline_plan.replace_task(
        ranges[0].task_index,
        workgroup_size=128,
        memory_strategy="shared_staged_1d",
    )
    direct_kernel = _bind_offload_execution_plan(stencil, baseline_plan)
    staged_kernel = _bind_offload_execution_plan(stencil, staged_plan)
    direct_manifest = tuple(direct_kernel.task_manifest(source, direct_output))
    staged_manifest = tuple(staged_kernel.task_manifest(source, staged_output))
    baseline_plan.validate_materialization(direct_manifest)
    staged_plan.validate_materialization(staged_manifest)
    direct_range = next(
        task for task in direct_manifest if task.task_type == "range_for"
    )
    staged_range = next(
        task for task in staged_manifest if task.task_type == "range_for"
    )

    source_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "source", ti.f32, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1)
    build_ms: dict[str, float] = {}
    started = time.perf_counter_ns()
    direct_builder = ti.graph.GraphBuilder()
    direct_builder.dispatch(direct_kernel, source_arg, output_arg)
    direct_graph = direct_builder.compile()
    build_ms["direct"] = (time.perf_counter_ns() - started) / 1.0e6
    started = time.perf_counter_ns()
    staged_builder = ti.graph.GraphBuilder()
    staged_builder._dispatch_shared_staged_1d(staged_kernel, source_arg, output_arg)
    staged_graph = staged_builder.compile()
    build_ms["staged"] = (time.perf_counter_ns() - started) / 1.0e6

    graph_manifests = {
        "direct": tuple(direct_graph.task_manifest()),
        "staged": tuple(staged_graph.task_manifest()),
    }
    staged_graph_range = next(
        task for task in graph_manifests["staged"] if task.task_type == "range_for"
    )

    counter_names = (
        "describe_storage",
        "validate_storage_owner",
        "analyze_storage_alias",
    )
    counters = {name: 0 for name in counter_names}
    originals = {name: getattr(graph_impl, name) for name in counter_names}

    def counted(name: str) -> Callable[..., Any]:
        original = originals[name]

        def invoke(*values: Any, **keywords: Any) -> Any:
            counters[name] += 1
            return original(*values, **keywords)

        return invoke

    for counter_name in counter_names:
        setattr(graph_impl, counter_name, counted(counter_name))

    direct_binding = direct_graph.bind({"source": source, "output": direct_output})
    staged_binding = staged_graph.bind({"source": source, "output": staged_output})
    alternate_binding = staged_graph.bind(
        {"source": alternate_source, "output": alternate_output}
    )
    after_publish = dict(counters)
    stable_counter_baseline = dict(counters)

    direct_binding_stats = _jsonable(direct_binding.statistics())
    staged_binding_stats = _jsonable(staged_binding.statistics())
    alternate_binding_stats = _jsonable(alternate_binding.statistics())
    staged_binding_plan = _jsonable(staged_graph.binding_plan())
    expected_grid = (count - 2 * radius + 127) // 128
    route_checks = {
        "shared_semantic_kernel_identity": (
            baseline_plan.semantic_kernel_identity
            == staged_plan.semantic_kernel_identity
        ),
        "distinct_compilation_identity": (
            baseline_plan.compilation_identity != staged_plan.compilation_identity
        ),
        "direct_plan_materialized": direct_range.requested_memory_strategy == "direct",
        "staged_plan_materialized": (
            staged_range.requested_memory_strategy == "shared_staged_1d"
            and staged_range.range_mapping == "shared_tiled_one_to_one"
            and staged_range.selected_block_size == 128
            and staged_range.selected_grid_size == expected_grid
            and staged_range.staged_external_arg_index == 0
            and (staged_range.staged_halo_low, staged_range.staged_halo_high)
            == (-radius, radius)
            and staged_range.static_shared_bytes == (128 + 2 * radius) * 4
        ),
        "staged_graph_manifest_materialized": (
            staged_graph_range.requested_memory_strategy == "shared_staged_1d"
            and staged_graph_range.range_mapping == "shared_tiled_one_to_one"
            and staged_graph_range.selected_block_size == 128
            and staged_graph_range.selected_grid_size == expected_grid
            and (
                staged_graph_range.staged_halo_low,
                staged_graph_range.staged_halo_high,
            )
            == (-radius, radius)
        ),
        "direct_binding_fast": bool(direct_binding.fast_path_qualified),
        "staged_binding_fast": bool(staged_binding.fast_path_qualified),
        "alternate_binding_fast": bool(alternate_binding.fast_path_qualified),
        "staged_memory_certificate": bool(
            staged_binding_stats.get("memory_recipe_publish_validated")
            and staged_binding_stats.get("memory_recipe_certified")
            and "dynamic_memory_recipe"
            not in staged_binding_stats.get("volatile_reasons", ())
        ),
        "alternate_memory_certificate": bool(
            alternate_binding_stats.get("memory_recipe_publish_validated")
            and alternate_binding_stats.get("memory_recipe_certified")
        ),
        "binding_plan_publish_certificate": bool(
            staged_binding_plan.get("memory_recipe_publish_certificate_required")
            and staged_binding_plan.get("memory_recipe_publish_frame_stable")
            and set(staged_binding_plan.get("memory_recipe_names", ()))
            == {"source", "output"}
        ),
    }
    route_evidence = {
        "passed": all(route_checks.values()),
        "checks": route_checks,
        "baseline_plan_identity": baseline_plan.identity,
        "staged_plan_identity": staged_plan.identity,
        "baseline_compilation_identity": baseline_plan.compilation_identity,
        "staged_compilation_identity": staged_plan.compilation_identity,
        "direct_kernel_manifest": [_manifest_row(item) for item in direct_manifest],
        "staged_kernel_manifest": [_manifest_row(item) for item in staged_manifest],
        "direct_graph_manifest": [
            _manifest_row(item) for item in graph_manifests["direct"]
        ],
        "staged_graph_manifest": [
            _manifest_row(item) for item in graph_manifests["staged"]
        ],
        "direct_physical_plan": _jsonable(direct_graph.physical_plan()),
        "staged_physical_plan": _jsonable(staged_graph.physical_plan()),
        "staged_binding_plan": staged_binding_plan,
        "binding_statistics_at_publish": {
            "direct": direct_binding_stats,
            "staged": staged_binding_stats,
            "alternate": alternate_binding_stats,
        },
    }

    routes = {
        "direct": lambda: direct_graph.run(direct_binding),
        "staged": lambda: staged_graph.run(staged_binding),
    }
    expected = np.zeros(count, dtype=np.float32)
    expected[radius : count - radius] = float(2 * radius + 1)
    alternate_expected = np.zeros(count, dtype=np.float32)
    alternate_expected[radius : count - radius] = float(2 * (2 * radius + 1))

    for route in ROUTES:
        routes[route]()
    ti.sync()
    correctness = {
        "direct_initial_exact": _exact_output(direct_output, expected, np),
        "staged_initial_exact": _exact_output(staged_output, expected, np),
    }

    smoke: dict[str, bool] = {}
    staged_graph.submit(staged_binding).wait()
    smoke["submit"] = _exact_output(staged_output, expected, np)
    pacer = ti.graph.SubmissionPacer(1)
    staged_graph.submit(
        staged_binding, pacer=pacer, lane="graph-memory-s4-pacer-smoke"
    ).wait()
    smoke["paced_submit"] = _exact_output(staged_output, expected, np)
    staged_graph.run(staged_binding)
    staged_graph.run(alternate_binding)
    staged_graph.run(staged_binding)
    ti.sync()
    smoke["a_b_a"] = bool(
        _exact_output(staged_output, expected, np)
        and _exact_output(alternate_output, alternate_expected, np)
    )

    for _ in range(16):
        routes["direct"]()
        routes["staged"]()
    ti.sync()
    common_replays, calibration = _calibrate_common_replays(
        ti, routes, float(args.minimum_block_ms)
    )
    timing_blocks = _measure_paired_blocks(
        ti,
        routes,
        process_order,
        common_replays,
        float(args.minimum_block_ms),
        int(args.blocks),
    )
    correctness.update(
        {
            "direct_after_timing_exact": _exact_output(direct_output, expected, np),
            "staged_after_timing_exact": _exact_output(staged_output, expected, np),
        }
    )

    _run_stability_wave(ti, routes, int(args.stability_replays))
    first_memory = _memory_observation(ti)
    _run_stability_wave(ti, routes, int(args.stability_replays))
    second_memory = _memory_observation(ti)
    memory_plateau = _plateau_comparison(first_memory, second_memory)
    memory_plateau.update(
        {
            "replays_per_wave": int(args.stability_replays),
            "routes_per_replay": list(ROUTES),
            "first_wave": first_memory,
            "second_wave": second_memory,
        }
    )
    correctness.update(
        {
            "direct_after_stability_exact": _exact_output(direct_output, expected, np),
            "staged_after_stability_exact": _exact_output(staged_output, expected, np),
        }
    )

    after_stable = dict(counters)
    stable_delta = _counter_delta(after_stable, stable_counter_baseline)
    graph_statistics_before_raw = {
        "direct": _jsonable(direct_graph.binding_statistics()),
        "staged": _jsonable(staged_graph.binding_statistics()),
        "direct_graph_stats": _jsonable(direct_graph._graph_stats),
        "staged_graph_stats": _jsonable(staged_graph._graph_stats),
    }

    raw_arguments = {"source": source, "output": staged_output}
    raw_before = dict(counters)
    raw_started = time.perf_counter_ns()
    for _ in range(int(args.raw_diagnostic_replays)):
        staged_graph.run(raw_arguments)
    ti.sync()
    raw_elapsed_ns = time.perf_counter_ns() - raw_started
    raw_after = dict(counters)
    correctness["raw_diagnostic_exact"] = _exact_output(staged_output, expected, np)
    raw_diagnostic = {
        "binding_mode": RAW_BINDING_MODE,
        "admission_eligible": False,
        "reason": "mutable dictionaries are fully validated on every replay",
        "replays": int(args.raw_diagnostic_replays),
        "elapsed_ms": raw_elapsed_ns / 1.0e6,
        "ns_per_replay": raw_elapsed_ns / int(args.raw_diagnostic_replays),
        "forbidden_call_delta": _counter_delta(raw_after, raw_before),
    }

    noise_rows_after = gpu_compute_processes()
    noise_conflicts_after = gpu_conflicting_processes(
        noise_rows_after, ignored_pids=(os.getpid(), os.getppid())
    )
    gpu_after = gpu_snapshot()
    noise = {
        "passed": not noise_conflicts_before and not noise_conflicts_after,
        "compute_processes_before": noise_rows_before,
        "compute_processes_after": noise_rows_after,
        "conflicts_before": noise_conflicts_before,
        "conflicts_after": noise_conflicts_after,
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
    }
    ratios = [float(block["staged_over_direct"]) for block in timing_blocks]
    forbidden_calls = {
        "after_binding_publication": after_publish,
        "stable_path_baseline": stable_counter_baseline,
        "after_all_stable_operations": after_stable,
        "stable_path_delta": stable_delta,
        "publish_instrumented": all(after_publish[name] > 0 for name in counter_names),
        "raw_diagnostic_delta": _counter_delta(raw_after, raw_before),
    }
    result = {
        "schema": SCHEMA,
        "evidence_kind": "fresh_process_graph_memory_worker",
        "scope": args.scope,
        "radius": radius,
        "count": count,
        "worker_index": int(args.worker_index),
        "pid": os.getpid(),
        "process_instance_id": str(uuid.uuid4()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "order": list(process_order),
        "primary_binding_mode": PRIMARY_BINDING_MODE,
        "common_replays": common_replays,
        "calibration_diagnostic": calibration,
        "timing_blocks": timing_blocks,
        "block_ratio_median": statistics.median(ratios),
        "process_ratio": statistics.median(ratios),
        "correctness": correctness,
        "route_evidence": route_evidence,
        "provenance": provenance,
        "noise": noise,
        "forbidden_calls": forbidden_calls,
        "memory_plateau": memory_plateau,
        "smoke": smoke,
        "raw_dict_diagnostic": raw_diagnostic,
        "graph_statistics_before_raw": graph_statistics_before_raw,
        "build_ms_diagnostic": build_ms,
    }
    result["policy_errors"] = _worker_policy_errors(
        result, float(args.minimum_block_ms), int(args.blocks)
    )
    result["structural_gates_passed"] = not result["policy_errors"]
    return _jsonable(result)


def _extract_worker_result(stdout: str) -> dict[str, Any]:
    encoded = next(
        (
            line[len(RESULT_PREFIX) :]
            for line in reversed(stdout.splitlines())
            if line.startswith(RESULT_PREFIX)
        ),
        None,
    )
    if encoded is None:
        raise RuntimeError("worker did not emit machine-readable S4 evidence")
    value = json.loads(encoded)
    if not isinstance(value, dict):
        raise RuntimeError("worker S4 evidence is not a JSON object")
    return value


def _worker_environment(repo_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    python_root = str((repo_root / "python").resolve())
    previous_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        python_root
        if not previous_pythonpath
        else os.pathsep.join((python_root, previous_pythonpath))
    )
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["TI_VISIBLE_DEVICE"] = "0"
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    for name in (
        "TAICHI_FORGE_INTERNAL_MAP_FUSION",
        "TAICHI_FORGE_INTERNAL_GRAPH_FUSION_QUALIFICATION",
        "TAICHI_FORGE_INTERNAL_GRAPH_FUSION_EXPECTED_REPLAYS",
    ):
        environment.pop(name, None)
    return environment


def _repository_provenance(
    repo_root: Path,
    script: Path,
    allow_dirty_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    metadata = git_metadata(repo_root)
    git = ["git", "-c", f"safe.directory={repo_root.as_posix()}"]
    origin_head = command_output([*git, "rev-parse", "origin/master"], repo_root)
    allowlist = _normalize_allow_dirty_paths(repo_root, allow_dirty_paths)
    entries = _git_status_entries(repo_root)
    metadata["dirty"] = bool(entries)
    metadata["status_short"] = [entry["raw"] for entry in entries]
    status_paths = tuple(entry["path"] for entry in entries)
    status_shape_supported = all(
        entry["status"] not in ("??", "!!")
        and " -> " not in entry["path"]
        and bool(entry["path"])
        for entry in entries
    )
    paths_in_allowlist = set(status_paths).issubset(set(allowlist))
    eol_checks = [
        _eol_only_diff(repo_root, path)
        for path in status_paths
        if status_shape_supported and path in allowlist
    ]
    dirty_allowed = bool(
        entries
        and allowlist
        and status_shape_supported
        and paths_in_allowlist
        and len(eol_checks) == len(entries)
        and all(check["passed"] for check in eol_checks)
    )
    worktree_accepted = metadata.get("dirty") is False or dirty_allowed
    checks = {
        "head_available": bool(metadata.get("head")),
        "worktree_clean_or_explicit_eol_only_allowlist": worktree_accepted,
        "head_matches_origin_master": bool(
            metadata.get("head") and metadata.get("head") == origin_head
        ),
        "driver_inside_repository": _path_within(script, repo_root),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "git": metadata,
        "origin_master": origin_head,
        "driver_path": str(script),
        "driver_sha256": sha256_file(script),
        "dirty_allowance": {
            "default_is_empty": True,
            "requested_paths": list(allowlist),
            "status_entries": list(entries),
            "status_paths": list(status_paths),
            "status_shape_supported": status_shape_supported,
            "all_status_paths_in_allowlist": paths_in_allowlist,
            "eol_neutral_diff_checks": eol_checks,
            "accepted": dirty_allowed,
            "policy": (
                "every dirty status path must be explicitly listed and have "
                "zero git diff HEAD --ignore-space-at-eol"
            ),
        },
    }


def _default_output(repo_root: Path, head: str | None) -> Path:
    revision = (head or "unknown")[:8]
    return (
        repo_root
        / ".agent"
        / "experiments"
        / f"graph-memory-s4-{revision}"
        / "qualification.json"
    )


def _parent(args: Any) -> int:
    policy_errors = qualification_policy_errors(args)
    if policy_errors:
        raise ValueError("; ".join(policy_errors))
    script = Path(__file__).resolve()
    repo_root = Path(args.repo_root or script.parents[2]).resolve()
    repository = _repository_provenance(repo_root, script, tuple(args.allow_dirty_path))
    if not repository["passed"]:
        raise RuntimeError(
            "formal S4 requires HEAD equal to origin/master and either a clean "
            "worktree or an explicitly verified EOL-only allowlist: "
            + json.dumps(repository["checks"], sort_keys=True)
        )
    head = str(repository["git"]["head"])
    output = (
        Path(args.output).resolve() if args.output else _default_output(repo_root, head)
    )
    worker_dir = output.parent / "workers"

    pre_rows = gpu_compute_processes()
    pre_conflicts = gpu_conflicting_processes(pre_rows, ignored_pids=(os.getpid(),))
    pre_gpu = gpu_snapshot()
    if pre_conflicts:
        raise RuntimeError(
            "formal S4 found conflicting GPU processes: "
            + json.dumps(pre_conflicts, sort_keys=True)
        )
    if not pre_gpu:
        raise RuntimeError("formal S4 could not capture an NVIDIA GPU snapshot")

    environment = _worker_environment(repo_root)
    all_workers: dict[str, list[dict[str, Any]]] = {}
    for scope in SCOPES:
        workers: list[dict[str, Any]] = []
        for worker_index, order in enumerate(_balanced_orders(int(args.processes))):
            order_text = ",".join(order)
            command = [
                sys.executable,
                str(script),
                "--worker",
                "--scope",
                scope,
                "--order",
                order_text,
                "--worker-index",
                str(worker_index),
                "--expected-head",
                head,
                "--repo-root",
                str(repo_root),
                "--count",
                str(args.count),
                "--blocks",
                str(args.blocks),
                "--minimum-block-ms",
                str(args.minimum_block_ms),
                "--stability-replays",
                str(args.stability_replays),
                "--raw-diagnostic-replays",
                str(args.raw_diagnostic_replays),
            ]
            completed = subprocess.run(
                command,
                cwd=repo_root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            worker_stem = f"{scope}-{worker_index:02d}-{order_text.replace(',', '-') }"
            worker_dir.mkdir(parents=True, exist_ok=True)
            (worker_dir / f"{worker_stem}.stdout.txt").write_text(
                completed.stdout, encoding="utf-8"
            )
            (worker_dir / f"{worker_stem}.stderr.txt").write_text(
                completed.stderr, encoding="utf-8"
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"GraphMemory S4 worker failed: {scope} {worker_index}\n"
                    + completed.stdout
                    + completed.stderr
                )
            worker = _extract_worker_result(completed.stdout)
            write_json(worker_dir / f"{worker_stem}.json", worker)
            workers.append(worker)
            print(
                f"{scope}: fresh worker {worker_index + 1}/"
                f"{REQUIRED_FRESH_PROCESSES}",
                flush=True,
            )
        all_workers[scope] = workers

    post_rows = gpu_compute_processes()
    post_conflicts = gpu_conflicting_processes(post_rows, ignored_pids=(os.getpid(),))
    post_gpu = gpu_snapshot()
    summaries = {
        scope: _aggregate_scope(
            scope,
            workers,
            float(args.minimum_block_ms),
            int(args.blocks),
        )
        for scope, workers in all_workers.items()
    }
    worker_provenance_passed = all(
        worker.get("provenance", {}).get("passed") is True
        for workers in all_workers.values()
        for worker in workers
    )
    core_hashes = sorted(
        {
            worker["provenance"]["native_core_sha256"]
            for workers in all_workers.values()
            for worker in workers
            if worker.get("provenance", {}).get("native_core_sha256")
        }
    )
    provenance = {
        "passed": bool(
            repository["passed"] and worker_provenance_passed and len(core_hashes) == 1
        ),
        "repository": repository,
        "all_workers_match_source_and_native": worker_provenance_passed,
        "native_core_sha256_values": core_hashes,
        "one_native_binary_across_workers": len(core_hashes) == 1,
    }
    noise = {
        "passed": bool(not pre_conflicts and not post_conflicts),
        "compute_processes_before": pre_rows,
        "compute_processes_after": post_rows,
        "conflicts_before": pre_conflicts,
        "conflicts_after": post_conflicts,
        "gpu_before": pre_gpu,
        "gpu_after": post_gpu,
    }
    eligible_scopes = [
        scope
        for scope, summary in summaries.items()
        if summary["status"] == "qualified_positive"
    ]
    negative_scopes = [
        scope
        for scope, summary in summaries.items()
        if summary["status"] == "negative_retained"
    ]
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "evidence_class": "strict_fresh_process_balanced_ab_ba",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "fresh_processes_per_scope": REQUIRED_FRESH_PROCESSES,
            "balanced_orders": ["direct,staged", "staged,direct"],
            "paired_blocks_per_process": REQUIRED_BLOCKS,
            "minimum_block_ms": float(args.minimum_block_ms),
            "one_common_replay_count_per_worker": True,
            "primary_binding_mode": PRIMARY_BINDING_MODE,
            "raw_dict_admission_eligible": False,
            "correctness_required": "exact full-array f32",
            "route_materialization_required": True,
            "stable_forbidden_call_delta_required": 0,
            "memory_policy": (
                "second 10000+ replay wave nonincreasing; no fixed memory cap"
            ),
            "strict_worst_positive_required_per_eligible_scope": True,
            "compile_and_publish_time": "diagnostic only",
            "submit_paced_and_a_b_a": "untimed safety smoke only",
        },
        "configuration": {
            "count": int(args.count),
            "minimum_block_ms": float(args.minimum_block_ms),
            "stability_replays_per_wave": int(args.stability_replays),
            "raw_diagnostic_replays": int(args.raw_diagnostic_replays),
        },
        "provenance": provenance,
        "noise": noise,
        "host": host_metadata(),
        "scopes": summaries,
        "admission": {
            "compileiq_changed": False,
            "recipe_visibility_changed": False,
            "eligible_private_recipe_scopes": eligible_scopes,
            "negative_scopes_retained": negative_scopes,
            "raw_dict_eligible": False,
        },
    }
    report_errors = _report_policy_errors(report)
    report["policy_errors"] = report_errors
    report["strict_gate_passed"] = bool(not report_errors and eligible_scopes)
    report["status"] = (
        "qualified_positive"
        if report["strict_gate_passed"] and not negative_scopes
        else (
            "partially_qualified"
            if report["strict_gate_passed"]
            else "negative_retained" if not report_errors else "invalid_evidence"
        )
    )
    write_json(output, report)
    print(
        json.dumps(
            {
                "output": str(output),
                "status": report["status"],
                "strict_gate_passed": report["strict_gate_passed"],
                "eligible_private_recipe_scopes": eligible_scopes,
                "negative_scopes_retained": negative_scopes,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 2 if report_errors else 0


def _parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Qualify private Graph shared_staged_1d against an exact direct "
            "recipe using stable GraphBindingSet replay."
        )
    )
    parser.add_argument("--processes", type=int, default=REQUIRED_FRESH_PROCESSES)
    parser.add_argument("--blocks", type=int, default=REQUIRED_BLOCKS)
    parser.add_argument("--minimum-block-ms", type=float, default=MINIMUM_BLOCK_MS)
    parser.add_argument("--count", type=int, default=1 << 24)
    parser.add_argument(
        "--stability-replays", type=int, default=MINIMUM_STABILITY_REPLAYS
    )
    parser.add_argument("--raw-diagnostic-replays", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "JSON artifact path; default: "
            ".agent/experiments/graph-memory-s4-<head8>/qualification.json"
        ),
    )
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument(
        "--allow-dirty-path",
        action="append",
        default=[],
        metavar="REPO_RELATIVE_PATH",
        help=(
            "allow one explicitly named dirty path only when git diff HEAD "
            "--ignore-space-at-eol is empty; repeat per path (default: none)"
        ),
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--scope", choices=SCOPES, help=argparse.SUPPRESS)
    parser.add_argument("--order", help=argparse.SUPPRESS)
    parser.add_argument("--worker-index", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--expected-head", help=argparse.SUPPRESS)
    args = parser.parse_args(arguments)
    if args.worker:
        missing = [
            name
            for name in ("scope", "order", "expected_head", "repo_root")
            if getattr(args, name) in (None, "")
        ]
        if missing:
            missing_options = ", ".join(
                f"--{name.replace('_', '-')}" for name in missing
            )
            parser.error("--worker requires " + missing_options)
        if args.worker_index < 0:
            parser.error("--worker requires a nonnegative --worker-index")
    return args


def main(arguments: Sequence[str] | None = None) -> int:
    args = _parse_args(arguments)
    if args.worker:
        errors = qualification_policy_errors(args)
        if errors:
            raise ValueError("; ".join(errors))
        result = _graph_memory_worker(args)
        print(RESULT_PREFIX + json.dumps(result, sort_keys=True))
        return 0 if result["structural_gates_passed"] else 2
    return _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
