"""Isolated Warp external baseline for THIN-002 affine transform.

This program intentionally produces absolute Warp measurements only.  It does
not create a Forge/Warp speedup because an external framework is not a drop-in
runtime replacement for the Forge or vanilla Taichi public API.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Callable, Sequence

try:
    from .runtime_common import (
        git_metadata,
        gpu_snapshot,
        host_metadata,
        logical_bandwidth_gbps,
        normalize_gpu_uuid,
        process_gpu_memory_mib,
        sha256_file,
        summarize_samples,
        working_set_bytes,
        write_csv,
        write_json,
    )
    from .single_kernel_microbench import (
        QUALIFICATION_MAX_CPU_UTIL_PERCENT,
        QUALIFICATION_MAX_GPU_TEMPERATURE_C,
        QUALIFICATION_MAX_GPU_UTIL_PERCENT,
        QUALIFICATION_MAX_CV_PERCENT,
        QUALIFICATION_MINIMUMS,
        _ExclusiveBenchmarkLock,
        _apply_affinity,
        _noise_observation,
        _resolve_affinity,
    )
except ImportError:  # Direct execution from this directory.
    from runtime_common import (
        git_metadata,
        gpu_snapshot,
        host_metadata,
        logical_bandwidth_gbps,
        normalize_gpu_uuid,
        process_gpu_memory_mib,
        sha256_file,
        summarize_samples,
        working_set_bytes,
        write_csv,
        write_json,
    )
    from single_kernel_microbench import (
        QUALIFICATION_MAX_CPU_UTIL_PERCENT,
        QUALIFICATION_MAX_GPU_TEMPERATURE_C,
        QUALIFICATION_MAX_GPU_UTIL_PERCENT,
        QUALIFICATION_MAX_CV_PERCENT,
        QUALIFICATION_MINIMUMS,
        _ExclusiveBenchmarkLock,
        _apply_affinity,
        _noise_observation,
        _resolve_affinity,
    )


SCHEMA = "taichi_forge.warp_external_baseline.v1"
PRESETS = {
    "small": 65_536,
    "medium": 1_048_576,
    "large": 16_777_216,
}


def qualification_policy_errors(args: argparse.Namespace) -> list[str]:
    if args.intent != "qualification":
        return []
    errors = []
    for name in ("samples", "warmups", "target_sample_ms",
                 "stability_replays"):
        if getattr(args, name) < QUALIFICATION_MINIMUMS[name]:
            errors.append(
                f"{name}={getattr(args, name)} is below qualification minimum "
                f"{QUALIFICATION_MINIMUMS[name]}")
    if args.cpu_affinity == "none":
        errors.append("qualification requires explicit or automatic CPU affinity")
    if args.max_cpu_util > QUALIFICATION_MAX_CPU_UTIL_PERCENT:
        errors.append("qualification CPU-utilization ceiling is too permissive")
    if args.max_gpu_util > QUALIFICATION_MAX_GPU_UTIL_PERCENT:
        errors.append("qualification GPU-utilization ceiling is too permissive")
    if args.max_gpu_temp > QUALIFICATION_MAX_GPU_TEMPERATURE_C:
        errors.append("qualification GPU-temperature ceiling is too permissive")
    return errors


def select_calibrated_batch(elapsed_ms: float, batch_size: int,
                            target_ms: float, maximum: int = 65_536) -> int:
    if elapsed_ms < 0.0 or batch_size <= 0 or target_ms <= 0.0:
        raise ValueError("calibration inputs must be non-negative and positive")
    estimate = (batch_size * 2 if elapsed_ms == 0.0 else
                math.ceil(batch_size * target_ms / elapsed_ms))
    return min(maximum, max(batch_size * 2, estimate))


def _windows_process_ancestors(pid: int) -> list[int]:
    """Returns only proven parent PIDs from a Toolhelp process snapshot.

    Windows venv launchers may remain as a Python parent while the base
    interpreter runs the program.  Ignoring that proven ancestor avoids a
    false self-conflict without allowing an unrelated same-path process.
    """
    if os.name != "nt":
        return []
    from ctypes import wintypes

    class ProcessEntry32(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.Process32FirstW.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(ProcessEntry32)]
    kernel32.Process32FirstW.restype = wintypes.BOOL
    kernel32.Process32NextW.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(ProcessEntry32)]
    kernel32.Process32NextW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
    if snapshot == wintypes.HANDLE(-1).value:
        return []
    parents: dict[int, int] = {}
    try:
        entry = ProcessEntry32()
        entry.dwSize = ctypes.sizeof(entry)
        if kernel32.Process32FirstW(snapshot, ctypes.byref(entry)):
            while True:
                parents[int(entry.th32ProcessID)] = int(
                    entry.th32ParentProcessID)
                if not kernel32.Process32NextW(snapshot, ctypes.byref(entry)):
                    break
    finally:
        kernel32.CloseHandle(snapshot)
    ancestors = []
    seen = {pid}
    current = pid
    while current in parents:
        parent = parents[current]
        if parent <= 0 or parent in seen:
            break
        ancestors.append(parent)
        seen.add(parent)
        current = parent
    return ancestors


def _path_in_prefix(path: Path, prefix: Path) -> bool:
    try:
        path.resolve().relative_to(prefix.resolve())
        return True
    except ValueError:
        return False


def _environment_provenance(wp: Any, np: Any) -> dict[str, Any]:
    prefix = Path(sys.prefix).resolve()
    package_path = Path(wp.__file__).resolve()
    numpy_path = Path(np.__file__).resolve()
    try:
        distribution_version = importlib_metadata.version("warp-lang")
    except importlib_metadata.PackageNotFoundError:
        distribution_version = None
    external_site_paths = []
    for entry in sys.path:
        if entry and "site-packages" in entry.lower():
            path = Path(entry).resolve()
            if not _path_in_prefix(path, prefix):
                external_site_paths.append(str(path))
    result = {
        "sys_prefix": str(prefix),
        "sys_base_prefix": str(Path(sys.base_prefix).resolve()),
        "venv_active": prefix != Path(sys.base_prefix).resolve(),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "warp_distribution_version": distribution_version,
        "warp_module_version": str(wp.__version__),
        "warp_package_path": str(package_path),
        "warp_package_inside_environment": _path_in_prefix(package_path, prefix),
        "warp_init_path": str(package_path),
        "warp_init_sha256": sha256_file(package_path),
        "numpy_version": str(np.__version__),
        "numpy_package_path": str(numpy_path),
        "numpy_package_inside_environment": _path_in_prefix(numpy_path, prefix),
        "external_site_paths": external_site_paths,
        "python_no_user_site": os.environ.get("PYTHONNOUSERSITE") == "1",
        "pythonpath_present": bool(os.environ.get("PYTHONPATH")),
    }
    result["isolated"] = bool(
        result["venv_active"]
        and result["warp_package_inside_environment"]
        and result["numpy_package_inside_environment"]
        and not external_site_paths
        and result["python_no_user_site"]
        and not result["pythonpath_present"]
    )
    return result


def _device_identity(device: Any) -> dict[str, Any]:
    rows = gpu_snapshot()
    runtime_uuid = normalize_gpu_uuid(getattr(device, "uuid", None))
    normalized_rows = [{
        "index": row.get("index"),
        "name": row.get("name"),
        "uuid": row.get("uuid"),
        "normalized_uuid": normalize_gpu_uuid(row.get("uuid")),
    } for row in rows]
    matches = [row for row in normalized_rows
               if row["normalized_uuid"] == runtime_uuid]
    return {
        "device_alias": str(device.alias),
        "device_name": str(device.name),
        "arch": int(device.arch),
        "runtime_uuid": runtime_uuid,
        "total_memory_bytes": int(device.total_memory),
        "mempool_enabled": bool(device.is_mempool_enabled),
        "nvidia_smi_devices": normalized_rows,
        "matching_devices": matches,
        "binding_verified": len(matches) == 1,
    }


def _memory_observation(wp: Any, device: Any) -> dict[str, Any]:
    return {
        "rss_bytes": working_set_bytes(),
        "process_gpu_memory_mib": process_gpu_memory_mib(os.getpid()),
        "mempool_used_current_bytes": int(
            wp.get_mempool_used_mem_current(device)),
        "mempool_used_high_bytes": int(wp.get_mempool_used_mem_high(device)),
        "device_free_memory_bytes": int(device.free_memory),
    }


def _memory_plateau(before: dict[str, Any],
                    after: dict[str, Any]) -> dict[str, Any]:
    def delta(name: str) -> float | int | None:
        left, right = before.get(name), after.get(name)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return right - left
        return None

    rss_delta = delta("rss_bytes")
    gpu_delta = delta("process_gpu_memory_mib")
    mempool_delta = delta("mempool_used_current_bytes")
    return {
        "rss_delta_bytes": rss_delta,
        "process_gpu_memory_delta_mib": gpu_delta,
        "mempool_used_current_delta_bytes": mempool_delta,
        "passed": bool(
            (rss_delta is None or rss_delta <= 64 * 1024 * 1024)
            and (gpu_delta is None or gpu_delta <= 64.0)
            and mempool_delta is not None
            and mempool_delta <= 0
        ),
    }


def _timed_batch(wp: Any, device: Any, launch: Callable[[], None],
                 batch_size: int) -> float:
    wp.synchronize_device(device)
    started = time.perf_counter_ns()
    for _ in range(batch_size):
        launch()
    wp.synchronize_device(device)
    return (time.perf_counter_ns() - started) / 1.0e6


def _calibrate(wp: Any, device: Any, launch: Callable[[], None],
               target_ms: float) -> tuple[int, list[dict[str, Any]]]:
    batch_size = 1
    attempts = []
    while True:
        elapsed_ms = _timed_batch(wp, device, launch, batch_size)
        attempts.append({"batch_size": batch_size, "elapsed_ms": elapsed_ms})
        if elapsed_ms >= target_ms or batch_size >= 65_536:
            return batch_size, attempts
        batch_size = select_calibrated_batch(
            elapsed_ms, batch_size, target_ms)


def _write_reports(output_dir: Path, result: dict[str, Any]) -> None:
    summary = result.get("summary") or {}
    stability = result.get("stability") or {}
    readiness = result.get("ready_for_external_absolute_baseline", False)
    status_zh = "通过" if readiness else "未通过"
    status_en = "pass" if readiness else "fail"
    median = summary.get("median_ms")
    p95 = summary.get("p95_ms")
    cv = summary.get("cv_percent")
    median_text = "n/a" if median is None else f"{median:.6f} ms"
    p95_text = "n/a" if p95 is None else f"{p95:.6f} ms"
    cv_text = "n/a" if cv is None else f"{cv:.3f}%"
    (output_dir / "report.zh-CN.md").write_text(
        "# Warp 外部绝对基线：i32 affine transform\n\n"
        f"- 运行状态：`{result['status']}`\n"
        f"- 外部绝对基线发布资格：{status_zh}\n"
        f"- Warp：`{result.get('warp_version', 'n/a')}`\n"
        f"- 设备：`{result.get('device_identity', {}).get('device_name', 'n/a')}`\n"
        f"- 元素数：{result.get('elements', 'n/a')}\n"
        f"- batch size：{result.get('batch_size', 'n/a')}\n"
        f"- 中位数：{median_text}/launch\n"
        f"- p95：{p95_text}/launch\n"
        f"- CV：{cv_text}\n"
        f"- 稳定性回放：{stability.get('replays', 0)}\n"
        f"- 内存平台检查：{'通过' if stability.get('memory_plateau', {}).get('passed') else '失败'}\n\n"
        "## 解释边界\n\n"
        "该结果只是一项 Warp 绝对基线。它不属于 Forge/vanilla 的同 API 成对"
        "统计，不生成或支持总体 Forge-vs-Warp 加速比。初始化、JIT 编译、传输和"
        "正确性检查均不在稳态 kernel 计时内，但单独记录。\n",
        encoding="utf-8")
    (output_dir / "report.en.md").write_text(
        "# Warp external absolute baseline: i32 affine transform\n\n"
        f"- Run status: `{result['status']}`\n"
        f"- External absolute-baseline eligibility: {status_en}\n"
        f"- Warp: `{result.get('warp_version', 'n/a')}`\n"
        f"- Device: `{result.get('device_identity', {}).get('device_name', 'n/a')}`\n"
        f"- Elements: {result.get('elements', 'n/a')}\n"
        f"- Batch size: {result.get('batch_size', 'n/a')}\n"
        f"- Median: {median_text}/launch\n"
        f"- p95: {p95_text}/launch\n"
        f"- CV: {cv_text}\n"
        f"- Stability replays: {stability.get('replays', 0)}\n"
        f"- Memory plateau: {'pass' if stability.get('memory_plateau', {}).get('passed') else 'fail'}\n\n"
        "## Interpretation boundary\n\n"
        "This is an absolute Warp baseline only. It is not part of the paired, "
        "same-API Forge/vanilla statistics and does not create or support an "
        "overall Forge-vs-Warp speedup. Initialization, JIT compilation, transfer, "
        "and correctness checks are excluded from steady-state kernel timing and "
        "are recorded separately.\n",
        encoding="utf-8")


def _run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import warp as wp

    cache_dir = (Path(__file__).resolve().parents[2] / "temp_outputs" /
                 "qualification" / "warp_kernel_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    wp.config.kernel_cache_dir = str(cache_dir)
    affinity = _apply_affinity(_resolve_affinity(
        args.cpu_affinity, args.cpu_threads))
    environment = _environment_provenance(wp, np)
    device = wp.get_device("cuda:0")
    identity = _device_identity(device)
    elements = PRESETS[args.preset]
    host = ((np.arange(elements, dtype=np.int64) % 1009) - 504).astype(np.int32)
    expected = (host * np.int32(3) + np.int32(7)).astype(np.int32)

    @wp.kernel
    def affine_transform(source: wp.array(dtype=wp.int32),
                         destination: wp.array(dtype=wp.int32)):
        index = wp.tid()
        destination[index] = source[index] * 3 + 7

    allocation_started = time.perf_counter_ns()
    source = wp.array(host, dtype=wp.int32, device=device)
    destination = wp.zeros(elements, dtype=wp.int32, device=device)
    wp.synchronize_device(device)
    allocation_ms = (time.perf_counter_ns() - allocation_started) / 1.0e6

    def launch() -> None:
        wp.launch(affine_transform, dim=elements, inputs=[source, destination],
                  device=device)

    first_call_started = time.perf_counter_ns()
    launch()
    wp.synchronize_device(device)
    first_call_ms = (time.perf_counter_ns() - first_call_started) / 1.0e6
    actual = destination.numpy()
    mismatch_before = int(np.count_nonzero(actual != expected))
    if mismatch_before:
        raise RuntimeError(f"pre-timing exact oracle failed: {mismatch_before}")

    batch_size, calibration = _calibrate(
        wp, device, launch, args.target_sample_ms)
    for _ in range(args.warmups):
        _timed_batch(wp, device, launch, batch_size)
    raw_batch_ms = [
        _timed_batch(wp, device, launch, batch_size)
        for _ in range(args.samples)
    ]
    samples_ms = [elapsed / batch_size for elapsed in raw_batch_ms]
    summary = summarize_samples(samples_ms)
    summary["logical_bandwidth_gbps_at_median"] = logical_bandwidth_gbps(
        elements * 8, float(summary["median_ms"]))

    stability_before = _memory_observation(wp, device)
    windows = []
    completed = 0
    while completed < args.stability_replays:
        count = min(args.stability_checkpoint,
                    args.stability_replays - completed)
        elapsed = _timed_batch(wp, device, launch, count)
        windows.append(elapsed / count)
        completed += count
    stability_after = _memory_observation(wp, device)
    plateau = _memory_plateau(stability_before, stability_after)

    wp.synchronize_device(device)
    actual_after = destination.numpy()
    mismatch_after = int(np.count_nonzero(actual_after != expected))
    result = {
        "schema": SCHEMA,
        "status": "passed",
        "runtime": "warp",
        "comparison_class": "external-absolute-baseline",
        "case_id": "EXTERNAL-001-THIN-002-TRANSFORM",
        "preset": args.preset,
        "intent": args.intent,
        "elements": elements,
        "dtype": "i32",
        "semantics": "dst_i_equals_src_i_times_3_plus_7",
        "timing_scope": "repeated kernel launches plus one outer device sync",
        "excluded_from_timing": [
            "Warp import and initialization", "allocation", "host-device transfer",
            "first call and JIT compilation", "correctness copies and oracle",
        ],
        "warp_version": str(wp.__version__),
        "kernel_cache_dir": str(cache_dir),
        "environment": environment,
        "affinity": affinity,
        "device_identity": identity,
        "allocation_and_input_transfer_ms": allocation_ms,
        "first_call_and_jit_ms": first_call_ms,
        "batch_size": batch_size,
        "calibration": calibration,
        "raw_batch_ms": raw_batch_ms,
        "samples_ms": samples_ms,
        "summary": summary,
        "correctness": {
            "comparison": "exact_i32_affine_transform",
            "mismatch_before": mismatch_before,
            "mismatch_after": mismatch_after,
            "passed": mismatch_before == 0 and mismatch_after == 0,
        },
        "stability": {
            "replays": args.stability_replays,
            "checkpoint": args.stability_checkpoint,
            "window_per_launch_ms": windows,
            "window_summary": summarize_samples(windows),
            "memory_before": stability_before,
            "memory_after": stability_after,
            "memory_plateau": plateau,
        },
        "measurement_config": {
            "samples": args.samples,
            "warmups": args.warmups,
            "target_sample_ms": args.target_sample_ms,
            "stability_replays": args.stability_replays,
            "stability_checkpoint": args.stability_checkpoint,
            "cpu_threads": args.cpu_threads,
            "cpu_affinity": args.cpu_affinity,
        },
        "workload_contract": {
            "same_as_taichi_case": "THIN-002-TRANSFORM",
            "same_elements_dtype_input_semantics": True,
            "same_outer_synchronization": True,
            "exact_oracle": True,
            "framework_api_equivalent": False,
        },
    }
    method_checks = {
        "isolated_environment": environment["isolated"],
        "physical_device_binding": identity["binding_verified"],
        "correctness": result["correctness"]["passed"],
        "scored_timing_window": (
            float(summary["median_ms"]) * batch_size >= args.target_sample_ms),
        "stability_complete": args.stability_replays > 0,
        "memory_plateau": plateau["passed"],
    }
    result["method_checks"] = method_checks
    result["ready_for_external_absolute_baseline"] = bool(
        args.intent == "qualification"
        and not qualification_policy_errors(args)
        and all(method_checks.values())
        and float(summary["cv_percent"]) <= QUALIFICATION_MAX_CV_PERCENT
    )
    del destination
    del source
    gc.collect()
    wp.synchronize_device(device)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=tuple(PRESETS), default="small")
    parser.add_argument("--intent", choices=("diagnostic", "qualification"),
                        default="diagnostic")
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--target-sample-ms", type=float, default=100.0)
    parser.add_argument("--stability-replays", type=int, default=100)
    parser.add_argument("--stability-checkpoint", type=int, default=10)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--cpu-affinity", default="auto")
    parser.add_argument("--max-cpu-util", type=float, default=20.0)
    parser.add_argument("--max-gpu-util", type=float, default=15.0)
    parser.add_argument("--max-gpu-temp", type=float, default=65.0)
    parser.add_argument("--output-root", default="temp_outputs/qualification/results")
    parser.add_argument("--run-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    errors = qualification_policy_errors(args)
    if errors:
        raise ValueError("; ".join(errors))
    if args.samples <= 0 or args.warmups < 0 or args.target_sample_ms <= 0:
        raise ValueError("samples/target must be positive and warmups non-negative")
    if args.stability_replays <= 0 or args.stability_checkpoint <= 0:
        raise ValueError("stability replay settings must be positive")
    repo_root = Path(__file__).resolve().parents[2]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"warp-transform-{args.preset}-{timestamp}"
    if Path(run_id).name != run_id:
        raise ValueError("run_id must be one path component")
    output_dir = (repo_root / args.output_root / run_id).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "run_id": run_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_metadata(repo_root),
        "host": host_metadata(),
        "config": vars(args),
        "noise_observations": [],
    }
    try:
        with _ExclusiveBenchmarkLock() as lock:
            manifest["exclusive_driver_lock"] = lock
            ignored_pids = [os.getpid(), *_windows_process_ancestors(os.getpid())]
            manifest["noise_ignored_process_lineage"] = ignored_pids
            before = _noise_observation(
                "cuda", ignored_pids, args.max_cpu_util,
                args.max_gpu_util, args.max_gpu_temp)
            manifest["noise_observations"].append({"label": "before", **before})
            write_json(output_dir / "manifest.json", manifest)
            if not before["passed"]:
                raise RuntimeError("noise admission failed before run: " +
                                   "; ".join(before["reasons"]))
            result = _run(args)
            after = _noise_observation(
                "cuda", ignored_pids, args.max_cpu_util,
                args.max_gpu_util, args.max_gpu_temp)
            manifest["noise_observations"].append({"label": "after", **after})
            result["noise_admission"] = {
                "before": before,
                "after": after,
                "passed": before["passed"] and after["passed"],
            }
            result["method_checks"]["exclusive_driver_lock"] = True
            result["method_checks"]["noise_admission"] = after["passed"]
            result["ready_for_external_absolute_baseline"] = bool(
                result["ready_for_external_absolute_baseline"]
                and after["passed"])
            manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(output_dir / "manifest.json", manifest)
            write_json(output_dir / "result.json", result)
            write_csv(output_dir / "samples.csv", [
                {"sample": index + 1, "batch_ms": batch,
                 "per_launch_ms": per_launch}
                for index, (batch, per_launch) in enumerate(zip(
                    result["raw_batch_ms"], result["samples_ms"]))
            ])
            _write_reports(output_dir, result)
            print(json.dumps({
                "run_id": run_id,
                "status": result["status"],
                "median_ms": result["summary"]["median_ms"],
                "p95_ms": result["summary"]["p95_ms"],
                "cv_percent": result["summary"]["cv_percent"],
                "ready": result["ready_for_external_absolute_baseline"],
                "output_dir": str(output_dir),
            }, sort_keys=True))
            return 0
    except Exception as error:
        failure = {
            "schema": SCHEMA,
            "run_id": run_id,
            "status": "failed",
            "reason": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "ready_for_external_absolute_baseline": False,
        }
        manifest["failure"] = failure
        write_json(output_dir / "manifest.json", manifest)
        write_json(output_dir / "failure.json", failure)
        (output_dir / "failure.zh-CN.md").write_text(
            "# Warp 外部基线运行失败\n\n"
            f"- 原因：`{failure['reason']}`\n"
            "- 发布资格：未通过\n",
            encoding="utf-8")
        (output_dir / "failure.en.md").write_text(
            "# Warp external baseline failed\n\n"
            f"- Reason: `{failure['reason']}`\n"
            "- Publication eligibility: fail\n",
            encoding="utf-8")
        print(json.dumps(failure, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
