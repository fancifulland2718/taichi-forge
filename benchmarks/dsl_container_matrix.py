from __future__ import annotations

import argparse
import csv
import ctypes
import ctypes.wintypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_PREFIX = "DSL_CONTAINER_MATRIX "
OPS = [
    "scan",
    "reduce",
    "transform",
    "gather",
    "scatter",
    "scatter_add",
    "compact",
    "histogram",
    "bucket",
    "grouped_reduce",
    "sort",
]


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.wintypes.DWORD),
        ("PageFaultCount", ctypes.wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _process_rss_mb(pid: int) -> float | None:
    access = 0x0400 | 0x0010
    handle = ctypes.windll.kernel32.OpenProcess(access, False, int(pid))
    if not handle:
        return None
    try:
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        )
        if not ok:
            return None
        return float(counters.WorkingSetSize) / (1024.0 * 1024.0)
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def _gpu_process_dedicated_mb(pid: int) -> float | None:
    ps = (
        "$pidToFind = "
        + str(int(pid))
        + "; "
        "$pattern = 'pid_' + $pidToFind + '_*'; "
        "$sum = 0; "
        "try { "
        "  (Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage').CounterSamples | "
        "    Where-Object { $_.InstanceName -like $pattern } | "
        "    ForEach-Object { $sum += $_.CookedValue }; "
        "  [Console]::WriteLine([math]::Round($sum / 1MB, 3)) "
        "} catch { [Console]::WriteLine(-1) }"
    )
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", ps],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5.0,
        ).strip()
        value = float(out)
        return None if value < 0 else value
    except Exception:
        return None


def _child_env(pythonpath: str | None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["TAICHI_OFFLINE_CACHE"] = "0"
    env["PYTHONPATH"] = pythonpath or str(ROOT / "python")
    return env


def _parse_prefixed(stdout: str, prefix: str) -> dict:
    for line in stdout.splitlines():
        if line.startswith(prefix):
            return json.loads(line[len(prefix) :])
    raise ValueError(f"missing result prefix {prefix!r}")


def _parse_json_payload(stdout: str):
    candidates = [idx for idx, char in enumerate(stdout) if char == "["]
    for idx in reversed(candidates):
        try:
            payload = json.loads(stdout[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, list):
            return payload
    raise ValueError("missing JSON list payload")


def _build_cmd(args, container: str, arch: str, op: str, n: int) -> list[str]:
    if container == "field":
        field_op = "bucket_builder" if op == "bucket" else op
        cmd = [
            args.python,
            str(ROOT / "benchmarks" / "s4_dense_field_native_bench.py"),
            "--child",
            "--package",
            "forge",
            "--arch",
            arch,
            "--op",
            field_op,
            "--n",
            str(n),
            "--repeats",
            str(args.repeats),
            "--warmups",
            str(args.warmups),
            "--indices-storage",
            args.field_indices_storage,
        ]
    else:
        script = (
            "ndarray_primitives.py"
            if container == "ndarray"
            else "struct_ndarray_primitives.py"
        )
        cmd = [
            args.python,
            str(ROOT / "benchmarks" / script),
            "--arch",
            arch,
            "--primitive",
            op,
            "--sizes",
            str(n),
            "--repeats",
            str(args.repeats),
            "--warmups",
            str(args.warmups),
            "--method-mode",
            args.method_mode,
        ]
    if args.internal_stats:
        cmd.append("--internal-stats")
    return cmd


def _sample_process(proc: subprocess.Popen, arch: str, interval_s: float):
    rss_samples = []
    gpu_samples = []
    deadline = time.perf_counter() + interval_s
    while True:
        ret = proc.poll()
        rss = _process_rss_mb(proc.pid)
        if rss is not None:
            rss_samples.append(rss)
        if arch != "cpu":
            gpu = _gpu_process_dedicated_mb(proc.pid)
            if gpu is not None:
                gpu_samples.append(gpu)
        if ret is not None:
            break
        remaining = deadline - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        deadline = time.perf_counter() + interval_s
    return rss_samples, gpu_samples


def _sum_internal_counts(counts: dict, prefix: str) -> int:
    return int(sum(value for key, value in counts.items() if str(key).startswith(prefix)))


def _internal_summary(internal: dict) -> dict:
    primitive = internal.get("primitive", {}) or {}
    legacy = internal.get("legacy_helper_fallbacks", {}) or {}
    sync = internal.get("sync", {}) or {}
    return {
        "sync_count": int(sync.get("count", 0) or 0),
        "sync_extra_calls": int(sync.get("extra_calls", 0) or 0),
        "sync_total_ms": float(sync.get("total_ms", 0.0) or 0.0),
        "legacy_fallback_total": int(sum(legacy.values())) if legacy else 0,
        "native_plan_hot_hits": int(primitive.get("native_plan.lookup.hot_hit", 0) or 0),
        "native_plan_cache_hits": int(primitive.get("native_plan.lookup.cache_hit", 0) or 0),
        "native_plan_active_hits": int(primitive.get("native_plan.lookup.active_hit", 0) or 0),
        "native_plan_misses": _sum_internal_counts(primitive, "native_plan.lookup.miss"),
        "native_plan_records": int(primitive.get("native_plan.record.calls", 0) or 0),
        "native_group_hot_hits": int(primitive.get("native_plan_group.lookup.hot_hit", 0) or 0),
        "native_group_cache_hits": int(primitive.get("native_plan_group.lookup.cache_hit", 0) or 0),
        "native_group_active_hits": int(primitive.get("native_plan_group.lookup.active_hit", 0) or 0),
        "native_group_misses": _sum_internal_counts(primitive, "native_plan_group.lookup.miss"),
        "native_group_records": int(primitive.get("native_plan_group.record.calls", 0) or 0),
        "program_method_invokes": int(primitive.get("program_method.invoke.calls", 0) or 0),
        "capability_has_probes": int(primitive.get("program_capability.has.probes", 0) or 0),
        "capability_available_probes": int(primitive.get("program_capability.available.probes", 0) or 0),
        "capability_value_probes": int(primitive.get("program_capability.value_available.probes", 0) or 0),
    }
def _normalize_row(container: str, arch: str, op: str, n: int, raw: dict, rss, gpu):
    if container == "field":
        runtime = raw.get("runtime", {})
        workspace = raw.get("workspace_peak_bytes", 0)
        ok = True
    else:
        if raw.get("skipped"):
            return {
                "container": container,
                "arch": arch,
                "op": op,
                "n": n,
                "status": "skip",
                "ok": False,
                "skip_reason": "native method unavailable",
            }
        runtime = {
            "median_ms": raw.get("median_ms"),
            "mean_ms": raw.get("mean_ms"),
            "min_ms": raw.get("min_ms"),
            "max_ms": raw.get("max_ms"),
            "samples": None,
        }
        workspace = raw.get("workspace_peak", 0)
        ok = bool(raw.get("ok"))
    rss_peak = max(rss) if rss else None
    rss_first = rss[0] if rss else None
    gpu_peak = max(gpu) if gpu else None
    gpu_first = gpu[0] if gpu else None
    internal = raw.get("internal", {}) or {}
    internal_summary = _internal_summary(internal)
    return {
        "container": container,
        "arch": arch,
        "op": op,
        "n": n,
        "status": "ok" if ok else "wrong",
        "ok": ok,
        "first_call_ms": raw.get("first_call_ms"),
        "runtime": runtime,
        "workspace_peak_bytes": int(workspace or 0),
        "internal": internal,
        "internal_summary": internal_summary,
        "process_rss_mb": {
            "first": rss_first,
            "peak": rss_peak,
            "peak_delta": None if rss_first is None or rss_peak is None else rss_peak - rss_first,
        },
        "gpu_dedicated_mb": {
            "first": gpu_first,
            "peak": gpu_peak,
            "peak_delta": None if gpu_first is None or gpu_peak is None else gpu_peak - gpu_first,
        },
    }


def _run_case(args, out_dir: Path, container: str, arch: str, op: str, n: int):
    cmd = _build_cmd(args, container, arch, op, n)
    stem = f"{container}_{arch}_{op}_{n}"
    print("RUN " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=_child_env(args.pythonpath),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    start = time.perf_counter()
    rss_samples = []
    gpu_samples = []
    timed_out = False
    while proc.poll() is None:
        if time.perf_counter() - start > args.timeout_s:
            proc.kill()
            timed_out = True
            break
        rss = _process_rss_mb(proc.pid)
        if rss is not None:
            rss_samples.append(rss)
        if arch != "cpu" and args.sample_gpu_memory:
            gpu = _gpu_process_dedicated_mb(proc.pid)
            if gpu is not None:
                gpu_samples.append(gpu)
        time.sleep(args.sample_interval_s)
    stdout, stderr = proc.communicate()
    rss = _process_rss_mb(proc.pid)
    if rss is not None:
        rss_samples.append(rss)
    if arch != "cpu" and args.sample_gpu_memory:
        gpu = _gpu_process_dedicated_mb(proc.pid)
        if gpu is not None:
            gpu_samples.append(gpu)
    (out_dir / f"{stem}.stdout.txt").write_text(stdout, encoding="utf-8")
    (out_dir / f"{stem}.stderr.txt").write_text(stderr, encoding="utf-8")
    if timed_out:
        return None, {"container": container, "arch": arch, "op": op, "n": n, "reason": "timeout"}
    if proc.returncode != 0:
        return None, {
            "container": container,
            "arch": arch,
            "op": op,
            "n": n,
            "returncode": proc.returncode,
            "stderr_tail": stderr[-4000:],
        }
    try:
        if container == "field":
            raw = _parse_prefixed(stdout, "S4_DENSE_FIELD_BENCH ")
        else:
            payload = _parse_json_payload(stdout)
            raw = payload[0] if payload else {}
        row = _normalize_row(container, arch, op, n, raw, rss_samples, gpu_samples)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return None, {
            "container": container,
            "arch": arch,
            "op": op,
            "n": n,
            "reason": "parse_error",
            "error": str(exc),
            "stderr_tail": stderr[-4000:],
        }
    runtime = row.get("runtime", {}) or {}
    print(
        f"{row['status'].upper()} {container} {arch} {op} n={n} "
        f"first={row.get('first_call_ms')} median={runtime.get('median_ms')} "
        f"workspace={row.get('workspace_peak_bytes')}",
        flush=True,
    )
    return row, None

def _write_csv(rows, path: Path):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "container",
                "arch",
                "op",
                "n",
                "status",
                "ok",
                "first_call_ms",
                "runtime_median_ms",
                "runtime_mean_ms",
                "runtime_min_ms",
                "runtime_max_ms",
                "workspace_peak_bytes",
                "process_peak_mb",
                "process_peak_delta_mb",
                "gpu_peak_mb",
                "gpu_peak_delta_mb",
            ],
        )
        writer.writeheader()
        for row in rows:
            runtime = row.get("runtime", {}) or {}
            proc = row.get("process_rss_mb", {}) or {}
            gpu = row.get("gpu_dedicated_mb", {}) or {}
            writer.writerow(
                {
                    "container": row.get("container"),
                    "arch": row.get("arch"),
                    "op": row.get("op"),
                    "n": row.get("n"),
                    "status": row.get("status"),
                    "ok": row.get("ok"),
                    "first_call_ms": row.get("first_call_ms"),
                    "runtime_median_ms": runtime.get("median_ms"),
                    "runtime_mean_ms": runtime.get("mean_ms"),
                    "runtime_min_ms": runtime.get("min_ms"),
                    "runtime_max_ms": runtime.get("max_ms"),
                    "workspace_peak_bytes": row.get("workspace_peak_bytes"),
                    "process_peak_mb": proc.get("peak"),
                    "process_peak_delta_mb": proc.get("peak_delta"),
                    "gpu_peak_mb": gpu.get("peak"),
                    "gpu_peak_delta_mb": gpu.get("peak_delta"),
                }
            )


def run_matrix(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []
    for container in args.containers:
        for arch in args.arches:
            for op in args.ops:
                for n in args.sizes:
                    row, failure = _run_case(args, out_dir, container, arch, op, int(n))
                    if row is not None:
                        rows.append(row)
                    if failure is not None:
                        print(f"FAIL {container} {arch} {op} n={n}: {failure.get('reason', failure.get('returncode'))}", flush=True)
                        failures.append(failure)
    summary = {
        "method_mode": args.method_mode,
        "containers": args.containers,
        "arches": args.arches,
        "ops": args.ops,
        "sizes": args.sizes,
        "repeats": args.repeats,
        "warmups": args.warmups,
        "rows": rows,
        "failures": failures,
    }
    summary_path = out_dir / "summary.json"
    csv_path = out_dir / "summary.csv"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(rows, csv_path)
    print(f"WROTE {summary_path}")
    print(f"WROTE {csv_path}")
    if failures:
        print(json.dumps({"failures": failures}, indent=2, ensure_ascii=False))
        return 1
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--pythonpath", default=str(ROOT / "python"))
    parser.add_argument("--out-dir", default=str(ROOT / "benchmarks" / "results" / "dsl_container_matrix"))
    parser.add_argument("--arches", nargs="+", default=["cpu", "cuda", "vulkan"], choices=["cpu", "cuda", "vulkan"])
    parser.add_argument("--containers", nargs="+", default=["field", "ndarray", "struct_ndarray"], choices=["field", "ndarray", "struct_ndarray"])
    parser.add_argument("--ops", nargs="+", default=OPS, choices=OPS)
    parser.add_argument("--sizes", nargs="+", type=int, default=[4096, 65536])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--method-mode", choices=["native", "auto"], default="auto")
    parser.add_argument("--field-indices-storage", choices=["ndarray", "field"], default="field")
    parser.add_argument("--timeout-s", type=float, default=240.0)
    parser.add_argument("--sample-interval-s", type=float, default=0.25)
    parser.add_argument("--sample-gpu-memory", action="store_true")
    parser.add_argument("--internal-stats", action="store_true")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    return run_matrix(args)


if __name__ == "__main__":
    raise SystemExit(main())