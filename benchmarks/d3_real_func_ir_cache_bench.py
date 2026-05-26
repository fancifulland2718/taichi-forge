import argparse
import ctypes
import ctypes.wintypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_PREFIX = "D3_REAL_FUNC_IR "


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


def _process_rss_mb(pid):
    if os.name != "nt":
        return None
    access = 0x0400 | 0x0010
    handle = ctypes.windll.kernel32.OpenProcess(access, False, int(pid))
    if not handle:
        return None
    try:
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        if not ok:
            return None
        return float(counters.WorkingSetSize) / (1024.0 * 1024.0)
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def _child_env(pythonpath):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["TAICHI_OFFLINE_CACHE"] = "0"
    env["PYTHONPATH"] = pythonpath or str(ROOT / "python")
    return env


def _sample_process(proc, interval_s):
    samples = []
    deadline = time.perf_counter() + interval_s
    while True:
        ret = proc.poll()
        rss = _process_rss_mb(proc.pid)
        if rss is not None:
            samples.append(rss)
        if ret is not None:
            break
        remaining = deadline - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        deadline = time.perf_counter() + interval_s
    return samples


def _parse_prefixed(stdout):
    for line in stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX):])
    raise ValueError(f"missing result prefix {RESULT_PREFIX!r}")


def _mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["arch"], []).append(row)
    summary = []
    for arch, items in sorted(groups.items()):
        ok_items = [row for row in items if row.get("ok")]
        summary.append(
            {
                "arch": arch,
                "trials": len(items),
                "ok_trials": len(ok_items),
                "first_fast_ms_mean": _mean([row.get("first_fast_ms") for row in ok_items]),
                "first_full_ms_mean": _mean([row.get("first_full_ms") for row in ok_items]),
                "reset_fast_ms_mean": _mean([row.get("reset_fast_ms") for row in ok_items]),
                "warm_fast_ms_mean": _mean([row.get("warm_fast_ms") for row in ok_items]),
                "warm_full_ms_mean": _mean([row.get("warm_full_ms") for row in ok_items]),
                "variant_count_mean": _mean([row.get("variant_count") for row in ok_items]),
                "max_rss_mb": max(
                    [row.get("rss_peak_mb") for row in ok_items if row.get("rss_peak_mb") is not None],
                    default=None,
                ),
            }
        )
    return summary


def _write_outputs(out_dir, rows, summary):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rows.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# D3 real_func IR Cache Benchmark",
        "",
        "| arch | ok/trials | first fast ms | first full ms | reset fast ms | warm fast ms | warm full ms | variants | max RSS MB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['arch']} | {row['ok_trials']}/{row['trials']} | "
            f"{_fmt(row['first_fast_ms_mean'])} | {_fmt(row['first_full_ms_mean'])} | "
            f"{_fmt(row['reset_fast_ms_mean'])} | {_fmt(row['warm_fast_ms_mean'])} | "
            f"{_fmt(row['warm_full_ms_mean'])} | {_fmt(row['variant_count_mean'])} | "
            f"{_fmt(row['max_rss_mb'])} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value):
    if value is None:
        return ""
    return f"{float(value):.3f}"


def run_parent(args):
    rows = []
    for arch in args.arches:
        for trial in range(args.trials):
            cmd = [
                args.python,
                str(Path(__file__).resolve()),
                "--child",
                "--arch",
                arch,
                "--n",
                str(args.n),
                "--warmups",
                str(args.warmups),
                "--steps",
                str(args.steps),
            ]
            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                env=_child_env(args.pythonpath),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            rss = _sample_process(proc, args.sample_interval_s)
            stdout, stderr = proc.communicate()
            row = {"arch": arch, "trial": trial, "returncode": proc.returncode}
            row["rss_peak_mb"] = max(rss) if rss else None
            row["stderr_tail"] = "\n".join(stderr.splitlines()[-20:])
            if proc.returncode == 0:
                try:
                    row.update(_parse_prefixed(stdout))
                except Exception as exc:
                    row["ok"] = False
                    row["parse_error"] = str(exc)
                    row["stdout_tail"] = "\n".join(stdout.splitlines()[-20:])
            else:
                row["ok"] = False
                row["stdout_tail"] = "\n".join(stdout.splitlines()[-20:])
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    summary = _summarize(rows)
    _write_outputs(Path(args.out_dir), rows, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if any(not row.get("ok") for row in rows):
        raise SystemExit(1)


def _expected(n, salt):
    total = 0
    for i in range(n):
        acc = (i * 17 + salt * 13 + 5) % 1000003
        for k in range(6):
            acc = (acc * 3 + i * (k + 1) + salt + k * 11) % 1000003
        total += acc % 65521
    return int(total)


def run_child(args):
    import taichi_forge as ti

    arches = {"cpu": ti.cpu, "cuda": ti.cuda}
    ti.init(arch=arches[args.arch], offline_cache=False)

    @ti.real_func
    def heavy_value(i: ti.i32, salt: ti.i32) -> ti.i32:
        acc = (i * 17 + salt * 13 + 5) % 1000003
        for k in ti.static(range(6)):
            acc = (acc * 3 + i * (k + 1) + salt + k * 11) % 1000003
        return acc

    @ti.kernel(opt_level="fast")
    def run_fast(n: ti.i32, salt: ti.i32) -> ti.i32:
        total = 0
        for i in range(n):
            total += heavy_value(i, salt) % 65521
        return total

    @ti.kernel(opt_level="full")
    def run_full(n: ti.i32, salt: ti.i32) -> ti.i32:
        total = 0
        for i in range(n):
            total += heavy_value(i, salt) % 65521
        return total

    n = int(args.n)
    expected = _expected(n, 123)

    start = time.perf_counter()
    fast_value = int(run_fast(n, 123))
    first_fast_ms = (time.perf_counter() - start) * 1000.0
    variants_after_fast = len(heavy_value.func.taichi_functions)

    start = time.perf_counter()
    full_value = int(run_full(n, 123))
    first_full_ms = (time.perf_counter() - start) * 1000.0
    variants_after_full = len(heavy_value.func.taichi_functions)

    for _ in range(args.warmups):
        run_fast(n, 123)
        run_full(n, 123)

    fast_samples = []
    full_samples = []
    for _ in range(args.steps):
        start = time.perf_counter()
        run_fast(n, 123)
        fast_samples.append((time.perf_counter() - start) * 1000.0)
        start = time.perf_counter()
        run_full(n, 123)
        full_samples.append((time.perf_counter() - start) * 1000.0)

    ti.reset()
    ti.init(arch=arches[args.arch], offline_cache=False)
    start = time.perf_counter()
    reset_value = int(run_fast(n, 123))
    reset_fast_ms = (time.perf_counter() - start) * 1000.0
    variants_after_reset = len(heavy_value.func.taichi_functions)

    payload = {
        "arch": args.arch,
        "ok": bool(fast_value == expected and full_value == expected and reset_value == expected),
        "n": n,
        "first_fast_ms": first_fast_ms,
        "first_full_ms": first_full_ms,
        "reset_fast_ms": reset_fast_ms,
        "warm_fast_ms": sum(fast_samples) / len(fast_samples),
        "warm_full_ms": sum(full_samples) / len(full_samples),
        "variants_after_fast": variants_after_fast,
        "variants_after_full": variants_after_full,
        "variants_after_reset": variants_after_reset,
        "variant_count": variants_after_full,
    }
    print(RESULT_PREFIX + json.dumps(payload, sort_keys=True), flush=True)
    if not payload["ok"]:
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--arch", choices=["cpu", "cuda"])
    parser.add_argument("--arches", nargs="+", default=["cpu", "cuda"])
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--pythonpath", default=str(ROOT / "python"))
    parser.add_argument("--out-dir", default=str(ROOT / "benchmarks" / "results" / "d3_real_func_ir_cache"))
    parser.add_argument("--sample-interval-s", type=float, default=0.5)
    args = parser.parse_args()
    if args.child:
        run_child(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()