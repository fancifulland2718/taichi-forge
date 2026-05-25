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
RESULT_PREFIX = "D3_FUNC_CACHE "


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
    if os.name != "nt":
        return None
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


def _child_env(pythonpath: str | None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["TAICHI_OFFLINE_CACHE"] = "0"
    env["PYTHONPATH"] = pythonpath or str(ROOT / "python")
    env["TI_FUNC_EXPANSION_PROFILE"] = "1"
    return env


def _parse_prefixed(stdout: str) -> dict:
    for line in stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX) :])
    raise ValueError(f"missing result prefix {RESULT_PREFIX!r}")


def _sample_process(proc: subprocess.Popen, interval_s: float):
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


def _mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _median(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def _summarize(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        groups.setdefault(row["arch"], []).append(row)
    out = []
    for arch, items in sorted(groups.items()):
        ok_items = [row for row in items if row.get("ok")]
        out.append(
            {
                "arch": arch,
                "trials": len(items),
                "ok_trials": len(ok_items),
                "first_call_total_ms_mean": _mean(
                    [row.get("first_call_total_ms") for row in ok_items]
                ),
                "first_call_total_ms_median": _median(
                    [row.get("first_call_total_ms") for row in ok_items]
                ),
                "warm_step_ms_mean": _mean([row.get("warm_step_ms") for row in ok_items]),
                "warm_step_ms_median": _median(
                    [row.get("warm_step_ms") for row in ok_items]
                ),
                "max_rss_mb": max(
                    [row.get("rss_peak_mb") for row in ok_items if row.get("rss_peak_mb") is not None],
                    default=None,
                ),
                "func_expansion_total_us_mean": _mean(
                    [row.get("func_expansion_total_us") for row in ok_items]
                ),
                "func_expansion_calls_mean": _mean(
                    [row.get("func_expansion_calls") for row in ok_items]
                ),
            }
        )
    return out


def _write_outputs(out_dir: Path, rows: list[dict], summary: list[dict]):
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "rows.json").open("w", encoding="utf-8") as fp:
        json.dump(rows, fp, indent=2, sort_keys=True)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, sort_keys=True)
    csv_fields = [
        "arch",
        "trial",
        "ok",
        "first_call_total_ms",
        "warm_step_ms",
        "rss_peak_mb",
        "func_expansion_total_us",
        "func_expansion_calls",
        "stderr_tail",
    ]
    with (out_dir / "rows.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in csv_fields})
    with (out_dir / "summary.md").open("w", encoding="utf-8") as fp:
        fp.write("# D3 @ti.func Cache Benchmark\n\n")
        fp.write(
            "| arch | ok/trials | first total ms mean | warm step ms mean | "
            "func expansion us mean | expansion calls mean | max RSS MB |\n"
        )
        fp.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in summary:
            fp.write(
                f"| {row['arch']} | {row['ok_trials']}/{row['trials']} | "
                f"{_fmt(row['first_call_total_ms_mean'])} | "
                f"{_fmt(row['warm_step_ms_mean'])} | "
                f"{_fmt(row['func_expansion_total_us_mean'])} | "
                f"{_fmt(row['func_expansion_calls_mean'])} | "
                f"{_fmt(row['max_rss_mb'])} |\n"
            )


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
                "--extra-globals",
                str(args.extra_globals),
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
                except Exception as exc:  # pylint: disable=broad-except
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


def _heavy_inner_py(a: int, b: int, salt: int) -> int:
    acc = (a * 17 + b * 13 + salt * 7 + 97) % 1000003
    for k in range(6):
        t = (acc + b * (k + 3) + salt * (k + 5) + k * 11) % 1000003
        if t % 5 < 2:
            acc = (acc + t * 3 + k * 19 + 23) % 1000003
        else:
            acc = (acc + t * 2 + b + k * 7 + 31) % 1000003
    return int(acc)


def _heavy_cell_py(x: int, y: int, salt: int) -> int:
    acc = (x + y * 5 + salt * 3 + 101) % 1000003
    for k in range(4):
        acc = _heavy_inner_py(acc, y + k, salt + k * 3)
        t = (acc + x * (k + 1) + salt) % 1000003
        if t % 7 < 3:
            acc = (acc + t + k * 29) % 1000003
        else:
            acc = (acc + t * 2 + y + k * 17) % 1000003
    return int(acc)


def _expected_values(inp, n: int, mode: int, salt: int):
    out = []
    checksum = 0
    for i in range(n):
        v = int(inp[i])
        x = _heavy_cell_py(v, i + mode, salt + mode * 17)
        y = _heavy_cell_py(x, v + mode, salt + mode * 19 + 5)
        z = _heavy_cell_py(y, i + v + mode, salt + mode * 23 + 11)
        value = (x + y * 3 + z * 5 + mode * 101) % 1000003
        out.append(value)
        checksum += value % 65521
    return out, int(checksum)


def run_child(args):
    import numpy as np
    import taichi_forge as ti
    from taichi_forge.lang import impl

    arches = {"cpu": ti.cpu, "cuda": ti.cuda, "vulkan": ti.vulkan}
    ti.init(arch=arches[args.arch], offline_cache=False)

    for i in range(int(args.extra_globals)):
        globals()[f"_d3_extra_global_{i}"] = i

    @ti.func
    def heavy_inner(a, b, salt):
        acc = (a * 17 + b * 13 + salt * 7 + 97) % 1000003
        for k in ti.static(range(6)):
            t = (acc + b * (k + 3) + salt * (k + 5) + k * 11) % 1000003
            if t % 5 < 2:
                acc = (acc + t * 3 + k * 19 + 23) % 1000003
            else:
                acc = (acc + t * 2 + b + k * 7 + 31) % 1000003
        return acc

    @ti.func
    def heavy_cell(x, y, salt):
        acc = (x + y * 5 + salt * 3 + 101) % 1000003
        for k in ti.static(range(4)):
            acc = heavy_inner(acc, y + k, salt + k * 3)
            t = (acc + x * (k + 1) + salt) % 1000003
            if t % 7 < 3:
                acc = (acc + t + k * 29) % 1000003
            else:
                acc = (acc + t * 2 + y + k * 17) % 1000003
        return acc

    n = int(args.n)
    inp_np = ((np.arange(n, dtype=np.int32) * 13 + 7) % 1000003).astype(np.int32)
    inp = ti.field(ti.i32, shape=n)
    out = ti.field(ti.i32, shape=n)
    inp.from_numpy(inp_np)

    @ti.kernel
    def d3_kernel(n: ti.i32, salt: ti.i32, mode: ti.template()) -> ti.i32:
        checksum = 0
        for i in range(n):
            v = inp[i]
            x = heavy_cell(v, i + mode, salt + mode * 17)
            y = heavy_cell(x, v + mode, salt + mode * 19 + 5)
            z = heavy_cell(y, i + v + mode, salt + mode * 23 + 11)
            value = (x + y * 3 + z * 5 + mode * 101) % 1000003
            out[i] = value
            checksum += value % 65521
        return checksum

    modes = list(range(4))
    first_calls = []
    checksums = {}
    for mode in modes:
        start = time.perf_counter()
        checksums[mode] = int(d3_kernel(n, 12345, mode))
        first_calls.append((time.perf_counter() - start) * 1000.0)

    out_np = out.to_numpy()
    expected, checksum = _expected_values(inp_np, n, modes[-1], 12345)
    correct = (
        checksums[modes[-1]] == checksum
        and np.array_equal(out_np.astype(np.int64), np.array(expected, dtype=np.int64))
    )

    for _ in range(args.warmups):
        for mode in modes:
            d3_kernel(n, 12345, mode)

    step_samples = []
    for _ in range(args.steps):
        start = time.perf_counter()
        for mode in modes:
            d3_kernel(n, 12345, mode)
        step_samples.append((time.perf_counter() - start) * 1000.0)

    stats = impl.get_runtime()._ti_func_expansion_stats
    expansion_total_us = 0.0
    expansion_calls = 0
    for entry in stats.values():
        expansion_total_us += float(entry.get("cumulative_ns", 0)) / 1000.0
        expansion_calls += int(entry.get("call_count", 0))

    payload = {
        "arch": args.arch,
        "ok": bool(correct),
        "n": n,
        "first_call_ms": first_calls,
        "first_call_total_ms": sum(first_calls),
        "warm_step_ms": sum(step_samples) / len(step_samples),
        "warm_step_ms_min": min(step_samples),
        "warm_step_ms_max": max(step_samples),
        "func_expansion_total_us": expansion_total_us,
        "func_expansion_calls": expansion_calls,
        "func_expansion_entries": len(stats),
    }
    print(RESULT_PREFIX + json.dumps(payload, sort_keys=True), flush=True)
    if not correct:
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--arch", choices=["cpu", "cuda", "vulkan"])
    parser.add_argument("--arches", nargs="+", default=["cpu", "cuda", "vulkan"])
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--extra-globals", type=int, default=0)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--pythonpath", default=str(ROOT / "python"))
    parser.add_argument("--out-dir", default=str(ROOT / "benchmarks" / "results" / "d3_func_cache"))
    parser.add_argument("--sample-interval-s", type=float, default=0.5)
    args = parser.parse_args()
    if args.child:
        run_child(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
