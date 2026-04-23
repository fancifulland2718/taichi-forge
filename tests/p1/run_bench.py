"""
P1 comprehensive cold-start compile benchmark.

Usage
-----
  python tests/p1/run_bench.py                    # default: llvm_opt=[3,2,1,0]; spv_opt=[3,2,1]
  python tests/p1/run_bench.py --llvm-opt 3 2     # only levels 3 and 2
  python tests/p1/run_bench.py --runs 5           # 5 cold runs per cell
  python tests/p1/run_bench.py --csv results.csv  # dump raw CSV
  python tests/p1/run_bench.py --baseline-commit 812be1f0c  # label for the RESULTS.md entry

Each (kernel × config) cell is measured in a fresh subprocess so LLVM JIT state
does not accumulate across measurements.  The cache dir is deleted before each
subprocess invocation to enforce a true cold start.

Output
------
  - console: summary table
  - tests/p1/RESULTS.md: append run record (commit, date, machine, full table)
  - optional --csv: raw per-run CSV
"""

from __future__ import annotations
import argparse
import csv
import datetime
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys

_HERE  = os.path.dirname(os.path.abspath(__file__))
_ROOT  = os.path.join(_HERE, "..", "..")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_cache_dir() -> str:
    """Return the taichi offline-cache directory without importing taichi."""
    home = os.path.expanduser("~")
    # taichi uses ~/.cache/taichi/ticache on Linux/Mac and
    # %LOCALAPPDATA%\taichi\ticache on Windows if not overridden
    local = os.environ.get("LOCALAPPDATA", "")
    if sys.platform == "win32" and local:
        return os.path.join(local, "taichi", "ticache")
    return os.path.join(home, ".cache", "taichi", "ticache")


def _clear_cache() -> None:
    cache_dir = _get_cache_dir()
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)


def _git_head() -> str:
    try:
        r = subprocess.run(
            ["git", "-C", _ROOT, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True
        )
        return r.stdout.strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Single-cell runner (subprocess)
# ---------------------------------------------------------------------------

_RUNNER_SRC = os.path.join(_HERE, "_hard_runner.py")


def _run_cell(kernel_name: str, llvm_opt: int, spv_opt: int) -> float | None:
    """Fork a subprocess, time one cold compile+run, return ms or None on failure."""
    _clear_cache()
    env = os.environ.copy()
    env["_HK_NAME"]     = kernel_name
    env["_HK_LLVM_OPT"] = str(llvm_opt)
    env["_HK_SPV_OPT"]  = str(spv_opt)
    # Do NOT override PYTHONPATH; use the ambient conda/venv environment.

    result = subprocess.run(
        [sys.executable, _RUNNER_SRC],
        capture_output=True, text=True, env=env,
        timeout=300
    )
    if result.returncode != 0:
        print(f"  [FAIL] {kernel_name} llvm={llvm_opt} spv={spv_opt}:\n{result.stderr[-400:]}")
        return None
    # Taichi may print kernel IR or banners to stdout; the ms value is always the
    # last bare floating-point number on its own line.
    for line in reversed(result.stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return float(line)
        except ValueError:
            continue
    print(f"  [BAD OUTPUT] {kernel_name}: {result.stdout[-200:]!r}")
    return None


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark(
    llvm_levels: list[int],
    spv_levels: list[int],
    runs: int,
    csv_path: str | None,
) -> dict:
    from hard_kernels import HARD_KERNELS  # type: ignore

    sys.path.insert(0, _HERE)

    raw: list[dict] = []          # one row per (kernel, llvm, spv, run)
    summary: dict   = {}          # (kernel, llvm, spv) -> mean_ms

    configs = [(ll, sv) for ll in llvm_levels for sv in spv_levels]
    total_cells = len(HARD_KERNELS) * len(configs) * runs
    done = 0

    print(f"Running {total_cells} measurements  "
          f"(kernels={len(HARD_KERNELS)}, configs={len(configs)}, runs={runs})\n")

    for kname, _factory in HARD_KERNELS:
        for llvm_opt, spv_opt in configs:
            label = f"llvm={llvm_opt} spv={spv_opt}"
            samples: list[float] = []
            for r in range(runs):
                ms = _run_cell(kname, llvm_opt, spv_opt)
                done += 1
                if ms is not None:
                    samples.append(ms)
                    raw.append({
                        "kernel": kname, "llvm_opt": llvm_opt, "spv_opt": spv_opt,
                        "run": r, "compile_ms": f"{ms:.1f}"
                    })
                    print(f"  [{done:3d}/{total_cells}] {kname:26s} {label}  run={r}  {ms:7.1f} ms")
                else:
                    done += 0   # already incremented above
            if samples:
                summary[(kname, llvm_opt, spv_opt)] = statistics.mean(samples)

    # Write CSV
    if csv_path:
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["kernel","llvm_opt","spv_opt","run","compile_ms"])
            w.writeheader(); w.writerows(raw)
        print(f"\nRaw CSV → {csv_path}")

    return summary


# ---------------------------------------------------------------------------
# Pretty table + RESULTS.md append
# ---------------------------------------------------------------------------

def _render_table(
    summary: dict,
    llvm_levels: list[int],
    spv_levels: list[int],
    kernels: list[str],
) -> str:
    lines = []
    # header
    cols = [f"llvm={ll}/spv={sv}" for ll in llvm_levels for sv in spv_levels]
    header = "| kernel | " + " | ".join(cols) + " |"
    sep    = "|" + "|".join(["---"] * (len(cols) + 1)) + "|"
    lines += [header, sep]

    for kname in kernels:
        cells = []
        baseline_ms = summary.get((kname, max(llvm_levels), max(spv_levels)))
        for ll in llvm_levels:
            for sv in spv_levels:
                ms = summary.get((kname, ll, sv))
                if ms is None:
                    cells.append("—")
                else:
                    if baseline_ms and (ll, sv) != (max(llvm_levels), max(spv_levels)):
                        ratio = ms / baseline_ms
                        marker = "↓" if ratio < 0.95 else ("↑" if ratio > 1.05 else "~")
                        cells.append(f"{ms:.0f} ms {marker}{ratio:.2f}x")
                    else:
                        cells.append(f"{ms:.0f} ms")
        lines.append("| " + kname + " | " + " | ".join(cells) + " |")

    return "\n".join(lines)


def _append_results_md(summary: dict, llvm_levels: list[int], spv_levels: list[int]) -> None:
    from hard_kernels import HARD_KERNELS  # type: ignore
    kernels = [k for k, _ in HARD_KERNELS]

    commit  = _git_head()
    now     = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    cpu     = platform.processor()
    py_ver  = sys.version.split()[0]

    table = _render_table(summary, llvm_levels, spv_levels, kernels)

    note = ("Baseline column = max(llvm_opt)/max(spv_opt).  "
            "Ratio shown relative to that baseline.  "
            "↓ = faster compile, ↑ = slower compile (regression).")

    entry = f"""
---

## Run on {now}

| Field | Value |
|---|---|
| commit | `{commit}` |
| machine | {cpu} |
| python | {py_ver} |
| platform | {sys.platform} |
| llvm_levels | {llvm_levels} |
| spv_levels  | {spv_levels}  |

{table}

> {note}
"""

    results_path = os.path.join(_HERE, "RESULTS.md")
    if not os.path.isfile(results_path):
        with open(results_path, "w") as fh:
            fh.write("# P1 Compile Benchmark Results\n\n")
            fh.write("Auto-generated by `tests/p1/run_bench.py`.  "
                     "Each entry appended on a new run.\n")
    with open(results_path, "a") as fh:
        fh.write(entry)
    print(f"\nResults appended → {results_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--llvm-opt", nargs="+", type=int, default=[3, 2, 1, 0],
                   metavar="N", help="llvm_opt_level values to test (default: 3 2 1 0)")
    p.add_argument("--spv-opt",  nargs="+", type=int, default=[3],
                   metavar="N", help="external_optimization_level / spv_opt values (default: 3)")
    p.add_argument("--runs", type=int, default=3, help="Cold runs per cell (default: 3)")
    p.add_argument("--csv",  default=None, help="Write raw CSV to this path")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    sys.path.insert(0, _HERE)

    summary = run_benchmark(
        llvm_levels=args.llvm_opt,
        spv_levels=args.spv_opt,
        runs=args.runs,
        csv_path=args.csv,
    )

    print("\n\n=== SUMMARY TABLE ===\n")
    from hard_kernels import HARD_KERNELS  # type: ignore
    kernels = [k for k, _ in HARD_KERNELS]
    table = _render_table(summary, args.llvm_opt, args.spv_opt, kernels)
    print(table)

    _append_results_md(summary, args.llvm_opt, args.spv_opt)
