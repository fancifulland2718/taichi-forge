"""Compile-time comparison: Taichi 1.7.4 (baseline) vs local 1.8.0 build (candidate).

Usage
-----
  python tests/p2/run_vs_baseline.py
  python tests/p2/run_vs_baseline.py --arches cpu cuda
  python tests/p2/run_vs_baseline.py --kernels mat14 mat16 mpm_p2g
  python tests/p2/run_vs_baseline.py --runs 1 --timeout 300
  python tests/p2/run_vs_baseline.py --baseline-python /path/to/ti174/python.exe

Design
------
  - Each (kernel × arch × version) cell runs inside a fresh subprocess.
  - Cache dir is cleared before every subprocess → guaranteed cold compile.
  - Timeout per cell: 600 s by default (kernels may take 1–2 minutes).
  - Results are appended to tests/p2/RESULTS.md and optionally a CSV file.
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
_RUNNER = os.path.join(_HERE, "_runner.py")
_RESULTS_MD = os.path.join(_HERE, "RESULTS.md")

# Default Python executables.
_PY_BASELINE  = (r"C:\Users\Administrator\AppData\Local\ti-build-cache"
                 r"\miniforge\envs\ti174\python.exe")
_PY_CANDIDATE = (r"C:\Users\Administrator\AppData\Local\ti-build-cache"
                 r"\miniforge\envs\3.10\python.exe")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_cache_dir() -> str:
    local = os.environ.get("LOCALAPPDATA", "")
    if sys.platform == "win32" and local:
        return os.path.join(local, "taichi", "ticache")
    return os.path.join(os.path.expanduser("~"), ".cache", "taichi", "ticache")


def _clear_cache() -> None:
    d = _get_cache_dir()
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)


def _git_head(root: str) -> str:
    try:
        r = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True)
        return r.stdout.strip()
    except Exception:
        return "unknown"


def _taichi_version(python_exe: str) -> str:
    try:
        r = subprocess.run(
            [python_exe, "-c", "import taichi; print(taichi.__version__)"],
            capture_output=True, text=True, timeout=15
        )
        # Last line of output (strips banners)
        for line in reversed(r.stdout.splitlines()):
            line = line.strip()
            if line:
                return line
    except Exception:
        pass
    return "unknown"


def _parse_ms(stdout: str) -> float | None:
    """Extract the last parseable float from stdout."""
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return float(line)
        except ValueError:
            continue
    return None


def _run_cell(python_exe: str, kernel_name: str, arch: str,
              timeout: int = 600) -> float | None:
    """Cold-compile one kernel with one Python executable; return ms or None."""
    _clear_cache()
    env = os.environ.copy()
    env["_HK_NAME"] = kernel_name
    env["_HK_ARCH"] = arch
    env["TI_PRINT_IR"] = "0"

    try:
        result = subprocess.run(
            [python_exe, _RUNNER],
            capture_output=True, text=True,
            env=env, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT ({timeout}s): {kernel_name}/{arch}", flush=True)
        return None
    except Exception as exc:
        print(f"  ERROR: {kernel_name}/{arch}: {exc}", flush=True)
        return None

    if result.returncode not in (0,):
        short_err = result.stderr[-600:] if result.stderr else ""
        # Exit code 3 = arch fallback — skip silently for non-cpu
        if result.returncode == 3:
            print(f"  SKIP ({kernel_name}/{arch}): arch not supported",
                  flush=True)
            return None
        print(f"  FAIL ({kernel_name}/{arch} rc={result.returncode}): "
              f"{short_err}", flush=True)
        return None

    ms = _parse_ms(result.stdout)
    if ms is None:
        print(f"  PARSE_FAIL ({kernel_name}/{arch}): stdout={result.stdout[-200:]}")
    return ms


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    kernels: list[str],
    arches: list[str],
    baseline_py: str,
    candidate_py: str,
    runs: int,
    timeout: int,
    csv_path: str | None,
) -> dict:
    from heavy_kernels import HEAVY_KERNELS
    kernel_meta = {n: (est_cpu, est_cuda) for n, _, est_cpu, est_cuda in HEAVY_KERNELS}

    total_cells = len(kernels) * len(arches) * 2 * runs
    done = 0

    # summary[kernel][arch] = {"baseline": [ms...], "candidate": [ms...]}
    summary: dict[str, dict[str, dict]] = {}

    csv_rows: list[dict] = []

    for arch in arches:
        for kname in kernels:
            summary.setdefault(kname, {})[arch] = {"baseline": [], "candidate": []}
            for run_idx in range(runs):
                for label, pyexe in [("baseline", baseline_py),
                                      ("candidate", candidate_py)]:
                    done += 1
                    pct = 100 * done // total_cells
                    print(f"[{done:3d}/{total_cells}] {kname:<16} {arch:<6} "
                          f"{label:<10} run={run_idx} ...", end=" ", flush=True)
                    ms = _run_cell(pyexe, kname, arch, timeout=timeout)
                    if ms is not None:
                        print(f"{ms:8.0f} ms", flush=True)
                        summary[kname][arch][label].append(ms)
                    else:
                        print("(skip)", flush=True)
                    if csv_path:
                        csv_rows.append({
                            "kernel": kname, "arch": arch,
                            "version": label, "run": run_idx,
                            "ms": ms if ms is not None else "",
                        })

    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["kernel", "arch", "version",
                                               "run", "ms"])
            w.writeheader()
            w.writerows(csv_rows)

    return summary


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _fmt(vals: list[float]) -> str:
    if not vals:
        return "n/a"
    mean = statistics.mean(vals)
    if mean >= 10000:
        return f"{mean/1000:.1f} s"
    return f"{mean:.0f} ms"


def _speedup(baseline: list[float], candidate: list[float]) -> str:
    if not baseline or not candidate:
        return "n/a"
    b = statistics.mean(baseline)
    c = statistics.mean(candidate)
    if c == 0:
        return "∞"
    ratio = b / c
    flag = ""
    if ratio >= 1.10:
        flag = "↑"  # candidate faster
    elif ratio <= 0.91:
        flag = "↓"  # candidate slower
    return f"{ratio:.2f}x{flag}"


def write_results_md(
    summary: dict,
    kernels: list[str],
    arches: list[str],
    baseline_py: str,
    candidate_py: str,
    baseline_ver: str,
    candidate_ver: str,
) -> None:
    ts    = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    head  = _git_head(_ROOT)
    mach  = platform.processor() or platform.machine()

    lines: list[str] = []
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"## Baseline vs Candidate – {ts}")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append(f"| baseline | taichi **{baseline_ver}** (ti174 env) |")
    lines.append(f"| candidate | taichi **{candidate_ver}** (our build, commit `{head}`) |")
    lines.append(f"| machine | {mach} |")
    lines.append(f"| python | {sys.version.split()[0]} |")
    lines.append(f"| platform | {sys.platform} |")
    lines.append(f"| arches | {arches} |")
    lines.append("")

    for arch in arches:
        lines.append(f"### arch = {arch}")
        lines.append("")
        lines.append("| kernel | baseline | candidate | speedup |")
        lines.append("|---|---|---|---|")
        for kname in kernels:
            cell = summary.get(kname, {}).get(arch, {})
            bl = cell.get("baseline", [])
            cd = cell.get("candidate", [])
            lines.append(f"| {kname} | {_fmt(bl)} | {_fmt(cd)} | {_speedup(bl, cd)} |")
        lines.append("")
        lines.append("> speedup = baseline / candidate. ↑ = candidate faster, ↓ = candidate slower.")
        lines.append("")

    block = "\n".join(lines)
    print(block)

    with open(_RESULTS_MD, "a", encoding="utf-8") as f:
        f.write(block + "\n")
    print(f"\nResults appended → {_RESULTS_MD}", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    from heavy_kernels import HEAVY_KERNELS
    all_kernel_names = [n for n, *_ in HEAVY_KERNELS]

    p = argparse.ArgumentParser(
        description="Compare Taichi 1.7.4 baseline vs local build on heavy kernels")
    p.add_argument("--kernels", nargs="+", default=all_kernel_names,
                   help="subset of kernel names to run")
    p.add_argument("--arches", nargs="+", default=["cpu", "cuda"],
                   help="list of arches to test (cpu cuda)")
    p.add_argument("--runs", type=int, default=1,
                   help="cold-compile runs per cell (1 is usually enough for heavy kernels)")
    p.add_argument("--timeout", type=int, default=600,
                   help="per-cell timeout in seconds (default: 600)")
    p.add_argument("--baseline-python", default=_PY_BASELINE,
                   help="Python executable for baseline Taichi")
    p.add_argument("--candidate-python", default=_PY_CANDIDATE,
                   help="Python executable for candidate Taichi")
    p.add_argument("--csv", default=None,
                   help="optional raw CSV output path")
    args = p.parse_args()

    print("=== Heavy Kernel Compile-Time Comparison ===")
    print(f"Baseline  python: {args.baseline_python}")
    print(f"Candidate python: {args.candidate_python}")

    baseline_ver  = _taichi_version(args.baseline_python)
    candidate_ver = _taichi_version(args.candidate_python)
    print(f"Baseline  version: {baseline_ver}")
    print(f"Candidate version: {candidate_ver}")
    print(f"Kernels: {args.kernels}")
    print(f"Arches:  {args.arches}")
    print(f"Runs:    {args.runs}")
    print(f"Timeout: {args.timeout} s")
    print()

    summary = run_benchmark(
        kernels=args.kernels,
        arches=args.arches,
        baseline_py=args.baseline_python,
        candidate_py=args.candidate_python,
        runs=args.runs,
        timeout=args.timeout,
        csv_path=args.csv,
    )

    write_results_md(
        summary=summary,
        kernels=args.kernels,
        arches=args.arches,
        baseline_py=args.baseline_python,
        candidate_py=args.candidate_python,
        baseline_ver=baseline_ver,
        candidate_ver=candidate_ver,
    )


if __name__ == "__main__":
    main()
