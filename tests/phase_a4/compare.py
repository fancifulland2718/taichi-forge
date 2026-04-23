"""Compare two bench_suite.py JSON outputs (baseline vs candidate).

Usage:
    python compare.py baseline.json candidate.json [--rel-tol 1e-4]

Prints a human-readable report and exits non-zero if any checksum
differs by more than `--rel-tol` (relative) for deterministic kernels.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def _rel_diff(a: float, b: float) -> float:
    base = max(abs(a), abs(b))
    if base == 0:
        return 0.0
    return abs(a - b) / base


def _fmt_ms(d: dict[str, Any] | None) -> str:
    if not d:
        return "     n/a"
    return f"{d.get('mean', float('nan')):8.2f}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("baseline")
    p.add_argument("candidate")
    p.add_argument("--rel-tol", type=float, default=1e-4,
                   help="relative tolerance for checksum comparison")
    p.add_argument("--abs-tol", type=float, default=1e-6,
                   help="absolute floor for checksum comparison")
    args = p.parse_args()

    with open(args.baseline, encoding="utf-8") as f:
        base = json.load(f)
    with open(args.candidate, encoding="utf-8") as f:
        cand = json.load(f)

    print(f"Baseline : {base['label']} taichi={base['taichi_version']}")
    print(f"Candidate: {cand['label']} taichi={cand['taichi_version']}")
    print()

    fails: list[str] = []
    archs = sorted(set(base["archs"]) | set(cand["archs"]))

    for arch in archs:
        b_arch = base["archs"].get(arch, {})
        c_arch = cand["archs"].get(arch, {})
        if "error" in b_arch or "error" in c_arch:
            print(f"[{arch}] skipped (error: base={b_arch.get('error')} "
                  f"cand={c_arch.get('error')})")
            continue
        b_benches = b_arch.get("benches", {})
        c_benches = c_arch.get("benches", {})
        names = sorted(set(b_benches) | set(c_benches))
        print(f"=== arch = {arch} ===")
        header = (f"  {'bench':<12} | {'compile (ms)':>22} | "
                  f"{'steady (ms/iter)':>22} | {'result match':>16}")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for name in names:
            b = b_benches.get(name, {})
            c = c_benches.get(name, {})
            if "error" in b or "error" in c:
                print(f"  {name:<12} | ERR base={b.get('error')} "
                      f"cand={c.get('error')}")
                continue
            b_comp = b.get("compile_ms", {})
            c_comp = c.get("compile_ms", {})
            b_run = b.get("run_ms", {})
            c_run = c.get("run_ms", {})

            # speedups: >1 means candidate is FASTER
            comp_speedup = (
                b_comp.get("mean", 0) / c_comp["mean"]
                if c_comp.get("mean") else float("nan")
            )
            run_speedup = (
                b_run.get("mean", 0) / c_run["mean"]
                if c_run.get("mean") else float("nan")
            )

            # checksum diff
            b_ck = b.get("checksum", {})
            c_ck = c.get("checksum", {})
            worst_key = ""
            worst_rel = 0.0
            for k in set(b_ck) & set(c_ck):
                r = _rel_diff(b_ck[k], c_ck[k])
                if r > worst_rel:
                    worst_rel = r
                    worst_key = k
            match = "OK"
            if worst_rel > args.rel_tol and worst_rel > args.abs_tol:
                match = f"FAIL({worst_key}:{worst_rel:.2e})"
                fails.append(f"{arch}/{name}: {match}")
            else:
                match = f"OK({worst_rel:.2e})"

            print(f"  {name:<12} | "
                  f"{_fmt_ms(b_comp)} -> {_fmt_ms(c_comp)} "
                  f"({comp_speedup:4.2f}x) | "
                  f"{_fmt_ms(b_run)} -> {_fmt_ms(c_run)} "
                  f"({run_speedup:4.2f}x) | {match:>16}")
        print()

    if fails:
        print("FAIL - numerical mismatches:")
        for f in fails:
            print(f"  {f}")
        return 1
    print("All numeric checksums match within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
