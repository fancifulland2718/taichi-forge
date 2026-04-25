"""P2.c inline tier spot — single-process, no subprocess, no rmtree.

Uses ti.reset() between tiers. For cold-compile measurement, we rely on
offline_cache=False (forces fresh codegen each ti.init). Simpler and
terminal-friendly — just prints plain text to stdout.
"""
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ARCH = sys.argv[1] if len(sys.argv) >= 2 else "cpu"
OUT = HERE / f"tier_single_{ARCH}.out"

import taichi as ti  # noqa: E402
from heavy_kernels import make_mat14  # noqa: E402

ARCH_MAP = {"cpu": ti.cpu, "cuda": ti.cuda, "vulkan": ti.vulkan}

results = []
with OUT.open("w", encoding="utf-8") as log:
    log.write(f"[tier_single] arch={ARCH}\n")
    log.flush()
    for tier in ("fast", "balanced", "full"):
        log.write(f"--- {tier} ---\n")
        log.flush()
        ti.reset()
        ti.init(arch=ARCH_MAP[ARCH], offline_cache=False, compile_tier=tier,
                kernel_profiler=False)
        k = make_mat14(ti)
        t0 = time.perf_counter()
        k()
        ti.sync()
        dt = time.perf_counter() - t0
        line = f"tier={tier:8s}  sec={dt:.3f}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()
        results.append({"tier": tier, "arch": ARCH, "sec": dt})
    log.write("=== SUMMARY ===\n")
    log.write(json.dumps(results, indent=2) + "\n")

print(json.dumps(results))
