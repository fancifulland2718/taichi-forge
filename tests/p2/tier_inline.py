"""P2.c tier-spot inline benchmark.

Runs mat14 cold once per tier, each tier in its own subprocess to avoid
taichi global-state contamination. Writes all results (including tracebacks)
to an output file so we do not depend on terminal stdout buffering.
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "tier_inline.out"
CHILD_FLAG = "--child"


def child(tier: str, arch: str) -> None:
    """Runs in a fresh subprocess. Prints a single __R__{json} line."""
    try:
        # wipe taichi offline cache to force cold compile
        ticache = pathlib.Path(os.path.expanduser("~")) / ".cache" / "taichi"
        shutil.rmtree(ticache, ignore_errors=True)

        import taichi as ti
        arch_map = {"cpu": ti.cpu, "cuda": ti.cuda, "vulkan": ti.vulkan}
        ti.init(arch=arch_map[arch], offline_cache=False, compile_tier=tier,
                kernel_profiler=False)

        sys.path.insert(0, str(HERE))
        from heavy_kernels import make_mat14  # type: ignore

        k = make_mat14(ti)
        t0 = time.perf_counter()
        k()
        ti.sync()
        dt = time.perf_counter() - t0
        print("__R__" + json.dumps({"tier": tier, "arch": arch, "sec": dt}),
              flush=True)
    except Exception:
        print("__E__" + json.dumps({"tier": tier, "arch": arch,
                                    "trace": traceback.format_exc()}),
              flush=True)


def parent(arch: str) -> None:
    results: list[dict] = []
    with OUT.open("w", encoding="utf-8") as f:
        f.write(f"[tier_inline] arch={arch}\n")
        f.flush()
        for tier in ("fast", "balanced", "full"):
            f.write(f"--- launching tier={tier} ---\n")
            f.flush()
            cp = subprocess.run(
                [sys.executable, "-u", __file__, CHILD_FLAG, tier, arch],
                capture_output=True, text=True, timeout=600,
            )
            f.write(f"[tier={tier}] rc={cp.returncode}\n")
            f.write("STDOUT:\n" + (cp.stdout or "") + "\n")
            f.write("STDERR:\n" + (cp.stderr or "") + "\n")
            f.flush()
            for line in (cp.stdout or "").splitlines():
                if line.startswith("__R__"):
                    results.append(json.loads(line[5:]))
                elif line.startswith("__E__"):
                    results.append(json.loads(line[5:]))
        f.write("=== SUMMARY ===\n")
        for r in results:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == CHILD_FLAG:
        child(sys.argv[2], sys.argv[3])
    else:
        arch = sys.argv[1] if len(sys.argv) >= 2 else "cpu"
        parent(arch)
