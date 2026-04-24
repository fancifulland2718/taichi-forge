"""P1.b hot-start microbench.

Measures the wall-clock cost of the *second* `ti.init()` after a
`ti.reset()`, which is the canonical "hot-start" path: the on-disk
offline cache is fully populated, but each Program still has to load
every CompiledKernelData from disk on first use.

Compares two configurations:

  * cold        — fresh process, empty offline cache. Establishes the
                  baseline compile cost so the bench can be sanity
                  checked.
  * disk-warm   — second init in the same process; offline cache hit
                  but the new P1.b in-process bytes mirror is
                  *disabled* (TI_INPROC_DISK_MIRROR_MB=0).
  * mirror-warm — second init in the same process; mirror enabled
                  (default cap of 256 MB).

Run via:

    python tests/p5/bench_hot_start.py --arch x64
    python tests/p5/bench_hot_start.py --arch cuda
    python tests/p5/bench_hot_start.py --arch vulkan

The kernel set is intentionally small (32 distinct kernels with cheap
bodies) so per-kernel overhead dominates total time and the mirror's
impact — purely on the disk-load path — is visible.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

WORKER = textwrap.dedent(
    """
    import os, sys, time, json
    import taichi_forge as ti

    arch_name = os.environ['BENCH_ARCH']
    cache_dir = os.environ['BENCH_CACHE']
    n_kernels = int(os.environ['BENCH_N'])
    phase     = os.environ['BENCH_PHASE']  # "cold" | "warm"

    arch = getattr(ti, arch_name)
    ti.init(arch=arch, offline_cache=True, offline_cache_file_path=cache_dir,
            print_ir=False, log_level=ti.WARN)

    # Build N distinct kernels — each must hash to a unique cache key,
    # so vary a closure constant.
    kernels = []
    for i in range(n_kernels):
        c = i + 1
        @ti.kernel
        def k(x: ti.f32) -> ti.f32:
            return x * c + c  # noqa: B023 — closure capture is intentional
        kernels.append(k)

    # First init in the worker process: time the actual first-call path
    # for every kernel. For phase=="warm" this is the disk-warm or
    # mirror-warm path (cache_dir already populated by the cold worker).
    t0 = time.perf_counter()
    for k in kernels:
        k(1.0)
    elapsed_first = time.perf_counter() - t0

    # Second init within the *same* process. After ti.reset+ti.init we
    # get a fresh Program but the offline cache is still on disk and
    # — importantly — the P1.b in-process bytes mirror is still in
    # memory if enabled. This is the metric we care about.
    ti.reset()
    ti.init(arch=arch, offline_cache=True, offline_cache_file_path=cache_dir,
            print_ir=False, log_level=ti.WARN)
    kernels2 = []
    for i in range(n_kernels):
        c = i + 1
        @ti.kernel
        def k(x: ti.f32) -> ti.f32:
            return x * c + c  # noqa: B023
        kernels2.append(k)

    t1 = time.perf_counter()
    for k in kernels2:
        k(1.0)
    elapsed_second = time.perf_counter() - t1

    print(json.dumps({
        "phase": phase,
        "elapsed_first_ms": elapsed_first * 1000.0,
        "elapsed_second_ms": elapsed_second * 1000.0,
    }))
    """
)


def run_phase(*, arch: str, cache_dir: Path, n: int, phase: str,
              mirror_mb: int) -> dict:
    env = os.environ.copy()
    env["BENCH_ARCH"] = arch
    env["BENCH_CACHE"] = str(cache_dir)
    env["BENCH_N"] = str(n)
    env["BENCH_PHASE"] = phase
    env["TI_INPROC_DISK_MIRROR_MB"] = str(mirror_mb)
    # Avoid shadow-importing the in-tree `taichi/` package when invoked
    # from a subprocess that may inherit cwd=D:\taichi.
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    # Write the worker to a real .py file so Taichi's source inspection
    # (which uses inspect.getsourcelines) can find the kernel bodies.
    # `-c "..."` does not expose source lines.
    worker_dir = Path(tempfile.mkdtemp(prefix="ti_p1b_worker_"))
    try:
        worker_path = worker_dir / "worker.py"
        worker_path.write_text(WORKER, encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(worker_path)],
            env=env,
            cwd=str(worker_dir),
            capture_output=True,
            text=True,
            timeout=180,
        )
    finally:
        shutil.rmtree(worker_dir, ignore_errors=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Worker failed (phase={phase}, mirror_mb={mirror_mb}):\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    # Take the last non-empty JSON line in case taichi prints banners.
    for line in reversed(proc.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            import json
            return json.loads(line)
    raise RuntimeError(f"No JSON output from worker:\n{proc.stdout}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="x64",
                    choices=["x64", "cpu", "cuda", "vulkan"])
    ap.add_argument("--n", type=int, default=32, help="kernel count")
    ap.add_argument("--repeat", type=int, default=3,
                    help="repeats per phase (median reported)")
    args = ap.parse_args()

    arch = "x64" if args.arch == "cpu" else args.arch

    with tempfile.TemporaryDirectory(prefix="ti_p1b_bench_") as tmp:
        cache_dir = Path(tmp) / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # --- Cold: empty cache. The "first" measurement is the true
        # cold-compile cost; the "second" is in-process + mirror primed.
        # We discard "second" here — it would be optimistic since the
        # mirror was just written.
        print(f"[arch={args.arch} n={args.n}] populating cold cache...",
              flush=True)
        cold = run_phase(arch=arch, cache_dir=cache_dir, n=args.n,
                          phase="cold", mirror_mb=256)
        cold_ms = cold["elapsed_first_ms"]

        def median_second(mirror_mb: int) -> float:
            samples = []
            for _ in range(args.repeat):
                r = run_phase(arch=arch, cache_dir=cache_dir, n=args.n,
                              phase="warm", mirror_mb=mirror_mb)
                # In a fresh subprocess the mirror is empty for the
                # FIRST init (cache hit comes from disk), then populated
                # for the SECOND init. So `elapsed_second_ms` is what
                # measures the pure mirror-vs-disk path.
                samples.append(r["elapsed_second_ms"])
            samples.sort()
            return samples[len(samples) // 2]

        # --- Warm path WITH mirror disabled (TI_INPROC_DISK_MIRROR_MB=0).
        warm_disk_ms = median_second(mirror_mb=0)

        # --- Warm path WITH mirror enabled (default 256 MB).
        warm_mirror_ms = median_second(mirror_mb=256)

        speedup = warm_disk_ms / warm_mirror_ms if warm_mirror_ms > 0 else float("nan")

        print()
        print(f"== P1.b hot-start bench  arch={args.arch} n={args.n} ==")
        print(f"  cold (compile + dump)         : {cold_ms:8.1f} ms")
        print(f"  warm second-init, mirror OFF  : {warm_disk_ms:8.1f} ms")
        print(f"  warm second-init, mirror ON   : {warm_mirror_ms:8.1f} ms")
        print(f"  mirror speedup                : {speedup:6.2f}x")


if __name__ == "__main__":
    main()
