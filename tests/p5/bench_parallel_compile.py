"""P5.b — A/B bench for parallel batch compile across CPU/CUDA/Vulkan.

Uses subprocess isolation so each cold-compile run sees a fresh interpreter
(avoids stale `ti.f32` references after `ti.reset()`).

Usage:
    python tests/p5/bench_parallel_compile.py [--arch cpu|cuda|vulkan|all]
                                               [--n 16] [--threads 4] [--repeat 2]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


WORKER_SCRIPT = r"""
import json, os, shutil, sys, time
import taichi as ti

mode   = sys.argv[1]
arch   = sys.argv[2]
n      = int(sys.argv[3])
thread = int(sys.argv[4])
cache_dirs = sys.argv[5].split('|') if len(sys.argv) > 5 else []

for d in cache_dirs:
    if d and os.path.isdir(d):
        try: shutil.rmtree(d)
        except Exception: pass

arch_obj = {'cpu': ti.cpu, 'cuda': ti.cuda, 'vulkan': ti.vulkan}[arch]
ti.init(arch=arch_obj, offline_cache=False,
        num_compile_threads=thread, log_level=ti.WARN)

kernels = []
for idx in range(n):
    a = float(idx + 1)
    b = float((idx * 7) % 13 + 1)

    @ti.kernel
    def _k() -> ti.f32:
        s = 0.0
        x = 0.5
        # Deliberately unrolled big loop → heavier LLVM optimization.
        for i in range(256):
            s += ti.sin(x * a + i * 0.1) * ti.cos(x * b - i * 0.07) \
                 + ti.log(1.0 + ti.abs(x + i * 0.01)) * a \
                 - ti.exp(-ti.abs(x + i * 0.02) * 0.01) * b
        return s

    _k.__name__ = f"_k_{idx}"
    kernels.append(_k)

t0 = time.perf_counter()
if mode == 'parallel':
    # Phase split: how much is Python-side materialize vs. C++ parallel compile?
    from taichi.lang import impl as _impl
    t_mat0 = time.perf_counter()
    specs = []
    for k in kernels:
        primal = k._primal if hasattr(k, '_primal') and not hasattr(k, 'ensure_compiled') else k
        key = primal.ensure_compiled()
        specs.append(primal.compiled_kernels[key])
    t_mat = time.perf_counter() - t_mat0
    prog = _impl.get_runtime().prog
    t_cxx0 = time.perf_counter()
    prog.compile_kernels(prog.config(), specs)
    t_cxx = time.perf_counter() - t_cxx0
    extra = {'t_py_materialize': t_mat, 't_cxx_compile': t_cxx}
else:
    for k in kernels:
        k()
    extra = {}
elapsed = time.perf_counter() - t0
result = {'elapsed': elapsed}
result.update(extra)
print('__BENCH_RESULT__' + json.dumps(result))
"""


def _cache_dirs():
    return "|".join(filter(None, [
        os.path.expanduser("~/.cache/taichi"),
        os.path.join(os.path.expanduser("~"), "AppData", "Local", "taichi", "taichi"),
    ]))


def _run_worker(mode: str, arch: str, n: int, threads: int) -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    worker_path = os.path.join(here, "_bench_worker.py")
    with open(worker_path, "w", encoding="utf-8") as f:
        f.write(WORKER_SCRIPT)

    proc = subprocess.run(
        [sys.executable, worker_path, mode, arch, str(n), str(threads), _cache_dirs()],
        capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"worker failed rc={proc.returncode}: "
            f"stderr={proc.stderr[-300:]}"
        )
    for line in proc.stdout.splitlines():
        if line.startswith("__BENCH_RESULT__"):
            return json.loads(line[len("__BENCH_RESULT__"):])
    raise RuntimeError(f"no __BENCH_RESULT__ in stdout: {proc.stdout[-300:]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="all", choices=["cpu", "cuda", "vulkan", "all"])
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--repeat", type=int, default=2)
    args = ap.parse_args()

    archs = ["cpu", "cuda", "vulkan"] if args.arch == "all" else [args.arch]
    print(f"[p5b-bench] n={args.n} threads={args.threads} repeat={args.repeat}")
    print("-" * 72)
    for arch in archs:
        serial, parallel = [], []
        s_extras, p_extras = [], []
        for _ in range(args.repeat):
            try:
                r = _run_worker("serial", arch, args.n, args.threads)
                serial.append(r["elapsed"])
                s_extras.append(r)
            except Exception as e:
                print(f"  {arch:8s} skipped (serial): {str(e)[:160]}")
                break
            try:
                r = _run_worker("parallel", arch, args.n, args.threads)
                parallel.append(r["elapsed"])
                p_extras.append(r)
            except Exception as e:
                print(f"  {arch:8s} skipped (parallel): {str(e)[:160]}")
                break
        if not serial or not parallel:
            continue
        s = min(serial) * 1000
        p = min(parallel) * 1000
        speedup = s / p if p > 0 else float("inf")
        s_all = [f"{x*1000:.0f}" for x in serial]
        p_all = [f"{x*1000:.0f}" for x in parallel]
        best_p = min(p_extras, key=lambda r: r["elapsed"])
        py_ms = best_p.get("t_py_materialize", 0) * 1000
        cxx_ms = best_p.get("t_cxx_compile", 0) * 1000
        print(
            f"  {arch:8s} serial={s:7.1f}ms  parallel={p:7.1f}ms  "
            f"speedup={speedup:.2f}x  (parallel split: py={py_ms:.1f}ms "
            f"cxx={cxx_ms:.1f}ms)"
        )
        print(f"           serial_all={s_all}  parallel_all={p_all}")


if __name__ == "__main__":
    sys.exit(main() or 0)
