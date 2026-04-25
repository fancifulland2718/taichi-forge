"""P5.b — heavier A/B bench across CPU/CUDA/Vulkan with a construction-test
workload designed to maximize per-kernel compile time so parallelism is
observable.

Kernel shape:
  * One heavy parallel ti.kernel per item (outer offload).
  * A 3-layer nested ti.static loop (8x8x4 = 256 fully unrolled ops) forcing
    LLVM/SPIR-V optimizer to chew on a real-sized function body.
  * Distinct constants per kernel → no cross-kernel redundancy caching.

Invocation:
    py -3.10 tests/p5/bench_parallel_compile_heavy.py --arch all --n 24 --threads 8
"""

from __future__ import annotations

import argparse, json, os, subprocess, sys


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

SZ = 64
fields = []
kernels = []
for idx in range(n):
    a = float(idx + 1)
    b = float((idx * 7) % 13 + 1)
    c = float((idx * 11) % 19 + 1)

    f = ti.field(ti.f32, shape=SZ)
    fields.append(f)

    @ti.kernel
    def _k(tf: ti.template()):
        for i in tf:
            x = ti.cast(i, ti.f32) * 0.01 + a
            acc = 0.0
            # 3 static-unrolled layers → single offload with big body
            for p in ti.static(range(8)):
                for q in ti.static(range(8)):
                    for r in ti.static(range(4)):
                        t = x + p * 0.11 + q * 0.07 - r * 0.03
                        acc += ti.sin(t * a) * ti.cos(t * b) \
                              + ti.log(1.0 + ti.abs(t * c)) \
                              - ti.exp(-ti.abs(t) * 0.05)
            tf[i] = acc

    _k.__name__ = f"_k_{idx}"
    kernels.append((_k, (f,)))

t0 = time.perf_counter()
if mode == 'parallel':
    from taichi.lang import impl as _impl
    t_mat0 = time.perf_counter()
    specs = []
    for (k, args) in kernels:
        primal = k._primal if hasattr(k, '_primal') and not hasattr(k, 'ensure_compiled') else k
        # Template kernels need args to drive specialization.
        key = primal.ensure_compiled(*args)
        specs.append(primal.compiled_kernels[key])
    t_mat = time.perf_counter() - t_mat0
    prog = _impl.get_runtime().prog
    t_cxx0 = time.perf_counter()
    prog.compile_kernels(prog.config(), specs)
    t_cxx = time.perf_counter() - t_cxx0
    extra = {'t_py_materialize': t_mat, 't_cxx_compile': t_cxx}
else:
    for (k, args) in kernels:
        k(*args)
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


def _run_worker(py, mode, arch, n, threads):
    here = os.path.dirname(os.path.abspath(__file__))
    worker_path = os.path.join(here, "_bench_worker_heavy.py")
    with open(worker_path, "w", encoding="utf-8") as f:
        f.write(WORKER_SCRIPT)
    proc = subprocess.run(
        [*py, worker_path, mode, arch, str(n), str(threads), _cache_dirs()],
        capture_output=True, text=True, timeout=900,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"worker failed rc={proc.returncode}: stderr={proc.stderr[-400:]}"
        )
    for line in proc.stdout.splitlines():
        if line.startswith("__BENCH_RESULT__"):
            return json.loads(line[len("__BENCH_RESULT__"):])
    raise RuntimeError(f"no __BENCH_RESULT__ in stdout: {proc.stdout[-400:]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="all", choices=["cpu", "cuda", "vulkan", "all"])
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--py", default="py,-3.10")
    args = ap.parse_args()
    py = args.py.split(",")

    archs = ["cpu", "cuda", "vulkan"] if args.arch == "all" else [args.arch]
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f"bench_heavy_result_{args.arch}.txt")
    out_lines = [f"[p5b-heavy] n={args.n} threads={args.threads} repeat={args.repeat} py={py}",
                 "-" * 78]
    print(out_lines[0]); print(out_lines[1])
    for arch in archs:
        serial, parallel, p_extras = [], [], []
        for _ in range(args.repeat):
            try:
                r = _run_worker(py, "serial", arch, args.n, args.threads)
                serial.append(r["elapsed"])
            except Exception as e:
                msg = f"  {arch:8s} skipped (serial): {str(e)[:300]}"
                print(msg); out_lines.append(msg)
                break
            try:
                r = _run_worker(py, "parallel", arch, args.n, args.threads)
                parallel.append(r["elapsed"])
                p_extras.append(r)
            except Exception as e:
                msg = f"  {arch:8s} skipped (parallel): {str(e)[:300]}"
                print(msg); out_lines.append(msg)
                break
        if not serial or not parallel:
            continue
        s = min(serial) * 1000; p = min(parallel) * 1000
        speedup = s / p if p > 0 else float("inf")
        best_p = min(p_extras, key=lambda r: r["elapsed"])
        py_ms = best_p.get("t_py_materialize", 0) * 1000
        cxx_ms = best_p.get("t_cxx_compile", 0) * 1000
        line1 = (f"  {arch:8s} serial={s:8.1f}ms  parallel={p:8.1f}ms  "
                 f"speedup={speedup:.2f}x  (split: py={py_ms:.1f}ms cxx={cxx_ms:.1f}ms)")
        line2 = (f"           serial_all={[round(x*1000) for x in serial]}  "
                 f"parallel_all={[round(x*1000) for x in parallel]}")
        print(line1); print(line2)
        out_lines.append(line1); out_lines.append(line2)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"[wrote] {out_path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
