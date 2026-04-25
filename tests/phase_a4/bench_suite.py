"""Phase A.4 comprehensive comparison benchmark.

Runs a fixed set of Taichi kernels, measures compile time (first-call)
and runtime (steady-state), records numeric checksums so two installs
can be diff-compared.

Output: JSON on stdout. Keep the script stdlib-only on the Python side
so it works with any reasonable Taichi version (>=1.3).

Note: deliberately NOT using `from __future__ import annotations` -
Taichi needs annotations to resolve to real objects at kernel-build
time, and PEP 563 stringified annotations break that.
"""

import argparse
import gc
import json
import math
import platform
import sys
import time
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import taichi_forge as ti

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> float:
    return time.perf_counter()


def _sync() -> None:
    # Works for both CPU (no-op) and GPU archs.
    ti.sync()


def _checksum(arr: np.ndarray) -> dict[str, float]:
    """Stable summary of a numeric array (f64 accumulation)."""
    a = arr.astype(np.float64, copy=False).ravel()
    return {
        "sum": float(a.sum()),
        "sum_abs": float(np.abs(a).sum()),
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
        # L2-norm is the most sensitive to small mis-computations.
        "l2": float(math.sqrt((a * a).sum())),
    }


def _time_compile(fn: Callable[[], Any]) -> float:
    """Measure the time of a single call that triggers compilation."""
    _sync()
    t0 = _now()
    fn()
    _sync()
    return _now() - t0


def _time_steady(fn: Callable[[], Any], iters: int) -> float:
    """Average per-iter wall time of `iters` consecutive calls."""
    _sync()
    t0 = _now()
    for _ in range(iters):
        fn()
    _sync()
    return (_now() - t0) / iters


# ---------------------------------------------------------------------------
# Individual benchmarks
# ---------------------------------------------------------------------------

def bench_saxpy(n: int, iters: int) -> dict[str, Any]:
    x = ti.field(ti.f32, shape=n)
    y = ti.field(ti.f32, shape=n)

    @ti.kernel
    def init():
        for i in x:
            x[i] = ti.cast(i, ti.f32) * 1e-3
            y[i] = ti.cast(i, ti.f32) * 2e-3

    @ti.kernel
    def saxpy(a: ti.f32):
        for i in x:
            y[i] = a * x[i] + y[i]

    init()
    _sync()
    compile_ms = _time_compile(lambda: saxpy(1.5)) * 1e3
    run_ms = _time_steady(lambda: saxpy(1.5), iters) * 1e3
    ck = _checksum(y.to_numpy())
    return {"compile_ms": compile_ms, "run_ms": run_ms, "checksum": ck}


def bench_reduce(n: int, iters: int) -> dict[str, Any]:
    x = ti.field(ti.f32, shape=n)

    @ti.kernel
    def init():
        for i in x:
            x[i] = ti.sin(ti.cast(i, ti.f32) * 1e-4)

    @ti.kernel
    def sum_sq() -> ti.f32:
        s = 0.0
        for i in x:
            s += x[i] * x[i]
        return s

    init()
    _sync()
    compile_ms = _time_compile(sum_sq) * 1e3
    run_ms = _time_steady(sum_sq, iters) * 1e3
    return {
        "compile_ms": compile_ms,
        "run_ms": run_ms,
        "checksum": {"value": float(sum_sq())},
    }


def bench_matmul(m: int, iters: int) -> dict[str, Any]:
    A = ti.field(ti.f32, shape=(m, m))
    B = ti.field(ti.f32, shape=(m, m))
    C = ti.field(ti.f32, shape=(m, m))

    @ti.kernel
    def init():
        for i, j in A:
            A[i, j] = ti.cast((i * 7 + j) % 17, ti.f32) * 0.01
            B[i, j] = ti.cast((i * 3 + j * 5) % 13, ti.f32) * 0.01

    @ti.kernel
    def gemm():
        for i, j in C:
            s = 0.0
            for k in range(m):
                s += A[i, k] * B[k, j]
            C[i, j] = s

    init()
    _sync()
    compile_ms = _time_compile(gemm) * 1e3
    run_ms = _time_steady(gemm, iters) * 1e3
    ck = _checksum(C.to_numpy())
    return {"compile_ms": compile_ms, "run_ms": run_ms, "checksum": ck}


def bench_stencil(n: int, iters: int) -> dict[str, Any]:
    u = ti.field(ti.f32, shape=(n, n))
    v = ti.field(ti.f32, shape=(n, n))

    @ti.kernel
    def init():
        for i, j in u:
            x = (ti.cast(i, ti.f32) / n - 0.5) * 2
            y = (ti.cast(j, ti.f32) / n - 0.5) * 2
            u[i, j] = ti.exp(-(x * x + y * y) * 5.0)
            v[i, j] = 0.0

    @ti.kernel
    def step():
        for i, j in ti.ndrange((1, n - 1), (1, n - 1)):
            v[i, j] = 0.2 * (u[i, j]
                              + u[i - 1, j] + u[i + 1, j]
                              + u[i, j - 1] + u[i, j + 1])

    @ti.kernel
    def swap():
        for i, j in u:
            u[i, j] = v[i, j]

    init()
    _sync()
    compile_ms = _time_compile(step) * 1e3
    # warm up swap too
    swap(); _sync()

    def one_iter():
        step()
        swap()

    run_ms = _time_steady(one_iter, iters) * 1e3
    ck = _checksum(u.to_numpy())
    return {"compile_ms": compile_ms, "run_ms": run_ms, "checksum": ck}


def bench_nbody(n: int, iters: int) -> dict[str, Any]:
    pos = ti.Vector.field(3, ti.f32, shape=n)
    vel = ti.Vector.field(3, ti.f32, shape=n)

    @ti.kernel
    def init():
        for i in pos:
            # deterministic pseudo-random init
            fi = ti.cast(i, ti.f32)
            pos[i] = ti.Vector([ti.sin(fi * 0.1),
                                ti.cos(fi * 0.13),
                                ti.sin(fi * 0.07)])
            vel[i] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def step(dt: ti.f32):
        for i in pos:
            acc = ti.Vector([0.0, 0.0, 0.0])
            for j in range(n):
                if i != j:
                    d = pos[j] - pos[i]
                    r2 = d.dot(d) + 1e-3
                    acc += d / (r2 * ti.sqrt(r2))
            vel[i] += dt * acc
            pos[i] += dt * vel[i]

    init()
    _sync()
    compile_ms = _time_compile(lambda: step(1e-4)) * 1e3
    run_ms = _time_steady(lambda: step(1e-4), iters) * 1e3
    ck = _checksum(pos.to_numpy())
    return {"compile_ms": compile_ms, "run_ms": run_ms, "checksum": ck}


def bench_mandelbrot(n: int, iters: int) -> dict[str, Any]:
    out = ti.field(ti.f32, shape=(n, n))

    @ti.kernel
    def render(cx: ti.f32, cy: ti.f32, scale: ti.f32):
        for i, j in out:
            x0 = (ti.cast(i, ti.f32) / n - 0.5) * scale + cx
            y0 = (ti.cast(j, ti.f32) / n - 0.5) * scale + cy
            x = 0.0
            y = 0.0
            k = 0
            max_iter = 128
            while x * x + y * y <= 4.0 and k < max_iter:
                x, y = x * x - y * y + x0, 2.0 * x * y + y0
                k += 1
            out[i, j] = ti.cast(k, ti.f32) / max_iter

    compile_ms = _time_compile(lambda: render(-0.75, 0.0, 2.5)) * 1e3
    run_ms = _time_steady(lambda: render(-0.75, 0.0, 2.5), iters) * 1e3
    ck = _checksum(out.to_numpy())
    return {"compile_ms": compile_ms, "run_ms": run_ms, "checksum": ck}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

BENCHES: dict[str, tuple[Callable[..., dict[str, Any]], dict[str, Any]]] = {
    "saxpy":      (bench_saxpy,      {"n": 1 << 20, "iters": 50}),
    "reduce":     (bench_reduce,     {"n": 1 << 20, "iters": 50}),
    "matmul":     (bench_matmul,     {"m": 256,     "iters": 10}),
    "stencil":    (bench_stencil,    {"n": 512,     "iters": 20}),
    "nbody":      (bench_nbody,      {"n": 1024,    "iters": 5}),
    "mandelbrot": (bench_mandelbrot, {"n": 512,     "iters": 10}),
}


def run_arch(arch_name: str, repeats: int) -> dict[str, Any]:
    arch = getattr(ti, arch_name)
    results: dict[str, Any] = {"arch": arch_name, "benches": {}}
    # Repeat each bench `repeats` times (in a fresh runtime) to get stability stats.
    for bname, (fn, kwargs) in BENCHES.items():
        samples = []
        for r in range(repeats):
            ti.init(arch=arch, offline_cache=False, random_seed=42)
            gc.collect()
            try:
                res = fn(**kwargs)
                samples.append(res)
            except Exception as e:
                samples.append({"error": f"{type(e).__name__}: {e}"})
            ti.reset()
        # aggregate
        ok = [s for s in samples if "error" not in s]
        if not ok:
            results["benches"][bname] = {"error": samples[0].get("error")}
            continue
        compile_ms = [s["compile_ms"] for s in ok]
        run_ms = [s["run_ms"] for s in ok]
        checksums = [s["checksum"] for s in ok]
        agg = {
            "compile_ms": {
                "min": min(compile_ms),
                "mean": sum(compile_ms) / len(compile_ms),
                "max": max(compile_ms),
            },
            "run_ms": {
                "min": min(run_ms),
                "mean": sum(run_ms) / len(run_ms),
                "max": max(run_ms),
                "stdev": (
                    float(np.std(run_ms, ddof=0)) if len(run_ms) > 1 else 0.0
                ),
            },
            # keep first checksum (they should be identical for a deterministic kernel)
            "checksum": checksums[0],
            # also surface max relative diff across repeats as a stability signal
            "checksum_stability": _checksum_stability(checksums),
            "repeats": len(ok),
        }
        results["benches"][bname] = agg
    return results


def _checksum_stability(checksums: list[dict[str, float]]) -> float:
    if len(checksums) < 2:
        return 0.0
    keys = list(checksums[0].keys())
    worst = 0.0
    for k in keys:
        vals = [c[k] for c in checksums]
        base = max(abs(v) for v in vals) or 1.0
        spread = max(vals) - min(vals)
        worst = max(worst, spread / base)
    return worst


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--archs", default="cpu,cuda",
                   help="comma-separated taichi archs to test")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--label", default="unknown")
    p.add_argument("--output", default="-")
    args = p.parse_args()

    info = {
        "label": args.label,
        "taichi_version": ti.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "repeats": args.repeats,
        "archs": {},
    }

    for a in args.archs.split(","):
        a = a.strip()
        if not a:
            continue
        try:
            info["archs"][a] = run_arch(a, args.repeats)
        except Exception as e:
            info["archs"][a] = {"error": f"{type(e).__name__}: {e}"}

    payload = json.dumps(info, indent=2, default=str)
    if args.output == "-":
        print(payload)
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(payload)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
