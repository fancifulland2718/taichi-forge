"""P3.c A/B cold-compile bench — subprocess per datapoint.

Runs two kernels (scalar-only saxpy, matrix-heavy 8x8 matmul) across a sweep
of inner sizes / unroll counts, and also a synthetic "many-kernel" scenario
where scalarize is invoked per-kernel. ``ti.init`` uses DEFAULT config
(P3.a unrolling_hard_limit = 0, i.e. the guard is OFF), so any delta is
attributable to the P3.c scalarize early-exit alone.

Invoke:
    py -3.10 tests\p3\bench_p3c_ab.py [--label WHEEL]
Prints one TSV row per datapoint, easy to diff two runs.
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time

SCALAR = r"""
import time, taichi as ti
ti.init(arch=ti.cpu, offline_cache=False, log_level=ti.ERROR)
N = 1 << 18
a = ti.field(ti.f32, shape=N); b = ti.field(ti.f32, shape=N); c = ti.field(ti.f32, shape=N)
@ti.kernel
def k(alpha: ti.f32):
    for i in a:
        x = a[i]; y = b[i]; z = ti.f32(0.0)
        for u in ti.static(range({UNROLL})):
            z += alpha * x + float(u) * y
        c[i] = z
t0 = time.perf_counter()
k(1.5); ti.sync()
print(f"DT={{time.perf_counter()-t0:.6f}}")
"""

MATRIX = r"""
import time, taichi as ti
ti.init(arch=ti.cpu, offline_cache=False, log_level=ti.ERROR)
N = 1 << 12
A = ti.Matrix.field(8, 8, ti.f32, shape=N)
B = ti.Matrix.field(8, 8, ti.f32, shape=N)
C = ti.Matrix.field(8, 8, ti.f32, shape=N)
@ti.kernel
def k():
    for i in A:
        acc = A[i] @ B[i]
        for u in ti.static(range({UNROLL})):
            acc = acc @ B[i]
        C[i] = acc
t0 = time.perf_counter()
k(); ti.sync()
print(f"DT={{time.perf_counter()-t0:.6f}}")
"""

# Many-kernel scalar scenario — stresses 'scalarize called per kernel'.
MANY = r"""
import time, taichi as ti
ti.init(arch=ti.cpu, offline_cache=False, log_level=ti.ERROR)
N = 1 << 14
fields = [ti.field(ti.f32, shape=N) for _ in range({KCOUNT})]
def make_kernel(idx):
    f = fields[idx]
    @ti.kernel
    def k(alpha: ti.f32):
        for i in f:
            x = f[i]; z = ti.f32(0.0)
            for u in ti.static(range(32)):
                z += alpha * x + float(u)
            f[i] = z
    return k
ks = [make_kernel(i) for i in range({KCOUNT})]
t0 = time.perf_counter()
for k in ks: k(1.5)
ti.sync()
print(f"DT={{time.perf_counter()-t0:.6f}}")
"""

def run_once(code: str) -> float:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as fh:
        fh.write(code); path = fh.name
    try:
        p = subprocess.run([sys.executable, "-W", "ignore", path],
                           capture_output=True, text=True, encoding="utf-8")
        for line in (p.stdout or "").splitlines():
            if line.startswith("DT="): return float(line[3:])
        sys.stderr.write(p.stderr[:600]); return float("nan")
    finally:
        try: os.unlink(path)
        except OSError: pass

def median3(code: str) -> float:
    vs = sorted(run_once(code) for _ in range(3))
    return vs[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="wheel")
    args = ap.parse_args()
    print(f"# label={args.label}")
    print("scenario\tparam\tmedian_s", flush=True)
    # Scalar scenarios — P3.c early-exit path applies.
    for u in (32, 128, 400, 800):
        print(f"scalar\tunroll={u}\t{median3(SCALAR.format(UNROLL=u)):.4f}", flush=True)


if __name__ == "__main__":
    main()
