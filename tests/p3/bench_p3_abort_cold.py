"""P3 cold-compile verification — each row runs in a FRESH python process.

The existing bench_p3_abort.py reuses one interpreter (with ti.reset() between
rows). The user asked whether the 707x speed-up was "cold" or "hot" — here we
spawn a clean subprocess per (N, hard_limit) to remove every in-process cache.
"""
import subprocess
import sys
import time
import os
import tempfile

RUNNER = r'''
import sys, time
import taichi as ti
from taichi.lang.exception import TaichiCompilationError

N = {N}
HL = {HL}
ti.init(arch=ti.cpu, offline_cache=False, unrolling_hard_limit=HL, log_level=ti.ERROR)
x = ti.field(ti.f32, shape=4)

@ti.kernel
def run():
    for i in x:
        s = ti.f32(0.0)
        for k in ti.static(range(N)):
            s += float(k) * ti.cast(i, ti.f32)
        x[i] = s

t0 = time.perf_counter()
status = "ok"
try:
    run()
    ti.sync()
except TaichiCompilationError:
    status = "aborted"
dt = time.perf_counter() - t0
print(f"DT={{dt:.4f}} STATUS={{status}}")
'''

def timed_run(N, HL):
    code = RUNNER.format(N=N, HL=HL)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False,
                                      encoding="utf-8") as f:
        f.write(code)
        script = f.name
    try:
        t0 = time.perf_counter()
        p = subprocess.run([sys.executable, "-W", "ignore", script],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        wall = time.perf_counter() - t0
    finally:
        try:
            os.unlink(script)
        except OSError:
            pass
    out = p.stdout.strip().splitlines()
    err = p.stderr.strip()
    dt, status = None, None
    for line in out:
        if line.startswith("DT="):
            parts = line.split()
            dt = float(parts[0][3:])
            status = parts[1][7:]
    if dt is None:
        print(f"[child failure] rc={p.returncode}")
        print("STDOUT:", p.stdout)
        print("STDERR:", err[:2000])
    return wall, dt, status


def main():
    # Single representative size — each row is a fresh interpreter.
    print(f"{'N':>6} {'HL':>6} {'proc wall (s)':>14} {'inner dt (s)':>13} {'status':>8}")
    for N, HL in [(800, 0), (800, 50), (1600, 0), (1600, 50)]:
        print(f"... running N={N} HL={HL}", flush=True)
        wall, dt, status = timed_run(N, HL)
        if dt is None:
            print(f"{N:>6} {HL:>6}  <child error>")
            continue
        print(f"{N:>6} {HL:>6} {wall:>14.3f} {dt:>13.4f} {status:>8}")


if __name__ == "__main__":
    main()
