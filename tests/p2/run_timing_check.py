"""
Quick timing validation script - runs each kernel sequentially and writes results.
Usage: python run_timing_check.py [arch]
"""
import subprocess
import sys
import os
import time

PY_CAND = sys.executable
ARCH = sys.argv[1] if len(sys.argv) > 1 else "cpu"

KERNELS = ["diffuse_auto", "mpm_p2g", "sph_force", "mat_chain3", "mat14"]

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "_runner.py")
OUT = os.path.join(HERE, "p2_timings_quick.txt")

results = []
for k in KERNELS:
    print(f"[{k}] starting...", flush=True)
    env = dict(os.environ)
    env["_HK_NAME"] = k
    env["_HK_ARCH"] = ARCH
    t0 = time.perf_counter()
    proc = subprocess.run(
        [PY_CAND, RUNNER],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    elapsed = time.perf_counter() - t0
    last_line = proc.stdout.strip().split("\n")[-1] if proc.stdout.strip() else ""
    status = "OK" if proc.returncode == 0 else f"ERR({proc.returncode})"
    row = f"{k}\t{ARCH}\t{elapsed:.1f}s\t{last_line}\t{status}"
    results.append(row)
    print(f"  => {row}", flush=True)
    if proc.returncode != 0:
        print(f"  STDERR: {proc.stderr[-400:]}", flush=True)

with open(OUT, "w") as f:
    f.write("\n".join(results) + "\n")

print(f"\nResults written to {OUT}")
