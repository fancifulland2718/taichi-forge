from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path




def process_rss_mb(pid: int) -> float:
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            process_query = 0x0400
            process_vm_read = 0x0010
            handle = ctypes.windll.kernel32.OpenProcess(
                process_query | process_vm_read, False, pid
            )
            if not handle:
                return 0.0
            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            ok = ctypes.WinDLL("psapi.dll").GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb
            )
            ctypes.windll.kernel32.CloseHandle(handle)
            if ok:
                return counters.WorkingSetSize / (1024.0 * 1024.0)
        except Exception:
            return 0.0
    else:
        try:
            status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
            for line in status.splitlines():
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
        except Exception:
            return 0.0
    return 0.0

WORKER = r"""
import json
import os
import sys
import time

import taichi_forge as ti


def current_rss_mb():
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        pass
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        psapi = ctypes.WinDLL("psapi.dll")
        ok = psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        )
        if ok:
            return counters.PeakWorkingSetSize / (1024.0 * 1024.0)
    except Exception:
        pass
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return 0.0
mode = sys.argv[1]
arch_name = sys.argv[2]
n = int(sys.argv[3])
threads = int(sys.argv[4])

arch = {"cpu": ti.cpu, "cuda": ti.cuda, "vulkan": ti.vulkan}[arch_name]
ti.init(arch=arch, offline_cache=False, num_compile_threads=threads, log_level=ti.WARN)

fields = []
kernels = []
for idx in range(n):
    f = ti.field(ti.f32, shape=64)
    fields.append(f)
    a = float(idx + 1)
    b = float((idx * 7) % 13 + 1)

    @ti.kernel
    def kernel(out: ti.template()):
        for i in out:
            x = ti.cast(i, ti.f32) * 0.01 + 0.5
            s = 0.0
            for j in range(64):
                y = x + ti.cast(j, ti.f32) * 0.025
                s += ti.sin(y * a) * ti.cos(y * b)
            out[i] = s

    kernels.append((kernel, (f,)))

t0 = time.perf_counter()
if mode == "serial":
    for kernel, args in kernels:
        kernel(*args)
    ti.sync()
    t_materialize = 0.0
    t_submit = 0.0
elif mode in ("parallel", "duplicate"):
    from taichi_forge.lang import impl as _impl

    specs = []
    t_mat0 = time.perf_counter()
    for kernel, args in kernels:
        primal = kernel._primal if hasattr(kernel, "_primal") and not hasattr(kernel, "ensure_compiled") else kernel
        key = primal.ensure_compiled(*args)
        specs.append(primal.compiled_kernels[key])
    if mode == "duplicate":
        specs = [spec for spec in specs for _ in range(3)]
    t_materialize = time.perf_counter() - t_mat0

    prog = _impl.get_runtime().prog
    t_submit0 = time.perf_counter()
    prog.compile_kernels(prog.config(), specs)
    t_submit = time.perf_counter() - t_submit0
else:
    raise RuntimeError(f"unknown mode {mode}")

t_compile = time.perf_counter() - t0

for kernel, args in kernels:
    kernel(*args)
ti.sync()

t_run0 = time.perf_counter()
for kernel, args in kernels:
    kernel(*args)
ti.sync()
t_warm = (time.perf_counter() - t_run0) / max(1, n)

checksum = sum(float(f[0]) for f in fields)
rss_mb = current_rss_mb()
print("__D4_BENCH__" + json.dumps({
    "arch": arch_name,
    "mode": mode,
    "n": n,
    "threads": threads,
    "compile_ms": t_compile * 1000.0,
    "materialize_ms": t_materialize * 1000.0,
    "submit_ms": t_submit * 1000.0,
    "warm_runtime_ms_per_kernel": t_warm * 1000.0,
    "rss_mb": rss_mb,
    "checksum": checksum,
}, sort_keys=True))
"""


def run_worker(script: Path, mode: str, arch: str, n: int, threads: int) -> dict:
    proc = subprocess.Popen(
        [sys.executable, str(script), mode, arch, str(n), str(threads)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    peak_rss_mb = 0.0
    deadline = time.time() + 600.0
    while proc.poll() is None:
        peak_rss_mb = max(peak_rss_mb, process_rss_mb(proc.pid))
        if time.time() > deadline:
            proc.kill()
            raise RuntimeError("worker timed out")
        time.sleep(0.02)
    stdout, stderr = proc.communicate()
    peak_rss_mb = max(peak_rss_mb, process_rss_mb(proc.pid))
    if proc.returncode != 0:
        raise RuntimeError(stderr[-1000:])
    for line in stdout.splitlines():
        if line.startswith("__D4_BENCH__"):
            row = json.loads(line[len("__D4_BENCH__") :])
            if peak_rss_mb > 0:
                row["rss_mb"] = peak_rss_mb
            return row
    raise RuntimeError(stdout[-1000:])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="all", choices=["cpu", "cuda", "vulkan", "all"])
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    archs = ["cpu", "cuda", "vulkan"] if args.arch == "all" else [args.arch]
    modes = ["serial", "parallel", "duplicate"]
    rows = []
    with tempfile.TemporaryDirectory(prefix="d4_cpp_ir_bench_") as tmp:
        worker = Path(tmp) / "worker.py"
        worker.write_text(textwrap.dedent(WORKER), encoding="utf-8")
        for arch in archs:
            for mode in modes:
                for _ in range(args.repeat):
                    try:
                        rows.append(run_worker(worker, mode, arch, args.n, args.threads))
                    except Exception as exc:
                        rows.append({
                            "arch": arch,
                            "mode": mode,
                            "error": str(exc),
                        })

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
