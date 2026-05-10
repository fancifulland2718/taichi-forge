"""Hash SNode correctness and performance comparison.

The script intentionally uses only the Python standard library plus
taichi_forge. Each Taichi case runs in a fresh child process so CPU/CUDA/Vulkan
runtime state and memory counters stay isolated.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


ARCHES = ("cpu", "cuda", "vulkan")
LAYOUTS = ("hash", "pointer_bitmasked", "bitmasked", "dense")


def make_key(i: int, domain: int) -> int:
    return (i * 131071 + 17) % domain


def make_value(key: int) -> int:
    return key % 97 + 1


def expected(active: int, domain: int) -> dict[str, int]:
    keys = [make_key(i, domain) for i in range(active)]
    values = [make_value(k) for k in keys]
    return {
        "count": active,
        "key_sum": sum(keys),
        "value_sum": sum(values),
    }


def stats_ms(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {}
    mean = statistics.fmean(samples)
    return {
        "samples": len(samples),
        "mean_ms": mean,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "cv_pct": statistics.pstdev(samples) / mean * 100.0 if mean else 0.0,
    }


def process_private_mb() -> float:
    if os.name != "nt":
        return -1.0

    class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    counters = PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = ctypes.sizeof(counters)
    psapi = ctypes.WinDLL("psapi.dll")
    kernel32 = ctypes.WinDLL("kernel32.dll")
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
        ctypes.c_ulong,
    ]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_bool
    ok = psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(),
        ctypes.byref(counters),
        counters.cb,
    )
    if not ok:
        return -1.0
    return counters.PrivateUsage / 1048576.0


def gpu_process_memory_counter_mb(pid: int | None = None, counter: str = "Dedicated Usage") -> float:
    if os.name != "nt":
        return -1.0
    if pid is None:
        pid = os.getpid()
    ps = (
        "$pidToFind = " + str(int(pid)) + "; "
        "$pattern = 'pid_' + $pidToFind + '_*'; "
        "$sum = 0; "
        "try { "
        f"  (Get-Counter '\\GPU Process Memory(*)\\{counter}').CounterSamples | "
        "    Where-Object { $_.InstanceName -like $pattern } | "
        "    ForEach-Object { $sum += $_.CookedValue }; "
        "  [Console]::WriteLine([math]::Round($sum / 1MB, 3)) "
        "} catch { [Console]::WriteLine(-1) }"
    )
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).strip()
        return float(out.splitlines()[-1]) if out else -1.0
    except BaseException:
        return -1.0


def nvidia_smi_compute_mb(pid: int | None = None) -> float:
    if pid is None:
        pid = os.getpid()
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).strip()
    except BaseException:
        return -1.0
    total = 0.0
    seen = False
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0] == str(int(pid)):
            try:
                total += float(parts[1])
            except ValueError:
                continue
            seen = True
    return total if seen else -1.0


def sample_memory() -> dict[str, float]:
    return {
        "process_private_mb": process_private_mb(),
        "gpu_dedicated_mb": gpu_process_memory_counter_mb(counter="Dedicated Usage"),
        "gpu_shared_mb": gpu_process_memory_counter_mb(counter="Shared Usage"),
        "nvidia_smi_compute_mb": nvidia_smi_compute_mb(),
    }


def run_python_dict(active: int, domain: int, steps: int, warmup: int, batch: int) -> dict:
    exp = expected(active, domain)
    t0 = time.perf_counter()
    data = {}
    for i in range(active):
        key = make_key(i, domain)
        data[key] = make_value(key)
    count = len(data)
    key_sum = sum(data.keys())
    value_sum = sum(data.values())
    compile_s = time.perf_counter() - t0
    ok = {"count": count, "key_sum": key_sum, "value_sum": value_sum} == exp

    for _ in range(warmup):
        for i in range(active):
            key = make_key(i, domain)
            data[key] = make_value(key)
    write_samples = []
    for _ in range(steps):
        t0 = time.perf_counter()
        for _ in range(batch):
            for i in range(active):
                key = make_key(i, domain)
                data[key] = make_value(key)
        write_samples.append((time.perf_counter() - t0) * 1000.0 / batch)

    reduce_samples = []
    for _ in range(warmup):
        count = len(data)
        key_sum = sum(data.keys())
        value_sum = sum(data.values())
    for _ in range(steps):
        t0 = time.perf_counter()
        for _ in range(batch):
            count = len(data)
            key_sum = sum(data.keys())
            value_sum = sum(data.values())
        reduce_samples.append((time.perf_counter() - t0) * 1000.0 / batch)

    return {
        "case": "python_dict",
        "arch": "python",
        "layout": "dict",
        "ok": ok,
        "compile_first_s": compile_s,
        "result": {"count": count, "key_sum": key_sum, "value_sum": value_sum},
        "expected": exp,
        "write": stats_ms(write_samples),
        "reduce": stats_ms(reduce_samples),
        "memory": sample_memory(),
    }


def run_taichi_case(
    arch_name: str,
    layout: str,
    active: int,
    domain: int,
    steps: int,
    warmup: int,
    batch: int,
) -> dict:
    import taichi_forge as ti

    arch = getattr(ti, arch_name)
    init_kwargs = {
        "arch": arch,
        "offline_cache": False,
        "debug": False,
        "log_level": "warn",
        "hash_snode_experimental": True,
    }
    if arch_name == "vulkan":
        init_kwargs.update(
            {
                "vulkan_sparse_experimental": True,
                "vulkan_listgen_dynamic_size": True,
            }
        )
    ti.init(**init_kwargs)

    mem_after_init = sample_memory()

    max_key_sum = active * max(0, domain - 1)
    if arch_name == "vulkan" and max_key_sum > 2_000_000_000:
        raise RuntimeError(
            "Vulkan hash benchmark key_sum exceeds the i32 accumulator range; "
            "use a smaller active/domain pair for Vulkan until i64 scalar "
            "atomics are part of the benchmark target."
        )
    acc_dtype = ti.i32 if arch_name == "vulkan" else ti.i64

    x = ti.field(ti.i32)
    count = ti.field(ti.i32, shape=())
    key_sum = ti.field(acc_dtype, shape=())
    value_sum = ti.field(acc_dtype, shape=())

    if layout == "hash":
        ti.root.hash(ti.i, domain, expected_active=active).place(x)
    elif layout == "pointer_bitmasked":
        block = 64
        ti.root.pointer(ti.i, max(1, domain // block)).bitmasked(ti.i, block).place(x)
    elif layout == "bitmasked":
        ti.root.bitmasked(ti.i, domain).place(x)
    elif layout == "dense":
        ti.root.dense(ti.i, domain).place(x)
    else:
        raise RuntimeError(f"unknown layout {layout}")

    @ti.func
    def bench_key(i):
        return (i * 131071 + 17) % domain

    @ti.func
    def bench_value(k):
        return k % 97 + 1

    @ti.kernel
    def write():
        for p in range(active):
            key = bench_key(p)
            x[key] = bench_value(key)

    @ti.kernel
    def clear_acc():
        count[None] = 0
        key_sum[None] = 0
        value_sum[None] = 0

    if layout == "dense":

        @ti.kernel
        def reduce():
            for i in x:
                v = x[i]
                if v != 0:
                    count[None] += 1
                    key_sum[None] += i
                    value_sum[None] += v

    else:

        @ti.kernel
        def reduce():
            for i in x:
                count[None] += 1
                key_sum[None] += i
                value_sum[None] += x[i]

    exp = expected(active, domain)

    t0 = time.perf_counter()
    write()
    clear_acc()
    reduce()
    ti.sync()
    compile_first_s = time.perf_counter() - t0
    mem_after_first = sample_memory()

    result = {
        "count": int(count[None]),
        "key_sum": int(key_sum[None]),
        "value_sum": int(value_sum[None]),
    }
    ok = result == exp

    for _ in range(warmup):
        write()
    ti.sync()
    write_samples = []
    for _ in range(steps):
        t0 = time.perf_counter()
        for _ in range(batch):
            write()
        ti.sync()
        write_samples.append((time.perf_counter() - t0) * 1000.0 / batch)

    for _ in range(warmup):
        clear_acc()
        reduce()
    ti.sync()
    reduce_samples = []
    for _ in range(steps):
        t0 = time.perf_counter()
        for _ in range(batch):
            clear_acc()
            reduce()
        ti.sync()
        reduce_samples.append((time.perf_counter() - t0) * 1000.0 / batch)

    clear_acc()
    reduce()
    ti.sync()
    final_result = {
        "count": int(count[None]),
        "key_sum": int(key_sum[None]),
        "value_sum": int(value_sum[None]),
    }

    return {
        "case": f"{arch_name}_{layout}",
        "arch": arch_name,
        "layout": layout,
        "ok": ok and final_result == exp,
        "compile_first_s": compile_first_s,
        "result": final_result,
        "expected": exp,
        "write": stats_ms(write_samples),
        "reduce": stats_ms(reduce_samples),
        "memory": {
            "after_init": mem_after_init,
            "after_first": mem_after_first,
            "after_bench": sample_memory(),
        },
    }


def print_table(results: list[dict]) -> None:
    print("\ncase,ok,compile_s,write_median_ms,reduce_median_ms,proc_mb,gpu_ded_mb")
    for item in results:
        mem = item.get("memory", {})
        if "after_bench" in mem:
            proc = mem["after_bench"].get("process_private_mb", -1.0)
            gpu = mem["after_bench"].get("gpu_dedicated_mb", -1.0)
        else:
            proc = mem.get("process_private_mb", -1.0)
            gpu = mem.get("gpu_dedicated_mb", -1.0)
        print(
            "{case},{ok},{compile:.6f},{write:.6f},{reduce:.6f},{proc:.3f},{gpu:.3f}".format(
                case=item.get("case"),
                ok=item.get("ok"),
                compile=item.get("compile_first_s", -1.0),
                write=item.get("write", {}).get("median_ms", -1.0),
                reduce=item.get("reduce", {}).get("median_ms", -1.0),
                proc=proc,
                gpu=gpu,
            )
        )


def child_main(args: argparse.Namespace) -> int:
    try:
        if args.case == "python_dict":
            result = run_python_dict(args.active, args.domain, args.steps, args.warmup, args.batch)
        else:
            arch, layout = args.case.split(":", 1)
            result = run_taichi_case(
                arch, layout, args.active, args.domain, args.steps, args.warmup, args.batch
            )
        print("HASH_SNODE_BENCH_RESULT " + json.dumps(result, sort_keys=True))
        return 0 if result.get("ok") else 2
    except BaseException as exc:
        result = {"case": args.case, "ok": False, "error": repr(exc)}
        print("HASH_SNODE_BENCH_RESULT " + json.dumps(result, sort_keys=True))
        return 1


def parent_main(args: argparse.Namespace) -> int:
    cases = args.cases
    if not cases:
        cases = ["python_dict"]
        cases.extend(f"{arch}:{layout}" for arch in ARCHES for layout in LAYOUTS)

    results = []
    for case in cases:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            "--case",
            case,
            "--active",
            str(args.active),
            "--domain",
            str(args.domain),
            "--steps",
            str(args.steps),
            "--warmup",
            str(args.warmup),
            "--batch",
            str(args.batch),
        ]
        print(f"[hash-bench] running {case}", file=sys.stderr, flush=True)
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=args.timeout,
        )
        parsed = None
        for line in proc.stdout.splitlines():
            if line.startswith("HASH_SNODE_BENCH_RESULT "):
                parsed = json.loads(line.split(" ", 1)[1])
        if parsed is None:
            parsed = {
                "case": case,
                "ok": False,
                "error": "missing result line",
                "returncode": proc.returncode,
                "output_tail": proc.stdout.splitlines()[-20:],
            }
        parsed["returncode"] = proc.returncode
        results.append(parsed)
        print(json.dumps(parsed, sort_keys=True), flush=True)

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print_table(results)
    return 0 if all(r.get("ok") for r in results if "error" not in r) else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--case", default="")
    parser.add_argument("--cases", nargs="*", default=[])
    parser.add_argument("--active", type=int, default=4096)
    parser.add_argument("--domain", type=int, default=65536)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    if args.child:
        return child_main(args)
    return parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
