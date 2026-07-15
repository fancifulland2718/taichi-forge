"""Fresh-process host allocator attribution for CPU/CUDA/Vulkan runtimes."""

import argparse
import ctypes
import gc
import json
import os
import platform
import time
from pathlib import Path

ti = None


def _windows_memory():
    class ProcessMemoryCountersEx(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("page_fault_count", ctypes.c_ulong),
            ("peak_working_set_bytes", ctypes.c_size_t),
            ("working_set_bytes", ctypes.c_size_t),
            ("quota_peak_paged_pool_bytes", ctypes.c_size_t),
            ("quota_paged_pool_bytes", ctypes.c_size_t),
            ("quota_peak_nonpaged_pool_bytes", ctypes.c_size_t),
            ("quota_nonpaged_pool_bytes", ctypes.c_size_t),
            ("pagefile_bytes", ctypes.c_size_t),
            ("peak_pagefile_bytes", ctypes.c_size_t),
            ("private_bytes", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ProcessMemoryCountersEx),
        ctypes.c_ulong,
    ]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int

    counters = ProcessMemoryCountersEx()
    counters.cb = ctypes.sizeof(counters)
    handle = kernel32.GetCurrentProcess()
    ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    if not ok:
        raise ctypes.WinError()
    return {
        "working_set_bytes": counters.working_set_bytes,
        "peak_working_set_bytes": counters.peak_working_set_bytes,
        "private_bytes": counters.private_bytes,
        "pagefile_bytes": counters.pagefile_bytes,
        "page_fault_count": counters.page_fault_count,
    }


def _linux_memory():
    import resource

    values = {}
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition(":")
            if separator and key in {"VmRSS", "VmSize", "VmData", "VmPeak"}:
                values[f"{key.lower()}_bytes"] = int(value.split()[0]) * 1024
    usage = resource.getrusage(resource.RUSAGE_SELF)
    values["minor_page_faults"] = usage.ru_minflt
    values["major_page_faults"] = usage.ru_majflt
    return values


def _process_memory():
    if platform.system() == "Windows":
        return _windows_memory()
    if platform.system() == "Linux":
        return _linux_memory()
    return {}


def _host_allocator():
    return dict(ti.tools.memory_pool_stats()["host"])


def _stage(name, started_at=None):
    result = {
        "stage": name,
        "process": _process_memory(),
        "host_allocator": _host_allocator(),
    }
    if started_at is not None:
        result["elapsed_ms"] = (time.perf_counter() - started_at) * 1000.0
    return result


def main():
    global ti

    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=("cpu", "cuda", "vulkan"), required=True)
    parser.add_argument(
        "--policy", choices=("adaptive", "legacy"), default="adaptive"
    )
    parser.add_argument("--arrays", type=int, default=256)
    parser.add_argument("--items", type=int, default=4096)
    args = parser.parse_args()
    if args.arrays <= 0 or args.items <= 0:
        parser.error("--arrays and --items must be positive")

    os.environ["TI_HOST_ALLOCATOR_ADAPTIVE_CHUNKS"] = (
        "1" if args.policy == "adaptive" else "0"
    )
    import taichi_forge as ti_module

    ti = ti_module
    arch = {"cpu": ti.cpu, "cuda": ti.cuda, "vulkan": ti.vulkan}[args.arch]
    stages = [_stage("imported")]

    started = time.perf_counter()
    ti.init(arch=arch, offline_cache=False)
    stages.append(_stage("initialized", started))

    field = ti.field(ti.f32, shape=args.items)

    @ti.kernel
    def initialize_field():
        for index in field:
            field[index] = index * 0.25

    @ti.kernel
    def advance(values: ti.types.ndarray()):
        for index in values:
            values[index] += field[index]

    started = time.perf_counter()
    initialize_field()
    warm = ti.ndarray(ti.f32, shape=args.items)
    advance(warm)
    ti.sync()
    stages.append(_stage("compiled", started))

    started = time.perf_counter()
    arrays = [ti.ndarray(ti.f32, shape=args.items) for _ in range(args.arrays)]
    for values in arrays:
        advance(values)
    ti.sync()
    stages.append(_stage("array_churn_live", started))

    started = time.perf_counter()
    arrays.clear()
    del warm
    gc.collect()
    ti.sync()
    stages.append(_stage("array_churn_released", started))

    started = time.perf_counter()
    ti.reset()
    stages.append(_stage("reset", started))
    print(
        json.dumps(
            {
                "schema": "taichi_forge.host_allocator_runtime.v1",
                "platform": platform.platform(),
                "pid": os.getpid(),
                "arch": args.arch,
                "policy": args.policy,
                "arrays": args.arrays,
                "items": args.items,
                "stages": stages,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
