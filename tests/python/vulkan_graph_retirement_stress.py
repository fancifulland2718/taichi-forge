"""Vulkan graph registration/retirement churn and memory probe.

Each graph uses all eight replay slots and then replays slot zero. Graphs are
destroyed without a per-graph sync so the runtime must retain in-flight state
only until its completion fence becomes ready.
"""

import argparse
import ctypes
import gc
import json
import os
import shutil
import subprocess

import numpy as np

import taichi_forge as ti


def _gpu_memory_mib():
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return None
    result = subprocess.run(
        [nvidia_smi, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    try:
        return int(result.stdout.splitlines()[0].strip())
    except (IndexError, ValueError):
        return None


def _rss_mib():
    class ProcessMemoryCounters(ctypes.Structure):
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
        ]

    if hasattr(ctypes, "WinDLL"):
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        ok = psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(),
            ctypes.byref(counters),
            counters.cb,
        )
        if ok:
            return round(counters.WorkingSetSize / (1024 * 1024), 3)

    # Linux release runners expose the current resident page count through
    # procfs. Keep this probe dependency-free so it also works in the wheel
    # validation environment where psutil is intentionally not required.
    if os.name == "posix":
        try:
            with open("/proc/self/statm", encoding="ascii") as statm:
                resident_pages = int(statm.read().split()[1])
            page_size = os.sysconf("SC_PAGE_SIZE")
            return round(resident_pages * page_size / (1024 * 1024), 3)
        except (OSError, ValueError, IndexError):
            pass
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", type=int, default=256)
    parser.add_argument("--items", type=int, default=4096)
    parser.add_argument("--runs-per-graph", type=int, default=9)
    args = parser.parse_args()
    if args.graphs < 1 or args.items < 1 or args.runs_per_graph < 9:
        parser.error("graphs/items must be positive and runs-per-graph >= 9")

    ti.init(arch=ti.vulkan, enable_fallback=False)

    @ti.kernel
    def add_bias(
        values: ti.types.ndarray(dtype=ti.i32, ndim=1), bias: ti.i32
    ):
        for i in values:
            values[i] += bias

    sym_values = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "values", ti.i32, ndim=1
    )
    sym_bias = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "bias", ti.i32)
    values = ti.ndarray(ti.i32, shape=args.items)
    values.fill(0)

    def run_graph(bias):
        builder = ti.graph.GraphBuilder()
        builder.dispatch(add_bias, sym_values, sym_bias)
        builder.dispatch(add_bias, sym_values, sym_bias)
        graph = builder.compile()
        runtime_args = {"values": values, "bias": bias}
        for _ in range(args.runs_per_graph):
            graph.run(runtime_args)
        del graph
        del builder

    run_graph(1)
    ti.sync()
    gc.collect()
    values.fill(0)
    rss_before_mib = _rss_mib()
    gpu_before_mib = _gpu_memory_mib()

    for iteration in range(args.graphs):
        run_graph(iteration + 1)
        if (iteration + 1) % 16 == 0:
            gc.collect()

    ti.sync()
    gc.collect()
    expected = (
        2
        * args.runs_per_graph
        * sum(range(1, args.graphs + 1))
    )
    np.testing.assert_array_equal(
        values.to_numpy(),
        np.full(args.items, expected, dtype=np.int32),
    )
    rss_after_mib = _rss_mib()
    gpu_after_mib = _gpu_memory_mib()
    print(
        json.dumps(
            {
                "graphs": args.graphs,
                "items": args.items,
                "runs_per_graph": args.runs_per_graph,
                "rss_before_mib": rss_before_mib,
                "rss_after_mib": rss_after_mib,
                "rss_delta_mib": (
                    None
                    if rss_before_mib is None or rss_after_mib is None
                    else round(rss_after_mib - rss_before_mib, 3)
                ),
                "gpu_memory_before_mib": gpu_before_mib,
                "gpu_memory_after_mib": gpu_after_mib,
                "result": "pass",
            },
            sort_keys=True,
        )
    )

    del values
    ti.reset()


if __name__ == "__main__":
    main()
