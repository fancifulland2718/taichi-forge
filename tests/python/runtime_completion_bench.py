"""Controlled F2 completion/default-submit microbenchmark.

GPU callers must enforce process/GPU-idle policy outside this process so the
monitor can observe workloads that appear at any point during the run.
"""

import argparse
import ctypes
import json
import os
import statistics
import sys
import time

import taichi_forge as ti
from taichi_forge.lang import impl


def summary(samples):
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, int((len(ordered) - 1) * 0.95 + 0.5))
    return {
        "median_us": statistics.median(samples),
        "p95_us": ordered[p95_index],
        "min_us": ordered[0],
        "max_us": ordered[-1],
        "samples_us": samples,
    }


def rss_bytes():
    if sys.platform == "win32":
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

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        get_process_memory_info = psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        )
        get_process_memory_info.restype = ctypes.c_bool
        process = kernel32.GetCurrentProcess()
        ok = get_process_memory_info(
            process, ctypes.byref(counters), counters.cb
        )
        return int(counters.WorkingSetSize) if ok else 0
    if sys.platform.startswith("linux"):
        with open("/proc/self/statm", "r", encoding="ascii") as statm:
            resident_pages = int(statm.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=("cpu", "cuda", "vulkan"), required=True)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--submit-iterations", type=int, default=2000)
    parser.add_argument("--completion-iterations", type=int, default=200)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    arch = {"cpu": ti.cpu, "cuda": ti.cuda, "vulkan": ti.vulkan}[args.arch]
    ti.init(arch=arch)
    state = ti.field(ti.i32, shape=1)

    @ti.kernel
    def step():
        state[0] += 1

    builder = ti.graph.GraphBuilder()
    builder.dispatch(step)
    graph = builder.compile()

    step()
    graph.run({})
    ti.sync()
    rss_before = rss_bytes()

    kernel_samples = []
    graph_samples = []
    for _ in range(args.rounds):
        begin = time.perf_counter_ns()
        for _ in range(args.submit_iterations):
            step()
        elapsed = time.perf_counter_ns() - begin
        kernel_samples.append(elapsed / args.submit_iterations / 1000.0)
        ti.sync()

        begin = time.perf_counter_ns()
        for _ in range(args.submit_iterations):
            graph.run({})
        elapsed = time.perf_counter_ns() - begin
        graph_samples.append(elapsed / args.submit_iterations / 1000.0)
        ti.sync()

    result = {
        "arch": args.arch,
        "rounds": args.rounds,
        "submit_iterations": args.submit_iterations,
        "ordinary_kernel": summary(kernel_samples),
        "ordinary_graph": summary(graph_samples),
        "rss_before_bytes": rss_before,
        "completion_available": hasattr(
            impl.get_runtime().prog, "_record_runtime_completion"
        ),
    }

    prog = impl.get_runtime().prog
    if result["completion_available"]:
        record_samples = []
        poll_samples = []
        wait_samples = []
        empty_samples = []
        for _ in range(args.rounds):
            record_total = 0
            poll_total = 0
            wait_total = 0
            empty_total = 0
            for _ in range(args.completion_iterations):
                step()
                begin = time.perf_counter_ns()
                ticket = prog._record_runtime_completion()
                record_total += time.perf_counter_ns() - begin

                begin = time.perf_counter_ns()
                ticket.done()
                poll_total += time.perf_counter_ns() - begin

                begin = time.perf_counter_ns()
                ticket.wait()
                wait_total += time.perf_counter_ns() - begin

                begin = time.perf_counter_ns()
                empty = prog._record_runtime_completion()
                empty_total += time.perf_counter_ns() - begin
                assert empty.done() and empty.sequence == ticket.sequence
            scale = args.completion_iterations * 1000.0
            record_samples.append(record_total / scale)
            poll_samples.append(poll_total / scale)
            wait_samples.append(wait_total / scale)
            empty_samples.append(empty_total / scale)
        result.update(
            {
                "completion_iterations": args.completion_iterations,
                "completion_record": summary(record_samples),
                "completion_poll": summary(poll_samples),
                "completion_wait": summary(wait_samples),
                "completion_empty": summary(empty_samples),
                "completion_stats": prog._debug_runtime_completion_stats(),
            }
        )

    ti.sync()
    result["rss_after_bytes"] = rss_bytes()
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(result, output, sort_keys=True)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
