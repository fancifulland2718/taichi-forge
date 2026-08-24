"""Process-level memory checkpoints for provider lifecycle qualification."""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


SCHEMA = "taichi_forge.hardware_process_memory.v1"
OUTPUT_ENV = "TI_HARDWARE_PROCESS_MEMORY_OUTPUT"
MINIMUM_QUALIFICATION_ITERATIONS = 1_000
RSS_PLATEAU_TOLERANCE_BYTES = 64 * 1024 * 1024
GPU_PLATEAU_TOLERANCE_BYTES = 16 * 1024 * 1024
_PHASES = ("before", "midpoint", "after")


def _rss_bytes():
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
        if get_process_memory_info(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            return int(counters.WorkingSetSize), "windows_working_set", None
        return None, None, f"GetProcessMemoryInfo failed: {ctypes.get_last_error()}"
    if sys.platform.startswith("linux"):
        try:
            resident_pages = int(
                Path("/proc/self/statm").read_text(encoding="ascii").split()[1]
            )
            return (
                resident_pages * int(os.sysconf("SC_PAGE_SIZE")),
                "linux_proc_statm",
                None,
            )
        except (OSError, ValueError, IndexError) as exc:
            return None, None, f"/proc/self/statm unavailable: {type(exc).__name__}"
    return None, None, f"current RSS is unsupported on {sys.platform}"


def _nvidia_process_gpu_bytes():
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None, None, "nvidia-smi_not_found"
    try:
        completed = subprocess.run(
            [
                executable,
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, None, f"nvidia-smi_failed:{type(exc).__name__}"
    if completed.returncode != 0:
        return None, None, "nvidia-smi_query_failed"
    values = []
    saw_process = False
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 1)]
        if len(fields) != 2 or fields[0] != str(os.getpid()):
            continue
        saw_process = True
        try:
            values.append(int(fields[1]))
        except ValueError:
            return None, None, "nvidia-smi_process_memory_unavailable"
    if not values:
        reason = (
            "nvidia-smi_process_memory_unavailable"
            if saw_process
            else "nvidia-smi_process_not_listed"
        )
        return None, None, reason
    return sum(values) * 1024 * 1024, "nvidia-smi_compute_process", None


def _append_record(path, record):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": SCHEMA, "records": []}
    if output.is_file():
        try:
            existing = json.loads(output.read_text(encoding="utf-8"))
            if existing.get("schema") == SCHEMA and isinstance(
                existing.get("records"), list
            ):
                payload = existing
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass
    payload["records"].append(record)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


class ProcessMemoryPlateau:
    """Record exact loop-aligned memory checkpoints when qualification enables it."""

    def __init__(self, case_id, providers, *, enabled=True):
        self.case_id = str(case_id)
        self.providers = tuple(str(provider) for provider in providers)
        self.output = os.environ.get(OUTPUT_ENV) if enabled else None
        self.samples = {}

    @property
    def enabled(self):
        return bool(self.output)

    def capture(self, phase):
        if not self.enabled:
            return
        if phase not in _PHASES or phase in self.samples:
            raise ValueError(f"invalid or duplicate process-memory phase {phase!r}")
        rss, rss_source, rss_reason = _rss_bytes()
        gpu, gpu_source, gpu_reason = _nvidia_process_gpu_bytes()
        self.samples[phase] = {
            "rss_bytes": rss,
            "rss_source": rss_source,
            "rss_unavailable_reason": rss_reason,
            "gpu_process_bytes": gpu,
            "gpu_process_source": gpu_source,
            "gpu_process_unavailable_reason": gpu_reason,
        }

    def finish(self, iterations):
        if not self.enabled:
            return None
        missing = tuple(phase for phase in _PHASES if phase not in self.samples)
        if missing:
            raise RuntimeError(f"missing process-memory checkpoints: {missing}")
        iterations = int(iterations)
        before = self.samples["before"]
        midpoint = self.samples["midpoint"]
        after = self.samples["after"]
        rss_available = all(
            sample["rss_bytes"] is not None for sample in self.samples.values()
        )
        gpu_available = all(
            sample["gpu_process_bytes"] is not None for sample in self.samples.values()
        )
        rss_growth = (
            after["rss_bytes"] - midpoint["rss_bytes"] if rss_available else None
        )
        gpu_growth = (
            after["gpu_process_bytes"] - midpoint["gpu_process_bytes"]
            if gpu_available
            else None
        )
        enough_iterations = iterations >= MINIMUM_QUALIFICATION_ITERATIONS
        rss_plateau = rss_available and rss_growth <= RSS_PLATEAU_TOLERANCE_BYTES
        gpu_plateau = gpu_available and gpu_growth <= GPU_PLATEAU_TOLERANCE_BYTES
        process_level_qualified = bool(
            enough_iterations and rss_plateau and gpu_plateau
        )
        record = {
            "case_id": self.case_id,
            "providers": self.providers,
            "pid": os.getpid(),
            "iterations": iterations,
            "samples": self.samples,
            "plateau": {
                "midpoint_to_after_rss_growth_bytes": rss_growth,
                "midpoint_to_after_gpu_process_growth_bytes": gpu_growth,
                "rss_tolerance_bytes": RSS_PLATEAU_TOLERANCE_BYTES,
                "gpu_process_tolerance_bytes": GPU_PLATEAU_TOLERANCE_BYTES,
            },
            "qualification": {
                "minimum_iterations": MINIMUM_QUALIFICATION_ITERATIONS,
                "minimum_iterations_met": enough_iterations,
                "rss_available": rss_available,
                "rss_plateau": rss_plateau,
                "gpu_process_available": gpu_available,
                "gpu_process_plateau": gpu_plateau,
                "process_level_memory_qualified": process_level_qualified,
            },
        }
        _append_record(self.output, record)
        return record


__all__ = [
    "GPU_PLATEAU_TOLERANCE_BYTES",
    "MINIMUM_QUALIFICATION_ITERATIONS",
    "OUTPUT_ENV",
    "ProcessMemoryPlateau",
    "RSS_PLATEAU_TOLERANCE_BYTES",
    "SCHEMA",
]
