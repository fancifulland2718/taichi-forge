"""Process-level memory checkpoints for provider lifecycle qualification."""

from __future__ import annotations

import ctypes
import ctypes.util
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
_NVML_SUCCESS = 0
_NVML_ERROR_INSUFFICIENT_SIZE = 7
_NVML_VALUE_NOT_AVAILABLE = (1 << 64) - 1


class _NvmlProcessInfo(ctypes.Structure):
    _fields_ = [
        ("pid", ctypes.c_uint),
        ("used_gpu_memory", ctypes.c_ulonglong),
        ("gpu_instance_id", ctypes.c_uint),
        ("compute_instance_id", ctypes.c_uint),
    ]


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


def _load_nvml_library():
    candidates = []
    if sys.platform == "win32":
        system_root = os.environ.get("SystemRoot", r"C:\Windows")
        candidates.extend(
            (str(Path(system_root) / "System32" / "nvml.dll"), "nvml.dll")
        )
        loader = ctypes.WinDLL
    elif sys.platform.startswith("linux"):
        discovered = ctypes.util.find_library("nvidia-ml")
        candidates.extend(
            candidate for candidate in (discovered, "libnvidia-ml.so.1") if candidate
        )
        loader = ctypes.CDLL
    else:
        return None, f"nvml_unsupported_platform:{sys.platform}"
    errors = []
    for candidate in dict.fromkeys(candidates):
        try:
            return loader(candidate), None
        except OSError as exc:
            errors.append(type(exc).__name__)
    detail = errors[-1] if errors else "no_candidate"
    return None, f"nvml_library_unavailable:{detail}"


def _nvml_device_processes(function, device):
    function.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(_NvmlProcessInfo),
    )
    function.restype = ctypes.c_int
    count = ctypes.c_uint(0)
    result = function(device, ctypes.byref(count), None)
    if result == _NVML_SUCCESS and count.value == 0:
        return (), None
    if result not in (_NVML_SUCCESS, _NVML_ERROR_INSUFFICIENT_SIZE):
        return None, f"query_failed:{result}"
    for _ in range(3):
        capacity = max(1, int(count.value) + 4)
        infos = (_NvmlProcessInfo * capacity)()
        count = ctypes.c_uint(capacity)
        result = function(device, ctypes.byref(count), infos)
        if result == _NVML_ERROR_INSUFFICIENT_SIZE:
            continue
        if result != _NVML_SUCCESS:
            return None, f"query_failed:{result}"
        return tuple(infos[index] for index in range(int(count.value))), None
    return None, "process_list_changed_repeatedly"


def _nvml_process_gpu_bytes():
    library, load_reason = _load_nvml_library()
    if library is None:
        return None, None, load_reason
    required = (
        "nvmlInit_v2",
        "nvmlShutdown",
        "nvmlDeviceGetCount_v2",
        "nvmlDeviceGetHandleByIndex_v2",
        "nvmlDeviceGetComputeRunningProcesses_v3",
        "nvmlDeviceGetGraphicsRunningProcesses_v3",
    )
    missing = tuple(name for name in required if not hasattr(library, name))
    if missing:
        return None, None, f"nvml_symbols_unavailable:{','.join(missing)}"

    initialize = library.nvmlInit_v2
    initialize.argtypes = ()
    initialize.restype = ctypes.c_int
    shutdown = library.nvmlShutdown
    shutdown.argtypes = ()
    shutdown.restype = ctypes.c_int
    get_count = library.nvmlDeviceGetCount_v2
    get_count.argtypes = (ctypes.POINTER(ctypes.c_uint),)
    get_count.restype = ctypes.c_int
    get_handle = library.nvmlDeviceGetHandleByIndex_v2
    get_handle.argtypes = (ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))
    get_handle.restype = ctypes.c_int

    result = initialize()
    if result != _NVML_SUCCESS:
        return None, None, f"nvml_initialize_failed:{result}"
    try:
        count = ctypes.c_uint(0)
        result = get_count(ctypes.byref(count))
        if result != _NVML_SUCCESS:
            return None, None, f"nvml_device_count_failed:{result}"
        pid = os.getpid()
        total = 0
        found_exact = False
        found_unavailable = False
        for index in range(int(count.value)):
            device = ctypes.c_void_p()
            result = get_handle(index, ctypes.byref(device))
            if result != _NVML_SUCCESS:
                return None, None, f"nvml_device_handle_failed:{result}"
            device_values = []
            for function_name in (
                "nvmlDeviceGetComputeRunningProcesses_v3",
                "nvmlDeviceGetGraphicsRunningProcesses_v3",
            ):
                infos, reason = _nvml_device_processes(
                    getattr(library, function_name), device
                )
                if infos is None:
                    return None, None, f"nvml_{function_name}:{reason}"
                for info in infos:
                    if int(info.pid) != pid:
                        continue
                    if int(info.used_gpu_memory) == _NVML_VALUE_NOT_AVAILABLE:
                        found_unavailable = True
                    else:
                        device_values.append(int(info.used_gpu_memory))
            if device_values:
                found_exact = True
                # Compute and graphics queries both report process-total memory.
                # Use the larger value per device instead of double counting it.
                total += max(device_values)
        if found_unavailable:
            return None, None, "nvml_process_memory_unavailable"
        if not found_exact:
            return None, None, "nvml_process_not_listed"
        return total, "nvml_compute_graphics_process_v3", None
    finally:
        shutdown()


def _nvidia_smi_process_gpu_bytes():
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


def _windows_process_gpu_bytes():
    if sys.platform != "win32":
        return None, None, f"windows_gpu_process_unsupported:{sys.platform}"
    executable = shutil.which("powershell.exe") or shutil.which("powershell")
    if executable is None:
        return None, None, "windows_powershell_not_found"
    command = (
        "$ErrorActionPreference='Stop';"
        f"$targetPid={os.getpid()};"
        "$rows=@(Get-CimInstance "
        "Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory "
        "| Where-Object { $_.Name -match ('^pid_'+$targetPid+'_') });"
        "$total=($rows | Measure-Object -Property TotalCommitted -Sum).Sum;"
        "if($null -eq $total){$total=0};"
        "[pscustomobject]@{total=[uint64]$total;instances=$rows.Count}"
        "| ConvertTo-Json -Compress"
    )
    try:
        completed = subprocess.run(
            [
                executable,
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                command,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, None, f"windows_gpu_process_failed:{type(exc).__name__}"
    if completed.returncode != 0:
        return None, None, "windows_gpu_process_query_failed"
    try:
        payload = json.loads(completed.stdout)
        total = int(payload["total"])
        instances = int(payload["instances"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None, None, "windows_gpu_process_output_invalid"
    if total < 0 or instances < 0:
        return None, None, "windows_gpu_process_output_invalid"
    return total, "windows_gpu_process_total_committed", None


def _process_gpu_bytes():
    value, source, nvml_reason = _nvml_process_gpu_bytes()
    if value is not None:
        return value, source, None
    windows_reason = None
    if sys.platform == "win32":
        value, source, windows_reason = _windows_process_gpu_bytes()
        if value is not None:
            return value, source, None
    value, source, smi_reason = _nvidia_smi_process_gpu_bytes()
    if value is not None:
        return value, source, None
    reasons = [f"nvml:{nvml_reason}"]
    if windows_reason is not None:
        reasons.append(f"windows:{windows_reason}")
    reasons.append(f"nvidia-smi:{smi_reason}")
    return None, None, ";".join(reasons)


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
        gpu, gpu_source, gpu_reason = _process_gpu_bytes()
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
