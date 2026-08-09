from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence


def percentile(values: Sequence[float], percent: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_samples(samples_ms: Sequence[float]) -> dict[str, float | int]:
    values = [float(value) for value in samples_ms]
    if not values:
        raise ValueError("at least one timing sample is required")
    mean = statistics.fmean(values)
    median = statistics.median(values)
    stddev = statistics.pstdev(values)
    absolute_deviations = [abs(value - median) for value in values]
    return {
        "count": len(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "mean_ms": mean,
        "median_ms": median,
        "p95_ms": percentile(values, 95.0),
        "p99_ms": percentile(values, 99.0),
        "stddev_ms": stddev,
        "cv_percent": 0.0 if mean == 0.0 else stddev / mean * 100.0,
        "mad_ms": statistics.median(absolute_deviations),
    }


def logical_bandwidth_gbps(logical_bytes: int, milliseconds: float) -> float:
    if milliseconds <= 0.0:
        return math.inf
    return logical_bytes / (milliseconds / 1000.0) / 1.0e9


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: Sequence[str], cwd: Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def git_metadata(repo_root: Path) -> dict[str, Any]:
    git = ["git", "-c", f"safe.directory={repo_root.as_posix()}"]
    head = command_output([*git, "rev-parse", "HEAD"], repo_root)
    status = command_output([*git, "status", "--short"], repo_root)
    return {
        "head": head,
        "dirty": bool(status),
        "status_short": [] if not status else status.splitlines(),
    }


def nvidia_smi_rows(query: str) -> list[dict[str, str]]:
    fields = [field.strip() for field in query.split(",")]
    output = command_output([
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ])
    if output is None:
        return []
    rows: list[dict[str, str]] = []
    for line in output.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == len(fields):
            rows.append(dict(zip(fields, values)))
    return rows


def gpu_snapshot() -> list[dict[str, str]]:
    return nvidia_smi_rows(
        "index,uuid,name,driver_version,memory.total,memory.used,"
        "utilization.gpu,temperature.gpu,power.draw,clocks.sm,clocks.mem"
    )


def normalize_gpu_uuid(value: str | bytes | None) -> str | None:
    """Normalizes CUDA/nvidia-smi UUIDs to one lower-case hexadecimal value."""
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.hex().lower()
    normalized = str(value).strip().lower()
    if normalized.startswith("gpu-"):
        normalized = normalized[4:]
    normalized = normalized.replace("-", "")
    if len(normalized) != 32 or any(
            character not in "0123456789abcdef" for character in normalized):
        return None
    return normalized


def runtime_device_identity(ti: Any, backend: str) -> dict[str, Any]:
    """Records and, where possible, proves the physical GPU used by Taichi.

    A single enumerated GPU plus explicit device-zero environment binding is a
    sufficient local proof for runtimes that expose no current-device UUID.
    Forge's CUDA runtime provides a stronger UUID that is matched against
    nvidia-smi. Multi-GPU runs fail closed unless that stronger proof exists.
    """
    if backend == "cpu":
        return {
            "backend": backend,
            "required": False,
            "binding_verified": True,
            "verification": "not_applicable",
        }
    rows = gpu_snapshot()
    normalized_rows = [
        {
            "index": row.get("index"),
            "name": row.get("name"),
            "uuid": row.get("uuid"),
            "normalized_uuid": normalize_gpu_uuid(row.get("uuid")),
        }
        for row in rows
    ]
    current_uuid = None
    uuid_error = None
    getter = getattr(ti._lib.core, "_current_cuda_external_device_uuid", None)
    if backend == "cuda" and getter is not None:
        try:
            program = ti.lang.impl.get_runtime().prog
            current_uuid = normalize_gpu_uuid(bytes(getter(program)))
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            uuid_error = repr(error)
    matches = [
        row for row in normalized_rows
        if current_uuid is not None and row["normalized_uuid"] == current_uuid
    ]
    forced_zero = (
        os.environ.get("TI_VISIBLE_DEVICE") == "0"
        and (backend != "cuda" or os.environ.get("CUDA_VISIBLE_DEVICES") == "0")
    )
    if current_uuid is not None:
        verified = len(matches) == 1
        verification = "runtime_uuid_matches_nvidia_smi"
    else:
        verified = len(normalized_rows) == 1 and forced_zero
        verification = "single_gpu_with_explicit_device_zero_binding"
    return {
        "backend": backend,
        "required": True,
        "binding_verified": verified,
        "verification": verification,
        "runtime_uuid": current_uuid,
        "runtime_uuid_error": uuid_error,
        "visible_device_environment": {
            "TI_VISIBLE_DEVICE": os.environ.get("TI_VISIBLE_DEVICE"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "nvidia_smi_devices": normalized_rows,
        "matching_devices": matches,
    }


def runtime_memory_observation(ti: Any) -> dict[str, Any]:
    """Returns optional Forge runtime and allocator snapshots without fallback."""
    runtime_memory = None
    runtime_error = None
    pool_stats = None
    pool_error = None
    try:
        program = ti.lang.impl.get_runtime().prog
        snapshot = getattr(program, "_runtime_statistics_snapshot", None)
        if snapshot is not None:
            raw = dict(snapshot())
            runtime_memory = {
                "schema_version": raw.get("schema_version"),
                "backend": raw.get("backend"),
                "program_domain": raw.get("program_domain"),
                "memory": dict(raw.get("memory", {})),
            }
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        runtime_error = repr(error)
    host_getter = getattr(ti._lib.core, "get_host_memory_pool_stats", None)
    device_getter = getattr(ti._lib.core, "get_device_memory_pool_stats", None)
    if host_getter is not None and device_getter is not None:
        try:
            pool_stats = {
                "host": dict(host_getter()),
                "device": dict(device_getter()),
            }
        except (RuntimeError, TypeError, ValueError) as error:
            pool_error = repr(error)
    return {
        "available": runtime_memory is not None and pool_stats is not None,
        "runtime": runtime_memory,
        "pools": pool_stats,
        "runtime_error": runtime_error,
        "pool_error": pool_error,
    }


def gpu_compute_processes() -> list[dict[str, str]]:
    fields = ["pid", "process_name", "used_gpu_memory"]
    output = command_output([
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ])
    if not output:
        return []
    rows: list[dict[str, str]] = []
    for line in output.splitlines():
        values = [value.strip() for value in line.split(",", 2)]
        if len(values) == len(fields):
            rows.append(dict(zip(fields, values)))
    return rows


def gpu_conflicting_processes(rows: Sequence[dict[str, str]],
                              ignored_pids: Sequence[int] = ()) -> list[dict[str,
                                                                            str]]:
    """Filters WDDM's broad compute-app list down to likely compute workloads."""
    ignored = set(ignored_pids)
    compute_names = (
        "python.exe",
        "pythonw.exe",
        "cuda",
        "blender.exe",
        "octane",
        "render",
    )
    conflicts: list[dict[str, str]] = []
    for row in rows:
        try:
            if int(row["pid"]) in ignored:
                continue
        except (KeyError, ValueError):
            pass
        name = row.get("process_name", "").lower()
        memory = row.get("used_gpu_memory", "").lower()
        numeric_memory = memory.replace("mib", "").strip()
        has_accounted_memory = False
        try:
            has_accounted_memory = float(numeric_memory) > 0.0
        except ValueError:
            pass
        if has_accounted_memory or any(token in name for token in compute_names):
            conflicts.append(row)
    return conflicts


def process_gpu_memory_mib(pid: int) -> float | None:
    for row in gpu_compute_processes():
        try:
            if int(row["pid"]) == pid:
                value = row["used_gpu_memory"].replace("MiB", "").strip()
                return float(value)
        except (KeyError, ValueError):
            continue
    if os.name == "nt":
        script = (
            f"$pidToFind = {int(pid)}; "
            "$pattern = 'pid_' + $pidToFind + '_*'; "
            "$sum = 0; "
            "try { "
            "(Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage')."
            "CounterSamples | Where-Object { $_.InstanceName -like $pattern } | "
            "ForEach-Object { $sum += $_.CookedValue }; "
            "[Console]::WriteLine([math]::Round($sum / 1MB, 3)) "
            "} catch { [Console]::WriteLine(-1) }")
        try:
            completed = subprocess.run(
                ["powershell", "-NoProfile", "-Command", script],
                check=True,
                capture_output=True,
                text=True,
                timeout=3,
            )
            value = float(completed.stdout.strip().splitlines()[-1])
            return None if value < 0.0 else value
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            return None
    return None


def working_set_bytes() -> int | None:
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCountersEx(ctypes.Structure):
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
                    ("PrivateUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCountersEx()
            counters.cb = ctypes.sizeof(counters)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.argtypes = []
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCountersEx),
                wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            handle = kernel32.GetCurrentProcess()
            if not psapi.GetProcessMemoryInfo(
                    handle, ctypes.byref(counters), counters.cb):
                return None
            return int(counters.WorkingSetSize)
        except (AttributeError, OSError, ValueError):
            return None
    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value * (1024 if sys.platform != "darwin" else 1))
    except (ImportError, ValueError):
        return None


def host_metadata() -> dict[str, Any]:
    metadata = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "python": sys.version,
        "python_executable": sys.executable,
    }
    if os.name == "nt":
        script = (
            "$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1; "
            "$cs = Get-CimInstance Win32_ComputerSystem; "
            "$os = Get-CimInstance Win32_OperatingSystem; "
            "[pscustomobject]@{cpu_name=$cpu.Name; physical_cores="
            "$cpu.NumberOfCores; logical_processors=$cpu.NumberOfLogicalProcessors; "
            "memory_bytes=$cs.TotalPhysicalMemory; os_caption=$os.Caption; "
            "os_build=$os.BuildNumber} | ConvertTo-Json -Compress")
        try:
            completed = subprocess.run(
                ["powershell", "-NoProfile", "-Command", script],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            metadata["windows_hardware"] = json.loads(completed.stdout)
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
            metadata["windows_hardware"] = None
        hardware = metadata.get("windows_hardware") or {}
        if not hardware.get("cpu_name"):
            try:
                import winreg

                with winreg.OpenKey(
                        winreg.HKEY_LOCAL_MACHINE,
                        r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                    hardware["cpu_name"] = winreg.QueryValueEx(
                        key, "ProcessorNameString")[0].strip()
            except (OSError, ImportError):
                hardware["cpu_name"] = None
        if not hardware.get("memory_bytes"):
            try:
                import ctypes

                class MemoryStatusEx(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                memory = MemoryStatusEx()
                memory.dwLength = ctypes.sizeof(memory)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(
                        ctypes.byref(memory)):
                    hardware["memory_bytes"] = int(memory.ullTotalPhys)
            except (AttributeError, OSError, ValueError):
                hardware["memory_bytes"] = None
        hardware["logical_processors"] = hardware.get(
            "logical_processors") or os.cpu_count()
        hardware["os_build"] = hardware.get("os_build") or platform.version()
        metadata["windows_hardware"] = hardware
        metadata["power_scheme"] = command_output(["powercfg", "/getactivescheme"])
    return metadata


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, values: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for value in values:
            stream.write(json.dumps(value, sort_keys=True, ensure_ascii=False))
            stream.write("\n")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
