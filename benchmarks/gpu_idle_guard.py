"""Admission checks for trustworthy GPU performance measurements.

The module has no Taichi dependency so it can run before a benchmark creates
its own GPU context. GUI/display processes are recorded but only other Python
compute processes reject a measurement.
"""

import csv
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import PurePath


@dataclass(frozen=True)
class GpuIdleEvidence:
    verified: bool
    python_idle: bool
    tool: str
    reason: str
    other_python_processes: tuple
    other_compute_processes: tuple

    def to_dict(self):
        return asdict(self)


def _process_record(pid, name):
    return {"pid": int(pid), "name": str(name).strip()}


def _is_python_process(name):
    executable = PurePath(str(name).strip()).name.lower()
    return executable.startswith("python") or executable.startswith("py.exe")


def parse_nvidia_compute_apps(output, *, own_pid=None):
    processes = []
    for row in csv.reader(str(output).splitlines()):
        if not row:
            continue
        raw_pid = row[0].strip()
        if not raw_pid.isdigit():
            continue
        pid = int(raw_pid)
        if own_pid is not None and pid == int(own_pid):
            continue
        name = ",".join(row[1:]).strip() if len(row) > 1 else ""
        processes.append(_process_record(pid, name))
    return tuple(processes)


def probe_nvidia_python_gpu_idle(*, own_pid=None, runner=subprocess.run):
    if own_pid is None:
        own_pid = os.getpid()
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = runner(
            command,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return GpuIdleEvidence(
            verified=False,
            python_idle=False,
            tool="nvidia-smi",
            reason=f"probe_failed:{type(exc).__name__}",
            other_python_processes=(),
            other_compute_processes=(),
        )

    if completed.returncode != 0:
        return GpuIdleEvidence(
            verified=False,
            python_idle=False,
            tool="nvidia-smi",
            reason=f"probe_exit_{completed.returncode}",
            other_python_processes=(),
            other_compute_processes=(),
        )

    processes = parse_nvidia_compute_apps(completed.stdout, own_pid=own_pid)
    python_processes = tuple(
        process for process in processes if _is_python_process(process["name"])
    )
    return GpuIdleEvidence(
        verified=True,
        python_idle=not python_processes,
        tool="nvidia-smi",
        reason="verified",
        other_python_processes=python_processes,
        other_compute_processes=processes,
    )


def probe_nvidia_environment(*, runner=subprocess.run):
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = runner(
            command,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "verified": False,
            "tool": "nvidia-smi",
            "reason": f"probe_failed:{type(exc).__name__}",
            "gpu": None,
            "driver": None,
        }
    rows = tuple(csv.reader(completed.stdout.splitlines()))
    if completed.returncode != 0 or not rows or len(rows[0]) < 2:
        return {
            "verified": False,
            "tool": "nvidia-smi",
            "reason": f"probe_exit_{completed.returncode}",
            "gpu": None,
            "driver": None,
        }
    return {
        "verified": True,
        "tool": "nvidia-smi",
        "reason": "verified",
        "gpu": rows[0][0].strip(),
        "driver": rows[0][1].strip(),
    }


def require_nvidia_python_gpu_idle(*, own_pid=None, runner=subprocess.run):
    evidence = probe_nvidia_python_gpu_idle(own_pid=own_pid, runner=runner)
    if not evidence.verified:
        raise RuntimeError(
            "GPU performance measurement requires an automatic idle check; "
            f"{evidence.tool} could not verify the GPU ({evidence.reason})."
        )
    if not evidence.python_idle:
        details = ", ".join(
            f"{item['pid']}:{item['name']}"
            for item in evidence.other_python_processes
        )
        raise RuntimeError(
            "GPU performance measurement refused because other Python GPU "
            f"compute processes are active: {details}"
        )
    return evidence


def prepare_performance_measurement(arch, *, requested):
    gpu_backend = arch in ("cuda", "vulkan")
    gpu_environment = (
        probe_nvidia_environment()
        if gpu_backend
        else {
            "verified": False,
            "tool": None,
            "reason": "cpu_backend",
            "gpu": None,
            "driver": None,
        }
    )
    gpu_idle = None
    if gpu_backend:
        gpu_idle = (
            require_nvidia_python_gpu_idle()
            if requested
            else probe_nvidia_python_gpu_idle()
        )
    performance_valid = bool(
        requested
        and (
            not gpu_backend
            or (gpu_idle.verified and gpu_idle.python_idle)
        )
    )
    return {
        "measurement": {
            "performance_requested": bool(requested),
            "performance_valid": performance_valid,
            "gpu_idle": None if gpu_idle is None else gpu_idle.to_dict(),
            "note": (
                "Eligible for performance comparison."
                if performance_valid
                else "Functional/diagnostic timing only; not a performance baseline."
            ),
        },
        "environment": {
            "os": platform.platform(),
            "python": sys.version,
            "python_executable": sys.executable,
            "gpu": gpu_environment.get("gpu"),
            "driver": gpu_environment.get("driver"),
            "gpu_probe": gpu_environment,
        },
    }


def finalize_performance_measurement(
    context,
    *,
    correct=None,
    skipped=False,
    reason=None,
):
    finalized = {
        "measurement": dict(context["measurement"]),
        "environment": dict(context["environment"]),
    }
    if skipped or correct is False:
        finalized["measurement"]["performance_valid"] = False
        if reason is None:
            reason = "case skipped" if skipped else "correctness check failed"
        finalized["measurement"]["note"] = (
            f"Functional/diagnostic result only: {reason}."
        )
    return finalized


__all__ = [
    "GpuIdleEvidence",
    "finalize_performance_measurement",
    "parse_nvidia_compute_apps",
    "probe_nvidia_environment",
    "probe_nvidia_python_gpu_idle",
    "prepare_performance_measurement",
    "require_nvidia_python_gpu_idle",
]
