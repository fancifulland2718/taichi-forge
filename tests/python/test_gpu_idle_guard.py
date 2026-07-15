import importlib.util
import subprocess
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_GUARD_PATH = _REPO_ROOT / "benchmarks" / "gpu_idle_guard.py"
_SPEC = importlib.util.spec_from_file_location("gpu_idle_guard", _GUARD_PATH)
gpu_idle_guard = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gpu_idle_guard)


class _Completed:
    def __init__(self, returncode=0, stdout=""):
        self.returncode = returncode
        self.stdout = stdout


def test_parse_nvidia_compute_apps_ignores_self_and_non_rows():
    output = "321, python.exe\nnot-a-pid, N/A\n654, renderer.exe\n"

    assert gpu_idle_guard.parse_nvidia_compute_apps(output, own_pid=321) == (
        {"pid": 654, "name": "renderer.exe"},
    )


def test_probe_reports_other_python_compute_processes():
    def runner(*_args, **_kwargs):
        return _Completed(stdout="123, python.exe\n456, game.exe\n")

    evidence = gpu_idle_guard.probe_nvidia_python_gpu_idle(
        own_pid=999,
        runner=runner,
    )

    assert evidence.verified
    assert not evidence.python_idle
    assert evidence.other_python_processes == (
        {"pid": 123, "name": "python.exe"},
    )
    assert len(evidence.other_compute_processes) == 2


def test_probe_failure_cannot_be_used_as_idle_evidence():
    def runner(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("nvidia-smi", 5)

    evidence = gpu_idle_guard.probe_nvidia_python_gpu_idle(runner=runner)

    assert not evidence.verified
    assert not evidence.python_idle
    with pytest.raises(RuntimeError, match="automatic idle check"):
        gpu_idle_guard.require_nvidia_python_gpu_idle(runner=runner)


def test_require_idle_accepts_non_python_compute_processes():
    def runner(*_args, **_kwargs):
        return _Completed(stdout="456, desktop-renderer.exe\n")

    evidence = gpu_idle_guard.require_nvidia_python_gpu_idle(runner=runner)

    assert evidence.verified
    assert evidence.python_idle
    assert evidence.other_python_processes == ()


def test_probe_nvidia_environment_records_gpu_and_driver():
    def runner(*_args, **_kwargs):
        return _Completed(stdout="Test GPU, 555.42\n")

    environment = gpu_idle_guard.probe_nvidia_environment(runner=runner)

    assert environment == {
        "verified": True,
        "tool": "nvidia-smi",
        "reason": "verified",
        "gpu": "Test GPU",
        "driver": "555.42",
    }


def test_prepare_measurement_marks_verified_gpu_performance(monkeypatch):
    evidence = gpu_idle_guard.GpuIdleEvidence(
        verified=True,
        python_idle=True,
        tool="nvidia-smi",
        reason="verified",
        other_python_processes=(),
        other_compute_processes=(),
    )
    monkeypatch.setattr(
        gpu_idle_guard,
        "require_nvidia_python_gpu_idle",
        lambda: evidence,
    )
    monkeypatch.setattr(
        gpu_idle_guard,
        "probe_nvidia_environment",
        lambda: {
            "verified": True,
            "tool": "nvidia-smi",
            "reason": "verified",
            "gpu": "Test GPU",
            "driver": "555.42",
        },
    )

    context = gpu_idle_guard.prepare_performance_measurement(
        "cuda",
        requested=True,
    )

    assert context["measurement"]["performance_valid"] is True
    assert context["measurement"]["gpu_idle"]["python_idle"] is True
    assert context["environment"]["gpu"] == "Test GPU"


def test_prepare_measurement_keeps_default_timings_diagnostic(monkeypatch):
    evidence = gpu_idle_guard.GpuIdleEvidence(
        verified=True,
        python_idle=True,
        tool="nvidia-smi",
        reason="verified",
        other_python_processes=(),
        other_compute_processes=(),
    )
    monkeypatch.setattr(
        gpu_idle_guard,
        "probe_nvidia_python_gpu_idle",
        lambda: evidence,
    )
    monkeypatch.setattr(
        gpu_idle_guard,
        "probe_nvidia_environment",
        lambda: {
            "verified": True,
            "tool": "nvidia-smi",
            "reason": "verified",
            "gpu": "Test GPU",
            "driver": "555.42",
        },
    )

    context = gpu_idle_guard.prepare_performance_measurement(
        "vulkan",
        requested=False,
    )

    assert context["measurement"]["performance_requested"] is False
    assert context["measurement"]["performance_valid"] is False
    assert "not a performance baseline" in context["measurement"]["note"]


def test_finalize_measurement_rejects_skipped_or_incorrect_rows():
    context = {
        "measurement": {
            "performance_requested": True,
            "performance_valid": True,
            "gpu_idle": {"verified": True, "python_idle": True},
            "note": "Eligible for performance comparison.",
        },
        "environment": {"gpu": "Test GPU"},
    }

    skipped = gpu_idle_guard.finalize_performance_measurement(
        context,
        correct=False,
        skipped=True,
        reason="provider unavailable",
    )
    incorrect = gpu_idle_guard.finalize_performance_measurement(
        context,
        correct=False,
    )

    assert skipped["measurement"]["performance_valid"] is False
    assert "provider unavailable" in skipped["measurement"]["note"]
    assert incorrect["measurement"]["performance_valid"] is False
    assert context["measurement"]["performance_valid"] is True
