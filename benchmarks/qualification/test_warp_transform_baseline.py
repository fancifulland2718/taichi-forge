from __future__ import annotations

import argparse

from benchmarks.qualification.warp_transform_baseline import (
    qualification_policy_errors,
    select_calibrated_batch,
)


def _args(**overrides):
    values = {
        "intent": "qualification",
        "samples": 30,
        "warmups": 5,
        "target_sample_ms": 100.0,
        "stability_replays": 1_000,
        "cpu_affinity": "auto",
        "max_cpu_util": 20.0,
        "max_gpu_util": 15.0,
        "max_gpu_temp": 65.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_warp_qualification_policy_accepts_exact_gate() -> None:
    assert qualification_policy_errors(_args()) == []


def test_warp_qualification_policy_rejects_relaxed_or_short_run() -> None:
    errors = qualification_policy_errors(_args(
        samples=29, stability_replays=999, max_gpu_util=16.0))
    assert any("samples=29" in error for error in errors)
    assert any("stability_replays=999" in error for error in errors)
    assert any("GPU-utilization" in error for error in errors)


def test_warp_diagnostic_does_not_apply_qualification_minimums() -> None:
    assert qualification_policy_errors(_args(
        intent="diagnostic", samples=1, stability_replays=1,
        cpu_affinity="none")) == []


def test_calibrated_batch_never_shrinks_and_obeys_cap() -> None:
    assert select_calibrated_batch(25.0, 8, 100.0) == 32
    assert select_calibrated_batch(0.0, 8, 100.0) == 16
    assert select_calibrated_batch(0.01, 32_768, 100.0) == 65_536
