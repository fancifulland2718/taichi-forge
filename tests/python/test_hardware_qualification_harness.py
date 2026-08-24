import json

import numpy as np

import hardware_acceleration_qualification as qualification
from taichi_forge.hardware import load_provider_admission_evidence


def _worker(
    order,
    hardware,
    baseline,
    paired,
    *,
    block_satisfied=True,
    block_ms=10.0,
    admission_scope=None,
    pid=1,
):
    worker = {
        "status": "passed",
        "order": order,
        "pid": pid,
        "timestamp_ns": pid,
        "backend": "cuda",
        "cuda_compute_capability": 80,
        "forge_version": (0, 6, 3),
        "forge_commit": "test-revision",
        "python": "3.10",
        "platform": "test",
        "timing": {
            "calibration": {
                variant: {
                    "requested_repetitions": 1,
                    "effective_repetitions": 10,
                    "observed_block_ms": block_ms,
                    "minimum_block_ms": block_ms,
                    "satisfied": block_satisfied,
                }
                for variant in ("hardware", "baseline")
            },
            "samples_ms": {
                "hardware": hardware,
                "baseline": baseline,
            },
            "paired_speedups": paired,
            "cold_ms": {"hardware": 1.1, "baseline": 2.1},
        },
        "workload": {"name": "synthetic"},
        "correctness": {"max_abs": 0.0},
        "route": {"selection": "eligible"},
    }
    if admission_scope is not None:
        worker["admission_scope"] = admission_scope
        worker["forge_version"] = "0.6.3"
        worker["cuda_device_uuid"] = admission_scope["device_scope"]["cuda_device_uuid"]
    return worker


def test_aggregate_accepts_only_stable_conservative_speedup():
    workers = (
        _worker("ab", [1.00, 1.01, 0.99], [2.00, 2.01, 1.99], [2.0] * 3),
        _worker("ba", [1.01, 1.00, 0.99], [2.01, 2.00, 1.99], [2.0] * 3),
    )

    report = qualification._aggregate("synthetic", workers, 0.10, 0.10)

    assert report["correctness_and_route_qualified"]
    assert report["noise_status"] == "stable"
    assert report["performance_claim_eligible"]
    assert report["performance_state"] == "stable_positive"
    assert report["performance_scope"]["revision"]["forge_commit"] == ("test-revision")
    assert report["paired_speedup"]["p05"] == 2.0
    assert "p05_ms" not in report["paired_speedup"]
    assert len(report["worker_calibration"]) == 2
    assert report["worker_calibration"][0]["variants"]["hardware"]["satisfied"]


def test_aggregate_rejects_cross_order_drift_despite_positive_speedup():
    workers = (
        _worker("ab", [1.0] * 5, [2.0] * 5, [2.0] * 5),
        _worker("ba", [2.0] * 5, [4.0] * 5, [2.0] * 5),
    )

    report = qualification._aggregate("synthetic", workers, 0.10, 0.10)

    assert report["correctness_and_route_qualified"]
    assert report["noise_status"] == "unstable"
    assert report["performance_state"] == "unstable"
    assert not report["performance_claim_eligible"]
    assert not report["variants"]["hardware"]["stable"]
    assert not report["variants"]["baseline"]["stable"]


def test_aggregate_rejects_no_speedup_and_worker_errors():
    workers = (
        _worker("ab", [2.0] * 5, [1.0] * 5, [0.5] * 5),
        _worker("ba", [2.0] * 5, [1.0] * 5, [0.5] * 5),
    )
    report = qualification._aggregate("synthetic", workers, 0.10, 0.10)
    assert report["noise_status"] == "stable"
    assert report["performance_state"] == "stable_negative"
    assert not report["performance_claim_eligible"]

    error = qualification._aggregate(
        "synthetic",
        ({"status": "error", "error": "provider failed"},),
        0.10,
        0.10,
    )
    assert error["status"] == "error"
    assert not error["performance_claim_eligible"]
    assert error["performance_state"] == "not_measured"
    assert error["performance_scope"] == {}


def test_aggregate_rejects_undersized_timing_blocks():
    workers = (
        _worker(
            "ab",
            [1.0] * 5,
            [2.0] * 5,
            [2.0] * 5,
            block_satisfied=False,
        ),
        _worker("ba", [1.0] * 5, [2.0] * 5, [2.0] * 5),
    )

    report = qualification._aggregate("synthetic", workers, 0.10, 0.10)

    assert report["noise_status"] == "stable"
    assert report["performance_state"] == "unstable"
    assert not report["minimum_block_qualified"]
    assert not report["performance_claim_eligible"]


def test_balanced_worker_schedule_cancels_first_order_bias():
    assert qualification._balanced_worker_schedule(1) == (("ab", 0), ("ba", 0))
    assert qualification._balanced_worker_schedule(2) == (
        ("ab", 0),
        ("ba", 0),
        ("ba", 1),
        ("ab", 1),
    )


def test_complex_error_checks_both_components():
    actual = np.array([1.0 + 3.0j], dtype=np.complex64)
    expected = np.array([1.0 + 1.0j], dtype=np.complex64)

    absolute, relative = qualification._error(actual, expected)

    assert absolute == 2.0
    assert relative > 1.0


def test_artifact_provenance_records_identity_and_digest(tmp_path):
    artifact = tmp_path / "runtime.bin"
    artifact.write_bytes(b"taichi-forge-runtime")

    provenance = qualification._artifact_provenance(artifact)

    assert provenance["path"] == str(artifact.resolve())
    assert provenance["bytes"] == 20
    assert (
        provenance["sha256"]
        == "3bdb141509c6111dee71c967b1c7e38875c39a5f646009caeb61aa7fc2c5a418"
    )


def test_aggregate_emits_loadable_strict_auto_admission_evidence(tmp_path):
    scope = {
        "operation_id": "linalg.spmv.cusparse",
        "provider_id": "cusparse",
        "baseline_id": "cuda_driver_kernel",
        "backend": "cuda",
        "device_scope": {
            "cuda_device_uuid": "00112233445566778899aabbccddeeff",
            "cuda_compute_capability": 80,
        },
        "provider_scope": {
            "provider_abi": "cusparse-dynamic-symbols-v1",
            "provider_version": {"major": 12, "minor": 4, "patch": 0},
        },
        "workload_scope": {
            "rows": 1024,
            "cols": 1024,
            "nnz": 7168,
            "storage_format": "csr",
            "block_size": None,
            "topology_fingerprint": ("tf-sp-v1:0123456789abcdef0123456789abcdef"),
        },
        "runtime_scope": {
            "forge_version": "0.6.3",
            "forge_commit": "test-revision",
            "python_provider_contract_sha256": "a" * 64,
        },
        "transfer_ns": 0.0,
        "conversion_ns": 0.0,
    }
    workers = tuple(
        _worker(
            "ab" if index < 4 else "ba",
            [1.0] * 5,
            [2.0] * 5,
            [2.0] * 5,
            block_ms=100.0,
            admission_scope=scope,
            pid=index + 1,
        )
        for index in range(8)
    )

    case = qualification._aggregate(
        "cuda-spmv",
        workers,
        0.05,
        0.05,
        auto_admission_expected_reuse=100,
        auto_admission_minimum_margin=0.05,
    )

    assert case["auto_admission"]["eligible"]
    assert case["auto_admission"]["evidence"]["performance"]["fresh_processes"] == 8
    assert (
        case["auto_admission"]["evidence"]["performance"][
            "baseline_first_use_overhead_ns"
        ]
        == 100_000.0
    )
    artifact = tmp_path / "qualification.json"
    artifact.write_text(
        json.dumps(
            {
                "schema": qualification.SCHEMA,
                "cases": [case],
            }
        ),
        encoding="utf-8",
    )
    evidence = load_provider_admission_evidence(artifact, case="cuda-spmv")
    assert evidence.operation_id == "linalg.spmv.cusparse"
