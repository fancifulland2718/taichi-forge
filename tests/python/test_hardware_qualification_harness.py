import numpy as np

import hardware_acceleration_qualification as qualification


def _worker(order, hardware, baseline, paired):
    return {
        "status": "passed",
        "order": order,
        "pid": 1,
        "timestamp_ns": 1,
        "backend": "cuda",
        "cuda_compute_capability": 80,
        "forge_version": (0, 6, 3),
        "python": "3.10",
        "platform": "test",
        "timing": {
            "samples_ms": {
                "hardware": hardware,
                "baseline": baseline,
            },
            "paired_speedups": paired,
        },
        "workload": {"name": "synthetic"},
        "correctness": {"max_abs": 0.0},
        "route": {"selection": "eligible"},
    }


def test_aggregate_accepts_only_stable_conservative_speedup():
    workers = (
        _worker("ab", [1.00, 1.01, 0.99], [2.00, 2.01, 1.99], [2.0] * 3),
        _worker("ba", [1.01, 1.00, 0.99], [2.01, 2.00, 1.99], [2.0] * 3),
    )

    report = qualification._aggregate("synthetic", workers, 0.10, 0.10)

    assert report["correctness_and_route_qualified"]
    assert report["noise_status"] == "stable"
    assert report["performance_claim_eligible"]
    assert report["paired_speedup"]["p05"] == 2.0
    assert "p05_ms" not in report["paired_speedup"]


def test_aggregate_rejects_cross_order_drift_despite_positive_speedup():
    workers = (
        _worker("ab", [1.0] * 5, [2.0] * 5, [2.0] * 5),
        _worker("ba", [2.0] * 5, [4.0] * 5, [2.0] * 5),
    )

    report = qualification._aggregate("synthetic", workers, 0.10, 0.10)

    assert report["correctness_and_route_qualified"]
    assert report["noise_status"] == "unstable"
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
    assert not report["performance_claim_eligible"]

    error = qualification._aggregate(
        "synthetic",
        ({"status": "error", "error": "provider failed"},),
        0.10,
        0.10,
    )
    assert error["status"] == "error"
    assert not error["performance_claim_eligible"]


def test_complex_error_checks_both_components():
    actual = np.array([1.0 + 3.0j], dtype=np.complex64)
    expected = np.array([1.0 + 1.0j], dtype=np.complex64)

    absolute, relative = qualification._error(actual, expected)

    assert absolute == 2.0
    assert relative > 1.0
