import json

import numpy as np

import hardware_acceleration_qualification as qualification
from taichi_forge.hardware import load_provider_admission_evidence


def test_physics_workload_registry_and_cpu_oracles_are_stable():
    assert "cuda-fft-poisson" in qualification.CASES
    assert "cuda-cudss-refactor-solve" in qualification.CASES
    assert "cuda-cudss-tet-fem" in qualification.CASES
    assert "cuda-spmv-krylov" in qualification.CASES
    assert "vulkan-offscreen-simulation" in qualification.CASES

    length = 32
    coordinates = 2.0 * np.pi * np.arange(length) / length
    rhs = np.stack((np.sin(coordinates), np.cos(3.0 * coordinates))).astype(np.float32)
    solution = qualification._periodic_poisson_reference(rhs)
    residual = qualification._periodic_poisson_residual(solution, rhs)
    residual_tolerance = qualification._periodic_poisson_residual_tolerance(
        solution, rhs
    )
    inverse = qualification._periodic_poisson_inverse_eigenvalues(length)
    assert inverse[0] == 0.0
    assert inverse[1] == inverse[-1]
    assert residual[1] < min(5e-6, residual_tolerance)

    rows, columns, values = qualification._implicit_grid_csr(3, 0.2)
    dense = np.zeros((9, 9), dtype=np.float64)
    for row in range(9):
        dense[row, columns[rows[row] : rows[row + 1]]] = values[
            rows[row] : rows[row + 1]
        ]
    np.testing.assert_allclose(dense, dense.T)
    assert np.min(np.linalg.eigvalsh(dense)) > 0.0

    wide_rows, wide_columns, wide_values = qualification._implicit_grid_csr(
        5, 0.2, stencil_radius=2
    )
    wide_dense = np.zeros((25, 25), dtype=np.float64)
    for row in range(25):
        wide_dense[row, wide_columns[wide_rows[row] : wide_rows[row + 1]]] = (
            wide_values[wide_rows[row] : wide_rows[row + 1]]
        )
    np.testing.assert_allclose(wide_dense, wide_dense.T)
    assert np.count_nonzero(wide_dense[12]) == 25
    assert np.min(np.linalg.eigvalsh(wide_dense)) > 0.0

    coordinates, tetrahedra, rows, columns, low_values = (
        qualification._irregular_tet_fem_csr(3, 2.0)
    )
    _, high_tetrahedra, high_rows, high_columns, high_values = (
        qualification._irregular_tet_fem_csr(3, 5.0)
    )
    assert coordinates.shape == (27, 3)
    assert tetrahedra.shape == (48, 4)
    assert np.array_equal(tetrahedra, high_tetrahedra)
    assert np.array_equal(rows, high_rows)
    assert np.array_equal(columns, high_columns)
    assert not np.array_equal(low_values, high_values)
    regular_center = np.full(3, 0.5, dtype=np.float32)
    assert not np.allclose(coordinates[13], regular_center)
    dense = np.zeros((81, 81), dtype=np.float64)
    for row in range(81):
        dense[row, columns[rows[row] : rows[row + 1]]] = low_values[
            rows[row] : rows[row + 1]
        ]
    np.testing.assert_allclose(dense, dense.T, atol=2e-6)
    assert np.min(np.linalg.eigvalsh(dense)) > 0.0


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


def test_aggregate_separates_stable_speedup_from_formal_claim_evidence():
    workers = (
        _worker("ab", [1.00, 1.01, 0.99], [2.00, 2.01, 1.99], [2.0] * 3),
        _worker("ba", [1.01, 1.00, 0.99], [2.01, 2.00, 1.99], [2.0] * 3),
    )
    workers[0]["memory"] = {"runtime_after_close": {"inflight_resources": 0}}
    workers[1]["provider_statistics"] = {"successes": 3, "failures": 0}

    report = qualification._aggregate("synthetic", workers, 0.10, 0.10)

    assert report["correctness_and_route_qualified"]
    assert report["noise_status"] == "stable"
    assert not report["performance_claim_eligible"]
    assert report["performance_state"] == "stable_positive"
    assert not report["performance_evidence"]["qualified"]
    assert report["performance_evidence"]["reasons"] == (
        "insufficient_fresh_process_coverage",
        "insufficient_timing_samples",
        "undersized_timing_blocks",
    )
    assert report["performance_scope"]["revision"]["forge_commit"] == ("test-revision")
    assert report["paired_speedup"]["p05"] == 2.0
    assert "p05_ms" not in report["paired_speedup"]
    assert len(report["worker_calibration"]) == 2
    assert report["worker_calibration"][0]["variants"]["hardware"]["satisfied"]
    assert report["memory"] == [workers[0]["memory"]]
    assert report["provider_statistics"] == [workers[1]["provider_statistics"]]


def test_aggregate_allows_claim_after_formal_fresh_process_coverage():
    workers = tuple(
        _worker(
            "ab" if index < 4 else "ba",
            [1.0] * 5,
            [2.0] * 5,
            [2.0] * 5,
            block_ms=100.0,
            pid=index + 1,
        )
        for index in range(8)
    )

    report = qualification._aggregate("synthetic", workers, 0.10, 0.10)

    assert report["performance_state"] == "stable_positive"
    assert report["performance_evidence"]["qualified"]
    assert report["performance_evidence"]["reasons"] == ()
    assert report["performance_claim_eligible"]


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


def test_runtime_bitcode_provenance_records_only_loaded_artifact_kinds(tmp_path):
    (tmp_path / "runtime_cuda.bc").write_bytes(b"cuda")
    (tmp_path / "runtime_x64.bc").write_bytes(b"x64")
    (tmp_path / "slim_libdevice.10.bc").write_bytes(b"libdevice")
    (tmp_path / "unrelated.bc").write_bytes(b"unrelated")

    provenance = qualification._runtime_bitcode_provenance(tmp_path)

    assert [
        artifact["path"].replace("\\", "/").rsplit("/", 1)[-1]
        for artifact in provenance
    ] == [
        "runtime_cuda.bc",
        "runtime_x64.bc",
        "slim_libdevice.10.bc",
    ]


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
            "native_runtime_build_identity": "b" * 64,
            "native_runtime_artifacts": {
                "python_extension_sha256": "c" * 64,
                "native_runtime_binary_sha256": "d" * 64,
                "runtime_bitcode_bundle_sha256": "e" * 64,
            },
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
