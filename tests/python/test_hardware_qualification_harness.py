import argparse
import json

import numpy as np
import pytest

import hardware_acceleration_qualification as qualification
import hardware_physics_crossover_qualification as physics_crossover
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
    residual_tolerance = qualification._periodic_poisson_residual_tolerance(solution, rhs)
    inverse = qualification._periodic_poisson_inverse_eigenvalues(length)
    assert inverse[0] == 0.0
    assert inverse[1] == inverse[-1]
    assert residual[1] < min(5e-6, residual_tolerance)

    rows, columns, values = qualification._implicit_grid_csr(3, 0.2)
    dense = np.zeros((9, 9), dtype=np.float64)
    for row in range(9):
        dense[row, columns[rows[row] : rows[row + 1]]] = values[rows[row] : rows[row + 1]]
    np.testing.assert_allclose(dense, dense.T)
    assert np.min(np.linalg.eigvalsh(dense)) > 0.0

    wide_rows, wide_columns, wide_values = qualification._implicit_grid_csr(5, 0.2, stencil_radius=2)
    wide_dense = np.zeros((25, 25), dtype=np.float64)
    for row in range(25):
        wide_dense[row, wide_columns[wide_rows[row] : wide_rows[row + 1]]] = wide_values[
            wide_rows[row] : wide_rows[row + 1]
        ]
    np.testing.assert_allclose(wide_dense, wide_dense.T)
    assert np.count_nonzero(wide_dense[12]) == 25
    assert np.min(np.linalg.eigvalsh(wide_dense)) > 0.0

    coordinates, tetrahedra, rows, columns, low_values = qualification._irregular_tet_fem_csr(3, 2.0)
    _, high_tetrahedra, high_rows, high_columns, high_values = qualification._irregular_tet_fem_csr(3, 5.0)
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
        dense[row, columns[rows[row] : rows[row + 1]]] = low_values[rows[row] : rows[row + 1]]
    np.testing.assert_allclose(dense, dense.T, atol=2e-6)
    assert np.min(np.linalg.eigvalsh(dense)) > 0.0


def test_physics_crossover_points_are_ordered_and_dimensioned():
    parser = physics_crossover._positive_ints
    assert parser("1,4,16") == (1, 4, 16)
    with pytest.raises(argparse.ArgumentTypeError, match="strictly increasing"):
        parser("4,1")

    args = argparse.Namespace(
        poisson_length=4096,
        poisson_batches=(1, 4, 16),
        krylov_grids=(64, 128, 256),
        krylov_iterations=48,
        krylov_stencil_radius=1,
        fem_grids=(4, 6, 8),
    )
    points = physics_crossover._family_points(args)
    assert [point["work_units"] for point in points["cuda-fft-poisson-batch"]] == [4096, 16384, 65536]
    assert [point["work_units"] for point in points["cuda-spmv-krylov-grid"]] == [4096, 16384, 65536]
    assert [point["work_units"] for point in points["cuda-cudss-tet-fem-grid"]] == [192, 648, 1536]


def test_physics_crossover_summary_fails_closed_and_reports_reversal():
    def point(label, work_units, state, *, evidence=True):
        return {
            "status": "passed",
            "correctness_and_route_qualified": True,
            "performance_evidence": {"qualified": evidence},
            "performance_state": state,
            "paired_speedup": {"p05": 1.05 if state == "stable_positive" else 0.95},
            "point": {"label": label, "work_units": work_units, "parameters": {}},
        }

    summary = physics_crossover._crossover_summary(
        (
            point("small", 1, "stable_negative"),
            point("medium", 2, "stable_positive"),
            point("large", 3, "stable_positive", evidence=False),
        )
    )
    assert summary["status"] == "crossover_observed"
    assert summary["first_qualified_positive_point"]["label"] == "medium"
    assert summary["qualified_positive_points"] == ("medium",)
    assert summary["reversals_after_first_positive"] == ("large",)
    assert not summary["monotonic_after_first_positive"]
    assert not summary["all_points_performance_qualified"]


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
    assert report["retention_eligible"]
    assert report["retention_qualification"]["qualified"]
    assert report["performance_claim_eligible"]


def test_aggregate_retains_robust_paired_gain_despite_absolute_process_noise():
    scales = (0.8, 1.2, 0.9, 1.1) * 2
    workers = tuple(
        _worker(
            "ab" if index < 4 else "ba",
            [scale] * 5,
            [scale * 1.3] * 5,
            [1.3] * 5,
            block_ms=100.0,
            pid=index + 1,
        )
        for index, scale in enumerate(scales)
    )

    report = qualification._aggregate("synthetic", workers, 0.05, 0.05)

    assert report["performance_state"] == "unstable"
    assert report["variants"]["hardware"]["cv"] > 0.05
    assert report["variants"]["baseline"]["cv"] > 0.05
    assert report["paired_speedup"]["p05"] == pytest.approx(1.3)
    assert report["paired_speedup"]["cv"] == pytest.approx(0.0)
    assert report["retention_eligible"]
    assert report["retention_qualification"]["absolute_variant_cv_is_diagnostic"]
    assert report["retention_qualification"]["reasons"] == ()
    assert not report["auto_admission"]["eligible"]
    assert not report["performance_claim_eligible"]


def test_aggregate_rejects_noisy_paired_gain_from_retention():
    paired = ([0.9, 1.0, 1.1, 1.3, 1.6],) * 8
    workers = tuple(
        _worker(
            "ab" if index < 4 else "ba",
            [1.0] * 5,
            [1.2] * 5,
            paired[index],
            block_ms=100.0,
            pid=index + 1,
        )
        for index in range(8)
    )

    report = qualification._aggregate("synthetic", workers, 0.10, 0.10)

    assert not report["retention_eligible"]
    assert "paired_margin_gate" in report["retention_qualification"]["reasons"]
    assert "unstable_paired_ratio" in report["retention_qualification"]["reasons"]


def test_build_provenance_qualification_requires_one_matching_worker_revision():
    workers = ({"forge_commit": "source-revision"}, {"forge_commit": "source-revision"})

    exact = qualification._build_provenance_qualification("source-revision", workers)
    abbreviated = qualification._build_provenance_qualification(
        "0955034a10c53d8848af897487abc332f90f0eb3", [{"forge_commit": "0955034a"}]
    )
    mismatch = qualification._build_provenance_qualification("other-revision", workers)
    mixed = qualification._build_provenance_qualification(
        "source-revision", (*workers, {"forge_commit": "other-revision"})
    )
    missing = qualification._build_provenance_qualification("source-revision", (*workers, {}))

    assert exact["qualified"]
    assert exact["reasons"] == ()
    assert abbreviated["qualified"]
    assert not mismatch["qualified"]
    assert mismatch["reasons"] == ("source_worker_revision_mismatch",)
    assert not mixed["qualified"]
    assert "mixed_worker_revisions" in mixed["reasons"]
    assert "source_worker_revision_mismatch" in mixed["reasons"]
    assert not missing["qualified"]
    assert "worker_revision_unavailable" in missing["reasons"]


def test_build_provenance_gate_fails_closed_for_claims_and_admission():
    case = {
        "status": "passed",
        "worker_provenance": [{"forge_commit": "built-revision"}],
        "performance_evidence": {"qualified": True, "reasons": ()},
        "performance_claim_eligible": True,
        "retention_eligible": True,
        "retention_qualification": {"qualified": True, "reasons": ()},
        "auto_admission": {"eligible": True, "evidence": {"unsafe": True}},
        "replay_proof_gate": {
            "physics_roi_gate_passed": True,
            "performance_gate_passed": True,
            "retention_gate_passed": True,
        },
    }

    overall = qualification._apply_build_provenance_gate([case], "source-revision")

    assert not overall["qualified"]
    assert not case["build_provenance"]["qualified"]
    assert not case["performance_evidence"]["qualified"]
    assert case["performance_evidence"]["reasons"] == ("build_provenance_unqualified",)
    assert not case["performance_claim_eligible"]
    assert not case["retention_eligible"]
    assert not case["retention_qualification"]["qualified"]
    assert case["retention_qualification"]["reasons"] == ("build_provenance_unqualified",)
    assert case["auto_admission"] == {"eligible": False, "reason": "build_provenance_unqualified"}
    assert case["replay_proof_gate"]["gate_reason"] == "build_provenance_unqualified"
    assert not case["replay_proof_gate"]["performance_gate_passed"]
    assert not case["replay_proof_gate"]["retention_gate_passed"]


def test_build_provenance_gate_preserves_exact_claims():
    case = {
        "status": "passed",
        "worker_provenance": [{"forge_commit": "source-revision"}],
        "performance_evidence": {"qualified": True, "reasons": ()},
        "performance_claim_eligible": True,
        "retention_eligible": True,
        "retention_qualification": {"qualified": True, "reasons": ()},
        "auto_admission": {"eligible": True, "evidence": {"safe": True}},
    }

    overall = qualification._apply_build_provenance_gate([case], "source-revision")

    assert overall["qualified"]
    assert case["build_provenance"]["qualified"]
    assert case["performance_evidence"]["qualified"]
    assert case["performance_claim_eligible"]
    assert case["retention_eligible"]
    assert case["auto_admission"]["eligible"]


def test_windows_performance_counter_payload_rejects_impossible_values():
    valid = qualification._validate_windows_performance_counter_payload(
        {
            "cpu": {"PercentProcessorPerformance": 118, "ProcessorFrequency": 5440},
            "gpu": [
                {"Name": "3D", "UtilizationPercentage": 75},
                {"Name": "Copy", "UtilizationPercentage": 12},
            ],
        }
    )
    invalid = qualification._validate_windows_performance_counter_payload(
        {
            "cpu": {"PercentProcessorPerformance": 118, "ProcessorFrequency": 5440},
            "gpu": [{"Name": "3D", "UtilizationPercentage": 1.8e14}],
        }
    )

    assert valid["qualified"]
    assert valid["gpu_engine_max_utilization_percent"] == 75
    assert not invalid["qualified"]
    assert invalid["reasons"] == ("invalid_gpu_engine_counter",)


def test_unqualified_performance_environment_blocks_formal_claim():
    workers = [
        _worker(
            "ab" if index < 4 else "ba",
            [1.0] * 5,
            [2.0] * 5,
            [2.0] * 5,
            block_ms=100.0,
            pid=index + 1,
        )
        for index in range(8)
    ]
    for worker in workers:
        worker["performance_environment"] = {"qualified": True, "reasons": ()}
    workers[-1]["performance_environment"] = {
        "qualified": False,
        "reasons": ("invalid_gpu_engine_counter",),
    }

    report = qualification._aggregate("synthetic", tuple(workers), 0.10, 0.10)

    assert report["performance_state"] == "stable_positive"
    assert not report["performance_environment"]["qualified"]
    assert report["performance_environment"]["reasons"] == ("invalid_gpu_engine_counter",)
    assert not report["performance_evidence"]["qualified"]
    assert "performance_environment_unqualified" in report["performance_evidence"]["reasons"]
    assert not report["performance_claim_eligible"]


def _cuda_replay_graph_statistics():
    return {
        "diagnostics_counters_complete": True,
        "backend": "cuda",
        "capture_attempts": 1,
        "captures": 1,
        "exact_replays": 39,
        "patched_replays": 0,
        "recaptures": 0,
        "ordinary_fallbacks": 0,
        "transient_failures": 0,
        "capture_exceptions": 0,
        "last_path": "cuda_exact_replay",
        "backend_replay_signature_slots": 1,
        "backend_replay_signature_slot_capacity": 2,
    }


def test_cuda_replay_proof_uses_cuda_counters_and_separate_gates():
    workers = [
        _worker(
            "ab" if index < 4 else "ba",
            [1.0] * 5,
            [2.0] * 5,
            [2.0] * 5,
            block_ms=100.0,
            pid=index + 1,
        )
        for index in range(8)
    ]
    for worker in workers:
        worker["replay_proof"] = {
            "enabled": True,
            "baseline_mode": "rerecord",
            "graph_statistics": _cuda_replay_graph_statistics(),
            "lifecycle": {
                "scope": "fresh_process_capture_replay_runtime_reset",
                "runtime_reset_completed": True,
            },
        }

    report = qualification._aggregate("cuda-spmv-krylov", tuple(workers), 0.10, 0.10)

    gate = report["replay_proof_gate"]
    assert gate["scope"] == "cuda_mixed_capture_vs_rerecord"
    assert gate["counters_qualified"]
    assert gate["lifecycle_gate_passed"]
    assert gate["performance_gate_passed"]
    assert gate["retention_gate_passed"]
    assert not report["performance_claim_eligible"]


def test_replay_proof_fails_closed_when_fresh_process_scope_is_incomplete():
    workers = [
        _worker(
            "ab" if index < 4 else "ba",
            [1.0] * 5,
            [2.0] * 5,
            [2.0] * 5,
            block_ms=100.0,
            pid=index + 1,
        )
        for index in range(8)
    ]
    for worker in workers[:-1]:
        worker["replay_proof"] = {
            "enabled": True,
            "baseline_mode": "taichi",
            "graph_statistics": _cuda_replay_graph_statistics(),
            "lifecycle": {"runtime_reset_completed": True},
        }

    report = qualification._aggregate("cuda-spmv-krylov", tuple(workers), 0.10, 0.10)

    assert report["replay_proof_gate"] == {
        "scope": "unqualified_replay_proof",
        "gate_reason": "incomplete_or_mixed_worker_scope",
        "counters_qualified": False,
        "lifecycle_gate_passed": False,
        "performance_gate_passed": False,
        "retention_gate_passed": False,
    }
    assert not report["performance_claim_eligible"]


def test_vulkan_replay_proof_keeps_retained_graphics_gate_shape():
    workers = [
        _worker(
            "ab" if index < 4 else "ba",
            [1.0] * 5,
            [2.0] * 5,
            [2.0] * 5,
            block_ms=100.0,
            pid=index + 1,
        )
        for index in range(8)
    ]
    for worker in workers:
        worker["backend"] = "vulkan"
        worker["workload"]["retained_binding_sets"] = 1
        worker["correctness"] = {
            "binding_sets": [
                {
                    "hardware_nonempty": True,
                    "rerecord_exact_image_match": True,
                }
            ]
        }
        worker["memory"] = {"pipeline_closed": {"lifecycle_state": "closed"}}
        worker["replay_proof"] = {
            "enabled": True,
            "baseline_mode": "rerecord",
            "runtime_statistics": {
                "retained_replay_prewarms": 1,
                "retained_replay_records": 1,
                "retained_replay_replays": 39,
                "retained_replay_busy_fallbacks": 0,
                "retained_replay_submit_failures": 0,
                "retained_replay_bridge_failures": 0,
                "retained_replay_slots": 1,
                "retained_replay_slot_capacity": 2,
                "retained_replay_peak_slots": 1,
            },
        }

    report = qualification._aggregate("vulkan-offscreen-simulation", tuple(workers), 0.10, 0.10)

    gate = report["replay_proof_gate"]
    assert gate["scope"] == "mechanism_retained_vs_rerecord"
    assert gate["counters_qualified"]
    assert gate["lifecycle_gate_passed"]
    assert gate["wall_gate_passed"]
    assert not gate["gpu_stage_gate_passed"]
    assert not gate["performance_gate_passed"]
    assert not gate["retention_gate_passed"]


def test_vulkan_binding_rotation_is_lifecycle_only_not_replay_performance():
    workers = [
        _worker(
            "ab" if index < 4 else "ba",
            [1.0] * 5,
            [2.0] * 5,
            [2.0] * 5,
            block_ms=100.0,
            pid=index + 1,
        )
        for index in range(8)
    ]
    for worker in workers:
        worker["backend"] = "vulkan"
        worker["workload"]["retained_binding_sets"] = 2
        worker["correctness"] = {
            "binding_sets": [
                {
                    "hardware_nonempty": True,
                    "rerecord_exact_image_match": True,
                },
                {
                    "hardware_nonempty": True,
                    "rerecord_exact_image_match": True,
                },
            ]
        }
        worker["memory"] = {"pipeline_closed": {"lifecycle_state": "closed"}}
        worker["replay_proof"] = {
            "enabled": True,
            "baseline_mode": "rerecord",
            "runtime_statistics": {
                "retained_replay_attempts": 100,
                "retained_replay_binding_misses": 99,
                "retained_replay_invalidations": 99,
                "retained_replay_prewarms": 100,
                "retained_replay_records": 0,
                "retained_replay_replays": 0,
                "retained_replay_busy_fallbacks": 0,
                "retained_replay_submit_failures": 0,
                "retained_replay_bridge_failures": 0,
                "retained_replay_slots": 0,
                "retained_replay_slot_capacity": 2,
                "retained_replay_peak_slots": 1,
            },
        }

    report = qualification._aggregate("vulkan-offscreen-simulation", tuple(workers), 0.10, 0.10)

    gate = report["replay_proof_gate"]
    assert gate["scope"] == "vulkan_binding_rotation_lifecycle"
    assert gate["counters_qualified"]
    assert gate["lifecycle_gate_passed"]
    assert not gate["performance_gate_passed"]
    assert not gate["retention_gate_passed"]
    assert not report["performance_claim_eligible"]


def test_vulkan_multi_packet_requires_low_sync_lifecycle_and_performance():
    workers = [
        _worker(
            "ab" if index < 4 else "ba",
            [1.0] * 5,
            [1.1] * 5,
            [1.1] * 5,
            block_ms=100.0,
            pid=index + 1,
        )
        for index in range(8)
    ]
    for worker in workers:
        worker["backend"] = "vulkan"
        worker["workload"].update({"retained_binding_sets": 1, "retained_packets_per_burst": 2})
        worker["correctness"] = {
            "binding_sets": [
                {
                    "hardware_nonempty": True,
                    "rerecord_exact_image_match": True,
                }
            ]
        }
        worker["memory"] = {"pipeline_closed": {"lifecycle_state": "closed"}}
        worker["replay_proof"] = {
            "enabled": True,
            "baseline_mode": "rerecord",
            "runtime_statistics": {
                "retained_replay_prewarms": 1,
                "retained_replay_records": 2,
                "retained_replay_replays": 100,
                "retained_replay_busy_fallbacks": 0,
                "retained_replay_submit_failures": 0,
                "retained_replay_bridge_failures": 0,
                "retained_replay_slots": 2,
                "retained_replay_slot_capacity": 2,
                "retained_replay_peak_slots": 2,
            },
        }
        worker["packet_timing"] = {
            "scope": "two fixed-binding packets with one terminal wait",
            "samples_ms": {"hardware": [1.0] * 5, "baseline": [1.1] * 5},
            "paired_speedups": [1.1] * 5,
            "calibration": {
                "hardware": {"satisfied": True, "observed_block_ms": 100.0},
                "baseline": {"satisfied": True, "observed_block_ms": 100.0},
            },
        }
        worker["packet_lifecycle"] = {
            "packets_per_burst": 2,
            "binding_sets": 1,
            "calls": {
                "hardware": {"bursts": 5, "submissions": 10, "completion_waits": 5},
                "baseline": {"bursts": 5, "submissions": 10, "completion_waits": 5},
            },
            "hardware_workspace_lane_waits_delta": 0,
            "baseline_workspace_lane_waits_delta": 0,
            "hardware_workspace_lanes_busy_after": 0,
            "baseline_workspace_lanes_busy_after": 0,
            "retained_replay_busy_fallbacks_delta": 0,
            "retained_replay_submit_failures_delta": 0,
            "retained_replay_bridge_failures_delta": 0,
        }

    report = qualification._aggregate("vulkan-offscreen-simulation", tuple(workers), 0.10, 0.10)
    gate = report["replay_proof_gate"]
    assert gate["scope"] == "vulkan_fixed_binding_multi_packet"
    assert gate["lifecycle_gate_passed"]
    assert gate["low_sync_gate_passed"]
    assert gate["packet_performance_gate_passed"]
    assert gate["retention_gate_passed"]
    assert not report["performance_claim_eligible"]

    workers[0]["packet_lifecycle"]["hardware_workspace_lane_waits_delta"] = 1
    report = qualification._aggregate("vulkan-offscreen-simulation", tuple(workers), 0.10, 0.10)
    gate = report["replay_proof_gate"]
    assert gate["lifecycle_gate_passed"]
    assert not gate["low_sync_gate_passed"]
    assert not gate["retention_gate_passed"]

    qualification._apply_build_provenance_gate([report], "different-revision")
    assert not report["packet_timing"]["performance_evidence_qualified"]
    assert not report["packet_timing"]["gate_passed"]
    assert report["packet_timing"]["gate_reason"] == "build_provenance_unqualified"


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
    assert provenance["sha256"] == "3bdb141509c6111dee71c967b1c7e38875c39a5f646009caeb61aa7fc2c5a418"


def test_runtime_bitcode_provenance_records_only_loaded_artifact_kinds(tmp_path):
    (tmp_path / "runtime_cuda.bc").write_bytes(b"cuda")
    (tmp_path / "runtime_x64.bc").write_bytes(b"x64")
    (tmp_path / "slim_libdevice.10.bc").write_bytes(b"libdevice")
    (tmp_path / "unrelated.bc").write_bytes(b"unrelated")

    provenance = qualification._runtime_bitcode_provenance(tmp_path)

    assert [artifact["path"].replace("\\", "/").rsplit("/", 1)[-1] for artifact in provenance] == [
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
    assert case["auto_admission"]["evidence"]["performance"]["baseline_first_use_overhead_ns"] == 100_000.0
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
