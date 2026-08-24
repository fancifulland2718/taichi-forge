import copy
import json

import pytest

from taichi_forge.hardware import (
    ProviderAdmissionEvidence,
    load_provider_admission_evidence,
)
from taichi_forge.hardware._admission import evaluate_provider_admission


def _evidence_record():
    return {
        "schema": "taichi_forge.provider_admission.v1",
        "schema_version": 1,
        "operation_id": "linalg.spmv.cusparse",
        "provider_id": "cusparse",
        "baseline_id": "cuda_driver_kernel",
        "backend": "cuda",
        "device_scope": {
            "cuda_device_uuid": "00112233445566778899aabbccddeeff",
            "cuda_compute_capability": 80,
        },
        "provider_scope": {
            "provider_abi": "cusparse-generic-spmv",
            "provider_version": {"major": 12, "minor": 4, "patch": 0},
        },
        "workload_scope": {
            "rows": 131072,
            "cols": 131072,
            "nnz": 917504,
            "storage_format": "csr",
            "block_size": None,
            "topology_fingerprint": "tf-sp-v1:0123456789abcdef0123456789abcdef",
        },
        "runtime_scope": {
            "forge_version": "0.6.3",
            "forge_commit": "test-revision",
            "python_provider_contract_sha256": "a" * 64,
        },
        "performance": {
            "expected_reuse": 10,
            "provider_median_ns": 50_000.0,
            "baseline_median_ns": 100_000.0,
            "provider_first_use_overhead_ns": 10_000.0,
            "baseline_first_use_overhead_ns": 0.0,
            "transfer_ns": 0.0,
            "conversion_ns": 0.0,
            "provider_samples": 48,
            "baseline_samples": 48,
            "provider_cv": 0.02,
            "baseline_cv": 0.02,
            "order_drift": 0.01,
            "minimum_block_ms": 100.0,
            "minimum_margin": 0.05,
            "paired_p05": 1.8,
            "fresh_processes": 8,
            "order_processes": {"ab": 4, "ba": 4},
        },
        "qualification": {
            "correctness_and_route_qualified": True,
            "stable": True,
            "minimum_block_qualified": True,
        },
    }


def _artifact(record=None):
    return {
        "schema": "taichi_forge.hardware_acceleration_qualification.v4",
        "cases": [
            {
                "case": "cuda-spmv",
                "auto_admission": {
                    "eligible": True,
                    "evidence": _evidence_record() if record is None else record,
                },
            }
        ],
    }


def _write_artifact(tmp_path, report):
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _evaluate(evidence, **overrides):
    values = {
        "operation_id": evidence.operation_id,
        "provider_id": evidence.provider_id,
        "baseline_id": evidence.baseline_id,
        "backend": evidence.backend,
        "device_scope": dict(evidence.device_scope),
        "provider_scope": {
            "provider_abi": evidence.provider_scope["provider_abi"],
            "provider_version": dict(evidence.provider_scope["provider_version"]),
        },
        "workload_scope": dict(evidence.workload_scope),
        "runtime_scope": dict(evidence.runtime_scope),
    }
    values.update(overrides)
    return evaluate_provider_admission(evidence, **values)


def test_admission_evidence_loads_only_from_qualified_artifact(tmp_path):
    path = _write_artifact(tmp_path, _artifact())

    evidence = load_provider_admission_evidence(path, case="cuda-spmv")

    assert evidence.operation_id == "linalg.spmv.cusparse"
    assert evidence.fresh_processes == 8
    assert len(evidence.source_artifact_sha256) == 64
    assert evidence.to_dict()["performance"]["paired_p05"] == 1.8
    with pytest.raises(TypeError):
        evidence.device_scope["cuda_compute_capability"] = 90
    with pytest.raises(TypeError, match="must be loaded"):
        ProviderAdmissionEvidence()


def test_admission_matches_all_scopes_and_rechecks_amortized_cost(tmp_path):
    evidence = load_provider_admission_evidence(_write_artifact(tmp_path, _artifact()))

    admitted = _evaluate(evidence)
    assert admitted.admitted
    assert admitted.route == "provider"
    assert admitted.reason == "qualified_cost_advantage"
    assert admitted.provider_amortized_ns == 51_000.0
    assert admitted.baseline_amortized_ns == 100_000.0

    mismatched = dict(evidence.workload_scope)
    mismatched["topology_fingerprint"] = "tf-sp-v1:different"
    rejected = _evaluate(evidence, workload_scope=mismatched)
    assert not rejected.admitted
    assert rejected.reason == "workload_scope_mismatch"


@pytest.mark.parametrize(
    ("path", "value", "message"),
    (
        (("performance", "fresh_processes"), 6, "fresh-process"),
        (("performance", "provider_samples"), 39, "timing samples"),
        (("performance", "minimum_block_ms"), 99.0, "too short"),
        (("performance", "provider_cv"), 0.051, "CV"),
        (("performance", "order_drift"), 0.051, "order drift"),
        (("performance", "paired_p05"), 1.01, "paired p05"),
    ),
)
def test_admission_rejects_underqualified_statistics(tmp_path, path, value, message):
    record = _evidence_record()
    target = record
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = value
    if path[-1] == "fresh_processes":
        record["performance"]["order_processes"] = {"ab": 3, "ba": 3}

    artifact = _write_artifact(tmp_path, _artifact(record))

    with pytest.raises(ValueError, match=message):
        load_provider_admission_evidence(artifact)


def test_admission_rejects_cost_that_does_not_cover_first_use(tmp_path):
    record = copy.deepcopy(_evidence_record())
    record["performance"]["provider_median_ns"] = 93_000.0
    record["performance"]["provider_first_use_overhead_ns"] = 30_000.0

    with pytest.raises(ValueError, match="amortized provider cost"):
        load_provider_admission_evidence(_write_artifact(tmp_path, _artifact(record)))


def test_admission_amortizes_baseline_first_use_cost_symmetrically(tmp_path):
    record = copy.deepcopy(_evidence_record())
    record["performance"]["provider_median_ns"] = 90_000.0
    record["performance"]["provider_first_use_overhead_ns"] = 500_000.0
    record["performance"]["baseline_first_use_overhead_ns"] = 1_000_000.0
    record["performance"]["paired_p05"] = 1.1

    evidence = load_provider_admission_evidence(
        _write_artifact(tmp_path, _artifact(record))
    )
    admitted = _evaluate(evidence)

    assert admitted.admitted
    assert admitted.provider_amortized_ns == 140_000.0
    assert admitted.baseline_amortized_ns == 200_000.0


def test_admission_accepts_exact_cudss_auto_workload_scope(tmp_path):
    record = copy.deepcopy(_evidence_record())
    record.update(
        {
            "operation_id": "linalg.solve.cudss_auto",
            "provider_id": "cudss",
            "baseline_id": "cusolver_sp",
            "provider_scope": {
                "provider_abi": "cudss-c-api-0.8",
                "provider_version": {"major": 0, "minor": 8, "patch": 1},
            },
        }
    )
    record["workload_scope"].update(
        {
            "solver_type": "LLT",
            "ordering": "AMD",
            "matrix_type": "spd",
            "matrix_view": "full",
            "workflow": "analyze_factorize_then_repeated_solve",
        }
    )
    report = _artifact(record)
    report["cases"][0]["case"] = "cuda-cudss-solve"

    evidence = load_provider_admission_evidence(
        _write_artifact(tmp_path, report), case="cuda-cudss-solve"
    )

    assert evidence.operation_id == "linalg.solve.cudss_auto"
    assert evidence.provider_scope["provider_abi"] == "cudss-c-api-0.8"
    assert evidence.workload_scope["workflow"] == (
        "analyze_factorize_then_repeated_solve"
    )


def test_admission_requires_unambiguous_case_selection(tmp_path):
    report = _artifact()
    second = copy.deepcopy(report["cases"][0])
    second["case"] = "cuda-spmv-second"
    report["cases"].append(second)
    path = _write_artifact(tmp_path, report)

    with pytest.raises(ValueError, match="ambiguous"):
        load_provider_admission_evidence(path)
    selected = load_provider_admission_evidence(path, case="cuda-spmv")
    assert selected.operation_id == "linalg.spmv.cusparse"
