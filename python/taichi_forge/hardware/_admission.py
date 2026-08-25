"""Validated, workload-scoped performance admission for automatic providers."""

import copy
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Optional


PROVIDER_ADMISSION_SCHEMA = "taichi_forge.provider_admission.v2"
PROVIDER_ADMISSION_SCHEMA_VERSION = 2
_QUALIFICATION_SOURCE_SCHEMA = "taichi_forge.hardware_acceleration_qualification.v7"

_MINIMUM_FRESH_PROCESSES = 8
_MINIMUM_PROCESSES_PER_ORDER = 4
_MINIMUM_SAMPLES_PER_VARIANT = 40
_MINIMUM_BLOCK_MS = 100.0
_MAXIMUM_CV = 0.05
_MAXIMUM_ORDER_DRIFT = 0.05
_MINIMUM_MARGIN = 0.05


def _json_value(value, name):
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{name} keys must be nonempty strings")
            result[key] = _json_value(item, f"{name}.{key}")
        return result
    if isinstance(value, (tuple, list)):
        return [_json_value(item, name) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must not contain non-finite values")
        return value
    raise TypeError(f"{name} contains unsupported value {type(value).__name__}")


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value):
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return copy.deepcopy(value)


def _nonempty_string(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _sha256_string(value, name):
    value = _nonempty_string(value, name)
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _positive_number(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def _nonnegative_number(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be a nonnegative finite number")
    return float(value)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True, init=False)
class ProviderAdmissionEvidence:
    """Immutable evidence emitted by a Forge fresh-process qualification run.

    Instances cannot be assembled from handwritten timing fields. Use
    :func:`load_provider_admission_evidence` on a qualification artifact.
    """

    schema_version: int
    operation_id: str
    provider_id: str
    baseline_id: str
    backend: str
    device_scope: Mapping
    provider_scope: Mapping
    workload_scope: Mapping
    runtime_scope: Mapping
    expected_reuse: int
    provider_median_ns: float
    baseline_median_ns: float
    provider_first_use_overhead_ns: float
    baseline_first_use_overhead_ns: float
    transfer_ns: float
    conversion_ns: float
    provider_samples: int
    baseline_samples: int
    provider_cv: float
    baseline_cv: float
    order_drift: float
    minimum_block_ms: float
    minimum_margin: float
    paired_p05: float
    fresh_processes: int
    order_processes: Mapping
    source_schema: str
    source_artifact_sha256: str

    def __init__(self, *args, **kwargs):
        del args, kwargs
        raise TypeError(
            "ProviderAdmissionEvidence must be loaded from a Forge "
            "qualification artifact"
        )

    @classmethod
    def _from_record(cls, record, *, source_schema, source_digest):
        if not isinstance(record, Mapping):
            raise TypeError("provider admission evidence must be a mapping")
        record = _json_value(record, "provider admission evidence")
        if record.get("schema") != PROVIDER_ADMISSION_SCHEMA:
            raise ValueError("provider admission evidence schema mismatch")
        if record.get("schema_version") != PROVIDER_ADMISSION_SCHEMA_VERSION:
            raise ValueError("provider admission evidence schema version mismatch")
        operation_id = _nonempty_string(record.get("operation_id"), "operation_id")
        qualification = record.get("qualification")
        if not isinstance(qualification, Mapping):
            raise ValueError("provider admission qualification is missing")
        required_checks = (
            "correctness_and_route_qualified",
            "stable",
            "minimum_block_qualified",
        )
        if any(qualification.get(name) is not True for name in required_checks):
            raise ValueError("provider admission qualification did not pass")

        performance = record.get("performance")
        if not isinstance(performance, Mapping):
            raise ValueError("provider admission performance record is missing")
        expected_reuse = _positive_integer(
            performance.get("expected_reuse"), "expected_reuse"
        )
        provider_median_ns = _positive_number(
            performance.get("provider_median_ns"), "provider_median_ns"
        )
        baseline_median_ns = _positive_number(
            performance.get("baseline_median_ns"), "baseline_median_ns"
        )
        first_use_overhead_ns = _nonnegative_number(
            performance.get("provider_first_use_overhead_ns"),
            "provider_first_use_overhead_ns",
        )
        baseline_first_use_overhead_ns = _nonnegative_number(
            performance.get("baseline_first_use_overhead_ns"),
            "baseline_first_use_overhead_ns",
        )
        transfer_ns = _nonnegative_number(performance.get("transfer_ns"), "transfer_ns")
        conversion_ns = _nonnegative_number(
            performance.get("conversion_ns"), "conversion_ns"
        )
        provider_samples = _positive_integer(
            performance.get("provider_samples"), "provider_samples"
        )
        baseline_samples = _positive_integer(
            performance.get("baseline_samples"), "baseline_samples"
        )
        provider_cv = _nonnegative_number(performance.get("provider_cv"), "provider_cv")
        baseline_cv = _nonnegative_number(performance.get("baseline_cv"), "baseline_cv")
        order_drift = _nonnegative_number(performance.get("order_drift"), "order_drift")
        minimum_block_ms = _positive_number(
            performance.get("minimum_block_ms"), "minimum_block_ms"
        )
        minimum_margin = _nonnegative_number(
            performance.get("minimum_margin"), "minimum_margin"
        )
        paired_p05 = _positive_number(performance.get("paired_p05"), "paired_p05")
        fresh_processes = _positive_integer(
            performance.get("fresh_processes"), "fresh_processes"
        )
        order_processes = performance.get("order_processes")
        if not isinstance(order_processes, Mapping):
            raise ValueError("order_processes must be a mapping")
        if set(order_processes) != {"ab", "ba"}:
            raise ValueError("order_processes must contain exactly ab and ba")
        order_processes = {
            order: _positive_integer(count, f"order_processes.{order}")
            for order, count in order_processes.items()
        }
        if sum(order_processes.values()) != fresh_processes:
            raise ValueError("fresh_processes does not match order_processes")
        if minimum_margin < _MINIMUM_MARGIN or minimum_margin >= 1.0:
            raise ValueError("minimum_margin is outside the auto-admission policy")
        if fresh_processes < _MINIMUM_FRESH_PROCESSES or any(
            count < _MINIMUM_PROCESSES_PER_ORDER for count in order_processes.values()
        ):
            raise ValueError("insufficient fresh-process qualification coverage")
        if (
            provider_samples < _MINIMUM_SAMPLES_PER_VARIANT
            or baseline_samples < _MINIMUM_SAMPLES_PER_VARIANT
        ):
            raise ValueError("insufficient timing samples for auto admission")
        if minimum_block_ms < _MINIMUM_BLOCK_MS:
            raise ValueError("timing blocks are too short for auto admission")
        if provider_cv > _MAXIMUM_CV or baseline_cv > _MAXIMUM_CV:
            raise ValueError("timing CV is too high for auto admission")
        if order_drift > _MAXIMUM_ORDER_DRIFT:
            raise ValueError("AB/BA order drift is too high for auto admission")
        if paired_p05 < 1.0 / (1.0 - minimum_margin):
            raise ValueError("paired p05 does not meet the admission margin")
        amortized_ns = (
            provider_median_ns
            + first_use_overhead_ns / expected_reuse
            + transfer_ns
            + conversion_ns
        )
        baseline_amortized_ns = (
            baseline_median_ns + baseline_first_use_overhead_ns / expected_reuse
        )
        if amortized_ns >= baseline_amortized_ns * (1.0 - minimum_margin):
            raise ValueError(
                "amortized provider cost does not meet the admission margin"
            )

        scope_names = (
            "device_scope",
            "provider_scope",
            "workload_scope",
            "runtime_scope",
        )
        scopes = {}
        for name in scope_names:
            scope = record.get(name)
            if not isinstance(scope, Mapping) or not scope:
                raise ValueError(f"{name} must be a nonempty mapping")
            scopes[name] = _freeze(_json_value(scope, name))
        device_scope = _thaw(scopes["device_scope"])
        provider_scope = _thaw(scopes["provider_scope"])
        workload_scope = _thaw(scopes["workload_scope"])
        runtime_scope = _thaw(scopes["runtime_scope"])
        _nonempty_string(
            device_scope.get("cuda_device_uuid"),
            "device_scope.cuda_device_uuid",
        )
        _nonempty_string(
            provider_scope.get("provider_abi"),
            "provider_scope.provider_abi",
        )
        if not isinstance(
            provider_scope.get("provider_version"), Mapping
        ) or not provider_scope.get("provider_version"):
            raise ValueError("provider_scope.provider_version must be a mapping")
        _nonempty_string(
            runtime_scope.get("forge_version"),
            "runtime_scope.forge_version",
        )
        _nonempty_string(
            runtime_scope.get("forge_commit"),
            "runtime_scope.forge_commit",
        )
        _sha256_string(
            runtime_scope.get("python_provider_contract_sha256"),
            "runtime_scope.python_provider_contract_sha256",
        )
        _sha256_string(
            runtime_scope.get("native_runtime_build_identity"),
            "runtime_scope.native_runtime_build_identity",
        )
        native_artifacts = runtime_scope.get("native_runtime_artifacts")
        if not isinstance(native_artifacts, Mapping) or not native_artifacts:
            raise ValueError(
                "runtime_scope.native_runtime_artifacts must be a mapping"
            )
        for artifact_name in (
            "python_extension_sha256",
            "native_runtime_binary_sha256",
            "runtime_bitcode_bundle_sha256",
        ):
            _sha256_string(
                native_artifacts.get(artifact_name),
                f"runtime_scope.native_runtime_artifacts.{artifact_name}",
            )
        if operation_id in (
            "linalg.spmv.cusparse",
            "linalg.solve.cudss_auto",
        ):
            _nonempty_string(
                workload_scope.get("topology_fingerprint"),
                "workload_scope.topology_fingerprint",
            )
        if operation_id == "linalg.solve.cudss_auto":
            _sha256_string(
                provider_scope.get("provider_binary_sha256"),
                "provider_scope.provider_binary_sha256",
            )

        self = object.__new__(cls)
        fields = {
            "schema_version": PROVIDER_ADMISSION_SCHEMA_VERSION,
            "operation_id": operation_id,
            "provider_id": _nonempty_string(record.get("provider_id"), "provider_id"),
            "baseline_id": _nonempty_string(record.get("baseline_id"), "baseline_id"),
            "backend": _nonempty_string(record.get("backend"), "backend"),
            **scopes,
            "expected_reuse": expected_reuse,
            "provider_median_ns": provider_median_ns,
            "baseline_median_ns": baseline_median_ns,
            "provider_first_use_overhead_ns": first_use_overhead_ns,
            "baseline_first_use_overhead_ns": baseline_first_use_overhead_ns,
            "transfer_ns": transfer_ns,
            "conversion_ns": conversion_ns,
            "provider_samples": provider_samples,
            "baseline_samples": baseline_samples,
            "provider_cv": provider_cv,
            "baseline_cv": baseline_cv,
            "order_drift": order_drift,
            "minimum_block_ms": minimum_block_ms,
            "minimum_margin": minimum_margin,
            "paired_p05": paired_p05,
            "fresh_processes": fresh_processes,
            "order_processes": _freeze(order_processes),
            "source_schema": _nonempty_string(source_schema, "source_schema"),
            "source_artifact_sha256": _nonempty_string(
                source_digest, "source_artifact_sha256"
            ),
        }
        for name, value in fields.items():
            object.__setattr__(self, name, value)
        return self

    def to_dict(self):
        """Return a detached diagnostic representation of the evidence."""

        return {
            "schema_version": self.schema_version,
            "operation_id": self.operation_id,
            "provider_id": self.provider_id,
            "baseline_id": self.baseline_id,
            "backend": self.backend,
            "device_scope": _thaw(self.device_scope),
            "provider_scope": _thaw(self.provider_scope),
            "workload_scope": _thaw(self.workload_scope),
            "runtime_scope": _thaw(self.runtime_scope),
            "performance": {
                "expected_reuse": self.expected_reuse,
                "provider_median_ns": self.provider_median_ns,
                "baseline_median_ns": self.baseline_median_ns,
                "provider_first_use_overhead_ns": self.provider_first_use_overhead_ns,
                "baseline_first_use_overhead_ns": self.baseline_first_use_overhead_ns,
                "transfer_ns": self.transfer_ns,
                "conversion_ns": self.conversion_ns,
                "provider_samples": self.provider_samples,
                "baseline_samples": self.baseline_samples,
                "provider_cv": self.provider_cv,
                "baseline_cv": self.baseline_cv,
                "order_drift": self.order_drift,
                "minimum_block_ms": self.minimum_block_ms,
                "minimum_margin": self.minimum_margin,
                "paired_p05": self.paired_p05,
                "fresh_processes": self.fresh_processes,
                "order_processes": _thaw(self.order_processes),
            },
            "source_schema": self.source_schema,
            "source_artifact_sha256": self.source_artifact_sha256,
        }


@dataclass(frozen=True)
class ProviderAdmissionDecision:
    """One fail-closed automatic-provider decision."""

    admitted: bool
    route: str
    reason: str
    provider_amortized_ns: Optional[float] = None
    baseline_median_ns: Optional[float] = None
    baseline_amortized_ns: Optional[float] = None
    expected_reuse: Optional[int] = None
    evidence_expected_reuse: Optional[int] = None

    def to_dict(self):
        return {
            "admitted": self.admitted,
            "route": self.route,
            "reason": self.reason,
            "provider_amortized_ns": self.provider_amortized_ns,
            "baseline_median_ns": self.baseline_median_ns,
            "baseline_amortized_ns": self.baseline_amortized_ns,
            "expected_reuse": self.expected_reuse,
            "evidence_expected_reuse": self.evidence_expected_reuse,
        }


def load_provider_admission_evidence(path, *, case=None):
    """Load one validated evidence record from a Forge qualification artifact."""

    artifact_path = Path(path)
    payload = artifact_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    try:
        report = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("provider admission artifact is not valid JSON") from exc
    if not isinstance(report, Mapping):
        raise ValueError("provider admission artifact root must be a mapping")
    source_schema = report.get("schema")
    _nonempty_string(source_schema, "qualification artifact schema")
    if source_schema != _QUALIFICATION_SOURCE_SCHEMA:
        raise ValueError("qualification artifact schema is not admission-capable")
    cases = report.get("cases")
    if not isinstance(cases, list):
        raise ValueError("qualification artifact must contain a cases list")
    candidates = []
    for item in cases:
        if not isinstance(item, Mapping):
            continue
        if case is not None and item.get("case") != case:
            continue
        admission = item.get("auto_admission")
        if isinstance(admission, Mapping) and admission.get("eligible") is True:
            candidates.append(admission.get("evidence"))
    if not candidates:
        suffix = "" if case is None else f" for case {case!r}"
        raise ValueError(f"qualification artifact has no eligible evidence{suffix}")
    if len(candidates) != 1:
        raise ValueError("qualification artifact evidence is ambiguous; specify case")
    return ProviderAdmissionEvidence._from_record(
        candidates[0], source_schema=source_schema, source_digest=digest
    )


def evaluate_provider_admission(
    evidence,
    *,
    operation_id,
    provider_id,
    baseline_id,
    backend,
    device_scope,
    provider_scope,
    workload_scope,
    runtime_scope,
    provider_warmed=False,
    expected_reuse=None,
):
    """Match validated evidence to a current operation and evaluate its cost gate."""

    if evidence is None:
        return ProviderAdmissionDecision(
            False, "fallback", "missing_admission_evidence"
        )
    if not isinstance(evidence, ProviderAdmissionEvidence):
        return ProviderAdmissionDecision(
            False, "fallback", "invalid_admission_evidence"
        )
    if expected_reuse is None:
        expected_reuse = evidence.expected_reuse
    elif (
        isinstance(expected_reuse, bool)
        or not isinstance(expected_reuse, int)
        or expected_reuse <= 0
    ):
        return ProviderAdmissionDecision(
            False,
            "fallback",
            "invalid_expected_reuse",
            evidence_expected_reuse=evidence.expected_reuse,
        )
    reuse_scope = {
        "expected_reuse": expected_reuse,
        "evidence_expected_reuse": evidence.expected_reuse,
    }
    expected_scalars = {
        "operation_id": operation_id,
        "provider_id": provider_id,
        "baseline_id": baseline_id,
        "backend": backend,
    }
    for name, actual in expected_scalars.items():
        if getattr(evidence, name) != actual:
            return ProviderAdmissionDecision(
                False, "fallback", f"{name}_mismatch", **reuse_scope
            )
    expected_scopes = {
        "device_scope": device_scope,
        "provider_scope": provider_scope,
        "workload_scope": workload_scope,
        "runtime_scope": runtime_scope,
    }
    for name, actual in expected_scopes.items():
        try:
            normalized = _json_value(actual, name)
        except (TypeError, ValueError):
            return ProviderAdmissionDecision(
                False, "fallback", f"{name}_unavailable", **reuse_scope
            )
        if _thaw(getattr(evidence, name)) != normalized:
            return ProviderAdmissionDecision(
                False, "fallback", f"{name}_mismatch", **reuse_scope
            )
    first_use = 0.0 if provider_warmed else evidence.provider_first_use_overhead_ns
    provider_cost = (
        evidence.provider_median_ns
        + first_use / expected_reuse
        + evidence.transfer_ns
        + evidence.conversion_ns
    )
    baseline_cost = (
        evidence.baseline_median_ns
        + evidence.baseline_first_use_overhead_ns / expected_reuse
    )
    if provider_cost >= baseline_cost * (1.0 - evidence.minimum_margin):
        return ProviderAdmissionDecision(
            False,
            "fallback",
            "cost_gate",
            provider_amortized_ns=provider_cost,
            baseline_median_ns=evidence.baseline_median_ns,
            baseline_amortized_ns=baseline_cost,
            expected_reuse=expected_reuse,
            evidence_expected_reuse=evidence.expected_reuse,
        )
    return ProviderAdmissionDecision(
        True,
        "provider",
        "qualified_cost_advantage",
        provider_amortized_ns=provider_cost,
        baseline_median_ns=evidence.baseline_median_ns,
        baseline_amortized_ns=baseline_cost,
        expected_reuse=expected_reuse,
        evidence_expected_reuse=evidence.expected_reuse,
    )


def _file_sha256(path):
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_bitcode_identity(directory):
    root = Path(directory)
    candidates = [root / "runtime_cuda.bc", root / "runtime_x64.bc"]
    candidates.extend(sorted(root.glob("slim_libdevice.*.bc")))
    artifacts = {
        candidate.name: _file_sha256(candidate)
        for candidate in candidates
        if candidate.is_file()
    }
    if not artifacts:
        return None
    payload = json.dumps(
        artifacts, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@lru_cache(maxsize=1)
def _current_runtime_scope():
    from taichi_forge._lib import core as _ti_core  # pylint: disable=C0415
    from taichi_forge._lib import utils as runtime_utils  # pylint: disable=C0415

    contract_digest = hashlib.sha256()
    package_root = Path(__file__).resolve().parents[1]
    contract_paths = (
        Path(__file__).resolve(),
        package_root / "linalg" / "sparse_matrix.py",
        package_root / "linalg" / "sparse_solver.py",
    )
    for contract_path in contract_paths:
        contract_digest.update(contract_path.name.encode("utf-8"))
        contract_digest.update(b"\0")
        contract_digest.update(contract_path.read_bytes())
        contract_digest.update(b"\0")
    native_artifacts = {
        "python_extension_sha256": _file_sha256(
            getattr(_ti_core, "__file__", None)
        ),
        "native_runtime_binary_sha256": _file_sha256(
            getattr(runtime_utils, "_loaded_native_runtime_path", None)
        ),
        "runtime_bitcode_bundle_sha256": _runtime_bitcode_identity(
            runtime_utils._runtime_bitcode_dir()  # pylint: disable=W0212
        ),
    }
    identity_payload = json.dumps(
        native_artifacts, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "forge_version": _ti_core.get_version_string(),
        "forge_commit": _ti_core.get_commit_hash(),
        "python_provider_contract_sha256": contract_digest.hexdigest(),
        "native_runtime_build_identity": hashlib.sha256(
            identity_payload
        ).hexdigest(),
        "native_runtime_artifacts": native_artifacts,
    }


def _current_cuda_device_scope():
    from taichi_forge.interop import (  # pylint: disable=C0415
        current_cuda_device_uuid,
    )
    from taichi_forge.lang import impl  # pylint: disable=C0415

    try:
        device_uuid = current_cuda_device_uuid().hex()
    except (RuntimeError, ValueError):
        device_uuid = None
    try:
        compute_capability = int(impl.get_cuda_compute_capability())
    except (RuntimeError, ValueError):
        compute_capability = None
    return {
        "cuda_device_uuid": device_uuid,
        "cuda_compute_capability": compute_capability,
    }


__all__ = [
    "PROVIDER_ADMISSION_SCHEMA",
    "PROVIDER_ADMISSION_SCHEMA_VERSION",
    "ProviderAdmissionDecision",
    "ProviderAdmissionEvidence",
    "load_provider_admission_evidence",
]
