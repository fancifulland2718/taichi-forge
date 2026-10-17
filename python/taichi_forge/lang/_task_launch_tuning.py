"""Internal qualification-gated launch specialization selection.

This module deliberately does not benchmark kernels.  A physics workload may
mutate fields, consume queues, or depend on replay state, so ordinary execution
cannot safely call it several times.  Qualification tooling publishes an
immutable record after correctness and fresh-process AB/BA gates; the runtime
only validates and consumes that record.
"""

from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import threading

from taichi_forge._lib import core as _ti_core
from taichi_forge.lang import impl


_SCHEMA_VERSION = 2
_CANDIDATE_BLOCK_DIMS = (64, 128, 256, 512)
_VULKAN_CANDIDATE_BLOCK_DIMS = (32, 64, 128, 256, 512)
_HOT_CALL_THRESHOLD = 8
_MIN_INDEPENDENT_BLOCKS = 5
_MAX_RECORD_CACHE_ENTRIES = 256
_MAX_OBSERVED_KERNELS = 1024


@dataclass(frozen=True)
class _TaskLaunchTuningDecision:
    status: str
    reason: str
    record_id: str
    kernel_key: str
    hardware_scope: tuple
    candidates: tuple
    block_dim: int | None = None
    observed_calls: int = 0
    workload_profile_id: str = ""
    record_scope_json: str = ""


@dataclass(frozen=True)
class _TaskLaunchWorkloadProfile:
    """Explicit, qualification-owned identity for one replayable workload."""

    workload_id: str
    input_distribution_id: str
    shape: tuple = ()
    shape_bucket: str = ""
    graph_capacity: int | None = None
    graph_signature: str = ""
    sparse_active_ratio_bucket: str = "not_applicable"
    topology_stability: str = "static"
    ir_identity: str = ""
    ptx_identity: str = ""
    oracle_identity: str = ""
    replay_identity: str = ""
    spirv_identity: str = ""

    def __post_init__(self):
        for name in (
            "workload_id",
            "input_distribution_id",
            "ir_identity",
            "oracle_identity",
            "replay_identity",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or "\n" in value:
                raise ValueError(f"{name} must be a non-empty single-line string")
        for name in ("ptx_identity", "spirv_identity"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or "\n" in value
                or (value and not value.strip())
            ):
                raise ValueError(f"{name} must be a single-line string")
        if not self.ptx_identity and not self.spirv_identity:
            raise ValueError("ptx_identity or spirv_identity is required")
        if not isinstance(self.shape, tuple) or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in self.shape
        ):
            raise ValueError("shape must be a tuple of positive integers")
        if not self.shape and not self.shape_bucket:
            raise ValueError("an exact shape or a shape_bucket is required")
        if not isinstance(self.shape_bucket, str) or "\n" in self.shape_bucket:
            raise ValueError("shape_bucket must be a single-line string")
        if self.graph_capacity is not None and (
            isinstance(self.graph_capacity, bool)
            or not isinstance(self.graph_capacity, int)
            or self.graph_capacity <= 0
        ):
            raise ValueError("graph_capacity must be a positive integer")
        if bool(self.graph_signature) != (self.graph_capacity is not None):
            raise ValueError(
                "graph_signature and graph_capacity must either both be set or both be absent"
            )
        if self.sparse_active_ratio_bucket not in (
            "not_applicable",
            "dense",
            "low",
            "medium",
            "high",
        ):
            raise ValueError("unsupported sparse_active_ratio_bucket")
        if self.topology_stability not in (
            "static",
            "bounded_dynamic",
            "dynamic",
        ):
            raise ValueError("unsupported topology_stability")
        scope = asdict(self)
        if not self.spirv_identity:
            del scope["spirv_identity"]  # Preserve existing CUDA record identities.
        stable_scope_json = _canonical_json(scope)
        object.__setattr__(self, "_stable_scope_json", stable_scope_json)
        object.__setattr__(
            self,
            "_identity",
            f"tlw1:{hashlib.sha256(stable_scope_json.encode('utf-8')).hexdigest()}",
        )

    @property
    def stable_scope(self):
        return json.loads(self._stable_scope_json)

    @property
    def identity(self):
        return self._identity


def _canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_json(value):
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@lru_cache(maxsize=16)
def _file_sha256(path, size, mtime_ns):
    del size, mtime_ns
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_file_identity(configured):
    if not configured:
        return None
    try:
        path = Path(configured).resolve(strict=True)
        stat = path.stat()
        if not path.is_file():
            return {"path": str(path), "sha256": None}
        return {
            "path": str(path),
            "sha256": _file_sha256(str(path), stat.st_size, stat.st_mtime_ns),
        }
    except OSError:
        return {"path": str(configured), "sha256": None}


def _compiler_scope():
    mode = os.environ.get("TI_CUDA_PTXAS_MODE", "driver").strip().lower()
    if not mode:
        mode = "driver"
    result = {
        "mode": mode,
        "llvm_version": _ti_core.get_llvm_target_support(),
    }
    if mode != "external":
        return result

    ptxas = os.environ.get("TI_CUDA_PTXAS_PATH")
    if not ptxas:
        ptxas = shutil.which("ptxas") or shutil.which("ptxas.exe")
    result["ptxas"] = _optional_file_identity(ptxas)
    direct_acf = os.environ.get("TI_CUDA_PTXAS_ACF_PATH")
    worker = os.environ.get("TI_CUDA_COMPILEIQ_WORKER")
    active_request = os.environ.get("TI_CUDA_COMPILEIQ_ACTIVE_REQUEST") == "1"
    if direct_acf:
        result["advanced_controls_mode"] = "apply_explicit_acf"
        result["direct_acf"] = _optional_file_identity(direct_acf)
    elif worker and not active_request:
        result["advanced_controls_mode"] = "request_tuning"
        result["compileiq_worker"] = _optional_file_identity(worker)
        result["compileiq_python"] = _optional_file_identity(
            os.environ.get("TI_CUDA_COMPILEIQ_PYTHON")
        )
    else:
        result["advanced_controls_mode"] = "baseline"
    return result


def _hardware_scope():
    program = impl.get_runtime().prog
    if impl.current_cfg().arch == _ti_core.vulkan:
        from taichi_forge.tools.vulkan_diagnostics import vulkan_runtime_diagnostics

        info = vulkan_runtime_diagnostics()
        device = info["device"]
        if not device["device_uuid"].strip("0") or device["subgroup_size"] <= 0:
            raise RuntimeError(
                "qualified Vulkan profiles require exact device and subgroup identity"
            )
        return {
            **device,
            "backend": "vulkan",
            "forge_version": info["forge_version"],
            "forge_commit": info["forge_commit"],
            "compiler": {"provider": "spirv_codegen"},
        }
    device = program._cuda_device_identity()
    try:
        from taichi_forge.interop import current_cuda_device_uuid

        device_uuid = current_cuda_device_uuid().hex()
    except (RuntimeError, ValueError) as error:
        raise RuntimeError(
            "qualified CUDA workload profiles require a stable device UUID"
        ) from error
    multiprocessor_count = int(device["multiprocessor_count"])
    if len(device_uuid) != 32 or multiprocessor_count <= 0:
        raise RuntimeError(
            "qualified CUDA workload profiles require exact UUID and SM identity"
        )
    return {
        "backend": "cuda",
        "device_name": str(device["device_name"]),
        "device_uuid": device_uuid,
        "multiprocessor_count": multiprocessor_count,
        "compute_capability": int(device["device_compute_capability"]),
        "codegen_compute_capability": int(device["codegen_compute_capability"]),
        "target": str(device["target"]),
        "driver_api_version": _ti_core.cuda_driver_api_version(),
        "driver_provider": _ti_core.cuda_driver_provider(),
        "forge_version": _ti_core.get_version_string(),
        "forge_commit": _ti_core.get_commit_hash(),
        "compiler": _compiler_scope(),
    }


def _cache_root_from_config(config):
    configured = str(config.offline_cache_file_path).strip()
    if not configured:
        return None
    return Path(configured) / "task_launch_tuning_v2"


def _candidate_blocks(tasks, max_threads, semantics=None, backend="cuda"):
    canonical = (
        _VULKAN_CANDIDATE_BLOCK_DIMS if backend == "vulkan" else _CANDIDATE_BLOCK_DIMS
    )
    if semantics is not None:
        from taichi_forge.lang._gpu_semantics import _GpuAvailability
        from taichi_forge.lang._gpu_semantics_tuning import (
            _WORKGROUP_DIMENSION,
            _derive_gpu_tuning_dimensions,
            _dimension_by_name,
        )

        dimensions = _derive_gpu_tuning_dimensions(
            semantics,
            max_threads=max_threads,
            canonical_workgroup_sizes=canonical,
            require_safe_serial_setup=False,
        )
        workgroup = _dimension_by_name(dimensions, _WORKGROUP_DIMENSION)
        if workgroup.status.availability != _GpuAvailability.PROVEN:
            return (), workgroup.status.reason
        return workgroup.legal_values, f"semantic {backend} workgroup candidates"
    if backend == "vulkan":
        return (), "Vulkan candidates require compiled shader semantics"
    ranges = tuple(task for task in tasks if task.task_type == "range_for")
    if len(ranges) != 1:
        return (), "requires exactly one parallel range task"
    task = ranges[0]
    if task.static_shared_bytes or task.dynamic_shared_bytes:
        return (), "shared-memory kernels require a resource-aware tuning stage"
    limit = min(1024, int(max_threads)) if int(max_threads) > 0 else 1024
    candidates = tuple(value for value in _CANDIDATE_BLOCK_DIMS if value <= limit)
    if not candidates:
        return (), "device thread limit rejects every canonical candidate"
    return candidates, "canonical CUDA block candidates"


def _task_scope(tasks):
    return [
        {
            "ordinal": index,
            "task_type": task.task_type,
            "logical_task_id": task.logical_task_id,
            "optimization_spec_id": task.optimization_spec_id,
        }
        for index, task in enumerate(tasks)
    ]


def _record_scope(
    kernel_key, tasks, hardware, workload_profile, artifact_identity=None
):
    scope = {
        "schema_version": _SCHEMA_VERSION,
        "kernel": {
            "cache_key": kernel_key,
            "tasks": _task_scope(tasks),
            "ir_identity": workload_profile.ir_identity,
            "ptx_identity": workload_profile.ptx_identity,
        },
        "workload": workload_profile.stable_scope,
        "hardware": hardware,
    }
    if hardware["backend"] == "vulkan":
        scope["kernel"].pop("ptx_identity")
        scope["kernel"]["spirv_identity"] = workload_profile.spirv_identity
        scope["kernel"]["compiled_artifact_identity"] = (
            list(artifact_identity) if artifact_identity else None
        )
    return scope


def _record_path(cache_root, record_id):
    return Path(cache_root) / "qualified" / f"{record_id}.json"


def _validated_record(path, scope, record_id):
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    checksum = record.pop("record_sha256", None)
    try:
        checksum_matches = checksum == _sha256_json(record)
    except (TypeError, ValueError):
        return None
    if not checksum_matches:
        return None
    if record.get("scope") != scope or record.get("record_id") != record_id:
        return None
    candidates = record.get("candidates")
    canonical = (
        _VULKAN_CANDIDATE_BLOCK_DIMS
        if scope["hardware"]["backend"] == "vulkan"
        else _CANDIDATE_BLOCK_DIMS
    )
    if (
        not candidates
        or not isinstance(candidates, list)
        or not all(type(item) is int and item in canonical for item in candidates)
    ):
        return None
    evidence = record.get("evidence")
    if not isinstance(evidence, dict):
        return None
    if record.get("admission") != "explicit_profile":
        return None
    if evidence.get("correctness_passed") is not True:
        return None
    if scope["hardware"]["backend"] == "vulkan" and (
        evidence.get("cost_metric") != "completion_time"
        or evidence.get("completion_waited") is not True
    ):
        return None
    independent_blocks = evidence.get("independent_abba_blocks")
    if (
        type(independent_blocks) is not int
        or independent_blocks < _MIN_INDEPENDENT_BLOCKS
    ):
        return None
    worst_ratio = evidence.get("worst_candidate_over_baseline_ratio")
    if (
        isinstance(worst_ratio, bool)
        or not isinstance(worst_ratio, (int, float))
        or not math.isfinite(float(worst_ratio))
        or not 0.0 < float(worst_ratio) < 1.0
    ):
        return None
    block_dim = record.get("block_dim")
    if type(block_dim) is not int or block_dim not in candidates:
        return None
    return record


class _TaskLaunchTuningCoordinator:
    def __init__(self):
        self._lock = threading.RLock()
        self._records = OrderedDict()
        self._observed_calls = OrderedDict()
        self._generation = 0

    @staticmethod
    def _store_bounded(mapping, key, value, limit):
        mapping[key] = value
        mapping.move_to_end(key)
        while len(mapping) > limit:
            mapping.popitem(last=False)

    @property
    def generation(self):
        return self._generation

    def invalidate(self, record_id):
        with self._lock:
            for key in tuple(self._records):
                if key[1] == record_id:
                    self._records.pop(key, None)
            self._generation += 1

    def observe_cached(self, decision):
        with self._lock:
            observed = self._observed_calls.get(decision.record_id, 0) + 1
            self._store_bounded(
                self._observed_calls,
                decision.record_id,
                observed,
                _MAX_OBSERVED_KERNELS,
            )
        if decision.status in ("cache_miss", "qualification_required"):
            status = (
                "qualification_required"
                if observed >= _HOT_CALL_THRESHOLD
                else "cache_miss"
            )
            reason = (
                "hot specialization has no qualified record; runtime measurement "
                "is forbidden without reset/replay and a correctness oracle"
                if status == "qualification_required"
                else "no exact qualified record; retaining compiler default"
            )
            return replace(
                decision,
                status=status,
                reason=reason,
                observed_calls=observed,
            )
        return replace(decision, observed_calls=observed)

    def resolve(
        self,
        *,
        kernel_key,
        tasks,
        config,
        observe,
        workload_profile=None,
        hardware=None,
        cache_root=None,
        semantics=None,
        artifact_identity=None,
    ):
        hardware = dict(_hardware_scope() if hardware is None else hardware)
        backend = hardware["backend"]
        max_threads = int(config.max_block_dim)
        if backend == "vulkan":
            max_threads = min(
                max_threads or 1024,
                hardware["max_workgroup_invocations"],
                hardware["max_workgroup_size_x"],
            )
        if tasks is None:
            candidates = None
            candidate_reason = "task manifest not materialized"
        else:
            candidates, candidate_reason = _candidate_blocks(
                tasks, max_threads, semantics, backend
            )
        hardware["runtime_max_block_dim"] = int(config.max_block_dim)
        hardware_tuple = tuple(
            sorted((key, _canonical_json(value)) for key, value in hardware.items())
        )
        if workload_profile is None:
            return _TaskLaunchTuningDecision(
                "profile_required",
                "qualified records require an explicit workload profile binding",
                "",
                kernel_key,
                hardware_tuple,
                () if candidates is None else candidates,
            )
        if not isinstance(workload_profile, _TaskLaunchWorkloadProfile):
            raise TypeError("workload_profile must be a _TaskLaunchWorkloadProfile")
        if tasks is None:
            return _TaskLaunchTuningDecision(
                "manifest_required",
                "an exact workload profile requires a materialized task manifest",
                "",
                kernel_key,
                hardware_tuple,
                (),
                workload_profile_id=workload_profile.identity,
            )
        if backend == "vulkan" and (
            not workload_profile.spirv_identity or not artifact_identity
        ):
            raise ValueError(
                "Vulkan qualification requires SPIR-V profile and compiled artifact identities"
            )
        if backend == "cuda" and not workload_profile.ptx_identity:
            raise ValueError("CUDA qualification requires a PTX profile identity")
        scope = _record_scope(
            kernel_key, tasks, hardware, workload_profile, artifact_identity
        )
        scope_json = _canonical_json(scope)
        record_id = _sha256_json(scope)
        root = _cache_root_from_config(config) if cache_root is None else cache_root
        cache_key = (None if root is None else str(Path(root).resolve()), record_id)

        with self._lock:
            observed = self._observed_calls.get(record_id, 0)
            if observe:
                observed += 1
                self._store_bounded(
                    self._observed_calls,
                    record_id,
                    observed,
                    _MAX_OBSERVED_KERNELS,
                )
            record = self._records.get(cache_key, ...)
            if record is not ...:
                self._records.move_to_end(cache_key)

        if candidates == ():
            return _TaskLaunchTuningDecision(
                "ineligible",
                candidate_reason,
                record_id,
                kernel_key,
                hardware_tuple,
                candidates,
                observed_calls=observed,
                workload_profile_id=workload_profile.identity,
                record_scope_json=scope_json,
            )

        if record is ...:
            path = None if root is None else _record_path(root, record_id)
            record = None if path is None else _validated_record(path, scope, record_id)
            with self._lock:
                self._store_bounded(
                    self._records,
                    cache_key,
                    record,
                    _MAX_RECORD_CACHE_ENTRIES,
                )

        if record is not None and tuple(record["candidates"]) != candidates:
            record = None
        if record is not None:
            record_candidates = tuple(int(value) for value in record["candidates"])
            return _TaskLaunchTuningDecision(
                "qualified",
                "exact qualified record matched workload, kernel, device, driver, and compiler",
                record_id,
                kernel_key,
                hardware_tuple,
                record_candidates,
                block_dim=int(record["block_dim"]),
                observed_calls=observed,
                workload_profile_id=workload_profile.identity,
                record_scope_json=scope_json,
            )
        if observed >= _HOT_CALL_THRESHOLD:
            status = "qualification_required"
            reason = (
                "hot specialization has no qualified record; runtime measurement "
                "is forbidden without reset/replay and a correctness oracle"
            )
        else:
            status = "cache_miss"
            reason = "no exact qualified record; retaining compiler default"
        return _TaskLaunchTuningDecision(
            status,
            reason,
            record_id,
            kernel_key,
            hardware_tuple,
            candidates,
            observed_calls=observed,
            workload_profile_id=workload_profile.identity,
            record_scope_json=scope_json,
        )


_coordinator = _TaskLaunchTuningCoordinator()


def _publish_qualified_record(
    *,
    decision,
    cache_root,
    block_dim,
    evidence,
    admission="explicit_profile",
):
    if block_dim not in decision.candidates:
        raise ValueError("qualified block_dim must be one of the decision candidates")
    if not decision.record_scope_json or not decision.workload_profile_id:
        raise ValueError("decision is not bound to an explicit workload profile")
    scope = json.loads(decision.record_scope_json)
    if _sha256_json(scope) != decision.record_id:
        raise ValueError("decision scope does not match its record id")
    record = {
        "schema_version": _SCHEMA_VERSION,
        "record_id": decision.record_id,
        "scope": scope,
        "candidates": list(decision.candidates),
        "admission": admission,
        "block_dim": int(block_dim),
        "evidence": dict(evidence),
    }
    record["record_sha256"] = _sha256_json(record)
    destination = _record_path(cache_root, decision.record_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json(record) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f"{decision.record_id}.",
        suffix=".tmp",
        delete=False,
    ) as output:
        temporary = Path(output.name)
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    try:
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    _coordinator.invalidate(decision.record_id)
    return destination


def _bind_workload_profile(binding, profile):
    if not isinstance(profile, _TaskLaunchWorkloadProfile):
        raise TypeError("profile must be a _TaskLaunchWorkloadProfile")
    return binding._with_workload_profile(profile)


__all__ = []
