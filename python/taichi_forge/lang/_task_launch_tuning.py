"""Internal qualification-gated launch specialization selection.

This module deliberately does not benchmark kernels.  A physics workload may
mutate fields, consume queues, or depend on replay state, so ordinary execution
cannot safely call it several times.  Qualification tooling publishes an
immutable record after correctness and fresh-process AB/BA gates; the runtime
only validates and consumes that record.
"""

from collections import OrderedDict
from dataclasses import dataclass, replace
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


_SCHEMA_VERSION = 1
_CANDIDATE_BLOCK_DIMS = (64, 128, 256, 512)
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
    result = {"mode": mode}
    if mode != "external":
        return result

    ptxas = os.environ.get("TI_CUDA_PTXAS_PATH")
    if not ptxas:
        ptxas = shutil.which("ptxas") or shutil.which("ptxas.exe")
    result["ptxas"] = _optional_file_identity(ptxas)
    result["direct_acf"] = _optional_file_identity(
        os.environ.get("TI_CUDA_PTXAS_ACF_PATH")
    )
    result["compileiq_worker"] = _optional_file_identity(
        os.environ.get("TI_CUDA_COMPILEIQ_WORKER")
    )
    result["compileiq_python"] = _optional_file_identity(
        os.environ.get("TI_CUDA_COMPILEIQ_PYTHON")
    )
    return result


def _hardware_scope():
    program = impl.get_runtime().prog
    device = program._cuda_device_identity()
    return {
        "backend": "cuda",
        "device_name": str(device["device_name"]),
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
    return Path(configured) / "task_launch_tuning_v1"


def _candidate_blocks(tasks, max_threads):
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


def _record_scope(kernel_key, hardware):
    return {
        "schema_version": _SCHEMA_VERSION,
        "kernel_key": kernel_key,
        "hardware": hardware,
    }


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
    if not candidates or not isinstance(candidates, list) or not all(
        type(item) is int and item in _CANDIDATE_BLOCK_DIMS
        for item in candidates
    ):
        return None
    evidence = record.get("evidence")
    if not isinstance(evidence, dict):
        return None
    if record.get("admission") != "auto":
        return None
    if evidence.get("correctness_passed") is not True:
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
            self._records.pop(record_id, None)
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
        hardware=None,
        cache_root=None,
    ):
        if tasks is None:
            candidates = None
            candidate_reason = "task manifest not materialized"
        else:
            candidates, candidate_reason = _candidate_blocks(
                tasks, config.max_block_dim
            )
        hardware = dict(_hardware_scope() if hardware is None else hardware)
        hardware["runtime_max_block_dim"] = int(config.max_block_dim)
        hardware_tuple = tuple(
            sorted((key, _canonical_json(value)) for key, value in hardware.items())
        )
        scope = _record_scope(kernel_key, hardware)
        record_id = _sha256_json(scope)

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
            record = self._records.get(record_id, ...)
            if record is not ...:
                self._records.move_to_end(record_id)

        if candidates == ():
            return _TaskLaunchTuningDecision(
                "ineligible",
                candidate_reason,
                record_id,
                kernel_key,
                hardware_tuple,
                candidates,
                observed_calls=observed,
            )

        root = _cache_root_from_config(config) if cache_root is None else cache_root
        if record is ...:
            path = None if root is None else _record_path(root, record_id)
            record = None if path is None else _validated_record(path, scope, record_id)
            with self._lock:
                self._store_bounded(
                    self._records,
                    record_id,
                    record,
                    _MAX_RECORD_CACHE_ENTRIES,
                )

        if record is not None:
            record_candidates = tuple(int(value) for value in record["candidates"])
            return _TaskLaunchTuningDecision(
                "qualified",
                "exact qualified record matched kernel, device, driver, and compiler",
                record_id,
                kernel_key,
                hardware_tuple,
                record_candidates,
                block_dim=int(record["block_dim"]),
                observed_calls=observed,
            )

        if candidates is None:
            return _TaskLaunchTuningDecision(
                "manifest_required",
                "no exact qualified record; task manifest is required for qualification",
                record_id,
                kernel_key,
                hardware_tuple,
                (),
                observed_calls=observed,
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
        )


_coordinator = _TaskLaunchTuningCoordinator()


def _publish_qualified_record(
    *,
    decision,
    cache_root,
    block_dim,
    evidence,
    admission="auto",
):
    if block_dim not in decision.candidates:
        raise ValueError("qualified block_dim must be one of the decision candidates")
    hardware = {
        key: json.loads(value) for key, value in decision.hardware_scope
    }
    scope = _record_scope(decision.kernel_key, hardware)
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


__all__ = []
