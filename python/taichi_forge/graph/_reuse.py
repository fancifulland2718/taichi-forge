"""Portable identities and artifacts for complete-Graph recipe reuse."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass


def _canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _identity(prefix, value):
    return prefix + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _normalize_json(value, *, path="facts"):
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, (list, tuple)):
        return tuple(
            _normalize_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, Mapping):
        normalized = {}
        keys = tuple(value)
        if any(not isinstance(key, str) or not key for key in keys):
            raise ValueError(f"{path} keys must be non-empty strings")
        for key in sorted(keys):
            normalized[key] = _normalize_json(value[key], path=f"{path}.{key}")
        return normalized
    raise TypeError(f"{path} must contain only canonical JSON-safe values")


class _CanonicalFactSet:
    SCHEMA = ""
    ID_PREFIX = ""
    ID_FIELD = ""

    __slots__ = ("_facts_json", "_identity")

    def __init__(self, facts):
        if not isinstance(facts, Mapping):
            raise TypeError("Graph reuse facts must be a mapping")
        normalized = _normalize_json(facts)
        self._facts_json = _canonical_json(normalized)
        self._identity = _identity(
            self.ID_PREFIX,
            {
                "schema": self.SCHEMA,
                "facts": normalized,
            },
        )

    @property
    def facts(self):
        return json.loads(self._facts_json)

    @property
    def identity(self):
        return self._identity

    def to_dict(self):
        return {
            "schema": self.SCHEMA,
            self.ID_FIELD: self.identity,
            "facts": self.facts,
        }

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, Mapping):
            raise TypeError(f"{cls.__name__} must be restored from a mapping")
        if value.get("schema") != cls.SCHEMA:
            raise ValueError(f"{cls.__name__} schema is unsupported")
        restored = cls(value.get("facts"))
        if value.get(cls.ID_FIELD) != restored.identity:
            raise ValueError(f"{cls.__name__} identity mismatch")
        return restored

    def __eq__(self, other):
        return type(self) is type(other) and self.identity == other.identity

    def __hash__(self):
        return hash((type(self), self.identity))

    def __repr__(self):
        return f"{type(self).__name__}(facts={self.facts!r})"


class GraphWorkloadContext(_CanonicalFactSet):
    """Caller-owned facts that identify one workload distribution and shape."""

    SCHEMA = "taichi_forge.graph_workload_context.v1"
    ID_PREFIX = "graph-workload-context-v1:"
    ID_FIELD = "workload_context_id"

    @property
    def workload_context_id(self):
        return self.identity


class GraphEvaluationContract(_CanonicalFactSet):
    """Caller-owned measurement, synchronization, and correctness contract."""

    SCHEMA = "taichi_forge.graph_evaluation_contract.v1"
    ID_PREFIX = "graph-evaluation-contract-v1:"
    ID_FIELD = "evaluation_contract_id"

    @property
    def evaluation_contract_id(self):
        return self.identity


class GraphBackendEnvironment(_CanonicalFactSet):
    """Caller-owned hardware, driver, runtime, and backend environment facts."""

    SCHEMA = "taichi_forge.graph_backend_environment.v1"
    ID_PREFIX = "graph-backend-environment-v1:"
    ID_FIELD = "backend_environment_id"

    @property
    def backend_environment_id(self):
        return self.identity


class _CanonicalArtifact(Mapping):
    SCHEMA = ""
    ID_PREFIX = ""
    ID_FIELD = ""

    __slots__ = ("_payload_json",)

    def __init__(self, payload):
        normalized = _normalize_json(payload, path=type(self).__name__)
        self._validate(normalized)
        self._payload_json = _canonical_json(normalized)

    @classmethod
    def create(cls, **sections):
        payload = {
            "schema": cls.SCHEMA,
            **sections,
        }
        payload[cls.ID_FIELD] = _identity(cls.ID_PREFIX, payload)
        return cls(payload)

    @classmethod
    def from_dict(cls, value):
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError(f"{cls.__name__} must be restored from a mapping")
        return cls(dict(value))

    def _validate(self, payload):
        if payload.get("schema") != self.SCHEMA:
            raise ValueError(f"{type(self).__name__} schema is unsupported")
        artifact_id = payload.get(self.ID_FIELD)
        if not isinstance(artifact_id, str) or not artifact_id:
            raise ValueError(f"{type(self).__name__} has no identity")
        content = dict(payload)
        content.pop(self.ID_FIELD)
        if artifact_id != _identity(self.ID_PREFIX, content):
            raise ValueError(f"{type(self).__name__} identity mismatch")

    @property
    def identity(self):
        return self[self.ID_FIELD]

    def to_dict(self):
        return json.loads(self._payload_json)

    def as_dict(self):
        return self.to_dict()

    def __getitem__(self, key):
        return self.to_dict()[key]

    def __iter__(self):
        return iter(self.to_dict())

    def __len__(self):
        return len(self.to_dict())


class GraphRecipeSearchCheckpointV1(_CanonicalArtifact):
    """Serializable Forge generation plus modified-CompileIQ observations."""

    SCHEMA = "taichi_forge.graph_recipe_search_checkpoint.v1"
    ID_PREFIX = "graph-recipe-search-checkpoint-v1:"
    ID_FIELD = "checkpoint_id"

    def _validate(self, payload):
        super()._validate(payload)
        for name in ("contract", "generation", "compileiq_checkpoint"):
            if not isinstance(payload.get(name), Mapping):
                raise ValueError(f"Graph recipe checkpoint requires {name}")

    @property
    def checkpoint_id(self):
        return self.identity

    @property
    def contract(self):
        return self["contract"]

    @property
    def generation(self):
        return self["generation"]

    @property
    def compileiq_checkpoint(self):
        return self["compileiq_checkpoint"]


class GraphRecipeSelectionArtifact(_CanonicalArtifact):
    """Portable stable-key recipe selection without an executable or callback."""

    SCHEMA = "taichi_forge.graph_recipe_selection_artifact.v1"
    ID_PREFIX = "graph-recipe-selection-artifact-v1:"
    ID_FIELD = "artifact_id"

    def _validate(self, payload):
        super()._validate(payload)
        for name in ("structure", "recipe_manifest", "evidence"):
            if not isinstance(payload.get(name), Mapping):
                raise ValueError(f"Graph recipe selection artifact requires {name}")

    @property
    def artifact_id(self):
        return self.identity

    @property
    def structure(self):
        return self["structure"]

    @property
    def recipe_manifest(self):
        return self["recipe_manifest"]

    @property
    def evidence(self):
        return self["evidence"]


class GraphRecipeReuseError(ValueError):
    """Structured failure to reconstruct one portable recipe artifact."""

    def __init__(self, message, *, error_key):
        super().__init__(message)
        if not isinstance(error_key, str) or not error_key:
            raise ValueError("Graph recipe reuse error_key must be non-empty text")
        self.error_key = error_key

    def to_dict(self):
        return {
            "error_key": self.error_key,
            "message": str(self),
        }


@dataclass(frozen=True)
class GraphRecipeApplicabilityReport:
    """Separate structural reconstruction from historical evidence reuse."""

    artifact_id: str
    status: str
    structurally_resolvable: bool
    evidence_applicable: bool
    drift_fields: tuple[str, ...]
    reason: str

    def __post_init__(self):
        allowed = {
            "applicable",
            "structurally_resolvable_evidence_drift",
            "contract_drift",
            "recipe_unavailable",
            "provider_unavailable",
            "backend_unavailable",
        }
        if self.status not in allowed:
            raise ValueError("Graph recipe applicability status is unsupported")
        if not isinstance(self.artifact_id, str) or not self.artifact_id:
            raise ValueError("Graph recipe applicability requires an artifact ID")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("Graph recipe applicability requires a reason")
        object.__setattr__(
            self,
            "drift_fields",
            tuple(sorted(set(self.drift_fields))),
        )

    def to_dict(self):
        return {
            "schema": "taichi_forge.graph_recipe_applicability_report.v1",
            "artifact_id": self.artifact_id,
            "status": self.status,
            "structurally_resolvable": self.structurally_resolvable,
            "evidence_applicable": self.evidence_applicable,
            "drift_fields": self.drift_fields,
            "reason": self.reason,
        }


__all__ = [
    "GraphBackendEnvironment",
    "GraphEvaluationContract",
    "GraphRecipeApplicabilityReport",
    "GraphRecipeReuseError",
    "GraphRecipeSearchCheckpointV1",
    "GraphRecipeSelectionArtifact",
    "GraphWorkloadContext",
]
