"""Shared evidence contracts for Forge-owned opaque recipe searches."""

from dataclasses import dataclass
import hashlib
import json
import math
import statistics


_MAX_QUALIFICATION_CANDIDATES = 32
_MIN_QUALIFICATION_BLOCKS = 10


def _compileiq_import_error():
    return RuntimeError(
        "CompileIQ is an optional external dependency; install the reviewed "
        "modified fork before requesting an offline search space"
    )


def _canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


@dataclass(frozen=True)
class _CompileIQPairedTrial:
    variant_id: str
    block: int
    order: tuple[str, str]


@dataclass(frozen=True)
class _CompileIQCandidateEvidence:
    variant_id: str
    ratios: tuple[float, ...]
    median_ratio: float
    worst_ratio: float
    best_ratio: float
    ratio_cv: float

    @property
    def worst_positive(self):
        return self.worst_ratio < 1.0


@dataclass(frozen=True)
class _CompileIQSearchStage:
    stage_id: str
    candidate_kind: str
    candidate_ids: tuple[str, ...]
    blocks: int

    def __post_init__(self):
        if not self.stage_id or "\n" in self.stage_id:
            raise ValueError("stage_id must be a non-empty single-line string")
        if self.candidate_kind != "qualification":
            raise ValueError("only independent qualification stages are supported")
        if not self.candidate_ids or len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("qualification candidates must be non-empty and unique")
        if len(self.candidate_ids) > _MAX_QUALIFICATION_CANDIDATES:
            raise ValueError("qualification may contain at most 32 candidates")
        if any(
            not isinstance(candidate_id, str) or not candidate_id or "\n" in candidate_id
            for candidate_id in self.candidate_ids
        ):
            raise ValueError("candidate IDs must be non-empty single-line strings")
        _balanced_paired_schedule((), blocks=self.blocks)
        if self.blocks < _MIN_QUALIFICATION_BLOCKS:
            raise ValueError("final qualification requires at least 10 blocks")

    @property
    def schedule(self):
        return _balanced_paired_schedule(
            self.candidate_ids,
            blocks=self.blocks,
        )

    def manifest(self):
        return {
            "stage_id": self.stage_id,
            "candidate_kind": self.candidate_kind,
            "candidate_ids": self.candidate_ids,
            "blocks": self.blocks,
            "execution": "fresh_process_balanced_ab_ba",
            "selection": "worst_ratio_then_median",
        }


@dataclass(frozen=True)
class _CompileIQWinnerScope:
    final_candidate_id: str
    forge_specialization_id: str
    workload_profile_id: str
    shape_scope_id: str
    replay_scope_id: str
    runtime_scope_id: str
    compiler_scope_id: str
    provider_scope_id: str
    variant_manifest_id: str

    def __post_init__(self):
        for name, value in self.__dict__.items():
            if not isinstance(value, str) or not value or "\n" in value:
                raise ValueError(f"{name} must be a non-empty single-line string")

    @property
    def stable_scope(self):
        return dict(self.__dict__)

    @property
    def identity(self):
        digest = hashlib.sha256(_canonical_json(self.stable_scope).encode("utf-8")).hexdigest()
        return f"ciqs1:{digest}"


@dataclass(frozen=True)
class _CompileIQFinalCandidate:
    forge_object_kind: str
    forge_object_id: str
    provider_candidate_id: str = "baseline"

    def __post_init__(self):
        if self.forge_object_kind not in (
            "executable_spec",
            "primitive_provider_recipe",
        ):
            raise ValueError("unsupported Forge final candidate object kind")
        for name, value in self.__dict__.items():
            if not isinstance(value, str) or not value or "\n" in value:
                raise ValueError(f"{name} must be a non-empty single-line string")

    @property
    def identity(self):
        digest = hashlib.sha256(_canonical_json(self.__dict__).encode("utf-8")).hexdigest()
        return f"ciqc1:{digest}"


@dataclass(frozen=True)
class _CompileIQQualificationDecision:
    status: str
    reason: str
    selected_candidate_id: str | None
    selected_forge_object_kind: str | None
    selected_forge_object_id: str | None
    selected_provider_candidate_id: str | None
    scope_id: str
    evidence: tuple[_CompileIQCandidateEvidence, ...]

    @property
    def admitted(self):
        return self.status == "qualified"


def _balanced_paired_schedule(candidate_ids, *, blocks=2):
    if isinstance(blocks, bool) or not isinstance(blocks, int):
        raise TypeError("blocks must be an integer")
    if blocks < 2 or blocks % 2:
        raise ValueError("blocks must be an even integer >= 2")
    return tuple(
        _CompileIQPairedTrial(
            variant_id=candidate_id,
            block=block,
            order=(("baseline", "candidate") if block % 2 == 0 else ("candidate", "baseline")),
        )
        for candidate_id in candidate_ids
        for block in range(blocks)
    )


def _rank_complete_paired_evidence(
    measurements,
    candidate_ids,
    *,
    blocks=2,
    candidate_kind="candidate",
    collection_name="candidates",
):
    _balanced_paired_schedule((), blocks=blocks)
    candidate_ids = tuple(candidate_ids)
    if not isinstance(measurements, dict):
        raise TypeError("measurements must map candidate IDs to paired ratios")
    missing = tuple(candidate_id for candidate_id in candidate_ids if candidate_id not in measurements)
    extra = tuple(candidate_id for candidate_id in measurements if candidate_id not in candidate_ids)
    if missing or extra:
        raise ValueError(f"paired measurements do not match {collection_name}; " f"missing={missing}, extra={extra}")

    evidence = []
    for candidate_id in candidate_ids:
        raw_ratios = measurements[candidate_id]
        if not isinstance(raw_ratios, (tuple, list)):
            raise TypeError("paired ratios must be a tuple or list")
        if len(raw_ratios) != blocks:
            raise ValueError(f"{candidate_kind} {candidate_id!r} requires exactly " f"{blocks} paired ratios")
        ratios = tuple(float(value) for value in raw_ratios)
        if any(not math.isfinite(value) or value <= 0.0 for value in ratios):
            raise ValueError("paired ratios must be finite and positive")
        evidence.append(
            _CompileIQCandidateEvidence(
                variant_id=candidate_id,
                ratios=ratios,
                median_ratio=float(statistics.median(ratios)),
                worst_ratio=max(ratios),
                best_ratio=min(ratios),
                ratio_cv=(float(statistics.stdev(ratios) / statistics.mean(ratios)) if len(ratios) > 1 else 0.0),
            )
        )
    return tuple(
        sorted(
            evidence,
            key=lambda item: (
                item.worst_ratio,
                item.median_ratio,
                item.variant_id,
            ),
        )
    )


def _qualify_complete_paired_candidates(
    measurements,
    finalists,
    *,
    scopes,
    correctness,
    memory_stable,
    blocks,
):
    finalists = tuple(finalists)
    candidate_ids = tuple(finalist.identity for finalist in finalists)
    evidence = _rank_complete_paired_evidence(
        measurements,
        candidate_ids,
        blocks=blocks,
        candidate_kind="qualification candidate",
        collection_name="independent qualification",
    )
    expected = set(candidate_ids)
    if not isinstance(scopes, dict) or set(scopes) != expected:
        raise ValueError("scopes must exactly match qualification candidates")
    if any(not isinstance(scope, _CompileIQWinnerScope) for scope in scopes.values()):
        raise TypeError("scope values must be _CompileIQWinnerScope values")
    if any(scope.final_candidate_id != candidate_id for candidate_id, scope in scopes.items()):
        raise ValueError("each scope must bind its exact final candidate ID")
    for name, gate in (
        ("correctness", correctness),
        ("memory_stable", memory_stable),
    ):
        if not isinstance(gate, dict) or set(gate) != expected:
            raise ValueError(f"{name} must exactly match qualification candidates")
        if any(type(value) is not bool for value in gate.values()):
            raise TypeError(f"{name} values must be booleans")
    eligible = tuple(item for item in evidence if correctness[item.variant_id] and memory_stable[item.variant_id])
    if not eligible:
        return _CompileIQQualificationDecision(
            status="keep_baseline",
            reason="no candidate passed correctness and memory gates",
            selected_candidate_id=None,
            selected_forge_object_kind=None,
            selected_forge_object_id=None,
            selected_provider_candidate_id=None,
            scope_id="",
            evidence=evidence,
        )
    selected = eligible[0]
    if not selected.worst_positive:
        return _CompileIQQualificationDecision(
            status="keep_baseline",
            reason="best valid candidate failed the worst-positive gate",
            selected_candidate_id=None,
            selected_forge_object_kind=None,
            selected_forge_object_id=None,
            selected_provider_candidate_id=None,
            scope_id="",
            evidence=evidence,
        )
    selected_candidate = next(finalist for finalist in finalists if finalist.identity == selected.variant_id)
    return _CompileIQQualificationDecision(
        status="qualified",
        reason=("exact candidate passed correctness, memory, and " "worst-positive gates"),
        selected_candidate_id=selected.variant_id,
        selected_forge_object_kind=selected_candidate.forge_object_kind,
        selected_forge_object_id=selected_candidate.forge_object_id,
        selected_provider_candidate_id=(selected_candidate.provider_candidate_id),
        scope_id=scopes[selected.variant_id].identity,
        evidence=evidence,
    )


__all__ = [
    "_CompileIQCandidateEvidence",
    "_CompileIQFinalCandidate",
    "_CompileIQPairedTrial",
    "_CompileIQQualificationDecision",
    "_CompileIQSearchStage",
    "_CompileIQWinnerScope",
    "_balanced_paired_schedule",
    "_compileiq_import_error",
    "_qualify_complete_paired_candidates",
    "_rank_complete_paired_evidence",
]
