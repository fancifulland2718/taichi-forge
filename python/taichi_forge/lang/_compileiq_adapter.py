"""Private, optional CompileIQ boundary for Forge kernel variants."""

from dataclasses import dataclass
import hashlib
import json
import math
import re
import statistics
from types import MappingProxyType


_FORGE_VARIANT_PARAMETER = "forge_variant"
_FORGE_VARIANT_PARAMETER_PREFIX = f"{_FORGE_VARIANT_PARAMETER}_"
_SEARCH_STAGES = ("structural", "launch", "full")
_PARAMETER_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_MAX_EXHAUSTIVE_STAGE_CANDIDATES = 32
_MIN_QUALIFICATION_BLOCKS = 10


def _compileiq_import_error():
    return RuntimeError(
        "CompileIQ is an optional external dependency; install it in the "
        "qualification environment before requesting a CompileIQ search space"
    )


@dataclass(frozen=True)
class _CompileIQVariantSelection:
    variant_id: str
    compilation_id: str


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
    compilation_id: str = ""
    forge_variant_id: str = ""

    def __post_init__(self):
        if not self.stage_id or "\n" in self.stage_id:
            raise ValueError("stage_id must be a non-empty single-line string")
        if self.candidate_kind not in (
            "forge_structural",
            "forge_launch",
            "ptxas_control",
            "qualification",
        ):
            raise ValueError("unsupported CompileIQ stage candidate kind")
        if not self.candidate_ids or len(set(self.candidate_ids)) != len(
            self.candidate_ids
        ):
            raise ValueError("stage candidates must be non-empty and unique")
        if len(self.candidate_ids) > _MAX_EXHAUSTIVE_STAGE_CANDIDATES:
            raise ValueError("an exhaustive stage may contain at most 32 candidates")
        if any(
            not isinstance(candidate_id, str)
            or not candidate_id
            or "\n" in candidate_id
            for candidate_id in self.candidate_ids
        ):
            raise ValueError("candidate IDs must be non-empty single-line strings")
        _balanced_paired_schedule((), blocks=self.blocks)
        if self.candidate_kind == "forge_launch" and not self.compilation_id:
            raise ValueError("a Forge launch stage requires compilation_id")
        if self.candidate_kind == "ptxas_control" and not self.forge_variant_id:
            raise ValueError("a PTXAS stage requires an exact Forge variant")
        if (
            self.candidate_kind == "qualification"
            and self.blocks < _MIN_QUALIFICATION_BLOCKS
        ):
            raise ValueError("final qualification requires at least 10 blocks")

    @property
    def schedule(self):
        return _balanced_paired_schedule(self.candidate_ids, blocks=self.blocks)

    def manifest(self):
        return {
            "stage_id": self.stage_id,
            "candidate_kind": self.candidate_kind,
            "candidate_ids": self.candidate_ids,
            "blocks": self.blocks,
            "compilation_id": self.compilation_id or None,
            "forge_variant_id": self.forge_variant_id or None,
            "execution": "fresh_process_balanced_ab_ba",
            "selection": "worst_ratio_then_median",
        }


def _canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


@dataclass(frozen=True)
class _CompileIQWinnerScope:
    final_candidate_id: str
    kernel_specialization_id: str
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
        digest = hashlib.sha256(
            _canonical_json(self.stable_scope).encode("utf-8")
        ).hexdigest()
        return f"ciqs1:{digest}"


@dataclass(frozen=True)
class _CompileIQFinalCandidate:
    forge_variant_id: str
    provider_candidate_id: str = "baseline"

    def __post_init__(self):
        for name, value in self.__dict__.items():
            if not isinstance(value, str) or not value or "\n" in value:
                raise ValueError(f"{name} must be a non-empty single-line string")

    @property
    def identity(self):
        digest = hashlib.sha256(
            _canonical_json(self.__dict__).encode("utf-8")
        ).hexdigest()
        return f"ciqc1:{digest}"


@dataclass(frozen=True)
class _CompileIQQualificationDecision:
    status: str
    reason: str
    selected_candidate_id: str | None
    selected_forge_variant_id: str | None
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
            order=(
                ("baseline", "candidate")
                if block % 2 == 0
                else ("candidate", "baseline")
            ),
        )
        for candidate_id in candidate_ids
        for block in range(blocks)
    )


def _rank_complete_paired_evidence(
    measurements,
    candidate_ids,
    *,
    blocks=2,
    candidate_kind="variant",
    collection_name="stage candidates",
):
    _balanced_paired_schedule((), blocks=blocks)
    candidate_ids = tuple(candidate_ids)
    if not isinstance(measurements, dict):
        raise TypeError("measurements must map candidate IDs to paired ratios")
    missing = tuple(
        candidate_id
        for candidate_id in candidate_ids
        if candidate_id not in measurements
    )
    extra = tuple(
        candidate_id
        for candidate_id in measurements
        if candidate_id not in candidate_ids
    )
    if missing or extra:
        raise ValueError(
            f"paired measurements do not match {collection_name}; "
            f"missing={missing}, extra={extra}"
        )

    evidence = []
    for candidate_id in candidate_ids:
        raw_ratios = measurements[candidate_id]
        if not isinstance(raw_ratios, (tuple, list)):
            raise TypeError("paired ratios must be a tuple or list")
        if len(raw_ratios) != blocks:
            raise ValueError(
                f"{candidate_kind} {candidate_id!r} requires exactly "
                f"{blocks} paired ratios"
            )
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
                ratio_cv=(
                    float(statistics.stdev(ratios) / statistics.mean(ratios))
                    if len(ratios) > 1
                    else 0.0
                ),
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


class _CompileIQVariantAdapter:
    """Expose stable Forge variant IDs without importing CompileIQ at startup.

    The adapter deliberately keeps Forge structural and PTXAS searches as
    separate stages.  This avoids coupling the wheel to CompileIQ and also
    works on CompileIQ releases whose Windows core rejects mixed search-space
    lists.
    """

    def __init__(self, session, *, parameter=_FORGE_VARIANT_PARAMETER):
        if not isinstance(parameter, str):
            raise TypeError("CompileIQ parameter must be a string")
        if not _PARAMETER_PATTERN.fullmatch(parameter):
            raise ValueError(
                "CompileIQ parameter must start with a letter and contain only "
                "letters, digits, and underscores"
            )
        variants = tuple(
            session.variant(variant_id) for variant_id in session.variant_ids()
        )
        if not variants:
            raise ValueError("a CompileIQ adapter requires at least one Forge variant")
        groups = tuple(session.compilation_groups)
        if not groups:
            raise ValueError("a CompileIQ adapter requires compilation groups")

        self._session = session
        self._parameter = parameter
        self._variants = MappingProxyType(
            {variant.variant_id: variant for variant in variants}
        )
        self._groups = MappingProxyType(
            {group.compilation_id: group for group in groups}
        )
        self._structural_ids = tuple(
            group.representative_variant_id for group in groups
        )

    def variant_ids(self, stage="structural", *, compilation_id=None):
        if stage not in _SEARCH_STAGES:
            raise ValueError("stage must be 'structural', 'launch', or 'full'")
        if stage == "structural":
            if compilation_id is not None:
                raise ValueError("structural search does not accept compilation_id")
            return self._structural_ids
        if stage == "full":
            if compilation_id is not None:
                raise ValueError("full search does not accept compilation_id")
            return tuple(self._variants)
        if compilation_id is None:
            raise ValueError("launch search requires compilation_id")
        try:
            return self._groups[compilation_id].variant_ids
        except KeyError as error:
            raise KeyError(
                f"unknown Forge compilation group {compilation_id!r}"
            ) from error

    def search_space(self, stage="structural", *, compilation_id=None):
        """Build a raw CompileIQ choice space through a lazy import."""

        if stage == "full":
            raise ValueError(
                "full Forge Cartesian search is disabled; use staged_plan() "
                "and exhaust structural and launch candidates separately"
            )

        try:
            from compileiq.search_spaces.base import choice
        except ImportError as error:
            raise _compileiq_import_error() from error
        return {
            self._parameter: choice(
                self.variant_ids(stage, compilation_id=compilation_id)
            )
        }

    def paired_schedule(
        self,
        stage="structural",
        *,
        compilation_id=None,
        blocks=2,
    ):
        """Return a deterministic AB/BA plan for a finite Forge stage.

        Forge variant stages are deliberately enumerated instead of delegated
        to CompileIQ's genetic core.  This prevents a noisy first-generation
        sample from permanently eliminating a legal structural or launch
        candidate.
        """

        return _balanced_paired_schedule(
            self.variant_ids(stage, compilation_id=compilation_id),
            blocks=blocks,
        )

    def rank_paired(
        self,
        measurements,
        stage="structural",
        *,
        compilation_id=None,
        blocks=2,
    ):
        """Rank complete paired evidence by worst ratio, then median ratio."""

        return _rank_complete_paired_evidence(
            measurements,
            self.variant_ids(stage, compilation_id=compilation_id),
            blocks=blocks,
        )

    @staticmethod
    def ptxas_search_space(*, version="13.3", variant="default", tag="latest"):
        """Create a separate optional PTXAS provider through a lazy import."""

        try:
            from compileiq.search_spaces.compilers import PtxasSearchSpace
        except ImportError as error:
            raise _compileiq_import_error() from error
        return PtxasSearchSpace(version=version, variant=variant, tag=tag)

    def select(self, parameters):
        if not isinstance(parameters, dict):
            raise TypeError("CompileIQ parameters must be a dictionary")
        if self._parameter not in parameters:
            raise KeyError(f"CompileIQ parameters require {self._parameter!r}")
        variant_id = parameters[self._parameter]
        if not isinstance(variant_id, str):
            raise TypeError(f"{self._parameter} must be a string")
        try:
            variant = self._variants[variant_id]
        except KeyError as error:
            raise KeyError(f"unknown Forge kernel variant {variant_id!r}") from error
        return _CompileIQVariantSelection(
            variant_id=variant_id,
            compilation_id=variant.compilation_id,
        )

    def bind(self, parameters):
        return self._session.bind(self.select(parameters).variant_id)

    def manifest(self):
        """Return a dependency-free, serializable worker/replay manifest."""

        from taichi_forge.lang._gpu_semantics_tuning import (
            _gpu_tuning_dimension_manifest,
        )

        return {
            "schema_version": 2,
            "parameter": self._parameter,
            "structural_variant_ids": self._structural_ids,
            "dimensions": tuple(
                _gpu_tuning_dimension_manifest(dimension)
                for dimension in getattr(self._session, "dimensions", ())
            ),
            "variants": tuple(
                {
                    "variant_id": variant.variant_id,
                    "compilation_id": variant.compilation_id,
                    "spec": variant.spec.stable_payload,
                    "selections": tuple(getattr(variant, "selections", ())),
                }
                for variant in self._variants.values()
            ),
        }

    def staged_plan(
        self,
        *,
        structural_blocks=4,
        launch_blocks=4,
        qualification_blocks=_MIN_QUALIFICATION_BLOCKS,
        structural_shortlist=4,
    ):
        return _CompileIQStagedSearchPlan(
            self,
            structural_blocks=structural_blocks,
            launch_blocks=launch_blocks,
            qualification_blocks=qualification_blocks,
            structural_shortlist=structural_shortlist,
        )


class _CompileIQStagedSearchPlan:
    """Dependency-free coordinator for finite, Windows-safe Forge searches.

    CompileIQ remains optional and may only own the PTXAS-control stage. Forge
    structural and launch candidates are exhaustively paired, while the final
    winner is independently measured against the global baseline.
    """

    def __init__(
        self,
        adapter,
        *,
        structural_blocks,
        launch_blocks,
        qualification_blocks,
        structural_shortlist,
    ):
        if (
            isinstance(structural_shortlist, bool)
            or not isinstance(structural_shortlist, int)
            or structural_shortlist <= 0
        ):
            raise ValueError("structural_shortlist must be a positive integer")
        if structural_shortlist > len(adapter._groups):
            structural_shortlist = len(adapter._groups)
        self._adapter = adapter
        self._structural_blocks = structural_blocks
        self._launch_blocks = launch_blocks
        self._qualification_blocks = qualification_blocks
        self._structural_shortlist = structural_shortlist
        self._structural_stage = _CompileIQSearchStage(
            stage_id="forge-structural",
            candidate_kind="forge_structural",
            candidate_ids=adapter.variant_ids("structural"),
            blocks=structural_blocks,
        )
        _CompileIQSearchStage(
            stage_id="qualification-contract",
            candidate_kind="qualification",
            candidate_ids=(adapter.variant_ids("structural")[0],),
            blocks=qualification_blocks,
        )

    @property
    def structural_stage(self):
        return self._structural_stage

    def shortlisted_compilation_ids(self, structural_measurements):
        ranked = _rank_complete_paired_evidence(
            structural_measurements,
            self._structural_stage.candidate_ids,
            blocks=self._structural_stage.blocks,
            candidate_kind="structural variant",
            collection_name="structural stage",
        )
        return tuple(
            self._adapter._variants[item.variant_id].compilation_id
            for item in ranked[: self._structural_shortlist]
        )

    def launch_stages(self, structural_measurements):
        return tuple(
            _CompileIQSearchStage(
                stage_id=f"forge-launch:{compilation_id}",
                candidate_kind="forge_launch",
                candidate_ids=self._adapter.variant_ids(
                    "launch", compilation_id=compilation_id
                ),
                blocks=self._launch_blocks,
                compilation_id=compilation_id,
            )
            for compilation_id in self.shortlisted_compilation_ids(
                structural_measurements
            )
        )

    def launch_finalists(self, structural_measurements, launch_measurements):
        stages = self.launch_stages(structural_measurements)
        expected = {stage.compilation_id for stage in stages}
        if not isinstance(launch_measurements, dict):
            raise TypeError("launch_measurements must be a dictionary")
        if set(launch_measurements) != expected:
            raise ValueError(
                "launch measurements must exactly match shortlisted groups"
            )
        return tuple(
            _rank_complete_paired_evidence(
                launch_measurements[stage.compilation_id],
                stage.candidate_ids,
                blocks=stage.blocks,
                candidate_kind="launch variant",
                collection_name=f"launch group {stage.compilation_id}",
            )[0].variant_id
            for stage in stages
        )

    def ptxas_stage(self, forge_variant_id, control_ids, *, blocks=4):
        if forge_variant_id not in self._adapter._variants:
            raise KeyError(f"unknown Forge kernel variant {forge_variant_id!r}")
        return _CompileIQSearchStage(
            stage_id=f"ptxas:{forge_variant_id}",
            candidate_kind="ptxas_control",
            candidate_ids=tuple(control_ids),
            blocks=blocks,
            forge_variant_id=forge_variant_id,
        )

    def qualification_stage(self, finalist_ids):
        finalists = tuple(finalist_ids)
        if not finalists or any(
            not isinstance(finalist, _CompileIQFinalCandidate)
            for finalist in finalists
        ):
            raise TypeError(
                "qualification finalists must be _CompileIQFinalCandidate values"
            )
        if any(
            finalist.forge_variant_id not in self._adapter._variants
            for finalist in finalists
        ):
            raise KeyError("qualification contains an unknown Forge variant")
        return _CompileIQSearchStage(
            stage_id="independent-qualification",
            candidate_kind="qualification",
            candidate_ids=tuple(finalist.identity for finalist in finalists),
            blocks=self._qualification_blocks,
        )

    def final_candidate(self, forge_variant_id, provider_candidate_id="baseline"):
        if forge_variant_id not in self._adapter._variants:
            raise KeyError(f"unknown Forge kernel variant {forge_variant_id!r}")
        return _CompileIQFinalCandidate(
            forge_variant_id=forge_variant_id,
            provider_candidate_id=provider_candidate_id,
        )

    def qualify(
        self,
        measurements,
        finalist_ids,
        *,
        scopes,
        correctness,
        memory_stable,
    ):
        finalists = tuple(finalist_ids)
        stage = self.qualification_stage(finalists)
        evidence = _rank_complete_paired_evidence(
            measurements,
            stage.candidate_ids,
            blocks=stage.blocks,
            candidate_kind="qualification candidate",
            collection_name="independent qualification",
        )
        expected = set(stage.candidate_ids)
        if not isinstance(scopes, dict) or set(scopes) != expected:
            raise ValueError("scopes must exactly match qualification candidates")
        if any(
            not isinstance(scope, _CompileIQWinnerScope)
            for scope in scopes.values()
        ):
            raise TypeError("scope values must be _CompileIQWinnerScope values")
        if any(
            scope.final_candidate_id != candidate_id
            for candidate_id, scope in scopes.items()
        ):
            raise ValueError("each scope must bind its exact final candidate ID")
        for name, gate in (
            ("correctness", correctness),
            ("memory_stable", memory_stable),
        ):
            if not isinstance(gate, dict) or set(gate) != expected:
                raise ValueError(f"{name} must exactly match qualification candidates")
            if any(type(value) is not bool for value in gate.values()):
                raise TypeError(f"{name} values must be booleans")
        eligible = tuple(
            item
            for item in evidence
            if correctness[item.variant_id] and memory_stable[item.variant_id]
        )
        if not eligible:
            return _CompileIQQualificationDecision(
                status="keep_baseline",
                reason="no candidate passed correctness and memory gates",
                selected_candidate_id=None,
                selected_forge_variant_id=None,
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
                selected_forge_variant_id=None,
                selected_provider_candidate_id=None,
                scope_id="",
                evidence=evidence,
            )
        selected_candidate = next(
            finalist
            for finalist in finalists
            if finalist.identity == selected.variant_id
        )
        return _CompileIQQualificationDecision(
            status="qualified",
            reason=(
                "exact candidate passed correctness, memory, and "
                "worst-positive gates"
            ),
            selected_candidate_id=selected.variant_id,
            selected_forge_variant_id=selected_candidate.forge_variant_id,
            selected_provider_candidate_id=(
                selected_candidate.provider_candidate_id
            ),
            scope_id=scopes[selected.variant_id].identity,
            evidence=evidence,
        )

    def manifest(self):
        return {
            "schema_version": 1,
            "execution": "windows_subprocess_serial",
            "forge_stages": ("structural", "launch"),
            "compileiq_stage": "optional_ptxas_control_only",
            "qualification": "independent_global_baseline",
            "structural_stage": self.structural_stage.manifest(),
            "structural_shortlist": self._structural_shortlist,
            "launch_blocks": self._launch_blocks,
            "qualification_blocks": self._qualification_blocks,
            "max_exhaustive_stage_candidates": (
                _MAX_EXHAUSTIVE_STAGE_CANDIDATES
            ),
        }


class _CompileIQVariantBundle:
    """Bind independently staged kernel winners as one workload candidate.

    Only structural representatives may be exposed as a bounded diagnostic
    choice space. Full and launch Cartesian products are rejected; callers
    stage each adapter, independently qualify exact finalists, and then use
    this bundle for workload binding. No PTXAS provider is imported here.
    """

    def __init__(self, sessions):
        if not isinstance(sessions, dict):
            raise TypeError("CompileIQ variant sessions must be a dictionary")
        if not sessions:
            raise ValueError("CompileIQ variant bundle requires at least one kernel")
        adapters = {}
        for name, session in sessions.items():
            if not isinstance(name, str) or not _PARAMETER_PATTERN.fullmatch(name):
                raise ValueError(
                    "kernel names must start with a letter and contain only "
                    "letters, digits, and underscores"
                )
            adapters[name] = _CompileIQVariantAdapter(
                session,
                parameter=f"{_FORGE_VARIANT_PARAMETER_PREFIX}{name}",
            )
        self._adapters = MappingProxyType(adapters)

    @property
    def kernel_names(self):
        return tuple(self._adapters)

    def search_space(self, stage="structural"):
        if stage != "structural":
            raise ValueError(
                "joint Forge Cartesian search is disabled; stage each kernel "
                "adapter before composing exact finalists"
            )
        search_space = {}
        for adapter in self._adapters.values():
            search_space.update(adapter.search_space(stage))
        return search_space

    def select(self, parameters):
        if not isinstance(parameters, dict):
            raise TypeError("CompileIQ parameters must be a dictionary")
        return MappingProxyType(
            {
                name: adapter.select(parameters)
                for name, adapter in self._adapters.items()
            }
        )

    def bind(self, parameters):
        if not isinstance(parameters, dict):
            raise TypeError("CompileIQ parameters must be a dictionary")
        return MappingProxyType(
            {
                name: adapter.bind(parameters)
                for name, adapter in self._adapters.items()
            }
        )

    def manifest(self):
        return {
            "schema_version": 1,
            "provider": "compileiq_user_space",
            "uses_ptxas_search_space": False,
            "kernels": {
                name: adapter.manifest()
                for name, adapter in self._adapters.items()
            },
        }


__all__ = [
    "_CompileIQVariantBundle",
    "_CompileIQVariantAdapter",
    "_CompileIQCandidateEvidence",
    "_CompileIQFinalCandidate",
    "_CompileIQQualificationDecision",
    "_CompileIQPairedTrial",
    "_CompileIQSearchStage",
    "_CompileIQStagedSearchPlan",
    "_CompileIQVariantSelection",
    "_CompileIQWinnerScope",
    "_balanced_paired_schedule",
    "_rank_complete_paired_evidence",
]
