"""Private executable-plan optimization identities for Forge Graphs."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from taichi_forge.graph._ir import graph_ir_to_dict


_GRAPH_FUSION_QUALIFICATION_SCHEMA = "taichi_forge.graph_fusion_qualification.v1"
_GRAPH_FUSION_QUALIFICATION_MAX_BYTES = 4 * 1024 * 1024
_INTERNAL_STRUCTURED_CONTROL_ENV = (
    "TAICHI_FORGE_INTERNAL_STRUCTURED_CONTROL_RECIPE"
)
_CUDA_CONDITIONAL_CONTROL_RECIPE_ID = "control:cuda_conditional_graph:v1"
_CUDA_MASKED_CONTROL_RECIPE_ID = "control:cuda_masked_bounded_graph:v1"
_CUDA_CONTROL_RECIPE_IDS = (
    _CUDA_CONDITIONAL_CONTROL_RECIPE_ID,
    _CUDA_MASKED_CONTROL_RECIPE_ID,
)


def _canonical_hash(value):
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _ExecutableOptimizationSpec:
    spec_id: str
    semantic_plan_id: str
    backend: str
    fusion_recipe_ids: tuple
    compilation_identity: str
    execution_identity: str
    control_recipe_id: str = ""

    def __post_init__(self):
        if not self.spec_id.startswith("executable:"):
            raise ValueError("executable optimization spec ID is invalid")
        if not self.semantic_plan_id.startswith("semantic-plan:"):
            raise ValueError("semantic plan ID is invalid")
        if not isinstance(self.backend, str) or not self.backend:
            raise ValueError("executable optimization backend is invalid")
        if len(set(self.fusion_recipe_ids)) != len(self.fusion_recipe_ids):
            raise ValueError("fusion recipe IDs must be unique")
        if not isinstance(self.control_recipe_id, str):
            raise ValueError("control recipe ID must be a string")
        if self.control_recipe_id and self.control_recipe_id not in (
            _CUDA_CONTROL_RECIPE_IDS
        ):
            raise ValueError("control recipe ID is unsupported")
        if self.control_recipe_id and self.fusion_recipe_ids:
            raise ValueError(
                "R5 executable specs cannot combine control and fusion recipes"
            )
        if not self.compilation_identity or not self.execution_identity:
            raise ValueError("executable optimization identities are required")

    def to_dict(self):
        value = {
            "schema_version": 1,
            "spec_id": self.spec_id,
            "semantic_plan_id": self.semantic_plan_id,
            "backend": self.backend,
            "fusion_recipe_ids": self.fusion_recipe_ids,
            "compilation_identity": self.compilation_identity,
            "execution_identity": self.execution_identity,
        }
        # Preserve the exact v1 map-fusion manifest and identities when this
        # optional physical axis is absent.
        if self.control_recipe_id:
            value["control_recipe_id"] = self.control_recipe_id
        return value


@dataclass(frozen=True)
class _ExecutableOptimizationSpace:
    semantic_plan_id: str
    baseline: _ExecutableOptimizationSpec
    candidates: tuple
    selected_spec_id: object
    selection_status: str

    @property
    def selected(self):
        for spec in (self.baseline, *self.candidates):
            if spec.spec_id == self.selected_spec_id:
                return spec
        return None

    def to_dict(self):
        return {
            "schema_version": 1,
            "semantic_plan_id": self.semantic_plan_id,
            "baseline": self.baseline.to_dict(),
            "candidates": tuple(spec.to_dict() for spec in self.candidates),
            "selected_spec_id": self.selected_spec_id,
            "selected": None if self.selected is None else self.selected.to_dict(),
            "selection_status": self.selection_status,
        }


def _required_string(value, role):
    if not isinstance(value, str) or not value or "\n" in value:
        raise ValueError(f"{role} must be a nonempty single-line string")
    return value


def _optional_finite_number(value, role):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{role} must be a finite number or null")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{role} must be finite")
    return value


def _positive_shape(value, role):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{role} must be a nonempty shape")
    shape = tuple(int(extent) for extent in value)
    if any(
        isinstance(extent, bool) or not isinstance(extent, int) for extent in value
    ) or any(extent <= 0 for extent in shape):
        raise ValueError(f"{role} extents must be positive integers")
    return shape


@dataclass(frozen=True)
class _GraphFusionBindingScope:
    name: str
    kind: str
    dtype: str = ""
    rank: int = 0
    element_shape: tuple = ()
    shape_min: tuple = ()
    shape_max: tuple = ()
    scalar_min: object = None
    scalar_max: object = None

    def __post_init__(self):
        _required_string(self.name, "fusion binding name")
        if self.kind not in ("ndarray", "scalar"):
            raise ValueError("fusion binding kind must be ndarray or scalar")
        if self.kind == "ndarray":
            _required_string(self.dtype, "fusion ndarray dtype")
            if (
                isinstance(self.rank, bool)
                or not isinstance(self.rank, int)
                or self.rank < 1
                or len(self.shape_min) != self.rank
                or len(self.shape_max) != self.rank
            ):
                raise ValueError("fusion ndarray rank and shape bounds disagree")
            if any(
                minimum <= 0 or maximum < minimum
                for minimum, maximum in zip(self.shape_min, self.shape_max)
            ):
                raise ValueError("fusion ndarray shape bounds are invalid")
            if self.scalar_min is not None or self.scalar_max is not None:
                raise ValueError("fusion ndarray scope cannot contain scalar bounds")
        elif (
            self.dtype
            or self.rank
            or self.element_shape
            or self.shape_min
            or self.shape_max
        ):
            raise ValueError("fusion scalar scope cannot contain ndarray metadata")
        if (
            self.scalar_min is not None
            and self.scalar_max is not None
            and self.scalar_max < self.scalar_min
        ):
            raise ValueError("fusion scalar bounds are invalid")

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("fusion binding scope must be an object")
        kind = value.get("kind")
        if kind == "ndarray":
            shape_min = _positive_shape(
                value.get("shape_min"), "fusion ndarray shape_min"
            )
            shape_max = _positive_shape(
                value.get("shape_max"), "fusion ndarray shape_max"
            )
            rank = value.get("rank")
            if isinstance(rank, bool) or not isinstance(rank, int):
                raise ValueError("fusion ndarray rank must be an integer")
            element_shape = value.get("element_shape", ())
            if not isinstance(element_shape, (list, tuple)):
                raise ValueError("fusion ndarray element_shape must be a shape")
            element_shape = tuple(int(extent) for extent in element_shape)
            if any(extent <= 0 for extent in element_shape):
                raise ValueError(
                    "fusion ndarray element_shape extents must be positive"
                )
            return cls(
                name=_required_string(value.get("name"), "fusion binding name"),
                kind=kind,
                dtype=_required_string(value.get("dtype"), "fusion ndarray dtype"),
                rank=rank,
                element_shape=element_shape,
                shape_min=shape_min,
                shape_max=shape_max,
            )
        if kind == "scalar":
            return cls(
                name=_required_string(value.get("name"), "fusion binding name"),
                kind=kind,
                scalar_min=_optional_finite_number(
                    value.get("minimum"), "fusion scalar minimum"
                ),
                scalar_max=_optional_finite_number(
                    value.get("maximum"), "fusion scalar maximum"
                ),
            )
        raise ValueError("fusion binding kind must be ndarray or scalar")

    def matches(self, descriptor):
        if not isinstance(descriptor, dict) or descriptor.get("kind") != self.kind:
            return False
        if self.kind == "ndarray":
            shape = tuple(descriptor.get("shape", ()))
            return bool(
                descriptor.get("dtype") == self.dtype
                and int(descriptor.get("rank", -1)) == self.rank
                and tuple(descriptor.get("element_shape", ())) == self.element_shape
                and len(shape) == self.rank
                and all(
                    minimum <= extent <= maximum
                    for extent, minimum, maximum in zip(
                        shape, self.shape_min, self.shape_max
                    )
                )
            )
        value = descriptor.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        value = float(value)
        return bool(
            math.isfinite(value)
            and (self.scalar_min is None or value >= self.scalar_min)
            and (self.scalar_max is None or value <= self.scalar_max)
        )

    def to_dict(self):
        if self.kind == "ndarray":
            return {
                "name": self.name,
                "kind": self.kind,
                "dtype": self.dtype,
                "rank": self.rank,
                "element_shape": self.element_shape,
                "shape_min": self.shape_min,
                "shape_max": self.shape_max,
            }
        return {
            "name": self.name,
            "kind": self.kind,
            "minimum": self.scalar_min,
            "maximum": self.scalar_max,
        }


@dataclass(frozen=True)
class _GraphFusionQualificationEntry:
    semantic_plan_id: str
    backend: str
    baseline_execution_identity: str
    selected_spec_id: str
    execution_identity: str
    source_commit: str
    runtime_scope: tuple
    binding_scopes: tuple
    minimum_expected_replays: int
    evidence_id: str

    def __post_init__(self):
        if not self.semantic_plan_id.startswith("semantic-plan:"):
            raise ValueError("fusion qualification semantic plan ID is invalid")
        _required_string(self.backend, "fusion qualification backend")
        _required_string(
            self.baseline_execution_identity,
            "fusion qualification baseline execution identity",
        )
        if not self.selected_spec_id.startswith("executable:"):
            raise ValueError("fusion qualification executable spec ID is invalid")
        _required_string(
            self.execution_identity, "fusion qualification execution identity"
        )
        if (
            not isinstance(self.source_commit, str)
            or len(self.source_commit) != 40
            or any(
                character not in "0123456789abcdef" for character in self.source_commit
            )
        ):
            raise ValueError("fusion qualification source commit is invalid")
        if not self.runtime_scope:
            raise ValueError("fusion qualification runtime scope is required")
        if len({scope.name for scope in self.binding_scopes}) != len(
            self.binding_scopes
        ):
            raise ValueError("fusion qualification binding names must be unique")
        if (
            isinstance(self.minimum_expected_replays, bool)
            or not isinstance(self.minimum_expected_replays, int)
            or self.minimum_expected_replays < 1
        ):
            raise ValueError(
                "fusion qualification minimum_expected_replays must be positive"
            )
        _required_string(self.evidence_id, "fusion qualification evidence ID")

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("fusion qualification entry must be an object")
        qualification = value.get("qualification")
        if not isinstance(qualification, dict) or any(
            qualification.get(name) is not True
            for name in ("correctness", "memory_stable", "worst_positive")
        ):
            raise ValueError(
                "fusion qualification entry did not pass every admission gate"
            )
        runtime_scope = value.get("runtime_scope")
        if not isinstance(runtime_scope, dict) or not runtime_scope:
            raise ValueError("fusion qualification runtime_scope is required")
        normalized_scope = []
        for key, item in sorted(runtime_scope.items()):
            _required_string(key, "fusion runtime scope key")
            if isinstance(item, bool) or not isinstance(item, (str, int)):
                raise ValueError(
                    "fusion runtime scope values must be strings or integers"
                )
            normalized_scope.append((key, item))
        raw_bindings = value.get("binding_scope", ())
        if not isinstance(raw_bindings, (list, tuple)):
            raise ValueError("fusion qualification binding_scope must be a list")
        minimum_expected_replays = value.get("minimum_expected_replays")
        if isinstance(minimum_expected_replays, bool) or not isinstance(
            minimum_expected_replays, int
        ):
            raise ValueError(
                "fusion qualification minimum_expected_replays must be an integer"
            )
        return cls(
            semantic_plan_id=_required_string(
                value.get("semantic_plan_id"),
                "fusion qualification semantic plan ID",
            ),
            backend=_required_string(
                value.get("backend"), "fusion qualification backend"
            ),
            baseline_execution_identity=_required_string(
                value.get("baseline_execution_identity"),
                "fusion qualification baseline execution identity",
            ),
            selected_spec_id=_required_string(
                value.get("selected_spec_id"),
                "fusion qualification executable spec ID",
            ),
            execution_identity=_required_string(
                value.get("execution_identity"),
                "fusion qualification execution identity",
            ),
            source_commit=_required_string(
                value.get("source_commit"), "fusion qualification source commit"
            ).lower(),
            runtime_scope=tuple(normalized_scope),
            binding_scopes=tuple(
                _GraphFusionBindingScope.from_dict(item) for item in raw_bindings
            ),
            minimum_expected_replays=minimum_expected_replays,
            evidence_id=_required_string(
                value.get("evidence_id"), "fusion qualification evidence ID"
            ),
        )

    @property
    def identity(self):
        return _canonical_hash(self.to_dict())

    def matches(
        self,
        *,
        semantic_plan_id,
        backend,
        source_commit,
        runtime_scope,
        bindings,
        expected_replays,
    ):
        if (
            semantic_plan_id != self.semantic_plan_id
            or backend != self.backend
            or source_commit != self.source_commit
            or expected_replays < self.minimum_expected_replays
            or tuple(sorted(runtime_scope.items())) != self.runtime_scope
        ):
            return False
        return len(self.binding_scopes) == len(bindings) and all(
            scope.name in bindings and scope.matches(bindings[scope.name])
            for scope in self.binding_scopes
        )

    def to_dict(self):
        return {
            "semantic_plan_id": self.semantic_plan_id,
            "backend": self.backend,
            "baseline_execution_identity": self.baseline_execution_identity,
            "selected_spec_id": self.selected_spec_id,
            "execution_identity": self.execution_identity,
            "source_commit": self.source_commit,
            "runtime_scope": dict(self.runtime_scope),
            "binding_scope": tuple(scope.to_dict() for scope in self.binding_scopes),
            "minimum_expected_replays": self.minimum_expected_replays,
            "evidence_id": self.evidence_id,
            "qualification": {
                "correctness": True,
                "memory_stable": True,
                "worst_positive": True,
            },
        }


@dataclass(frozen=True)
class _GraphFusionQualificationCache:
    entries: tuple
    source_path: str = ""

    @classmethod
    def from_dict(cls, value, *, source_path=""):
        if not isinstance(value, dict):
            raise ValueError("graph fusion qualification cache must be an object")
        if value.get("schema") != _GRAPH_FUSION_QUALIFICATION_SCHEMA:
            raise ValueError("graph fusion qualification cache schema is invalid")
        raw_entries = value.get("entries")
        if not isinstance(raw_entries, list):
            raise ValueError("graph fusion qualification entries must be a list")
        entries = tuple(
            _GraphFusionQualificationEntry.from_dict(item) for item in raw_entries
        )
        identities = tuple(entry.identity for entry in entries)
        if len(set(identities)) != len(identities):
            raise ValueError("graph fusion qualification entries must be unique")
        return cls(entries=entries, source_path=str(source_path))

    @classmethod
    def load(cls, path):
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"graph fusion qualification cache does not exist: {path}")
        if path.stat().st_size > _GRAPH_FUSION_QUALIFICATION_MAX_BYTES:
            raise ValueError("graph fusion qualification cache is too large")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("graph fusion qualification cache is invalid") from exc
        return cls.from_dict(value, source_path=str(path))

    def select(self, **scope):
        matches = tuple(entry for entry in self.entries if entry.matches(**scope))
        if not matches:
            return None, "no_exact_qualification"
        if len(matches) != 1:
            return None, "ambiguous_qualification"
        return matches[0], "qualified"

    def to_dict(self):
        return {
            "schema": _GRAPH_FUSION_QUALIFICATION_SCHEMA,
            "entries": tuple(entry.to_dict() for entry in self.entries),
        }


def _make_spec(
    semantic_plan_id,
    backend,
    fusion_recipe_ids,
    control_recipe_id="",
):
    fusion_recipe_ids = tuple(fusion_recipe_ids)
    dispatch_reduction = 0
    for recipe_id in fusion_recipe_ids:
        fields = recipe_id.split(":")
        source_count = 2
        if not (
            len(fields) != 3
            or fields[0] != "fusion"
            or not fields[1].startswith("map")
        ):
            try:
                parsed_count = int(fields[1][3:])
            except ValueError:
                parsed_count = 2
            if 2 <= parsed_count <= 4:
                source_count = parsed_count
        dispatch_reduction += source_count - 1
    compilation_payload = {
        "semantic_plan_id": semantic_plan_id,
        "backend": backend,
        "fusion_recipe_ids": fusion_recipe_ids,
    }
    if control_recipe_id:
        compilation_payload["control_recipe_id"] = control_recipe_id
    compilation_identity = _canonical_hash(compilation_payload)
    execution_identity = _canonical_hash(
        {
            "compilation_identity": compilation_identity,
            "physical_dispatch_delta": -dispatch_reduction,
        }
    )
    return _ExecutableOptimizationSpec(
        spec_id=f"executable:{compilation_identity[:24]}",
        semantic_plan_id=semantic_plan_id,
        backend=backend,
        fusion_recipe_ids=fusion_recipe_ids,
        compilation_identity=compilation_identity,
        execution_identity=execution_identity,
        control_recipe_id=control_recipe_id,
    )


def _build_executable_optimization_space(
    root,
    fusion_plan,
    backend,
    *,
    control_recipe_ids=(),
    selected_control_recipe_id="",
):
    semantic_digest = _canonical_hash(graph_ir_to_dict(root))
    semantic_plan_id = f"semantic-plan:{semantic_digest[:24]}"
    control_recipe_ids = tuple(control_recipe_ids)
    if control_recipe_ids:
        if control_recipe_ids != _CUDA_CONTROL_RECIPE_IDS:
            raise ValueError("structured-control recipe domain is not the R5 domain")
        baseline = _make_spec(
            semantic_plan_id,
            backend,
            (),
            control_recipe_ids[0],
        )
        candidates = tuple(
            _make_spec(semantic_plan_id, backend, (), recipe_id)
            for recipe_id in control_recipe_ids[1:]
        )
        specs = (baseline, *candidates)
        if fusion_plan.applied_groups:
            selected_spec_id = None
            selection_status = "control_recipe_requires_unfused_source"
        else:
            selected = next(
                (
                    spec
                    for spec in specs
                    if spec.control_recipe_id == selected_control_recipe_id
                ),
                None,
            )
            selected_spec_id = None if selected is None else selected.spec_id
            selection_status = (
                "control_recipe_not_materialized"
                if selected is None
                else (
                    "selected_control_baseline"
                    if selected is baseline
                    else "selected_control_recipe"
                )
            )
        return _ExecutableOptimizationSpace(
            semantic_plan_id=semantic_plan_id,
            baseline=baseline,
            candidates=candidates,
            selected_spec_id=selected_spec_id,
            selection_status=selection_status,
        )

    baseline = _make_spec(semantic_plan_id, backend, ())
    candidate_recipe_sets = []
    for partition in fusion_plan.candidate_partitions:
        partition = tuple(partition)
        if partition and partition not in candidate_recipe_sets:
            candidate_recipe_sets.append(partition)
    applied_recipe_ids = tuple(fusion_plan.applied_recipe_ids)
    if applied_recipe_ids and applied_recipe_ids not in candidate_recipe_sets:
        candidate_recipe_sets.append(applied_recipe_ids)
    candidates = tuple(
        _make_spec(semantic_plan_id, backend, candidate)
        for candidate in candidate_recipe_sets
    )
    if fusion_plan.applied_groups == 0:
        selected_spec_id = baseline.spec_id
        selection_status = "selected_baseline"
    elif (
        applied_recipe_ids
        and fusion_plan.applied_groups == len(applied_recipe_ids)
        and fusion_plan.unmatched_applied_groups == 0
    ):
        selected_spec_id = next(
            spec.spec_id
            for spec in candidates
            if spec.fusion_recipe_ids == applied_recipe_ids
        )
        selection_status = "selected_map_recipe"
    else:
        selected_spec_id = None
        selection_status = "applied_group_count_mismatch"
    return _ExecutableOptimizationSpace(
        semantic_plan_id=semantic_plan_id,
        baseline=baseline,
        candidates=candidates,
        selected_spec_id=selected_spec_id,
        selection_status=selection_status,
    )


__all__ = [
    "_CUDA_CONDITIONAL_CONTROL_RECIPE_ID",
    "_CUDA_CONTROL_RECIPE_IDS",
    "_CUDA_MASKED_CONTROL_RECIPE_ID",
    "_GRAPH_FUSION_QUALIFICATION_SCHEMA",
    "_INTERNAL_STRUCTURED_CONTROL_ENV",
    "_ExecutableOptimizationSpace",
    "_ExecutableOptimizationSpec",
    "_GraphFusionBindingScope",
    "_GraphFusionQualificationCache",
    "_GraphFusionQualificationEntry",
    "_build_executable_optimization_space",
]
