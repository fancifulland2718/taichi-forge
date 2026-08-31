"""Private optional CompileIQ adapter for Forge executable plans."""

from dataclasses import dataclass
from types import MappingProxyType
import json
import re

from taichi_forge.graph._optimization import (
    _CUDA_CONTROL_RECIPE_IDS,
    _CUDA_NESTED_CONTROL_RECIPE_IDS,
    _CUDA_STRUCTURED_CONTROL_RECIPE_DOMAINS,
    _GRAPH_FUSION_QUALIFICATION_SCHEMA,
    _INTERNAL_STRUCTURED_CONTROL_ENV,
    _ExecutableOptimizationSpace,
    _GraphFusionQualificationCache,
)
from taichi_forge.lang._compileiq_adapter import (
    _CompileIQFinalCandidate,
    _CompileIQQualificationDecision,
    _CompileIQSearchStage,
    _balanced_paired_schedule,
    _compileiq_import_error,
    _qualify_complete_paired_candidates,
    _rank_complete_paired_evidence,
)


_EXECUTABLE_PARAMETER = "forge_executable_spec"
_INTERNAL_MAP_FUSION_ENV = "TAICHI_FORGE_INTERNAL_MAP_FUSION"
_PARAMETER_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_MAP_RECIPE_PATTERN = re.compile(r"^fusion:map([2-4]):[0-9a-f]{24}$")
_CONTROL_MATERIALIZATION = {
    _CUDA_CONTROL_RECIPE_IDS[0]: "cuda_conditional_graph",
    _CUDA_CONTROL_RECIPE_IDS[1]: "cuda_masked_bounded_graph",
    _CUDA_NESTED_CONTROL_RECIPE_IDS[0]: "cuda_nested_device_update",
    _CUDA_NESTED_CONTROL_RECIPE_IDS[1]: "cuda_nested_masked_bounded",
}
_CONTROL_DOMAIN_NAMES = {
    _CUDA_CONTROL_RECIPE_IDS: "cuda_flat",
    _CUDA_NESTED_CONTROL_RECIPE_IDS: "cuda_nested_while_while",
}


def _materialization_recipe(spec):
    if not spec.fusion_recipe_ids:
        return "baseline"
    group_sizes = []
    for recipe_id in spec.fusion_recipe_ids:
        match = _MAP_RECIPE_PATTERN.fullmatch(recipe_id)
        if match is None:
            raise ValueError(f"unsupported executable fusion recipe {recipe_id!r}")
        group_sizes.append(int(match.group(1)))
    return f"map{max(group_sizes)}"


def _control_materialization_recipe(spec):
    if not spec.control_recipe_id:
        return "auto"
    try:
        return _CONTROL_MATERIALIZATION[spec.control_recipe_id]
    except KeyError as error:
        raise ValueError(
            f"unsupported structured-control recipe {spec.control_recipe_id!r}"
        ) from error


@dataclass(frozen=True)
class GraphExecutableRecipeSelection:
    spec_id: str
    semantic_plan_id: str
    backend: str
    fusion_recipe_ids: tuple
    compilation_identity: str
    execution_identity: str
    materialization_recipe: str
    control_recipe_id: str = ""
    control_materialization_recipe: str = "auto"

    @property
    def worker_environment(self):
        """Return an environment overlay without mutating process state."""

        return MappingProxyType(
            {
                _INTERNAL_MAP_FUSION_ENV: self.materialization_recipe,
                _INTERNAL_STRUCTURED_CONTROL_ENV: (
                    self.control_materialization_recipe
                ),
            }
        )

    def to_dict(self):
        value = {
            "spec_id": self.spec_id,
            "semantic_plan_id": self.semantic_plan_id,
            "backend": self.backend,
            "fusion_recipe_ids": self.fusion_recipe_ids,
            "compilation_identity": self.compilation_identity,
            "execution_identity": self.execution_identity,
            "materialization_recipe": self.materialization_recipe,
        }
        if self.control_recipe_id:
            value["control_recipe_id"] = self.control_recipe_id
            value["control_materialization_recipe"] = (
                self.control_materialization_recipe
            )
        return value


_CompileIQExecutableSelection = GraphExecutableRecipeSelection


class _CompileIQExecutableAdapter:
    """Expose bounded Graph recipes as an offline categorical dimension.

    Forge owns candidate legality, materialization, and identity.  CompileIQ
    only selects one stable spec ID.  The evaluation worker applies the
    returned environment overlay before constructing the Graph and must call
    :meth:`verify_materialized` after compilation.  Ordinary import, Graph
    construction, launch, and replay never import or invoke CompileIQ.
    """

    def __init__(self, space, *, parameter=_EXECUTABLE_PARAMETER):
        if not isinstance(space, _ExecutableOptimizationSpace):
            raise TypeError(
                "CompileIQ executable adapter requires an optimization space"
            )
        if not isinstance(parameter, str):
            raise TypeError("CompileIQ parameter must be a string")
        if not _PARAMETER_PATTERN.fullmatch(parameter):
            raise ValueError(
                "CompileIQ parameter must start with a letter and contain only "
                "letters, digits, and underscores"
            )

        specs = (space.baseline, *space.candidates)
        if space.baseline.fusion_recipe_ids:
            raise ValueError("executable baseline must not contain fusion recipes")
        if any(spec.semantic_plan_id != space.semantic_plan_id for spec in specs):
            raise ValueError("executable specs must share one semantic plan")
        if any(spec.backend != space.baseline.backend for spec in specs):
            raise ValueError("executable specs must share one backend")
        if len({spec.spec_id for spec in specs}) != len(specs):
            raise ValueError("executable spec IDs must be unique")
        control_recipe_ids = tuple(spec.control_recipe_id for spec in specs)
        if space.baseline.control_recipe_id:
            if (
                control_recipe_ids not in _CUDA_STRUCTURED_CONTROL_RECIPE_DOMAINS
                or any(spec.fusion_recipe_ids for spec in specs)
            ):
                raise ValueError(
                    "structured-control space must contain one exact Forge domain"
                )
        elif any(control_recipe_ids):
            raise ValueError(
                "map-fusion space cannot contain a control recipe candidate"
            )

        materialization_by_spec = {
            spec.spec_id: _materialization_recipe(spec) for spec in specs
        }
        control_materialization_by_spec = {
            spec.spec_id: _control_materialization_recipe(spec) for spec in specs
        }
        materializations = tuple(
            (
                materialization_by_spec[spec.spec_id],
                control_materialization_by_spec[spec.spec_id],
            )
            for spec in specs
        )
        if len(set(materializations)) != len(materializations):
            raise ValueError(
                "executable candidates must map to unique materialization recipes"
            )

        self._space = space
        self._parameter = parameter
        self._specs = MappingProxyType({spec.spec_id: spec for spec in specs})
        self._materialization_by_spec = MappingProxyType(materialization_by_spec)
        self._control_materialization_by_spec = MappingProxyType(
            control_materialization_by_spec
        )
        self._candidate_ids = tuple(spec.spec_id for spec in space.candidates)
        self._structured_control_domain = _CONTROL_DOMAIN_NAMES.get(
            control_recipe_ids,
            "",
        )

    @classmethod
    def from_graph(cls, graph, *, parameter=_EXECUTABLE_PARAMETER):
        try:
            space = graph._executable_optimization_space
        except AttributeError as error:
            raise TypeError(
                "CompileIQ executable adapter requires a compiled Forge Graph"
            ) from error
        return cls(space, parameter=parameter)

    @property
    def semantic_plan_id(self):
        return self._space.semantic_plan_id

    @property
    def backend(self):
        return self._space.baseline.backend

    @property
    def recipe_kind(self):
        return (
            "structured_control"
            if self._space.baseline.control_recipe_id
            else "map_fusion"
        )

    @property
    def structured_control_domain(self):
        return self._structured_control_domain

    @property
    def parameter(self):
        return self._parameter

    @property
    def baseline_spec_id(self):
        return self._space.baseline.spec_id

    def spec_ids(self, *, include_baseline=True):
        if not isinstance(include_baseline, bool):
            raise TypeError("include_baseline must be a bool")
        if include_baseline:
            return (self._space.baseline.spec_id, *self._candidate_ids)
        return self._candidate_ids

    def search_space(self):
        """Build the CompileIQ choice only when explicitly requested."""

        try:
            from compileiq.search_spaces.base import choice
        except ImportError as error:
            raise _compileiq_import_error() from error
        return {self._parameter: choice(self.spec_ids())}

    def select(self, parameters):
        if not isinstance(parameters, dict):
            raise TypeError("CompileIQ parameters must be a dictionary")
        if self._parameter not in parameters:
            raise KeyError(f"CompileIQ parameters require {self._parameter!r}")
        spec_id = parameters[self._parameter]
        if not isinstance(spec_id, str):
            raise TypeError(f"{self._parameter} must be a string")
        try:
            spec = self._specs[spec_id]
        except KeyError as error:
            raise KeyError(f"unknown Forge executable spec {spec_id!r}") from error
        return GraphExecutableRecipeSelection(
            spec_id=spec.spec_id,
            semantic_plan_id=spec.semantic_plan_id,
            backend=spec.backend,
            fusion_recipe_ids=spec.fusion_recipe_ids,
            compilation_identity=spec.compilation_identity,
            execution_identity=spec.execution_identity,
            materialization_recipe=self._materialization_by_spec[spec.spec_id],
            control_recipe_id=spec.control_recipe_id,
            control_materialization_recipe=(
                self._control_materialization_by_spec[spec.spec_id]
            ),
        )

    def verify_materialized(self, parameters, actual_space):
        """Fail closed unless a rebuilt Graph exactly matches the selection."""

        selection = self.select(parameters)
        if not isinstance(actual_space, _ExecutableOptimizationSpace):
            raise TypeError("materialized result must be an optimization space")
        if actual_space.semantic_plan_id != selection.semantic_plan_id:
            raise ValueError("materialized Graph semantic plan does not match")
        if actual_space.baseline.backend != selection.backend:
            raise ValueError("materialized Graph backend does not match")
        if actual_space.selected_spec_id != selection.spec_id:
            raise ValueError("materialized Graph did not select the requested spec")
        actual = actual_space.selected
        if actual is None or (
            actual.compilation_identity != selection.compilation_identity
            or actual.execution_identity != selection.execution_identity
            or actual.fusion_recipe_ids != selection.fusion_recipe_ids
            or actual.control_recipe_id != selection.control_recipe_id
        ):
            raise ValueError("materialized Graph identity does not match")
        return selection

    def verify_materialized_graph(self, parameters, graph):
        try:
            actual_space = graph._executable_optimization_space
        except AttributeError as error:
            raise TypeError(
                "materialized result must be a compiled Forge Graph"
            ) from error
        return self.verify_materialized(parameters, actual_space)

    def paired_schedule(self, *, blocks=2):
        """Enumerate every non-baseline recipe with balanced AB/BA order."""

        return _balanced_paired_schedule(
            self._candidate_ids,
            blocks=blocks,
        )

    def rank_paired(self, measurements, *, blocks=2):
        """Rank complete evidence while retaining the baseline sentinel."""

        return _rank_complete_paired_evidence(
            measurements,
            self._candidate_ids,
            blocks=blocks,
            candidate_kind="spec",
            collection_name="executable candidates",
        )

    def final_candidate(self, spec_id, provider_candidate_id="baseline"):
        if spec_id not in self._specs:
            raise KeyError(f"unknown Forge executable spec {spec_id!r}")
        return _CompileIQFinalCandidate(
            forge_object_kind="executable_spec",
            forge_object_id=spec_id,
            provider_candidate_id=provider_candidate_id,
        )

    def qualification_stage(self, finalists, *, blocks=10):
        finalists = tuple(finalists)
        if not finalists or any(
            not isinstance(finalist, _CompileIQFinalCandidate) for finalist in finalists
        ):
            raise TypeError(
                "qualification finalists must be _CompileIQFinalCandidate values"
            )
        if any(
            finalist.forge_object_kind != "executable_spec"
            or finalist.forge_object_id not in self._specs
            for finalist in finalists
        ):
            raise KeyError("qualification contains an unknown executable spec")
        return _CompileIQSearchStage(
            stage_id="executable-independent-qualification",
            candidate_kind="qualification",
            candidate_ids=tuple(finalist.identity for finalist in finalists),
            blocks=blocks,
        )

    def qualify(
        self,
        measurements,
        finalists,
        *,
        scopes,
        correctness,
        memory_stable,
        blocks=10,
    ):
        finalists = tuple(finalists)
        stage = self.qualification_stage(finalists, blocks=blocks)
        return _qualify_complete_paired_candidates(
            measurements,
            finalists,
            scopes=scopes,
            correctness=correctness,
            memory_stable=memory_stable,
            blocks=stage.blocks,
        )

    def qualification_cache(
        self,
        decision,
        *,
        source_commit,
        runtime_scope,
        binding_scope,
        minimum_expected_replays,
        evidence_id,
        runtime_provider_candidate_id,
    ):
        """Emit one runtime-consumable cache entry after independent gates.

        Search measurements are intentionally insufficient.  The caller must
        pass an admitted decision from :meth:`qualify`, the exact runtime and
        binding scopes used by the independent fresh-process qualification,
        and the provider candidate that the ordinary runtime will reproduce.
        The returned value is validated by the same strict parser used at
        runtime, without importing CompileIQ.
        """

        if not isinstance(decision, _CompileIQQualificationDecision):
            raise TypeError(
                "qualification cache requires a CompileIQ qualification decision"
            )
        if not decision.admitted:
            raise ValueError("qualification cache requires an admitted decision")
        if decision.selected_forge_object_kind != "executable_spec":
            raise ValueError("qualification decision did not select an executable spec")
        try:
            selected = self._specs[decision.selected_forge_object_id]
        except KeyError as error:
            raise ValueError(
                "qualification decision selected an unknown executable spec"
            ) from error
        if selected is self._space.baseline or not selected.fusion_recipe_ids:
            if selected is self._space.baseline:
                raise ValueError("qualification cache cannot select the baseline spec")
            raise ValueError(
                "structured-control qualification is offline-only in R5; "
                "runtime cache admission is unavailable"
            )
        if (
            not isinstance(runtime_provider_candidate_id, str)
            or not runtime_provider_candidate_id
        ):
            raise ValueError("runtime provider candidate ID is required")
        if decision.selected_provider_candidate_id != runtime_provider_candidate_id:
            raise ValueError(
                "qualified provider candidate does not match the runtime provider"
            )
        if not isinstance(runtime_scope, dict):
            raise TypeError("runtime_scope must be a dictionary")
        if not isinstance(binding_scope, (tuple, list)):
            raise TypeError("binding_scope must be a tuple or list")
        if not isinstance(evidence_id, str) or not evidence_id:
            raise ValueError("evidence_id is required")

        value = {
            "schema": _GRAPH_FUSION_QUALIFICATION_SCHEMA,
            "entries": [
                {
                    "semantic_plan_id": self.semantic_plan_id,
                    "backend": self.backend,
                    "baseline_execution_identity": (
                        self._space.baseline.execution_identity
                    ),
                    "selected_spec_id": selected.spec_id,
                    "execution_identity": selected.execution_identity,
                    "source_commit": source_commit,
                    "runtime_scope": dict(runtime_scope),
                    "binding_scope": list(binding_scope),
                    "minimum_expected_replays": minimum_expected_replays,
                    "evidence_id": (
                        f"{evidence_id}|compileiq_scope={decision.scope_id}"
                    ),
                    "qualification": {
                        "correctness": True,
                        "memory_stable": True,
                        "worst_positive": True,
                    },
                }
            ],
        }
        validated = _GraphFusionQualificationCache.from_dict(value).to_dict()
        # Cache.to_dict() uses immutable tuples internally.  Normalize through
        # JSON so this method returns the exact list-based shape accepted by the
        # runtime file parser and ready for direct serialization.
        return json.loads(json.dumps(validated, sort_keys=True))

    def qualification_cache_json(self, decision, **scope):
        return json.dumps(
            self.qualification_cache(decision, **scope),
            indent=2,
            sort_keys=True,
        )

    def manifest(self):
        value = {
            "schema_version": 1,
            "provider": "compileiq_user_space",
            "parameter": self._parameter,
            "semantic_plan_id": self.semantic_plan_id,
            "backend": self.backend,
            "baseline_spec_id": self._space.baseline.spec_id,
            "search_protocol": "exhaustive_then_independent_qualification",
            "specs": tuple(self._spec_manifest(spec) for spec in self._specs.values()),
        }
        if self.recipe_kind == "structured_control":
            value["recipe_kind"] = self.recipe_kind
            value["runtime_admission"] = "offline_explicit_reconstruction_only"
            if self.structured_control_domain == "cuda_nested_while_while":
                value["structured_control_domain"] = (
                    self.structured_control_domain
                )
        return value

    def _spec_manifest(self, spec):
        value = {
            **spec.to_dict(),
            "materialization_recipe": self._materialization_by_spec[spec.spec_id],
        }
        if spec.control_recipe_id:
            value["control_materialization_recipe"] = (
                self._control_materialization_by_spec[spec.spec_id]
            )
        return value


__all__ = [
    "GraphExecutableRecipeSelection",
    "_CompileIQExecutableAdapter",
    "_CompileIQExecutableSelection",
]
