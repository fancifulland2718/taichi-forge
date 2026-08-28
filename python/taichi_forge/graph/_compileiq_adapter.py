"""Private optional CompileIQ adapter for Forge executable plans."""

from dataclasses import dataclass
from types import MappingProxyType
import re

from taichi_forge.graph._optimization import _ExecutableOptimizationSpace
from taichi_forge.lang._compileiq_adapter import (
    _balanced_paired_schedule,
    _compileiq_import_error,
    _rank_complete_paired_evidence,
)


_EXECUTABLE_PARAMETER = "forge_executable_spec"
_INTERNAL_MAP_FUSION_ENV = "TAICHI_FORGE_INTERNAL_MAP_FUSION"
_PARAMETER_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_MAP_RECIPE_PATTERN = re.compile(r"^fusion:map([2-4]):[0-9a-f]{24}$")


def _materialization_recipe(spec):
    if not spec.fusion_recipe_ids:
        return "baseline"
    group_sizes = []
    for recipe_id in spec.fusion_recipe_ids:
        match = _MAP_RECIPE_PATTERN.fullmatch(recipe_id)
        if match is None:
            raise ValueError(
                f"unsupported executable fusion recipe {recipe_id!r}"
            )
        group_sizes.append(int(match.group(1)))
    return f"map{max(group_sizes)}"


@dataclass(frozen=True)
class _CompileIQExecutableSelection:
    spec_id: str
    semantic_plan_id: str
    backend: str
    fusion_recipe_ids: tuple
    compilation_identity: str
    execution_identity: str
    materialization_recipe: str

    @property
    def worker_environment(self):
        """Return an environment overlay without mutating process state."""

        return MappingProxyType(
            {_INTERNAL_MAP_FUSION_ENV: self.materialization_recipe}
        )


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

        materialization_by_spec = {
            spec.spec_id: _materialization_recipe(spec) for spec in specs
        }
        materializations = tuple(materialization_by_spec.values())
        if len(set(materializations)) != len(materializations):
            raise ValueError(
                "executable candidates must map to unique materialization recipes"
            )

        self._space = space
        self._parameter = parameter
        self._specs = MappingProxyType({spec.spec_id: spec for spec in specs})
        self._materialization_by_spec = MappingProxyType(
            materialization_by_spec
        )
        self._candidate_ids = tuple(spec.spec_id for spec in space.candidates)

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
            raise KeyError(
                f"unknown Forge executable spec {spec_id!r}"
            ) from error
        return _CompileIQExecutableSelection(
            spec_id=spec.spec_id,
            semantic_plan_id=spec.semantic_plan_id,
            backend=spec.backend,
            fusion_recipe_ids=spec.fusion_recipe_ids,
            compilation_identity=spec.compilation_identity,
            execution_identity=spec.execution_identity,
            materialization_recipe=self._materialization_by_spec[spec.spec_id],
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

    def manifest(self):
        return {
            "schema_version": 1,
            "provider": "compileiq_user_space",
            "parameter": self._parameter,
            "semantic_plan_id": self.semantic_plan_id,
            "backend": self.backend,
            "baseline_spec_id": self._space.baseline.spec_id,
            "specs": tuple(
                {
                    **spec.to_dict(),
                    "materialization_recipe": self._materialization_by_spec[
                        spec.spec_id
                    ],
                }
                for spec in self._specs.values()
            ),
        }


__all__ = [
    "_CompileIQExecutableAdapter",
    "_CompileIQExecutableSelection",
]
