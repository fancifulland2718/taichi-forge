"""Modified-CompileIQ search for one qualified primitive provider domain."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from importlib import import_module
from pathlib import Path
from types import MappingProxyType

from taichi_forge._compileiq_opaque import (
    CompileIQOpaqueUnavailableError,
    _CompileIQOpaqueRecipeTransport,
    _identity,
    _validated_compileiq_capability as _validate_shared_compileiq_capability,
)
from taichi_forge._lib import core as _ti_core
from taichi_forge.algorithms._algorithms import (
    _REDUCE_FIELD_PRIVATE_MIN_N,
    _check_reduce_request,
    _primitive_view,
    experimental_reduce,
    resolve_primitive_capability,
)
from taichi_forge.algorithms._autodiff import is_fwd_mode_active, is_tape_active
from taichi_forge.lang._compileiq_adapter import (
    _CompileIQFinalCandidate,
    _CompileIQSearchStage,
    _balanced_paired_schedule,
    _qualify_complete_paired_candidates,
    _rank_complete_paired_evidence,
)
from taichi_forge.lang.impl import current_cfg
from taichi_forge.lang.misc import cuda
from taichi_forge.types.primitive_types import i32


_BASELINE_RECIPE_ID = "reduce-provider:cuda-device:v1"
_FIELD_RECIPE_ID = "reduce-provider:field-atomic:v1"
_RECIPE_METHODS = MappingProxyType(
    {
        _BASELINE_RECIPE_ID: "cuda_device",
        _FIELD_RECIPE_ID: "field_atomic",
    }
)
_PROVIDER_SOURCE_FILES = (
    "_compileiq_opaque.py",
    "_kernels.py",
    "algorithms/_algorithms.py",
    "algorithms/_compileiq_opaque.py",
    "algorithms/_primitive_capabilities.py",
    "lang/_compileiq_adapter.py",
)


class CompileIQReduceProviderUnavailableError(CompileIQOpaqueUnavailableError):
    """The reviewed modified CompileIQ or exact reduce domain is unavailable."""


def _validated_compileiq_capability():
    return _validate_shared_compileiq_capability(
        importer=import_module,
        error_type=CompileIQReduceProviderUnavailableError,
    )


def _method_manifest(method):
    return {
        "method": method.method,
        "backend": method.backend,
        "program_available": method.program_available,
        "provider_probes": tuple(method.provider_probes),
        "implementation": method.implementation,
        "dependency_class": method.dependency_class,
    }


@lru_cache(maxsize=1)
def _provider_source_identity():
    package_root = Path(__file__).resolve().parents[1]
    entries = []
    for relative_path in _PROVIDER_SOURCE_FILES:
        path = package_root / Path(relative_path)
        try:
            normalized = (
                path.read_text(encoding="utf-8")
                .replace("\r\n", "\n")
                .replace("\r", "\n")
            )
        except (OSError, UnicodeError) as error:
            raise CompileIQReduceProviderUnavailableError(
                f"reduce provider source file {relative_path!r} is unavailable"
            ) from error
        entries.append(
            (
                relative_path,
                "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
            )
        )
    entries = tuple(entries)
    return MappingProxyType(
        {
            "lock": _identity(
                "forge-reduce-source-v1:",
                {
                    "schema": "taichi_forge.reduce-provider-source-lock.v1",
                    "files": entries,
                },
            ),
            "files": entries,
        }
    )


def _reduce_scope(values, output, op):
    if current_cfg().arch != cuda:
        raise CompileIQReduceProviderUnavailableError(
            "reduce provider search currently requires an initialized CUDA runtime"
        )
    if is_tape_active() or is_fwd_mode_active():
        raise CompileIQReduceProviderUnavailableError(
            "reduce provider search is unavailable during automatic differentiation"
        )
    if op != "sum":
        raise CompileIQReduceProviderUnavailableError(
            "reduce provider search currently supports only exact i32 sum"
        )
    for method in ("cuda_device", "field_atomic"):
        _check_reduce_request(values, output, op, method, None)

    values_view = _primitive_view(values)
    output_view = _primitive_view(output)
    if (
        values_view is None
        or output_view is None
        or not values_view.is_dense_field
        or not output_view.is_scalar_field
        or values.dtype != i32
        or output.dtype != i32
    ):
        raise CompileIQReduceProviderUnavailableError(
            "reduce provider search requires a dense 1D i32 field and scalar "
            "i32 field output"
        )

    resolved = resolve_primitive_capability("reduce")
    methods = {method.method: method for method in resolved.methods}
    required = ("cuda_device", "field_atomic")
    if any(
        method not in methods or not methods[method].program_available
        for method in required
    ):
        raise CompileIQReduceProviderUnavailableError(
            "the active CUDA runtime does not expose both reviewed reduce providers"
        )
    size = int(values.shape[0])
    source_identity = _provider_source_identity()
    return {
        "schema": "taichi_forge.algorithms.reduce-provider-scope.v1",
        "taichi_commit": _ti_core.get_commit_hash(),
        "backend": "cuda",
        "operation": "sum",
        "dtype": "i32",
        "size": size,
        "input": {
            "storage": "dense_field",
            "shape": (size,),
            "offset": int(values_view.offset),
            "stride": int(values_view.stride),
        },
        "output": {"storage": "scalar_field", "shape": ()},
        "field_private_threshold": _REDUCE_FIELD_PRIVATE_MIN_N,
        "provider_source_lock": source_identity["lock"],
        "provider_source_files": source_identity["files"],
        "providers": tuple(_method_manifest(methods[name]) for name in required),
    }


@dataclass(frozen=True)
class CompileIQReduceProviderSelection:
    recipe_id: str
    method: str
    backend: str
    operation: str
    dtype: str
    size: int
    provider_plan: str

    def to_dict(self):
        return dict(self.__dict__)


class CompileIQReduceProviderSearch:
    """Baseline-inclusive opaque search over exact i32 reduce providers.

    This object is an offline evaluation boundary. It does not modify
    ``experimental_reduce(method='auto')`` and does not install a runtime cache.
    """

    __slots__ = ("_scope", "_semantic_fingerprint", "_transport")

    def __init__(self, values, output, *, op="sum"):
        capability_components = _validated_compileiq_capability()
        scope = _reduce_scope(values, output, op)
        semantic_payload = {
            "schema": "taichi_forge.algorithms.compileiq-reduce-semantics.v1",
            "scope": scope,
            "baseline_recipe_id": _BASELINE_RECIPE_ID,
            "recipes": tuple(
                {"recipe_id": recipe_id, "method": method}
                for recipe_id, method in _RECIPE_METHODS.items()
            ),
        }
        semantic_fingerprint = _identity("forge-reduce-semantics-v1:", semantic_payload)
        transport = _CompileIQOpaqueRecipeTransport(
            provider_namespace="taichi_forge.algorithms.reduce_provider",
            domain_version="dense-field-i32-sum.v1",
            provider_semantic_fingerprint=semantic_fingerprint,
            recipe_ids=tuple(_RECIPE_METHODS),
            baseline_recipe_id=_BASELINE_RECIPE_ID,
            capability_components=capability_components,
            domain_owner="reduce provider",
            recipe_description="reduce provider recipe",
        )
        self._scope = MappingProxyType(scope)
        self._semantic_fingerprint = semantic_fingerprint
        self._transport = transport

    @property
    def capability(self):
        return self._transport.capability

    @property
    def search_space(self):
        return self._transport.search_space

    @property
    def worker_type(self):
        return self._transport.worker_type

    @property
    def python_source_lock(self):
        return self._transport.python_source_lock

    @property
    def domain_fingerprint(self):
        return self._transport.domain_fingerprint

    @property
    def recipe_ids(self):
        return self._transport.recipe_ids

    @property
    def baseline_recipe_id(self):
        return _BASELINE_RECIPE_ID

    @property
    def scope(self):
        return MappingProxyType(copy.deepcopy(dict(self._scope)))

    def _selection(self, recipe_id):
        method = _RECIPE_METHODS[recipe_id]
        provider_plan = (
            "cuda_driver_reduce"
            if method == "cuda_device"
            else (
                "field_private_two_stage"
                if self._scope["size"] >= _REDUCE_FIELD_PRIVATE_MIN_N
                else "field_scalar_atomic"
            )
        )
        return CompileIQReduceProviderSelection(
            recipe_id=recipe_id,
            method=method,
            backend="cuda",
            operation="sum",
            dtype="i32",
            size=self._scope["size"],
            provider_plan=provider_plan,
        )

    def select(self, parameters):
        return self._selection(self._transport.decode(parameters))

    def execute(self, parameters, values, output, *, workspace=None):
        """Materialize one selected route for offline objective evaluation."""

        actual_scope = _reduce_scope(values, output, "sum")
        if actual_scope != dict(self._scope):
            raise ValueError(
                "reduce provider materialization does not match the frozen domain"
            )
        selection = self.select(parameters)
        experimental_reduce(
            values,
            output,
            op="sum",
            method=selection.method,
            workspace=workspace,
        )
        return selection

    def recipe_manifest(self, recipe_id):
        if recipe_id not in self.recipe_ids:
            raise KeyError(f"unknown reduce provider recipe {recipe_id!r}")
        selection = self._selection(recipe_id)
        return MappingProxyType(
            {
                **selection.to_dict(),
                "is_baseline": recipe_id == self.baseline_recipe_id,
            }
        )

    def search_coverage(self, compileiq_search):
        return self._transport.search_coverage(compileiq_search)

    def require_complete_search(self, compileiq_search):
        return self._transport.require_complete_search(compileiq_search)

    def select_best_result(self, compileiq_search, result):
        recipe_id = self._transport.select_best_recipe_id(compileiq_search, result)
        return self._selection(recipe_id)

    def paired_schedule(self, *, blocks=2):
        candidate_ids = tuple(
            recipe_id
            for recipe_id in self.recipe_ids
            if recipe_id != self.baseline_recipe_id
        )
        return _balanced_paired_schedule(candidate_ids, blocks=blocks)

    def rank_paired(self, measurements, *, blocks=2):
        candidate_ids = tuple(
            recipe_id
            for recipe_id in self.recipe_ids
            if recipe_id != self.baseline_recipe_id
        )
        return _rank_complete_paired_evidence(
            measurements,
            candidate_ids,
            blocks=blocks,
            candidate_kind="provider recipe",
            collection_name="reduce provider recipes",
        )

    def final_candidate(self, recipe_id):
        if recipe_id == self.baseline_recipe_id:
            raise ValueError(
                "the reduce provider baseline is a sentinel, not a finalist"
            )
        selection = self._selection(recipe_id)
        return _CompileIQFinalCandidate(
            forge_object_kind="primitive_provider_recipe",
            forge_object_id=recipe_id,
            provider_candidate_id=selection.method,
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
            finalist.forge_object_kind != "primitive_provider_recipe"
            or finalist.forge_object_id not in self.recipe_ids
            or finalist.forge_object_id == self.baseline_recipe_id
            for finalist in finalists
        ):
            raise KeyError("qualification contains an unknown provider recipe")
        return _CompileIQSearchStage(
            stage_id="reduce-provider-independent-qualification",
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

    def manifest(self):
        return {
            "schema": "taichi_forge.algorithms.compileiq-reduce-provider.v1",
            **self._transport.manifest(),
            "object": "experimental_reduce_provider",
            "scope": copy.deepcopy(dict(self._scope)),
            "recipes": tuple(
                dict(self.recipe_manifest(recipe_id)) for recipe_id in self.recipe_ids
            ),
            "qualification": "independent_forge_worst_positive_v1",
            "runtime_admission": "explicit_selection_only_no_auto_policy_mutation",
        }


def compileiq_reduce_provider_search(values, output, *, op="sum"):
    """Build the exact modified-CompileIQ reduce-provider recipe domain."""

    return CompileIQReduceProviderSearch(values, output, op=op)


__all__ = [
    "CompileIQReduceProviderSearch",
    "CompileIQReduceProviderSelection",
    "CompileIQReduceProviderUnavailableError",
    "compileiq_reduce_provider_search",
]
