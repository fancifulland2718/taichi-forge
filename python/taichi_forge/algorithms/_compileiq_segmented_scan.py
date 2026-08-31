"""Modified-CompileIQ search for one exact integer segmented-scan domain."""

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
    _SEGMENTED_SCAN_CUDA_GLOBAL_MIN_ITEMS,
    _SEGMENTED_SCAN_CUDA_GLOBAL_MIN_SEGMENT_LENGTH,
    _check_segmented_request,
    _primitive_view,
    _prog_available,
    experimental_segmented_scan,
)
from taichi_forge.algorithms._autodiff import is_fwd_mode_active, is_tape_active
from taichi_forge.lang import impl
from taichi_forge.lang._compileiq_qualification import (
    _CompileIQFinalCandidate,
    _CompileIQSearchStage,
    _balanced_paired_schedule,
    _qualify_complete_paired_candidates,
    _rank_complete_paired_evidence,
)
from taichi_forge.lang.impl import current_cfg
from taichi_forge.lang.misc import cuda
from taichi_forge.types.primitive_types import i32, u32


_SERIAL_RECIPE_ID = "segmented-scan:serial:v1"
_GLOBAL_RECIPE_ID = "segmented-scan:cuda-global-scan:v1"
_RECIPE_METHODS = MappingProxyType(
    {
        _SERIAL_RECIPE_ID: "serial",
        _GLOBAL_RECIPE_ID: "global_scan",
    }
)
_SOURCE_FILES = (
    "_compileiq_opaque.py",
    "algorithms/__init__.py",
    "algorithms/_algorithms.py",
    "algorithms/_compileiq_segmented_scan.py",
    "algorithms/_primitive_capabilities.py",
    "lang/_compileiq_qualification.py",
)


class CompileIQSegmentedScanUnavailableError(CompileIQOpaqueUnavailableError):
    """The exact fork or the reviewed segmented-scan domain is unavailable."""


def _validated_compileiq_capability():
    return _validate_shared_compileiq_capability(
        importer=import_module,
        error_type=CompileIQSegmentedScanUnavailableError,
    )


@lru_cache(maxsize=1)
def _source_identity():
    package_root = Path(__file__).resolve().parents[1]
    entries = []
    for relative_path in _SOURCE_FILES:
        path = package_root / Path(relative_path)
        try:
            normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
        except (OSError, UnicodeError) as error:
            raise CompileIQSegmentedScanUnavailableError(
                f"segmented-scan source file {relative_path!r} is unavailable"
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
                "forge-segmented-scan-source-v1:",
                {
                    "schema": ("taichi_forge.segmented-scan-source-lock.v1"),
                    "files": entries,
                },
            ),
            "files": entries,
        }
    )


def _topology_fingerprint(layout):
    return _identity(
        "forge-segmented-layout-v1:",
        {
            "encoding": layout._encoding,
            "offsets": tuple(layout._offsets_host),
            "capacity": layout.capacity,
            "num_items": layout.num_items,
        },
    )


def _segmented_scan_scope(values, layout, output, inclusive, op):
    if current_cfg().arch != cuda:
        raise CompileIQSegmentedScanUnavailableError("segmented-scan search requires an initialized CUDA runtime")
    if is_tape_active() or is_fwd_mode_active():
        raise CompileIQSegmentedScanUnavailableError(
            "segmented-scan search is unavailable during automatic differentiation"
        )
    if op != "sum":
        raise CompileIQSegmentedScanUnavailableError("segmented-scan search currently supports only integer sum")
    if not isinstance(inclusive, bool):
        raise CompileIQSegmentedScanUnavailableError("segmented-scan search inclusive must be a bool")
    request_modes = []
    for method in ("serial", "global_scan"):
        request_modes.append(
            _check_segmented_request(
                "compileiq_segmented_scan_search()",
                values,
                layout,
                output,
                method=method,
                workspace=None,
                scan=True,
            )
        )
    ndarray_mode, in_place = request_modes[0]
    if any(mode != request_modes[0] for mode in request_modes[1:]):
        raise CompileIQSegmentedScanUnavailableError("segmented-scan physical routes disagree on the request contract")
    values_view = _primitive_view(values)
    output_view = _primitive_view(output)
    if (
        not ndarray_mode
        or in_place
        or values_view is None
        or output_view is None
        or not values_view.is_plain_ndarray
        or not output_view.is_plain_ndarray
        or values.dtype not in (i32, u32)
        or output.dtype != values.dtype
    ):
        raise CompileIQSegmentedScanUnavailableError(
            "segmented-scan search requires disjoint plain 1D i32/u32 ndarrays"
        )

    impl.get_runtime().materialize()
    prog = impl.get_runtime().prog
    if not _prog_available(prog, "cuda_device_scan_available"):
        raise CompileIQSegmentedScanUnavailableError(
            "the active CUDA runtime does not expose the reviewed global scan plan"
        )

    dtype = "i32" if values.dtype == i32 else "u32"
    global_baseline = (
        layout.num_items >= _SEGMENTED_SCAN_CUDA_GLOBAL_MIN_ITEMS
        and layout.max_segment_length >= _SEGMENTED_SCAN_CUDA_GLOBAL_MIN_SEGMENT_LENGTH
    )
    source_identity = _source_identity()
    return {
        "schema": "taichi_forge.algorithms.segmented-scan-scope.v1",
        "taichi_commit": _ti_core.get_commit_hash(),
        "backend": "cuda",
        "operation": "sum",
        "dtype": dtype,
        "inclusive": inclusive,
        "source_method": "auto",
        "baseline_method": "global_scan" if global_baseline else "serial",
        "input": {
            "storage": "plain_ndarray",
            "shape": (int(values.shape[0]),),
        },
        "output": {
            "storage": "plain_ndarray",
            "shape": (int(output.shape[0]),),
            "aliases_input": False,
        },
        "layout": {
            "encoding": layout._encoding,
            "capacity": layout.capacity,
            "num_items": layout.num_items,
            "num_segments": layout.num_segments,
            "max_segment_length": layout.max_segment_length,
            "topology_fingerprint": _topology_fingerprint(layout),
        },
        "auto_threshold": {
            "minimum_items": _SEGMENTED_SCAN_CUDA_GLOBAL_MIN_ITEMS,
            "minimum_max_segment_length": (_SEGMENTED_SCAN_CUDA_GLOBAL_MIN_SEGMENT_LENGTH),
        },
        "provider_source_lock": source_identity["lock"],
        "provider_source_files": source_identity["files"],
        "providers": (
            {
                "method": "serial",
                "backend": "cuda",
                "implementation": "segment_local_jit",
                "dependency_class": "builtin",
            },
            {
                "method": "global_scan",
                "backend": "cuda",
                "implementation": "native_scan_plus_segment_correction",
                "dependency_class": "driver",
                "provider_probe": "cuda_device_scan_available",
            },
        ),
    }


@dataclass(frozen=True)
class CompileIQSegmentedScanSelection:
    recipe_id: str
    method: str
    backend: str
    operation: str
    dtype: str
    inclusive: bool
    capacity: int
    num_items: int
    num_segments: int
    max_segment_length: int
    topology_fingerprint: str
    provider_plan: str

    def to_dict(self):
        return dict(self.__dict__)


class CompileIQSegmentedScanSearch:
    """Opaque offline search over two complete integer segmented-scan plans.

    The search never mutates ``method="auto"`` and cannot emit a runtime cache.
    """

    __slots__ = ("_scope", "_semantic_fingerprint", "_transport")

    def __init__(self, values, layout, output, *, inclusive=True, op="sum"):
        capability_components = _validated_compileiq_capability()
        scope = _segmented_scan_scope(values, layout, output, inclusive, op)
        baseline_recipe_id = next(
            recipe_id for recipe_id, method in _RECIPE_METHODS.items() if method == scope["baseline_method"]
        )
        semantic_payload = {
            "schema": ("taichi_forge.algorithms.compileiq-segmented-scan-semantics.v1"),
            "scope": scope,
            "baseline_recipe_id": baseline_recipe_id,
            "recipes": tuple(
                {"recipe_id": recipe_id, "method": method} for recipe_id, method in _RECIPE_METHODS.items()
            ),
        }
        semantic_fingerprint = _identity("forge-segmented-scan-semantics-v1:", semantic_payload)
        transport = _CompileIQOpaqueRecipeTransport(
            provider_namespace="taichi_forge.algorithms.segmented_scan",
            domain_version="cuda-integer-immutable-layout.v1",
            provider_semantic_fingerprint=semantic_fingerprint,
            recipe_ids=tuple(_RECIPE_METHODS),
            baseline_recipe_id=baseline_recipe_id,
            capability_components=capability_components,
            domain_owner="segmented scan",
            recipe_description="segmented-scan recipe",
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
        return self._transport.baseline_recipe_id

    @property
    def scope(self):
        return MappingProxyType(copy.deepcopy(dict(self._scope)))

    def _selection(self, recipe_id):
        method = _RECIPE_METHODS[recipe_id]
        layout = self._scope["layout"]
        return CompileIQSegmentedScanSelection(
            recipe_id=recipe_id,
            method=method,
            backend="cuda",
            operation="sum",
            dtype=self._scope["dtype"],
            inclusive=self._scope["inclusive"],
            capacity=layout["capacity"],
            num_items=layout["num_items"],
            num_segments=layout["num_segments"],
            max_segment_length=layout["max_segment_length"],
            topology_fingerprint=layout["topology_fingerprint"],
            provider_plan=("segment_local_serial" if method == "serial" else "cuda_global_scan_segment_correction"),
        )

    def select(self, parameters):
        return self._selection(self._transport.decode(parameters))

    def execute(
        self,
        parameters,
        values,
        layout,
        output,
        *,
        workspace=None,
    ):
        actual_scope = _segmented_scan_scope(
            values,
            layout,
            output,
            self._scope["inclusive"],
            "sum",
        )
        if actual_scope != dict(self._scope):
            raise ValueError("segmented-scan materialization does not match the frozen domain")
        selection = self.select(parameters)
        experimental_segmented_scan(
            values,
            layout,
            output,
            inclusive=selection.inclusive,
            op="sum",
            method=selection.method,
            workspace=workspace,
        )
        return selection

    def recipe_manifest(self, recipe_id):
        if recipe_id not in self.recipe_ids:
            raise KeyError(f"unknown segmented-scan recipe {recipe_id!r}")
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
        candidate_ids = tuple(recipe_id for recipe_id in self.recipe_ids if recipe_id != self.baseline_recipe_id)
        return _balanced_paired_schedule(candidate_ids, blocks=blocks)

    def rank_paired(self, measurements, *, blocks=2):
        candidate_ids = tuple(recipe_id for recipe_id in self.recipe_ids if recipe_id != self.baseline_recipe_id)
        return _rank_complete_paired_evidence(
            measurements,
            candidate_ids,
            blocks=blocks,
            candidate_kind="segmented-scan recipe",
            collection_name="segmented-scan recipes",
        )

    def final_candidate(self, recipe_id):
        if recipe_id == self.baseline_recipe_id:
            raise ValueError("the segmented-scan baseline is a sentinel, not a finalist")
        selection = self._selection(recipe_id)
        return _CompileIQFinalCandidate(
            forge_object_kind="primitive_provider_recipe",
            forge_object_id=recipe_id,
            provider_candidate_id=selection.method,
        )

    def qualification_stage(self, finalists, *, blocks=10):
        finalists = tuple(finalists)
        if not finalists or any(not isinstance(finalist, _CompileIQFinalCandidate) for finalist in finalists):
            raise TypeError("qualification finalists must be _CompileIQFinalCandidate values")
        if any(
            finalist.forge_object_kind != "primitive_provider_recipe"
            or finalist.forge_object_id not in self.recipe_ids
            or finalist.forge_object_id == self.baseline_recipe_id
            for finalist in finalists
        ):
            raise KeyError("qualification contains an unknown segmented-scan recipe")
        return _CompileIQSearchStage(
            stage_id="segmented-scan-independent-qualification",
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
            "schema": "taichi_forge.algorithms.compileiq-segmented-scan.v1",
            **self._transport.manifest(),
            "object": "experimental_segmented_scan",
            "scope": copy.deepcopy(dict(self._scope)),
            "recipes": tuple(dict(self.recipe_manifest(recipe_id)) for recipe_id in self.recipe_ids),
            "qualification": "independent_forge_worst_positive_v1",
            "runtime_admission": ("explicit_selection_only_no_auto_policy_mutation"),
        }


def compileiq_segmented_scan_search(
    values,
    layout,
    output,
    *,
    inclusive=True,
    op="sum",
):
    """Build the exact modified-CompileIQ integer segmented-scan domain."""

    return CompileIQSegmentedScanSearch(
        values,
        layout,
        output,
        inclusive=inclusive,
        op=op,
    )


__all__ = [
    "CompileIQSegmentedScanSearch",
    "CompileIQSegmentedScanSelection",
    "CompileIQSegmentedScanUnavailableError",
    "compileiq_segmented_scan_search",
]
