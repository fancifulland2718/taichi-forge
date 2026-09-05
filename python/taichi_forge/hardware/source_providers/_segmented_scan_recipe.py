"""Explicit Toolkit-addon plans for existing Graph segmented-scan semantics."""

import ctypes

from taichi_forge.graph._native_algorithm import _GraphSegmentedScanRecipeSource
from taichi_forge.graph._recipes.families import (
    GraphRuntimeFragmentProvider,
    _fragment,
    _native_source_coverage,
    runtime_family_provider_descriptor,
)
from taichi_forge.graph._recipes.fragments import (
    GraphFragmentResourceRequirement,
    GraphFragmentTask,
)
from taichi_forge.graph._recipes.providers import GraphRecipeProviderError
from taichi_forge.hardware._cub_segmented_capture import _CubSegmentedScanExecutable
from taichi_forge.hardware._cub_source_provider import (
    _Invocation,
    _OPERATION_SPECS,
    load_cub_source_provider,
)
from taichi_forge.hardware._native_adapter import validate_runtime_generation
from taichi_forge.lang import impl
from taichi_forge.types.primitive_types import u32


class CubSegmentedScanRecipeProvider(GraphRuntimeFragmentProvider):
    """Add a captured reset-monoid strategy to whole-Graph recipe search.

    Construct explicitly from a separately built CUB addon manifest, then pass
    alongside ``ti.graph.default_recipe_providers()``. It discovers frozen
    ``GraphBuilder.segmented_scan()`` operations; it never routes ordinary
    algorithms or changes runtime auto. Only i32/u32 modular sum is supported.
    The addon and its Toolkit/driver requirements remain external to the wheel.
    """

    def __init__(self, manifest_path):
        from taichi_forge._lib import core

        namespace = "taichi_forge.graph.segmented_scan_addon"
        if not hasattr(core.GraphBuilder, "_dispatch_cuda_addon_capture_recipe"):
            raise GraphRecipeProviderError(
                "This native runtime does not provide addon Graph capture",
                error_key="native_addon_capture_unavailable",
                provider_namespace=namespace,
            )
        self._provider = load_cub_source_provider(manifest_path)
        if self._provider._library.info.features & 0x30 != 0x30:
            raise GraphRecipeProviderError(
                "This addon does not provide reset-monoid segmented scan",
                error_key="segmented_scan_addon_unavailable",
                provider_namespace=namespace,
            )
        self._component = self._provider.manifest.build_report()
        self.descriptor = runtime_family_provider_descriptor(
            "segmented_scan_addon",
            capabilities=("modular-segmented-scan", "native-addon-capture"),
            domain_version="reset-monoid-graph-v1",
            semantic_fingerprint=f"reset-monoid-i32-u32-v1:{self._component['build_identity']}",
        )
        self._workspace_queries = {}

    def _sources(self, definition):
        validate_runtime_generation(
            self._provider, "source addon belongs to another runtime generation"
        )
        if definition.backend != "cuda":
            return ()
        return tuple(
            source
            for source in definition._runtime_spec._graph_native_algorithm_sources
            if isinstance(source, _GraphSegmentedScanRecipeSource)
            and source.layout.num_items > 0
        )

    def _workspace(self, source):
        mode = "inclusive" if source.inclusive else "exclusive"
        operation = _OPERATION_SPECS[f"segmented_{mode}_scan_u32"]["code"]
        key = operation, int(source.layout.num_items)
        if key not in self._workspace_queries:
            query = _Invocation(
                struct_size=ctypes.sizeof(_Invocation),
                operation=operation,
                num_items=key[1],
            )
            self._workspace_queries[key] = self._provider._library.workspace_bytes(
                query
            )
        return self._workspace_queries[key]

    def fragments(self, definition):
        result = []
        for source in self._sources(definition):
            coverage = _native_source_coverage(definition, source)
            if not coverage:
                continue
            key = source._recipe_source_key
            head_bytes = 4 * max(1, (source.layout.num_items + 31) // 32)
            workspace_bytes = max(1, self._workspace(source))
            task = GraphFragmentTask.create(
                f"{key}:reset-monoid",
                "captured_segmented_reset_scan",
                physical={
                    "semantic_contract": source.semantics,
                    "strategy": "reset_monoid_lookback",
                    "component": self._component,
                    "head_bitset_bytes": head_bytes,
                    "workspace_bytes": workspace_bytes,
                    "vendor_internal_kernel_topology": "unobserved",
                },
            )
            resources = tuple(
                GraphFragmentResourceRequirement(
                    name=f"{key}:{name}",
                    kind=kind,
                    bytes=size,
                    alignment=4 if name == "heads" else 1,
                    ownership="fragment",
                    lifetime="graph",
                    exclusive_submission=exclusive,
                )
                for name, kind, size, exclusive in (
                    ("heads", "frozen_layout_bitset", head_bytes, False),
                    ("workspace", "scan_tile_state", workspace_bytes, True),
                )
            )
            result.append(
                _fragment(
                    definition,
                    family="segmented_scan_addon",
                    source_key=key,
                    choice_id="reset_monoid_lookback",
                    coverage=coverage,
                    tasks=(task,),
                    resources=resources,
                    exclusive_submission=True,
                    provider_descriptor=self.descriptor,
                )
            )
        return tuple(result)

    def contribute_runtime(self, assembly, selection):
        import numpy as np
        from taichi_forge.graph._graph import GraphBuilder

        source = assembly.find_source(
            self._sources(assembly.definition), selection.source_key
        )
        if selection.materialization_choice != "reset_monoid_lookback":
            raise ValueError("unsupported segmented addon physical strategy")
        # Frozen topology upload belongs to materialization, never to replay.
        offsets = np.asarray(source.offsets, dtype=np.int64)
        starts = offsets[:-1][offsets[:-1] < offsets[1:]]
        packed = np.zeros(max(1, (source.layout.num_items + 31) // 32), np.uint32)
        np.bitwise_or.at(
            packed, starts >> 5, np.uint32(1) << (starts & 31).astype(np.uint32)
        )
        heads = impl.ndarray(u32, shape=len(packed))
        heads.from_numpy(packed)
        executable = _CubSegmentedScanExecutable(
            self._provider,
            source.values,
            heads,
            source.output,
            num_items=int(source.layout.num_items),
            inclusive=source.inclusive,
            binding_prefix=f"_forge_reset_{source._recipe_node_index}",
        )
        builder = GraphBuilder(
            _capture_recipe_sources=False, _explicit_map_source_groups=()
        )
        builder._append_native_executable(executable, admission="explicit")
        builder._flush_graph_builder()
        if len(builder._nodes) != 1:
            raise ValueError("segmented addon did not materialize one Graph region")
        replacement = builder._nodes[0]
        assembly.rewrite_node(source._recipe_node_index, lambda _node: replacement)

    def describe(self, definition, fragment_key):
        fragment = self.resolve(definition, fragment_key)
        return {
            **fragment.provider_metadata,
            "physical_strategy": fragment.tasks[0].physical,
            "limitations": (
                "fixed one-dimensional i32/u32 arrays and frozen segmented sum layout",
                "modular 32-bit arithmetic; not floating-point reassociation",
                "separately built addon with explicit Toolkit/driver/target compatibility",
                "persistent head bitset and scan workspace are candidate costs, not free storage",
                "kernel topology and production performance require measurement",
            ),
        }


__all__ = ["CubSegmentedScanRecipeProvider"]
