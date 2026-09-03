"""Semantic-operation physical recipe providers for complete Graph search."""

from taichi_forge.graph._recipes.families import (
    GraphRuntimeFragmentProvider,
    _fragment,
    _operation_source_coverage,
    _semantic_source_fragments,
    _subtree_regions,
    _choice_fragment,
    runtime_family_provider_descriptor,
)
from taichi_forge.graph._recipes.fragments import GraphFragmentTask


class GraphBoundedExecutionRecipeProvider(GraphRuntimeFragmentProvider):
    descriptor = runtime_family_provider_descriptor(
        "bounded_execution",
        capabilities=("device-bounded-execution", "typed-runtime-fragment"),
    )

    def fragments(self, definition):
        from taichi_forge.graph._graph import _cuda_nested_device_update_available
        from taichi_forge.graph._optimization import (
            _GraphBoundedExecutionRecipeManifest,
        )

        spec = definition._runtime_spec
        sources = tuple(spec._graph_bounded_sources)
        bounded = tuple(
            dispatch
            for stage in spec.pipeline_definition
            for dispatch in stage["bounded_dispatches"]
        )
        if not sources or len(sources) != len(bounded):
            return ()
        grouped = {}
        for source, dispatch in zip(sources, bounded):
            domain = dispatch["domain"]
            if (
                domain.count_source != "device_extent"
                or domain.ordered
                or domain.physical_grid_policy != "auto"
                or not domain.semantic_kernel_identity
            ):
                return ()
            grouped.setdefault(dispatch["publication_key"], []).append(
                (source, dispatch)
            )

        result = []
        for ordinal, members in enumerate(grouped.values()):
            source_key = f"bounded-group:{ordinal}"
            coverage = []
            selected_strategies = set()
            for source, _ in members:
                source._recipe_group_key = source_key
                selected_strategies.add(source.selected_strategy)
                source_coverage = _operation_source_coverage(
                    definition,
                    spec,
                    "bounded",
                    source,
                )
                if not source_coverage:
                    coverage = []
                    break
                coverage.extend(source_coverage)
            if not coverage or len(selected_strategies) != 1:
                continue
            _, first_dispatch = members[0]
            publication = {
                "count_name": first_dispatch["count_name"],
                "capacity": first_dispatch["capacity"],
                "block_dim": first_dispatch["domain"].block_dim,
                "publication_epoch": first_dispatch["domain"].publication_epoch,
                "consumer_count": len(members),
            }
            strategies = ["logical_exact"]
            if _cuda_nested_device_update_available():
                strategies.append("adaptive_per_node")
                if len(members) >= 2:
                    strategies.append("adaptive_grouped")
            strategies.append("masked_capacity")
            manifests = tuple(
                _GraphBoundedExecutionRecipeManifest.from_payload(
                    {
                        "strategy": strategy,
                        "source_physical_grid_policy": "auto",
                        "bounded_dispatch_count": len(members),
                        "publication_groups": (publication,),
                    }
                )
                for strategy in strategies
            )
            selected_strategy = next(iter(selected_strategies))
            selected_id = next(
                manifest.recipe_id
                for manifest in manifests
                if manifest.strategy == selected_strategy
            )
            result.extend(
                _choice_fragment(
                    definition,
                    "bounded_execution",
                    source_key,
                    tuple(dict.fromkeys(coverage)),
                    manifest,
                    materialization_choice=manifest.strategy,
                    provider_descriptor=self.descriptor,
                )
                for manifest in manifests
                if manifest.recipe_id != selected_id
            )
        return tuple(result)

    def contribute_runtime(self, assembly, selection):
        sources = tuple(
            source
            for source in assembly.spec._graph_bounded_sources
            if getattr(source, "_recipe_group_key", None) == selection.source_key
        )
        if not sources:
            raise ValueError(
                f"bounded-execution source is unavailable: {selection.source_key}"
            )
        for source in sources:

            def rewrite_operation(builder, _operation, source=source):
                source.materialize(
                    builder,
                    selection.materialization_choice,
                )

            assembly.select_operation(source, rewrite_operation)


class GraphStructuredControlRecipeProvider(GraphRuntimeFragmentProvider):
    descriptor = runtime_family_provider_descriptor(
        "structured_control",
        capabilities=("structured-control-route", "typed-runtime-fragment"),
    )

    def fragments(self, definition):
        spec = definition._runtime_spec
        domains = spec.control_recipe_domains
        if not domains:
            return ()
        structured_kinds = frozenset(
            ("while", "if", "switch", "while_region", "if_region", "switch_region")
        )
        regions_by_id = {region.region_id: region for region in definition.regions}

        def has_structured_ancestor(region):
            parent_id = region.parent_region_id
            while parent_id is not None:
                parent = regions_by_id[parent_id]
                if parent.kind in structured_kinds:
                    return True
                parent_id = parent.parent_region_id
            return False

        result = []
        for node_index, recipe_ids, selected_recipe_id in domains:
            source_prefix = f"graph/{node_index}:"
            roots = tuple(
                region
                for region in definition.regions
                if region.path.startswith(source_prefix)
                and region.kind in structured_kinds
                and not has_structured_ancestor(region)
            )
            if not roots:
                continue
            coverage = tuple(
                region_id
                for root in roots
                for region_id in _subtree_regions(definition, root.region_id)
            )
            source_key = f"structured-control:node:{node_index}"
            for recipe_id in recipe_ids:
                if recipe_id == selected_recipe_id:
                    continue
                task = GraphFragmentTask.create(
                    f"{source_key}:task:0",
                    "structured_control",
                    physical={
                        "family": "structured_control",
                        "recipe_id": recipe_id,
                    },
                )
                result.append(
                    _fragment(
                        definition,
                        family="structured_control",
                        source_key=source_key,
                        choice_id=recipe_id,
                        coverage=coverage,
                        tasks=(task,),
                        provider_descriptor=self.descriptor,
                    )
                )
        return tuple(result)

    def contribute_runtime(self, assembly, selection):
        from taichi_forge.graph._graph import (
            _CONTROL_RECIPE_ROUTES,
            _clone_control_recipe_runtime_node,
        )

        prefix = "structured-control:node:"
        if not selection.source_key.startswith(prefix):
            raise ValueError("structured-control fragment has no source-node partition")
        try:
            node_index = int(selection.source_key[len(prefix) :])
            control_recipe = _CONTROL_RECIPE_ROUTES[selection.choice_id]
        except (KeyError, ValueError) as error:
            raise ValueError("structured-control selection is invalid") from error
        assembly.rewrite_node(
            node_index,
            lambda node: _clone_control_recipe_runtime_node(
                node,
                control_recipe,
            ),
        )


class GraphReductionRecipeProvider(GraphRuntimeFragmentProvider):
    descriptor = runtime_family_provider_descriptor(
        "graph_reduction",
        capabilities=("graph-reduction-phase-plan", "typed-runtime-fragment"),
    )

    def fragments(self, definition):
        spec = definition._runtime_spec
        return _semantic_source_fragments(
            definition,
            spec,
            spec._graph_reduction_sources,
            "graph_reduction",
            provider_descriptor=self.descriptor,
        )

    def contribute_runtime(self, assembly, selection):
        source = assembly.find_source(
            assembly.spec._graph_reduction_sources,
            selection.source_key,
        )

        def rewrite_operation(builder, operation):
            source.materialize(
                builder,
                selection.choice_id,
                label=operation[2],
                record_selection=False,
            )

        assembly.select_operation(source, rewrite_operation)


class GraphNativeAlgorithmRecipeProvider(GraphRuntimeFragmentProvider):
    descriptor = runtime_family_provider_descriptor(
        "native_algorithm",
        capabilities=("native-algorithm-plan", "typed-runtime-fragment"),
    )

    def fragments(self, definition):
        spec = definition._runtime_spec
        return _semantic_source_fragments(
            definition,
            spec,
            spec._graph_native_algorithm_sources,
            "native_algorithm",
            provider_descriptor=self.descriptor,
        )

    def contribute_runtime(self, assembly, selection):
        from taichi_forge.graph._graph import GraphBuilder

        source = assembly.find_source(
            assembly.spec._graph_native_algorithm_sources,
            selection.source_key,
        )
        builder = GraphBuilder(
            _capture_recipe_sources=False,
            _explicit_map_source_groups=(),
        )
        source.materialize(
            builder,
            selection.choice_id,
            record_selection=False,
        )
        builder._flush_graph_builder()
        if len(builder._nodes) != 1:
            raise ValueError(
                "native-algorithm recipe did not materialize one Graph node"
            )
        replacement = builder._nodes[0]
        assembly.rewrite_node(
            source._recipe_node_index,
            lambda _node: replacement,
        )


__all__ = [
    "GraphBoundedExecutionRecipeProvider",
    "GraphNativeAlgorithmRecipeProvider",
    "GraphReductionRecipeProvider",
    "GraphStructuredControlRecipeProvider",
]
