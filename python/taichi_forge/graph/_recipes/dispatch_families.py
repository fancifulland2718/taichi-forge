"""Per-dispatch physical recipe providers for complete Graph search."""

from taichi_forge.graph._recipes.families import (
    GraphRuntimeFragmentProvider,
    _choice_fragment,
    runtime_family_provider_descriptor,
)


def _dispatch_choice_fragments(definition, sources, family, descriptor):
    unmatched = [source for source in definition.sources if source.kind == "dispatch"]
    result = []
    for source in sources:
        manifests = source.manifests()
        if not manifests:
            continue
        semantic_identity = manifests[0].to_dict()["semantic_kernel_identity"]
        match = next(
            (item for item in unmatched if item.semantic_identity == semantic_identity),
            None,
        )
        if match is None:
            continue
        unmatched.remove(match)
        for manifest in manifests:
            if manifest.recipe_id == source.selected_recipe_id:
                continue
            result.append(
                _choice_fragment(
                    definition,
                    family,
                    source._recipe_source_key,
                    (match.region_id,),
                    manifest,
                    provider_descriptor=descriptor,
                )
            )
    return tuple(result)


class GraphOffloadPhaseFusionRecipeProvider(GraphRuntimeFragmentProvider):
    descriptor = runtime_family_provider_descriptor(
        "offload_phase_fusion",
        capabilities=("offload-phase-fusion", "typed-runtime-fragment"),
    )

    def fragments(self, definition):
        return _dispatch_choice_fragments(
            definition,
            definition._runtime_spec._graph_offload_fusion_sources,
            "offload_phase_fusion",
            self.descriptor,
        )


class GraphSparseTraversalRecipeProvider(GraphRuntimeFragmentProvider):
    descriptor = runtime_family_provider_descriptor(
        "sparse_traversal",
        capabilities=("sparse-traversal-plan", "typed-runtime-fragment"),
    )

    def fragments(self, definition):
        sources = tuple(
            source
            for source in definition._runtime_spec._graph_sparse_traversal_sources
            if len(source.manifests()) >= 2
        )
        return _dispatch_choice_fragments(
            definition,
            sources,
            "sparse_traversal",
            self.descriptor,
        )


__all__ = [
    "GraphOffloadPhaseFusionRecipeProvider",
    "GraphSparseTraversalRecipeProvider",
]
