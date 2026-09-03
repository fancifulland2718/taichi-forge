"""GraphMemory fragments on the versioned complete-recipe provider path."""

from taichi_forge.graph._recipes.families import (
    GraphRuntimeFragmentProvider,
    _choice_fragment,
)
from taichi_forge.graph._recipes.providers import (
    GraphRecipeProviderDescriptor,
    RUNTIME_GRAPH_ASSEMBLY_V1,
)


class GraphMemoryRecipeProvider(GraphRuntimeFragmentProvider):
    """Discover and rebuild physical memory-plan alternatives by stable key."""

    descriptor = GraphRecipeProviderDescriptor(
        namespace="taichi_forge.graph.graph_memory",
        provider_version="complete-graph-family-v1",
        domain_version="graph-memory-domain-v1",
        semantic_fingerprint="graph-memory-fragment-generation-v1",
        assembly_protocols=(RUNTIME_GRAPH_ASSEMBLY_V1,),
        capabilities=("graph-memory-plan", "typed-runtime-fragment"),
        fragment_key_schema="graph_memory:source:choice.v1",
    )

    def fragments(self, definition):
        spec = definition._runtime_spec
        unmatched = [
            source for source in definition.sources if source.kind == "dispatch"
        ]
        result = []
        for ordinal, source in enumerate(spec._graph_memory_sources):
            manifests = source.manifests()
            if not manifests:
                continue
            semantic_identity = manifests[0].to_dict()["semantic_kernel_identity"]
            match = next(
                (
                    item
                    for item in unmatched
                    if item.semantic_identity == semantic_identity
                ),
                None,
            )
            if match is None:
                continue
            unmatched.remove(match)
            source_key = f"memory:{ordinal}"
            for manifest in manifests:
                if manifest.recipe_id == source.selected_recipe_id:
                    continue
                result.append(
                    _choice_fragment(
                        definition,
                        "graph_memory",
                        source_key,
                        (match.region_id,),
                        manifest,
                        provider_descriptor=self.descriptor,
                    )
                )
        return tuple(result)


__all__ = ["GraphMemoryRecipeProvider"]
