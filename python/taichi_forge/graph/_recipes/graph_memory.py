"""GraphMemory fragments on the versioned complete-recipe provider path."""

from taichi_forge.graph._recipes.families import (
    GraphFamilySelection,
    _choice_fragment,
)
from taichi_forge.graph._recipes.materialize import GraphMaterializedFragment
from taichi_forge.graph._recipes.providers import (
    GraphRecipeProviderDescriptor,
    RUNTIME_GRAPH_ASSEMBLY_V1,
)


class GraphMemoryRecipeProvider:
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

    def discover(self, definition):
        return self.fragments(definition)

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

    def resolve(self, definition, fragment_key):
        matches = tuple(
            fragment
            for fragment in self.fragments(definition)
            if fragment.fragment_key == fragment_key
        )
        if len(matches) != 1:
            raise KeyError(f"GraphMemory fragment is unavailable: {fragment_key}")
        return matches[0]

    def expand(self, definition, fragment_key):
        self.resolve(definition, fragment_key)
        return ()

    def materialize(self, scope, fragment):
        selection = GraphFamilySelection.from_fragment(fragment)
        if fragment.coverage_region_ids != selection.coverage_region_ids:
            raise ValueError("GraphMemory fragment coverage changed before build")
        return GraphMaterializedFragment.create(fragment, selection)

    def describe(self, definition, fragment_key):
        return self.resolve(definition, fragment_key).provider_metadata


__all__ = ["GraphMemoryRecipeProvider"]
