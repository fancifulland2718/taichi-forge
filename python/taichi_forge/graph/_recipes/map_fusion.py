"""Exact map-partition fragments on the complete-recipe provider path."""

from taichi_forge.graph._recipes.families import (
    GraphRuntimeFragmentProvider,
    _fragment,
    runtime_family_provider_descriptor,
)
from taichi_forge.graph._recipes.fragments import GraphFragmentTask


class GraphMapFusionRecipeProvider(GraphRuntimeFragmentProvider):
    """Own discovery and stable resolution of complete map partitions."""

    descriptor = runtime_family_provider_descriptor(
        "map_fusion",
        capabilities=("exact-map-partition", "typed-runtime-fragment"),
    )

    def fragments(self, definition):
        spec = definition._runtime_spec
        dispatch_regions = tuple(
            source.region_id
            for source in definition.sources
            if source.kind == "dispatch"
        )
        result = []
        for recipe in spec.fusion_plan.candidate_recipes:
            logical_indices = []
            for source_id in recipe.source_dispatch_ids:
                marker = source_id.rsplit("/dispatch:", 1)
                if len(marker) != 2 or not marker[1].isdigit():
                    logical_indices = []
                    break
                logical_indices.append(int(marker[1]))
            if not logical_indices or any(
                index >= len(dispatch_regions) for index in logical_indices
            ):
                continue
            coverage = tuple(dispatch_regions[index] for index in logical_indices)
            source_key = "dispatches:" + ",".join(
                str(index) for index in logical_indices
            )
            task = GraphFragmentTask.create(
                f"{source_key}:task:0",
                "fused_map",
                physical={
                    "family": "map_fusion",
                    "recipe": recipe.to_dict(),
                    "source_dispatch_indices": tuple(logical_indices),
                },
            )
            result.append(
                _fragment(
                    definition,
                    family="map_fusion",
                    source_key=source_key,
                    choice_id=recipe.recipe_id,
                    coverage=coverage,
                    tasks=(task,),
                    provider_descriptor=self.descriptor,
                )
            )
        return tuple(result)

    def contribute_runtime(self, assembly, selection):
        prefix = "dispatches:"
        if not selection.source_key.startswith(prefix):
            raise ValueError("map-fusion fragment has no source partition")
        try:
            source_group = tuple(
                int(value) for value in selection.source_key[len(prefix) :].split(",")
            )
        except ValueError as error:
            raise ValueError("map-fusion source partition is invalid") from error
        assembly.add_map_source_group(source_group)


__all__ = ["GraphMapFusionRecipeProvider"]
