"""Branch/join schedule fragments on the complete-recipe provider path."""

from taichi_forge.graph._recipes.families import (
    GraphFamilySelection,
    _fragment,
)
from taichi_forge.graph._recipes.fragments import GraphFragmentTask
from taichi_forge.graph._recipes.materialize import GraphMaterializedFragment
from taichi_forge.graph._recipes.providers import (
    GraphRecipeProviderDescriptor,
    RUNTIME_GRAPH_ASSEMBLY_V1,
)


class GraphBranchJoinRecipeProvider:
    """Expose legal coarse branch DAGs with stable task dependencies."""

    descriptor = GraphRecipeProviderDescriptor(
        namespace="taichi_forge.graph.branch_join_schedule",
        provider_version="complete-graph-family-v1",
        domain_version="branch-join-domain-v1",
        semantic_fingerprint="branch-join-fragment-generation-v1",
        assembly_protocols=(RUNTIME_GRAPH_ASSEMBLY_V1,),
        capabilities=("branch-join-dag", "typed-runtime-fragment"),
        fragment_key_schema="branch_join_schedule:source:choice.v1",
    )

    def discover(self, definition):
        return self.fragments(definition)

    def fragments(self, definition):
        spec = definition._runtime_spec
        result = []
        for source in spec._graph_branch_join_sources:
            runtime_node = spec.nodes[source.node_index]
            logical_nodes = tuple(runtime_node.ir_node.children)
            task_by_dispatch = {}
            tasks = []
            for branch_index, group in enumerate(source.branch_groups):
                previous = None
                for dispatch_index in group:
                    task_id = f"{source.source_key}:dispatch:{dispatch_index}"
                    task = GraphFragmentTask.create(
                        task_id,
                        "cuda_branch_dispatch",
                        depends_on=(() if previous is None else (previous,)),
                        effects=logical_nodes[dispatch_index].effects,
                        bindings=logical_nodes[dispatch_index].bindings,
                        temporaries=logical_nodes[dispatch_index].temporaries,
                        physical={
                            "family": "branch_join_schedule",
                            "queue": f"cuda_branch:{branch_index}",
                            "dispatch_index": dispatch_index,
                        },
                    )
                    tasks.append(task)
                    task_by_dispatch[dispatch_index] = task_id
                    previous = task_id

            join_task_id = f"{source.source_key}:dispatch:{source.join_index}"
            join_node = logical_nodes[source.join_index]
            tasks.append(
                GraphFragmentTask.create(
                    join_task_id,
                    "cuda_branch_join",
                    depends_on=tuple(
                        task_by_dispatch[group[-1]]
                        for group in source.branch_groups
                    ),
                    effects=join_node.effects,
                    bindings=join_node.bindings,
                    temporaries=join_node.temporaries,
                    physical={
                        "family": "branch_join_schedule",
                        "queue": "default",
                        "dispatch_index": source.join_index,
                        "branch_groups": source.branch_groups,
                        "disjoint_pairs": source.disjoint_pairs,
                        "sequential_temporary_bytes": (
                            source.sequential_temporary_bytes
                        ),
                        "parallel_temporary_bytes": (
                            source.parallel_temporary_bytes
                        ),
                    },
                )
            )
            dispatch_indices = tuple(
                index
                for group in source.branch_groups
                for index in group
            ) + (source.join_index,)
            coverage = tuple(
                definition.sources[index].region_id for index in dispatch_indices
            )
            result.append(
                _fragment(
                    definition,
                    family="branch_join_schedule",
                    source_key=source.source_key,
                    choice_id=source.recipe_id,
                    coverage=coverage,
                    tasks=tuple(tasks),
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
            raise KeyError(f"branch/join fragment is unavailable: {fragment_key}")
        return matches[0]

    def expand(self, definition, fragment_key):
        self.resolve(definition, fragment_key)
        return ()

    def materialize(self, scope, fragment):
        selection = GraphFamilySelection.from_fragment(fragment)
        if fragment.coverage_region_ids != selection.coverage_region_ids:
            raise ValueError("branch/join fragment coverage changed before build")
        return GraphMaterializedFragment.create(fragment, selection)

    def describe(self, definition, fragment_key):
        return self.resolve(definition, fragment_key).provider_metadata


__all__ = ["GraphBranchJoinRecipeProvider"]
