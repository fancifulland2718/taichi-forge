"""Whole-submission physical recipe providers for complete Graph search."""

from taichi_forge.graph._recipes.families import (
    GraphRuntimeFragmentProvider,
    _fragment,
    runtime_family_provider_descriptor,
)
from taichi_forge.graph._recipes.fragments import (
    GraphFragmentResourceRequirement,
    GraphFragmentTask,
)


class GraphRecordingPartitionRecipeProvider(GraphRuntimeFragmentProvider):
    descriptor = runtime_family_provider_descriptor(
        "recording_partition",
        capabilities=("binding-frontier-partition", "typed-runtime-fragment"),
    )

    def fragments(self, definition):
        coverage = tuple(source.region_id for source in definition.sources)
        result = []
        for source in definition._runtime_spec._graph_recording_partition_sources:
            first_id = f"{source.source_key}:segment:0"
            tasks = (
                GraphFragmentTask.create(
                    first_id,
                    "cuda_recording_segment",
                    physical={
                        "family": "recording_partition",
                        "queue": "default",
                        "dispatch_range": (0, source.cut_index),
                        "isolated_bindings": source.isolated_bindings,
                    },
                ),
                GraphFragmentTask.create(
                    f"{source.source_key}:segment:1",
                    "cuda_recording_segment",
                    depends_on=(first_id,),
                    physical={
                        "family": "recording_partition",
                        "queue": "default",
                        "dispatch_range": (
                            source.cut_index,
                            source.dispatch_count,
                        ),
                        "isolated_bindings": source.isolated_bindings,
                    },
                ),
            )
            result.append(
                _fragment(
                    definition,
                    family="recording_partition",
                    source_key=source.source_key,
                    choice_id=source.recipe_id,
                    coverage=coverage,
                    tasks=tasks,
                    provider_descriptor=self.descriptor,
                )
            )
        return tuple(result)

    def contribute_runtime(self, assembly, selection):
        from taichi_forge.graph._graph import _replay_recording_partition_nodes

        source = assembly.find_source(
            assembly.spec._graph_recording_partition_sources,
            selection.source_key,
            attribute="source_key",
        )
        if selection.choice_id != source.recipe_id:
            raise ValueError(
                "recording-partition selection does not match its frozen source"
            )
        assembly.expand_node(
            source.node_index,
            lambda node: _replay_recording_partition_nodes(node, source),
        )


class GraphWorkspaceConcurrencyRecipeProvider(GraphRuntimeFragmentProvider):
    descriptor = runtime_family_provider_descriptor(
        "workspace_concurrency",
        capabilities=("complete-graph-concurrency", "typed-runtime-fragment"),
    )

    def fragments(self, definition):
        from taichi_forge.graph._graph import _workspace_concurrency_spec_eligible

        spec = definition._runtime_spec
        if not _workspace_concurrency_spec_eligible(spec, definition.backend):
            return ()
        coverage = tuple(source.region_id for source in definition.sources)
        source_key = "whole-graph-pair"
        first_id = f"{source_key}:invoke:0"
        second_id = f"{source_key}:invoke:1"
        tasks = (
            GraphFragmentTask.create(
                first_id,
                "cuda_complete_graph_invocation",
                effects=spec.pre_optimization_ir_root.effects,
                bindings=spec.pre_optimization_ir_root.bindings,
                physical={
                    "family": "workspace_concurrency",
                    "queue": "cuda_workspace:0",
                    "invocation": 0,
                },
            ),
            GraphFragmentTask.create(
                second_id,
                "cuda_complete_graph_invocation",
                effects=spec.pre_optimization_ir_root.effects,
                bindings=spec.pre_optimization_ir_root.bindings,
                physical={
                    "family": "workspace_concurrency",
                    "queue": "cuda_workspace:1",
                    "invocation": 1,
                },
            ),
            GraphFragmentTask.create(
                f"{source_key}:join",
                "cuda_complete_graph_pair_join",
                depends_on=(first_id, second_id),
                physical={
                    "family": "workspace_concurrency",
                    "queue": "default",
                    "event_join": True,
                },
            ),
        )
        return (
            _fragment(
                definition,
                family="workspace_concurrency",
                source_key=source_key,
                choice_id="cuda-concurrent-pair-v1",
                coverage=coverage,
                tasks=tasks,
                resources=(
                    GraphFragmentResourceRequirement(
                        name=f"{source_key}:second-private-workspace",
                        kind="workspace_lane",
                        bytes=int(spec.internal_storage_bytes),
                        alignment=1,
                        ownership="graph_instance",
                        lifetime="graph",
                        exclusive_submission=True,
                    ),
                ),
                provider_descriptor=self.descriptor,
            ),
        )

    def contribute_runtime(self, assembly, selection):
        if (
            selection.source_key != "whole-graph-pair"
            or selection.choice_id != "cuda-concurrent-pair-v1"
        ):
            raise ValueError(
                "workspace concurrency selection is not a supported complete recipe"
            )
        assembly.enable_workspace_pair()


__all__ = [
    "GraphRecordingPartitionRecipeProvider",
    "GraphWorkspaceConcurrencyRecipeProvider",
]
