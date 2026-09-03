"""Migrate established Forge Graph families into complete recipe fragments.

This module is the only boundary between older family-specific manifests and
the complete-Graph recipe system. Providers describe region replacements;
materialization happens once at the whole-Graph boundary and never reconstructs
a choice from process environment state.
"""

from dataclasses import dataclass

from taichi_forge.graph._recipes.fragments import (
    GraphFragmentBindingRequirement,
    GraphFragmentResourceRequirement,
    GraphFragmentSubmissionRequirement,
    GraphFragmentTask,
    GraphRecipeFragment,
)
from taichi_forge.graph._recipes.materialize import (
    GraphMaterializationProduct,
    GraphMaterializedFragment,
)
from taichi_forge.graph._recipes.physical import observe_graph_physical_manifest
from taichi_forge.graph._recipes.providers import (
    GraphRecipeProviderDescriptor,
    RUNTIME_GRAPH_ASSEMBLY_V1,
)

_PROVIDER_VERSION = "complete-graph-family-v1"
_FAMILY_NAMESPACES = tuple(
    "taichi_forge.graph." + family
    for family in (
        "bounded_execution",
        "branch_join_schedule",
        "graph_reduction",
        "map_fusion",
        "native_algorithm",
        "offload_phase_fusion",
        "recording_partition",
        "sparse_traversal",
        "structured_control",
        "workspace_concurrency",
    )
)
_RUNTIME_ASSEMBLY_PROVIDER_DESCRIPTOR = GraphRecipeProviderDescriptor(
    namespace="taichi_forge.graph.runtime_assembly",
    provider_version=_PROVIDER_VERSION,
    domain_version="existing-family-domain-v1",
    semantic_fingerprint="existing-family-fragment-generation-v1",
    assembly_protocols=(RUNTIME_GRAPH_ASSEMBLY_V1,),
    capabilities=("legacy-family-adapter", "typed-runtime-graph-assembly"),
    owned_fragment_namespaces=_FAMILY_NAMESPACES,
    fragment_key_schema="family:source:choice.v1",
)


@dataclass(frozen=True)
class GraphFamilySelection:
    """One explicit provider choice consumed by the whole-Graph assembler."""

    family: str
    source_key: str
    choice_id: str
    materialization_choice: str
    coverage_region_ids: tuple[str, ...]

    def to_dict(self):
        return {
            "family": self.family,
            "source_key": self.source_key,
            "choice_id": self.choice_id,
            "materialization_choice": self.materialization_choice,
            "coverage_region_ids": self.coverage_region_ids,
        }

    @classmethod
    def from_fragment(cls, fragment):
        payload = fragment.provider_metadata.get("family_selection")
        if not isinstance(payload, dict):
            raise ValueError("Graph family fragment has no stable selection metadata")
        return cls(
            family=str(payload["family"]),
            source_key=str(payload["source_key"]),
            choice_id=str(payload["choice_id"]),
            materialization_choice=str(payload["materialization_choice"]),
            coverage_region_ids=tuple(payload["coverage_region_ids"]),
        )


def _subtree_regions(definition, root_region_id):
    root = definition.region(root_region_id)
    return tuple(
        region.region_id
        for region in definition.regions
        if region.path == root.path or region.path.startswith(root.path + "/")
    )


def _binding_requirements(definition, coverage):
    coverage = frozenset(coverage)
    return tuple(
        GraphFragmentBindingRequirement(
            item.name,
            kinds=item.kinds,
            required=item.required,
            scope=item.scope,
        )
        for item in definition.binding_abi
        if not item.region_ids or coverage.intersection(item.region_ids)
    )


def _workspace_resource(source_key, manifest):
    payload = manifest.to_dict()
    workspace = dict(payload.get("workspace", {}))
    size = int(workspace.get("bytes", workspace.get("action_owned_bytes", 0)) or 0)
    ownership = workspace.get("ownership", "none")
    if size <= 0 or ownership == "none":
        return ()
    normalized_ownership = {
        "graph_instance": "graph_instance",
        "graph_native_action": "graph_instance",
        "fragment": "fragment",
        "shared": "shared",
    }.get(ownership, "fragment")
    return (
        GraphFragmentResourceRequirement(
            name=f"{source_key}:workspace",
            kind="workspace",
            bytes=size,
            alignment=1,
            ownership=normalized_ownership,
            lifetime="graph",
            exclusive_submission=bool(
                workspace.get(
                    "exclusive_submission",
                    normalized_ownership == "graph_instance",
                )
            ),
        ),
    )


def _manifest_tasks(family, source_key, manifest):
    payload = manifest.to_dict()
    stages = tuple(payload.get("physical_stages", ()))
    if not stages:
        stages = (
            {
                "name": payload.get("strategy", family),
                "kind": family,
                "manifest": payload,
            },
        )
    tasks = []
    for index, stage in enumerate(stages):
        stage = dict(stage)
        tasks.append(
            GraphFragmentTask.create(
                f"{source_key}:task:{index}",
                str(stage.get("name") or stage.get("kind") or family),
                depends_on=(() if index == 0 else (tasks[-1].task_id,)),
                physical={"family": family, "stage": stage},
            )
        )
    return tuple(tasks)


def _fragment(
    definition,
    *,
    family,
    source_key,
    choice_id,
    materialization_choice=None,
    coverage,
    tasks,
    resources=(),
    exclusive_submission=False,
    provider_descriptor=None,
):
    provider_descriptor = (
        _RUNTIME_ASSEMBLY_PROVIDER_DESCRIPTOR
        if provider_descriptor is None
        else provider_descriptor
    )
    coverage = tuple(coverage)
    selection = GraphFamilySelection(
        family=family,
        source_key=source_key,
        choice_id=choice_id,
        materialization_choice=(
            choice_id if materialization_choice is None else materialization_choice
        ),
        coverage_region_ids=coverage,
    )
    return GraphRecipeFragment.create(
        definition,
        provider_namespace=f"taichi_forge.graph.{family}",
        provider_version=provider_descriptor.provider_version,
        provider_domain_version=provider_descriptor.domain_version,
        fragment_key=f"{family}:{source_key}:{choice_id}",
        coverage_region_ids=coverage,
        tasks=tasks,
        binding_requirements=_binding_requirements(definition, coverage),
        resources=resources,
        submission=GraphFragmentSubmissionRequirement(
            recording_scope="whole_graph",
            exclusive_submission=bool(exclusive_submission),
        ),
        backend_requirements=(definition.backend,),
        assembly_protocol=RUNTIME_GRAPH_ASSEMBLY_V1,
        assembly_provider_namespace=(
            _RUNTIME_ASSEMBLY_PROVIDER_DESCRIPTOR.namespace
        ),
        provider_metadata={"family_selection": selection.to_dict()},
    )


def _choice_fragment(
    definition,
    family,
    source_key,
    coverage,
    manifest,
    *,
    materialization_choice=None,
    provider_descriptor=None,
):
    payload = manifest.to_dict()
    exclusive = bool(
        payload.get("submission", {}).get("exclusive", False)
        or payload.get("workspace", {}).get("exclusive_submission", False)
    )
    resources = _workspace_resource(source_key, manifest)
    return _fragment(
        definition,
        family=family,
        source_key=source_key,
        choice_id=manifest.recipe_id,
        materialization_choice=materialization_choice,
        coverage=coverage,
        tasks=_manifest_tasks(family, source_key, manifest),
        resources=resources,
        exclusive_submission=exclusive
        or any(item.exclusive_submission for item in resources),
        provider_descriptor=provider_descriptor,
    )


def _fusion_fragments(definition, spec):
    dispatch_regions = tuple(
        source.region_id for source in definition.sources if source.kind == "dispatch"
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
        source_key = "dispatches:" + ",".join(str(index) for index in logical_indices)
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
            )
        )
    return tuple(result)


def _offload_phase_fusion_fragments(definition, spec):
    unmatched = [source for source in definition.sources if source.kind == "dispatch"]
    result = []
    for source in spec._graph_offload_fusion_sources:
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
                    "offload_phase_fusion",
                    source._recipe_source_key,
                    (match.region_id,),
                    manifest,
                )
            )
    return tuple(result)


def _sparse_traversal_fragments(definition, spec):
    unmatched = [source for source in definition.sources if source.kind == "dispatch"]
    result = []
    for source in spec._graph_sparse_traversal_sources:
        manifests = source.manifests()
        if len(manifests) < 2:
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
                    "sparse_traversal",
                    source._recipe_source_key,
                    (match.region_id,),
                    manifest,
                )
            )
    return tuple(result)


def _branch_join_fragments(definition, spec):
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
                    "parallel_temporary_bytes": source.parallel_temporary_bytes,
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
            )
        )
    return tuple(result)


def _recording_partition_fragments(definition, spec):
    """Expose bounded binding-frontier cuts as opaque complete recipes."""

    coverage = tuple(source.region_id for source in definition.sources)
    result = []
    for source in spec._graph_recording_partition_sources:
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
            )
        )
    return tuple(result)


def _workspace_concurrency_fragments(definition, spec):
    from taichi_forge.graph._graph import _workspace_concurrency_spec_eligible

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
        ),
    )


def _native_source_coverage(definition, source):
    prefix = f"graph/{source._recipe_node_index}:"
    root = next(
        (
            region
            for region in definition.regions
            if region.parent_region_id == definition.regions[0].region_id
            and region.path.startswith(prefix)
        ),
        None,
    )
    return () if root is None else _subtree_regions(definition, root.region_id)


def _recipe_operation_dispatch_count(operation):
    kind = operation[0]
    if kind == "dispatch":
        return 1
    if kind == "sequential":
        return operation[1]._dispatch_count
    if kind == "bounded":
        return operation[1]._recipe_physical_dispatches
    if kind == "graph_reduction":
        return operation[1].selected_physical_dispatches
    raise ValueError(f"unknown frozen Graph recipe operation {kind!r}")


def _operation_source_coverage(
    definition,
    spec,
    operation_kind,
    requested_source,
):
    regions_by_path = {region.path: region for region in definition.regions}
    for node_index, node in enumerate(spec.nodes):
        operations = getattr(node, "recipe_operations", ())
        if not operations:
            continue
        node_path = f"graph/{node_index}:{node.ir_node.kind}"
        dispatch_index = 0
        for operation in operations:
            count = _recipe_operation_dispatch_count(operation)
            if operation[0] == operation_kind and operation[1] is requested_source:
                coverage = []
                for index in range(dispatch_index, dispatch_index + count):
                    region = regions_by_path.get(f"{node_path}/{index}:dispatch")
                    if region is None:
                        return ()
                    coverage.extend(_subtree_regions(definition, region.region_id))
                return tuple(coverage)
            dispatch_index += count
    return ()


def _semantic_source_fragments(definition, spec, sources, family):
    if len(sources) == 1 and len(definition.sources) == 1:
        coverage_by_source = (
            _subtree_regions(definition, definition.sources[0].region_id),
        )
    elif family == "native_algorithm":
        coverage_by_source = tuple(
            _native_source_coverage(definition, source) for source in sources
        )
    elif family == "graph_reduction":
        coverage_by_source = tuple(
            _operation_source_coverage(
                definition,
                spec,
                "graph_reduction",
                source,
            )
            for source in sources
        )
    else:
        return ()

    result = []
    for source, coverage in zip(sources, coverage_by_source):
        if not coverage:
            continue
        for manifest in source.manifests():
            if manifest.recipe_id == source.selected_recipe_id:
                continue
            result.append(
                _choice_fragment(
                    definition,
                    family,
                    source._recipe_source_key,
                    coverage,
                    manifest,
                )
            )
    return tuple(result)


def _bounded_fragments(definition, spec):
    from taichi_forge.graph._graph import _cuda_nested_device_update_available
    from taichi_forge.graph._optimization import (
        _GraphBoundedExecutionRecipeManifest,
    )

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
        grouped.setdefault(dispatch["publication_key"], []).append((source, dispatch))

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
            )
            for manifest in manifests
            if manifest.recipe_id != selected_id
        )
    return tuple(result)


def _control_fragments(definition, spec):
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
                )
            )
    return tuple(result)


class GraphRuntimeAssemblyProvider:
    """Assemble typed runtime fragments while legacy families migrate."""

    descriptor = _RUNTIME_ASSEMBLY_PROVIDER_DESCRIPTOR

    def discover(self, definition):
        return self.fragments(definition)

    def fragments(self, definition):
        spec = definition._runtime_spec
        fragments = []
        fragments.extend(_fusion_fragments(definition, spec))
        fragments.extend(_offload_phase_fusion_fragments(definition, spec))
        fragments.extend(_sparse_traversal_fragments(definition, spec))
        fragments.extend(_branch_join_fragments(definition, spec))
        fragments.extend(_recording_partition_fragments(definition, spec))
        fragments.extend(_workspace_concurrency_fragments(definition, spec))
        fragments.extend(_bounded_fragments(definition, spec))
        fragments.extend(_control_fragments(definition, spec))
        fragments.extend(
            _semantic_source_fragments(
                definition,
                spec,
                spec._graph_reduction_sources,
                "graph_reduction",
            )
        )
        fragments.extend(
            _semantic_source_fragments(
                definition,
                spec,
                spec._graph_native_algorithm_sources,
                "native_algorithm",
            )
        )
        return tuple(fragments)

    def resolve(self, definition, fragment_key):
        matches = tuple(
            fragment
            for fragment in self.fragments(definition)
            if fragment.fragment_key == fragment_key
        )
        if len(matches) != 1:
            raise KeyError(
                f"existing Graph family fragment is unavailable: {fragment_key}"
            )
        return matches[0]

    def expand(self, definition, fragment_key):
        self.resolve(definition, fragment_key)
        return ()

    def materialize(self, scope, fragment):
        selection = GraphFamilySelection.from_fragment(fragment)
        if fragment.coverage_region_ids != selection.coverage_region_ids:
            raise ValueError("Graph family fragment coverage changed before build")
        return GraphMaterializedFragment.create(fragment, selection)

    def assemble(self, scope, definition, recipe, materialized_fragments):
        return assemble_existing_family_recipe(
            scope,
            definition,
            recipe,
            materialized_fragments,
        )

    def describe(self, definition, fragment_key):
        return self.resolve(definition, fragment_key).provider_metadata


# Transitional private compatibility name.  New code should use the assembly
# role; Phase 7 removes the remaining family discovery responsibilities.
GraphExistingFamilyProvider = GraphRuntimeAssemblyProvider


def default_graph_recipe_providers():
    """Return the built-in providers required by the public recipe path."""

    from taichi_forge.graph._recipes.graph_memory import GraphMemoryRecipeProvider

    return (GraphRuntimeAssemblyProvider(), GraphMemoryRecipeProvider())


def assemble_existing_family_recipe(
    scope,
    definition,
    recipe,
    materialized_fragments,
):
    """Materialize all selected families as one runtime Graph transaction."""

    selections = tuple(item.payload for item in materialized_fragments)
    if not all(isinstance(item, GraphFamilySelection) for item in selections):
        raise TypeError("whole-Graph family assembler received an unknown payload")
    keys = tuple((item.family, item.source_key) for item in selections)
    if len(keys) != len(set(keys)):
        raise ValueError("complete Graph recipe selects a family source more than once")
    graph = definition._runtime_spec.materialize_complete_recipe(
        definition,
        recipe,
        selections,
        workspace_lanes=scope._context.workspace_lanes,
        workspace_saturation=scope._context.workspace_saturation,
    )
    manifest = observe_graph_physical_manifest(definition, recipe, graph)
    return GraphMaterializationProduct(graph, manifest)


def materialize_existing_family_baseline(scope, definition, recipe):
    """Materialize the exact all-baseline whole-Graph recipe.

    Reusing ``GraphDefinition.compile()`` here would also reuse any map fusion
    that the ordinary GraphBuilder selected while the definition was frozen.
    Complete recipes instead treat every uncovered map region as a singleton,
    so route the empty selection through the same exact whole-Graph assembler
    as non-baseline recipes.
    """

    if definition._runtime_spec.fusion_plan.candidate_recipes:
        graph = definition._runtime_spec.materialize_complete_recipe(
            definition,
            recipe,
            (),
            workspace_lanes=scope._context.workspace_lanes,
            workspace_saturation=scope._context.workspace_saturation,
        )
    else:
        # No map-fusion axis exists, so the frozen executor already is the
        # exact all-baseline recipe.  Reuse it to preserve structured-control
        # ownership and provider baselines that require no reconstruction.
        graph = definition.compile(
            workspace_lanes=scope._context.workspace_lanes,
            workspace_saturation=scope._context.workspace_saturation,
        )
    manifest = observe_graph_physical_manifest(definition, recipe, graph)
    return GraphMaterializationProduct(graph, manifest)


__all__ = [
    "GraphRuntimeAssemblyProvider",
    "GraphExistingFamilyProvider",
    "GraphFamilySelection",
    "assemble_existing_family_recipe",
    "default_graph_recipe_providers",
    "materialize_existing_family_baseline",
]
