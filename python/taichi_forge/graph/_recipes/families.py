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

_PROVIDER_VERSION = "complete-graph-family-v1"


@dataclass(frozen=True)
class GraphFamilySelection:
    """One explicit provider choice consumed by the whole-Graph assembler."""

    family: str
    source_key: str
    choice_id: str
    materialization_choice: str
    coverage_region_ids: tuple[str, ...]


@dataclass(frozen=True)
class _FamilyFragmentMaterializer:
    selection: GraphFamilySelection

    def materialize(self, scope, fragment):
        if fragment.coverage_region_ids != self.selection.coverage_region_ids:
            raise ValueError(
                "Graph family fragment coverage changed before build")
        return GraphMaterializedFragment.create(fragment, self.selection)


def _subtree_regions(definition, root_region_id):
    root = definition.region(root_region_id)
    return tuple(
        region.region_id for region in definition.regions
        if region.path == root.path or region.path.startswith(root.path + "/"))


def _binding_requirements(definition, coverage):
    coverage = frozenset(coverage)
    return tuple(
        GraphFragmentBindingRequirement(
            item.name,
            kinds=item.kinds,
            required=item.required,
            scope=item.scope,
        ) for item in definition.binding_abi
        if not item.region_ids or coverage.intersection(item.region_ids))


def _workspace_resource(source_key, manifest):
    payload = manifest.to_dict()
    workspace = dict(payload.get("workspace", {}))
    size = int(
        workspace.get("bytes", workspace.get("action_owned_bytes", 0)) or 0)
    ownership = workspace.get("ownership", "none")
    if size <= 0 or ownership == "none":
        return ()
    normalized_ownership = {
        "graph_instance": "graph_instance",
        "graph_native_action": "graph_instance",
        "fragment": "fragment",
        "shared": "shared",
    }.get(ownership, "fragment")
    return (GraphFragmentResourceRequirement(
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
            )),
    ), )


def _manifest_tasks(family, source_key, manifest):
    payload = manifest.to_dict()
    stages = tuple(payload.get("physical_stages", ()))
    if not stages:
        stages = ({
            "name": payload.get("strategy", family),
            "kind": family,
            "manifest": payload,
        }, )
    tasks = []
    for index, stage in enumerate(stages):
        stage = dict(stage)
        tasks.append(
            GraphFragmentTask.create(
                f"{source_key}:task:{index}",
                str(stage.get("name") or stage.get("kind") or family),
                depends_on=(() if index == 0 else (tasks[-1].task_id, )),
                physical={
                    "family": family,
                    "stage": stage
                },
            ))
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
):
    coverage = tuple(coverage)
    selection = GraphFamilySelection(
        family=family,
        source_key=source_key,
        choice_id=choice_id,
        materialization_choice=(choice_id if materialization_choice is None
                                else materialization_choice),
        coverage_region_ids=coverage,
    )
    return GraphRecipeFragment.create(
        definition,
        provider_namespace=f"taichi_forge.graph.{family}",
        provider_version=_PROVIDER_VERSION,
        coverage_region_ids=coverage,
        tasks=tasks,
        binding_requirements=_binding_requirements(definition, coverage),
        resources=resources,
        submission=GraphFragmentSubmissionRequirement(
            recording_scope="whole_graph",
            exclusive_submission=bool(exclusive_submission),
        ),
        backend_requirements=(definition.backend, ),
        materializer_key=f"{family}:{source_key}:{choice_id}",
        materializer=_FamilyFragmentMaterializer(selection),
    )


def _choice_fragment(
    definition,
    family,
    source_key,
    coverage,
    manifest,
    *,
    materialization_choice=None,
):
    payload = manifest.to_dict()
    exclusive = bool(
        payload.get("submission", {}).get("exclusive", False)
        or payload.get("workspace", {}).get("exclusive_submission", False))
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
    )


def _fusion_fragments(definition, spec):
    dispatch_regions = tuple(source.region_id for source in definition.sources
                             if source.kind == "dispatch")
    result = []
    for recipe in spec.fusion_plan.candidate_recipes:
        logical_indices = []
        for source_id in recipe.source_dispatch_ids:
            marker = source_id.rsplit("/dispatch:", 1)
            if len(marker) != 2 or not marker[1].isdigit():
                logical_indices = []
                break
            logical_indices.append(int(marker[1]))
        if not logical_indices or any(index >= len(dispatch_regions)
                                      for index in logical_indices):
            continue
        coverage = tuple(dispatch_regions[index] for index in logical_indices)
        source_key = "dispatches:" + ",".join(
            str(index) for index in logical_indices)
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
                tasks=(task, ),
            ))
    return tuple(result)


def _memory_fragments(definition, spec):
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
            (item for item in unmatched
             if item.semantic_identity == semantic_identity),
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
                    (match.region_id, ),
                    manifest,
                ))
    return tuple(result)


def _native_source_coverage(definition, source):
    prefix = f"graph/{source._recipe_node_index}:"
    root = next(
        (region for region in definition.regions
         if region.parent_region_id == definition.regions[0].region_id
         and region.path.startswith(prefix)),
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


def _reduction_source_coverage(definition, spec, requested_source):
    regions_by_path = {region.path: region for region in definition.regions}
    for node_index, node in enumerate(spec.nodes):
        operations = getattr(node, "recipe_operations", ())
        if not operations:
            continue
        node_path = f"graph/{node_index}:{node.ir_node.kind}"
        dispatch_index = 0
        for operation in operations:
            count = _recipe_operation_dispatch_count(operation)
            if operation[0] == "graph_reduction" and operation[
                    1] is requested_source:
                coverage = []
                for index in range(dispatch_index, dispatch_index + count):
                    region = regions_by_path.get(
                        f"{node_path}/{index}:dispatch")
                    if region is None:
                        return ()
                    coverage.extend(
                        _subtree_regions(definition, region.region_id))
                return tuple(coverage)
            dispatch_index += count
    return ()


def _semantic_source_fragments(definition, spec, sources, family):
    if len(sources) == 1 and len(definition.sources) == 1:
        coverage_by_source = (_subtree_regions(
            definition, definition.sources[0].region_id), )
    elif family == "native_algorithm":
        coverage_by_source = tuple(
            _native_source_coverage(definition, source) for source in sources)
    elif family == "graph_reduction":
        coverage_by_source = tuple(
            _reduction_source_coverage(definition, spec, source)
            for source in sources)
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
                ))
    return tuple(result)


def _bounded_fragments(definition, spec):
    from taichi_forge.graph._graph import _graph_bounded_recipe_scope

    manifests, selected_recipe_id, status = _graph_bounded_recipe_scope(
        spec.pipeline_definition)
    if status != "complete_recipe_domain":
        return ()
    bounded_regions = []
    physical_sources = []

    def visit(node):
        if not node.children:
            physical_sources.append(node)
            return
        for child in node.children:
            visit(child)

    visit(spec.pre_optimization_ir_root)
    if len(physical_sources) != len(definition.sources):
        return ()
    for semantic, physical in zip(definition.sources, physical_sources):
        if getattr(physical, "bounded_domain", None) is not None:
            bounded_regions.append(semantic.region_id)
    if not bounded_regions:
        return ()
    if len(spec._graph_bounded_sources) != len(bounded_regions):
        # Host-count and legacy bounded dispatches are semantically valid, but
        # they do not preserve an exact replay source for complete recipes.
        return ()
    source_key = "bounded:complete-scope"
    return tuple(
        _choice_fragment(
            definition,
            "bounded_execution",
            source_key,
            tuple(bounded_regions),
            manifest,
            materialization_choice=manifest.strategy,
        ) for manifest in manifests
        if manifest.recipe_id != selected_recipe_id)


def _control_fragments(definition, spec):
    recipe_ids = spec.control_recipe_ids
    if not recipe_ids:
        return ()
    structured_kinds = frozenset(("while", "if", "switch", "while_region",
                                  "if_region", "switch_region"))
    regions_by_id = {region.region_id: region for region in definition.regions}

    def has_structured_ancestor(region):
        parent_id = region.parent_region_id
        while parent_id is not None:
            parent = regions_by_id[parent_id]
            if parent.kind in structured_kinds:
                return True
            parent_id = parent.parent_region_id
        return False

    roots = tuple(region for region in definition.regions
                  if region.kind in structured_kinds
                  and not has_structured_ancestor(region))
    if not roots:
        return ()
    coverage = tuple(
        region_id for root in roots
        for region_id in _subtree_regions(definition, root.region_id))
    source_key = "structured-control:complete-scope"
    result = []
    for recipe_id in recipe_ids:
        if recipe_id == spec.selected_control_recipe_id:
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
                tasks=(task, ),
            ))
    return tuple(result)


class GraphExistingFamilyProvider:
    """Expose every currently materializable family without early return."""
    def fragments(self, definition):
        spec = definition._runtime_spec
        fragments = []
        fragments.extend(_fusion_fragments(definition, spec))
        fragments.extend(_memory_fragments(definition, spec))
        fragments.extend(_bounded_fragments(definition, spec))
        fragments.extend(_control_fragments(definition, spec))
        fragments.extend(
            _semantic_source_fragments(
                definition,
                spec,
                spec._graph_reduction_sources,
                "graph_reduction",
            ))
        fragments.extend(
            _semantic_source_fragments(
                definition,
                spec,
                spec._graph_native_algorithm_sources,
                "native_algorithm",
            ))
        return tuple(fragments)


def assemble_existing_family_recipe(
    scope,
    definition,
    recipe,
    materialized_fragments,
):
    """Materialize all selected families as one runtime Graph transaction."""

    selections = tuple(item.payload for item in materialized_fragments)
    if not all(isinstance(item, GraphFamilySelection) for item in selections):
        raise TypeError(
            "whole-Graph family assembler received an unknown payload")
    keys = tuple((item.family, item.source_key) for item in selections)
    if len(keys) != len(set(keys)):
        raise ValueError(
            "complete Graph recipe selects a family source more than once")
    graph = definition._runtime_spec.materialize_complete_recipe(
        definition,
        recipe,
        selections,
        workspace_lanes=scope._context.workspace_lanes,
        workspace_saturation=scope._context.workspace_saturation,
    )
    manifest = observe_graph_physical_manifest(definition, recipe, graph)
    return GraphMaterializationProduct(graph, manifest)


__all__ = [
    "GraphExistingFamilyProvider",
    "GraphFamilySelection",
    "assemble_existing_family_recipe",
]
