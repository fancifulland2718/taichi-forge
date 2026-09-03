import json
from types import SimpleNamespace

import pytest
import taichi_forge as ti
from taichi_forge.graph._ir import (
    DispatchNode,
    GraphAccess,
    ResourceEffect,
    RuntimeBinding,
    SequentialRegion,
    TemporaryRequirement,
)
from taichi_forge.graph._recipes import (
    GraphDefinition,
    GraphFragmentBindingRequirement,
    GraphFragmentResourceRequirement,
    GraphFragmentSubmissionRequirement,
    GraphFragmentTask,
    GraphRecipeCatalog,
    GraphRecipeComposer,
    GraphRecipeCompositionError,
    GraphRecipeFragment,
    GraphRecipeProviderDescriptor,
    GraphRecipeProviderError,
    GraphRecipeProviderSet,
    PROVIDER_OWNED_WHOLE_GRAPH_V1,
)

from tests import test_utils


def _definition():
    nodes = (
        DispatchNode(
            name="load",
            effects=(
                ResourceEffect("values", GraphAccess.READ),
                ResourceEffect("middle_a", GraphAccess.WRITE),
            ),
            bindings=(RuntimeBinding("values", "ndarray"),),
            logical_kernel_identity="kernel:load",
        ),
        DispatchNode(
            name="map",
            effects=(
                ResourceEffect("middle_a", GraphAccess.READ),
                ResourceEffect("middle_b", GraphAccess.WRITE),
            ),
            logical_kernel_identity="kernel:map",
        ),
        DispatchNode(
            name="store",
            effects=(
                ResourceEffect("middle_b", GraphAccess.READ),
                ResourceEffect("output", GraphAccess.WRITE),
            ),
            bindings=(RuntimeBinding("output", "ndarray"),),
            logical_kernel_identity="kernel:store",
        ),
    )
    root = SequentialRegion(nodes, name="graph")
    spec = SimpleNamespace(
        pre_optimization_ir_root=root,
        definition_semantic_root=root,
        definition_semantic_sources=(),
        ir_root=root,
        runtime_arg_names=frozenset(("values", "output")),
        fixed_runtime_args={},
        temporary_runtime_arg_names=frozenset(),
        derived_runtime_arg_names=frozenset(),
        execution_definition={
            "nodes": (),
            "dispatch_count": 3,
            "native_count": 0,
            "observation_count": 0,
            "structured_control_count": 0,
            "max_structured_depth": 0,
            "runtime_arg_count": 2,
            "fixed_runtime_arg_count": 0,
            "internal_storage_bytes": 0,
            "temporary_memory_plan": {},
        },
    )
    return GraphDefinition._from_graph_spec(
        spec,
        "cuda",
        core_commit="test-build",
    )


def _task(task_id, route, *, effects=(), depends_on=(), temporaries=()):
    return GraphFragmentTask.create(
        task_id,
        "kernel",
        depends_on=depends_on,
        effects=effects,
        temporaries=temporaries,
        physical={"route": route},
    )


def _provider_descriptor(namespace="test.provider"):
    return GraphRecipeProviderDescriptor(
        namespace=namespace,
        provider_version="1",
        domain_version="test-domain-v1",
        semantic_fingerprint="test-provider-semantics-v1",
    )


def _fragment(
    definition,
    source_indices,
    route,
    *,
    provider="test.provider",
    tasks=None,
    bindings=(),
    resources=(),
    submission=None,
    backends=(),
    capabilities=(),
    assembly_protocol=None,
    assembly_provider_namespace=None,
):
    sources = tuple(definition.sources[index] for index in source_indices)
    tasks = tasks or (_task(f"task_{route}", route),)
    return GraphRecipeFragment.create(
        definition,
        provider_namespace=provider,
        provider_version="1",
        provider_domain_version="test-domain-v1",
        fragment_key=f"fragment:{route}",
        coverage_region_ids=tuple(source.region_id for source in sources),
        tasks=tasks,
        binding_requirements=bindings,
        resources=resources,
        submission=submission,
        backend_requirements=backends,
        capability_requirements=capabilities,
        **(
            {}
            if assembly_protocol is None
            else {"assembly_protocol": assembly_protocol}
        ),
        **(
            {}
            if assembly_provider_namespace is None
            else {"assembly_provider_namespace": assembly_provider_namespace}
        ),
    )


def test_composer_fills_baseline_and_selects_two_disjoint_regions_independently():
    definition = _definition()
    composer = GraphRecipeComposer(definition)
    baseline = composer.compose()
    assert baseline.recipe_id == definition.baseline_recipe.recipe_id
    assert baseline.planned_physical_id == (
        definition.baseline_recipe.planned_physical_id
    )
    assert tuple(step.source for step in baseline.execution_steps) == (
        "baseline",
        "baseline",
        "baseline",
    )

    first = _fragment(definition, (0,), "first")
    third = _fragment(definition, (2,), "third")
    recipe = composer.compose((third, first))
    reordered = composer.compose((first, third))

    assert recipe == reordered
    json.dumps(recipe.to_dict(), sort_keys=True, allow_nan=False)
    assert recipe.fragment_ids == (first.fragment_id, third.fragment_id)
    assert len(recipe.region_selections) == len(definition.regions)
    assert {item.region_id for item in recipe.region_selections} == {
        region.region_id for region in definition.regions
    }
    assert recipe.baseline_coverage_region_ids == (
        definition.regions[0].region_id,
        definition.sources[1].region_id,
    )
    assert recipe.recipe_id != baseline.recipe_id
    assert recipe.planned_physical_id != baseline.planned_physical_id
    assert tuple(step.source for step in recipe.execution_steps) == (
        "fragment",
        "baseline",
        "fragment",
    )


def test_cross_region_fragment_composes_with_independent_resource_and_submission():
    definition = _definition()
    fusion = _fragment(
        definition,
        (0, 1),
        "fused_load_map",
        tasks=(
            _task(
                "fused",
                "fused_load_map",
                effects=(
                    ResourceEffect("values", GraphAccess.READ),
                    ResourceEffect("middle_b", GraphAccess.WRITE),
                ),
                temporaries=(TemporaryRequirement("fusion_tile", 128, 16),),
            ),
        ),
        bindings=(GraphFragmentBindingRequirement("values", ("ndarray",)),),
        resources=(
            GraphFragmentResourceRequirement(
                "fusion_workspace",
                "device_buffer",
                bytes=1024,
                alignment=256,
                ownership="graph_instance",
                lifetime="graph",
                exclusive_submission=True,
            ),
        ),
    )
    tail = _fragment(
        definition,
        (2,),
        "vector_store",
        tasks=(
            _task(
                "store",
                "vector_store",
                effects=(
                    ResourceEffect("middle_b", GraphAccess.READ),
                    ResourceEffect("output", GraphAccess.WRITE),
                ),
            ),
        ),
        bindings=(GraphFragmentBindingRequirement("output", ("ndarray",)),),
    )

    recipe = GraphRecipeComposer(definition).compose((fusion, tail))
    assert recipe.baseline_coverage_region_ids == (definition.regions[0].region_id,)
    assert recipe.declared_persistent_resource_bytes == 1024
    assert recipe.declared_transient_resource_bytes == 128
    assert recipe.exclusive_submission
    assert tuple(len(step.region_ids) for step in recipe.execution_steps) == (2, 1)


def test_composer_rejects_only_structural_and_physical_incompatibilities():
    definition = _definition()
    first = _fragment(definition, (0,), "first")
    overlapping = _fragment(definition, (0, 1), "overlapping")
    with pytest.raises(GraphRecipeCompositionError, match="overlapping"):
        GraphRecipeComposer(definition).compose((first, overlapping))

    incomplete_subtree = GraphRecipeFragment.create(
        definition,
        provider_namespace="test.control",
        provider_version="1",
        provider_domain_version="test-domain-v1",
        fragment_key="fragment:control",
        coverage_region_ids=(definition.regions[0].region_id,),
        tasks=(_task("control", "control"),),
    )
    with pytest.raises(GraphRecipeCompositionError, match="include its subtree"):
        GraphRecipeComposer(definition).compose((incomplete_subtree,))

    skipped_middle = _fragment(definition, (0, 2), "skipped_middle")
    with pytest.raises(GraphRecipeCompositionError, match="skip an intermediate"):
        GraphRecipeComposer(definition).compose((skipped_middle,))

    wrong_binding = _fragment(
        definition,
        (1,),
        "wrong_binding",
        bindings=(GraphFragmentBindingRequirement("missing", ("ndarray",)),),
    )
    with pytest.raises(GraphRecipeCompositionError, match="unknown public binding"):
        GraphRecipeComposer(definition).compose((wrong_binding,))

    partial_owned_graph = _fragment(
        definition,
        (1,),
        "partial-owned-graph",
        assembly_protocol=PROVIDER_OWNED_WHOLE_GRAPH_V1,
    )
    with pytest.raises(GraphRecipeCompositionError, match="exact-coverage"):
        GraphRecipeComposer(definition).compose((partial_owned_graph,))

    capability = _fragment(
        definition,
        (1,),
        "cooperative",
        backends=("cuda",),
        capabilities=("cooperative_groups",),
    )
    with pytest.raises(GraphRecipeCompositionError, match="unavailable capabilities"):
        GraphRecipeComposer(definition).compose((capability,))
    GraphRecipeComposer(
        definition,
        available_capabilities=("cooperative_groups",),
    ).compose((capability,))


def test_cross_queue_hazards_need_a_barrier_and_private_resources_cannot_collide():
    definition = _definition()
    producer = _fragment(
        definition,
        (0,),
        "producer",
        tasks=(
            _task(
                "producer",
                "producer",
                effects=(ResourceEffect("middle_b", GraphAccess.WRITE),),
            ),
        ),
        resources=(
            GraphFragmentResourceRequirement("workspace", "device_buffer", bytes=64),
        ),
        submission=GraphFragmentSubmissionRequirement(queue="compute"),
    )
    consumer = _fragment(
        definition,
        (2,),
        "consumer",
        tasks=(
            _task(
                "consumer",
                "consumer",
                effects=(ResourceEffect("middle_b", GraphAccess.READ),),
            ),
        ),
        resources=(
            GraphFragmentResourceRequirement("workspace", "device_buffer", bytes=128),
        ),
        submission=GraphFragmentSubmissionRequirement(queue="copy"),
    )
    composer = GraphRecipeComposer(definition)
    with pytest.raises(GraphRecipeCompositionError, match="incompatible resources"):
        composer.compose((producer, consumer))

    consumer_without_collision = _fragment(
        definition,
        (2,),
        "consumer_unique",
        tasks=consumer.tasks,
        resources=(
            GraphFragmentResourceRequirement(
                "consumer_workspace", "device_buffer", bytes=128
            ),
        ),
        submission=GraphFragmentSubmissionRequirement(queue="copy"),
    )
    with pytest.raises(GraphRecipeCompositionError, match="explicit barrier"):
        composer.compose((producer, consumer_without_collision))

    consumer_with_barrier = _fragment(
        definition,
        (2,),
        "consumer_barrier",
        tasks=consumer.tasks,
        resources=consumer_without_collision.resources,
        submission=GraphFragmentSubmissionRequirement(
            queue="copy",
            barrier_before=True,
        ),
    )
    recipe = composer.compose((producer, consumer_with_barrier))
    assert recipe.queues == ("compute", "copy")
    assert recipe.barrier_count == 1


def test_catalog_stages_explicit_composition_neighbors_and_physical_dedup():
    definition = _definition()
    read = ResourceEffect("values", GraphAccess.READ)
    write = ResourceEffect("middle_b", GraphAccess.WRITE)
    first_task = (
        _task(
            "provider_a_task",
            "same_physical_route",
            effects=(read, write),
        ),
    )
    second_task = (
        _task(
            "provider_b_task",
            "same_physical_route",
            effects=(write, read),
        ),
    )
    first_provider = _fragment(
        definition,
        (1,),
        "same_a",
        provider="provider.a",
        tasks=first_task,
        assembly_provider_namespace="provider.shared_assembly",
    )
    second_provider = _fragment(
        definition,
        (1,),
        "same_b",
        provider="provider.b",
        tasks=second_task,
        assembly_provider_namespace="provider.shared_assembly",
    )
    assert first_provider.fragment_id != second_provider.fragment_id
    assert first_provider.planned_physical_id == second_provider.planned_physical_id

    deduplicated = GraphRecipeCatalog(definition)
    deduplicated.register_fragment(first_provider)
    deduplicated.register_fragment(second_provider)
    admitted = deduplicated.build_single_region_stage()
    assert admitted[0].recipe.recipe_id == admitted[1].recipe.recipe_id
    assert len(deduplicated.entries(stage="single-region")) == 1
    assert len(deduplicated.physical_duplicates) == 1

    neighbor = _fragment(definition, (0,), "neighbor")
    base = _fragment(
        definition,
        (0,),
        "base",
    )
    tail = _fragment(definition, (2,), "tail")
    catalog = GraphRecipeCatalog(definition)

    class Provider:
        descriptor = _provider_descriptor()

        def fragments(self, requested_definition):
            assert requested_definition is definition
            return (base, tail)

        def resolve(self, requested_definition, fragment_key):
            assert requested_definition is definition
            return {
                base.fragment_key: base,
                tail.fragment_key: tail,
                neighbor.fragment_key: neighbor,
            }[fragment_key]

        def expand(self, requested_definition, fragment_key):
            assert requested_definition is definition
            return (neighbor,) if fragment_key == base.fragment_key else ()

    assert catalog.discover((Provider(),)) == (base, tail)
    single_entries = catalog.build_single_region_stage()
    base_entry = next(
        entry
        for entry in single_entries
        if entry.recipe.fragment_ids == (base.fragment_id,)
    )
    neighbor_entries = catalog.expand_neighbors(base_entry.recipe.recipe_id)
    assert len(neighbor_entries) == 1
    assert neighbor_entries[0].parent_recipe_ids == (base_entry.recipe.recipe_id,)
    assert neighbor_entries[0].recipe.fragment_ids == (neighbor.fragment_id,)

    composed = catalog.compose_compatible(
        ((base.fragment_id, tail.fragment_id),),
        parent_recipe_ids=tuple(entry.recipe.recipe_id for entry in single_entries),
    )
    assert len(composed) == 1
    assert composed[0].recipe.fragment_ids == (base.fragment_id, tail.fragment_id)
    assert len(catalog.entries(stage="compatible-composition")) == 1


def test_catalog_builds_budget_bounded_compatible_complete_recipes():
    definition = _definition()
    head_a = _fragment(definition, (0,), "head_a")
    head_b = _fragment(definition, (0,), "head_b")
    middle = _fragment(definition, (1,), "middle")
    tail = _fragment(definition, (2,), "tail")
    catalog = GraphRecipeCatalog(definition)

    class Provider:
        descriptor = _provider_descriptor()

        def fragments(self, requested_definition):
            assert requested_definition is definition
            return (head_a, head_b, middle, tail)

    catalog.discover((Provider(),))
    singletons = catalog.build_single_region_stage()
    admitted = catalog.build_compatible_stage(candidate_limit=3)

    assert len(singletons) == 4
    assert len(admitted) == 3
    assert all(len(entry.recipe.fragments) >= 2 for entry in admitted)
    assert all(entry.parent_recipe_ids for entry in admitted)
    assert all(
        len(entry.parent_recipe_ids) == len(entry.recipe.fragments)
        for entry in admitted
    )
    assert all(
        not (
            {head_a.fragment_id, head_b.fragment_id}
            <= set(entry.recipe.fragment_ids)
        )
        for entry in admitted
    )
    assert catalog.build_compatible_stage(candidate_limit=0) == ()


def test_provider_set_identity_is_order_independent_and_definition_bound():
    definition = _definition()

    class Provider:
        def __init__(self, namespace):
            self.descriptor = GraphRecipeProviderDescriptor(
                namespace=namespace,
                provider_version="1",
                domain_version="test-domain-v1",
                semantic_fingerprint=f"semantics:{namespace}",
            )

        def discover(self, requested_definition):
            assert requested_definition is definition
            return ()

    first = Provider("test.first")
    second = Provider("test.second")
    forward = GraphRecipeProviderSet(definition, (first, second))
    reverse = GraphRecipeProviderSet(definition, (second, first))

    assert forward.provider_registry_id == reverse.provider_registry_id
    assert forward.generation_domain_id == reverse.generation_domain_id
    assert tuple(item.namespace for item in forward.descriptors) == (
        "test.first",
        "test.second",
    )
    json.dumps(forward.to_dict(), sort_keys=True, allow_nan=False)


def test_provider_set_rejects_namespace_conflict_and_capability_drift():
    definition = _definition()

    class Provider:
        def __init__(self, descriptor):
            self.descriptor = descriptor

        def discover(self, _definition):
            return ()

    descriptor = _provider_descriptor("test.duplicate")
    with pytest.raises(GraphRecipeProviderError) as duplicate:
        GraphRecipeProviderSet(
            definition,
            (Provider(descriptor), Provider(descriptor)),
        )
    assert duplicate.value.error_key == "provider_namespace_duplicate"

    capability_descriptor = GraphRecipeProviderDescriptor(
        namespace="test.capability",
        provider_version="1",
        domain_version="test-domain-v1",
        semantic_fingerprint="test-capability-semantics-v1",
        required_capabilities=("cuda.graph.capture",),
    )
    with pytest.raises(GraphRecipeProviderError) as unavailable:
        GraphRecipeProviderSet(
            definition,
            (Provider(capability_descriptor),),
            available_capabilities=(),
        )
    assert unavailable.value.to_dict()["error_key"] == (
        "provider_capability_unavailable"
    )


def test_fragment_task_dag_rejects_cycles_before_catalog_admission():
    definition = _definition()
    with pytest.raises(ValueError, match="cycle"):
        _fragment(
            definition,
            (1,),
            "cycle",
            tasks=(
                _task("left", "left", depends_on=("right",)),
                _task("right", "right", depends_on=("left",)),
            ),
        )


def test_multi_region_fragment_does_not_cross_structural_parents_implicitly():
    left = SequentialRegion(
        (DispatchNode(name="left", logical_kernel_identity="kernel:left"),),
        name="left_branch",
    )
    right = SequentialRegion(
        (DispatchNode(name="right", logical_kernel_identity="kernel:right"),),
        name="right_branch",
    )
    root = SequentialRegion((left, right), name="graph")
    spec = SimpleNamespace(
        pre_optimization_ir_root=root,
        definition_semantic_root=root,
        definition_semantic_sources=(),
        ir_root=root,
        runtime_arg_names=frozenset(),
        fixed_runtime_args={},
        temporary_runtime_arg_names=frozenset(),
        derived_runtime_arg_names=frozenset(),
        execution_definition={
            "nodes": (),
            "dispatch_count": 2,
            "native_count": 0,
            "observation_count": 0,
            "structured_control_count": 0,
            "max_structured_depth": 0,
            "runtime_arg_count": 0,
            "fixed_runtime_arg_count": 0,
            "internal_storage_bytes": 0,
            "temporary_memory_plan": {},
        },
    )
    definition = GraphDefinition._from_graph_spec(spec, "cuda")
    implicit_cross_parent = _fragment(definition, (0, 1), "implicit_cross_parent")
    with pytest.raises(GraphRecipeCompositionError, match="sequential parent"):
        GraphRecipeComposer(definition).compose((implicit_cross_parent,))

    whole_subtree = GraphRecipeFragment.create(
        definition,
        provider_namespace="test.explicit_topology",
        provider_version="1",
        provider_domain_version="test-domain-v1",
        fragment_key="fragment:whole_graph",
        coverage_region_ids=tuple(region.region_id for region in definition.regions),
        tasks=(_task("whole_graph", "explicit_whole_graph_dag"),),
    )
    recipe = GraphRecipeComposer(definition).compose((whole_subtree,))
    assert recipe.baseline_coverage_region_ids == ()


@test_utils.test(arch=ti.cpu)
def test_actual_graph_definition_feeds_the_fragment_composer_without_v1_space():
    @ti.kernel
    def add_one(
        values: ti.types.ndarray(dtype=ti.i32, ndim=1),
        middle: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in values:
            middle[i] = values[i] + 1

    @ti.kernel
    def double(
        middle: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in middle:
            output[i] = middle[i] * 2

    values = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "values", ti.i32, ndim=1)
    middle = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "middle", ti.i32, ndim=1)
    output = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(add_one, values, middle)
    builder.dispatch(double, middle, output)
    definition = builder.freeze()

    combined = GraphRecipeFragment.create(
        definition,
        provider_namespace="test.actual_graph_fusion",
        provider_version="1",
        provider_domain_version="test-domain-v1",
        fragment_key="fragment:fused_add_double",
        coverage_region_ids=tuple(source.region_id for source in definition.sources),
        tasks=(
            _task(
                "fused_add_double",
                "fused_add_double",
                effects=(
                    ResourceEffect("values", GraphAccess.READ),
                    ResourceEffect("output", GraphAccess.WRITE),
                ),
            ),
        ),
        binding_requirements=(
            GraphFragmentBindingRequirement("values"),
            GraphFragmentBindingRequirement("output"),
        ),
        backend_requirements=("cpu",),
    )
    recipe = GraphRecipeComposer(definition).compose((combined,))

    assert recipe.semantic_graph_id == definition.semantic_graph_id
    assert set(recipe.baseline_coverage_region_ids) == {
        region.region_id
        for region in definition.regions
        if region.region_id not in combined.coverage_region_ids
    }
    assert not hasattr(recipe, "executable_optimization_spec")
