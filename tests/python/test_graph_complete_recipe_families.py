from itertools import pairwise

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge._lib import core as ti_core

from tests import test_utils

_RECIPE_ENVIRONMENTS = (
    "TAICHI_FORGE_INTERNAL_MAP_FUSION",
    "TAICHI_FORGE_INTERNAL_GRAPH_MEMORY_RECIPE",
    "TAICHI_FORGE_INTERNAL_GRAPH_REDUCTION_RECIPE",
    "TAICHI_FORGE_INTERNAL_GRAPH_NATIVE_ALGORITHM_RECIPE",
    "TAICHI_FORGE_INTERNAL_STRUCTURED_CONTROL_RECIPE",
    "TI_CUDA_BOUNDED_DISPATCH_MODE",
    "TI_GRAPH_CUDA_BOUNDED_UPDATE_POLICY",
)


def _family_fragments(catalog, family):
    namespace = f"taichi_forge.graph.{family}"
    return tuple(fragment for fragment in catalog.fragments
                 if fragment.provider_namespace == namespace)


def _selection(fragment):
    return fragment.materializer.selection


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_complete_recipe_composes_fusion_and_memory_without_environment(
        monkeypatch):
    count = 1027

    @ti.kernel
    def scale(
            source: ti.types.ndarray(dtype=ti.f32, ndim=1),
            temporary: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(count):
            temporary[index] = source[index] * 2.0

    @ti.kernel
    def bias(
            temporary: ti.types.ndarray(dtype=ti.f32, ndim=1),
            staged_input: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(count):
            staged_input[index] = temporary[index] + 1.0

    @ti.kernel
    def stencil(
            staged_input: ti.types.ndarray(dtype=ti.f32, ndim=1),
            output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(1, count - 1):
            output[index] = (staged_input[index - 1] + staged_input[index] +
                             staged_input[index + 1])

    symbolic = {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1)
        for name in ("source", "temporary", "staged_input", "output")
    }
    monkeypatch.setenv("TAICHI_FORGE_INTERNAL_MAP_FUSION", "baseline")
    builder = ti.graph.GraphBuilder()
    builder.dispatch(scale, symbolic["source"], symbolic["temporary"])
    builder.dispatch(bias, symbolic["temporary"], symbolic["staged_input"])
    builder.dispatch(stencil, symbolic["staged_input"], symbolic["output"])
    definition = builder.freeze()
    catalog = definition.recipe_catalog()

    fusion = next(fragment
                  for fragment in _family_fragments(catalog, "map_fusion")
                  if len(fragment.coverage_region_ids) == 2)
    memory_source = definition._runtime_spec._graph_memory_sources[2]
    staged_id = next(manifest.recipe_id
                     for manifest in memory_source.manifests()
                     if manifest.strategy == "shared_staged_1d")
    memory = next(fragment
                  for fragment in _family_fragments(catalog, "graph_memory")
                  if _selection(fragment).choice_id == staged_id)
    assert not set(fusion.coverage_region_ids).intersection(
        memory.coverage_region_ids)
    entry = catalog.compose(
        (fusion.fragment_id, memory.fragment_id),
        stage="compatible-composition",
        parent_recipe_ids=(catalog.baseline.recipe.recipe_id, ),
    )

    for name in _RECIPE_ENVIRONMENTS:
        monkeypatch.setenv(name, "intentionally-invalid-after-freeze")
    with definition.materialization_context() as context:
        materialized = context.materialize(entry.recipe)
        graph = materialized.executor
        assert graph.definition is definition
        assert materialized.manifest.recipe_id == entry.recipe.recipe_id
        task_manifest = graph.task_manifest()
        assert any(task.source_dispatch_count == 2 for task in task_manifest)
        assert any(task.requested_memory_strategy == "shared_staged_1d"
                   for task in task_manifest)

        source = ti.ndarray(ti.f32, shape=count)
        temporary = ti.ndarray(ti.f32, shape=count)
        staged_input = ti.ndarray(ti.f32, shape=count)
        output = ti.ndarray(ti.f32, shape=count)
        values = np.arange(count, dtype=np.float32) * 0.25
        source.from_numpy(values)
        output.fill(0)
        graph.run(
            graph.bind({
                "source": source,
                "temporary": temporary,
                "staged_input": staged_input,
                "output": output,
            }))
        ti.sync()
        intermediate = values * 2.0 + 1.0
        expected = np.zeros(count, dtype=np.float32)
        expected[
            1:-1] = intermediate[:-2] + intermediate[1:-1] + intermediate[2:]
        np.testing.assert_allclose(output.to_numpy(), expected, rtol=0, atol=0)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_complete_reduction_recipe_owns_workspace_and_survives_bad_environment(
    monkeypatch, ):
    count = 4097
    values_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                              "values",
                              ti.i32,
                              ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                              "output",
                              ti.i32,
                              ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.reduce(values_arg, output_arg, count=count)
    definition = builder.freeze()
    catalog = definition.recipe_catalog()
    source = definition._runtime_spec._graph_reduction_sources[0]
    phased_id = next(manifest.recipe_id for manifest in source.manifests()
                     if manifest.strategy == "block_partial_finalize")
    fragment = next(item
                    for item in _family_fragments(catalog, "graph_reduction")
                    if _selection(item).choice_id == phased_id)
    recipe = catalog.compose(
        (fragment.fragment_id, ),
        stage="single-region",
        parent_recipe_ids=(catalog.baseline.recipe.recipe_id, ),
    ).recipe
    assert recipe.declared_persistent_resource_bytes == (
        (count + 1023) // 1024) * 4

    for name in _RECIPE_ENVIRONMENTS:
        monkeypatch.setenv(name, "intentionally-invalid-after-freeze")
    with definition.materialization_context() as context:
        materialized = context.materialize(recipe)
        graph = materialized.executor
        assert graph.execution_stats().memory.persistent_bytes >= (
            recipe.declared_persistent_resource_bytes)
        values = ti.ndarray(ti.i32, shape=count)
        output = ti.ndarray(ti.i32, shape=1)
        host = ((np.arange(count, dtype=np.int32) % 23) - 11).astype(np.int32)
        values.from_numpy(host)
        graph.run(graph.bind({"values": values, "output": output}))
        ti.sync()
        expected = np.asarray(host.sum(dtype=np.int64), dtype=np.int32).item()
        assert output.to_numpy()[0] == expected


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_complete_native_algorithm_recipe_materializes_both_physical_routes(
    monkeypatch, ):
    capacity = 8192
    offsets = np.arange(0, capacity + 1, 2048, dtype=np.int32)
    layout = ti.algorithms.SegmentedLayout.from_offsets(
        offsets,
        capacity=capacity,
    )
    values = ti.ndarray(ti.i32, shape=capacity)
    output = ti.ndarray(ti.i32, shape=capacity)
    builder = ti.graph.GraphBuilder()
    builder.segmented_scan(values, layout, output)
    definition = builder.freeze()
    catalog = definition.recipe_catalog()
    fragments = _family_fragments(catalog, "native_algorithm")
    assert len(fragments) == 1

    host = ((np.arange(capacity, dtype=np.int64) % 7) + 1).astype(np.int32)
    expected = np.empty_like(host)
    for begin, end in pairwise(offsets):
        expected[begin:end] = np.cumsum(host[begin:end], dtype=np.int32)

    for name in _RECIPE_ENVIRONMENTS:
        monkeypatch.setenv(name, "intentionally-invalid-after-freeze")
    physical_identities = set()
    persistent_bytes = set()
    with definition.materialization_context() as context:
        recipes = [catalog.baseline.recipe]
        recipes.extend(
            catalog.compose(
                (fragment.fragment_id, ),
                stage="single-region",
                parent_recipe_ids=(catalog.baseline.recipe.recipe_id, ),
            ).recipe for fragment in fragments)
        for recipe in recipes:
            materialized = context.materialize(recipe)
            values.from_numpy(host)
            output.fill(0)
            materialized.executor.run({})
            ti.sync()
            np.testing.assert_array_equal(output.to_numpy(), expected)
            physical_identities.add(
                materialized.manifest.materialized_physical_id)
            persistent_bytes.add(
                materialized.manifest.persistent_requested_bytes)
    assert len(physical_identities) == 2
    assert persistent_bytes == {0, layout.num_segments * 4}


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_complete_bounded_recipe_replays_all_scope_strategies_without_environment(
    monkeypatch, ):
    probe = dict(ti_core.cuda_bounded_dispatch_probe())
    if not probe["exact_device_grid_available"]:
        pytest.skip(probe["unavailable_reason"])

    capacity = 257
    block_dim = 32

    @ti.kernel
    def publish(
            requested: ti.i32,
            extent: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        ti.device_extent_publish(extent, capacity, requested)

    @ti.kernel
    def consume(
            extent: ti.types.ndarray(dtype=ti.i32, ndim=1),
            observed: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        ti.loop_config(block_dim=block_dim)
        for index in range(capacity):
            if index < ti.device_extent_count(extent):
                ti.atomic_add(observed[0], index + 1)

    requested_arg = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "requested", ti.i32)
    extent_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                              "extent",
                              ti.i32,
                              ndim=1)
    first_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "first", ti.i32, ndim=1)
    second_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                              "second",
                              ti.i32,
                              ndim=1)

    monkeypatch.setenv("TI_CUDA_BOUNDED_DISPATCH_MODE", "auto")
    monkeypatch.setenv("TI_GRAPH_CUDA_BOUNDED_UPDATE_POLICY", "auto")
    builder = ti.graph.GraphBuilder()
    builder.dispatch(publish, requested_arg, extent_arg)
    for output_arg in (first_arg, second_arg):
        builder.dispatch_bounded(
            consume,
            extent_arg,
            output_arg,
            extent=extent_arg,
            capacity=capacity,
            block_dim=block_dim,
        )
    definition = builder.freeze()
    catalog = definition.recipe_catalog()
    fragments = _family_fragments(catalog, "bounded_execution")
    expected_strategies = {
        "logical_exact",
        "adaptive_per_node",
        "adaptive_grouped",
        "masked_capacity",
    }
    assert {_selection(item).materialization_choice
            for item in fragments} == (expected_strategies - {"logical_exact"})

    for name in _RECIPE_ENVIRONMENTS:
        monkeypatch.setenv(name, "intentionally-invalid-after-freeze")
    extent = ti.DeviceExtent(capacity)
    first = ti.ndarray(ti.i32, shape=1)
    second = ti.ndarray(ti.i32, shape=1)
    arguments = {
        "requested": 17,
        "extent": extent,
        "first": first,
        "second": second,
    }
    physical_identities = {}
    persistent_control_bytes = {}
    with definition.materialization_context() as context:
        recipes = [("logical_exact", catalog.baseline.recipe)]
        recipes.extend((
            _selection(fragment).materialization_choice,
            catalog.compose(
                (fragment.fragment_id, ),
                stage="single-region",
                parent_recipe_ids=(catalog.baseline.recipe.recipe_id, ),
            ).recipe,
        ) for fragment in fragments)
        for strategy, recipe in recipes:
            materialized = context.materialize(recipe)
            first.fill(0)
            second.fill(0)
            materialized.executor.run(arguments)
            ti.sync()
            expected = arguments["requested"] * (arguments["requested"] +
                                                 1) // 2
            assert int(first.to_numpy()[0]) == expected
            assert int(second.to_numpy()[0]) == expected
            physical_identities[strategy] = (
                materialized.manifest.materialized_physical_id)
            persistent_control_bytes[strategy] = (
                materialized.executor.execution_stats(
                ).memory.persistent_bounded_control_bytes)

    assert len(set(
        physical_identities.values())) == len(expected_strategies), (
            physical_identities)
    assert persistent_control_bytes["logical_exact"] == 0
    assert persistent_control_bytes["masked_capacity"] == 0
    assert persistent_control_bytes["adaptive_per_node"] > 0
    assert persistent_control_bytes["adaptive_grouped"] > 0


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_complete_structured_control_recipe_rebuilds_both_routes_without_environment(
    monkeypatch, ):
    capabilities = dict(ti_core.cuda_conditional_graph_capabilities())
    if not capabilities.get("general_graph_exact_control_available", False):
        pytest.skip("general CUDA conditional Graph is unavailable")
    if not capabilities.get("internal_masked_graph_available", False):
        pytest.skip("internal masked CUDA Graph control is unavailable")

    @ti.kernel
    def initialize(
            state: ti.types.ndarray(dtype=ti.i32, ndim=0),
            predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
            counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
    ):
        state[None] = 0
        predicate[None] = 0
        counter[None] = 0

    @ti.kernel
    def condition(
            state: ti.types.ndarray(dtype=ti.i32, ndim=0),
            predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
            target: ti.i32,
    ):
        predicate[None] = int(state[None] < target)

    @ti.kernel
    def step(
            state: ti.types.ndarray(dtype=ti.i32, ndim=0),
            predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
            counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
    ):
        if predicate[None] != 0:
            state[None] += 1
            counter[None] += 1

    def scalar(name):
        return ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.i32, ndim=0)

    state = scalar("state")
    predicate = scalar("predicate")
    counter = scalar("counter")
    target = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "target", ti.i32)
    monkeypatch.delenv("TI_GRAPH_CUDA_FORCE_MASKED_CONTROL", raising=False)
    monkeypatch.setenv(
        "TAICHI_FORGE_INTERNAL_STRUCTURED_CONTROL_RECIPE",
        "cuda_conditional_graph",
    )
    builder = ti.graph.GraphBuilder()
    builder.dispatch(initialize, state, predicate, counter)
    condition_region = builder.create_sequential()
    condition_region.dispatch(condition, state, predicate, target)
    body = builder.create_sequential()
    body.dispatch(step, state, predicate, counter)
    builder.while_loop(
        condition_region,
        body,
        predicate=predicate,
        control_inputs=(state, target),
        carried_state=(state, ),
        counter=counter,
        max_iterations=8,
        name="complete_recipe_control",
    )
    definition = builder.freeze()
    catalog = definition.recipe_catalog()
    fragments = _family_fragments(catalog, "structured_control")
    assert len(fragments) == 1, tuple(
        (region.kind, region.path, region.parent_region_id)
        for region in definition.regions)

    for name in _RECIPE_ENVIRONMENTS:
        monkeypatch.setenv(name, "intentionally-invalid-after-freeze")
    physical_identities = set()
    routes = set()
    with definition.materialization_context() as context:
        recipes = [catalog.baseline.recipe]
        recipes.extend(
            catalog.compose(
                (fragment.fragment_id, ),
                stage="single-region",
                parent_recipe_ids=(catalog.baseline.recipe.recipe_id, ),
            ).recipe for fragment in fragments)
        for recipe in recipes:
            materialized = context.materialize(recipe)
            arguments = {
                "state": ti.ndarray(ti.i32, shape=()),
                "predicate": ti.ndarray(ti.i32, shape=()),
                "counter": ti.ndarray(ti.i32, shape=()),
                "target": 5,
            }
            materialized.executor.run(arguments)
            report = materialized.executor.control_flow_stats()[0]
            assert report.logical_iterations == 5
            assert arguments["state"].to_numpy()[()] == 5
            assert arguments["counter"].to_numpy()[()] == 5
            routes.add(report.lowering)
            physical_identities.add(
                materialized.manifest.materialized_physical_id)

    assert routes == {"cuda_conditional_graph", "cuda_masked_bounded_graph"}
    assert len(physical_identities) == 2
