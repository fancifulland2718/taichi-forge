from itertools import pairwise

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge._lib import core as ti_core
from taichi_forge.graph._recipes.physical import observe_graph_physical_manifest
from taichi_forge.lang.exception import TaichiRuntimeError

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
    return tuple(
        fragment
        for fragment in catalog.fragments
        if fragment.provider_namespace == namespace
    )


def _selection(fragment):
    return fragment.materializer.selection


@test_utils.test(
    arch=ti.cuda,
    offline_cache=False,
    cuda_sparse_pool_auto_size=True,
    cuda_sparse_per_snode_pool=True,
    listgen_static_grid_dim=False,
)
def test_complete_recipe_owns_sparse_listgen_grid_for_migrating_active_set():
    block_count = 33
    block_size = 8
    domain_size = block_count * block_size
    values = ti.field(ti.i32)
    fields = ti.FieldsBuilder()
    pointer = fields.pointer(
        ti.i,
        block_count,
        vk_max_active=block_count,
    )
    pointer.bitmasked(ti.i, block_size).place(values)
    tree = fields.finalize()
    output = ti.ndarray(ti.i32, shape=2)

    @ti.kernel
    def deactivate_all():
        for block in range(block_count):
            ti.deactivate(pointer, block)

    @ti.kernel
    def activate_phase(phase: ti.i32):
        for index in range(domain_size):
            block = index // block_size
            local = index % block_size
            if block % 3 == phase and (local + phase) % 4 == 1:
                values[index] = 1000 * (phase + 1) + index

    @ti.kernel
    def clear(result: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        result[0] = 0
        result[1] = 0

    @ti.kernel
    def reduce_active(result: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for index in values:
            ti.atomic_add(result[0], values[index])
            ti.atomic_add(result[1], 1)

    result_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY,
        "result",
        ti.i32,
        ndim=1,
    )
    builder = ti.graph.GraphBuilder()
    builder.dispatch(clear, result_arg)
    builder.dispatch(reduce_active, result_arg)
    definition = builder.freeze()

    sources = definition._runtime_spec._graph_sparse_traversal_sources
    assert len(sources) == 1
    manifests = sources[0].manifests()
    assert tuple(manifest.strategy for manifest in manifests) == (
        "saturating",
        "parent_capacity_bound",
    )
    bounded_manifest = manifests[1]
    listgens = bounded_manifest.to_dict()["listgen_tasks"]
    assert len(listgens) == 2
    assert all(task["policy"] == "parent_capacity_bound" for task in listgens)
    assert all(
        task["actual_grid"]
        == min(task["selected_grid"], task["parent_grid_bound"])
        for task in listgens
    )
    assert any(
        task["actual_grid"] < task["selected_grid"] for task in listgens
    )

    catalog = definition.recipe_catalog()
    sparse_fragments = _family_fragments(catalog, "sparse_traversal")
    assert len(sparse_fragments) == 1
    fragment = sparse_fragments[0]
    entry = catalog.compose(
        (fragment.fragment_id,),
        stage="sparse-traversal",
        parent_recipe_ids=(catalog.baseline.recipe.recipe_id,),
    )
    session = definition.search_recipes(
        engine="compileiq",
        target=ti.graph.GraphOptimizationTarget(
            objectives=(("device_time_ns", "min"),)
        ),
        budget=ti.graph.GraphSearchBudget(evaluation_limit=2),
    )
    assert len(session.recipes) == 2
    assert any(
        handle.manifest.families == ("sparse_traversal",)
        for handle in session.recipes
    )

    prog = ti.lang.impl.get_runtime().prog
    with definition.materialization_context() as context:
        product = context.materialize(entry.recipe)
        task_manifests = product.executor.task_manifest()
        physical_listgens = tuple(
            task for task in task_manifests if task.sparse_list_op == 2
        )
        assert len(physical_listgens) == 2
        assert all(
            task.requested_sparse_list_policy == "parent_capacity_bound"
            and task.actual_grid_size
            == min(task.selected_grid_size, task.sparse_list_parent_grid_bound)
            for task in physical_listgens
        )
        bindings = product.executor.bind({"result": output})
        prog._debug_reset_sparse_listgen_stats()
        for phase in (0, 1):
            deactivate_all()
            activate_phase(phase)
            expected_values = [
                1000 * (phase + 1) + index
                for index in range(domain_size)
                if index // block_size % 3 == phase
                and (index % block_size + phase) % 4 == 1
            ]
            for _ in range(4):
                product.executor.run(bindings)
            ti.sync()
            np.testing.assert_array_equal(
                output.to_numpy(),
                np.asarray((sum(expected_values), len(expected_values)), np.int32),
            )

        stats = dict(prog._debug_sparse_snode_tree_stats(tree.id))["listgen"]
        totals = dict(stats["totals"])
        assert totals["rebuilds"] >= 4
        assert totals["reuse_hits"] >= 8
        assert totals["candidate_slots_dispatched"] == 2 * sum(
            task.actual_grid_size * task.selected_block_size
            for task in physical_listgens
        )
        graph_stats = product.executor._graph_stats[0]
        assert graph_stats["captures"] == 0
        assert graph_stats["last_path"] == "ordinary_fallback"

    tree.destroy()


@test_utils.test(arch=ti.cuda, offline_cache=False, kernel_profiler=True)
def test_complete_recipe_searches_and_materializes_offload_phase_fusion():
    count = (1 << 17) + 19

    @ti.kernel
    def three_phase(
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        first: ti.types.ndarray(dtype=ti.i32, ndim=1),
        second: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for index in range(count):
            first[index] = source[index] * 2 + 1
        for index in range(count):
            second[index] = first[index] * 3 - 4
        for index in range(count):
            output[index] = second[index] ^ 0x55AA

    symbolic = {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.i32, ndim=1)
        for name in ("source", "first", "second", "output")
    }
    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        three_phase,
        symbolic["source"],
        symbolic["first"],
        symbolic["second"],
        symbolic["output"],
    )
    definition = builder.freeze()
    source = definition._runtime_spec._graph_offload_fusion_sources[0]
    manifests = source.manifests()
    fused_manifest = next(
        manifest
        for manifest in manifests
        if manifest.strategy == "exact_pointwise_phase_fusion"
        and len(manifest.to_dict()["materialized_tasks"]) == 1
    )
    payload = fused_manifest.to_dict()
    assert len(payload["source_task_lineage"]) == 1
    assert len(payload["source_task_lineage"][0]) == 3
    assert payload["fusion_groups"] == [[0, 1, 2]]

    catalog = definition.recipe_catalog()
    fragment = next(
        item
        for item in _family_fragments(catalog, "offload_phase_fusion")
        if _selection(item).choice_id == fused_manifest.recipe_id
    )
    entry = catalog.compose(
        (fragment.fragment_id,),
        stage="compiler-ir-topology",
        parent_recipe_ids=(catalog.baseline.recipe.recipe_id,),
    )
    session = definition.search_recipes(
        engine="compileiq",
        target=ti.graph.GraphOptimizationTarget(
            objectives=(("device_time_ns", "min"),)
        ),
        budget=ti.graph.GraphSearchBudget(evaluation_limit=16),
    )
    assert any(
        handle.manifest.families == ("offload_phase_fusion",)
        for handle in session.recipes
    )

    with definition.materialization_context() as context:
        product = context.materialize(entry.recipe)
        task_manifest = tuple(
            task
            for task in product.executor.task_manifest()
            if task.task_type == "range_for"
        )
        assert len(task_manifest) == 1
        assert (
            task_manifest[0].optimization_spec_id
            == payload["offload_compilation_identity"]
        )

        arrays = {name: ti.ndarray(ti.i32, shape=count) for name in symbolic}
        host = np.arange(count, dtype=np.int32) - 31
        arrays["source"].from_numpy(host)
        arrays["output"].fill(0)
        bindings = product.executor.bind(arrays)
        published = product.executor.binding_statistics()
        for _ in range(4):
            product.executor.run(bindings)
        ti.sync()
        replayed = product.executor.binding_statistics()
        assert replayed["version_builds"] == published["version_builds"]
        assert replayed["raw_replay_validations"] == published["raw_replay_validations"]
        assert replayed["version_fast_replays"] == (
            published["version_fast_replays"] + 4
        )
        expected = ((host * 2 + 1) * 3 - 4) ^ 0x55AA
        np.testing.assert_array_equal(arrays["output"].to_numpy(), expected)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_complete_recipe_excludes_cross_lane_offload_phase_fusion():
    count = 4096

    @ti.kernel
    def shifted_read(
        data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for index in range(count):
            data[index] = index + 1
        for index in range(count):
            output[index] = data[(index + 1) % count]

    data = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "data", ti.i32, ndim=1)
    output = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(shifted_read, data, output)
    definition = builder.freeze()
    source = definition._runtime_spec._graph_offload_fusion_sources[0]

    assert source.manifests() == ()
    assert "non-pointwise external access" in source.candidate_failure
    assert not _family_fragments(definition.recipe_catalog(), "offload_phase_fusion")


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_complete_recipe_composes_fusion_and_memory_without_environment(monkeypatch):
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
            output[index] = (
                staged_input[index - 1] + staged_input[index] + staged_input[index + 1]
            )

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

    fusion = next(
        fragment
        for fragment in _family_fragments(catalog, "map_fusion")
        if len(fragment.coverage_region_ids) == 2
    )
    memory_source = definition._runtime_spec._graph_memory_sources[2]
    staged_id = next(
        manifest.recipe_id
        for manifest in memory_source.manifests()
        if manifest.strategy == "shared_staged_1d"
    )
    memory = next(
        fragment
        for fragment in _family_fragments(catalog, "graph_memory")
        if _selection(fragment).choice_id == staged_id
    )
    assert not set(fusion.coverage_region_ids).intersection(memory.coverage_region_ids)
    entry = catalog.compose(
        (fusion.fragment_id, memory.fragment_id),
        stage="compatible-composition",
        parent_recipe_ids=(catalog.baseline.recipe.recipe_id,),
    )
    public_session = definition.search_recipes(
        engine="compileiq",
        target=ti.graph.GraphOptimizationTarget(
            objectives=(("device_time_ns", "min"),)
        ),
        budget=ti.graph.GraphSearchBudget(evaluation_limit=64),
    )
    assert any(
        set(handle.manifest.families) == {"map_fusion", "graph_memory"}
        and handle.manifest.selected_fragment_count == 2
        for handle in public_session.recipes
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
        assert any(
            task.requested_memory_strategy == "shared_staged_1d"
            for task in task_manifest
        )

        source = ti.ndarray(ti.f32, shape=count)
        temporary = ti.ndarray(ti.f32, shape=count)
        staged_input = ti.ndarray(ti.f32, shape=count)
        output = ti.ndarray(ti.f32, shape=count)
        values = np.arange(count, dtype=np.float32) * 0.25
        source.from_numpy(values)
        output.fill(0)
        graph.run(
            graph.bind(
                {
                    "source": source,
                    "temporary": temporary,
                    "staged_input": staged_input,
                    "output": output,
                }
            )
        )
        ti.sync()
        intermediate = values * 2.0 + 1.0
        expected = np.zeros(count, dtype=np.float32)
        expected[1:-1] = intermediate[:-2] + intermediate[1:-1] + intermediate[2:]
        np.testing.assert_allclose(output.to_numpy(), expected, rtol=0, atol=0)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_complete_reduction_recipe_owns_workspace_and_survives_bad_environment(
    monkeypatch,
):
    count = 4097
    values_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "values", ti.i32, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.reduce(values_arg, output_arg, count=count)
    definition = builder.freeze()
    catalog = definition.recipe_catalog()
    source = definition._runtime_spec._graph_reduction_sources[0]
    phased_id = next(
        manifest.recipe_id
        for manifest in source.manifests()
        if manifest.strategy == "block_partial_finalize"
    )
    fragment = next(
        item
        for item in _family_fragments(catalog, "graph_reduction")
        if _selection(item).choice_id == phased_id
    )
    recipe = catalog.compose(
        (fragment.fragment_id,),
        stage="single-region",
        parent_recipe_ids=(catalog.baseline.recipe.recipe_id,),
    ).recipe
    assert recipe.declared_persistent_resource_bytes == ((count + 1023) // 1024) * 4

    for name in _RECIPE_ENVIRONMENTS:
        monkeypatch.setenv(name, "intentionally-invalid-after-freeze")
    with definition.materialization_context() as context:
        materialized = context.materialize(recipe)
        graph = materialized.executor
        assert graph.execution_stats().memory.persistent_bytes >= (
            recipe.declared_persistent_resource_bytes
        )
        values = ti.ndarray(ti.i32, shape=count)
        output = ti.ndarray(ti.i32, shape=1)
        host = ((np.arange(count, dtype=np.int32) % 23) - 11).astype(np.int32)
        values.from_numpy(host)
        graph.run(graph.bind({"values": values, "output": output}))
        ti.sync()
        expected = np.asarray(host.sum(dtype=np.int64), dtype=np.int32).item()
        assert output.to_numpy()[0] == expected


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_complete_native_algorithm_recipe_materializes_all_physical_routes(
    monkeypatch,
):
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
    assert len(fragments) == 3

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
                (fragment.fragment_id,),
                stage="single-region",
                parent_recipe_ids=(catalog.baseline.recipe.recipe_id,),
            ).recipe
            for fragment in fragments
        )
        for recipe in recipes:
            materialized = context.materialize(recipe)
            values.from_numpy(host)
            output.fill(0)
            materialized.executor.run({})
            ti.sync()
            np.testing.assert_array_equal(output.to_numpy(), expected)
            physical_identities.add(materialized.manifest.materialized_physical_id)
            persistent_bytes.add(materialized.manifest.persistent_requested_bytes)
    assert len(physical_identities) == 4
    assert persistent_bytes == {0, layout.num_segments * 4}


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_complete_bounded_recipe_replays_all_scope_strategies_without_environment(
    monkeypatch,
):
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
    extent_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "extent", ti.i32, ndim=1)
    first_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "first", ti.i32, ndim=1)
    second_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "second", ti.i32, ndim=1)

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
    assert {_selection(item).materialization_choice for item in fragments} == (
        expected_strategies - {"logical_exact"}
    )

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
    manifest_persistent_bytes = {}
    execution_persistent_bytes = {}
    bounded_resource_bytes = {}
    with definition.materialization_context() as context:
        recipes = [("logical_exact", catalog.baseline.recipe)]
        recipes.extend(
            (
                _selection(fragment).materialization_choice,
                catalog.compose(
                    (fragment.fragment_id,),
                    stage="single-region",
                    parent_recipe_ids=(catalog.baseline.recipe.recipe_id,),
                ).recipe,
            )
            for fragment in fragments
        )
        for strategy, recipe in recipes:
            materialized = context.materialize(recipe)
            first.fill(0)
            second.fill(0)
            materialized.executor.run(arguments)
            ti.sync()
            observed_manifest = observe_graph_physical_manifest(
                definition,
                recipe,
                materialized.executor,
            )
            expected = arguments["requested"] * (arguments["requested"] + 1) // 2
            assert int(first.to_numpy()[0]) == expected
            assert int(second.to_numpy()[0]) == expected
            physical_identities[strategy] = observed_manifest.materialized_physical_id
            memory = materialized.executor.execution_stats().memory
            persistent_control_bytes[strategy] = memory.persistent_bounded_control_bytes
            execution_persistent_bytes[strategy] = memory.persistent_bytes
            manifest_persistent_bytes[strategy] = (
                observed_manifest.persistent_requested_bytes
            )
            bounded_resource_bytes[strategy] = sum(
                resource.requested_bytes
                for resource in observed_manifest.resources
                if resource.kind == "bounded_control_state"
            )

    assert len(set(physical_identities.values())) == len(
        expected_strategies
    ), physical_identities
    assert persistent_control_bytes["logical_exact"] == 0
    assert persistent_control_bytes["masked_capacity"] == 0
    assert persistent_control_bytes["adaptive_per_node"] > 0
    assert persistent_control_bytes["adaptive_grouped"] > 0
    assert manifest_persistent_bytes == execution_persistent_bytes
    assert bounded_resource_bytes == persistent_control_bytes


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_complete_bounded_recipe_composes_independent_publication_groups(monkeypatch):
    probe = dict(ti_core.cuda_bounded_dispatch_probe())
    if not probe["exact_device_grid_available"]:
        pytest.skip(probe["unavailable_reason"])

    capacity = 257

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
        ti.loop_config(block_dim=32)
        for index in range(capacity):
            if index < ti.device_extent_count(extent):
                ti.atomic_add(observed[0], index + 1)

    def ndarray_arg(name):
        return ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.i32, ndim=1)

    requested_a = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "requested_a", ti.i32)
    requested_b = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "requested_b", ti.i32)
    extent_a_arg = ndarray_arg("extent_a")
    extent_b_arg = ndarray_arg("extent_b")
    output_a_arg = ndarray_arg("output_a")
    output_b_arg = ndarray_arg("output_b")
    monkeypatch.setenv("TI_CUDA_BOUNDED_DISPATCH_MODE", "auto")
    monkeypatch.setenv("TI_GRAPH_CUDA_BOUNDED_UPDATE_POLICY", "auto")
    builder = ti.graph.GraphBuilder()
    builder.dispatch(publish, requested_a, extent_a_arg)
    builder.dispatch_bounded(
        consume,
        extent_a_arg,
        output_a_arg,
        extent=extent_a_arg,
        capacity=capacity,
        block_dim=32,
    )
    builder.dispatch(publish, requested_b, extent_b_arg)
    builder.dispatch_bounded(
        consume,
        extent_b_arg,
        output_b_arg,
        extent=extent_b_arg,
        capacity=capacity,
        block_dim=32,
    )
    definition = builder.freeze()
    catalog = definition.recipe_catalog()
    fragments = _family_fragments(catalog, "bounded_execution")
    by_source = {}
    for fragment in fragments:
        selection = _selection(fragment)
        by_source.setdefault(selection.source_key, {})[
            selection.materialization_choice
        ] = fragment
    assert len(by_source) == 2
    assert all("masked_capacity" in choices for choices in by_source.values())
    masked = tuple(
        choices["masked_capacity"].fragment_id for choices in by_source.values()
    )
    entry = catalog.compose(
        masked,
        stage="compatible-composition",
        parent_recipe_ids=(catalog.baseline.recipe.recipe_id,),
    )
    assert len(entry.recipe.fragments) == 2

    extent_a = ti.DeviceExtent(capacity)
    extent_b = ti.DeviceExtent(capacity)
    output_a = ti.ndarray(ti.i32, shape=1)
    output_b = ti.ndarray(ti.i32, shape=1)
    arguments = {
        "requested_a": 17,
        "requested_b": 29,
        "extent_a": extent_a,
        "extent_b": extent_b,
        "output_a": output_a,
        "output_b": output_b,
    }
    for output in (output_a, output_b):
        output.fill(0)
    with definition.materialization_context() as context:
        with context.materialize(entry.recipe) as materialized:
            materialized.executor.run(arguments)
            ti.sync()
            assert int(output_a.to_numpy()[0]) == 17 * 18 // 2
            assert int(output_b.to_numpy()[0]) == 29 * 30 // 2
            bounded = materialized.executor.execution_stats().memory
            assert bounded.persistent_bounded_control_bytes == 0


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_complete_structured_control_recipe_rebuilds_both_routes_without_environment(
    monkeypatch,
):
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
        carried_state=(state,),
        counter=counter,
        max_iterations=8,
        name="complete_recipe_control",
    )
    definition = builder.freeze()
    catalog = definition.recipe_catalog()
    fragments = _family_fragments(catalog, "structured_control")
    assert len(fragments) == 1, tuple(
        (region.kind, region.path, region.parent_region_id)
        for region in definition.regions
    )

    for name in _RECIPE_ENVIRONMENTS:
        monkeypatch.setenv(name, "intentionally-invalid-after-freeze")
    physical_identities = set()
    routes = set()
    with definition.materialization_context() as context:
        recipes = [catalog.baseline.recipe]
        recipes.extend(
            catalog.compose(
                (fragment.fragment_id,),
                stage="single-region",
                parent_recipe_ids=(catalog.baseline.recipe.recipe_id,),
            ).recipe
            for fragment in fragments
        )
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
            physical_identities.add(materialized.manifest.materialized_physical_id)

    assert routes == {"cuda_conditional_graph", "cuda_masked_bounded_graph"}
    assert len(physical_identities) == 2


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_complete_structured_control_composes_independent_top_level_domains(
    monkeypatch,
):
    capabilities = dict(ti_core.cuda_conditional_graph_capabilities())
    if not capabilities.get("general_graph_exact_control_available", False):
        pytest.skip("general CUDA conditional Graph is unavailable")
    if not capabilities.get("internal_masked_graph_available", False):
        pytest.skip("internal masked CUDA Graph control is unavailable")

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

    monkeypatch.delenv("TI_GRAPH_CUDA_FORCE_MASKED_CONTROL", raising=False)
    monkeypatch.setenv(
        "TAICHI_FORGE_INTERNAL_STRUCTURED_CONTROL_RECIPE",
        "cuda_conditional_graph",
    )
    builder = ti.graph.GraphBuilder()
    symbols = []
    for suffix in ("a", "b"):
        state = scalar(f"state_{suffix}")
        predicate = scalar(f"predicate_{suffix}")
        counter = scalar(f"counter_{suffix}")
        target = ti.graph.Arg(
            ti.graph.ArgKind.SCALAR,
            f"target_{suffix}",
            ti.i32,
        )
        condition_region = builder.create_sequential()
        condition_region.dispatch(condition, state, predicate, target)
        body = builder.create_sequential()
        body.dispatch(step, state, predicate, counter)
        builder.while_loop(
            condition_region,
            body,
            predicate=predicate,
            control_inputs=(state, target),
            carried_state=(state,),
            counter=counter,
            max_iterations=8,
            name=f"complete_recipe_control_{suffix}",
        )
        symbols.append((suffix, state, predicate, counter, target))

    definition = builder.freeze()
    catalog = definition.recipe_catalog()
    fragments = _family_fragments(catalog, "structured_control")
    by_source = {}
    for fragment in fragments:
        selection = _selection(fragment)
        by_source.setdefault(selection.source_key, {})[selection.choice_id] = fragment
    assert len(by_source) == 2
    assert all(len(choices) == 1 for choices in by_source.values())
    masked_fragment_ids = tuple(
        next(iter(choices.values())).fragment_id for choices in by_source.values()
    )
    masked_entry = catalog.compose(
        masked_fragment_ids,
        stage="compatible-composition",
        parent_recipe_ids=(catalog.baseline.recipe.recipe_id,),
    )

    physical_ids = set()
    with definition.materialization_context() as context:
        for recipe, expected_route in (
            (catalog.baseline.recipe, "cuda_conditional_graph"),
            (masked_entry.recipe, "cuda_masked_bounded_graph"),
        ):
            arguments = {}
            for index, (suffix, *_symbols) in enumerate(symbols):
                state = ti.ndarray(ti.i32, shape=())
                predicate = ti.ndarray(ti.i32, shape=())
                counter = ti.ndarray(ti.i32, shape=())
                state.fill(0)
                predicate.fill(0)
                counter.fill(0)
                arguments.update(
                    {
                        f"state_{suffix}": state,
                        f"predicate_{suffix}": predicate,
                        f"counter_{suffix}": counter,
                        f"target_{suffix}": 3 + index * 2,
                    }
                )
            with context.materialize(recipe) as materialized:
                materialized.executor.run(arguments)
                reports = materialized.executor.control_flow_stats()
                assert len(reports) == 2
                assert {report.lowering for report in reports} == {expected_route}
                assert [report.logical_iterations for report in reports] == [3, 5]
                assert arguments["state_a"].to_numpy()[()] == 3
                assert arguments["state_b"].to_numpy()[()] == 5
                physical_ids.add(materialized.materialized_physical_id)
    assert len(physical_ids) == 2


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_complete_recipe_materializes_coarse_cuda_branch_join_dag():
    count = 257

    @ti.kernel
    def advance_a(
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        target: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for index in range(count):
            target[index] = source[index] + 1

    @ti.kernel
    def advance_b(
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        target: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for index in range(count):
            target[index] = source[index] - 2

    @ti.kernel
    def join(
        lhs: ti.types.ndarray(dtype=ti.i32, ndim=1),
        rhs: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for index in range(count):
            output[index] = lhs[index] + rhs[index]

    symbols = {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.i32, ndim=1)
        for name in ("a0", "a1", "b0", "b1", "output")
    }
    builder = ti.graph.GraphBuilder()
    for stage in range(4):
        builder.dispatch(
            advance_a,
            symbols["a0" if stage % 2 == 0 else "a1"],
            symbols["a1" if stage % 2 == 0 else "a0"],
        )
    for stage in range(4):
        builder.dispatch(
            advance_b,
            symbols["b0" if stage % 2 == 0 else "b1"],
            symbols["b1" if stage % 2 == 0 else "b0"],
        )
    builder.dispatch(join, symbols["a0"], symbols["b0"], symbols["output"])
    definition = builder.freeze()

    sources = definition._runtime_spec._graph_branch_join_sources
    assert len(sources) == 1
    assert sources[0].branch_groups == ((0, 1, 2, 3), (4, 5, 6, 7))
    assert sources[0].join_index == 8
    assert sources[0].parallel_temporary_bytes == (
        sources[0].sequential_temporary_bytes
    )

    catalog = definition.recipe_catalog()
    fragments = _family_fragments(catalog, "branch_join_schedule")
    assert len(fragments) == 1
    entry = catalog.compose(
        (fragments[0].fragment_id,),
        stage="coarse-cuda-branch-join",
        parent_recipe_ids=(catalog.baseline.recipe.recipe_id,),
    )

    with definition.materialization_context() as context:
        with context.materialize(catalog.baseline.recipe) as baseline:
            baseline_physical_id = baseline.materialized_physical_id
        with context.materialize(entry.recipe) as product:
            assert product.materialized_physical_id != baseline_physical_id
            tasks = product.manifest.tasks
            assert tuple(task.queue for task in tasks) == (
                *("cuda_branch:0",) * 4,
                *("cuda_branch:1",) * 4,
                "default",
            )
            assert tuple(task.depends_on for task in tasks) == (
                (),
                (0,),
                (1,),
                (2,),
                (),
                (4,),
                (5,),
                (6,),
                (3, 7),
            )

            values = {
                name: ti.ndarray(ti.i32, shape=count) for name in symbols
            }
            initial = np.arange(count, dtype=np.int32)
            values["a0"].from_numpy(initial)
            values["a1"].fill(0)
            values["b0"].from_numpy(initial * 3)
            values["b1"].fill(0)
            values["output"].fill(0)
            bindings = product.executor.bind(values)
            assert bindings.fast_path_qualified
            published = product.executor.binding_statistics()
            for _ in range(32):
                product.executor.run(bindings)
            ti.sync()
            replayed = product.executor.binding_statistics()
            assert replayed["version_builds"] == published["version_builds"]
            assert replayed["raw_replay_validations"] == (
                published["raw_replay_validations"]
            )
            assert replayed["version_fast_replays"] == (
                published["version_fast_replays"] + 32
            )
            assert product.executor.execution_stats().execution_path == (
                "cuda_exact_replay"
            )
            np.testing.assert_array_equal(
                values["output"].to_numpy(),
                initial * 4 - 128,
            )

            aliased = dict(values)
            aliased["b0"] = aliased["a0"]
            with pytest.raises(
                TaichiRuntimeError,
                match="requires proven disjoint storage",
            ):
                product.executor.bind(aliased)

    short = ti.graph.GraphBuilder()
    short.dispatch(advance_a, symbols["a0"], symbols["a1"])
    short.dispatch(advance_b, symbols["b0"], symbols["b1"])
    short.dispatch(join, symbols["a1"], symbols["b1"], symbols["output"])
    assert not short.freeze()._runtime_spec._graph_branch_join_sources
