from itertools import pairwise

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.graph import compileiq_recipe_search

from tests import test_utils


def _layout(capacity, segment_length):
    offsets = np.arange(0, capacity + 1, segment_length, dtype=np.int32)
    if offsets[-1] != capacity:
        offsets = np.append(offsets, np.int32(capacity))
    return (
        ti.algorithms.SegmentedLayout.from_offsets(
            offsets,
            capacity=capacity,
        ),
        offsets,
    )


def _build(values, layout, output, *, inclusive=True, workspace_lanes=1):
    builder = ti.graph.GraphBuilder()
    builder.segmented_scan(
        values,
        layout,
        output,
        inclusive=inclusive,
    )
    return builder.compile(workspace_lanes=workspace_lanes)


def _parameters(search, recipe_id):
    return {
        "domain_fingerprint": search.domain_fingerprint,
        "recipe_id": recipe_id,
    }


@test_utils.test(arch=ti.cuda, offline_cache=False)
@pytest.mark.parametrize(
    "taichi_dtype,numpy_dtype,inclusive",
    ((ti.i32, np.int32, False), (ti.u32, np.uint32, True)),
)
def test_graph_native_segmented_scan_materializes_complete_domain(
    monkeypatch,
    taichi_dtype,
    numpy_dtype,
    inclusive,
):
    capacity = 8192
    layout, offsets = _layout(capacity, 2048)
    values = ti.ndarray(taichi_dtype, shape=capacity)
    output = ti.ndarray(taichi_dtype, shape=capacity)
    graph = _build(values, layout, output, inclusive=inclusive)

    search = compileiq_recipe_search(graph)
    manifest = search.manifest()
    assert manifest["recipe_kind"] == "graph_native_algorithm"
    assert (
        search.search_space.provider_namespace == "taichi_forge.graph.native_algorithm"
    )
    assert (
        search.search_space.domain_version
        == "graph-native-algorithm-complete-recipe.v2"
    )
    assert len(search.recipe_ids) == 6

    recipes = {
        recipe_id: search.recipe_manifest(recipe_id) for recipe_id in search.recipe_ids
    }
    strategies = {
        recipe["native_algorithm_recipe_manifest"]["strategy"]
        for recipe in recipes.values()
    }
    assert strategies == {
        "segment_local_serial",
        "global_scan_segment_correction",
        "warp_chunked_carry",
        "block_chunked_carry",
        "warp_shuffle_carry",
        "block_hierarchical_carry",
    }
    assert (
        recipes[search.baseline_recipe_id]["native_algorithm_recipe_manifest"][
            "strategy"
        ]
        == "segment_local_serial"
    )
    semantics = tuple(
        recipe["native_algorithm_recipe_manifest"]["semantics"]
        for recipe in recipes.values()
    )
    assert semantics[0] == semantics[1]
    assert semantics[0]["dtype"] == ("i32" if taichi_dtype == ti.i32 else "u32")
    assert semantics[0]["inclusive"] is inclusive
    assert semantics[0]["capacity"] == capacity
    assert semantics[0]["num_segments"] == len(offsets) - 1

    host = ((np.arange(capacity, dtype=np.uint64) % 7) + 1).astype(numpy_dtype)
    expected = np.empty_like(host)
    for begin, end in pairwise(offsets):
        segment = host[int(begin) : int(end)]
        accumulated = np.cumsum(segment, dtype=numpy_dtype)
        if inclusive:
            expected[int(begin) : int(end)] = accumulated
        else:
            expected[int(begin) : int(end)] = np.concatenate(
                (np.zeros(1, dtype=numpy_dtype), accumulated[:-1])
            )

    semantic_graph_ids = set()
    with graph.definition.materialization_context() as context:
        for recipe_id in search.recipe_ids:
            parameters = _parameters(search, recipe_id)
            with search.materialize(parameters, context=context) as materialized:
                rebuilt = materialized.executor
                semantic_graph_ids.add(rebuilt.definition.semantic_graph_id)
                search.verify_materialized_graph(parameters, materialized)
                values.from_numpy(host)
                output.fill(0)
                rebuilt.run({})
                executable = rebuilt._spec.nodes[0].executable
                strategy = recipes[recipe_id]["native_algorithm_recipe_manifest"][
                    "strategy"
                ]
                if strategy == "global_scan_segment_correction":
                    from taichi_forge.graph._retained_scan import RetainedScanRecording

                    def reject_host_provider(*_args, **_kwargs):
                        raise AssertionError("retained scan replay entered a host provider")

                    with monkeypatch.context() as stable_replay:
                        stable_replay.setattr(RetainedScanRecording, "execute", reject_host_provider)
                        rebuilt.run({})
                elif strategy == "segment_local_serial":
                    kernel_plan_type = type(executable._serial_call._plan)

                    def reject_repeated_kernel_resource_proof(*_args, **_kwargs):
                        raise AssertionError(
                            "stable Graph replay repeated a fixed-kernel resource proof"
                        )

                    with monkeypatch.context() as stable_replay:
                        stable_replay.setattr(
                            kernel_plan_type,
                            "matches",
                            reject_repeated_kernel_resource_proof,
                        )
                        rebuilt.run({})
                else:
                    assert executable.debug_info["nested_graph_replay"]
                    rebuilt.run({})
                ti.sync()
                np.testing.assert_array_equal(output.to_numpy(), expected)
                assert executable.debug_info["provider_preparations"] == 0
                assert executable.debug_info["kernel_plan_preparations"] == (
                    {
                        "segment_local_serial": 1,
                        "global_scan_segment_correction": 0,
                        "warp_chunked_carry": 0,
                        "block_chunked_carry": 0,
                        "warp_shuffle_carry": 0,
                        "block_hierarchical_carry": 0,
                    }[strategy]
                )
                assert executable.debug_info["nested_graph_replay"] == (
                    strategy != "segment_local_serial"
                )
                execution_report = rebuilt.execution_stats()
                assert execution_report.memory.provider_generation_report_count == 1
                assert execution_report.memory.provider_generation_known_resident_requested_bytes == (
                    (len(offsets) - 1) * 4 + int(executable._scan_scratch.shape[0]) * 4
                    if strategy == "global_scan_segment_correction"
                    else 0
                )
                assert (
                    execution_report.memory.provider_generation_requested_bytes_complete
                    is True
                )
    assert semantic_graph_ids == {graph.definition.semantic_graph_id}


@test_utils.test(arch=ti.cuda, offline_cache=False)
@pytest.mark.parametrize("dtype,inclusive", ((ti.i32, False), (ti.u32, True)))
def test_retained_global_scan_survives_arena_growth_clear_and_preserves_padding(dtype, inclusive):
    from taichi_forge.graph._retained_scan import scan_workspace_bytes

    # More than two hierarchy levels, empty segments, modular overflow, and
    # unused capacity exercise the real scan/correction/lifetime contract.
    lengths = np.array([0, 1, 31, 0, 1025, (1 << 20) + 19, 0], np.int32)
    offsets = np.concatenate((np.zeros(1, np.int32), np.cumsum(lengths, dtype=np.int32)))
    n, capacity = int(offsets[-1]), int(offsets[-1]) + 17
    host = np.random.default_rng(91).integers(0, 2**32, capacity, dtype=np.uint32)
    if dtype == ti.i32:
        host = host.view(np.int32)
    expected = np.full_like(host, 73)
    for begin, end in pairwise(offsets):
        if begin != end:
            summed = np.cumsum(host[begin:end], dtype=host.dtype)
            expected[begin:end] = summed if inclusive else np.concatenate((np.zeros(1, host.dtype), summed[:-1]))
    values, output = (ti.ndarray(dtype, capacity) for _ in range(2))
    values.from_numpy(host)
    output.fill(73)
    layout = ti.algorithms.SegmentedLayout.from_offsets(offsets, capacity=capacity)
    builder = ti.graph.GraphBuilder()
    builder.segmented_scan(values, layout, output, inclusive=inclusive)
    definition = builder.freeze()
    catalog = definition.recipe_catalog()
    # This large topology chooses the global strategy as its complete baseline;
    # baseline fragments are deliberately omitted by the composer.
    recipe = next(entry.recipe for entry in catalog.entries() if not entry.recipe.fragments)
    prog = ti.lang.impl.get_runtime().prog
    with definition.materialization_context() as context:
        with context.materialize(recipe) as materialized:
            graph = materialized.executor
            bound = graph.bind({})
            # Freeze/materialization/bind must not advance user computation.
            np.testing.assert_array_equal(output.to_numpy(), np.full_like(host, 73))
            for _ in range(5):
                graph.run(bound)
            # Ordinary scans grow and then discard the Program-shared arena.
            unrelated = ti.ndarray(ti.i32, 2 * capacity)
            unrelated.fill(1)
            prog.cuda_device_inclusive_scan_ndarray(unrelated.arr, 0)
            prog._primitive_workspace_clear()
            before = prog._primitive_workspace_stats()
            for _ in range(31):
                graph.run(bound)
            np.testing.assert_array_equal(output.to_numpy(), expected)
            np.testing.assert_array_equal(values.to_numpy(), host)
            after = prog._primitive_workspace_stats()
            assert after["acquisitions"] == before["acquisitions"]
            assert after["reserved_bytes"] == before["reserved_bytes"] == 0
            executable = graph._spec.nodes[0].executable
            assert executable._nested_graph._graph_stats[0]["last_path"] == "cuda_exact_replay"
            assert graph.execution_stats().memory.provider_generation_known_resident_requested_bytes == (
                len(lengths) * 4 + scan_workspace_bytes(n, dtype)
            )


@test_utils.test(arch=ti.cuda, offline_cache=False)
@pytest.mark.parametrize("inclusive", (False, True))
def test_graph_native_segmented_scan_global_correction_handles_irregular_long_segments(
    inclusive,
):
    lengths = np.asarray((1, 127, 128, 129, 4097), dtype=np.int32)
    offsets = np.concatenate(
        (np.zeros(1, dtype=np.int32), np.cumsum(lengths, dtype=np.int32))
    )
    capacity = int(offsets[-1])
    layout = ti.algorithms.SegmentedLayout.from_offsets(offsets, capacity=capacity)
    values = ti.ndarray(ti.i32, shape=capacity)
    output = ti.ndarray(ti.i32, shape=capacity)
    graph = _build(values, layout, output, inclusive=inclusive)
    search = compileiq_recipe_search(graph)
    recipes = {
        recipe_id: search.recipe_manifest(recipe_id) for recipe_id in search.recipe_ids
    }
    global_id = next(
        recipe_id
        for recipe_id, recipe in recipes.items()
        if recipe["native_algorithm_recipe_manifest"]["strategy"]
        == "global_scan_segment_correction"
    )
    manifest = recipes[global_id]["native_algorithm_recipe_manifest"]
    assert manifest["topology"] == {
        "kind": "global_scan_segment_correction",
        "correction_block_dim": 128,
        "correction_graph_nodes": 2,
        "submission": "retained_cuda_graph",
        "scan_scratch": "graph_bound_ndarray",
        "scan_extent": capacity,
    }

    host = ((np.arange(capacity, dtype=np.int64) * 104729) - (1 << 30)).astype(np.int32)
    expected = np.empty_like(host)
    for begin, end in pairwise(offsets):
        accumulated = np.cumsum(host[begin:end], dtype=np.int32)
        if inclusive:
            expected[begin:end] = accumulated
        else:
            expected[begin:end] = np.concatenate(
                (np.zeros(1, dtype=np.int32), accumulated[:-1])
            )
    values.from_numpy(host)
    with search.materialize(_parameters(search, global_id)) as materialized:
        candidate = materialized.executor
        for _ in range(3):
            output.fill(0x5A5A5A5A)
            candidate.run({})
        ti.sync()
        np.testing.assert_array_equal(output.to_numpy(), expected)
        executable = candidate._spec.nodes[0].executable
        assert executable.debug_info["provider_preparations"] == 0
        assert executable.debug_info["kernel_plan_preparations"] == 0
        assert executable.debug_info["nested_graph_replay"] is True
        assert executable.backend_command_plan.provider_replay is False


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_graph_native_segmented_scan_generates_and_replays_length_bucket_hybrid():
    lengths = np.asarray((1, 7, 32, 33, 129, 511), dtype=np.int32)
    offsets = np.concatenate(
        (np.zeros(1, dtype=np.int32), np.cumsum(lengths, dtype=np.int32))
    )
    capacity = int(offsets[-1])
    layout = ti.algorithms.SegmentedLayout.from_offsets(offsets, capacity=capacity)
    values = ti.ndarray(ti.i32, shape=capacity)
    output = ti.ndarray(ti.i32, shape=capacity)
    graph = _build(values, layout, output, inclusive=False)
    search = compileiq_recipe_search(graph)
    recipes = {
        recipe_id: search.recipe_manifest(recipe_id) for recipe_id in search.recipe_ids
    }
    hybrid_id = next(
        recipe_id
        for recipe_id, recipe in recipes.items()
        if recipe["native_algorithm_recipe_manifest"]["strategy"]
        == "length_bucket_hybrid"
    )
    manifest = recipes[hybrid_id]["native_algorithm_recipe_manifest"]
    assert manifest["topology"] == {
        "kind": "length_bucket_hybrid",
        "short_max_items": 32,
        "short_segment_count": 3,
        "long_segment_count": 3,
        "short_block_dim": 32,
        "long_block_dim": 128,
    }
    assert manifest["workspace"]["action_owned_bytes"] == len(lengths) * 4

    host = ((np.arange(capacity, dtype=np.int64) % 11) - 5).astype(np.int32)
    expected = np.empty_like(host)
    for begin, end in pairwise(offsets):
        inclusive = np.cumsum(host[begin:end], dtype=np.int32)
        expected[begin:end] = np.concatenate(
            (np.zeros(1, dtype=np.int32), inclusive[:-1])
        )
    values.from_numpy(host)
    with search.materialize(_parameters(search, hybrid_id)) as materialized:
        candidate = materialized.executor
        for _ in range(3):
            output.fill(123)
            candidate.run({})
        ti.sync()
        np.testing.assert_array_equal(output.to_numpy(), expected)
        executable = candidate._spec.nodes[0].executable
        assert executable.debug_info["nested_graph_replay"]
        assert executable.backend_command_plan.command_count == 2
        assert (
            candidate.execution_stats().memory.provider_generation_known_resident_requested_bytes
            == len(lengths) * 4
        )


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_graph_native_segmented_scan_large_auto_is_global_baseline():
    capacity = 65536
    layout, _ = _layout(capacity, 32768)
    values = ti.ndarray(ti.u32, shape=capacity)
    output = ti.ndarray(ti.u32, shape=capacity)
    graph = _build(values, layout, output)
    search = compileiq_recipe_search(graph)
    baseline = search.recipe_manifest(search.baseline_recipe_id)
    assert baseline["native_algorithm_recipe_manifest"]["strategy"] == (
        "global_scan_segment_correction"
    )


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_graph_native_segmented_scan_fixed_resources_require_one_lane():
    layout, _ = _layout(64, 16)
    values = ti.ndarray(ti.i32, shape=64)
    output = ti.ndarray(ti.i32, shape=64)
    with pytest.raises(
        RuntimeError,
        match="exclusive provider-owned fixed storage require workspace_lanes=1",
    ):
        _build(values, layout, output, workspace_lanes=2)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_graph_native_segmented_scan_recipe_composes_with_ordinary_dispatch():
    capacity = 64
    layout, offsets = _layout(capacity, 16)
    values = ti.ndarray(ti.i32, shape=capacity)
    output = ti.ndarray(ti.i32, shape=capacity)

    @ti.kernel
    def clear(array: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in array:
            array[i] = 0

    scratch_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY,
        "scratch",
        ti.i32,
        ndim=1,
    )
    builder = ti.graph.GraphBuilder()
    builder.segmented_scan(values, layout, output)
    builder.dispatch(clear, scratch_arg)
    graph = builder.compile()
    search = compileiq_recipe_search(graph)

    assert search.manifest()["families"] == ("native_algorithm",)
    assert len(search.recipe_ids) == 6
    assert graph.definition._runtime_spec._graph_memory_sources == ()

    host = (np.arange(capacity, dtype=np.int32) % 5) + 1
    expected = np.empty_like(host)
    for begin, end in pairwise(offsets):
        expected[int(begin) : int(end)] = np.cumsum(
            host[int(begin) : int(end)], dtype=np.int32
        )
    scratch = ti.ndarray(ti.i32, shape=capacity)
    physical_ids = set()
    with graph.definition.materialization_context() as context:
        for recipe_id in search.recipe_ids:
            parameters = _parameters(search, recipe_id)
            with search.materialize(parameters, context=context) as materialized:
                search.verify_materialized_graph(parameters, materialized)
                candidate = materialized.executor
                values.from_numpy(host)
                output.fill(0)
                scratch.fill(9)
                candidate.run(candidate.bind({"scratch": scratch}))
                ti.sync()
                np.testing.assert_array_equal(output.to_numpy(), expected)
                np.testing.assert_array_equal(
                    scratch.to_numpy(), np.zeros(capacity, dtype=np.int32)
                )
                physical_ids.add(materialized.materialized_physical_id)
    assert len(physical_ids) == 6


@test_utils.test(arch=ti.cuda, offline_cache=False)
@pytest.mark.parametrize("dtype", (ti.i32, ti.u32))
@pytest.mark.parametrize("inclusive", (False, True))
def test_segmented_shuffle_irregular_overflow_queued_replay(dtype, inclusive):
    # Empty segments, partial warps/blocks and many repeated carries coexist.
    lengths = np.array([0, 1, 31, 32, 33, 0, 127, 128, 129, 4097, 65537, 0], np.int32)
    offsets = np.concatenate((np.zeros(1, np.int32), np.cumsum(lengths, dtype=np.int32)))
    n = int(offsets[-1])
    layout = ti.algorithms.SegmentedLayout.from_offsets(offsets, capacity=n + 11)
    host = np.random.default_rng(733).integers(0, 2**32, n + 11, dtype=np.uint32)
    if dtype == ti.i32:
        host = host.view(np.int32)
    values, output = (ti.ndarray(dtype, n + 11) for _ in range(2))
    values.from_numpy(host)
    expected = np.full_like(host, 77)
    for begin, end in pairwise(offsets):
        if end == begin:
            continue
        summed = np.cumsum(host[begin:end], dtype=host.dtype)
        expected[begin:end] = summed if inclusive else np.concatenate((np.zeros(1, host.dtype), summed[:-1]))
    builder = ti.graph.GraphBuilder()
    builder.segmented_scan(values, layout, output, inclusive=inclusive)
    definition = builder.freeze()
    catalog = definition.recipe_catalog()
    stages = {"warp_register_segment_chunks", "hierarchical_warp_segment_chunks"}
    recipes = [
        entry.recipe
        for entry in catalog.entries()
        if any(task.kind in stages for fragment in entry.recipe.fragments for task in fragment.tasks)
    ]
    assert len(recipes) == 2
    physical_ids = set()
    with definition.materialization_context() as context:
        for recipe in recipes:
            with context.materialize(recipe) as materialized:
                graph = materialized.executor
                output.fill(77)
                bound = graph.bind({})
                for _ in range(73):
                    graph.run(bound)
                np.testing.assert_array_equal(output.to_numpy(), expected)
                np.testing.assert_array_equal(values.to_numpy(), host)
                assert graph.execution_stats().memory.provider_generation_known_resident_requested_bytes == 0
                physical_ids.add(materialized.materialized_physical_id)
    assert len(physical_ids) == 2


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_graph_keyed_aggregation_generates_complete_fixed_resource_domain():
    count = 4096
    num_groups = 64
    keys = ti.ndarray(ti.i32, shape=count)
    values = ti.ndarray(ti.i32, shape=count)
    output = ti.ndarray(ti.i32, shape=num_groups)
    host_keys = (np.arange(count, dtype=np.int32) * 37 + 11) % num_groups
    host_keys[::127] = -1
    host_keys[::191] = num_groups + 3
    host_values = ((np.arange(count, dtype=np.int32) % 17) - 8).astype(np.int32)
    expected = np.zeros(num_groups, dtype=np.int32)
    valid = (host_keys >= 0) & (host_keys < num_groups)
    np.add.at(expected, host_keys[valid], host_values[valid])
    keys.from_numpy(host_keys)
    values.from_numpy(host_values)

    builder = ti.graph.GraphBuilder()
    builder.keyed_reduce(keys, values, output)
    graph = builder.compile()
    search = compileiq_recipe_search(graph)
    assert len(search.recipe_ids) == 2
    recipes = {
        recipe_id: search.recipe_manifest(recipe_id) for recipe_id in search.recipe_ids
    }
    strategies = {
        recipe["native_algorithm_recipe_manifest"]["strategy"]
        for recipe in recipes.values()
    }
    assert strategies == {
        "global_atomic",
        "block_shared_dense",
    }
    physical_ids = set()

    with graph.definition.materialization_context() as context:
        for recipe_id, recipe in recipes.items():
            manifest = recipe["native_algorithm_recipe_manifest"]
            semantics = manifest["semantics"]
            assert semantics["operation"] == "sum"
            assert semantics["invalid_key_policy"] == "ignore"
            assert semantics["associativity"] == "modular_integer_sum"
            assert semantics["determinism"] == "exact"
            assert semantics["count"] == count
            assert semantics["num_groups"] == num_groups
            assert semantics["keys"]["fixed_resource"]
            assert semantics["values"]["fixed_resource"]
            assert semantics["output"]["fixed_resource"]
            strategy = manifest["strategy"]
            topology = manifest["topology"]
            assert topology["kind"] == strategy
            assert topology["stage_count"] == len(manifest["physical_stages"])
            if strategy == "block_shared_dense":
                assert topology == {
                    "kind": "block_shared_dense",
                    "block_dim": 256,
                    "stage_count": 2,
                    "dense_group_limit": 256,
                    "static_shared_bytes": num_groups * 4,
                }
            owned_bytes = 0
            assert manifest["workspace"]["action_owned_bytes"] == owned_bytes

            parameters = _parameters(search, recipe_id)
            with search.materialize(parameters, context=context) as materialized:
                search.verify_materialized_graph(parameters, materialized)
                candidate = materialized.executor
                for _ in range(2):
                    output.fill(123)
                    candidate.run({})
                    ti.sync()
                    np.testing.assert_array_equal(output.to_numpy(), expected)
                executable = candidate._spec.nodes[0].executable
                assert executable.debug_info["provider_preparations"] == (
                    1 if strategy == "global_atomic" else 0
                )
                assert executable.debug_info["kernel_plan_preparations"] == (
                    {
                        "global_atomic": 0,
                        "block_shared_dense": 0,
                    }[strategy]
                )
                assert executable.debug_info["action_owned_bytes"] == owned_bytes
                assert executable.backend_command_plan.command_count == (
                    {
                        "global_atomic": 1,
                        "block_shared_dense": 1,
                    }[strategy]
                )
                assert executable.backend_command_plan.command_count_exact == (
                    strategy != "global_atomic"
                )
                assert executable.backend_command_plan.provider_replay == (
                    strategy == "global_atomic"
                )
                assert executable.debug_info["nested_graph_replay"] == (
                    strategy == "block_shared_dense"
                )
                memory = candidate.execution_stats().memory
                assert memory.provider_generation_report_count == 1
                assert (
                    memory.provider_generation_known_resident_requested_bytes
                    == owned_bytes
                )
                assert memory.provider_generation_requested_bytes_complete
                physical_ids.add(materialized.materialized_physical_id)
    assert len(physical_ids) == 2


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_segmented_shuffle_public_search_selection_artifact_and_rebuild():
    def build():
        layout, _ = _layout(8192, 256)
        values, output = (ti.ndarray(ti.u32, 8192) for _ in range(2))
        values.fill(1)
        builder = ti.graph.GraphBuilder()
        builder.segmented_scan(values, layout, output)
        return builder.freeze(), output

    definition, output = build()
    catalog = definition.recipe_catalog()
    observed = set()
    expected = np.tile(np.arange(1, 257, dtype=np.uint32), 32)

    def evaluate(graph, handle):
        graph.run({})
        np.testing.assert_array_equal(output.to_numpy(), expected)
        observed.add(handle.recipe_id)
        recipe = catalog.entry(handle.recipe_id).recipe
        # A deterministic contract fixture, deliberately not a performance score.
        return {
            "fixture_hierarchical_selection": float(
                any(
                    task.kind == "hierarchical_warp_segment_chunks"
                    for fragment in recipe.fragments
                    for task in fragment.tasks
                )
            )
        }

    decision = definition.search_recipes(
        target=ti.graph.GraphOptimizationTarget(objectives=(("fixture_hierarchical_selection", "max"),)),
        budget=ti.graph.GraphSearchBudget(evaluation_limit=6, repeat_count=1),
        strategy=ti.graph.GraphRecipeSearchStrategy(mode="exact_if_bounded"),
    ).run(evaluate)
    assert decision.status == "selected" and decision.report.search_complete
    assert len(observed) == 6
    artifact = ti.graph.GraphRecipeSelectionArtifact.from_dict(decision.selection_artifact.to_dict())
    rebuilt, rebuilt_output = build()
    assert rebuilt.semantic_graph_id == definition.semantic_graph_id
    selected = rebuilt.resolve_recipe(artifact)
    with rebuilt.materialize(selected) as materialized:
        for _ in range(9):
            materialized.executor.run({})
        np.testing.assert_array_equal(rebuilt_output.to_numpy(), expected)
        assert materialized.executor._spec.nodes[0].executable.debug_info["strategy"] == "block_hierarchical_carry"


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_segmented_shuffle_does_not_advertise_an_empty_physical_plan():
    layout = ti.algorithms.SegmentedLayout.from_offsets(np.zeros(4, np.int32), capacity=7)
    values, output = (ti.ndarray(ti.u32, 7) for _ in range(2))
    builder = ti.graph.GraphBuilder()
    builder.segmented_scan(values, layout, output)
    catalog = builder.freeze().recipe_catalog()
    assert all(
        task.kind not in ("warp_register_segment_chunks", "hierarchical_warp_segment_chunks")
        for fragment in catalog.fragments
        for task in fragment.tasks
    )


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_graph_keyed_aggregation_wide_domain_keeps_only_exact_atomic_recipe():
    count = 1025
    num_groups = 8192
    keys = ti.ndarray(ti.i32, shape=count)
    values = ti.ndarray(ti.i32, shape=count)
    output = ti.ndarray(ti.i32, shape=num_groups)
    indices = np.arange(count, dtype=np.int32)
    host_keys = (indices * 37 + 11) % num_groups
    host_keys[::127] = -1
    host_keys[::191] = num_groups + 3
    host_values = (np.iinfo(np.int32).max - (indices % 31)).astype(np.int32)
    expected = np.zeros(num_groups, dtype=np.int32)
    valid = (host_keys >= 0) & (host_keys < num_groups)
    np.add.at(expected, host_keys[valid], host_values[valid])
    keys.from_numpy(host_keys)
    values.from_numpy(host_values)

    builder = ti.graph.GraphBuilder()
    builder.keyed_reduce(keys, values, output)
    graph = builder.compile()
    search = compileiq_recipe_search(graph)
    assert len(search.recipe_ids) == 1
    recipe_id = search.recipe_ids[0]
    assert "native_algorithm_recipe_manifest" not in search.recipe_manifest(recipe_id)
    parameters = _parameters(search, recipe_id)
    with search.materialize(parameters) as materialized:
        candidate = materialized.executor
        for _ in range(2):
            output.fill(123)
            candidate.run({})
            ti.sync()
            np.testing.assert_array_equal(output.to_numpy(), expected)
        executable = candidate._spec.nodes[0].executable
        assert executable.debug_info["strategy"] == "global_atomic"
        assert executable.backend_command_plan.command_count == 1
        assert executable.backend_command_plan.provider_replay
        assert not executable.debug_info["nested_graph_replay"]
        assert executable.debug_info["action_owned_bytes"] == 0


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_graph_keyed_aggregation_rejects_unscoped_floating_point_order():
    count = 64
    keys = ti.ndarray(ti.i32, shape=count)
    values = ti.ndarray(ti.f32, shape=count)
    output = ti.ndarray(ti.f32, shape=8)
    builder = ti.graph.GraphBuilder()
    with pytest.raises(
        RuntimeError,
        match="requires plain 1D i32 keys and i32 value/output ndarrays",
    ):
        builder.keyed_reduce(keys, values, output)
