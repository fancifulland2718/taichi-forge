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

    assert graph._compileiq_graph_native_algorithm_status == "complete_recipe_domain"
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
    assert len(search.recipe_ids) == 2

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
            assert dict(search.worker_environment(parameters)) == {}
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
                frozen_kernel_call = (
                    executable._serial_call
                    if strategy == "segment_local_serial"
                    else executable._gather_call
                )
                kernel_plan_type = type(frozen_kernel_call._plan)

                def reject_repeated_kernel_resource_proof(*_args, **_kwargs):
                    raise AssertionError(
                        "stable Graph replay repeated a fixed-kernel resource proof"
                    )

                if strategy == "global_scan_segment_correction":
                    plan_type = type(executable._scan_plan)

                    def reject_repeated_program_proof(*_args, **_kwargs):
                        raise AssertionError(
                            "stable Graph replay repeated a provider Program proof"
                        )

                    with monkeypatch.context() as stable_replay:
                        stable_replay.setattr(
                            plan_type,
                            "matches_program",
                            reject_repeated_program_proof,
                        )
                        stable_replay.setattr(
                            kernel_plan_type,
                            "matches",
                            reject_repeated_kernel_resource_proof,
                        )
                        rebuilt.run({})
                else:
                    with monkeypatch.context() as stable_replay:
                        stable_replay.setattr(
                            kernel_plan_type,
                            "matches",
                            reject_repeated_kernel_resource_proof,
                        )
                        rebuilt.run({})
                ti.sync()
                np.testing.assert_array_equal(output.to_numpy(), expected)
                assert executable.debug_info["provider_preparations"] == (
                    1 if strategy == "global_scan_segment_correction" else 0
                )
                assert executable.debug_info["kernel_plan_preparations"] == (
                    2 if strategy == "global_scan_segment_correction" else 1
                )
                execution_report = rebuilt.execution_stats()
                assert execution_report.memory.provider_generation_report_count == 1
                assert (
                    execution_report.memory.provider_generation_known_resident_requested_bytes
                    == (
                        (len(offsets) - 1) * 4
                        if strategy == "global_scan_segment_correction"
                        else 0
                    )
                )
                assert (
                    execution_report.memory.provider_generation_requested_bytes_complete
                    is True
                )
    assert semantic_graph_ids == {graph.definition.semantic_graph_id}


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
    assert len(search.recipe_ids) == 2
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
    assert len(physical_ids) == 2
