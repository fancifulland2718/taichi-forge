from types import MappingProxyType

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge._lib import core as ti_core
from taichi_forge.lang import impl
from taichi_forge.lang.exception import TaichiCompilationError
from taichi_forge.lang._offload_execution_plan import (
    _OffloadExecutionPlan,
    _bind_offload_execution_plan,
)
from taichi_forge.graph import _graph as graph_impl
from taichi_forge.graph import compileiq_recipe_search
from tests import test_utils


def _search_budget(evaluations):
    from compileiq.forge_support import ForgeOpaqueSearchBudgetV2

    return ForgeOpaqueSearchBudgetV2(
        evaluation_limit=evaluations,
        time_limit_seconds=300.0,
        materialized_memory_limit_bytes=1 << 30,
    )


def _shared_staged_plan(kernel, *probe_args, block_dim=128):
    baseline = _OffloadExecutionPlan.from_task_manifests(
        kernel.task_manifest(*probe_args)
    )
    ranges = tuple(task for task in baseline.tasks if task.task_kind == "range_for")
    assert len(ranges) == 1
    return baseline.replace_task(
        ranges[0].task_index,
        workgroup_size=block_dim,
        memory_strategy="shared_staged_1d",
    )


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_graph_memory_compileiq_materializes_complete_direct_and_staged():
    count = 1027

    @ti.kernel
    def stencil(
        source: ti.types.ndarray(dtype=ti.f32, ndim=1),
        output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in range(1, count - 1):
            output[i] = source[i - 1] + source[i] * 2.0 + source[i + 1]

    source_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "source", ti.f32, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(stencil, source_arg, output_arg)
    source_graph = builder.compile()

    search = compileiq_recipe_search(source_graph)
    assert len(search.recipe_ids) == 4
    assert search.search_space.provider_namespace == "taichi_forge.graph.memory"
    assert search.search_space.domain_version == "graph-memory-complete-recipe.v2"
    assert search.manifest()["recipe_kind"] == "graph_memory"
    manifests = {
        recipe_id: search.recipe_manifest(recipe_id) for recipe_id in search.recipe_ids
    }
    direct_id = next(
        recipe_id
        for recipe_id, manifest in manifests.items()
        if manifest["memory_recipe_manifest"]["strategy"] == "direct"
    )
    staged_id = next(
        recipe_id
        for recipe_id, manifest in manifests.items()
        if manifest["memory_recipe_manifest"]["strategy"] == "shared_staged_1d"
        and manifest["memory_recipe_manifest"]["staged_sources"][0]["arg_index"] == 0
        and manifest["memory_recipe_manifest"]["staged_sources"][0]["tile_elements"]
        == 130
    )
    assert direct_id == search.baseline_recipe_id
    assert not manifests[direct_id]["fusion_recipe_ids"]
    assert not manifests[staged_id]["fusion_recipe_ids"]
    assert not manifests[direct_id].get("control_recipe_id")
    assert not manifests[staged_id].get("control_recipe_id")

    def parameters(recipe_id):
        return {
            "domain_fingerprint": search.domain_fingerprint,
            "recipe_id": recipe_id,
        }

    def materialize(recipe_id):
        opaque_parameters = parameters(recipe_id)
        materialized = search.materialize(opaque_parameters)
        search.verify_materialized_graph(opaque_parameters, materialized)
        return materialized

    observed = []

    def objective(_graph, request):
        observed.append((request.recipe_id, request.recipe_id))
        return float(search.recipe_ids.index(request.recipe_id))

    with search.compileiq_search(
        objective,
        budget=_search_budget(len(search.recipe_ids)),
    ) as exhaustive:
        result = exhaustive.start()
        coverage = search.require_complete_search(exhaustive)
        selected = search.select_best_result(exhaustive, result)
    assert coverage["complete"]
    assert coverage["evaluation_count"] == len(search.recipe_ids)
    assert {item[0] for item in observed} == set(search.recipe_ids)
    assert all(requested == actual for requested, actual in observed)
    assert selected.spec_id == search.recipe_ids[0]

    direct_materialized = materialize(direct_id)
    staged_materialized = materialize(staged_id)
    direct_graph = direct_materialized.executor
    staged_graph = staged_materialized.executor
    assert (
        direct_graph.definition.semantic_graph_id
        == staged_graph.definition.semantic_graph_id
    )
    assert direct_graph.definition.binding_abi == staged_graph.definition.binding_abi
    staged_task = next(
        task for task in staged_graph.task_manifest() if task.task_type == "range_for"
    )
    assert staged_task.requested_memory_strategy == "shared_staged_1d"
    assert staged_task.range_mapping == "shared_tiled_one_to_one"
    assert staged_task.selected_block_size == 128

    source = ti.ndarray(ti.f32, shape=count)
    output = ti.ndarray(ti.f32, shape=count)
    values = np.arange(count, dtype=np.float32) * 0.25
    source.from_numpy(values)
    output.fill(0)
    bindings = staged_graph.bind({"source": source, "output": output})
    assert bindings.fast_path_qualified
    staged_graph.run(bindings)
    ti.sync()
    expected = np.zeros(count, dtype=np.float32)
    expected[1:-1] = values[:-2] + values[1:-1] * 2.0 + values[2:]
    np.testing.assert_allclose(output.to_numpy(), expected, rtol=0, atol=0)

    with pytest.raises(RuntimeError, match="requires proven disjoint storage"):
        staged_graph.bind({"source": source, "output": source})
    short_source = ti.ndarray(ti.f32, shape=count - 1)
    short_output = ti.ndarray(ti.f32, shape=count - 1)
    with pytest.raises(RuntimeError, match="at least 1027 scalar elements"):
        staged_graph.bind({"source": short_source, "output": short_output})
    direct_materialized.close()
    staged_materialized.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_graph_memory_2d_stencil_materializes_true_tiles_and_replays_exactly():
    rows = 67
    columns = 53

    @ti.kernel
    def stencil(
        source: ti.types.ndarray(dtype=ti.f32, ndim=2),
        output: ti.types.ndarray(dtype=ti.f32, ndim=2),
    ):
        for row, column in ti.ndrange((1, rows - 1), (1, columns - 1)):
            output[row, column] = (
                source[row - 1, column]
                + source[row, column - 1]
                + source[row, column] * 2.0
                + source[row, column + 1]
                + source[row + 1, column]
            )

    source_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "source", ti.f32, ndim=2)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=2)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(stencil, source_arg, output_arg)
    source_graph = builder.compile()

    metadata = source_graph._spec.compiled_graph()._dispatch_metadata[0]
    assert metadata["version"] == 4
    assert metadata["blocker"] == ""
    assert metadata["iteration_domain"] == {
        "kind": "constant_range",
        "begin": 0,
        "end": 65 * 51,
        "arg_id": [],
        "axis": -1,
        "logical_shape": [65, 51],
        "logical_origin": [1, 1],
    }
    source_effect = next(
        effect for effect in metadata["effects"] if effect["arg_id"] == [0]
    )
    assert source_effect["footprint"]["iteration_rank"] == 2
    assert source_effect["footprint"]["affine_index_offsets"] == [
        [-1, 0],
        [0, -1],
        [0, 0],
        [0, 1],
        [1, 0],
    ]
    assert source_effect["footprint"]["halo"] == [[-1, 1], [-1, 1]]

    search = compileiq_recipe_search(source_graph)
    manifests = {
        recipe_id: search.recipe_manifest(recipe_id) for recipe_id in search.recipe_ids
    }
    assert len(manifests) == 4
    staged_manifests = {
        recipe_id: manifest["memory_recipe_manifest"]
        for recipe_id, manifest in manifests.items()
        if manifest["memory_recipe_manifest"]["strategy"] == "shared_staged_2d"
    }
    assert {
        tuple(manifest["offload_plan"]["tasks"][0]["memory_tile_shape"])
        for manifest in staged_manifests.values()
    } == {(8, 8), (8, 16), (16, 16)}
    staged_id, staged = next(
        (recipe_id, manifest)
        for recipe_id, manifest in staged_manifests.items()
        if manifest["offload_plan"]["tasks"][0]["memory_tile_shape"] == [8, 16]
    )
    assert staged["schema_version"] == 5
    assert staged["memory_disjoint_pairs"] == [["source", "output"]]
    assert {
        (item[0], tuple(item[1]), item[2], item[3], tuple(item[4]), item[5])
        for item in staged["memory_layout_requirements"]
    } == {
        ("source", (rows, columns), 4, 4, (), "row_major_2d"),
        ("output", (rows - 1, columns - 1), 4, 4, (), "row_major_2d"),
    }
    staged_source = staged["staged_sources"][0]
    assert staged_source["iteration_shape"] == [65, 51]
    assert staged_source["iteration_origin"] == [1, 1]
    assert staged_source["tile_shape"] == [8, 16]
    assert staged_source["tile_extents"] == [10, 18]
    assert staged_source["tile_elements"] == 180
    assert staged_source["tile_bytes"] == 720
    assert staged_source["logical_output_count"] == 3315
    assert staged_source["direct_input_records"] == 16575
    assert staged_source["staged_input_records"] == 4897

    parameters = {
        "domain_fingerprint": search.domain_fingerprint,
        "recipe_id": staged_id,
    }
    with search.materialize(parameters) as materialized:
        search.verify_materialized_graph(parameters, materialized)
        graph = materialized.executor
        task = next(
            item for item in graph.task_manifest() if item.task_type == "range_for"
        )
        assert task.requested_memory_strategy == "shared_staged_2d"
        assert task.range_mapping == "shared_tiled_2d_one_to_one"
        assert task.selected_block_size == 128
        assert task.staged_iteration_shape == (65, 51)
        assert task.staged_iteration_origin == (1, 1)
        assert task.staged_tile_shape == (8, 16)
        assert task.static_shared_bytes == 720

        source = ti.ndarray(ti.f32, shape=(rows, columns))
        output = ti.ndarray(ti.f32, shape=(rows, columns))
        values = (
            np.arange(rows * columns, dtype=np.int64).reshape(rows, columns) % 41
        ).astype(np.float32)
        source.from_numpy(values)
        output.fill(0)
        bindings = graph.bind({"source": source, "output": output})
        assert bindings.fast_path_qualified
        for _ in range(5):
            graph.run(bindings)
        ti.sync()
        expected = np.zeros_like(values)
        expected[1:-1, 1:-1] = (
            values[:-2, 1:-1]
            + values[1:-1, :-2]
            + values[1:-1, 1:-1] * 2.0
            + values[1:-1, 2:]
            + values[2:, 1:-1]
        )
        np.testing.assert_array_equal(output.to_numpy(), expected)
        assert graph.binding_statistics()["version_fast_replays"] >= 5

        with pytest.raises(RuntimeError, match="requires proven disjoint storage"):
            graph.bind({"source": source, "output": source})


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_graph_memory_2d_two_source_domain_keeps_all_staging_subsets_distinct():
    rows = 35
    columns = 29

    @ti.kernel
    def two_source_stencil(
        vertical: ti.types.ndarray(dtype=ti.f32, ndim=2),
        horizontal: ti.types.ndarray(dtype=ti.f32, ndim=2),
        output: ti.types.ndarray(dtype=ti.f32, ndim=2),
    ):
        for row, column in ti.ndrange((1, rows - 1), (1, columns - 1)):
            output[row, column] = (
                vertical[row - 1, column]
                + vertical[row, column]
                + vertical[row + 1, column]
                + horizontal[row, column - 1]
                + horizontal[row, column + 1]
            )

    vertical_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "vertical", ti.f32, ndim=2)
    horizontal_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "horizontal", ti.f32, ndim=2
    )
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=2)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(two_source_stencil, vertical_arg, horizontal_arg, output_arg)
    source_graph = builder.compile()
    search = compileiq_recipe_search(source_graph)
    manifests = {
        recipe_id: search.recipe_manifest(recipe_id) for recipe_id in search.recipe_ids
    }

    assert len(manifests) == 10
    staged_variants = {
        (
            tuple(
                source["arg_index"]
                for source in manifest["memory_recipe_manifest"]["staged_sources"]
            ),
            tuple(
                manifest["memory_recipe_manifest"]["offload_plan"]["tasks"][0][
                    "memory_tile_shape"
                ]
            ),
        )
        for manifest in manifests.values()
        if manifest["memory_recipe_manifest"]["strategy"] == "shared_staged_2d"
    }
    assert staged_variants == {
        (source_indices, tile_shape)
        for source_indices in ((0,), (1,), (0, 1))
        for tile_shape in ((8, 8), (8, 16), (16, 16))
    }
    staged_id, staged = next(
        (recipe_id, manifest["memory_recipe_manifest"])
        for recipe_id, manifest in manifests.items()
        if manifest["memory_recipe_manifest"]["strategy"] == "shared_staged_2d"
        and tuple(
            source["arg_index"]
            for source in manifest["memory_recipe_manifest"]["staged_sources"]
        )
        == (0, 1)
        and manifest["memory_recipe_manifest"]["offload_plan"]["tasks"][0][
            "memory_tile_shape"
        ]
        == [8, 8]
    )
    assert [source["access_offsets"] for source in staged["staged_sources"]] == [
        [[-1, 0], [0, 0], [1, 0]],
        [[0, -1], [0, 1]],
    ]

    parameters = {
        "domain_fingerprint": search.domain_fingerprint,
        "recipe_id": staged_id,
    }
    with search.materialize(parameters) as materialized:
        graph = materialized.executor
        search.verify_materialized_graph(parameters, materialized)
        task = next(
            item for item in graph.task_manifest() if item.task_type == "range_for"
        )
        assert task.staged_external_arg_indices == (0, 1)
        assert task.staged_halo_lows_nd == ((-1, 0), (0, -1))
        assert task.staged_halo_highs_nd == ((1, 0), (0, 1))
        assert task.static_shared_bytes == 640

        vertical = ti.ndarray(ti.f32, shape=(rows, columns))
        horizontal = ti.ndarray(ti.f32, shape=(rows, columns))
        output = ti.ndarray(ti.f32, shape=(rows, columns))
        vertical_values = np.arange(rows * columns, dtype=np.float32).reshape(
            rows, columns
        )
        horizontal_values = vertical_values * 0.25 + 3.0
        vertical.from_numpy(vertical_values)
        horizontal.from_numpy(horizontal_values)
        output.fill(0)
        bindings = graph.bind(
            {"vertical": vertical, "horizontal": horizontal, "output": output}
        )
        for _ in range(3):
            graph.run(bindings)
        ti.sync()
        expected = np.zeros_like(vertical_values)
        expected[1:-1, 1:-1] = (
            vertical_values[:-2, 1:-1]
            + vertical_values[1:-1, 1:-1]
            + vertical_values[2:, 1:-1]
            + horizontal_values[1:-1, :-2]
            + horizontal_values[1:-1, 2:]
        )
        np.testing.assert_array_equal(output.to_numpy(), expected)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_graph_memory_2d_high_reuse_recipe_has_exact_physical_accounting():
    rows = 37
    columns = 31
    radius = 2

    @ti.kernel
    def stencil(
        source: ti.types.ndarray(dtype=ti.f32, ndim=2),
        output: ti.types.ndarray(dtype=ti.f32, ndim=2),
    ):
        for row, column in ti.ndrange(
            (radius, rows - radius), (radius, columns - radius)
        ):
            value = 0.0
            for delta_row, delta_column in ti.static(
                ti.ndrange((-radius, radius + 1), (-radius, radius + 1))
            ):
                value += source[row + delta_row, column + delta_column]
            output[row, column] = value

    source_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "source", ti.f32, ndim=2)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=2)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(stencil, source_arg, output_arg)
    source_graph = builder.compile()
    search = compileiq_recipe_search(source_graph)
    manifests = {
        recipe_id: search.recipe_manifest(recipe_id) for recipe_id in search.recipe_ids
    }

    assert len(manifests) == 4
    staged_id, staged = next(
        (recipe_id, manifest["memory_recipe_manifest"])
        for recipe_id, manifest in manifests.items()
        if manifest["memory_recipe_manifest"]["strategy"] == "shared_staged_2d"
        and manifest["memory_recipe_manifest"]["offload_plan"]["tasks"][0][
            "memory_tile_shape"
        ]
        == [16, 16]
    )
    staged_source = staged["staged_sources"][0]
    assert staged_source["iteration_shape"] == [33, 27]
    assert staged_source["iteration_origin"] == [2, 2]
    assert staged_source["tile_extents"] == [20, 20]
    assert staged_source["tile_elements"] == 400
    assert staged_source["logical_output_count"] == 891
    assert staged_source["direct_input_records"] == 22275
    assert staged_source["staged_input_records"] == 1575

    parameters = {
        "domain_fingerprint": search.domain_fingerprint,
        "recipe_id": staged_id,
    }
    with search.materialize(parameters) as materialized:
        search.verify_materialized_graph(parameters, materialized)
        graph = materialized.executor
        task = next(
            item for item in graph.task_manifest() if item.task_type == "range_for"
        )
        assert task.actual_geometry_kind == "static_exact_tiled_2d"
        assert task.actual_grid_size == 6
        assert task.actual_block_size == 256
        assert task.static_shared_bytes == 1600

        source = ti.ndarray(ti.f32, shape=(rows, columns))
        output = ti.ndarray(ti.f32, shape=(rows, columns))
        values = (
            np.arange(rows * columns, dtype=np.int64).reshape(rows, columns) % 29
        ).astype(np.float32)
        source.from_numpy(values)
        output.fill(0)
        bindings = graph.bind({"source": source, "output": output})
        for _ in range(3):
            graph.run(bindings)
        ti.sync()
        expected = np.zeros_like(values)
        for delta_row in range(-radius, radius + 1):
            for delta_column in range(-radius, radius + 1):
                expected[radius:-radius, radius:-radius] += values[
                    radius + delta_row : rows - radius + delta_row,
                    radius + delta_column : columns - radius + delta_column,
                ]
        np.testing.assert_array_equal(output.to_numpy(), expected)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_graph_memory_2d_compound_records_keep_packed_lane_layout():
    rows = 21
    columns = 19
    vec2 = ti.types.vector(2, ti.f32)

    @ti.kernel
    def stencil(
        source: ti.types.ndarray(dtype=vec2, ndim=2),
        output: ti.types.ndarray(dtype=vec2, ndim=2),
    ):
        for row, column in ti.ndrange((1, rows - 1), (1, columns - 1)):
            output[row, column] = (
                source[row - 1, column]
                + source[row, column - 1]
                + source[row, column]
                + source[row, column + 1]
                + source[row + 1, column]
            )

    source_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "source", vec2, ndim=2)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", vec2, ndim=2)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(stencil, source_arg, output_arg)
    source_graph = builder.compile()
    search = compileiq_recipe_search(source_graph)
    staged_id = next(
        recipe_id
        for recipe_id in search.recipe_ids
        if search.recipe_manifest(recipe_id)["memory_recipe_manifest"]["strategy"]
        == "shared_staged_2d"
        and search.recipe_manifest(recipe_id)["memory_recipe_manifest"]["offload_plan"][
            "tasks"
        ][0]["memory_tile_shape"]
        == [8, 8]
    )
    staged_source = search.recipe_manifest(staged_id)["memory_recipe_manifest"][
        "staged_sources"
    ][0]
    assert staged_source["element_shape"] == [2]
    assert staged_source["element_bytes"] == 8
    assert staged_source["tile_extents"] == [10, 10]
    assert staged_source["tile_bytes"] == 800

    parameters = {
        "domain_fingerprint": search.domain_fingerprint,
        "recipe_id": staged_id,
    }
    with search.materialize(parameters) as materialized:
        graph = materialized.executor
        search.verify_materialized_graph(parameters, materialized)
        task = next(
            item for item in graph.task_manifest() if item.task_type == "range_for"
        )
        assert task.staged_element_shapes == ((2,),)
        assert task.staged_element_bytes == (8,)
        assert task.static_shared_bytes == 800

        source = ti.Vector.ndarray(2, ti.f32, shape=(rows, columns))
        output = ti.Vector.ndarray(2, ti.f32, shape=(rows, columns))
        values = np.arange(rows * columns * 2, dtype=np.float32).reshape(
            rows, columns, 2
        )
        source.from_numpy(values)
        output.fill(0)
        bindings = graph.bind({"source": source, "output": output})
        for _ in range(3):
            graph.run(bindings)
        ti.sync()
        expected = np.zeros_like(values)
        expected[1:-1, 1:-1] = (
            values[:-2, 1:-1]
            + values[1:-1, :-2]
            + values[1:-1, 1:-1]
            + values[1:-1, 2:]
            + values[2:, 1:-1]
        )
        np.testing.assert_array_equal(output.to_numpy(), expected)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_graph_memory_2d_rejects_noncanonical_wraparound_coordinates():
    rows = 33
    columns = 31

    @ti.kernel
    def wrapped_stencil(
        source: ti.types.ndarray(dtype=ti.f32, ndim=2),
        output: ti.types.ndarray(dtype=ti.f32, ndim=2),
    ):
        for row, column in ti.ndrange(rows, columns):
            output[row, column] = (
                source[row, column]
                + source[(row + 1) % rows, column]
                + source[row, (column + 1) % columns]
            )

    source_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "source", ti.f32, ndim=2)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=2)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(wrapped_stencil, source_arg, output_arg)
    graph = builder.compile()
    search = compileiq_recipe_search(graph)

    assert len(search.recipe_ids) == 1
    assert graph._spec._graph_memory_sources[0].candidate_failure
    assert "memory_recipe_id" not in search.recipe_manifest(search.recipe_ids[0])


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_graph_memory_static_stencil_accumulator_is_not_a_global_atomic_effect():
    count = 1031
    radius = 4

    @ti.kernel
    def stencil(
        source: ti.types.ndarray(dtype=ti.f32, ndim=1),
        output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in range(radius, count - radius):
            value = 0.0
            for offset in ti.static(range(-radius, radius + 1)):
                value += source[i + offset]
            output[i] = value

    source_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "source", ti.f32, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(stencil, source_arg, output_arg)
    source_graph = builder.compile()

    metadata = source_graph._spec.compiled_graph()._dispatch_metadata[0]
    assert metadata["version"] == 4
    assert metadata["blocker"] == ""
    assert metadata["side_effects"] == []
    source_effect = next(
        effect for effect in metadata["effects"] if effect["arg_id"] == [0]
    )
    assert source_effect["footprint"]["affine_offsets"] == list(
        range(-radius, radius + 1)
    )

    search = compileiq_recipe_search(source_graph)
    assert len(search.recipe_ids) == 4
    staged_id, staged = next(
        (recipe_id, search.recipe_manifest(recipe_id)["memory_recipe_manifest"])
        for recipe_id in search.recipe_ids
        if search.recipe_manifest(recipe_id)["memory_recipe_manifest"]["strategy"]
        == "shared_staged_1d"
        and search.recipe_manifest(recipe_id)["memory_recipe_manifest"]["offload_plan"][
            "tasks"
        ][0]["workgroup_size"]
        == 128
    )
    staged_source = staged["staged_sources"][0]
    assert staged["schema_version"] == 5
    assert staged_source["access_offsets"] == list(range(-radius, radius + 1))
    assert staged_source["logical_output_count"] == 1023
    assert staged_source["direct_input_records"] == 9207
    assert staged_source["staged_input_records"] == 1087
    assert staged_source["direct_input_bytes"] == 36828
    assert staged_source["staged_input_bytes"] == 4348

    source = ti.ndarray(ti.f32, shape=count)
    output = ti.ndarray(ti.f32, shape=count)
    values = (np.arange(count, dtype=np.int64) % 17).astype(np.float32)
    source.from_numpy(values)
    expected = np.zeros(count, dtype=np.float32)
    for offset in range(-radius, radius + 1):
        expected[radius:-radius] += values[radius + offset : count - radius + offset]

    with search.materialize(
        {"domain_fingerprint": search.domain_fingerprint, "recipe_id": staged_id}
    ) as materialized:
        graph = materialized.executor
        binding = graph.bind({"source": source, "output": output})
        for _ in range(3):
            graph.run(binding)
        ti.sync()
        np.testing.assert_array_equal(output.to_numpy(), expected)


@pytest.mark.parametrize(
    ("dtype", "numpy_dtype", "element_bytes"),
    ((ti.f16, np.float16, 2), (ti.f64, np.float64, 8)),
)
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_graph_memory_compileiq_supports_two_and_eight_byte_scalar_stencils(
    dtype,
    numpy_dtype,
    element_bytes,
):
    count = 1027

    @ti.kernel
    def stencil(
        source: ti.types.ndarray(dtype=dtype, ndim=1),
        output: ti.types.ndarray(dtype=dtype, ndim=1),
    ):
        for i in range(1, count - 1):
            output[i] = source[i - 1] + source[i] * 2 + source[i + 1]

    source_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "source", dtype, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", dtype, ndim=1)

    def build():
        builder = ti.graph.GraphBuilder()
        builder.dispatch(stencil, source_arg, output_arg)
        return builder.compile()

    source_graph = build()
    search = compileiq_recipe_search(source_graph)
    manifests = {
        recipe_id: search.recipe_manifest(recipe_id) for recipe_id in search.recipe_ids
    }
    assert len(manifests) == 4
    staged_id, staged_manifest = next(
        (recipe_id, manifest)
        for recipe_id, manifest in manifests.items()
        if manifest["memory_recipe_manifest"]["strategy"] == "shared_staged_1d"
        and manifest["memory_recipe_manifest"]["staged_sources"][0]["tile_elements"]
        == 130
    )
    layout_requirements = {
        tuple(requirement)
        for requirement in staged_manifest["memory_recipe_manifest"][
            "memory_layout_requirements"
        ]
    }
    assert layout_requirements == {
        ("source", count, element_bytes, element_bytes),
        ("output", count - 1, element_bytes, element_bytes),
    }

    parameters = {
        "domain_fingerprint": search.domain_fingerprint,
        "recipe_id": staged_id,
    }
    materialized = search.materialize(parameters)

    staged_graph = materialized.executor

    search.verify_materialized_graph(parameters, materialized)
    staged_task = next(
        task for task in staged_graph.task_manifest() if task.task_type == "range_for"
    )
    assert staged_task.requested_memory_strategy == "shared_staged_1d"
    assert staged_task.range_mapping == "shared_tiled_one_to_one"
    assert staged_task.static_shared_bytes == (128 + 2) * element_bytes

    source = ti.ndarray(dtype, shape=count)
    output = ti.ndarray(dtype, shape=count)
    values = (np.arange(count, dtype=np.int64) % 17).astype(numpy_dtype)
    source.from_numpy(values)
    output.fill(0)
    bindings = staged_graph.bind({"source": source, "output": output})
    assert bindings.fast_path_qualified
    staged_graph.run(bindings)
    ti.sync()
    expected = np.zeros(count, dtype=numpy_dtype)
    expected[1:-1] = values[:-2] + values[1:-1] * 2 + values[2:]
    np.testing.assert_array_equal(output.to_numpy(), expected)

    materialized.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_graph_memory_compileiq_materializes_two_ordered_shared_sources(monkeypatch):
    count = 4099

    @ti.kernel
    def two_source_stencil(
        left: ti.types.ndarray(dtype=ti.f32, ndim=1),
        right: ti.types.ndarray(dtype=ti.f32, ndim=1),
        output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in range(2, count - 2):
            output[i] = (
                left[i - 2] + left[i] + left[i + 1] + right[i - 1] * 0.5 + right[i + 2]
            )

    left_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "left", ti.f32, ndim=1)
    right_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "right", ti.f32, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1)

    def build():
        builder = ti.graph.GraphBuilder()
        builder.dispatch(two_source_stencil, left_arg, right_arg, output_arg)
        return builder.compile()

    source_graph = build()
    search = compileiq_recipe_search(source_graph)
    manifests = {
        recipe_id: search.recipe_manifest(recipe_id) for recipe_id in search.recipe_ids
    }
    assert len(manifests) == 10
    generated_shapes = {
        (
            tuple(source["arg_index"] for source in manifest["staged_sources"]),
            manifest["offload_plan"]["tasks"][0]["workgroup_size"],
        )
        for manifest in (
            value["memory_recipe_manifest"] for value in manifests.values()
        )
        if manifest["strategy"] == "shared_staged_1d"
    }
    assert generated_shapes == {
        (source_indices, block_dim)
        for source_indices in ((0,), (1,), (0, 1))
        for block_dim in (64, 128, 256)
    }
    staged_id, staged = next(
        (recipe_id, manifest["memory_recipe_manifest"])
        for recipe_id, manifest in manifests.items()
        if manifest["memory_recipe_manifest"]["strategy"] == "shared_staged_1d"
        and tuple(
            source["arg_index"]
            for source in manifest["memory_recipe_manifest"]["staged_sources"]
        )
        == (0, 1)
        and manifest["memory_recipe_manifest"]["offload_plan"]["tasks"][0][
            "workgroup_size"
        ]
        == 128
    )
    assert staged["schema_version"] == 5
    assert staged["staged_sources"] == [
        {
            "alignment": 4,
            "arg_index": 0,
            "arg_name": "left",
            "byte_offset": 0,
            "element_bytes": 4,
            "element_shape": [],
            "halo_high": 1,
            "halo_low": -2,
            "lane_count": 1,
            "scalar_bytes": 4,
            "tile_bytes": 524,
            "tile_elements": 131,
            "access_offsets": [-2, 0, 1],
            "logical_output_count": 4095,
            "direct_input_records": 12285,
            "staged_input_records": 4191,
            "direct_input_bytes": 49140,
            "staged_input_bytes": 16764,
        },
        {
            "alignment": 4,
            "arg_index": 1,
            "arg_name": "right",
            "byte_offset": 524,
            "element_bytes": 4,
            "element_shape": [],
            "halo_high": 2,
            "halo_low": -1,
            "lane_count": 1,
            "scalar_bytes": 4,
            "tile_bytes": 524,
            "tile_elements": 131,
            "access_offsets": [-1, 2],
            "logical_output_count": 4095,
            "direct_input_records": 8190,
            "staged_input_records": 4191,
            "direct_input_bytes": 32760,
            "staged_input_bytes": 16764,
        },
    ]
    assert {tuple(pair) for pair in staged["memory_disjoint_pairs"]} == {
        ("left", "output"),
        ("right", "output"),
    }
    assert {tuple(item) for item in staged["memory_layout_requirements"]} == {
        ("left", count - 1, 4, 4),
        ("right", count, 4, 4),
        ("output", count - 2, 4, 4),
    }

    parameters = {
        "domain_fingerprint": search.domain_fingerprint,
        "recipe_id": staged_id,
    }
    materialized = search.materialize(parameters)
    staged_graph = materialized.executor
    search.verify_materialized_graph(parameters, materialized)
    task = next(
        item for item in staged_graph.task_manifest() if item.task_type == "range_for"
    )
    assert task.staged_external_arg_indices == (0, 1)
    assert task.staged_halo_lows == (-2, -1)
    assert task.staged_halo_highs == (1, 2)
    assert task.staged_byte_offsets == (0, 524)
    assert task.staged_element_bytes == (4, 4)
    assert task.staged_scalar_bytes == (4, 4)
    assert task.staged_element_shapes == ((), ())
    assert task.static_shared_bytes == 1048

    left = ti.ndarray(ti.f32, shape=count)
    right = ti.ndarray(ti.f32, shape=count)
    output = ti.ndarray(ti.f32, shape=count)
    left_values = np.arange(count, dtype=np.float32) * 0.25
    right_values = (np.arange(count, dtype=np.float32) % 23) * 0.5
    left.from_numpy(left_values)
    right.from_numpy(right_values)
    output.fill(0)
    bindings = staged_graph.bind({"left": left, "right": right, "output": output})
    assert bindings.fast_path_qualified
    assert staged_graph.bind(
        {"left": left, "right": left, "output": output}
    ).fast_path_qualified
    with pytest.raises(RuntimeError, match="requires proven disjoint storage"):
        staged_graph.bind({"left": left, "right": right, "output": left})
    with monkeypatch.context() as stable_replay:
        stable_replay.setattr(
            graph_impl,
            "analyze_storage_alias",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("stable replay repeated alias analysis")
            ),
        )
        stable_replay.setattr(
            graph_impl,
            "describe_storage",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("stable replay repeated storage description")
            ),
        )
        stable_replay.setattr(
            graph_impl,
            "validate_storage_owner",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("stable replay repeated owner validation")
            ),
        )
        staged_graph.run(bindings)
        staged_graph.run(bindings)
    ti.sync()
    expected = np.zeros(count, dtype=np.float32)
    expected[2:-2] = (
        left_values[:-4]
        + left_values[2:-2]
        + left_values[3:-1]
        + right_values[1:-3] * 0.5
        + right_values[4:]
    )
    np.testing.assert_array_equal(output.to_numpy(), expected)
    materialized.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_graph_memory_multi_source_layout_aligns_mixed_scalar_tiles():
    count = 259

    @ti.kernel
    def mixed_stencil(
        narrow: ti.types.ndarray(dtype=ti.f16, ndim=1),
        wide: ti.types.ndarray(dtype=ti.f64, ndim=1),
        output: ti.types.ndarray(dtype=ti.f64, ndim=1),
    ):
        for i in range(1, count - 1):
            output[i] = (
                ti.cast(narrow[i - 1], ti.f64)
                + ti.cast(narrow[i + 1], ti.f64)
                + wide[i - 1]
                + wide[i + 1]
            )

    narrow_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "narrow", ti.f16, ndim=1)
    wide_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "wide", ti.f64, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f64, ndim=1)

    def build():
        builder = ti.graph.GraphBuilder()
        builder.dispatch(mixed_stencil, narrow_arg, wide_arg, output_arg)
        return builder.compile()

    graph = build()
    search = compileiq_recipe_search(graph)
    staged_id, staged = next(
        (recipe_id, search.recipe_manifest(recipe_id)["memory_recipe_manifest"])
        for recipe_id in search.recipe_ids
        if search.recipe_manifest(recipe_id)["memory_recipe_manifest"]["strategy"]
        == "shared_staged_1d"
        and tuple(
            source["arg_index"]
            for source in search.recipe_manifest(recipe_id)["memory_recipe_manifest"][
                "staged_sources"
            ]
        )
        == (0, 1)
        and search.recipe_manifest(recipe_id)["memory_recipe_manifest"]["offload_plan"][
            "tasks"
        ][0]["workgroup_size"]
        == 128
    )
    assert [source["byte_offset"] for source in staged["staged_sources"]] == [
        0,
        264,
    ]
    assert [source["alignment"] for source in staged["staged_sources"]] == [2, 8]
    assert [source["tile_bytes"] for source in staged["staged_sources"]] == [
        260,
        1040,
    ]

    parameters = {
        "domain_fingerprint": search.domain_fingerprint,
        "recipe_id": staged_id,
    }
    materialized = search.materialize(parameters)

    staged_graph = materialized.executor

    search.verify_materialized_graph(parameters, materialized)
    task = next(
        item for item in staged_graph.task_manifest() if item.task_type == "range_for"
    )
    assert task.staged_byte_offsets == (0, 264)
    assert task.staged_element_bytes == (2, 8)
    assert task.static_shared_bytes == 1304

    narrow = ti.ndarray(ti.f16, shape=count)
    wide = ti.ndarray(ti.f64, shape=count)
    output = ti.ndarray(ti.f64, shape=count)
    narrow_values = (np.arange(count, dtype=np.int64) % 11).astype(np.float16)
    wide_values = (np.arange(count, dtype=np.int64) % 17).astype(np.float64)
    narrow.from_numpy(narrow_values)
    wide.from_numpy(wide_values)
    output.fill(0)
    staged_graph.run({"narrow": narrow, "wide": wide, "output": output})
    ti.sync()
    expected = np.zeros(count, dtype=np.float64)
    expected[1:-1] = (
        narrow_values[:-2].astype(np.float64)
        + narrow_values[2:].astype(np.float64)
        + wide_values[:-2]
        + wide_values[2:]
    )
    np.testing.assert_array_equal(output.to_numpy(), expected)

    materialized.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_graph_memory_compileiq_keeps_one_byte_scalar_stencils_out_of_domain():
    count = 257

    @ti.kernel
    def stencil(
        source: ti.types.ndarray(dtype=ti.i8, ndim=1),
        output: ti.types.ndarray(dtype=ti.i8, ndim=1),
    ):
        for i in range(1, count - 1):
            output[i] = source[i - 1] + source[i + 1]

    source_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "source", ti.i8, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.i8, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(stencil, source_arg, output_arg)
    graph = builder.compile()
    search = compileiq_recipe_search(graph)

    assert len(search.recipe_ids) == 1
    assert "only two-, four-, or eight-byte primitive lanes are supported" in (
        graph._spec._graph_memory_sources[0].candidate_failure
    )


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_graph_memory_compileiq_rejects_unsupported_candidates():
    count = 256

    @ti.kernel
    def pointwise(
        source: ti.types.ndarray(dtype=ti.f32, ndim=1),
        output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in range(count):
            output[i] = source[i] * 2.0

    source_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "source", ti.f32, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(pointwise, source_arg, output_arg)
    graph = builder.compile()
    search = compileiq_recipe_search(graph)

    assert graph._spec._graph_memory_sources[0].candidate_failure
    assert len(search.recipe_ids) == 1
    assert all(
        "memory_recipe_id" not in search.recipe_manifest(recipe_id)
        for recipe_id in search.recipe_ids
    )

    with pytest.raises(KeyError, match="unknown complete Graph recipe"):
        search.materialize(
            {
                "domain_fingerprint": search.domain_fingerprint,
                "recipe_id": "graph-recipe:" + "0" * 64,
            }
        )

    multi = ti.graph.GraphBuilder()
    multi.dispatch(pointwise, source_arg, output_arg)
    multi.dispatch(pointwise, source_arg, output_arg)
    multi_graph = multi.compile()
    multi_search = compileiq_recipe_search(multi_graph)
    assert all(
        "memory_recipe_id" not in multi_search.recipe_manifest(recipe_id)
        for recipe_id in multi_search.recipe_ids
    )


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_private_graph_shared_staged_recipe_materializes_and_replays_exactly(
    monkeypatch,
):
    count = 1027

    @ti.kernel
    def stencil(
        source: ti.types.ndarray(dtype=ti.f32, ndim=1),
        output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in range(1, count - 1):
            output[i] = source[i - 1] + source[i] * 2.0 + source[i + 1]

    source = ti.ndarray(ti.f32, shape=count)
    output = ti.ndarray(ti.f32, shape=count)
    values = np.arange(count, dtype=np.float32) * 0.25
    source.from_numpy(values)
    plan = _shared_staged_plan(stencil, source, output)
    bound = _bind_offload_execution_plan(stencil, plan)

    manifest = next(
        task
        for task in bound.task_manifest(source, output)
        if task.task_type == "range_for"
    )
    assert manifest.requested_memory_strategy == "shared_staged_1d"
    assert manifest.range_mapping == "shared_tiled_one_to_one"
    assert manifest.selected_block_size == 128
    assert manifest.selected_grid_size == (count - 2 + 127) // 128
    assert manifest.staged_external_arg_index == 0
    assert (manifest.staged_halo_low, manifest.staged_halo_high) == (-1, 1)
    assert manifest.static_shared_bytes == (128 + 2) * 4

    source_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "source", ti.f32, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder._dispatch_shared_staged_1d(bound, source_arg, output_arg)
    graph = builder.compile()
    graph._graph_stats

    alias_checks = 0
    description_calls = 0
    owner_validation_calls = 0
    original_alias_check = graph_impl.analyze_storage_alias
    original_describe_storage = graph_impl.describe_storage
    original_validate_storage_owner = graph_impl.validate_storage_owner

    def counted_alias_check(*args, **kwargs):
        nonlocal alias_checks
        alias_checks += 1
        return original_alias_check(*args, **kwargs)

    def counted_describe_storage(*args, **kwargs):
        nonlocal description_calls
        description_calls += 1
        return original_describe_storage(*args, **kwargs)

    def counted_validate_storage_owner(*args, **kwargs):
        nonlocal owner_validation_calls
        owner_validation_calls += 1
        return original_validate_storage_owner(*args, **kwargs)

    monkeypatch.setattr(graph_impl, "analyze_storage_alias", counted_alias_check)
    monkeypatch.setattr(graph_impl, "describe_storage", counted_describe_storage)
    monkeypatch.setattr(
        graph_impl, "validate_storage_owner", counted_validate_storage_owner
    )

    binding_plan = graph.binding_plan()
    assert binding_plan["memory_recipe_names"] == ("output", "source")
    assert binding_plan["memory_recipe_publish_certificate_required"]
    assert binding_plan["memory_recipe_publish_frame_stable"]
    bindings = graph.bind({"source": source, "output": output})
    binding_stats = bindings.statistics()
    assert bindings.fast_path_qualified
    assert binding_stats["memory_recipe_publish_validated"]
    assert binding_stats["memory_recipe_certified"]
    assert binding_stats["memory_recipe_names"] == ("output", "source")
    assert "dynamic_memory_recipe" not in binding_stats["volatile_reasons"]
    publish_description_calls = description_calls
    publish_owner_validation_calls = owner_validation_calls
    assert publish_description_calls == 2
    assert publish_owner_validation_calls == 2
    assert alias_checks == 1

    graph.run(bindings)
    ti.sync()
    expected = np.zeros(count, dtype=np.float32)
    expected[1:-1] = values[:-2] + values[1:-1] * 2.0 + values[2:]
    np.testing.assert_allclose(output.to_numpy(), expected, rtol=0, atol=0)

    first = graph._graph_stats[0]
    output.fill(0)
    graph.run(bindings)
    ti.sync()
    second = graph._graph_stats[0]
    assert first["captures"] == 1
    assert second["exact_replays"] == 1
    np.testing.assert_allclose(output.to_numpy(), expected, rtol=0, atol=0)

    for _ in range(16):
        graph.run(bindings)
    ti.sync()
    graph_identity = graph._instance_debug_info
    program = impl.get_runtime().prog
    runtime_before = program._runtime_statistics_snapshot()["memory"]
    host_before = dict(ti_core.get_host_memory_pool_stats())
    device_before = dict(ti_core.get_device_memory_pool_stats())
    for _ in range(10_000):
        graph.run(bindings)
    ti.sync()
    assert alias_checks == 1
    assert description_calls == publish_description_calls
    assert owner_validation_calls == publish_owner_validation_calls
    assert graph.binding_statistics()["version_fast_replays"] >= 10_018
    assert graph._instance_debug_info == graph_identity
    assert graph._graph_stats[0]["exact_replays"] >= 10_001
    runtime_after = program._runtime_statistics_snapshot()["memory"]
    for key in (
        "host_requested_live_bytes",
        "host_raw_bytes",
        "device_requested_live_bytes",
        "device_raw_bytes",
        "device_cached_bytes",
    ):
        if runtime_before[key] is not None and runtime_after[key] is not None:
            assert runtime_after[key] <= runtime_before[key]
    for before, after in (
        (host_before, dict(ti_core.get_host_memory_pool_stats())),
        (device_before, dict(ti_core.get_device_memory_pool_stats())),
    ):
        for key in (
            "raw_chunks",
            "requested_live_bytes",
            "raw_bytes",
            "reserved_bytes",
            "committed_bytes",
            "used_bytes",
            "cached_blocks",
            "cached_bytes",
        ):
            if key in before and key in after:
                assert after[key] <= before[key]

    # Recurring resource sets can each be published once. Switching A -> B -> A
    # then reuses both immutable Python certificates; the native CGraph cache
    # independently reuses its generation-qualified resource plans.
    alternate_values = values[::-1].copy()
    alternate_source = ti.ndarray(ti.f32, shape=count)
    alternate_output = ti.ndarray(ti.f32, shape=count)
    alternate_source.from_numpy(alternate_values)
    alternate_bindings = graph.bind(
        {"source": alternate_source, "output": alternate_output}
    )
    assert alternate_bindings.fast_path_qualified
    recurring_description_calls = description_calls
    recurring_owner_validation_calls = owner_validation_calls
    recurring_alias_checks = alias_checks

    output.fill(0)
    alternate_output.fill(0)
    graph.run(bindings)
    graph.run(alternate_bindings)
    graph.run(bindings)
    ti.sync()

    assert description_calls == recurring_description_calls
    assert owner_validation_calls == recurring_owner_validation_calls
    assert alias_checks == recurring_alias_checks
    np.testing.assert_allclose(output.to_numpy(), expected, rtol=0, atol=0)
    alternate_expected = np.zeros(count, dtype=np.float32)
    alternate_expected[1:-1] = (
        alternate_values[:-2] + alternate_values[1:-1] * 2.0 + alternate_values[2:]
    )
    np.testing.assert_allclose(
        alternate_output.to_numpy(), alternate_expected, rtol=0, atol=0
    )

    # Mutable compatibility dictionaries deliberately keep one exact owner
    # scan per replay. A new descriptor tuple proves aliasing once, then the
    # collision-free cache skips only the exhaustive alias/layout analysis.
    raw_source = ti.ndarray(ti.f32, shape=count)
    raw_output = ti.ndarray(ti.f32, shape=count)
    raw_source.from_numpy(values)
    raw_descriptions_before = description_calls
    raw_owner_validations_before = owner_validation_calls
    raw_alias_checks_before = alias_checks
    graph.run({"source": raw_source, "output": raw_output})
    graph.run({"source": raw_source, "output": raw_output})
    ti.sync()
    assert description_calls - raw_descriptions_before == 4
    assert owner_validation_calls - raw_owner_validations_before == 4
    assert alias_checks - raw_alias_checks_before == 1

    ti.reset()
    with pytest.raises(RuntimeError, match="compiled before ti.reset"):
        graph.run(bindings)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_private_graph_shared_staged_recipe_is_graph_owned_and_alias_safe():
    count = 257

    @ti.kernel
    def stencil(
        source: ti.types.ndarray(dtype=ti.f32, ndim=1),
        output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in range(1, count - 1):
            output[i] = source[i - 1] + source[i + 1]

    source = ti.ndarray(ti.f32, shape=count)
    output = ti.ndarray(ti.f32, shape=count)
    bound = _bind_offload_execution_plan(
        stencil, _shared_staged_plan(stencil, source, output)
    )
    source_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "source", ti.f32, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1)

    with pytest.raises(RuntimeError, match="Graph-owned"):
        bound(source, output)
    with pytest.raises(TaichiCompilationError, match="Graph-owned memory recipe"):
        ti.graph.GraphBuilder().dispatch(bound, source_arg, output_arg)

    builder = ti.graph.GraphBuilder()
    builder._dispatch_shared_staged_1d(bound, source_arg, output_arg)
    graph = builder.compile()
    bindings = graph.bind({"source": source, "output": output})
    assert bindings.fast_path_qualified
    replacement_source = ti.ndarray(ti.f32, shape=count)
    replacement_values = np.linspace(-2.0, 3.0, count, dtype=np.float32)
    replacement_source.from_numpy(replacement_values)
    bindings.update(source=replacement_source)
    assert bindings.fast_path_qualified
    assert bindings.statistics()["memory_recipe_certified"]
    output.fill(0)
    graph.run(bindings)
    ti.sync()
    expected = np.zeros(count, dtype=np.float32)
    expected[1:-1] = replacement_values[:-2] + replacement_values[2:]
    np.testing.assert_allclose(output.to_numpy(), expected, rtol=0, atol=0)

    revision = bindings.revision
    with pytest.raises(RuntimeError, match="requires proven disjoint storage"):
        bindings.update(output=replacement_source)
    assert bindings.revision == revision
    output.fill(0)
    graph.run(bindings)
    ti.sync()
    np.testing.assert_allclose(output.to_numpy(), expected, rtol=0, atol=0)

    with pytest.raises(RuntimeError, match="requires proven disjoint storage"):
        graph.run({"source": source, "output": source})

    short_source = ti.ndarray(ti.f32, shape=count - 1)
    short_output = ti.ndarray(ti.f32, shape=count - 1)
    with pytest.raises(RuntimeError, match="at least 257 scalar elements"):
        graph.bind({"source": short_source, "output": short_output})


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_private_graph_shared_staged_recipe_validates_final_provider_bindings():
    count = 257

    @ti.kernel
    def stencil(
        source: ti.types.ndarray(dtype=ti.f32, ndim=1),
        output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in range(1, count - 1):
            output[i] = source[i - 1] + source[i + 1]

    source = ti.ndarray(ti.f32, shape=count)
    output = ti.ndarray(ti.f32, shape=count)
    bound = _bind_offload_execution_plan(
        stencil, _shared_staged_plan(stencil, source, output)
    )
    source_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "source", ti.f32, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder._dispatch_shared_staged_1d(bound, source_arg, output_arg)
    graph = builder.compile()

    class _AliasingProvider:
        def __init__(self):
            self.binding = None

        def bind_graph_arguments(self, runtime_args):
            alias = graph_impl.ProviderOwnedNdarrayBinding(
                runtime_args["source"].arr,
                self,
            )
            self.binding = alias
            return graph_impl.PreparedGraphBindings(
                MappingProxyType({"output": alias}),
                (self,),
            )

    # A dynamic provider is normally contributed by a mixed native segment.
    # Inject its production binding result here so this shared-stage-only
    # fixture proves that memory contracts validate the final provider-owned
    # frame, after replacements and with its exact submission owner attached.
    provider = _AliasingProvider()
    graph._spec.lifetime_leases = (provider,)
    with pytest.raises(RuntimeError, match="requires proven disjoint storage"):
        graph.run({"source": source, "output": output})
    assert provider.binding is not None
    with pytest.raises(AttributeError):
        provider.binding.arr = output.arr


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_private_graph_shared_staged_recipe_snapshots_mapping_proxy(monkeypatch):
    count = 257

    @ti.kernel
    def stencil(
        source: ti.types.ndarray(dtype=ti.f32, ndim=1),
        output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in range(1, count - 1):
            output[i] = source[i - 1] + source[i + 1]

    source = ti.ndarray(ti.f32, shape=count)
    output = ti.ndarray(ti.f32, shape=count)
    values = np.arange(count, dtype=np.float32)
    source.from_numpy(values)
    output.fill(0)
    bound = _bind_offload_execution_plan(
        stencil, _shared_staged_plan(stencil, source, output)
    )
    source_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "source", ti.f32, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder._dispatch_shared_staged_1d(bound, source_arg, output_arg)
    graph = builder.compile()

    backing = {"source": source, "output": output}
    arguments = MappingProxyType(backing)
    original_validate = graph._spec._validate_bound_runtime_args

    def validate_then_mutate_backing(validation_args, **kwargs):
        certificate = original_validate(validation_args, **kwargs)
        backing["output"] = source
        return certificate

    monkeypatch.setattr(
        graph._spec,
        "_validate_bound_runtime_args",
        validate_then_mutate_backing,
    )
    graph.run(arguments)
    ti.sync()

    assert backing["output"] is source
    expected = np.zeros(count, dtype=np.float32)
    expected[1:-1] = values[:-2] + values[2:]
    np.testing.assert_allclose(output.to_numpy(), expected, rtol=0, atol=0)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_private_graph_shared_staged_recipe_rejects_pointwise_input():
    count = 256

    @ti.kernel
    def pointwise(
        source: ti.types.ndarray(dtype=ti.f32, ndim=1),
        output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in range(count):
            output[i] = source[i] * 2.0

    source = ti.ndarray(ti.f32, shape=count)
    output = ti.ndarray(ti.f32, shape=count)
    plan = _shared_staged_plan(pointwise, source, output)
    with pytest.raises(RuntimeError, match="at least two distinct affine offsets"):
        _bind_offload_execution_plan(pointwise, plan).task_manifest(source, output)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_graph_memory_compound_records_materialize_complete_lane_layout():
    count = 259

    for element_shape, scalar_dtype, numpy_dtype in (
        ((3,), ti.f32, np.float32),
        ((2, 2), ti.f64, np.float64),
    ):
        element_type = (
            ti.types.vector(element_shape[0], scalar_dtype)
            if len(element_shape) == 1
            else ti.types.matrix(element_shape[0], element_shape[1], scalar_dtype)
        )

        @ti.kernel
        def compound_stencil(
            source: ti.types.ndarray(dtype=element_type, ndim=1),
            output: ti.types.ndarray(dtype=element_type, ndim=1),
        ):
            for i in range(1, count - 1):
                output[i] = source[i - 1] + source[i] * 2 + source[i + 1]

        source_arg = ti.graph.Arg(
            ti.graph.ArgKind.NDARRAY, "source", element_type, ndim=1
        )
        output_arg = ti.graph.Arg(
            ti.graph.ArgKind.NDARRAY, "output", element_type, ndim=1
        )

        def build():
            builder = ti.graph.GraphBuilder()
            builder.dispatch(compound_stencil, source_arg, output_arg)
            return builder.compile()

        graph = build()
        search = compileiq_recipe_search(graph)
        staged_id, staged = next(
            (recipe_id, search.recipe_manifest(recipe_id)["memory_recipe_manifest"])
            for recipe_id in search.recipe_ids
            if search.recipe_manifest(recipe_id)["memory_recipe_manifest"]["strategy"]
            == "shared_staged_1d"
            and search.recipe_manifest(recipe_id)["memory_recipe_manifest"][
                "offload_plan"
            ]["tasks"][0]["workgroup_size"]
            == 128
        )
        scalar_bytes = np.dtype(numpy_dtype).itemsize
        lane_count = int(np.prod(element_shape))
        record_bytes = scalar_bytes * lane_count
        assert staged["schema_version"] == 5
        assert staged["staged_sources"] == [
            {
                "alignment": scalar_bytes,
                "arg_index": 0,
                "arg_name": "source",
                "byte_offset": 0,
                "element_bytes": record_bytes,
                "element_shape": list(element_shape),
                "halo_high": 1,
                "halo_low": -1,
                "lane_count": lane_count,
                "scalar_bytes": scalar_bytes,
                "tile_bytes": (128 + 2) * record_bytes,
                "tile_elements": 128 + 2,
                "access_offsets": [-1, 0, 1],
                "logical_output_count": 257,
                "direct_input_records": 771,
                "staged_input_records": 263,
                "direct_input_bytes": 771 * record_bytes,
                "staged_input_bytes": 263 * record_bytes,
            }
        ]
        assert staged["memory_layout_requirements"] == [
            [
                "output",
                count - 1,
                record_bytes,
                scalar_bytes,
                list(element_shape),
                "aos",
            ],
            ["source", count, record_bytes, scalar_bytes, list(element_shape), "aos"],
        ]

        parameters = {
            "domain_fingerprint": search.domain_fingerprint,
            "recipe_id": staged_id,
        }
        materialized = search.materialize(parameters)
        staged_graph = materialized.executor
        search.verify_materialized_graph(parameters, materialized)
        task = next(
            item
            for item in staged_graph.task_manifest()
            if item.task_type == "range_for"
        )
        assert task.staged_element_bytes == (record_bytes,)
        assert task.staged_scalar_bytes == (scalar_bytes,)
        assert task.staged_element_shapes == (element_shape,)
        assert task.static_shared_bytes == (128 + 2) * record_bytes

        source = (
            ti.Vector.ndarray(element_shape[0], scalar_dtype, shape=count)
            if len(element_shape) == 1
            else ti.Matrix.ndarray(
                element_shape[0], element_shape[1], scalar_dtype, shape=count
            )
        )
        output = (
            ti.Vector.ndarray(element_shape[0], scalar_dtype, shape=count)
            if len(element_shape) == 1
            else ti.Matrix.ndarray(
                element_shape[0], element_shape[1], scalar_dtype, shape=count
            )
        )
        values = np.arange(count * lane_count, dtype=numpy_dtype).reshape(
            (count, *element_shape)
        )
        source.from_numpy(values)
        output.fill(0)
        bindings = staged_graph.bind({"source": source, "output": output})
        assert bindings.fast_path_qualified
        staged_graph.run(bindings)
        ti.sync()
        expected = np.zeros_like(values)
        expected[1:-1] = values[:-2] + values[1:-1] * 2 + values[2:]
        np.testing.assert_array_equal(output.to_numpy(), expected)
        materialized.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_graph_memory_compound_recipe_rejects_partial_lane_policy():
    count = 259
    vec3 = ti.types.vector(3, ti.f32)

    @ti.kernel
    def partial_lane_stencil(
        source: ti.types.ndarray(dtype=vec3, ndim=1),
        output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in range(1, count - 1):
            output[i] = source[i - 1][0] + source[i + 1][0]

    source_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "source", vec3, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(partial_lane_stencil, source_arg, output_arg)
    graph = builder.compile()
    search = compileiq_recipe_search(graph)
    assert len(search.recipe_ids) == 1
    assert "compound staged inputs must read every lane" in (
        graph._spec._graph_memory_sources[0].candidate_failure
    )
