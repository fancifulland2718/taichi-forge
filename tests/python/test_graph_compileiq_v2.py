from itertools import pairwise

import numpy as np
import taichi_forge as ti
from taichi_forge.graph import compileiq_recipe_search

from tests import test_utils


def _budget(evaluations, *, memory=1 << 30):
    from compileiq.forge_support import ForgeOpaqueSearchBudgetV2

    return ForgeOpaqueSearchBudgetV2(
        evaluation_limit=evaluations,
        time_limit_seconds=300.0,
        materialized_memory_limit_bytes=memory,
    )


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_complete_recipe_v2_budget_resume_reuses_real_graph_measurements():
    count = 1025

    @ti.kernel
    def first(
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        temporary: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in source:
            temporary[i] = source[i] * 2

    @ti.kernel
    def second(
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        temporary: ti.types.ndarray(dtype=ti.i32, ndim=1),
        middle: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in source:
            middle[i] = temporary[i] + 3

    @ti.kernel
    def third(
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        middle: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in source:
            output[i] = middle[i] * 4

    symbolic = {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.i32, ndim=1)
        for name in ("source", "temporary", "middle", "output")
    }
    builder = ti.graph.GraphBuilder()
    builder.dispatch(first, symbolic["source"], symbolic["temporary"])
    builder.dispatch(
        second,
        symbolic["source"],
        symbolic["temporary"],
        symbolic["middle"],
    )
    builder.dispatch(
        third,
        symbolic["source"],
        symbolic["middle"],
        symbolic["output"],
    )
    baseline = builder.compile(workspace_lanes=2, workspace_saturation="raise")
    plans = compileiq_recipe_search(baseline)
    assert len(plans.recipe_ids) == 4
    canonical_batch = plans.batch(recipe_ids=plans.recipe_ids)
    reordered_batch = plans.batch(recipe_ids=reversed(plans.recipe_ids))
    assert reordered_batch == canonical_batch
    assert reordered_batch.batch_fingerprint == canonical_batch.batch_fingerprint

    source = ti.ndarray(ti.i32, shape=count)
    temporary = ti.ndarray(ti.i32, shape=count)
    middle = ti.ndarray(ti.i32, shape=count)
    output = ti.ndarray(ti.i32, shape=count)
    source_np = np.arange(count, dtype=np.int32)
    source.from_numpy(source_np)
    arguments = {
        "source": source,
        "temporary": temporary,
        "middle": middle,
        "output": output,
    }
    first_measurements = []

    def objective(graph, request):
        output.fill(0)
        graph.run(arguments)
        ti.sync()
        np.testing.assert_array_equal(output.to_numpy(), (source_np * 2 + 3) * 4)
        physical_dispatches = graph.physical_plan()["physical_dispatch_count"]
        first_measurements.append((request.measurement_key, request.recipe_id))
        return float(physical_dispatches)

    with plans.compileiq_search(
        objective,
        budget=_budget(2),
        repeat_count=2,
        deterministic_seed=23,
    ) as partial_session:
        partial = partial_session.start()
        checkpoint = partial.checkpoint()
        partial_coverage = plans.search_coverage(partial_session)

    assert partial.termination_reason == "evaluation_budget_exhausted"
    assert partial_coverage["evaluation_count"] == 2
    assert partial_coverage["baseline_observed"]
    assert len(first_measurements) == 2

    resumed_measurements = []

    def resumed_objective(graph, request):
        output.fill(0)
        graph.run(arguments)
        ti.sync()
        np.testing.assert_array_equal(output.to_numpy(), (source_np * 2 + 3) * 4)
        resumed_measurements.append((request.measurement_key, request.recipe_id))
        return float(graph.physical_plan()["physical_dispatch_count"])

    with plans.compileiq_search(
        resumed_objective,
        budget=_budget(len(plans.recipe_ids) * 2),
        repeat_count=2,
        deterministic_seed=23,
        checkpoint=checkpoint,
    ) as resumed_session:
        result = resumed_session.start()
        coverage = plans.require_complete_search(resumed_session)
        selected = plans.select_best_result(resumed_session, result)

    assert coverage["complete"]
    assert coverage["evaluation_count"] == len(plans.recipe_ids) * 2
    assert len(resumed_measurements) == len(plans.recipe_ids) * 2 - 2
    assert not {key for key, _ in first_measurements}.intersection(
        key for key, _ in resumed_measurements
    )
    measured = result.get_results()
    assert len({item["materialized_physical_id"] for item in measured}) >= 2
    assert selected.recipe_id in plans.recipe_ids
    assert result.get_best_result()["metrics"]["score"] == 1.0


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_complete_recipe_v2_multifidelity_bounded_racing_keeps_lineage():
    from compileiq.forge_support import (
        ForgeOpaqueObjectiveV1,
        ForgeOpaqueTargetContractV1,
    )

    capacity = 64

    @ti.kernel
    def publish(
        requested: ti.i32,
        extent: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        ti.device_extent_publish(extent, capacity, requested)

    @ti.kernel
    def consume(
        extent: ti.types.ndarray(dtype=ti.i32, ndim=1),
        total: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in range(capacity):
            if i < ti.device_extent_count(extent):
                ti.atomic_add(total[0], i + 1)

    requested_arg = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "requested", ti.i32)
    extent_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "extent", ti.i32, ndim=1)
    total_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "total", ti.i32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(publish, requested_arg, extent_arg)
    builder.dispatch_bounded(
        consume,
        extent_arg,
        total_arg,
        extent=extent_arg,
        capacity=capacity,
        block_dim=32,
    )
    baseline = builder.compile()
    plans = compileiq_recipe_search(baseline)
    assert len(plans.recipe_ids) == 3

    target_contract = ForgeOpaqueTargetContractV1(
        objectives=(
            ForgeOpaqueObjectiveV1(name="physical_dispatches", direction="min"),
            ForgeOpaqueObjectiveV1(name="persistent_bytes", direction="min"),
        )
    )
    extent = ti.DeviceExtent(capacity)
    total = ti.ndarray(ti.i32, shape=1)
    arguments = {"requested": 17, "extent": extent, "total": total}
    observed_recipes = set()

    def objective(graph, request):
        total.fill(0)
        graph.run(arguments)
        ti.sync()
        assert int(total.to_numpy()[0]) == 17 * 18 // 2
        observed_recipes.add(request.recipe_id)
        return {
            "physical_dispatches": float(
                graph.physical_plan()["physical_dispatch_count"]
            ),
            "persistent_bytes": float(
                graph.execution_stats().memory.persistent_bytes
            ),
        }

    stage0 = plans.batch(
        fidelity_name="screen",
        fidelity_ordinal=0,
        repeat_count=2,
        work_scale=0.25,
    )
    with plans.compileiq_search(
        objective,
        budget=_budget(32),
        target_contract=target_contract,
        repeat_count=2,
        fidelity_name="screen",
        deterministic_seed=11,
    ) as session:
        screened = session.submit_batch(stage0)
        survivors = screened.survivor_lineage()[0]["survivor_recipe_ids"]
        assert plans.baseline_recipe_id in survivors
        assert 1 < len(survivors) < len(plans.recipe_ids)

        stage1 = plans.batch(
            recipe_ids=survivors,
            stage_index=1,
            parent_batch=stage0,
            fidelity_name="full",
            fidelity_ordinal=1,
            repeat_count=2,
            work_scale=1.0,
        )
        final = session.submit_batch(stage1)
        checkpoint = session.checkpoint()

    assert len(checkpoint.stages) == 2
    assert checkpoint.stages[1].batch_fingerprint == stage1.batch_fingerprint
    assert checkpoint.stages[1].complete
    assert observed_recipes == set(plans.recipe_ids)
    assert all(item["observation_count"] == 2 for item in final.get_results())
    assert len(
        {item["materialized_physical_id"] for item in final.get_results()}
    ) == len(final.get_results())


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_complete_recipe_v2_memory_budget_skips_native_workspace_before_materialization():
    capacity = 1024
    offsets = np.arange(0, capacity + 1, 256, dtype=np.int32)
    layout = ti.algorithms.SegmentedLayout.from_offsets(offsets, capacity=capacity)
    values = ti.ndarray(ti.i32, shape=capacity)
    output = ti.ndarray(ti.i32, shape=capacity)
    builder = ti.graph.GraphBuilder()
    builder.segmented_scan(values, layout, output, inclusive=True)
    baseline = builder.compile()
    plans = compileiq_recipe_search(baseline)
    batch = plans.batch()
    estimates = {
        recipe.recipe_id: recipe.estimated_materialized_bytes
        for recipe in batch.recipes
    }
    assert estimates[plans.baseline_recipe_id] == 0
    candidate_id = next(
        recipe_id for recipe_id, estimate in estimates.items() if estimate > 0
    )

    host = ((np.arange(capacity, dtype=np.int64) % 7) + 1).astype(np.int32)
    expected = np.empty_like(host)
    for begin, end in pairwise(offsets):
        expected[begin:end] = np.cumsum(host[begin:end], dtype=np.int32)
    observed = []

    def objective(graph, request):
        observed.append(request.recipe_id)
        values.from_numpy(host)
        output.fill(0)
        graph.run({})
        ti.sync()
        np.testing.assert_array_equal(output.to_numpy(), expected)
        return 1.0

    with plans.compileiq_search(
        objective,
        budget=_budget(len(plans.recipe_ids), memory=0),
    ) as session:
        result = session.start()

    by_recipe = {item["recipe_id"]: item for item in result.get_results()}
    assert observed == [plans.baseline_recipe_id]
    assert by_recipe[plans.baseline_recipe_id]["feasible"]
    assert by_recipe[candidate_id]["feasible"] is False
    assert by_recipe[candidate_id]["failures"][0]["failure"]["code"] == (
        "estimated_memory_budget_exceeded"
    )
