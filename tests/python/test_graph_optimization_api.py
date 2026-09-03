from itertools import pairwise
import json

import numpy as np
import pytest
import taichi_forge as ti

from tests import test_utils


def test_graph_optimization_public_contract_rejects_ambiguous_inputs():
    with pytest.raises(ValueError, match="objective"):
        ti.graph.GraphOptimizationTarget(objectives=())
    with pytest.raises(ValueError, match="direction"):
        ti.graph.GraphOptimizationTarget(objectives=(("time", "fastest"),))
    with pytest.raises(ValueError, match="relation"):
        ti.graph.GraphOptimizationTarget(
            constraints=(("memory", "<", 1.0),)
        )
    with pytest.raises(ValueError, match="evaluation_limit"):
        ti.graph.GraphSearchBudget(0)
    with pytest.raises(ValueError, match="cover one complete recipe"):
        ti.graph.GraphSearchBudget(1, repeat_count=2)
    with pytest.raises(ValueError, match="materialized_memory_limit_bytes"):
        ti.graph.GraphSearchBudget(1, materialized_memory_limit_bytes=-1)
    with pytest.raises(ValueError, match="strategy mode"):
        ti.graph.GraphRecipeSearchStrategy(mode="guess")
    with pytest.raises(ValueError, match="cannot exceed"):
        ti.graph.GraphRecipeSearchStrategy(
            exact_composition_limit=4,
            max_generated_recipes=3,
        )


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_public_complete_recipe_search_materializes_measured_pareto_decision():
    lengths = np.asarray((1, 7, 32, 33, 129, 511), dtype=np.int32)
    offsets = np.concatenate(
        (np.zeros(1, dtype=np.int32), np.cumsum(lengths, dtype=np.int32))
    )
    capacity = int(offsets[-1])
    layout = ti.algorithms.SegmentedLayout.from_offsets(
        offsets,
        capacity=capacity,
    )
    values = ti.ndarray(ti.i32, shape=capacity)
    output = ti.ndarray(ti.i32, shape=capacity)
    builder = ti.graph.GraphBuilder()
    builder.segmented_scan(values, layout, output, inclusive=False)
    definition = builder.freeze()

    target = ti.graph.GraphOptimizationTarget(
        objectives=(
            ("physical_dispatches", "min"),
            ("materialized_memory_bytes", "min"),
        )
    )
    budget = ti.graph.GraphSearchBudget(
        evaluation_limit=10,
        repeat_count=2,
        deterministic_seed=17,
    )
    session = definition.search_recipes(
        engine="compileiq",
        target=target,
        budget=budget,
    )
    assert len(session.recipes) == 5
    assert session.baseline.manifest.is_baseline
    assert all(
        recipe.semantic_graph_id == definition.semantic_graph_id
        for recipe in session.recipes
    )
    assert all(
        "block_dim" not in json.dumps(recipe.to_dict(), sort_keys=True)
        for recipe in session.recipes
    )

    host = ((np.arange(capacity, dtype=np.int64) % 11) - 5).astype(np.int32)
    expected = np.empty_like(host)
    for begin, end in pairwise(offsets):
        inclusive = np.cumsum(host[begin:end], dtype=np.int32)
        expected[begin:end] = np.concatenate(
            (np.zeros(1, dtype=np.int32), inclusive[:-1])
        )
    observed = []

    def evaluator(graph, recipe):
        values.from_numpy(host)
        output.fill(123)
        graph.run({})
        ti.sync()
        np.testing.assert_array_equal(output.to_numpy(), expected)
        observed.append(recipe.recipe_id)
        return {
            "physical_dispatches": float(
                graph.physical_plan()["physical_dispatch_count"]
            )
        }

    decision = session.run(evaluator)
    report = decision.report
    assert report.search_complete
    assert report.evaluation_count == 10
    assert set(observed) == {recipe.recipe_id for recipe in session.recipes}
    assert report.missing_recipe_ids == ()
    assert report.selected_recipe_id == decision.selection.recipe_id
    assert decision.selection in decision.pareto_frontier
    assert all(
        "materialized_memory_bytes" in result["metrics"]
        for result in report.results
    )
    assert report.compileiq_capability["schema"] == (
        "compileiq.taichi-forge-recipe-search-capability.v2"
    )
    assert report.compileiq_provenance["verification"] == (
        "bundled_manifest_lock_at_search_start"
    )
    json.dumps(decision.to_dict(), sort_keys=True, allow_nan=False)

    with definition.materialize(decision.selection) as materialized:
        values.from_numpy(host)
        output.fill(123)
        materialized.executor.run({})
        ti.sync()
        np.testing.assert_array_equal(output.to_numpy(), expected)
        physical = materialized.materialization_report()
        assert physical["recipe_id"] == decision.selection.recipe_id
        assert physical["planned_physical_id"] == (
            decision.selection.planned_physical_id
        )
        assert physical["materialized_physical_id"]

    staged_strategy = ti.graph.GraphRecipeSearchStrategy(
        mode="staged",
        max_generation_rounds=2,
        max_generated_recipes=16,
    )
    partial_session = definition.search_recipes(
        engine="compileiq",
        target=target,
        budget=ti.graph.GraphSearchBudget(
            evaluation_limit=6,
            deterministic_seed=17,
        ),
        strategy=staged_strategy,
    )
    partial = partial_session.run(evaluator)
    assert partial.selection is None
    assert not partial.report.search_complete
    assert partial.report.status["terminal_state"] == "budget_exhausted"
    assert partial.report.status["generation_status"] == "not_finalized"
    partial_checkpoint = partial.report.checkpoint
    assert len(partial_checkpoint["batches"]) == 2
    assert not partial_checkpoint["stages"][-1]["complete"]
    assert partial_checkpoint["batches"][-1]["fidelity"]["terminal"]
    survivors = set(partial_checkpoint["stages"][0]["survivor_recipe_ids"])
    assert all(
        set(item["parent_recipe_ids"]).issubset(survivors)
        for item in partial_checkpoint["batches"][1]["recipes"]
    )

    resumed_observed = []

    def resumed_evaluator(graph, recipe):
        resumed_observed.append(recipe.recipe_id)
        return evaluator(graph, recipe)

    resumed_session = definition.search_recipes(
        engine="compileiq",
        target=target,
        budget=ti.graph.GraphSearchBudget(
            evaluation_limit=16,
            deterministic_seed=17,
        ),
        strategy=staged_strategy,
        checkpoint=partial_checkpoint,
    )
    resumed = resumed_session.run(resumed_evaluator)
    assert resumed.selection is not None
    assert resumed.report.search_complete
    assert resumed.report.termination_reason == "no_new_physical_identity"
    assert resumed.report.status["generation_status"] == "strategy_complete"
    assert resumed.report.status["terminal_fidelity_status"] == "complete"
    assert resumed.report.strategy.strategy_id == staged_strategy.strategy_id
    assert resumed_observed
    measurement_keys = tuple(
        item["request"]["measurement_key"]
        for item in resumed.report.checkpoint["records"]
    )
    assert len(measurement_keys) == len(set(measurement_keys))

    with pytest.raises(RuntimeError, match="single-use"):
        session.run(evaluator)


@test_utils.test(arch=ti.cpu)
def test_recipe_handle_is_bound_to_its_frozen_definition():
    @ti.kernel
    def fill(output: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for index in output:
            output[index] = index

    output_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY,
        "output",
        ti.i32,
        ndim=1,
    )
    first_builder = ti.graph.GraphBuilder()
    first_builder.dispatch(fill, output_arg)
    first = first_builder.freeze()
    second_builder = ti.graph.GraphBuilder()
    second_builder.dispatch(fill, output_arg)
    second_builder.dispatch(fill, output_arg)
    second = second_builder.freeze()
    handle = ti.graph.GraphRecipeHandle._from_recipe(
        first,
        first.recipe_catalog().baseline.recipe,
    )

    with pytest.raises(ValueError, match="different GraphDefinition"):
        second.materialize(handle)
