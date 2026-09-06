from itertools import pairwise
from contextlib import nullcontext
from dataclasses import replace
import json
import multiprocessing

import numpy as np
import pytest
import taichi_forge as ti

from tests import test_utils


def _resume_segmented_scan_in_fresh_process(
    checkpoint,
    workload_context,
    evaluation_contract,
    backend_environment,
    result_queue,
):
    try:
        ti.init(arch=ti.cuda, offline_cache=False)
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
        strategy = ti.graph.GraphRecipeSearchStrategy(
            mode="staged",
            max_generation_rounds=2,
            max_generated_recipes=16,
        )
        host = ((np.arange(capacity, dtype=np.int64) % 11) - 5).astype(np.int32)
        expected = np.empty_like(host)
        for begin, end in pairwise(offsets):
            inclusive = np.cumsum(host[begin:end], dtype=np.int32)
            expected[begin:end] = np.concatenate(
                (np.zeros(1, dtype=np.int32), inclusive[:-1])
            )
        evaluated = []

        def evaluator(graph, recipe):
            values.from_numpy(host)
            output.fill(123)
            graph.run({})
            ti.sync()
            np.testing.assert_array_equal(output.to_numpy(), expected)
            evaluated.append(recipe.recipe_id)
            return {
                "physical_dispatches": float(
                    graph.physical_plan()["physical_dispatch_count"]
                )
            }

        outcome = definition.search_recipes(
            target=target,
            budget=ti.graph.GraphSearchBudget(
                evaluation_limit=16,
                deterministic_seed=17,
            ),
            strategy=strategy,
            checkpoint=checkpoint,
            workload_context=ti.graph.GraphWorkloadContext.from_dict(workload_context),
            evaluation_contract=ti.graph.GraphEvaluationContract.from_dict(
                evaluation_contract
            ),
            backend_environment=ti.graph.GraphBackendEnvironment.from_dict(
                backend_environment
            ),
        ).run(evaluator)
        result_queue.put(
            {
                "search_complete": outcome.report.search_complete,
                "selected_recipe_id": outcome.selection.recipe_id,
                "evaluated_recipe_ids": tuple(evaluated),
                "evaluation_count": outcome.report.evaluation_count,
            }
        )
    except BaseException as error:
        result_queue.put(
            {
                "error": type(error).__name__,
                "message": str(error),
            }
        )
    finally:
        ti.reset()


def test_graph_optimization_public_contract_rejects_ambiguous_inputs():
    with pytest.raises(ValueError, match="objective"):
        ti.graph.GraphOptimizationTarget(objectives=())
    with pytest.raises(ValueError, match="direction"):
        ti.graph.GraphOptimizationTarget(objectives=(("time", "fastest"),))
    with pytest.raises(ValueError, match="relation"):
        ti.graph.GraphOptimizationTarget(constraints=(("memory", "<", 1.0),))
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
    offsets = np.concatenate((np.zeros(1, dtype=np.int32), np.cumsum(lengths, dtype=np.int32)))
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
    workload_context = ti.graph.GraphWorkloadContext(
        {
            "fixture": "segmented-exclusive-scan",
            "lengths": lengths.tolist(),
            "dtype": "i32",
        }
    )
    evaluation_contract = ti.graph.GraphEvaluationContract(
        {
            "warmup": 0,
            "synchronization": "ti.sync-after-run",
            "correctness": "numpy-exact",
            "metric": "physical-plan-manifest",
            "cost_profiles": {
                "contract_fixture": {
                    "scope": "synthetic arithmetic fixture, not performance evidence",
                    "unit": "us", "setup": "fixture_setup", "first": "fixture_first", "steady": "fixture_steady",
                    "amortization_model": "setup_plus_first_plus_remaining_steady",
                }
            },
        }
    )
    backend_environment = ti.graph.GraphBackendEnvironment(
        {
            "fixture": "pytest-current-cuda-device",
            "runtime": "taichi-forge-test-runtime",
        }
    )
    budget = ti.graph.GraphSearchBudget(
        evaluation_limit=14,
        repeat_count=2,
        deterministic_seed=17,
    )
    session = definition.search_recipes(
        engine="compileiq",
        target=target,
        budget=budget,
        workload_context=workload_context,
        evaluation_contract=evaluation_contract,
        backend_environment=backend_environment,
    )
    assert len(session.recipes) == 7
    assert session.baseline.manifest.is_baseline
    assert all(recipe.semantic_graph_id == definition.semantic_graph_id for recipe in session.recipes)
    assert all("block_dim" not in json.dumps(recipe.to_dict(), sort_keys=True) for recipe in session.recipes)

    host = ((np.arange(capacity, dtype=np.int64) % 11) - 5).astype(np.int32)
    expected = np.empty_like(host)
    for begin, end in pairwise(offsets):
        inclusive = np.cumsum(host[begin:end], dtype=np.int32)
        expected[begin:end] = np.concatenate((np.zeros(1, dtype=np.int32), inclusive[:-1]))
    observed = []

    def evaluator(graph, recipe):
        values.from_numpy(host)
        output.fill(123)
        graph.run({})
        ti.sync()
        np.testing.assert_array_equal(output.to_numpy(), expected)
        observed.append(recipe.recipe_id)
        return {
            "physical_dispatches": float(graph.physical_plan()["physical_dispatch_count"]),
            "fixture_setup": 100.0, "fixture_first": 2.0,
            "fixture_steady": 1.0 if recipe.manifest.is_baseline else 100.0,
        }

    decision = session.run(evaluator)
    report = decision.report
    assert isinstance(decision, ti.graph.GraphOptimizationOutcome)
    assert isinstance(report, ti.graph.GraphOptimizationReportV2)
    assert decision.status == "selected"
    assert decision.next_action == "apply_selection"
    assert report.search_complete
    assert report.evaluation_count == 14
    assert set(observed) == {recipe.recipe_id for recipe in session.recipes}
    assert report.missing_recipe_ids == ()
    assert report.selected_recipe_id == decision.selection.recipe_id
    assert decision.selection in decision.pareto_frontier
    assert all("materialized_memory_bytes" in result["metrics"] for result in report.results)
    assert report.compileiq_capability["schema"] == ("compileiq.taichi-forge-recipe-search-capability.v2")
    assert report.compileiq_provenance["verification"] == ("bundled_manifest_lock_at_search_start")
    assert report.compileiq_report.schema_id == ("compileiq.opaque-optimization-report.v1")
    assert report.compileiq_report.detail == "summary"
    assert report.compileiq_report.checkpoint.embedded is False
    assert report.compileiq_report.trials == ()
    assert report.selection_reason["provider_claims_used"] is False
    assert report.outcome_status == "selected"
    assert report.next_action == "apply_selection"
    assert report.recipe_annotations
    assert report.context["evaluation"] == evaluation_contract.to_dict()
    assert report.context["workload"] == workload_context.to_dict()
    assert report.context["backend"] == backend_environment.to_dict()
    assert all("fixture_setup" not in candidate.metrics for candidate in report.compileiq_report.candidates)
    selected_annotation = next(
        item for item in report.recipe_annotations if item["recipe_id"] == decision.selection.recipe_id
    )
    assert selected_annotation["measurement"]["complete"]
    assert selected_annotation["measurement"]["materialized_physical_ids"]
    # Cost observations must not silently become search objectives or filter
    # this structurally selected (but cost-negative) candidate from the report.
    assert selected_annotation["cost_profiles"]["contract_fixture"]["break_even"]["status"] == "no_positive_steady_saving"
    assert selected_annotation["cost_profiles"]["contract_fixture"]["phases"]["steady"]["median"] == 100.0
    assert selected_annotation["frozen_fragments"]
    assert all(
        claim["source"] == "provider_declared_not_measured"
        for item in report.recipe_annotations
        for claim in item["provider_claims"]
    )
    restored_report = ti.graph.GraphOptimizationReportV2.from_json(report.to_json())
    assert restored_report.to_dict() == report.to_dict()
    legacy_annotations = []
    for annotation in report.recipe_annotations:
        annotation.pop("trial_boundaries", None)
        annotation.pop("cost_profiles", None)
        annotation.pop("cost_observations", None)
        annotation.pop("frozen_fragments", None)
        annotation["measurement"].pop("materialized_memory_scope", None)
        legacy_annotations.append(annotation)
    legacy_reuse = report.reuse
    legacy_reuse.pop("context", None)
    legacy_report = replace(
        report, _recipe_annotations_json=json.dumps(legacy_annotations, sort_keys=True),
        _reuse_json=json.dumps(legacy_reuse, sort_keys=True),
    )
    restored_legacy = ti.graph.GraphOptimizationReportV2.from_json(legacy_report.to_json())
    assert restored_legacy.to_dict() == legacy_report.to_dict()
    assert "Graph resource observation boundaries" not in restored_legacy.to_markdown()
    assert restored_legacy.context is None
    assert "Caller-measured lifecycle costs" not in restored_legacy.to_markdown()
    tampered_report = report.to_dict()
    tampered_report["selection"]["reason"]["provider_claims_used"] = True
    with pytest.raises(ValueError, match="identity mismatch"):
        ti.graph.GraphOptimizationReportV2.from_dict(tampered_report)
    markdown = report.to_markdown()
    assert "Taichi Forge Graph Optimization Report" in markdown
    assert "CompileIQ measurement facts" in markdown
    assert "Provider notes above are declarations" in markdown
    assert "Caller-measured lifecycle costs" in markdown
    assert "synthetic arithmetic fixture" in markdown
    assert decision.selection_artifact is not None
    resolved = definition.resolve_recipe(decision.selection_artifact)
    assert resolved.recipe_id == decision.selection.recipe_id
    applicability = definition.check_recipe_applicability(
        decision.selection_artifact,
        workload_context=workload_context,
        evaluation_contract=evaluation_contract,
        backend_environment=backend_environment,
        target=target,
    )
    assert applicability.status == "applicable"
    drifted = definition.check_recipe_applicability(
        decision.selection_artifact,
        workload_context=ti.graph.GraphWorkloadContext({"fixture": "different-workload"}),
        evaluation_contract=evaluation_contract,
        backend_environment=backend_environment,
        target=target,
    )
    assert drifted.status == "structurally_resolvable_evidence_drift"
    assert drifted.drift_fields == ("workload_context_id",)
    json.dumps(decision.to_dict(), sort_keys=True, allow_nan=False)

    with definition.materialize(decision.selection) as materialized:
        values.from_numpy(host)
        output.fill(123)
        materialized.executor.run({})
        ti.sync()
        np.testing.assert_array_equal(output.to_numpy(), expected)
        physical = materialized.materialization_report()
        assert physical["recipe_id"] == decision.selection.recipe_id
        assert physical["planned_physical_id"] == (decision.selection.planned_physical_id)
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
            evaluation_limit=8,
            deterministic_seed=17,
        ),
        strategy=staged_strategy,
        workload_context=workload_context,
        evaluation_contract=evaluation_contract,
        backend_environment=backend_environment,
    )
    partial = partial_session.run(evaluator)
    assert partial.selection is None
    assert partial.status == "resumable"
    assert partial.next_action == "resume_search"
    assert not partial.report.search_complete
    assert partial.report.status["terminal_state"] == "budget_exhausted"
    assert partial.report.status["generation_status"] == "not_finalized"
    assert partial.report.selection_reason["rule"] == ("incomplete_evidence_no_selection")
    partial_checkpoint = partial.report.checkpoint
    compileiq_checkpoint = partial_checkpoint.compileiq_checkpoint
    assert len(compileiq_checkpoint["batches"]) == 2
    assert not compileiq_checkpoint["stages"][-1]["complete"]
    assert compileiq_checkpoint["batches"][-1]["fidelity"]["terminal"]
    survivors = set(compileiq_checkpoint["stages"][0]["survivor_recipe_ids"])
    assert all(
        set(item["parent_recipe_ids"]).issubset(survivors) for item in compileiq_checkpoint["batches"][1]["recipes"]
    )
    drifted_contract = dict(partial_checkpoint.contract)
    drifted_contract["workload_context_id"] = "graph-workload-context-v1:drift"
    contract_drift_checkpoint = ti.graph.GraphRecipeSearchCheckpointV1.create(
        contract=drifted_contract,
        generation=partial_checkpoint.generation,
        compileiq_checkpoint=compileiq_checkpoint,
    )
    with pytest.raises(ValueError, match="workload_context_id"):
        definition.search_recipes(
            engine="compileiq",
            target=target,
            budget=ti.graph.GraphSearchBudget(
                evaluation_limit=16,
                deterministic_seed=17,
            ),
            strategy=staged_strategy,
            checkpoint=contract_drift_checkpoint,
            workload_context=workload_context,
            evaluation_contract=evaluation_contract,
            backend_environment=backend_environment,
        )

    spawn_context = multiprocessing.get_context("spawn")
    result_queue = spawn_context.Queue()
    process = spawn_context.Process(
        target=_resume_segmented_scan_in_fresh_process,
        args=(
            partial_checkpoint.to_dict(),
            workload_context.to_dict(),
            evaluation_contract.to_dict(),
            backend_environment.to_dict(),
            result_queue,
        ),
    )
    process.start()
    process.join(timeout=60)
    assert not process.is_alive()
    assert process.exitcode == 0
    fresh_result = result_queue.get(timeout=5)
    assert "error" not in fresh_result, fresh_result
    assert fresh_result["search_complete"]
    assert fresh_result["selected_recipe_id"]
    assert fresh_result["evaluated_recipe_ids"]
    assert fresh_result["evaluation_count"] > partial.report.evaluation_count

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
        workload_context=workload_context,
        evaluation_contract=evaluation_contract,
        backend_environment=backend_environment,
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
        item["request"]["measurement_key"] for item in resumed.report.checkpoint.compileiq_checkpoint["records"]
    )
    assert len(measurement_keys) == len(set(measurement_keys))
    boundary_keys = tuple(
        trial["measurement_key"]
        for annotation in resumed.report.recipe_annotations
        for trial in annotation["trial_boundaries"]
    )
    assert sorted(boundary_keys) == sorted(measurement_keys)
    assert {
        trial["measurement_key"]
        for annotation in partial.report.recipe_annotations
        for trial in annotation["trial_boundaries"]
    }.issubset(boundary_keys)
    resumed_cost_keys = {
        item["measurement_key"] for annotation in resumed.report.recipe_annotations for item in annotation["cost_observations"]
    }
    assert {
        item["measurement_key"] for annotation in partial.report.recipe_annotations for item in annotation["cost_observations"]
    }.issubset(resumed_cost_keys)

    with pytest.raises(RuntimeError, match="single-use"):
        session.run(evaluator)


@test_utils.test(arch=ti.cuda, offline_cache=False)
@pytest.mark.parametrize("trace_enabled", (False, True))
def test_public_staged_search_materializes_new_survivor_compositions(monkeypatch, trace_enabled):
    lengths = np.asarray((3, 17, 53), dtype=np.int32)
    offsets = np.concatenate(
        (np.zeros(1, dtype=np.int32), np.cumsum(lengths, dtype=np.int32))
    )
    capacity = int(offsets[-1])
    layout = ti.algorithms.SegmentedLayout.from_offsets(
        offsets,
        capacity=capacity,
    )
    first_input = ti.ndarray(ti.i32, shape=capacity)
    first_output = ti.ndarray(ti.i32, shape=capacity)
    second_input = ti.ndarray(ti.i32, shape=capacity)
    second_output = ti.ndarray(ti.i32, shape=capacity)
    builder = ti.graph.GraphBuilder()
    builder.segmented_scan(first_input, layout, first_output, inclusive=False)
    builder.segmented_scan(second_input, layout, second_output, inclusive=False)
    definition = builder.freeze()

    target = ti.graph.GraphOptimizationTarget(
        objectives=(("selected_fragments", "max"),)
    )
    workload_context = ti.graph.GraphWorkloadContext(
        {
            "fixture": "two-independent-segmented-scans",
            "lengths": lengths.tolist(),
        }
    )
    evaluation_contract = ti.graph.GraphEvaluationContract(
        {
            "synchronization": "ti.sync-after-run",
            "correctness": "two-numpy-exact-oracles",
            "metric": "selected-fragment-count",
        }
    )
    backend_environment = ti.graph.GraphBackendEnvironment(
        {"fixture": "pytest-current-cuda-device"}
    )
    strategy = ti.graph.GraphRecipeSearchStrategy(
        mode="staged",
        max_generation_rounds=1,
        max_generated_recipes=14,
    )
    session = definition.search_recipes(
        target=target,
        budget=ti.graph.GraphSearchBudget(
            evaluation_limit=48,
            deterministic_seed=23,
            minimum_survivors=32,
        ),
        strategy=strategy,
        workload_context=workload_context,
        evaluation_contract=evaluation_contract,
        backend_environment=backend_environment,
    )
    initial_recipe_ids = {item.recipe_id for item in session.recipes}
    assert 2 < len(initial_recipe_ids) < strategy.max_generated_recipes

    first_host = ((np.arange(capacity) % 9) - 4).astype(np.int32)
    second_host = ((np.arange(capacity) % 7) + 2).astype(np.int32)

    def exclusive_scan(values):
        expected = np.empty_like(values)
        for begin, end in pairwise(offsets):
            inclusive = np.cumsum(values[begin:end], dtype=np.int32)
            expected[begin:end] = np.concatenate(
                (np.zeros(1, dtype=np.int32), inclusive[:-1])
            )
        return expected

    first_expected = exclusive_scan(first_host)
    second_expected = exclusive_scan(second_host)

    def evaluator(graph, recipe):
        first_input.from_numpy(first_host)
        second_input.from_numpy(second_host)
        first_output.fill(0)
        second_output.fill(0)
        graph.run({})
        ti.sync()
        np.testing.assert_array_equal(first_output.to_numpy(), first_expected)
        np.testing.assert_array_equal(second_output.to_numpy(), second_expected)
        return {"selected_fragments": float(recipe.manifest.selected_fragment_count)}

    from taichi_forge._lib import core as ti_core

    events = []
    stack = []
    native_push = ti_core._push_external_profiler_range
    native_pop = ti_core._pop_external_profiler_range

    def push(message, category, payload):
        assert trace_enabled, "ordinary search must not enter NVTX"
        events.append((message, category, payload))
        stack.append(category)
        native_push(message, category, payload)

    def pop():
        assert stack, "annotation scope must be balanced"
        stack.pop()
        native_pop()

    monkeypatch.setattr(ti_core, "_push_external_profiler_range", push)
    monkeypatch.setattr(ti_core, "_pop_external_profiler_range", pop)
    with ti.profiler.recipe_search_trace() if trace_enabled else nullcontext():
        outcome = session.run(evaluator)
    assert not stack
    assert outcome.selection is not None
    assert outcome.report.search_complete
    checkpoint = outcome.report.checkpoint.compileiq_checkpoint
    if trace_enabled:
        assert {category for _, category, _ in events} == {2, 3, 4, 5, 6}
        assert [(message, payload) for message, category, payload in events if category == 3] == [
            (batch["stage_fingerprint"], batch["stage_index"]) for batch in checkpoint["batches"]
        ]
        requests = [record["request"] for record in checkpoint["records"] if record["source"] == "objective"]
        # Checkpoints canonically sort records by identity, not execution order.
        assert sorted((message, payload) for message, category, payload in events if category == 6) == sorted(
            (
                f"measurement={request['measurement_key']} observation={request['observation_index']} "
                f"fidelity={request['fidelity_name']}",
                request["observation_index"],
            )
            for request in requests
        )
        for category in (4, 5):
            assert sorted(message for message, observed_category, _ in events if observed_category == category) == sorted(
                request["recipe_id"] for request in requests
            )
    else:
        assert not events
    assert len(checkpoint["batches"]) == 3
    stage_zero_ids = {item["recipe_id"] for item in checkpoint["batches"][0]["recipes"]}
    generated_ids = {
        item["recipe_id"] for item in checkpoint["batches"][1]["recipes"]
    } - stage_zero_ids
    assert generated_ids
    assert generated_ids.issubset({item.recipe_id for item in session.recipes})
    stage_zero_survivors = set(checkpoint["stages"][0]["survivor_recipe_ids"])
    assert all(
        set(item["parent_recipe_ids"]).issubset(stage_zero_survivors)
        for item in checkpoint["batches"][1]["recipes"]
    )
    assert all(
        item["recipe_id"] in generated_ids
        for item in checkpoint["batches"][1]["recipes"]
        if len(item["parent_recipe_ids"]) == 2
    )
    assert outcome.selection.recipe_id in {
        item["recipe_id"] for item in checkpoint["batches"][-1]["recipes"]
    }
    assert outcome.selection.manifest.selected_fragment_count == 2
    assert outcome.selection_artifact is not None
    resolved = definition.resolve_recipe(outcome.selection_artifact)
    assert resolved.recipe_id == outcome.selection.recipe_id


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

    session_only = first.search_recipes(
        budget=ti.graph.GraphSearchBudget(evaluation_limit=1)
    )
    session_only_outcome = session_only.run(
        lambda _graph, _recipe: {"device_time_ns": 1.0}
    )
    assert session_only_outcome.selection_artifact is not None
    assert (
        session_only_outcome.selection_artifact.evidence["reuse_scope"]
        == "session_only"
    )
    with pytest.raises(ValueError, match="session-only"):
        first.search_recipes(
            budget=ti.graph.GraphSearchBudget(evaluation_limit=1),
            checkpoint=session_only_outcome.report.checkpoint,
        )
