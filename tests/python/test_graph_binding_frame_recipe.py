"""Complete recipe reachability, binding publication and physical ownership."""

import gc
import json

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge._lib import core
from taichi_forge.graph._recipes.binding_frames import GraphBindingFrameRecipeProvider
from taichi_forge.graph._recipes.families import GraphRuntimeAssemblyProvider
from tests import test_utils


def _definition(size=131):
    @ti.kernel
    def stage(
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        scratch: ti.types.ndarray(dtype=ti.i32, ndim=1),
        scale: ti.i32,
    ):
        for i in source:
            scratch[i] = source[i] * scale

    @ti.kernel
    def finish(
        scratch: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in output:
            output[i] = scratch[i] + 7

    source = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "source", ti.i32, ndim=1)
    output = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1)
    scale = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "scale", ti.i32)
    builder = ti.graph.GraphBuilder()
    scratch = builder.private_ndarray("scratch", ti.i32, size)
    builder.dispatch(stage, source, scratch, scale)
    builder.dispatch(finish, scratch, output)
    return builder.freeze()


def _catalog(definition):
    if not core._CudaGraphBindingExecutor.available():
        pytest.skip("CUDA binding-frame execution is unavailable")
    return definition.recipe_catalog(providers=(GraphRuntimeAssemblyProvider(), GraphBindingFrameRecipeProvider()))


def _arrays(size=131):
    source = ti.ndarray(ti.i32, size)
    output = ti.ndarray(ti.i32, size)
    host = np.arange(size, dtype=np.int32) + 1
    source.from_numpy(host)
    output.fill(-53)
    return source, output, host


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_binding_recipe_publication_replay_update_and_failure_are_transactional():
    definition = _definition()
    catalog = _catalog(definition)
    assert len(catalog.entries()) == 2
    recipe = next(entry.recipe for entry in catalog.entries() if entry.recipe.fragments)
    assert recipe.planned_physical_id != catalog.baseline.recipe.planned_physical_id
    default = definition.recipe_catalog()
    assert any(fragment.provider_namespace.endswith(".binding_frames") for fragment in default.fragments)
    source, output, host = _arrays()
    with definition.materialization_context(provider_set=catalog.provider_set) as context:
        with context.materialize(recipe) as materialized:
            graph = materialized.executor
            native = graph._instance._backend_executable._native
            assert materialized.manifest.submissions[0].replay_mode == "cuda_immutable_argument_frames_exec_reuse"
            bindings = graph.bind(dict(source=source, output=output, scale=3))
            version = bindings._version
            assert version.execution_frame is not None
            assert "scratch" not in version.arguments
            assert "scratch" in version.execution_arguments
            np.testing.assert_array_equal(output.to_numpy(), np.full(host.size, -53, np.int32))
            before = native.snapshot()
            for _ in range(29):
                graph.run(bindings)
            np.testing.assert_array_equal(output.to_numpy(), host * 3 + 7)
            after = native.snapshot()
            assert before["preparation_upload_calls"] == after["preparation_upload_calls"]
            assert before["argument_bytes"] == after["argument_bytes"]
            with pytest.raises((ValueError, RuntimeError)):
                bindings.update(output=ti.ndarray(ti.f32, host.size))
            assert bindings._version is version
            bindings.update(scale=-5)
            assert bindings._version.execution_frame is not version.execution_frame
            graph.submit(bindings).wait()
            np.testing.assert_array_equal(output.to_numpy(), host * -5 + 7)
            del version
            gc.collect()
            native.snapshot()
            memory = graph.execution_stats().memory
            assert (
                memory.persistent_argument_bytes - memory.persistent_internal_storage_bytes
                == native.snapshot()["argument_bytes"]
                > 0
            )
            assert graph._graph_stats[0]["diagnostics_counters_complete"] is False
            candidate_id = materialized.materialized_physical_id
        assert native.snapshot()["closed"] == 1
        assert native.snapshot()["argument_bytes"] == 0
        with context.materialize(catalog.baseline.recipe) as baseline:
            assert baseline.materialized_physical_id != candidate_id
            assert baseline.manifest.submissions[0].replay_mode == "runtime_managed"


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_binding_recipe_raw_mapping_calls_prepare_without_silent_ordinary_fallback():
    definition = _definition()
    catalog = _catalog(definition)
    recipe = next(entry.recipe for entry in catalog.entries() if entry.recipe.fragments)
    source, output, host = _arrays()
    with definition.materialization_context(provider_set=catalog.provider_set) as context:
        with context.materialize(recipe) as candidate:
            graph = candidate.executor
            native = graph._instance._backend_executable._native
            for scale in (2, -1, 7):
                before = native.snapshot()["preparation_upload_calls"]
                graph.run(dict(source=source, output=output, scale=scale))
                np.testing.assert_array_equal(output.to_numpy(), host * scale + 7)
                assert native.snapshot()["preparation_upload_calls"] > before
            assert native.snapshot()["executables"] == 1
            assert native.snapshot()["frames"] == 1


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_binding_recipe_fork_search_report_and_fresh_definition_resolve():
    definition = _definition()
    catalog = _catalog(definition)
    providers = (GraphRuntimeAssemblyProvider(), GraphBindingFrameRecipeProvider())
    source, output, host = _arrays()
    # Deterministic contract selection, not a performance or acceleration claim.
    session = definition.search_recipes(
        providers=providers,
        target=ti.graph.GraphOptimizationTarget(objectives=(("prepared_binding_candidate", "max"),)),
        budget=ti.graph.GraphSearchBudget(evaluation_limit=4, repeat_count=1),
        strategy=ti.graph.GraphRecipeSearchStrategy(mode="exact_if_bounded"),
    )
    observed = set()

    def evaluate(graph, request):
        bindings = graph.bind(dict(source=source, output=output, scale=11))
        graph.run(bindings)
        np.testing.assert_array_equal(output.to_numpy(), host * 11 + 7)
        selected = bindings._version.execution_frame is not None
        observed.add(request.recipe_id)
        return {"prepared_binding_candidate": float(selected)}

    decision = session.run(evaluate)
    assert decision.status == "selected", decision.report.results
    assert decision.report.search_complete
    assert len(observed) == 2
    assert "raw mapping calls" in json.dumps(decision.report.recipe_annotations)
    fresh = _definition()
    assert fresh.semantic_graph_id == definition.semantic_graph_id
    selection = fresh.resolve_recipe(decision.selection_artifact, providers=providers)
    with fresh.materialize(selection) as materialized:
        evaluate(materialized.executor, selection)
        assert materialized.executor._instance.physical_submission_mode == "cuda_immutable_argument_frames_exec_reuse"


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_binding_recipe_scope_and_missing_capability_do_not_change_ordinary_execution(monkeypatch):
    definition = _definition()
    catalog = _catalog(definition)
    recipe = next(entry.recipe for entry in catalog.entries() if entry.recipe.fragments)
    with definition.materialization_context(provider_set=catalog.provider_set, workspace_lanes=2) as context:
        with pytest.raises((ValueError, RuntimeError), match="one workspace lane"):
            context.materialize(recipe)
    source, output, host = _arrays()

    class MissingNativeCapability:
        @staticmethod
        def available():
            return False

    monkeypatch.setattr(core, "_CudaGraphBindingExecutor", MissingNativeCapability)
    provider = GraphBindingFrameRecipeProvider()
    assert provider.fragments(definition) == ()
    ordinary = definition.compile()
    try:
        ordinary.run(dict(source=source, output=output, scale=2))
        np.testing.assert_array_equal(output.to_numpy(), host * 2 + 7)
    finally:
        ordinary._invalidate_runtime()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_binding_recipe_manifest_failure_retires_unpublished_executor(monkeypatch):
    from taichi_forge.graph._recipes import families

    definition = _definition()
    catalog = _catalog(definition)
    recipe = next(entry.recipe for entry in catalog.entries() if entry.recipe.fragments)
    source, output, host = _arrays()
    unpublished = []

    def fail_manifest(definition, recipe, graph):
        # Keep a strong reference: rollback must close resources immediately,
        # not rely on destruction when the failed materialization loses scope.
        unpublished.append(graph._instance._backend_executable._native)
        graph.bind(dict(source=source, output=output, scale=3))
        assert unpublished[-1].snapshot()["argument_bytes"] > 0
        raise ValueError("injected physical observation failure")

    with definition.materialization_context(provider_set=catalog.provider_set) as context:
        with monkeypatch.context() as patch:
            patch.setattr(families, "observe_graph_physical_manifest", fail_manifest)
            with pytest.raises((ValueError, RuntimeError), match="injected physical observation failure"):
                context.materialize(recipe)
        assert unpublished[0].snapshot()["closed"] == 1
        assert unpublished[0].snapshot()["argument_bytes"] == 0
        with context.materialize(recipe) as recovered:
            bindings = recovered.executor.bind(dict(source=source, output=output, scale=3))
            recovered.executor.run(bindings)
            np.testing.assert_array_equal(output.to_numpy(), host * 3 + 7)
