"""Whole-Graph configuration trials retain the compiler and deployment ABI."""

import numpy as np
import pytest
import taichi_forge as ti

from taichi_forge.graph._recipes.families import GraphRuntimeAssemblyProvider
from taichi_forge.graph._recipes.kernel_configuration import (
    GraphKernelConfigurationRecipeProvider,
)
from tests import test_utils


PROVIDERS = (GraphRuntimeAssemblyProvider(), GraphKernelConfigurationRecipeProvider())


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False, compile_tier="fast")
def test_multi_task_workgroup_keeps_order_dense_lowering_and_aot(tmp_path):
    field = ti.field(ti.i32, (17, 19))

    @ti.kernel
    def step(out: ti.types.ndarray(ti.i32, 1), offset: ti.i32):
        out[0] = offset + 7
        for I in ti.grouped(field):
            field[I] = I[0] * 19 + I[1] + out[0]
        for i, j in ti.ndrange(17, 19):
            out[1 + i * 19 + j] = field[i, j] * 3
        out[0] += 2

    builder = ti.graph.GraphBuilder()
    builder.dispatch(step, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1),
                     ti.graph.Arg(ti.graph.ArgKind.SCALAR, "offset", ti.i32))
    definition = builder.freeze()
    catalog = definition.recipe_catalog(providers=PROVIDERS)
    fragment = choose(definition, catalog, lambda spec: spec.backend.workgroup_size == 64)
    entry = catalog.compose((fragment.fragment_id,), stage="multi-range")
    source = definition._runtime_spec._graph_kernel_configuration_sources[0]
    before = tuple(t.selected_block_size for t in source.facts.snapshot.manifests)
    with definition.materialization_context(providers=PROVIDERS) as context:
        selected = context.materialize(entry.recipe)
        tasks = selected.executor.task_manifest()
        ranges = tuple(t for t in tasks if t.task_type == "range_for")
        assert len(ranges) == 2 and all(t.selected_block_size == 64 for t in ranges)
        out = ti.ndarray(ti.i32, 324)
        selected.executor.run(dict(out=out, offset=5))
        expected = np.concatenate(([14], (np.arange(323) + 12) * 3)).astype(np.int32)
        np.testing.assert_array_equal(out.to_numpy(), expected)
        module = ti.aot.Module()
        module.add_execution_plan("step", selected.executor, snapshot_state=True)
        archive = tmp_path / "multi.tcm"
        module.archive(str(archive))
        with ti.aot.borrow_runtime() as runtime, runtime.load_module(archive) as loaded:
            target = runtime.ndarray("i32", (324,))
            try:
                with loaded.prepare_execution("step", dict(out=target, offset=5)) as call:
                    call()
                np.testing.assert_array_equal(np.frombuffer(target.read(), np.int32), expected)
            finally:
                target.close()
        assert tuple(t.selected_block_size for t in source.facts.snapshot.manifests) == before


def choose(definition, catalog, predicate):
    source = definition._runtime_spec._graph_kernel_configuration_sources[0]
    choice = next(
        manifest.recipe_id for manifest, spec in source.choices() if predicate(spec)
    )
    return next(
        fragment
        for fragment in catalog.fragments
        if fragment.provider_metadata["family_selection"]["choice_id"] == choice
    )


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, compile_tier="fast"
)
def test_configuration_discovery_is_lazy_and_repeated_calls_share_code():
    field = ti.field(ti.i32, shape=257)

    @ti.kernel
    def add(offset: ti.i32):
        for i in range(257):
            field[i] += i + offset

    builder = ti.graph.GraphBuilder()
    builder.dispatch(add, ti.graph.Arg(ti.graph.ArgKind.SCALAR, "first", ti.i32))
    builder.dispatch(add, ti.graph.Arg(ti.graph.ArgKind.SCALAR, "second", ti.i32))
    definition = builder.freeze()
    before = len(add._primal.compiled_kernels)
    catalog = definition.recipe_catalog(providers=PROVIDERS)
    assert len(add._primal.compiled_kernels) == before
    if ti.lang.impl.current_cfg().arch == ti.cpu:
        assert not catalog.fragments
        with definition.materialization_context(providers=PROVIDERS) as context:
            baseline = context.materialize(catalog.baseline.recipe)
            baseline.executor.run(dict(first=2, second=7))
        np.testing.assert_array_equal(field.to_numpy(), np.arange(257) * 2 + 9)
        return
    fragment = choose(
        definition, catalog, lambda spec: spec.backend.workgroup_size is not None
    )
    source = definition._runtime_spec._graph_kernel_configuration_sources[0]
    block = next(
        spec.backend.workgroup_size
        for _, spec in source.choices()
        if spec.backend.workgroup_size is not None
    )
    assert len(fragment.coverage_region_ids) == 2
    entry = catalog.compose((fragment.fragment_id,), stage="configuration")
    with definition.materialization_context(providers=PROVIDERS) as context:
        baseline = context.materialize(catalog.baseline.recipe)
        selected = context.materialize(entry.recipe)
        again = context.materialize(entry.recipe)
        assert again.cache_hit
        assert len(add._primal.compiled_kernels) == before + 1
        assert selected.materialized_physical_id != baseline.materialized_physical_id
        assert all(
            task.selected_block_size == block
            for task in selected.executor.task_manifest()
            if task.task_type == "range_for"
        )
        selected.executor.run(dict(first=2, second=7))
        np.testing.assert_array_equal(field.to_numpy(), np.arange(257) * 2 + 9)
        again.close()
        selected.close()
        field.fill(0)
        baseline.executor.run(dict(first=3, second=8))
        np.testing.assert_array_equal(field.to_numpy(), np.arange(257) * 2 + 11)


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False, compile_tier="fast")
def test_configuration_keeps_source_block_and_template_owner():
    @ti.data_oriented
    class Model:
        def __init__(self):
            self.values = ti.field(ti.i32, shape=65)

        @ti.kernel
        def run(self, factor: ti.template(), offset: ti.i32):
            ti.loop_config(block_dim=64)
            for i in range(65):
                self.values[i] = i * factor + offset

    model = Model()
    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        model.run,
        ti.graph.Arg(ti.graph.ArgKind.SCALAR, "offset", ti.i32),
        template_args={"self": model, "factor": 7},
    )
    definition = builder.freeze()
    source = definition._runtime_spec._graph_kernel_configuration_sources[0]
    assert any(
        task.source_block_size_explicit for task in source.facts.snapshot.manifests
    )
    assert all(spec.backend.workgroup_size is None for _, spec in source.choices())
    catalog = definition.recipe_catalog(providers=PROVIDERS)
    fragment = choose(definition, catalog, lambda spec: spec.ir.compile_tier == "full")
    entry = catalog.compose((fragment.fragment_id,), stage="tier")
    with definition.materialization_context(providers=PROVIDERS) as context:
        selected = context.materialize(entry.recipe)
        assert all(
            task.selected_block_size == 64
            for task in selected.executor.task_manifest()
            if task.task_type == "range_for"
        )
        selected.executor.run(dict(offset=9))
        np.testing.assert_array_equal(model.values.to_numpy(), np.arange(65) * 7 + 9)


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False, compile_tier="fast")
def test_selected_configuration_exports_primal_and_original_reverse(tmp_path):
    from tests.python.test_aot_graph_recording_export import consume

    @ti.kernel
    def seed(dx: ti.types.ndarray(ti.f32, 1), dy: ti.types.ndarray(ti.f32, 1)):
        for i in dx:
            dx[i] = 0
            dy[i] = 1

    @ti.kernel
    def square(
        x: ti.types.ndarray(ti.f32, 1, needs_grad=True),
        y: ti.types.ndarray(ti.f32, 1, needs_grad=True),
        gain: ti.f32,
    ):
        for i in range(x.shape[0]):
            y[i] = gain * x[i] * x[i]

    arguments = {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1)
        for name in ("x", "y", "dx", "dy")
    }
    gain = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "gain", ti.f32)
    builder = ti.graph.GraphBuilder()
    prefix = builder.create_sequential()
    prefix.dispatch(seed, arguments["dx"], arguments["dy"])
    builder.append(prefix)
    builder.dispatch(square, arguments["x"], arguments["y"], gain)
    builder.dispatch(square.grad, arguments["x"], arguments["y"], gain)
    definition = builder.freeze()
    assert len(definition._runtime_spec._graph_kernel_configuration_sources) == 1
    catalog = definition.recipe_catalog(providers=PROVIDERS)
    fragment = choose(definition, catalog, lambda spec: spec.ir.compile_tier == "full")
    entry = catalog.compose((fragment.fragment_id,), stage="primal-tier")
    with definition.materialization_context(providers=PROVIDERS) as context:
        selected = context.materialize(entry.recipe)
        module = ti.aot.Module()
        module.add_execution_plan(
            "step", selected.executor, gradients=dict(x="dx", y="dy")
        )
        module.add_execution_plan(
            "again", selected.executor, gradients=dict(x="dx", y="dy")
        )
        assert len(module._exported_kernels) == 3
        arch = "cuda" if ti.lang.impl.current_cfg().arch == ti.cuda else "vulkan"
        consume(tmp_path, module, "square", arch)


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False, compile_tier="fast")
def test_configuration_search_resumes_and_retries_failed_materialization(monkeypatch):
    @ti.kernel
    def write(out: ti.types.ndarray(ti.i32, 1)):
        for i in range(out.shape[0]):
            out[i] = i * 3 + 2

    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        write, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1)
    )
    definition = builder.freeze()
    catalog = definition.recipe_catalog(providers=PROVIDERS)
    fragment = choose(definition, catalog, lambda spec: spec.ir.compile_tier == "full")
    entry = catalog.compose((fragment.fragment_id,), stage="retry")
    source = definition._runtime_spec._graph_kernel_configuration_sources[0]
    materialize = source.materialize
    attempts = []

    def transient(choice):
        attempts.append(choice)
        if len(attempts) == 1:
            raise RuntimeError("transient trial failure")
        return materialize(choice)

    monkeypatch.setattr(source, "materialize", transient)
    with definition.materialization_context(providers=PROVIDERS) as context:
        with pytest.raises(Exception, match="transient trial failure"):
            context.materialize(entry.recipe)
        retried = context.materialize(entry.recipe)
        assert len(attempts) == 2
        assert not retried.cache_hit
    monkeypatch.setattr(source, "materialize", materialize)
    out = ti.ndarray(ti.i32, 65)
    options = dict(
        providers=PROVIDERS,
        strategy=ti.graph.GraphRecipeSearchStrategy(mode="exact_if_bounded"),
        target=ti.graph.GraphOptimizationTarget(objectives=(("score", "min"),)),
        workload_context=ti.graph.GraphWorkloadContext(
            {"fixture": "configuration", "size": 65}
        ),
        evaluation_contract=ti.graph.GraphEvaluationContract(
            {"oracle": "exact integer affine", "score": "synthetic"}
        ),
        backend_environment=ti.graph.GraphBackendEnvironment(
            {"fixture": "current-test-device"}
        ),
    )
    observations = []

    def evaluate(graph, recipe):
        graph.run(dict(out=out))
        np.testing.assert_array_equal(out.to_numpy(), np.arange(65) * 3 + 2)
        observations.append(recipe.recipe_id)
        return {"score": float(recipe.manifest.is_baseline)}

    partial = definition.search_recipes(
        budget=ti.graph.GraphSearchBudget(2), **options
    ).run(evaluate)
    assert not partial.report.search_complete
    saved = ti.graph.GraphRecipeSearchCheckpointV1.from_dict(
        partial.checkpoint.to_dict()
    )
    completed = definition.search_recipes(
        budget=ti.graph.GraphSearchBudget(64),
        checkpoint=saved,
        **options,
    ).run(evaluate)
    assert completed.status == "selected"
    assert completed.report.search_complete
    assert not completed.selection.manifest.is_baseline
    keys = [
        item["request"]["measurement_key"]
        for item in completed.checkpoint.compileiq_checkpoint["records"]
    ]
    assert len(keys) == len(set(keys)) == len(observations)
    with definition.materialize(completed.selection, providers=PROVIDERS) as selected:
        evaluate(selected.executor, completed.selection)


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False, compile_tier="fast")
def test_configuration_neighbors_combine_controls_and_restore_without_history():
    @ti.kernel
    def write(out: ti.types.ndarray(ti.i32, 1)):
        for i in range(out.shape[0]):
            out[i] = 2 * i + 5

    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        write, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1)
    )
    definition = builder.freeze()
    catalog = definition.recipe_catalog(providers=PROVIDERS)
    tier = choose(definition, catalog, lambda spec: spec.ir.compile_tier == "full")
    tier_entry = catalog.compose((tier.fragment_id,), stage="tier")
    source = definition._runtime_spec._graph_kernel_configuration_sources[0]
    before = len(write._primal.compiled_kernels)
    generated = catalog.build_survivor_stage(
        (tier_entry.recipe.recipe_id,),
        seed_fragment_ids=tuple(item.fragment_id for item in catalog.fragments),
        candidate_limit=32,
    )
    selected = None
    for entry in generated:
        if len(entry.recipe.fragments) == 1:
            fragment = entry.recipe.fragments[0]
            choice = fragment.provider_metadata["family_selection"]["choice_id"]
            _, spec = source.resolve(choice)
            if (
                spec.ir.compile_tier == "full"
                and spec.backend.workgroup_size is not None
            ):
                selected = entry
                break
    assert selected is not None
    assert len(write._primal.compiled_kernels) == before
    fragment = selected.recipe.fragments[0]
    # Fresh discovery and key resolution, without retaining generated variants.
    source._choices = None
    fresh = GraphKernelConfigurationRecipeProvider()
    restored = fresh.resolve(definition, fragment.fragment_key)
    assert restored.to_dict() == fragment.to_dict()
    with pytest.raises(ValueError, match="unavailable"):
        source.resolve(
            fragment.provider_metadata["family_selection"]["choice_id"] + "0"
        )
    out = ti.ndarray(ti.i32, 65)
    with definition.materialization_context(providers=PROVIDERS) as context:
        materialized = context.materialize(selected.recipe)
        materialized.executor.run(dict(out=out))
        np.testing.assert_array_equal(out.to_numpy(), np.arange(65) * 2 + 5)
        assert len(write._primal.compiled_kernels) == before + 1


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False, compile_tier="fast")
def test_configuration_rejects_recaptured_host_constants():
    config = {"value": 3}
    result = ti.field(ti.i32, shape=65)

    @ti.kernel
    def write():
        for i in range(65):
            result[i] = config["value"]

    builder = ti.graph.GraphBuilder()
    builder.dispatch(write)
    definition = builder.freeze()
    catalog = definition.recipe_catalog(providers=PROVIDERS)
    fragment = choose(definition, catalog, lambda spec: spec.ir.compile_tier == "full")
    entry = catalog.compose((fragment.fragment_id,), stage="source-check")
    config["value"] = 7
    with definition.materialization_context(providers=PROVIDERS) as context:
        with pytest.raises(Exception, match="source changed after Graph freeze"):
            context.materialize(entry.recipe)
        baseline = context.materialize(catalog.baseline.recipe)
        baseline.executor.run({})
        np.testing.assert_array_equal(result.to_numpy(), np.full(65, 3))
