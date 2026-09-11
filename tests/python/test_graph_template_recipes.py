import numpy as np
import pytest

import taichi_forge as ti

from tests import test_utils


def _family(catalog, name):
    return tuple(f for f in catalog.fragments if f.provider_namespace == f"taichi_forge.graph.{name}")


def _compose(catalog, fragment):
    return catalog.compose(
        (fragment.fragment_id,),
        stage="template-specialization",
        parent_recipe_ids=(catalog.baseline.recipe.recipe_id,),
    ).recipe


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_template_field_offload_recipe_preserves_owner_and_live_contents():
    count = 4099

    @ti.data_oriented
    class Work:
        def __init__(self):
            self.count = count
            self.data = ti.field(ti.i32, shape=count)
            self.output = ti.field(ti.i32, shape=count)

        @ti.kernel
        def phases(self, scale: ti.template()):
            for i in range(self.count):
                self.output[i] = self.data[i] * scale
            for i in range(self.count):
                self.output[i] = self.output[i] * 3 + 7

    work = Work()
    template_args = {"self": work, "scale": 2}
    builder = ti.graph.GraphBuilder()
    builder.dispatch(work.phases, template_args=template_args)
    definition = builder.freeze()
    # Mutating the caller's argument mapping must not retarget a lazy recipe.
    template_args["scale"] = 99
    catalog = definition.recipe_catalog()
    fragments = _family(catalog, "offload_phase_fusion")
    assert fragments
    session = definition.search_recipes(
        target=ti.graph.GraphOptimizationTarget(objectives=(("physical_tasks", "min"),)),
        budget=ti.graph.GraphSearchBudget(evaluation_limit=8),
    )
    assert any("offload_phase_fusion" in r.manifest.families for r in session.recipes)
    with definition.materialization_context() as context:
        baseline = context.materialize(catalog.baseline.recipe)
        fused = context.materialize(_compose(catalog, fragments[0]))
        assert len(baseline.executor.task_manifest()) == 2
        assert len(fused.executor.task_manifest()) == 1
        for offset in (0, 19):
            host = np.arange(count, dtype=np.int32) + offset
            work.data.from_numpy(host)
            bindings = fused.executor.bind({})
            before = fused.executor.binding_statistics()
            for _ in range(3):
                fused.executor.run(bindings)
            np.testing.assert_array_equal(work.output.to_numpy(), host * 6 + 7)
            after = fused.executor.binding_statistics()
            assert after["raw_replay_validations"] == before["raw_replay_validations"]

    def evaluate(graph, _recipe):
        graph.run(graph.bind({}))
        np.testing.assert_array_equal(work.output.to_numpy(), (np.arange(count, dtype=np.int32) + 19) * 6 + 7)
        return {"physical_tasks": float(len(graph.task_manifest()))}

    decision = session.run(evaluate)
    assert decision.status == "selected"
    assert "offload_phase_fusion" in decision.selection.manifest.families
    for annotation in decision.report.recipe_annotations:
        for trial in annotation["trial_boundaries"]:
            route = trial["execution_after_evaluator"]
            assert route["status"] == "snapshot"
            assert route["backend_graph_segments"] == 1
            assert route["ordinary_fallback_segments"] == 0
            assert route["native_node_count"] == 0
    assert "Execution boundary snapshots" in decision.report.to_markdown()
    rebuilt = ti.graph.GraphBuilder()
    rebuilt.dispatch(work.phases, template_args={"self": work, "scale": 2})
    resolved = rebuilt.freeze().resolve_recipe(decision.selection_artifact)
    assert resolved.recipe_id == decision.selection.recipe_id


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_template_ndarray_memory_recipe_keeps_specialization_and_binding_checks():
    count = 1031

    @ti.data_oriented
    class Stencil:
        def __init__(self):
            self.count = count

        @ti.kernel
        def apply(
            self,
            source: ti.types.ndarray(dtype=ti.f32, ndim=1),
            output: ti.types.ndarray(dtype=ti.f32, ndim=1),
            scale: ti.template(),
        ):
            for i in range(1, self.count - 1):
                output[i] = source[i - 1] + source[i] * scale + source[i + 1]

    work = Stencil()
    source_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "source", ti.f32, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(work.apply, source_arg, output_arg, template_args={"self": work, "scale": 2.0})
    definition = builder.freeze()
    catalog = definition.recipe_catalog()
    fragments = _family(catalog, "graph_memory")
    assert fragments
    source = ti.ndarray(ti.f32, shape=count)
    output = ti.ndarray(ti.f32, shape=count)
    host = np.arange(count, dtype=np.float32)
    source.from_numpy(host)
    output.fill(-1)
    with definition.materialization_context() as context:
        product = context.materialize(_compose(catalog, fragments[0]))
        product.executor.run(product.executor.bind({"source": source, "output": output}))
        observed = output.to_numpy()
        np.testing.assert_array_equal(observed[1:-1], host[:-2] + host[1:-1] * 2 + host[2:])
        np.testing.assert_array_equal(observed[[0, -1]], [-1, -1])
        with pytest.raises((ValueError, RuntimeError), match="(?i)(alias|overlap|disjoint)"):
            product.executor.bind({"source": source, "output": source})


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_template_mixed_field_external_offload_requires_alias_proof():
    count = 1027
    data = ti.field(ti.i32, shape=count)

    @ti.kernel
    def phases(field: ti.template(), output: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(count):
            field[i] = i + 1
        for i in range(count):
            output[i] = field[i] * 3

    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(phases, output_arg, template_args={"field": data})
    definition = builder.freeze()
    catalog = definition.recipe_catalog()
    assert not _family(catalog, "offload_phase_fusion")
    source = definition._runtime_spec._graph_offload_fusion_sources[0]
    assert "mixed field and external memory requires a Graph alias contract" in source.candidate_failure
    observations = catalog.discovery_report()["providers"]
    explanation = next(
        item["provider_explanation"]
        for item in observations
        if item["provider_namespace"] == "taichi_forge.graph.offload_phase_fusion"
    )
    assert explanation["status"] == "sources_inspected"
    assert explanation["sources"][0]["status"] == "candidate_generation_rejected"
    assert "Graph alias contract" in explanation["sources"][0]["generation_rejections"][0]["reason"]
    # Reporting must read the frozen observation, not call the lazy compiler.
    source.candidates = lambda: pytest.fail("discovery report recompiled a source")
    assert catalog.discovery_report()["providers"] == observations
    # Lack of a fusion proof does not reject or modify the ordinary baseline.
    with definition.materialization_context() as context:
        baseline = context.materialize(catalog.baseline.recipe)
        output = ti.ndarray(ti.i32, shape=count)
        baseline.executor.run(baseline.executor.bind({"output": output}))
        np.testing.assert_array_equal(output.to_numpy(), (np.arange(count, dtype=np.int32) + 1) * 3)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_template_recipe_rejects_changed_compile_time_owner_semantics():
    @ti.data_oriented
    class Work:
        def __init__(self):
            self.scale = 2
            self.data = ti.field(ti.i32, shape=1024)

        @ti.kernel
        def phases(self):
            for i in range(1024):
                self.data[i] = i * self.scale
            for i in range(1024):
                self.data[i] = self.data[i] + 7

    work = Work()
    builder = ti.graph.GraphBuilder()
    builder.dispatch(work.phases, template_args={"self": work})
    definition = builder.freeze()
    work.scale = 99
    assert not _family(definition.recipe_catalog(), "offload_phase_fusion")
    source = definition._runtime_spec._graph_offload_fusion_sources[0]
    assert "materialized offload topology does not match the plan" in source.candidate_failure
    with definition.materialization_context() as context:
        baseline = context.materialize(definition.baseline_recipe)
        baseline.executor.run(baseline.executor.bind({}))
        np.testing.assert_array_equal(work.data.to_numpy(), np.arange(1024, dtype=np.int32) * 2 + 7)
