"""Compiled qualification shared by explicit candidates and complete Graphs."""

from contextlib import ExitStack

import numpy as np
import taichi_forge as ti
from taichi_forge.lang import impl

from taichi_forge.graph._recipes.dispatch_families import GraphRangeDispatchRecipeProvider
from taichi_forge.graph._recipes.families import GraphRuntimeAssemblyProvider
from taichi_forge.lang._range_dispatch import device_range_dispatch_rejection
from tests import test_utils


def definition():
    @ti.kernel
    def write(out: ti.types.ndarray(ti.i32, 1), limits: ti.types.ndarray(ti.i32, 1), n: ti.i32):
        for i in range(limits[0]):
            out[i] = i * 7 + n

    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        write,
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "limits", ti.i32, ndim=1),
        ti.graph.Arg(ti.graph.ArgKind.SCALAR, "n", ti.i32),
    )
    return builder.freeze()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_range_family_shares_code_and_preserves_mutable_bindings(tmp_path):
    graph_definition = definition()
    providers = (GraphRuntimeAssemblyProvider(), GraphRangeDispatchRecipeProvider())
    runtime = ti.lang.impl.get_runtime()
    runtime.set_kernel_executable_lifecycle_telemetry_enabled(True)
    runtime.debug_kernel_executable_lifecycle_stats(True)
    catalog = graph_definition.recipe_catalog(providers=providers)
    fragments = catalog.fragments
    if impl.current_cfg().arch == ti.cpu:
        assert not fragments
        runtime.set_kernel_executable_lifecycle_telemetry_enabled(False)
        return
    assert len(fragments) == 1
    entry = catalog.compose((fragments[0].fragment_id,), stage="range-controller")
    with graph_definition.materialization_context(providers=providers) as context:
        baseline = context.materialize(catalog.baseline.recipe)
        selected = context.materialize(entry.recipe)
        again = context.materialize(entry.recipe)
        assert again.cache_hit
        assert selected.materialized_physical_id == again.materialized_physical_id
        assert selected.materialized_physical_id != baseline.materialized_physical_id
        assert runtime.debug_kernel_executable_lifecycle_stats()["compiler_invocations"] == 0
        runtime.set_kernel_executable_lifecycle_telemetry_enabled(False)
        tasks = selected.executor.task_manifest()
        assert any(task.requested_range_dispatch == "device" for task in tasks)
        assert all(
            device_range_dispatch_rejection(task) is None
            for task in tasks
            if task.requested_range_dispatch == "device"
        )
        out = ti.ndarray(ti.i32, 257)
        limits = ti.ndarray(ti.i32, 1)
        ti.sync()
        with ti.graph.execution_domain(capacity=1) as domain:
            with selected.executor.prepare(
                dict(out=out, limits=limits, n=0), execution_domain=domain, mutable_parameters=("n",)
            ) as prepared:
                prepared.initialize()
                for size in (0, 1, 65, 257, 0):
                    out.fill(-1)
                    limits.from_numpy(np.array([size], np.int32))
                    ti.sync()
                    ticket = prepared.submit(values=dict(n=size))
                    ticket.wait()
                    ticket.close()
                    expected = np.full(257, -1, np.int32)
                    expected[:size] = np.arange(size) * 7 + size
                    np.testing.assert_array_equal(out.to_numpy(), expected)
        # The selected controller is exported without reselecting the source.
        module = ti.aot.Module(impl.current_cfg().arch)
        module.add_execution_plan("write", selected.executor)
        artifact = tmp_path / "range.tcm"
        module.archive(str(artifact))
        from taichi_forge_runtime.aot import Scalar

        with ti.aot.borrow_runtime() as owner, owner.load_module(artifact) as loaded:
            with ExitStack() as resources:
                target, count = owner.ndarray("i32", (257,)), owner.ndarray("i32", (1,))
                resources.callback(target.close)
                resources.callback(count.close)
                count.write(np.array([65], np.int32).tobytes())
                with loaded.prepare_execution("write", dict(out=target, limits=count, n=Scalar("i32", 9))) as call:
                    call()
                    np.testing.assert_array_equal(np.frombuffer(target.read(), np.int32)[:65], np.arange(65) * 7 + 9)
        again.close()
        selected.close()
        limits.from_numpy(np.array([65], np.int32))
        baseline.executor.run(dict(out=out, limits=limits, n=65))
        np.testing.assert_array_equal(out.to_numpy()[:65], np.arange(65) * 7 + 65)


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_explicit_range_search_runs_complete_graphs():
    graph_definition = definition()
    out = ti.ndarray(ti.i32, 65)
    limits = ti.ndarray(ti.i32, 1)
    limits.from_numpy(np.array([65], np.int32))
    session = graph_definition.search_recipes(
        providers=(GraphRuntimeAssemblyProvider(), GraphRangeDispatchRecipeProvider()),
        target=ti.graph.GraphOptimizationTarget(objectives=(("controller_selected", "max"),)),
        budget=ti.graph.GraphSearchBudget(evaluation_limit=2, repeat_count=1),
        strategy=ti.graph.GraphRecipeSearchStrategy(mode="exact_if_bounded"),
    )
    assert len(session.recipes) == 2
    observed = []

    def evaluate(graph, recipe):
        graph.run(dict(out=out, limits=limits, n=65))
        np.testing.assert_array_equal(out.to_numpy(), np.arange(65) * 7 + 65)
        selected = any(task.requested_range_dispatch == "device" for task in graph.task_manifest())
        observed.append(selected)
        return {"controller_selected": float(selected)}

    result = session.run(evaluate)
    assert result.status == "selected", result.report.results
    assert result.report.search_complete
    assert sorted(observed) == [False, True]
    ledger = result.report.search_cost_ledger
    assert ledger["history"]["record_count"] == 2
    assert ledger["history"]["generated_recipe_count"] == 2
    assert ledger["history"]["whole_search_host_seconds"] > 0
    assert len(ledger["trials_by_recipe"]) == 2
    assert sum(item["selected"] for item in ledger["trials_by_recipe"].values()) == 1
    restored = ti.graph.GraphOptimizationReportV2.from_json(result.report.to_json())
    assert restored.search_cost_ledger == ledger
    assert "Search cost ledger" in restored.to_markdown()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_kernel_search_does_not_confuse_host_reduction_with_device_controller():
    from taichi_forge.lang._compileiq_offload_execution_plan import CompileIQOffloadExecutionPlanSearch
    from taichi_forge.lang._offload_execution_plan import _OffloadExecutionPlan

    total = ti.field(ti.f32, shape=())

    @ti.kernel
    def sum_values(n: ti.i32):
        for i in range(n):
            total[None] += ti.cast(i, ti.f32)

    manifests = sum_values.task_manifest(65)
    reduction = next(task for task in manifests if task.thread_local_bytes)
    assert reduction.range_bounds.empty_group_elision == "compiler_tls_reduction"
    assert device_range_dispatch_rejection(reduction) is not None
    baseline = _OffloadExecutionPlan.from_task_manifests(manifests)
    recipes, eligibility = CompileIQOffloadExecutionPlanSearch._first_stage_recipes(baseline, manifests)
    assert all(not recipe.plan.device_range_tasks for recipe in recipes.values())
    status = dict(eligibility)[reduction.logical_task_id]
    assert status["thread_local"] == "manifest-proven TLS reduction"
    assert "compiler_tls_reduction" in status["range_dispatch"]
