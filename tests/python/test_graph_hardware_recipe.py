"""Hardware recordings retain their place and resources in whole-Graph recipes."""

import numpy as np
import pytest
import taichi_forge as ti

from taichi_forge.graph._recipes.families import GraphRuntimeAssemblyProvider
from taichi_forge.graph._recipes.map_fusion import GraphMapFusionRecipeProvider
from tests import test_utils


@test_utils.test(arch=ti.cuda, offline_cache=False)
@pytest.mark.parametrize("native_first", (True, False))
def test_captured_fft_survives_baseline_and_fused_recipe_reconstruction(native_first):
    if not ti.hardware.fft.is_available():
        pytest.skip("a compatible optional cuFFT library is unavailable")
    length = 64

    @ti.kernel
    def scale(source: ti.types.ndarray(dtype=ti.f32, ndim=1), output: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for i in range(length * 2):
            output[i] = source[i] * 2.0

    @ti.kernel
    def shift(source: ti.types.ndarray(dtype=ti.f32, ndim=1), output: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for i in range(length * 2):
            output[i] = source[i] + 0.25

    layout = ti.hardware.fft.CufftLayout(embed=(8, 8))
    plan = ti.hardware.fft.CufftPlanND((8, 8), input_layout=layout, output_layout=layout)
    symbols = [
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1) for name in ("input", "first", "second", "output")
    ]
    builder = ti.graph.GraphBuilder()
    if native_first:
        builder.append_native(plan.record(input="input", output="first"), admission="auto")
        builder.dispatch(scale, symbols[1], symbols[2])
        builder.dispatch(shift, symbols[2], symbols[3])
    else:
        builder.dispatch(scale, symbols[0], symbols[1])
        builder.dispatch(shift, symbols[1], symbols[2])
        builder.append_native(plan.record(input="second", output="output"), admission="auto")
    definition = builder.freeze()
    catalog = definition.recipe_catalog(providers=(GraphRuntimeAssemblyProvider(), GraphMapFusionRecipeProvider()))
    assert len(catalog.entries()) == 2, "baseline and an adjacent two-map fusion must both be reachable"
    arrays = {symbol.name: ti.ndarray(ti.f32, shape=length * 2) for symbol in symbols}
    host = np.linspace(-1, 2, length * 2, dtype=np.float32)
    fft_input = host if native_first else host * 2 + 0.25
    reference = np.fft.fft2((fft_input[0::2] + 1j * fft_input[1::2]).reshape(8, 8)).reshape(-1)
    expected = np.stack((reference.real, reference.imag), axis=1).reshape(-1)
    if native_first:
        expected = expected * 2 + 0.25
    identities = set()
    try:
        with definition.materialization_context(provider_set=catalog.provider_set) as context:
            for entry in catalog.entries():
                arrays["input"].from_numpy(host)
                for name in ("first", "second", "output"):
                    arrays[name].fill(-123)
                with context.materialize(entry.recipe) as materialized:
                    graph = materialized.executor
                    assert graph._debug_info["native_count"] == 1
                    manifest = materialized.manifest
                    assert not manifest.task_topology_exact
                    native_tasks = [task for task in manifest.tasks if task.kind == "native_action"]
                    assert len(native_tasks) == 1
                    assert manifest.tasks[0 if native_first else -1] is native_tasks[0]
                    assert native_tasks[0].kernel_indices == ()
                    bindings = graph.bind(arrays)
                    for _ in range(3):
                        graph.run(bindings)
                    np.testing.assert_allclose(arrays["output"].to_numpy(), expected, rtol=3e-5, atol=3e-5)
                    identities.add(materialized.materialized_physical_id)
            assert len(identities) == 2
    finally:
        plan.close()


def _hardware_recipe_providers(binding_frames):
    # Strategy/lifetime fixtures isolate their provider domain. Dedicated mixed
    # frame tests below use the full default set, including submission recipes.
    return tuple(
        provider
        for provider in ti.graph.default_recipe_providers()
        if binding_frames or not provider.descriptor.namespace.endswith(".binding_frames")
    )


def _spmm_definition(*, tolerance=2e-5, prepare=True, component_version=None, binding_frames=False):
    rows, columns, rhs_count = 7, 5, 8
    dense = np.zeros((rows, columns), np.float32)
    offsets, indices, values = [0], [], []
    for row in range(rows):
        for column in sorted({row % columns, (row + 2) % columns}):
            value = (row + 1) * 0.125 + column * 0.0625
            indices.append(column)
            values.append(value)
            dense[row, column] = value
        offsets.append(len(indices))
    devices = []
    for host, dtype in ((offsets, ti.i32), (indices, ti.i32), (values, ti.f32)):
        array = ti.ndarray(dtype, len(host))
        array.from_numpy(np.asarray(host, np.float32 if dtype == ti.f32 else np.int32))
        devices.append(array)
    matrix = ti.linalg.SparsePattern.csr(rows, columns, devices[0], devices[1]).matrix(devices[2])
    host = np.arange(columns * rhs_count, dtype=np.float32).reshape(columns, rhs_count) / 16 - 0.5
    bindings = {
        "input": ti.ndarray(ti.f32, (columns, rhs_count)),
        "product": ti.ndarray(ti.f32, (rows, rhs_count)),
        "output": ti.ndarray(ti.f32, (rows, rhs_count)),
    }
    bindings["input"].from_numpy(host)
    bindings["product"].fill(-123)
    operation = matrix.record_spmm(
        rhs_count,
        output="product",
        absolute_tolerance=tolerance,
        relative_tolerance=tolerance,
    )
    if component_version is not None:
        import json

        # Simulate a different installed vendor release without changing math.
        operation._component_json = json.dumps({**operation.component, "version": component_version})
    if prepare:
        before = matrix._debug_runtime_stats()
        prepared = operation.prepare(bindings["input"], bindings["product"])
        after = matrix._debug_runtime_stats()
        assert prepared["row_streamed"]["prepared"]
        assert prepared["tree_direct"]["preprocessed"] is False
        assert before["operations"]["spmm_calls"] == after["operations"]["spmm_calls"] == 0
        np.testing.assert_array_equal(bindings["product"].to_numpy(), -123)
        np.testing.assert_array_equal(bindings["input"].to_numpy(), host)

    @ti.kernel
    def finish(product: ti.types.ndarray(dtype=ti.f32, ndim=2), output: ti.types.ndarray(dtype=ti.f32, ndim=2)):
        for i, j in ti.ndrange(rows, rhs_count):
            output[i, j] = product[i, j] * 2.0 + 0.25

    builder = ti.graph.GraphBuilder()
    builder.append_native(operation, admission="auto")
    builder.dispatch(
        finish, *(ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=2) for name in ("product", "output"))
    )
    providers = (*_hardware_recipe_providers(binding_frames), ti.hardware.linalg.SparseSpmmRecipeProvider())
    return builder.freeze(), operation, providers, bindings, dense @ host * 2 + 0.25


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_spmm_complete_recipes_search_report_and_fresh_process_resolution(tmp_path):
    import json
    import subprocess
    import sys

    if not ti.hardware.linalg.cusparse_spmm_is_available():
        pytest.skip("the optional cuSPARSE SpMM runtime is unavailable")
    definition, operation, providers, bindings, expected = _spmm_definition()
    session = definition.search_recipes(
        providers=providers,
        target=ti.graph.GraphOptimizationTarget(objectives=(("preprocessed", "max"),)),
        budget=ti.graph.GraphSearchBudget(evaluation_limit=6, repeat_count=1),
        strategy=ti.graph.GraphRecipeSearchStrategy(mode="exact_if_bounded"),
    )
    # This structural objective exercises selection/resolve, not an acceleration claim.
    assert len(session.recipes) == len(operation.preparation_report())
    observed = set()

    def evaluate(graph, recipe):
        bound = graph.bind(bindings)
        for _ in range(3):
            graph.run(bound)
        np.testing.assert_allclose(bindings["output"].to_numpy(), expected, rtol=2e-5, atol=2e-5)
        assert graph._graph_stats[0]["last_path"] == "cuda_exact_replay", graph._graph_stats
        plans = [lease for lease in graph._spec.lifetime_leases if hasattr(lease, "_graph_spmm_source")]
        assert len(plans) == 1
        observed.add(plans[0]._graph_physical_plan_id)
        return {"preprocessed": float(plans[0].plan_info()["preprocessed"])}

    decision = session.run(evaluate)
    assert decision.status == "selected", decision.report.results
    assert decision.report.search_complete
    assert len(observed) == len(session.recipes)
    restored = ti.graph.GraphOptimizationReportV2.from_json(decision.report.to_json())
    assert restored.to_dict() == decision.report.to_dict()
    annotations = json.dumps(decision.report.recipe_annotations)
    assert "finite_inputs_only" in annotations
    assert "frozen_config" in annotations
    frozen = [fragment for item in decision.report.recipe_annotations for fragment in item["frozen_fragments"]]
    spmm_tasks = [
        task["physical"] for fragment in frozen if fragment["provider_namespace"].endswith(".sparse_spmm")
        for task in fragment["physical_tasks"]
    ]
    assert spmm_tasks
    assert all(task["semantic_contract"]["numerical_contract"] == operation.semantics["numerical_contract"] for task in spmm_tasks)
    assert all(task["component"] == operation.component for task in spmm_tasks)
    markdown = decision.report.to_markdown()
    assert '"bitwise_reproducibility": false' in markdown
    assert '"provider": "cusparse"' in markdown
    assert '"host_setup_seconds":' in markdown
    assert '"measurement_scope": "host_elapsed_for_spmm_preparation_or_cache_reuse"' in markdown
    assert '"selected_only_restore": "not_measured"' in markdown
    assert all(not annotation["cost_profiles"] for annotation in decision.report.recipe_annotations)
    assert restored.to_markdown() == markdown
    resolved = definition.resolve_recipe(decision.selection_artifact, providers=providers)
    with definition.materialize(resolved) as materialized:
        evaluate(materialized.executor, resolved)
        before_update = operation.matrix._debug_runtime_stats()["operations"]["spmm_plan_builds"]
        updated = ti.ndarray(ti.f32, operation.semantics["nonzeros"])
        updated.fill(0)
        operation.matrix.update_values(updated)
        materialized.executor.run(materialized.executor.bind(bindings))
        np.testing.assert_array_equal(bindings["output"].to_numpy(), 0.25)
        assert operation.matrix._debug_runtime_stats()["operations"]["spmm_plan_builds"] == before_update

    # The new definition is rebuilt from normal Python code, never unpickled.
    artifact = tmp_path / "selection.json"
    artifact.write_text(json.dumps(decision.selection_artifact.to_dict()), encoding="utf-8")
    child = """
import json, sys
from pathlib import Path
import numpy as np
import taichi_forge as ti
from tests.python.test_graph_hardware_recipe import _spmm_definition
ti.init(arch=ti.cuda, offline_cache=False)
definition, operation, providers, bindings, expected = _spmm_definition()
artifact = ti.graph.GraphRecipeSelectionArtifact.from_dict(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')))
resolved = definition.resolve_recipe(artifact, providers=providers)
with definition.materialize(resolved) as result:
    result.executor.run(result.executor.bind(bindings))
    np.testing.assert_allclose(bindings['output'].to_numpy(), expected, rtol=2e-5, atol=2e-5)
print('SPMM_RESOLVED:' + resolved.recipe_id)
ti.reset()
"""
    child_result = subprocess.run(
        [sys.executable, "-c", child, str(artifact)], capture_output=True, text=True, timeout=90
    )
    assert child_result.returncode == 0, child_result.stdout + child_result.stderr
    assert "SPMM_RESOLVED:" + resolved.recipe_id in child_result.stdout

    changed, _, changed_providers, _, _ = _spmm_definition(component_version=-1)
    assert changed.semantic_graph_id == definition.semantic_graph_id
    assert changed.baseline_recipe.recipe_id != definition.baseline_recipe.recipe_id
    with pytest.raises(ti.graph.GraphRecipeReuseError):
        changed.resolve_recipe(decision.selection_artifact, providers=changed_providers)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_spmm_semantic_drift_and_explicit_preparation():
    if not ti.hardware.linalg.cusparse_spmm_is_available():
        pytest.skip("the optional cuSPARSE SpMM runtime is unavailable")
    definition, operation, providers, bindings, _ = _spmm_definition(prepare=False)
    with pytest.raises(ti.graph.GraphRecipeProviderError) as error:
        definition.recipe_catalog(providers=providers)
    assert error.value.error_key == "spmm_plan_preparation_required"
    with pytest.raises(AttributeError):
        operation.rhs_count = 16
    operation.prepare(bindings["input"], bindings["product"])
    assert len(definition.recipe_catalog(providers=providers).entries()) >= 2
    other, _, _, _, _ = _spmm_definition(tolerance=3e-5, prepare=False)
    assert definition.semantic_graph_id != other.semantic_graph_id
    for tolerance in (True, float("nan"), -1.0):
        with pytest.raises((TypeError, ValueError)):
            operation.matrix.record_spmm(8, absolute_tolerance=tolerance, relative_tolerance=2e-5)


def _fft_definition(*, prepare=True, direction="forward", preparation=None, binding_frames=False):
    dimensions, batch = (24, 40), 3
    source = ti.linalg.record_fft(
        dimensions,
        batch_count=batch,
        direction=direction,
        output="transform",
        absolute_tolerance=2e-4,
        relative_tolerance=2e-4,
        preparation=preparation,
    )
    if prepare:
        source.prepare()
    shape = (batch, *dimensions, 2)
    bindings = {
        name: ti.ndarray(ti.f32, shape)
        for name in ("input", "transform", "middle", "output")
    }
    host = np.random.default_rng(49).uniform(-1, 1, shape).astype(np.float32)
    bindings["input"].from_numpy(host)
    complex_host = host[..., 0] + 1j * host[..., 1]
    expected = (
        np.fft.fft2(complex_host)
        if direction == "forward"
        else np.fft.ifft2(complex_host) * np.prod(dimensions)
    )
    expected = np.stack((expected.real, expected.imag), axis=-1) * 2 + 0.125

    @ti.kernel
    def scale(
        source: ti.types.ndarray(dtype=ti.f32, ndim=4),
        output: ti.types.ndarray(dtype=ti.f32, ndim=4),
    ):
        for b, i, j, c in ti.ndrange(batch, dimensions[0], dimensions[1], 2):
            output[b, i, j, c] = source[b, i, j, c] * 2

    @ti.kernel
    def shift(
        source: ti.types.ndarray(dtype=ti.f32, ndim=4),
        output: ti.types.ndarray(dtype=ti.f32, ndim=4),
    ):
        for b, i, j, c in ti.ndrange(batch, dimensions[0], dimensions[1], 2):
            output[b, i, j, c] = source[b, i, j, c] + 0.125

    symbols = {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=4)
        for name in bindings
    }
    builder = ti.graph.GraphBuilder()
    builder.append_native(source, admission="auto")
    builder.dispatch(scale, symbols["transform"], symbols["middle"])
    builder.dispatch(shift, symbols["middle"], symbols["output"])
    providers = (
        *_hardware_recipe_providers(binding_frames),
        ti.hardware.fft.FftRecipeProvider(),
    )
    return builder.freeze(), source, providers, bindings, expected


def _mixed_hardware_definition(*, binding_frames=False):
    _, spmm, _, sparse_bindings, sparse_expected = _spmm_definition()
    operation = ti.linalg.record_fft(
        (12, 20),
        input="fft_input",
        output="fft_output",
        absolute_tolerance=2e-4,
        relative_tolerance=2e-4,
    )
    operation.prepare()
    host = np.random.default_rng(93).uniform(-1, 1, (12, 20, 2)).astype(np.float32)
    bindings = {name: sparse_bindings[name] for name in ("input", "product")}
    for name in ("fft_input", "fft_output"):
        bindings[name] = ti.ndarray(ti.f32, host.shape)
    bindings["fft_input"].from_numpy(host)
    reference = np.fft.fft2(host[..., 0] + 1j * host[..., 1])
    expected = {
        "product": (sparse_expected - 0.25) / 2,
        "fft_output": np.stack((reference.real, reference.imag), axis=-1),
    }
    builder = ti.graph.GraphBuilder()
    builder.append_native(spmm, admission="auto")
    builder.append_native(operation, admission="auto")
    providers = (
        *_hardware_recipe_providers(binding_frames),
        ti.hardware.linalg.SparseSpmmRecipeProvider(),
        ti.hardware.fft.FftRecipeProvider(),
    )
    return builder.freeze(), operation, spmm, providers, bindings, expected


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_fft_and_spmm_complete_recipes_compose_search_and_resolve_in_fresh_process(
    tmp_path,
):
    import json
    import subprocess
    import sys

    if (
        not ti.hardware.fft.is_available()
        or not ti.hardware.linalg.cusparse_spmm_is_available()
    ):
        pytest.skip("the optional cuFFT/cuSPARSE runtime is unavailable")
    definition, operation, spmm, providers, bindings, expected = (
        _mixed_hardware_definition()
    )
    observed = set()

    def evaluate(graph, recipe):
        bound = graph.bind(bindings)
        for _ in range(3):
            graph.run(bound)
        for name, reference in expected.items():
            np.testing.assert_allclose(
                bindings[name].to_numpy(), reference, rtol=2e-4, atol=2e-4
            )
        assert (
            graph._graph_stats[0]["last_path"] == "cuda_exact_replay"
        ), graph._graph_stats
        record = next(
            lease
            for lease in graph._spec.lifetime_leases
            if hasattr(lease, "_graph_fft_source")
        )
        sparse_record = next(
            lease
            for lease in graph._spec.lifetime_leases
            if hasattr(lease, "_graph_spmm_source")
        )
        observed.add((record.plan._separable, sparse_record.algorithm))
        # A deterministic structural objective, not a performance claim.
        return {
            "phases": float(record.plan._separable)
            + float(sparse_record.plan_info()["preprocessed"])
        }

    session = definition.search_recipes(
        providers=providers,
        target=ti.graph.GraphOptimizationTarget(objectives=(("phases", "max"),)),
        budget=ti.graph.GraphSearchBudget(evaluation_limit=8, repeat_count=1),
        strategy=ti.graph.GraphRecipeSearchStrategy(mode="exact_if_bounded"),
    )
    expected_count = 2 * len(spmm.preparation_report())
    assert (
        len(session.recipes) == expected_count
    ), "FFT and SpMM strategies must compose across native regions"
    decision = session.run(evaluate)
    assert (
        decision.status == "selected" and decision.report.search_complete
    ), decision.report.results
    assert len(observed) == expected_count
    assert (
        len({result["materialized_physical_id"] for result in decision.report.results})
        == expected_count
    )
    assert "columns_in_place_per_batch" in json.dumps(
        decision.report.recipe_annotations
    )
    markdown = decision.report.to_markdown()
    assert '"measurement_scope": "host_elapsed_for_fft_plan_creation"' in markdown
    assert '"measurement_scope": "host_elapsed_for_spmm_preparation_or_cache_reuse"' in markdown
    assert '"shared_initialization": "not_separated"' in markdown
    artifact = tmp_path / "fft.json"
    artifact.write_text(
        json.dumps(decision.selection_artifact.to_dict()), encoding="utf-8"
    )
    child = """
import json,sys
from pathlib import Path
import numpy as np
import taichi_forge as ti
from tests.python.test_graph_hardware_recipe import _mixed_hardware_definition
ti.init(arch=ti.cuda,offline_cache=False)
definition,operation,spmm,providers,bindings,expected = _mixed_hardware_definition()
artifact = ti.graph.GraphRecipeSelectionArtifact.from_dict(json.loads(Path(sys.argv[1]).read_text()))
handle = definition.resolve_recipe(artifact,providers=providers)
with definition.materialize(handle) as result:
    bound=result.executor.bind(bindings)
    for _ in range(3):
        result.executor.run(bound)
    for name,reference in expected.items():
        np.testing.assert_allclose(bindings[name].to_numpy(), reference, rtol=2e-4, atol=2e-4)
print('FFT_RESOLVED:'+handle.recipe_id)
operation.close()
ti.reset()
"""
    child_result = subprocess.run(
        [sys.executable, "-c", child, str(artifact)],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert child_result.returncode == 0, child_result.stdout + child_result.stderr
    assert "FFT_RESOLVED:" + decision.report.selected_recipe_id in child_result.stdout
    operation.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_fft_binding_frame_retains_closed_plan_until_executor_retirement():
    from taichi_forge._lib import core
    from taichi_forge.lang import impl

    if not ti.hardware.fft.is_available():
        pytest.skip("optional FFT provider is unavailable")
    if not getattr(core._CudaGraphBindingExecutor, "supports_capture_commands", lambda: False)():
        pytest.skip("fixed-library binding frames are unavailable")
    definition, operation, _providers, bindings, expected = _fft_definition()
    graph = definition.compile()
    native = core._CudaGraphBindingExecutor(
        graph._spec.nodes[0].compiled_graph, impl.current_cfg(), impl.get_runtime().prog
    )
    arguments = {name: value.arr for name, value in bindings.items()}
    frame = native.prepare(arguments)
    plan = next(lease for lease in graph._spec.lifetime_leases if hasattr(lease, "_handle"))
    plan.close()
    operation.close()
    try:
        for _ in range(31):
            native.run(frame)
        np.testing.assert_allclose(bindings["output"].to_numpy(), expected, rtol=2e-4, atol=2e-4)
        with pytest.raises(RuntimeError, match="fixed plan"):
            native.prepare(arguments)
        # A rejected new binding does not poison the already published frame.
        native.run(frame)
        np.testing.assert_allclose(bindings["output"].to_numpy(), expected, rtol=2e-4, atol=2e-4)
    finally:
        native.close()
    with pytest.raises(RuntimeError, match="closed"):
        native.run(frame)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_fft_recipe_preparation_and_numerical_semantics_are_explicit():
    if not ti.hardware.fft.is_available():
        pytest.skip("the optional cuFFT runtime is unavailable")
    definition, operation, providers, bindings, expected = _fft_definition(
        prepare=False
    )
    with pytest.raises(ti.graph.GraphRecipeProviderError) as caught:
        definition.recipe_catalog(providers=providers)
    assert caught.value.error_key == "fft_plan_preparation_required"
    operation.prepare()
    catalog = definition.recipe_catalog(providers=providers)
    # Grouped/ndrange rank-four maps currently lack fusion metadata; neither
    # missing metadata nor unchanged maps are represented as fake candidates.
    assert len(catalog.entries()) == 2
    with definition.materialization_context(
        provider_set=catalog.provider_set
    ) as context:
        for entry in catalog.entries():
            with context.materialize(entry.recipe) as result:
                result.executor.run(result.executor.bind(bindings))
                np.testing.assert_allclose(
                    bindings["output"].to_numpy(), expected, rtol=2e-4, atol=2e-4
                )
    inverse, inverse_op, _, _, _ = _fft_definition(direction="inverse")
    assert definition.semantic_graph_id != inverse.semantic_graph_id
    for tolerance in (True, float("inf"), -1.0):
        with pytest.raises((TypeError, ValueError)):
            ti.linalg.record_fft((8, 8), absolute_tolerance=tolerance, relative_tolerance=1e-4)
    operation.close()
    inverse_op.close()


@pytest.mark.parametrize("use_alternative", (False, True))
@pytest.mark.parametrize("explicit_close", (False, True))
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_fft_search_owner_can_retire_without_invalidating_graph_plan_leases(use_alternative, explicit_close):
    import gc
    import weakref

    if not ti.hardware.fft.is_available():
        pytest.skip("the optional cuFFT runtime is unavailable")
    definition, operation, providers, bindings, expected = _fft_definition()
    owner = weakref.ref(operation)
    plans = {name: weakref.ref(plan) for name, plan in operation._plans.items()}
    catalog = definition.recipe_catalog(providers=providers)
    recipe = next(entry.recipe for entry in catalog.entries() if bool(entry.recipe.fragments) == use_alternative)
    context = definition.materialization_context(provider_set=catalog.provider_set)
    result = context.materialize(recipe)
    graph = result.executor
    bound = graph.bind(bindings)
    # Queue work, release only the search owner, then continue with the same
    # published binding. Native completion and Graph leases own live plans.
    for _ in range(11):
        graph.run(bound)
    if explicit_close:
        operation.close()
        with pytest.raises(RuntimeError, match="closed"):
            operation.prepare()
        with pytest.raises(RuntimeError, match="closed"):
            operation.compile()
    del operation
    gc.collect()
    assert owner() is None
    assert (plans["whole_transform"]() is not None) == (not use_alternative)
    assert (plans["row_batch_column_inplace"]() is not None) == use_alternative
    before = ti.hardware.fft.cache_statistics()
    for _ in range(31):
        graph.run(bound)
    np.testing.assert_allclose(bindings["output"].to_numpy(), expected, rtol=2e-4, atol=2e-4)
    assert ti.hardware.fft.cache_statistics().create_requests == before.create_requests
    result.close()
    del bound, graph, result
    gc.collect()
    assert plans["whole_transform"]() is None
    assert plans["row_batch_column_inplace"]() is None
    # Prepared facts remain resolvable without retaining the alternative.
    assert [entry.recipe.recipe_id for entry in definition.recipe_catalog(providers=providers).entries()] == [
        entry.recipe.recipe_id for entry in catalog.entries()
    ]
    assert ti.hardware.fft.cache_statistics().create_requests == before.create_requests
    alternative = next(entry.recipe for entry in catalog.entries() if entry.recipe.fragments)
    with context.materialize(alternative) as restored:
        restored.executor.run(restored.executor.bind(bindings))
        np.testing.assert_allclose(bindings["output"].to_numpy(), expected, rtol=2e-4, atol=2e-4)
        assert ti.hardware.fft.cache_statistics().create_requests == before.create_requests + 1
    context.close()

    # Public search transports the definition, not an unused baseline Graph.
    # Constructing a session must not reacquire either retired plan.
    before_session = ti.hardware.fft.cache_statistics()
    session = definition.search_recipes(providers=providers, budget=ti.graph.GraphSearchBudget(evaluation_limit=2))
    assert session.recipes
    assert ti.hardware.fft.cache_statistics().create_requests == before_session.create_requests


@pytest.mark.parametrize("drift", ("workspace", "component"))
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_fft_retired_plan_recreation_rejects_changed_facts_and_releases_the_new_handle(monkeypatch, drift):
    import gc
    from taichi_forge.linalg import _fft
    from taichi_forge.graph._recipes.materialize import GraphMaterializationError

    if not ti.hardware.fft.is_available():
        pytest.skip("the optional cuFFT runtime is unavailable")
    definition, operation, providers, _, _ = _fft_definition()
    catalog = definition.recipe_catalog(providers=providers)
    alternative = next(entry.recipe for entry in catalog.entries() if entry.recipe.fragments)
    operation.close()
    gc.collect()
    before = ti.hardware.fft.cache_statistics()
    factory = _fft._SeparableFftPlan

    def changed_plan(*args):
        plan = factory(*args)
        if drift == "workspace":
            plan._workspace_bytes += 1
        else:
            from types import SimpleNamespace

            original = plan._retained_identity.to_dict()
            plan._retained_identity = SimpleNamespace(to_dict=lambda: {**original, "provider_scope": {"drift": True}})
        return plan

    monkeypatch.setattr(_fft, "_SeparableFftPlan", changed_plan)
    with definition.materialization_context(provider_set=catalog.provider_set) as context:
        with pytest.raises(GraphMaterializationError, match="component or workspace"):
            context.materialize(alternative)
        after = ti.hardware.fft.cache_statistics()
        assert after.live_handles == before.live_handles
        assert after.workspace_bytes_live == before.workspace_bytes_live
        monkeypatch.setattr(_fft, "_SeparableFftPlan", factory)
        with context.materialize(alternative):
            assert ti.hardware.fft.cache_statistics().live_handles == before.live_handles + 1


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_fft_metadata_catalog_does_not_recreate_plans_after_runtime_reset():
    from taichi_forge.hardware._fft_recipe import _sources

    if not ti.hardware.fft.is_available():
        pytest.skip("the optional cuFFT runtime is unavailable")
    definition, operation, _, _, _ = _fft_definition()
    source = next(_sources(definition))[2]
    operation.close()
    ti.reset()
    with pytest.raises(RuntimeError, match="retired runtime"):
        source._recording("row_batch_column_inplace")


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_fft_frozen_description_preserves_other_live_baselines_and_recompiles_after_retirement():
    import gc
    import weakref

    if not ti.hardware.fft.is_available():
        pytest.skip("the optional cuFFT runtime is unavailable")
    definition, operation, providers, bindings, expected = _fft_definition()
    baseline_plan = weakref.ref(operation._plans["whole_transform"])
    original = definition.compile()
    original_bound = original.bind(bindings)
    original.run(original_bound)
    catalog = definition.recipe_catalog(providers=providers)
    alternative = next(entry.recipe for entry in catalog.entries() if entry.recipe.fragments)
    operation.close()
    with definition.materialization_context(provider_set=catalog.provider_set) as context:
        with context.materialize(alternative) as result:
            candidate = result.executor
            bound = candidate.bind(bindings)
            for _ in range(7):
                original.run(original_bound)
                candidate.run(bound)
            np.testing.assert_allclose(bindings["output"].to_numpy(), expected, rtol=2e-4, atol=2e-4)
            assert baseline_plan() is not None
            del original_bound, original
            gc.collect()
            assert baseline_plan() is None
            before = ti.hardware.fft.cache_statistics()
            assert before.live_plans == 1
            # Compile is the resource boundary, not bind/run. The definition
            # itself must not keep this recreated baseline after retirement.
            rebuilt = definition.compile()
            assert ti.hardware.fft.cache_statistics().create_requests == before.create_requests + 1
            rebuilt.run(rebuilt.bind(bindings))
            np.testing.assert_allclose(bindings["output"].to_numpy(), expected, rtol=2e-4, atol=2e-4)
            del rebuilt
            gc.collect()
            assert ti.hardware.fft.cache_statistics().live_plans == 1


@pytest.mark.parametrize("alternative", (False, True))
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_fft_fresh_process_resolves_without_baseline_plan_and_materializes_only_selection(tmp_path, alternative):
    import json
    import subprocess
    import sys

    if not ti.hardware.fft.is_available():
        pytest.skip("the optional cuFFT runtime is unavailable")
    definition, operation, providers, bindings, expected = _fft_definition()

    def evaluate(graph, recipe):
        graph.run(graph.bind(bindings))
        np.testing.assert_allclose(bindings["output"].to_numpy(), expected, rtol=2e-4, atol=2e-4)
        record = next(lease for lease in graph._spec.lifetime_leases if hasattr(lease, "_graph_fft_source"))
        return {"alternative": float(record.plan._separable)}

    decision = definition.search_recipes(
        providers=providers,
        budget=ti.graph.GraphSearchBudget(evaluation_limit=2, repeat_count=1),
        target=ti.graph.GraphOptimizationTarget(objectives=(("alternative", "max" if alternative else "min"),)),
    ).run(evaluate)
    assert decision.status == "selected", decision.report.results
    artifact = tmp_path / "fft-restore.json"
    artifact.write_text(
        json.dumps(
            {
                "preparation": operation.preparation_artifact(),
                "selection": decision.selection_artifact.to_dict(),
                "alternative": alternative,
            }
        ),
        encoding="utf-8",
    )
    child = """
import json,sys
from pathlib import Path
import numpy as np
import taichi_forge as ti
from tests.python.test_graph_hardware_recipe import _fft_definition
ti.init(arch=ti.cuda,offline_cache=False)
data=json.loads(Path(sys.argv[1]).read_text())
assert ti.hardware.fft.cache_statistics().create_requests == 0
definition,operation,providers,bindings,expected = _fft_definition(prepare=False, preparation=data['preparation'])
assert ti.hardware.fft.cache_statistics().create_requests == 0
assert not operation._plans
selection=ti.graph.GraphRecipeSelectionArtifact.from_dict(data['selection'])
handle=definition.resolve_recipe(selection,providers=providers)
assert ti.hardware.fft.cache_statistics().create_requests == 0
with definition.materialize(handle) as result:
    stats=ti.hardware.fft.cache_statistics()
    assert stats.create_requests == stats.live_handles == stats.live_plans == 1,stats
    record=next(lease for lease in result.executor._spec.lifetime_leases if hasattr(lease,'_graph_fft_source'))
    assert record.plan._separable == data['alternative']
    assert record._graph_fft_source._preparation_origin == 'imported_expected_facts_not_current_measurement'
    bound=result.executor.bind(bindings)
    operation.close()
    for _ in range(13):
        result.executor.run(bound)
    np.testing.assert_allclose(bindings['output'].to_numpy(),expected,rtol=2e-4,atol=2e-4)
    assert ti.hardware.fft.cache_statistics().create_requests == 1
    if data['alternative']:
        catalog=definition.recipe_catalog(providers=providers)
        fragment=next(entry.recipe.fragments[0] for entry in catalog.entries() if entry.recipe.fragments)
        description=providers[-1].describe(definition,fragment.fragment_key)
        observation=description['preparation_observation']
        assert observation['preparation_origin'] == 'imported_expected_facts_not_current_measurement'
        assert observation['restoration_observation']['unselected_plans_created'] == 0
        assert observation['host_setup_seconds'] == data['preparation']['plans']['row_batch_column_inplace']['host_setup_seconds']
        from taichi_forge.graph._report_context import _provider_preparation_markdown
        text='\\n'.join(_provider_preparation_markdown([{'recipe_id':handle.recipe_id,'provider_claims':[
            {'claims':description,'provider_namespace':fragment.provider_namespace,'fragment_key':fragment.fragment_key}]}]))
        assert 'host_elapsed_for_selected_fft_plan_recreation' in text
print('PLAN_FREE_RESOLVE:'+handle.recipe_id)
ti.reset()
"""
    completed = subprocess.run([sys.executable, "-c", child, str(artifact)], capture_output=True, text=True, timeout=90)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "PLAN_FREE_RESOLVE:" + decision.report.selected_recipe_id in completed.stdout
    operation.close()


@pytest.mark.parametrize("drift", ("schema", "semantics", "device", "component", "workspace_type", "elapsed"))
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_fft_preparation_rejects_drift_before_creating_any_native_plan(drift):
    if not ti.hardware.fft.is_available():
        pytest.skip("the optional cuFFT runtime is unavailable")
    _, operation, _, _, _ = _fft_definition()
    preparation = operation.preparation_artifact()
    if drift in ("schema", "semantics", "device", "component"):
        preparation[drift] = "drift"
    elif drift == "workspace_type":
        preparation["plans"]["whole_transform"]["workspace_bytes"] = True
    else:
        preparation["plans"]["whole_transform"]["host_setup_seconds"] = -1
    before = ti.hardware.fft.cache_statistics()
    with pytest.raises(ValueError, match="FFT preparation"):
        _fft_definition(prepare=False, preparation=preparation)
    assert ti.hardware.fft.cache_statistics() == before
    operation.close()


@pytest.mark.parametrize("family", ("fft", "spmm", "mixed"))
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_fixed_library_binding_frames_prepare_without_execution_and_preserve_queued_buffers(family):
    from taichi_forge._lib import core

    if not ti.hardware.fft.is_available() or not ti.hardware.linalg.cusparse_spmm_is_available():
        pytest.skip("optional FFT/SpMM providers are unavailable")
    if not getattr(core._CudaGraphBindingExecutor, "supports_capture_commands", lambda: False)():
        pytest.skip("this native runtime does not support fixed-library binding frames")
    if family == "mixed":
        definition, operation, spmm, providers, bindings, expected = _mixed_hardware_definition(binding_frames=True)
    else:
        factory = _fft_definition if family == "fft" else _spmm_definition
        definition, operation, providers, bindings, expected_output = factory(binding_frames=True)
        expected = {"output": expected_output}
    catalog = definition.recipe_catalog(providers=providers)
    recipe = next(
        entry.recipe
        for entry in catalog.entries()
        if any(fragment.provider_namespace.endswith(".binding_frames") for fragment in entry.recipe.fragments)
    )
    with definition.materialization_context(provider_set=catalog.provider_set) as context:
        with context.materialize(recipe) as result:
            graph = result.executor
            groups = []
            for _ in range(5):
                group = {}
                for name, source in bindings.items():
                    array = ti.ndarray(ti.f32, source.shape)
                    array.from_numpy(source.to_numpy())
                    if name in expected:
                        array.fill(-987)
                    group[name] = array
                groups.append(group)
            published = [graph.bind(group) for group in groups]
            for group in groups:
                for name in expected:
                    np.testing.assert_array_equal(group[name].to_numpy(), -987)
            before = graph._graph_stats[0]["binding_frame_state"]
            assert before["executables"] == 1
            assert before["frames"] == len(groups)
            assert before["preparation_upload_calls"] == (0 if family == "mixed" else len(groups))
            if family != "spmm":
                operation.close()
            import sys

            provider_calls = []

            def observe(frame, event, _arg):
                if event == "call" and frame.f_globals.get("__name__", "") in (
                    "taichi_forge.linalg._fft",
                    "taichi_forge.linalg._spmm",
                    "taichi_forge.hardware._fft",
                    "taichi_forge.hardware._linalg",
                ):
                    provider_calls.append(frame.f_code.co_name)

            sys.setprofile(observe)
            try:
                for step in range(157):
                    graph.run(published[(step * 3) % len(groups)])
            finally:
                sys.setprofile(None)
            assert not provider_calls
            for group in groups:
                for name, reference in expected.items():
                    np.testing.assert_allclose(group[name].to_numpy(), reference, rtol=2e-4, atol=2e-4)
            after = graph._graph_stats[0]["binding_frame_state"]
            assert after["preparation_upload_calls"] == before["preparation_upload_calls"]
            assert after["argument_bytes"] == before["argument_bytes"]
            assert after["executables"] == 1
            assert not result.manifest.task_topology_exact

            if family == "spmm":
                matrix = operation.matrix
                replacement = ti.ndarray(ti.f32, matrix._debug_runtime_stats()["identity"]["nnz"])
                replacement.fill(0)
                matrix.update_values(replacement)
                for binding in published:
                    graph.run(binding)
                # The post-SpMM map adds 0.25; updates change values, not frames.
                for group in groups:
                    np.testing.assert_allclose(group["output"].to_numpy(), 0.25)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_mixed_library_frames_are_reachable_through_default_complete_recipe_search():
    from taichi_forge._lib import core

    if not ti.hardware.fft.is_available() or not ti.hardware.linalg.cusparse_spmm_is_available():
        pytest.skip("optional FFT/SpMM providers are unavailable")
    if not getattr(core._CudaGraphBindingExecutor, "supports_capture_commands", lambda: False)():
        pytest.skip("this native runtime does not support fixed-library binding frames")
    definition, operation, spmm, providers, bindings, expected = _mixed_hardware_definition(binding_frames=True)
    observed = []

    def evaluate(graph, recipe):
        bound = graph.bind(bindings)
        for _ in range(3):
            graph.run(bound)
        for name, reference in expected.items():
            np.testing.assert_allclose(bindings[name].to_numpy(), reference, rtol=2e-4, atol=2e-4)
        framed = graph._graph_stats[0]["last_path"] == "cuda_prepared_binding_plan"
        observed.append(framed)
        return {"immutable_frames": float(framed)}

    decision = definition.search_recipes(
        providers=providers,
        target=ti.graph.GraphOptimizationTarget(objectives=(("immutable_frames", "max"),)),
        budget=ti.graph.GraphSearchBudget(evaluation_limit=16, repeat_count=1),
        strategy=ti.graph.GraphRecipeSearchStrategy(mode="exact_if_bounded"),
    ).run(evaluate)
    assert decision.status == "selected", decision.report.results
    assert decision.report.search_complete
    assert any(observed) and not all(observed)
    selected = decision.selection_artifact.recipe_manifest
    assert any(item["provider_namespace"].endswith(".binding_frames") for item in selected["fragments"])
    operation.close()
