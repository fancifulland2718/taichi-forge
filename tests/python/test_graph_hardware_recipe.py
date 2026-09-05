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


def _spmm_definition(*, tolerance=2e-5, prepare=True, component_version=None):
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
    providers = (*ti.graph.default_recipe_providers(), ti.hardware.linalg.SparseSpmmRecipeProvider())
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


def _fft_definition(*, prepare=True, direction="forward"):
    dimensions, batch = (24, 40), 3
    source = ti.linalg.record_fft(
        dimensions,
        batch_count=batch,
        direction=direction,
        output="transform",
        absolute_tolerance=2e-4,
        relative_tolerance=2e-4,
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
        *ti.graph.default_recipe_providers(),
        ti.hardware.fft.FftRecipeProvider(),
    )
    return builder.freeze(), source, providers, bindings, expected


def _mixed_hardware_definition():
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
        *ti.graph.default_recipe_providers(),
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
            ti.linalg.record_fft(
                (8, 8), absolute_tolerance=tolerance, relative_tolerance=1e-4
            )
    operation.close()
    inverse_op.close()
