"""Public complete-recipe discovery, composition, resolve and resource scope."""

import json
import os

import numpy as np
import pytest
import taichi_forge as ti

from taichi_forge.hardware.source_providers import CubSegmentedScanRecipeProvider
from tests import test_utils


def _definition(inclusive=True, dtype=ti.u32, two_scans=False):
    n = 8193
    offsets = np.array([0, 0, 1, 32, 32, 4096, n], dtype=np.int32)
    layout = ti.algorithms.SegmentedLayout.from_offsets(offsets, capacity=n + 7)
    arrays = [ti.ndarray(dtype, n + 7) for _ in range(3 if two_scans else 2)]
    host = np.random.default_rng(620).integers(0, 2**32, size=n + 7, dtype=np.uint32)
    if dtype == ti.i32:
        host = host.view(np.int32)
    arrays[0].from_numpy(host)
    for array in arrays[1:]:
        array.fill(77)
    builder = ti.graph.GraphBuilder()
    for source, output in zip(arrays, arrays[1:]):
        builder.segmented_scan(source, layout, output, inclusive=inclusive)
    expected = host.copy()
    for _ in arrays[1:]:
        for begin, end in zip(offsets, offsets[1:]):
            part = expected[begin:end].copy()
            if inclusive:
                expected[begin:end] = np.cumsum(part, dtype=host.dtype)
            elif end > begin:
                expected[begin] = 0
                expected[begin + 1 : end] = np.cumsum(part[:-1], dtype=host.dtype)
    expected[n:] = 77
    return builder.freeze(), arrays, expected


def _provider():
    path = os.environ.get("TI_FORGE_TEST_CUB_SOURCE_PROVIDER_MANIFEST")
    if not path:
        pytest.skip("an explicit reset-monoid source addon was not supplied")
    return CubSegmentedScanRecipeProvider(path)


@test_utils.test(arch=ti.cuda, offline_cache=False)
@pytest.mark.parametrize("inclusive,dtype", ((True, ti.u32), (False, ti.i32)))
def test_segmented_addon_public_search_resolve_and_semantic_identity(
    inclusive, dtype, tmp_path
):
    provider = _provider()
    definition, arrays, expected = _definition(inclusive, dtype)
    providers = (*ti.graph.default_recipe_providers(), provider)
    ordinary = definition.recipe_catalog()
    assert all(
        "segmented_scan_addon" not in item.provider_namespace
        for item in ordinary.fragments
    )
    catalog = definition.recipe_catalog(providers=providers)
    addon = tuple(
        item
        for item in catalog.fragments
        if item.provider_namespace == provider.descriptor.namespace
    )
    assert len(addon) == 1
    assert sum(item.bytes for item in addon[0].resources) > 4 * ((8193 + 31) // 32)
    session = definition.search_recipes(
        providers=providers,
        target=ti.graph.GraphOptimizationTarget(
            objectives=(("addon_selected", "max"),)
        ),
        budget=ti.graph.GraphSearchBudget(evaluation_limit=16, repeat_count=1),
        strategy=ti.graph.GraphRecipeSearchStrategy(mode="exact_if_bounded"),
    )
    observed = set()

    def evaluate(graph, recipe):
        bound = graph.bind({})
        for _ in range(3):
            graph.run(bound)
        # Capacity padding is outside the segmented layout's active prefix.
        # Existing global and local strategies do not agree on padding writes.
        np.testing.assert_array_equal(arrays[-1].to_numpy()[:8193], expected[:8193])
        selected = any(
            item.provider_namespace == provider.descriptor.namespace
            for item in catalog.entry(recipe.recipe_id).recipe.fragments
        )
        if selected:
            assert graph._graph_stats[0]["last_path"] == "cuda_exact_replay"
            assert graph.definition.semantic_graph_id == definition.semantic_graph_id
            assert graph.execution_stats().memory.provider_generation_known_resident_requested_bytes == sum(
                item.bytes for item in addon[0].resources
            )
        observed.add(recipe.recipe_id)
        return {"addon_selected": float(selected)}

    decision = session.run(evaluate)
    assert decision.status == "selected", decision.report.results
    assert decision.report.search_complete
    assert len(observed) == len(session.recipes)
    assert "reset_monoid_lookback" in json.dumps(decision.report.recipe_annotations)
    restored = definition.resolve_recipe(
        decision.selection_artifact, providers=providers
    )
    with definition.materialize(restored) as materialized:
        evaluate(materialized.executor, restored)
        candidate_physical = materialized.materialized_physical_id
    with definition.materialization_context(providers=providers) as context:
        with context.materialize(catalog.baseline.recipe) as baseline:
            assert candidate_physical != baseline.materialized_physical_id
    fresh, fresh_arrays, _ = _definition(inclusive, dtype)
    assert fresh.semantic_graph_id == definition.semantic_graph_id
    resolved = fresh.resolve_recipe(decision.selection_artifact, providers=providers)
    with fresh.materialize(resolved) as materialized:
        materialized.executor.run({})
        np.testing.assert_array_equal(fresh_arrays[-1].to_numpy(), expected)
    if inclusive:
        import subprocess
        import sys

        artifact = tmp_path / "selection.json"
        artifact.write_text(
            json.dumps(decision.selection_artifact.to_dict()), encoding="utf-8"
        )
        child = """
import json, sys
from pathlib import Path
import numpy as np
import taichi_forge as ti
from tests.python.test_graph_segmented_addon_recipe import _definition, _provider
ti.init(arch=ti.cuda, offline_cache=False)
definition, arrays, expected = _definition()
providers = (*ti.graph.default_recipe_providers(), _provider())
artifact = ti.graph.GraphRecipeSelectionArtifact.from_dict(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')))
resolved = definition.resolve_recipe(artifact, providers=providers)
with definition.materialize(resolved) as candidate:
    candidate.executor.run({})
    np.testing.assert_array_equal(arrays[-1].to_numpy(), expected)
print('ADDON_RESOLVED:' + resolved.recipe_id)
ti.reset()
"""
        completed = subprocess.run(
            [sys.executable, "-c", child, str(artifact)],
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert "ADDON_RESOLVED:" + resolved.recipe_id in completed.stdout


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_segmented_addon_composes_two_regions_with_independent_storage():
    provider = _provider()
    definition, arrays, expected = _definition(two_scans=True)
    providers = (*ti.graph.default_recipe_providers(), provider)
    catalog = definition.recipe_catalog(providers=providers)
    fragments = tuple(
        item
        for item in catalog.fragments
        if item.provider_namespace == provider.descriptor.namespace
    )
    assert len(fragments) == 2
    recipe = catalog.composer.compose(fragments)
    with definition.materialization_context(providers=providers) as context:
        with context.materialize(recipe) as materialized:
            graph = materialized.executor
            bound = graph.bind({})
            for _ in range(3):
                graph.run(bound)
            np.testing.assert_array_equal(arrays[-1].to_numpy(), expected)
            assert len(graph._spec.fixed_runtime_args) == 8
            workspaces = [
                value
                for name, value in graph._spec.fixed_runtime_args.items()
                if name.endswith("_workspace")
            ]
            assert workspaces[0] is not workspaces[1]
            tasks = materialized.manifest.tasks
            assert (
                tasks[0].properties["effects"][2]["resource"]["static_slot"]
                == tasks[1].properties["effects"][0]["resource"]["static_slot"]
            )
