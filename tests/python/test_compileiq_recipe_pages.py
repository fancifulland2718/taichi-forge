"""Small real trials exercise paging; large domains only test description size."""

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge._compileiq_opaque import _CompileIQOpaqueRecipeTransport
from tests import test_utils


def test_frame_byte_limit_and_missing_paging_capability_are_checked_before_trials():
    from taichi_forge._compileiq_opaque import _validated_compileiq_capability
    from taichi_forge._compileiq_opaque import CompileIQOpaqueUnavailableError

    components = _validated_compileiq_capability()
    ids = ("base", *(f"r:{i}:" + "界" * 900 for i in range(1700)))
    transport = _CompileIQOpaqueRecipeTransport(
        provider_namespace="test.pages",
        domain_version="v1",
        provider_semantic_fingerprint="all-long-ids",
        recipe_ids=ids,
        baseline_recipe_id="base",
        capability_components=components,
        domain_owner="test",
        recipe_description="test recipe",
        dynamic_only=True,
    )
    pages = tuple(transport.recipe_pages())
    assert len(pages) > 1
    assert set(key for page in pages for key in page) == set(ids)
    for page in pages:
        domain = type(transport.search_space)(
            **{
                **transport.search_space.model_dump(by_alias=True),
                "recipe_ids": page,
            }
        )
        assert domain.recipe_ids
    capability = dict(transport.capability)
    capability.pop("opaque_recipe_paging")
    transport._capability = capability
    with pytest.raises(CompileIQOpaqueUnavailableError, match="logical_stage_pages_v1"):
        transport.batch_v2(
            recipe_ids=("base",),
            stage_index=0,
            stage_fingerprint="stage",
            fidelity_name="full",
            fidelity_ordinal=0,
            repeat_count=1,
            terminal=True,
            page_group_fingerprint="group",
            page_final=False,
        )


def _small_pages(monkeypatch):
    original = _CompileIQOpaqueRecipeTransport.recipe_pages
    monkeypatch.setattr(
        _CompileIQOpaqueRecipeTransport,
        "recipe_pages",
        lambda self, ids=None, **kwargs: original(self, ids, page_size=2),
    )


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
@pytest.mark.parametrize("mode", ("exact_if_bounded", "staged"))
def test_graph_pages_resume_whole_stage_and_keep_earlier_winner(monkeypatch, mode):
    _small_pages(monkeypatch)

    @ti.kernel
    def first(a: ti.types.ndarray(ti.i32, ndim=1), b: ti.types.ndarray(ti.i32, ndim=1)):
        for i in a:
            b[i] = a[i] * 2

    @ti.kernel
    def second(
        a: ti.types.ndarray(ti.i32, ndim=1), b: ti.types.ndarray(ti.i32, ndim=1), c: ti.types.ndarray(ti.i32, ndim=1)
    ):
        for i in a:
            c[i] = b[i] + 3

    @ti.kernel
    def third(
        a: ti.types.ndarray(ti.i32, ndim=1), c: ti.types.ndarray(ti.i32, ndim=1), out: ti.types.ndarray(ti.i32, ndim=1)
    ):
        for i in a:
            out[i] = c[i] * 2

    symbolic = [ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.i32, ndim=1) for name in ("a", "b", "c", "out")]
    builder = ti.graph.GraphBuilder()
    builder.dispatch(first, symbolic[0], symbolic[1])
    builder.dispatch(second, symbolic[0], symbolic[1], symbolic[2])
    builder.dispatch(third, symbolic[0], symbolic[2], symbolic[3])
    definition = builder.freeze()
    providers = tuple(
        provider
        for provider in ti.graph.default_recipe_providers()
        if provider.descriptor.namespace in ("taichi_forge.graph.runtime_assembly", "taichi_forge.graph.map_fusion")
    )
    arrays = {name: ti.ndarray(ti.i32, shape=65) for name in ("a", "b", "c", "out")}
    host = np.arange(65, dtype=np.int32)
    arrays["a"].from_numpy(host)
    options = dict(
        providers=providers,
        target=ti.graph.GraphOptimizationTarget(objectives=(("score", "min"),)),
        strategy=ti.graph.GraphRecipeSearchStrategy(mode=mode, max_generation_rounds=0),
        workload_context=ti.graph.GraphWorkloadContext({"fixture": "paged-three-maps"}),
        evaluation_contract=ti.graph.GraphEvaluationContract({"metric": "synthetic-ranking-after-exact-oracle"}),
        backend_environment=ti.graph.GraphBackendEnvironment({"fixture": "current-test-device"}),
    )
    session = definition.search_recipes(budget=ti.graph.GraphSearchBudget(2), **options)
    assert len(session.recipes) >= 3
    first_candidate = sorted(key.recipe_id for key in session.recipes if key.recipe_id != session.baseline.recipe_id)[0]
    calls = []

    def evaluate(graph, recipe):
        graph.run(arrays)
        ti.sync()
        np.testing.assert_array_equal(arrays["out"].to_numpy(), (host * 2 + 3) * 2)
        calls.append(recipe.recipe_id)
        return {"score": 1.0 if recipe.recipe_id == first_candidate else 10.0}

    partial = session.run(evaluate)
    assert not partial.report.search_complete
    assert partial.selection is None
    assert len(calls) == 2
    saved = partial.checkpoint
    resumed = definition.search_recipes(
        budget=ti.graph.GraphSearchBudget(20),
        checkpoint=saved,
        **options,
    ).run(evaluate)
    assert resumed.report.search_complete
    assert resumed.selection.recipe_id == first_candidate
    checkpoint = resumed.checkpoint.compileiq_checkpoint
    assert checkpoint["batches"][-1]["page_final"]
    keys = [record["request"]["measurement_key"] for record in checkpoint["records"]]
    assert len(keys) == len(set(keys)) == len(calls)
    if mode == "exact_if_bounded":
        assert len(calls) == len(session.recipes)
    assert len(checkpoint["batches"]) > 1
    assert resumed.report.compileiq_report.baseline_comparison


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_staged_pages_resume_generated_compositions(monkeypatch):
    _small_pages(monkeypatch)
    offsets = np.array([0, 3, 7], dtype=np.int32)
    layout = ti.algorithms.SegmentedLayout.from_offsets(offsets, capacity=7)
    inputs = [ti.ndarray(ti.i32, shape=7) for _ in range(2)]
    outputs = [ti.ndarray(ti.i32, shape=7) for _ in range(2)]
    host = np.arange(7, dtype=np.int32)
    expected = np.concatenate((np.cumsum(host[:3]), np.cumsum(host[3:])))
    builder = ti.graph.GraphBuilder()
    for source, output in zip(inputs, outputs):
        source.from_numpy(host)
        builder.segmented_scan(source, layout, output, inclusive=True)
    definition = builder.freeze()
    options = dict(
        target=ti.graph.GraphOptimizationTarget(objectives=(("fragments", "max"),)),
        strategy=ti.graph.GraphRecipeSearchStrategy(mode="staged", max_generation_rounds=1, max_generated_recipes=14),
        workload_context=ti.graph.GraphWorkloadContext({"fixture": "two-tiny-scans"}),
        evaluation_contract=ti.graph.GraphEvaluationContract({"metric": "fragment-count-after-oracle"}),
        backend_environment=ti.graph.GraphBackendEnvironment({"fixture": "current-cuda-test-device"}),
    )

    def budget(limit):
        return ti.graph.GraphSearchBudget(limit, minimum_survivors=32)

    initial = definition.search_recipes(budget=budget(64), **options)
    initial_count = len(initial.recipes)
    assert initial_count > 2
    calls = []

    def evaluate(graph, recipe):
        graph.run({})
        ti.sync()
        for output in outputs:
            np.testing.assert_array_equal(output.to_numpy(), expected)
        calls.append(recipe.recipe_id)
        return {"fragments": float(recipe.manifest.selected_fragment_count)}

    partial = definition.search_recipes(budget=budget(initial_count + 1), **options).run(evaluate)
    assert not partial.report.search_complete
    assert partial.checkpoint.compileiq_checkpoint["batches"][-1]["fidelity"]["ordinal"] == 1
    resumed = definition.search_recipes(budget=budget(64), checkpoint=partial.checkpoint, **options).run(evaluate)
    assert resumed.report.search_complete
    cp = resumed.checkpoint.compileiq_checkpoint
    assert {batch["fidelity"]["ordinal"] for batch in cp["batches"]} == {0, 1, 2}
    assert len(resumed.checkpoint.generation["recipes"]) > initial_count
    keys = [record["request"]["measurement_key"] for record in cp["records"]]
    assert len(keys) == len(set(keys)) == len(calls)
