import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.graph import compileiq_recipe_search

from tests import test_utils

_RECIPE_ENV = "TAICHI_FORGE_INTERNAL_GRAPH_REDUCTION_RECIPE"


def _reduction_arguments(dtype):
    return (
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "values", dtype, ndim=1),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", dtype, ndim=1),
    )


def _build_reduction_graph(
    dtype,
    count,
    *,
    absolute_tolerance=None,
    relative_tolerance=None,
):
    values, output = _reduction_arguments(dtype)
    builder = ti.graph.GraphBuilder()
    builder.reduce(
        values,
        output,
        count=count,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    return builder.compile()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_graph_reduction_compileiq_reconstructs_complete_typed_domain(monkeypatch):
    count = 4097
    graph = _build_reduction_graph(ti.i32, count)
    assert graph._compileiq_graph_reduction_status == "complete_recipe_domain", (
        graph._compileiq_graph_reduction_status
    )
    search = compileiq_recipe_search(graph)

    assert search.manifest()["recipe_kind"] == "graph_reduction"
    assert search.search_space.provider_namespace == "taichi_forge.graph.reduction"
    assert search.search_space.domain_version == "graph-reduction-complete-recipe.v1"
    assert len(search.recipe_ids) == 2

    manifests = {
        recipe_id: search.recipe_manifest(recipe_id) for recipe_id in search.recipe_ids
    }
    strategies = {
        manifest["reduction_recipe_manifest"]["strategy"]
        for manifest in manifests.values()
    }
    assert strategies == {"direct_atomic_tls", "block_partial_finalize"}
    direct_id = search.baseline_recipe_id
    phased_id = next(
        recipe_id
        for recipe_id, manifest in manifests.items()
        if manifest["reduction_recipe_manifest"]["strategy"]
        == "block_partial_finalize"
    )
    assert direct_id == search.baseline_recipe_id
    for manifest in manifests.values():
        for stage in manifest["reduction_recipe_manifest"]["physical_stages"]:
            for task in stage["tasks"]:
                assert not {"task_id", "logical_task_id", "task_name"}.intersection(
                    task
                )

    expected_semantics = {
        "operation": "sum",
        "dtype": "i32",
        "count": count,
        "identity": 0,
        "associativity": "modular_integer_sum",
        "reduction_order": "unspecified_integer",
        "determinism": "exact",
        "absolute_tolerance": 0.0,
        "relative_tolerance": 0.0,
        "input": "values",
        "output": "output",
    }
    assert all(
        manifest["reduction_recipe_manifest"]["semantics"] == expected_semantics
        for manifest in manifests.values()
    )
    assert manifests[direct_id]["reduction_recipe_manifest"]["workspace"] == {
        "ownership": "none",
        "exclusive_submission": False,
        "elements": 0,
        "bytes": 0,
    }
    partial_count = (count + 256 * 4 - 1) // (256 * 4)
    assert manifests[phased_id]["reduction_recipe_manifest"]["workspace"] == {
        "ownership": "graph_instance",
        "exclusive_submission": True,
        "elements": partial_count,
        "bytes": partial_count * 4,
    }

    def rebuild(recipe_id):
        parameters = {
            "domain_fingerprint": search.domain_fingerprint,
            "recipe_id": recipe_id,
        }
        environment = search.worker_environment(parameters)
        assert environment[_RECIPE_ENV] == manifests[recipe_id]["reduction_recipe_id"]
        with monkeypatch.context() as reconstruction:
            for name, value in environment.items():
                reconstruction.setenv(name, value)
            rebuilt = _build_reduction_graph(ti.i32, count)
        search.verify_materialized_graph(parameters, rebuilt)
        return rebuilt

    values = ti.ndarray(ti.i32, shape=count)
    output = ti.ndarray(ti.i32, shape=1)
    host_values = ((np.arange(count, dtype=np.int32) % 23) - 11).astype(np.int32)
    values.from_numpy(host_values)
    expected = np.asarray(host_values.sum(dtype=np.int64), dtype=np.int32).item()
    for recipe_id in search.recipe_ids:
        rebuilt = rebuild(recipe_id)
        output.fill(123)
        bindings = rebuilt.bind({"values": values, "output": output})
        assert bindings.fast_path_qualified
        assert bindings.statistics()["memory_recipe_publish_validated"]
        assert bindings.statistics()["fixed_bindings_flattened"]
        rebuilt.run(bindings)
        ti.sync()
        assert output.to_numpy()[0] == expected
        statistics = rebuilt.binding_statistics()
        assert statistics["raw_replay_validations"] == 0
        assert statistics["version_volatile_replays"] == 0
        assert statistics["version_fast_replays"] == 1

    phased = rebuild(phased_id)
    with pytest.raises(RuntimeError, match="requires proven disjoint storage"):
        phased.bind({"values": values, "output": values})
    short_values = ti.ndarray(ti.i32, shape=count - 1)
    with pytest.raises(RuntimeError, match=f"at least {count} scalar elements"):
        phased.bind({"values": short_values, "output": output})


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_graph_reduction_requires_explicit_floating_point_equivalence(monkeypatch):
    with pytest.raises(ValueError, match="requires explicit"):
        _build_reduction_graph(ti.f32, 1024)
    with pytest.raises(ValueError, match="positive tolerance"):
        _build_reduction_graph(
            ti.f32,
            1024,
            absolute_tolerance=0.0,
            relative_tolerance=0.0,
        )
    with pytest.raises(ValueError, match="tolerances must be zero"):
        _build_reduction_graph(
            ti.i32,
            1024,
            absolute_tolerance=1.0,
            relative_tolerance=0.0,
        )

    graph = _build_reduction_graph(
        ti.f32,
        1024,
        absolute_tolerance=1e-4,
        relative_tolerance=1e-5,
    )
    search = compileiq_recipe_search(graph)
    assert len(search.recipe_ids) == 2
    values = ti.ndarray(ti.f32, shape=1024)
    output = ti.ndarray(ti.f32, shape=1)
    host_values = ((np.arange(1024, dtype=np.float32) % 17) - 8) * 0.125
    values.from_numpy(host_values)
    expected = host_values.sum(dtype=np.float32)
    for recipe_id in search.recipe_ids:
        parameters = {
            "domain_fingerprint": search.domain_fingerprint,
            "recipe_id": recipe_id,
        }
        manifest = search.recipe_manifest(recipe_id)
        semantics = manifest["reduction_recipe_manifest"]["semantics"]
        assert semantics["dtype"] == "f32"
        assert semantics["determinism"] == "within_tolerance"
        assert semantics["absolute_tolerance"] == 1e-4
        assert semantics["relative_tolerance"] == 1e-5
        with monkeypatch.context() as reconstruction:
            for name, value in search.worker_environment(parameters).items():
                reconstruction.setenv(name, value)
            rebuilt = _build_reduction_graph(
                ti.f32,
                1024,
                absolute_tolerance=1e-4,
                relative_tolerance=1e-5,
            )
        search.verify_materialized_graph(parameters, rebuilt)
        output.fill(123.0)
        rebuilt.run(rebuilt.bind({"values": values, "output": output}))
        ti.sync()
        np.testing.assert_allclose(output.to_numpy()[0], expected, rtol=1e-5, atol=1e-4)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_graph_reduction_does_not_mix_with_other_graph_recipe_domains():
    count = 1024

    @ti.kernel
    def copy(
        values: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for index in range(count):
            output[index] = values[index]

    values, output = _reduction_arguments(ti.i32)
    result = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "result", ti.i32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(copy, values, output)
    builder.reduce(output, result, count=count)
    graph = builder.compile()

    assert graph._compileiq_graph_reduction_status == "definition_out_of_scope"
    with pytest.raises(ValueError, match="exact map-partition search requires"):
        compileiq_recipe_search(graph)
