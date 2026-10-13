import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.graph import compileiq_recipe_search

from tests import test_utils


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
def test_graph_reduction_compileiq_materializes_complete_typed_domain():
    count = 4097
    graph = _build_reduction_graph(ti.i32, count)
    search = compileiq_recipe_search(graph)

    assert search.manifest()["recipe_kind"] == "graph_reduction"
    assert search.search_space.provider_namespace == "taichi_forge.graph.reduction"
    assert search.search_space.domain_version == "graph-reduction-complete-recipe.v2"
    assert len(search.recipe_ids) == 4

    manifests = {
        recipe_id: search.recipe_manifest(recipe_id) for recipe_id in search.recipe_ids
    }
    strategies = {
        manifest["reduction_recipe_manifest"]["strategy"]
        for manifest in manifests.values()
    }
    assert strategies == {"direct_atomic_tls", "block_partial_finalize"}
    topologies = {
        (
            manifest["reduction_recipe_manifest"]["topology"]["block_dim"],
            manifest["reduction_recipe_manifest"]["topology"]["items_per_thread"],
        )
        for manifest in manifests.values()
        if manifest["reduction_recipe_manifest"]["strategy"] == "block_partial_finalize"
    }
    assert topologies == {(256, 4), (128, 4), (64, 2)}
    assert all(
        manifest["reduction_recipe_manifest"]["topology"]["in_block_reduction"]
        == "warp_shuffle_shared_finalize"
        for manifest in manifests.values()
        if manifest["reduction_recipe_manifest"]["strategy"] == "block_partial_finalize"
    )
    direct_id = search.baseline_recipe_id
    phased_id = next(
        recipe_id
        for recipe_id, manifest in manifests.items()
        if manifest["reduction_recipe_manifest"]["strategy"] == "block_partial_finalize"
        and manifest["reduction_recipe_manifest"]["topology"]["block_dim"] == 256
        and manifest["reduction_recipe_manifest"]["topology"]["items_per_thread"] == 4
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

    def materialize(recipe_id, context):
        parameters = {
            "domain_fingerprint": search.domain_fingerprint,
            "recipe_id": recipe_id,
        }
        materialized = search.materialize(parameters, context=context)
        search.verify_materialized_graph(parameters, materialized)
        assert materialized.manifest.recipe_id == recipe_id
        return materialized

    values = ti.ndarray(ti.i32, shape=count)
    output = ti.ndarray(ti.i32, shape=1)
    host_values = ((np.arange(count, dtype=np.int32) % 23) - 11).astype(np.int32)
    values.from_numpy(host_values)
    expected = np.asarray(host_values.sum(dtype=np.int64), dtype=np.int32).item()
    semantic_graph_ids = set()
    with graph.definition.materialization_context() as context:
        for recipe_id in search.recipe_ids:
            with materialize(recipe_id, context) as materialized:
                rebuilt = materialized.executor
                semantic_graph_ids.add(rebuilt.definition.semantic_graph_id)
                output.fill(123)
                bindings = rebuilt.bind({"values": values, "output": output})
                assert bindings.fast_path_qualified
                assert bindings.statistics()["memory_recipe_publish_validated"]
                assert bindings.statistics()["fixed_bindings_flattened"]
                for _ in range(8):
                    rebuilt.run(bindings)
                ti.sync()
                assert output.to_numpy()[0] == expected
                segment = rebuilt.execution_stats().segments[0]
                assert segment.last_path == "cuda_exact_replay"
                assert segment.backend_replay_path
                assert segment.fallback_reason == "none"
                statistics = rebuilt.binding_statistics()
                assert statistics["raw_replay_validations"] == 0
                assert statistics["version_volatile_replays"] == 0
                assert statistics["version_fast_replays"] == 8
        assert semantic_graph_ids == {graph.definition.semantic_graph_id}

        with materialize(phased_id, context) as materialized:
            phased = materialized.executor
            with pytest.raises(RuntimeError, match="requires proven disjoint storage"):
                phased.bind({"values": values, "output": values})
            short_values = ti.ndarray(ti.i32, shape=count - 1)
            with pytest.raises(RuntimeError, match=f"at least {count} scalar elements"):
                phased.bind({"values": short_values, "output": output})


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_graph_reduction_materialization_preserves_private_endpoint_aliases():
    count = 257

    @ti.kernel
    def produce(x: ti.types.ndarray(ti.i32, ndim=1), private: ti.types.ndarray(ti.i32, ndim=1)):
        for i in range(count):
            private[i] = x[i] + 1

    @ti.kernel
    def publish(private: ti.types.ndarray(ti.i32, ndim=1), output: ti.types.ndarray(ti.i32, ndim=1)):
        output[0] = private[0]

    builder = ti.graph.GraphBuilder()
    values = builder.private_ndarray("private_values", ti.i32, count)
    result = builder.private_ndarray("private_result", ti.i32, 1)
    aliases = []
    for symbol in (values, result):
        alias = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, symbol.name + "_alias", ti.i32, ndim=1)
        builder._bind_internal_runtime_arg(alias, builder._declared_private_bindings[symbol.name][1])
        aliases.append(alias)
    public = {name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.i32, ndim=1) for name in ("x", "output")}
    builder.dispatch(produce, public["x"], values)
    builder.reduce(*aliases, count=count)
    builder.dispatch(publish, result, public["output"])
    graph = builder.compile()
    search = compileiq_recipe_search(graph)
    staged = next(
        recipe for recipe in search.recipe_ids
        if search.recipe_manifest(recipe).get("reduction_recipe_manifest", {}).get("strategy")
        == "block_partial_finalize"
    )
    arrays = dict(x=ti.ndarray(ti.i32, count), output=ti.ndarray(ti.i32, 1))
    for recipe in (search.baseline_recipe_id, staged):
        with search.materialize(dict(domain_fingerprint=search.domain_fingerprint, recipe_id=recipe)) as materialized:
            selected = materialized.executor
            assert set(selected._spec.runtime_arg_names) == {"x", "output"}
            for symbol, alias in zip((values, result), aliases):
                assert selected._spec.fixed_runtime_args[symbol.name] is selected._spec.fixed_runtime_args[alias.name]
            bindings = selected.bind(arrays)
            for shift in (0, 3):
                source = np.arange(count, dtype=np.int32) + shift
                arrays["x"].from_numpy(source)
                selected.run(bindings)
                assert arrays["output"].to_numpy()[0] == (source + 1).sum()
    graph.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_graph_reduction_chain_binds_its_declared_private_bridge(tmp_path):
    count = 257
    values, output = _reduction_arguments(ti.i32)
    builder = ti.graph.GraphBuilder()
    bridge = builder.private_ndarray("bridge", ti.i32, 1)
    builder.reduce(values, bridge, count=count)
    builder.reduce(bridge, output, count=1)
    graph = builder.compile()
    assert set(graph._spec.runtime_arg_names) == {"values", "output"}
    arrays = dict(values=ti.ndarray(ti.i32, count), output=ti.ndarray(ti.i32, 1))
    arrays["values"].from_numpy(np.arange(count, dtype=np.int32))
    graph.run(graph.bind(arrays))
    expected = count * (count - 1) // 2
    assert arrays["output"].to_numpy()[0] == expected

    search = compileiq_recipe_search(graph)
    selected_id = next(
        recipe for recipe in search.recipe_ids
        if search.recipe_manifest(recipe).get("reduction_recipe_manifest", {}).get("strategy")
        == "block_partial_finalize"
    )
    module = ti.aot.Module()
    with search.materialize(dict(domain_fingerprint=search.domain_fingerprint, recipe_id=selected_id)) as materialized:
        selected = materialized.executor
        assert set(selected._spec.runtime_arg_names) == {"values", "output"}
        selected.run(selected.bind(arrays))
        assert arrays["output"].to_numpy()[0] == expected
        module.add_execution_plan("step", selected)
    graph.close()
    artifact = tmp_path / "chain.tcm"
    module.archive(str(artifact))
    with ti.aot.borrow_runtime() as runtime:
        loaded = runtime.load_module(artifact)
        imported = {name: runtime.import_ndarray(value) for name, value in arrays.items()}
        prepared = loaded.prepare_execution("step", imported)
        for value in (2, 3):
            arrays["values"].fill(value)
            prepared()
            assert arrays["output"].to_numpy()[0] == count * value


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_graph_reduction_requires_explicit_floating_point_equivalence():
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
    assert len(search.recipe_ids) == 4
    values = ti.ndarray(ti.f32, shape=1024)
    output = ti.ndarray(ti.f32, shape=1)
    host_values = ((np.arange(1024, dtype=np.float32) % 17) - 8) * 0.125
    values.from_numpy(host_values)
    expected = host_values.sum(dtype=np.float32)

    with graph.definition.materialization_context() as context:
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
            with search.materialize(parameters, context=context) as materialized:
                search.verify_materialized_graph(parameters, materialized)
                rebuilt = materialized.executor
                output.fill(123.0)
                rebuilt.run(rebuilt.bind({"values": values, "output": output}))
                ti.sync()
                np.testing.assert_allclose(
                    output.to_numpy()[0], expected, rtol=1e-5, atol=1e-4
                )


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_graph_reduction_recipe_composes_with_an_ordinary_dispatch():
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

    search = compileiq_recipe_search(graph)
    assert search.manifest()["families"] == ("graph_reduction",)
    assert len(search.recipe_ids) == 4

    host_values = ((np.arange(count, dtype=np.int32) % 19) - 9).astype(np.int32)
    input_array = ti.ndarray(ti.i32, shape=count)
    intermediate = ti.ndarray(ti.i32, shape=count)
    result_array = ti.ndarray(ti.i32, shape=1)
    input_array.from_numpy(host_values)
    expected = np.asarray(host_values.sum(dtype=np.int64), dtype=np.int32).item()

    with graph.definition.materialization_context() as context:
        for recipe_id in search.recipe_ids:
            parameters = {
                "domain_fingerprint": search.domain_fingerprint,
                "recipe_id": recipe_id,
            }
            with search.materialize(parameters, context=context) as materialized:
                search.verify_materialized_graph(parameters, materialized)
                candidate = materialized.executor
                result_array.fill(123)
                candidate.run(
                    candidate.bind(
                        {
                            "values": input_array,
                            "output": intermediate,
                            "result": result_array,
                        }
                    )
                )
                ti.sync()
                np.testing.assert_array_equal(intermediate.to_numpy(), host_values)
                assert result_array.to_numpy()[0] == expected


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_graph_reduction_generates_and_materializes_hierarchical_topology():
    count = 270337
    graph = _build_reduction_graph(ti.i32, count)
    search = compileiq_recipe_search(graph)
    assert len(search.recipe_ids) == 5
    hierarchical_id, hierarchical = next(
        (recipe_id, search.recipe_manifest(recipe_id)["reduction_recipe_manifest"])
        for recipe_id in search.recipe_ids
        if search.recipe_manifest(recipe_id)["reduction_recipe_manifest"]["strategy"]
        == "hierarchical_partial_finalize"
    )
    assert hierarchical["topology"] == {
        "kind": "hierarchical_partial_finalize",
        "block_dim": 256,
        "items_per_thread": 4,
        "levels": 3,
        "load": "scalar_coalesced",
        "in_block_reduction": "warp_shuffle_shared_finalize",
    }
    first_partial_count = ((count + 4 - 1) // 4 + 256 - 1) // 256
    second_partial_count = (first_partial_count + 256 * 4 - 1) // (256 * 4)
    assert hierarchical["workspace"] == {
        "ownership": "graph_instance",
        "exclusive_submission": True,
        "elements": first_partial_count + second_partial_count,
        "bytes": (first_partial_count + second_partial_count) * 4,
    }
    assert len(hierarchical["physical_stages"]) == 3

    values = ti.ndarray(ti.i32, shape=count)
    output = ti.ndarray(ti.i32, shape=1)
    host_values = ((np.arange(count, dtype=np.int32) % 13) - 6).astype(np.int32)
    values.from_numpy(host_values)
    output.fill(123)
    expected = np.asarray(host_values.sum(dtype=np.int64), dtype=np.int32).item()
    parameters = {
        "domain_fingerprint": search.domain_fingerprint,
        "recipe_id": hierarchical_id,
    }
    with search.materialize(parameters) as materialized:
        search.verify_materialized_graph(parameters, materialized)
        materialized.executor.run(
            materialized.executor.bind({"values": values, "output": output})
        )
        ti.sync()
        assert output.to_numpy()[0] == expected
        assert (
            materialized.manifest.persistent_requested_bytes
            == hierarchical["workspace"]["bytes"]
        )


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_graph_reduction_warp_recipe_is_correct_across_grid_stride_reuse():
    # B64/I2 exceeds the CUDA backend's resident grid cap at this size, so a
    # physical block must process multiple logical reduction groups.
    count = 1_048_577
    graph = _build_reduction_graph(ti.i32, count)
    search = compileiq_recipe_search(graph)
    recipe_id = next(
        recipe_id
        for recipe_id in search.recipe_ids
        if (
            search.recipe_manifest(recipe_id)["reduction_recipe_manifest"]["topology"][
                "block_dim"
            ]
            == 64
        )
    )
    values = ti.ndarray(ti.i32, shape=count)
    output = ti.ndarray(ti.i32, shape=1)
    host = ((np.arange(count, dtype=np.int64) % 23) - 11).astype(np.int32)
    values.from_numpy(host)
    expected = np.asarray(host.sum(dtype=np.int64), dtype=np.int32).item()
    parameters = {
        "domain_fingerprint": search.domain_fingerprint,
        "recipe_id": recipe_id,
    }
    with search.materialize(parameters) as materialized:
        candidate = materialized.executor
        binding = candidate.bind({"values": values, "output": output})
        for _ in range(32):
            candidate.run(binding)
        ti.sync()
        assert output.to_numpy()[0] == expected
