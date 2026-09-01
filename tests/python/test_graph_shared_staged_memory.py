from types import MappingProxyType

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge._lib import core as ti_core
from taichi_forge.lang import impl
from taichi_forge.lang.exception import TaichiCompilationError
from taichi_forge.lang._offload_execution_plan import (
    _OffloadExecutionPlan,
    _bind_offload_execution_plan,
)
from taichi_forge.graph import _graph as graph_impl
from taichi_forge.graph import compileiq_recipe_search
from tests import test_utils


def _shared_staged_plan(kernel, *probe_args, block_dim=128):
    baseline = _OffloadExecutionPlan.from_task_manifests(
        kernel.task_manifest(*probe_args)
    )
    ranges = tuple(task for task in baseline.tasks if task.task_kind == "range_for")
    assert len(ranges) == 1
    return baseline.replace_task(
        ranges[0].task_index,
        workgroup_size=block_dim,
        memory_strategy="shared_staged_1d",
    )


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_graph_memory_compileiq_recipe_reconstructs_complete_direct_and_staged(
    monkeypatch,
):
    count = 1027

    @ti.kernel
    def stencil(
        source: ti.types.ndarray(dtype=ti.f32, ndim=1),
        output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in range(1, count - 1):
            output[i] = source[i - 1] + source[i] * 2.0 + source[i + 1]

    source_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "source", ti.f32, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(stencil, source_arg, output_arg)
    source_graph = builder.compile()

    search = compileiq_recipe_search(source_graph)
    assert source_graph._compileiq_graph_memory_status == "complete_recipe_domain"
    assert len(search.recipe_ids) == 2
    assert search.search_space.provider_namespace == "taichi_forge.graph.memory"
    assert search.search_space.domain_version == "graph-memory-complete-recipe"
    assert search.manifest()["recipe_kind"] == "graph_memory"
    manifests = {
        recipe_id: search.recipe_manifest(recipe_id) for recipe_id in search.recipe_ids
    }
    direct_id = next(
        recipe_id
        for recipe_id, manifest in manifests.items()
        if manifest["memory_recipe_manifest"]["strategy"] == "direct"
    )
    staged_id = next(
        recipe_id
        for recipe_id, manifest in manifests.items()
        if manifest["memory_recipe_manifest"]["strategy"] == "shared_staged_1d"
    )
    assert direct_id == search.baseline_recipe_id
    assert not manifests[direct_id]["fusion_recipe_ids"]
    assert not manifests[staged_id]["fusion_recipe_ids"]
    assert not manifests[direct_id].get("control_recipe_id")
    assert not manifests[staged_id].get("control_recipe_id")

    def parameters(recipe_id):
        return {
            "domain_fingerprint": search.domain_fingerprint,
            "recipe_id": recipe_id,
        }

    def rebuild(recipe_id):
        environment = search.worker_environment(parameters(recipe_id))
        assert environment["TAICHI_FORGE_INTERNAL_MAP_FUSION"] == "baseline"
        assert environment["TAICHI_FORGE_INTERNAL_STRUCTURED_CONTROL_RECIPE"] == "auto"
        assert (
            environment["TAICHI_FORGE_INTERNAL_GRAPH_MEMORY_RECIPE"]
            == manifests[recipe_id]["memory_recipe_id"]
        )
        with monkeypatch.context() as reconstruction:
            for name, value in environment.items():
                reconstruction.setenv(name, value)
            rebuilt_builder = ti.graph.GraphBuilder()
            rebuilt_builder.dispatch(stencil, source_arg, output_arg)
            rebuilt = rebuilt_builder.compile()
        search.verify_materialized_graph(parameters(recipe_id), rebuilt)
        return rebuilt

    observed = []

    def objective(opaque_parameters):
        selection = search.select(opaque_parameters)
        rebuilt = rebuild(selection.spec_id)
        observed.append(
            (
                selection.spec_id,
                rebuilt._compileiq_executable_optimization_space.selected_spec_id,
            )
        )
        return float(search.recipe_ids.index(selection.spec_id))

    exhaustive = search.compileiq_search(objective)
    result = exhaustive.start()
    coverage = search.require_complete_search(exhaustive)
    selected = search.select_best_result(exhaustive, result)
    assert coverage["complete"]
    assert coverage["evaluation_count"] == 2
    assert {item[0] for item in observed} == set(search.recipe_ids)
    assert all(requested == actual for requested, actual in observed)
    assert selected.spec_id == search.recipe_ids[0]

    direct_graph = rebuild(direct_id)
    staged_graph = rebuild(staged_id)
    assert (
        direct_graph._compileiq_executable_optimization_space.semantic_plan_id
        == staged_graph._compileiq_executable_optimization_space.semantic_plan_id
    )
    staged_task = next(
        task for task in staged_graph.task_manifest() if task.task_type == "range_for"
    )
    assert staged_task.requested_memory_strategy == "shared_staged_1d"
    assert staged_task.range_mapping == "shared_tiled_one_to_one"
    assert staged_task.selected_block_size == 128

    source = ti.ndarray(ti.f32, shape=count)
    output = ti.ndarray(ti.f32, shape=count)
    values = np.arange(count, dtype=np.float32) * 0.25
    source.from_numpy(values)
    output.fill(0)
    bindings = staged_graph.bind({"source": source, "output": output})
    assert bindings.fast_path_qualified
    staged_graph.run(bindings)
    ti.sync()
    expected = np.zeros(count, dtype=np.float32)
    expected[1:-1] = values[:-2] + values[1:-1] * 2.0 + values[2:]
    np.testing.assert_allclose(output.to_numpy(), expected, rtol=0, atol=0)

    with pytest.raises(RuntimeError, match="requires proven disjoint storage"):
        staged_graph.bind({"source": source, "output": source})
    short_source = ti.ndarray(ti.f32, shape=count - 1)
    short_output = ti.ndarray(ti.f32, shape=count - 1)
    with pytest.raises(RuntimeError, match="at least 1027 scalar elements"):
        staged_graph.bind({"source": short_source, "output": short_output})

    with monkeypatch.context() as reconstruction:
        reconstruction.setenv(
            "TAICHI_FORGE_INTERNAL_GRAPH_MEMORY_RECIPE",
            "graph-memory:shared-staged-1d:" + "0" * 24,
        )
        with pytest.raises(RuntimeError, match="absent from this Graph definition"):
            rejected = ti.graph.GraphBuilder()
            rejected.dispatch(stencil, source_arg, output_arg)


@pytest.mark.parametrize(
    ("dtype", "numpy_dtype", "element_bytes"),
    ((ti.f16, np.float16, 2), (ti.f64, np.float64, 8)),
)
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_graph_memory_compileiq_supports_two_and_eight_byte_scalar_stencils(
    monkeypatch,
    dtype,
    numpy_dtype,
    element_bytes,
):
    count = 1027

    @ti.kernel
    def stencil(
        source: ti.types.ndarray(dtype=dtype, ndim=1),
        output: ti.types.ndarray(dtype=dtype, ndim=1),
    ):
        for i in range(1, count - 1):
            output[i] = source[i - 1] + source[i] * 2 + source[i + 1]

    source_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "source", dtype, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", dtype, ndim=1)

    def build():
        builder = ti.graph.GraphBuilder()
        builder.dispatch(stencil, source_arg, output_arg)
        return builder.compile()

    source_graph = build()
    search = compileiq_recipe_search(source_graph)
    manifests = {
        recipe_id: search.recipe_manifest(recipe_id) for recipe_id in search.recipe_ids
    }
    assert len(manifests) == 2, source_graph._compileiq_graph_memory_status
    staged_id, staged_manifest = next(
        (recipe_id, manifest)
        for recipe_id, manifest in manifests.items()
        if manifest["memory_recipe_manifest"]["strategy"] == "shared_staged_1d"
    )
    layout_requirements = {
        tuple(requirement)
        for requirement in staged_manifest["memory_recipe_manifest"][
            "memory_layout_requirements"
        ]
    }
    assert layout_requirements == {
        ("source", count, element_bytes, element_bytes),
        ("output", count - 1, element_bytes, element_bytes),
    }

    parameters = {
        "domain_fingerprint": search.domain_fingerprint,
        "recipe_id": staged_id,
    }
    selection = search.select(parameters)
    with monkeypatch.context() as reconstruction:
        for name, value in selection.worker_environment.items():
            reconstruction.setenv(name, value)
        staged_graph = build()
    search.verify_materialized_graph(parameters, staged_graph)
    staged_task = next(
        task for task in staged_graph.task_manifest() if task.task_type == "range_for"
    )
    assert staged_task.requested_memory_strategy == "shared_staged_1d"
    assert staged_task.range_mapping == "shared_tiled_one_to_one"
    assert staged_task.static_shared_bytes == (128 + 2) * element_bytes

    source = ti.ndarray(dtype, shape=count)
    output = ti.ndarray(dtype, shape=count)
    values = (np.arange(count, dtype=np.int64) % 17).astype(numpy_dtype)
    source.from_numpy(values)
    output.fill(0)
    bindings = staged_graph.bind({"source": source, "output": output})
    assert bindings.fast_path_qualified
    staged_graph.run(bindings)
    ti.sync()
    expected = np.zeros(count, dtype=numpy_dtype)
    expected[1:-1] = values[:-2] + values[1:-1] * 2 + values[2:]
    np.testing.assert_array_equal(output.to_numpy(), expected)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_graph_memory_compileiq_keeps_one_byte_scalar_stencils_out_of_domain():
    count = 257

    @ti.kernel
    def stencil(
        source: ti.types.ndarray(dtype=ti.i8, ndim=1),
        output: ti.types.ndarray(dtype=ti.i8, ndim=1),
    ):
        for i in range(1, count - 1):
            output[i] = source[i - 1] + source[i + 1]

    source_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "source", ti.i8, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.i8, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(stencil, source_arg, output_arg)
    graph = builder.compile()
    search = compileiq_recipe_search(graph)

    assert len(search.recipe_ids) == 1
    assert "only two-, four-, or eight-byte scalar elements are supported" in (
        graph._compileiq_graph_memory_status
    )


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_graph_memory_compileiq_rejects_unsupported_and_multi_dispatch_domains(
    monkeypatch,
):
    count = 256

    @ti.kernel
    def pointwise(
        source: ti.types.ndarray(dtype=ti.f32, ndim=1),
        output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in range(count):
            output[i] = source[i] * 2.0

    source_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "source", ti.f32, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(pointwise, source_arg, output_arg)
    graph = builder.compile()
    search = compileiq_recipe_search(graph)

    assert graph._compileiq_graph_memory_status.startswith("candidate_rejected:")
    assert len(search.recipe_ids) == 1
    assert all(
        "memory_recipe_id" not in search.recipe_manifest(recipe_id)
        for recipe_id in search.recipe_ids
    )

    with monkeypatch.context() as reconstruction:
        reconstruction.setenv(
            "TAICHI_FORGE_INTERNAL_GRAPH_MEMORY_RECIPE",
            "graph-memory:shared-staged-1d:" + "0" * 24,
        )
        with pytest.raises(RuntimeError, match="cannot be materialized"):
            rejected = ti.graph.GraphBuilder()
            rejected.dispatch(pointwise, source_arg, output_arg)

    multi = ti.graph.GraphBuilder()
    multi.dispatch(pointwise, source_arg, output_arg)
    multi.dispatch(pointwise, source_arg, output_arg)
    multi_graph = multi.compile()
    multi_search = compileiq_recipe_search(multi_graph)
    assert multi_graph._compileiq_graph_memory_status == "definition_out_of_scope"
    assert all(
        "memory_recipe_id" not in multi_search.recipe_manifest(recipe_id)
        for recipe_id in multi_search.recipe_ids
    )


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_private_graph_shared_staged_recipe_materializes_and_replays_exactly(
    monkeypatch,
):
    count = 1027

    @ti.kernel
    def stencil(
        source: ti.types.ndarray(dtype=ti.f32, ndim=1),
        output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in range(1, count - 1):
            output[i] = source[i - 1] + source[i] * 2.0 + source[i + 1]

    source = ti.ndarray(ti.f32, shape=count)
    output = ti.ndarray(ti.f32, shape=count)
    values = np.arange(count, dtype=np.float32) * 0.25
    source.from_numpy(values)
    plan = _shared_staged_plan(stencil, source, output)
    bound = _bind_offload_execution_plan(stencil, plan)

    manifest = next(
        task
        for task in bound.task_manifest(source, output)
        if task.task_type == "range_for"
    )
    assert manifest.requested_memory_strategy == "shared_staged_1d"
    assert manifest.range_mapping == "shared_tiled_one_to_one"
    assert manifest.selected_block_size == 128
    assert manifest.selected_grid_size == (count - 2 + 127) // 128
    assert manifest.staged_external_arg_index == 0
    assert (manifest.staged_halo_low, manifest.staged_halo_high) == (-1, 1)
    assert manifest.static_shared_bytes == (128 + 2) * 4

    source_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "source", ti.f32, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder._dispatch_shared_staged_1d(bound, source_arg, output_arg)
    graph = builder.compile()
    graph._graph_stats

    alias_checks = 0
    description_calls = 0
    owner_validation_calls = 0
    original_alias_check = graph_impl.analyze_storage_alias
    original_describe_storage = graph_impl.describe_storage
    original_validate_storage_owner = graph_impl.validate_storage_owner

    def counted_alias_check(*args, **kwargs):
        nonlocal alias_checks
        alias_checks += 1
        return original_alias_check(*args, **kwargs)

    def counted_describe_storage(*args, **kwargs):
        nonlocal description_calls
        description_calls += 1
        return original_describe_storage(*args, **kwargs)

    def counted_validate_storage_owner(*args, **kwargs):
        nonlocal owner_validation_calls
        owner_validation_calls += 1
        return original_validate_storage_owner(*args, **kwargs)

    monkeypatch.setattr(graph_impl, "analyze_storage_alias", counted_alias_check)
    monkeypatch.setattr(graph_impl, "describe_storage", counted_describe_storage)
    monkeypatch.setattr(
        graph_impl, "validate_storage_owner", counted_validate_storage_owner
    )

    binding_plan = graph.binding_plan()
    assert binding_plan["memory_recipe_names"] == ("output", "source")
    assert binding_plan["memory_recipe_publish_certificate_required"]
    assert binding_plan["memory_recipe_publish_frame_stable"]
    bindings = graph.bind({"source": source, "output": output})
    binding_stats = bindings.statistics()
    assert bindings.fast_path_qualified
    assert binding_stats["memory_recipe_publish_validated"]
    assert binding_stats["memory_recipe_certified"]
    assert binding_stats["memory_recipe_names"] == ("output", "source")
    assert "dynamic_memory_recipe" not in binding_stats["volatile_reasons"]
    publish_description_calls = description_calls
    publish_owner_validation_calls = owner_validation_calls
    assert publish_description_calls == 2
    assert publish_owner_validation_calls == 2
    assert alias_checks == 1

    graph.run(bindings)
    ti.sync()
    expected = np.zeros(count, dtype=np.float32)
    expected[1:-1] = values[:-2] + values[1:-1] * 2.0 + values[2:]
    np.testing.assert_allclose(output.to_numpy(), expected, rtol=0, atol=0)

    first = graph._graph_stats[0]
    output.fill(0)
    graph.run(bindings)
    ti.sync()
    second = graph._graph_stats[0]
    assert first["captures"] == 1
    assert second["exact_replays"] == 1
    np.testing.assert_allclose(output.to_numpy(), expected, rtol=0, atol=0)

    for _ in range(16):
        graph.run(bindings)
    ti.sync()
    graph_identity = graph._instance_debug_info
    program = impl.get_runtime().prog
    runtime_before = program._runtime_statistics_snapshot()["memory"]
    host_before = dict(ti_core.get_host_memory_pool_stats())
    device_before = dict(ti_core.get_device_memory_pool_stats())
    for _ in range(10_000):
        graph.run(bindings)
    ti.sync()
    assert alias_checks == 1
    assert description_calls == publish_description_calls
    assert owner_validation_calls == publish_owner_validation_calls
    assert graph.binding_statistics()["version_fast_replays"] >= 10_018
    assert graph._instance_debug_info == graph_identity
    assert graph._graph_stats[0]["exact_replays"] >= 10_001
    runtime_after = program._runtime_statistics_snapshot()["memory"]
    for key in (
        "host_requested_live_bytes",
        "host_raw_bytes",
        "device_requested_live_bytes",
        "device_raw_bytes",
        "device_cached_bytes",
    ):
        if runtime_before[key] is not None and runtime_after[key] is not None:
            assert runtime_after[key] <= runtime_before[key]
    for before, after in (
        (host_before, dict(ti_core.get_host_memory_pool_stats())),
        (device_before, dict(ti_core.get_device_memory_pool_stats())),
    ):
        for key in (
            "raw_chunks",
            "requested_live_bytes",
            "raw_bytes",
            "reserved_bytes",
            "committed_bytes",
            "used_bytes",
            "cached_blocks",
            "cached_bytes",
        ):
            if key in before and key in after:
                assert after[key] <= before[key]

    # Recurring resource sets can each be published once. Switching A -> B -> A
    # then reuses both immutable Python certificates; the native CGraph cache
    # independently reuses its generation-qualified resource plans.
    alternate_values = values[::-1].copy()
    alternate_source = ti.ndarray(ti.f32, shape=count)
    alternate_output = ti.ndarray(ti.f32, shape=count)
    alternate_source.from_numpy(alternate_values)
    alternate_bindings = graph.bind(
        {"source": alternate_source, "output": alternate_output}
    )
    assert alternate_bindings.fast_path_qualified
    recurring_description_calls = description_calls
    recurring_owner_validation_calls = owner_validation_calls
    recurring_alias_checks = alias_checks

    output.fill(0)
    alternate_output.fill(0)
    graph.run(bindings)
    graph.run(alternate_bindings)
    graph.run(bindings)
    ti.sync()

    assert description_calls == recurring_description_calls
    assert owner_validation_calls == recurring_owner_validation_calls
    assert alias_checks == recurring_alias_checks
    np.testing.assert_allclose(output.to_numpy(), expected, rtol=0, atol=0)
    alternate_expected = np.zeros(count, dtype=np.float32)
    alternate_expected[1:-1] = (
        alternate_values[:-2] + alternate_values[1:-1] * 2.0 + alternate_values[2:]
    )
    np.testing.assert_allclose(
        alternate_output.to_numpy(), alternate_expected, rtol=0, atol=0
    )

    # Mutable compatibility dictionaries deliberately keep one exact owner
    # scan per replay. A new descriptor tuple proves aliasing once, then the
    # collision-free cache skips only the exhaustive alias/layout analysis.
    raw_source = ti.ndarray(ti.f32, shape=count)
    raw_output = ti.ndarray(ti.f32, shape=count)
    raw_source.from_numpy(values)
    raw_descriptions_before = description_calls
    raw_owner_validations_before = owner_validation_calls
    raw_alias_checks_before = alias_checks
    graph.run({"source": raw_source, "output": raw_output})
    graph.run({"source": raw_source, "output": raw_output})
    ti.sync()
    assert description_calls - raw_descriptions_before == 4
    assert owner_validation_calls - raw_owner_validations_before == 4
    assert alias_checks - raw_alias_checks_before == 1

    ti.reset()
    with pytest.raises(RuntimeError, match="compiled before ti.reset"):
        graph.run(bindings)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_private_graph_shared_staged_recipe_is_graph_owned_and_alias_safe():
    count = 257

    @ti.kernel
    def stencil(
        source: ti.types.ndarray(dtype=ti.f32, ndim=1),
        output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in range(1, count - 1):
            output[i] = source[i - 1] + source[i + 1]

    source = ti.ndarray(ti.f32, shape=count)
    output = ti.ndarray(ti.f32, shape=count)
    bound = _bind_offload_execution_plan(
        stencil, _shared_staged_plan(stencil, source, output)
    )
    source_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "source", ti.f32, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1)

    with pytest.raises(RuntimeError, match="Graph-owned"):
        bound(source, output)
    with pytest.raises(TaichiCompilationError, match="Graph-owned memory recipe"):
        ti.graph.GraphBuilder().dispatch(bound, source_arg, output_arg)

    builder = ti.graph.GraphBuilder()
    builder._dispatch_shared_staged_1d(bound, source_arg, output_arg)
    graph = builder.compile()
    bindings = graph.bind({"source": source, "output": output})
    assert bindings.fast_path_qualified
    replacement_source = ti.ndarray(ti.f32, shape=count)
    replacement_values = np.linspace(-2.0, 3.0, count, dtype=np.float32)
    replacement_source.from_numpy(replacement_values)
    bindings.update(source=replacement_source)
    assert bindings.fast_path_qualified
    assert bindings.statistics()["memory_recipe_certified"]
    output.fill(0)
    graph.run(bindings)
    ti.sync()
    expected = np.zeros(count, dtype=np.float32)
    expected[1:-1] = replacement_values[:-2] + replacement_values[2:]
    np.testing.assert_allclose(output.to_numpy(), expected, rtol=0, atol=0)

    revision = bindings.revision
    with pytest.raises(RuntimeError, match="requires proven disjoint storage"):
        bindings.update(output=replacement_source)
    assert bindings.revision == revision
    output.fill(0)
    graph.run(bindings)
    ti.sync()
    np.testing.assert_allclose(output.to_numpy(), expected, rtol=0, atol=0)

    with pytest.raises(RuntimeError, match="requires proven disjoint storage"):
        graph.run({"source": source, "output": source})

    short_source = ti.ndarray(ti.f32, shape=count - 1)
    short_output = ti.ndarray(ti.f32, shape=count - 1)
    with pytest.raises(RuntimeError, match="at least 257 scalar elements"):
        graph.bind({"source": short_source, "output": short_output})


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_private_graph_shared_staged_recipe_validates_final_provider_bindings():
    count = 257

    @ti.kernel
    def stencil(
        source: ti.types.ndarray(dtype=ti.f32, ndim=1),
        output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in range(1, count - 1):
            output[i] = source[i - 1] + source[i + 1]

    source = ti.ndarray(ti.f32, shape=count)
    output = ti.ndarray(ti.f32, shape=count)
    bound = _bind_offload_execution_plan(
        stencil, _shared_staged_plan(stencil, source, output)
    )
    source_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "source", ti.f32, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder._dispatch_shared_staged_1d(bound, source_arg, output_arg)
    graph = builder.compile()

    class _AliasingProvider:
        def __init__(self):
            self.binding = None

        def bind_graph_arguments(self, runtime_args):
            alias = graph_impl.ProviderOwnedNdarrayBinding(
                runtime_args["source"].arr,
                self,
            )
            self.binding = alias
            return graph_impl.PreparedGraphBindings(
                MappingProxyType({"output": alias}),
                (self,),
            )

    # A dynamic provider is normally contributed by a mixed native segment.
    # Inject its production binding result here so this shared-stage-only
    # fixture proves that memory contracts validate the final provider-owned
    # frame, after replacements and with its exact submission owner attached.
    provider = _AliasingProvider()
    graph._spec.lifetime_leases = (provider,)
    with pytest.raises(RuntimeError, match="requires proven disjoint storage"):
        graph.run({"source": source, "output": output})
    assert provider.binding is not None
    with pytest.raises(AttributeError):
        provider.binding.arr = output.arr


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_private_graph_shared_staged_recipe_snapshots_mapping_proxy(monkeypatch):
    count = 257

    @ti.kernel
    def stencil(
        source: ti.types.ndarray(dtype=ti.f32, ndim=1),
        output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in range(1, count - 1):
            output[i] = source[i - 1] + source[i + 1]

    source = ti.ndarray(ti.f32, shape=count)
    output = ti.ndarray(ti.f32, shape=count)
    values = np.arange(count, dtype=np.float32)
    source.from_numpy(values)
    output.fill(0)
    bound = _bind_offload_execution_plan(
        stencil, _shared_staged_plan(stencil, source, output)
    )
    source_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "source", ti.f32, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder._dispatch_shared_staged_1d(bound, source_arg, output_arg)
    graph = builder.compile()

    backing = {"source": source, "output": output}
    arguments = MappingProxyType(backing)
    original_validate = graph._spec._validate_bound_runtime_args

    def validate_then_mutate_backing(validation_args, **kwargs):
        certificate = original_validate(validation_args, **kwargs)
        backing["output"] = source
        return certificate

    monkeypatch.setattr(
        graph._spec,
        "_validate_bound_runtime_args",
        validate_then_mutate_backing,
    )
    graph.run(arguments)
    ti.sync()

    assert backing["output"] is source
    expected = np.zeros(count, dtype=np.float32)
    expected[1:-1] = values[:-2] + values[2:]
    np.testing.assert_allclose(output.to_numpy(), expected, rtol=0, atol=0)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_private_graph_shared_staged_recipe_rejects_pointwise_input():
    count = 256

    @ti.kernel
    def pointwise(
        source: ti.types.ndarray(dtype=ti.f32, ndim=1),
        output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in range(count):
            output[i] = source[i] * 2.0

    source = ti.ndarray(ti.f32, shape=count)
    output = ti.ndarray(ti.f32, shape=count)
    plan = _shared_staged_plan(pointwise, source, output)
    with pytest.raises(RuntimeError, match="at least two distinct affine offsets"):
        _bind_offload_execution_plan(pointwise, plan).task_manifest(source, output)
