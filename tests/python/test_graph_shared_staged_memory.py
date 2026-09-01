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
