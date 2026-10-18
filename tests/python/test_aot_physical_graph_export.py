"""Persist selected physical kernels without resurrecting their source maps."""

import gc
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_aot_selected_maps_forward_adjoint_and_alias(tmp_path):
    from taichi_forge.graph._recipes.families import GraphRuntimeAssemblyProvider
    from taichi_forge.graph._recipes.map_fusion import GraphMapFusionRecipeProvider

    @ti.kernel
    def scale(x: ti.types.ndarray(ti.f32, ndim=1), t: ti.types.ndarray(ti.f32, ndim=1), gain: ti.f32):
        for i in range(17):
            t[i] = gain * x[i]

    @ti.kernel
    def square(t: ti.types.ndarray(ti.f32, ndim=1), y: ti.types.ndarray(ti.f32, ndim=1), bias: ti.f32):
        for i in range(17):
            y[i] = t[i] * t[i] + bias

    @ti.kernel
    def seed(y: ti.types.ndarray(ti.f32, ndim=1), dy: ti.types.ndarray(ti.f32, ndim=1)):
        for i in range(17):
            dy[i] = 2 * y[i]

    @ti.kernel
    def backward(
        t: ti.types.ndarray(ti.f32, ndim=1),
        dy: ti.types.ndarray(ti.f32, ndim=1),
        dx: ti.types.ndarray(ti.f32, ndim=1),
        gain: ti.f32,
    ):
        for i in range(17):
            dx[i] = 2 * gain * t[i] * dy[i]

    @ti.kernel
    def bias(left: ti.types.ndarray(ti.f32, ndim=1), right: ti.types.ndarray(ti.f32, ndim=1), value: ti.f32):
        for i in range(17):
            left[i] = right[i] + value

    symbols = {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1) for name in ("x", "t", "y", "dy", "dx")
    }
    gain = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "gain", ti.f32)
    offset = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "bias", ti.f32)
    builder = ti.graph.GraphBuilder(_map_recipe="map2")
    builder.dispatch(scale, symbols["x"], symbols["t"], gain)
    builder.dispatch(square, symbols["t"], symbols["y"], offset)
    builder.dispatch(seed, symbols["y"], symbols["dy"])
    builder.dispatch(backward, symbols["t"], symbols["dy"], symbols["dx"], gain)
    graph = builder.compile()
    assert graph.physical_plan()["physical_dispatch_count"] == 2
    module = ti.aot.Module()
    plan = module.add_execution_plan("step", graph)
    operations = json.loads(plan.to_json())["operations"]
    assert len(operations) == len(module._exported_kernels) == 2
    assert len({op["entry"] for op in operations}) == 2
    module.add_execution_plan("again", graph)
    assert len(module._exported_kernels) == 2
    graph.close()
    del graph

    # Exercise explicit catalog selection as well as the convenience map recipe.
    alias_builder = ti.graph.GraphBuilder()
    alias_builder.dispatch(scale, symbols["x"], symbols["t"], gain)
    alias_builder.dispatch(bias, symbols["x"], symbols["t"], offset)
    definition = alias_builder.freeze()
    providers = (GraphRuntimeAssemblyProvider(), GraphMapFusionRecipeProvider())
    catalog = definition.recipe_catalog(providers=providers)
    recipe = next(entry.recipe for entry in catalog.entries() if entry.recipe.fragments)
    with definition.materialization_context(provider_set=catalog.provider_set) as context:
        with context.materialize(recipe) as materialized:
            assert len(materialized.manifest.kernels) == 1
            module.add_execution_plan("alias", materialized.executor)
    del materialized, context, definition, catalog, recipe
    gc.collect()
    assert len(module._exported_kernels) == 3
    artifact = tmp_path / "physical.tcm"
    module.archive(str(artifact))
    arch = "cpu" if ti.lang.impl.current_cfg().arch == ti.cpu else str(ti.lang.impl.current_cfg().arch).split(".")[-1]
    for mode in ("standalone", "borrowed"):
        command = [sys.executable] + (["-S"] if mode == "standalone" else [])
        result = subprocess.run(
            [*command, str(Path(__file__).with_name("aot_physical_graph_probe.py")), mode, arch, str(artifact)],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_aot_selected_private_storage_preserves_pool_topology(tmp_path):
    import array
    from dataclasses import replace
    from taichi_forge_runtime.aot import Runtime, Scalar
    from tests.python.test_graph_resource_lifetime_recipe import _definition, _providers

    definition = _definition(groups=2, size=17)
    catalog = definition.recipe_catalog(providers=_providers())
    separate = tuple(fragment for fragment in catalog.fragments if len(fragment.coverage_region_ids) == 2)
    recipe = catalog.composer.compose(separate)
    module = ti.aot.Module()
    with definition.materialization_context(provider_set=catalog.provider_set) as context:
        with context.materialize(recipe) as materialized:
            graph = materialized.executor
            assert len(graph.execution_stats().storage_pools) == 2
            plan = module.add_execution_plan("step", graph)
            document = json.loads(plan.to_json())
            pools = [r for r in document["resources"] if r["kind"] == "native" and r["provider"] == "memory"]
            arrays = [r for r in document["resources"] if r["kind"] == "ndarray"]
            assert len(pools) == len(arrays) == 2
            assert {a["allocator"] for a in arrays} == {p["name"] for p in pools}
            assert document["version"] == 19
            assert plan.argument_names == ("output_0", "output_1", "value_0", "value_1")
            artifact = tmp_path / "private_pools.tcm"
            module.archive(str(artifact))

            # Unknown allocation policies still fail before publishing any code.
            original = graph._spec._storage_plans
            graph._spec._storage_plans = tuple(replace(p, factory=lambda: None) for p in original)
            rejected = ti.aot.Module()
            try:
                with pytest.raises(ti.TaichiRuntimeError, match="storage/binding recipe"):
                    rejected.add_execution_plan("invalid", graph)
                assert not rejected._content and not rejected._exported_kernels and not rejected._execution_plans
                assert not rejected._graph_export_failed
            finally:
                graph._spec._storage_plans = original

    # Producer generations/pools are now closed. Both consumer ownership modes
    # reconstruct their own pools from the same package.
    for make_runtime in (lambda: Runtime("cuda"), ti.aot.borrow_runtime):
        with make_runtime() as runtime:
            loaded = runtime.load_module(artifact)
            outputs = [runtime.ndarray("i32", 17) for _ in range(4)]
            frames = [loaded.prepare_execution("step", {
                "output_0": outputs[2 * index], "output_1": outputs[2 * index + 1],
                "value_0": Scalar("i32", index + 1), "value_1": Scalar("i32", index + 9),
            }) for index in range(2)]
            assert all(frame.info["resource_bytes"] == 136 for frame in frames)
            for index, frame in enumerate(frames):
                frame()
                for group in range(2):
                    actual = array.array("i")
                    actual.frombytes(outputs[2 * index + group].read())
                    value = index + (1 if group == 0 else 9)
                    assert actual.tolist() == [(value + i * 3) * 2 + 7 for i in range(17)]
            frames[0]()
            frames[0].close()
            frames[1]()
            loaded.close()
            frames[1].close()


@pytest.mark.parametrize("dtype", [ti.i32, ti.f32])
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_aot_selected_ordered_temporary_storage_replays_without_callbacks(tmp_path, monkeypatch, dtype):
    import array
    from taichi_forge_runtime.aot import Runtime, Scalar
    from tests.python.test_graph_resource_lifetime_recipe import _TaggedTemporaryWork, _TemporaryAction, _providers

    builder = ti.graph.GraphBuilder()
    builder.append_native(_TaggedTemporaryWork(257, dtype))
    definition = builder.freeze()
    catalog = definition.recipe_catalog(providers=_providers())
    fragment = next(f for f in catalog.fragments if f.fragment_key.endswith(":queue-ordered-storage"))
    recipe = catalog.composer.compose((fragment,))
    module = ti.aot.Module()
    with definition.materialization_context(provider_set=catalog.provider_set) as context:
        with context.materialize(recipe) as materialized:
            def forbidden(*args, **kwargs):
                raise AssertionError("AOT export reran a resolved temporary callback")

            with monkeypatch.context() as cold:
                cold.setattr(_TemporaryAction, "bind_graph_temporaries", forbidden)
                plan = module.add_execution_plan("step", materialized.executor)
            document = json.loads(plan.to_json())
            arrays = [r for r in document["resources"] if r["kind"] == "ndarray"]
            assert len(arrays) == 1 and arrays[0]["shape"] == [257]
            assert arrays[0]["dtype"] == str(dtype)
            assert not arrays[0].get("zero_initialize")
            assert len(document["operations"]) == 2 and len(module._exported_kernels) == 2
            assert plan.argument_names == ("output", "tag")
            artifact = tmp_path / "ordered_temporary.tcm"
            module.archive(str(artifact))

    for make_runtime in (lambda: Runtime("cuda"), ti.aot.borrow_runtime):
        with make_runtime() as runtime:
            loaded = runtime.load_module(artifact)
            outputs = [runtime.ndarray(str(dtype), 257) for _ in range(3)]
            frames = [loaded.prepare_execution("step", dict(output=output, tag=Scalar("i32", tag)))
                      for output, tag in zip(outputs, (-3, 19, 41))]
            assert all(frame.info["resource_bytes"] == 1028 for frame in frames)
            for _ in range(5):
                for frame in reversed(frames):
                    frame()
            for output, tag in zip(outputs, (-3, 19, 41)):
                actual = array.array("f" if dtype == ti.f32 else "i")
                actual.frombytes(output.read())
                assert actual.tolist() == [(i * 3 + tag) * 2 + 5 for i in range(257)]
            frames[0]()
            frames[0].close()
            frames[1]()
            loaded.close()
            for frame in frames[1:]:
                frame.close()
