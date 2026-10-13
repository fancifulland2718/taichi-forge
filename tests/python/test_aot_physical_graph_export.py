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
def test_aot_selected_storage_is_not_silently_replaced(tmp_path):
    from tests.python.test_graph_resource_lifetime_recipe import _definition, _providers

    definition = _definition(groups=1, size=17)
    catalog = definition.recipe_catalog(providers=_providers())
    recipe = catalog.composer.compose((catalog.fragments[0],))
    module = ti.aot.Module()
    with definition.materialization_context(provider_set=catalog.provider_set) as context:
        with context.materialize(recipe) as materialized:
            assert materialized.executor.execution_stats().storage_pools
            with pytest.raises(ti.TaichiRuntimeError, match="storage/binding recipe"):
                module.add_execution_plan("invalid", materialized.executor)
            assert not module._content and not module._exported_kernels and not module._execution_plans
            assert not module._graph_export_failed
