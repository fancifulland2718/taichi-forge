"""Vulkan FFT resources exported from ordinary and selected mixed Graphs."""

import gc
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils
from tests.python.test_hardware_vulkan_fft import _adapter, _input
from tests.python.test_graph_vulkan_fft_recipe import _providers


@pytest.mark.parametrize("selected", ["ordinary", "partition", "frame_partition"])
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_fft_recording_export(tmp_path, monkeypatch, selected):
    dimensions, batches = ((8,) if selected == "ordinary" else (3, 5)), 5
    data, values = _input(dimensions, batches)
    plans = [
        ti.hardware.fft.VulkanFftPlan(data, dimensions, batch_count=batches, adapter_path=_adapter()),
        ti.hardware.fft.VulkanFftPlan(
            data, dimensions, batch_count=batches, direction="inverse", normalization="inverse", adapter_path=_adapter()
        ),
    ]
    size = int(np.prod(dimensions))

    @ti.kernel
    def produce(x: ti.types.ndarray(ti.f32), data: ti.types.ndarray(ti.f32)):
        for i in ti.grouped(x):
            data[i] = 2 * x[i]

    @ti.kernel
    def save(data: ti.types.ndarray(ti.f32), spectrum: ti.types.ndarray(ti.f32)):
        for i in ti.grouped(data):
            spectrum[i] = data[i]

    @ti.kernel
    def finish(data: ti.types.ndarray(ti.f32), y: ti.types.ndarray(ti.f32), dx: ti.types.ndarray(ti.f32)):
        for i in ti.grouped(data):
            y[i] = 0.5 * data[i]
            # Explicit adjoint of FFT(2*x) for 0.5*||FFT(2*x)||^2.
            dx[i] = 2.0 * size * data[i]

    args = {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=len(data.shape))
        for name in ("x", "data", "spectrum", "y", "dx")
    }
    builder = ti.graph.GraphBuilder()
    builder.dispatch(produce, args["x"], args["data"])
    builder.append_native(plans[0].record())
    builder.dispatch(save, args["data"], args["spectrum"])
    builder.append_native(plans[1].record())
    builder.dispatch(finish, args["data"], args["y"], args["dx"])
    materialized = None
    if selected != "ordinary":
        definition = builder.freeze()
        catalog = definition.recipe_catalog(providers=_providers())
        tiles = [
            f
            for f in catalog.fragments
            if f.provider_metadata.get("family_selection", {}).get("choice_id") == "batch-tile-2"
        ]
        assert len(tiles) == 2
        if selected == "frame_partition":
            tiles.append(
                next(
                    f
                    for f in catalog.fragments
                    if f.provider_metadata.get("family_selection", {}).get("source_key") == "whole-graph-bindings"
                )
            )
        recipe = catalog.compose(tuple(f.fragment_id for f in tiles), stage="contract").recipe
        materialized = definition.materialize(recipe, providers=_providers())
        graph = materialized.executor
    else:
        graph = builder.compile()
    arrays = {name: ti.ndarray(ti.f32, data.shape) for name in args}
    arrays["data"] = data
    arrays["x"].from_numpy(values)
    graph.run(arrays)
    np.testing.assert_allclose(arrays["y"].to_numpy(), values, atol=1e-5)
    np.testing.assert_allclose(arrays["dx"].to_numpy(), 4 * size * values, atol=1e-4)
    module = ti.aot.Module()
    before = data.to_numpy()
    description = json.loads(module.add_execution_plan("step", graph).to_json())
    np.testing.assert_array_equal(data.to_numpy(), before)
    assert len(description["resources"]) == 2
    assert [r["parameters"]["direction"] for r in description["resources"]] == ["forward", "inverse"]
    assert {r["parameters"]["batch_tile"] for r in description["resources"]} == {
        batches if selected == "ordinary" else 2
    }
    assert "adapter_path" not in json.dumps(description)
    if selected == "frame_partition":

        class UnqualifiedExecutor(graph._spec._binding_executor_factory):
            pass

        rejected = ti.aot.Module()
        with monkeypatch.context() as patch:
            patch.setattr(graph._spec, "_binding_executor_factory", UnqualifiedExecutor)
            with pytest.raises(ti.TaichiRuntimeError, match="selected storage/binding recipe"):
                rejected.add_execution_plan("step", graph)
        assert not rejected._exported_kernels
    if materialized is not None:
        materialized.close()
    else:
        graph.close()
    del graph, builder
    gc.collect()
    for plan in plans:
        plan.close()
    artifact = tmp_path / "vkfft.tcm"
    module.archive(str(artifact))
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_vulkan_fft_probe.py")),
                mode,
                str(artifact),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_fft_export_rejects_unqualified_variant():
    class Variant(ti.hardware.fft.VulkanFftPlan):
        pass

    data, _ = _input((8,), 1)
    with Variant(data, (8,), adapter_path=_adapter()) as plan:
        builder = ti.graph.GraphBuilder()
        builder.append_native(plan.record())
        graph = builder.compile()
        module = ti.aot.Module()
        with pytest.raises(ti.TaichiRuntimeError, match="reconstruction contract"):
            module.add_execution_plan("step", graph)
        assert not module._exported_kernels
        graph.close()
