"""Selected FFT plans retain decomposition, scaling, and explicit adjoints."""

import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.aot_fft_recipe_probe import case_info, reference


@pytest.mark.parametrize("case", ("forward", "inverse", "real", "callback_forward", "callback_inverse"))
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_selected_fft_export_and_adjoints(case, tmp_path, monkeypatch):
    from taichi_forge.hardware import _fft_lto

    directory = os.environ.get("TI_TEST_CUFFT_DIRECTORY")
    dll = os.add_dll_directory(directory) if directory and os.name == "nt" else None
    if not ti.hardware.fft.is_available():
        pytest.skip("external cuFFT required")
    callback = case.startswith("callback_")
    if callback:
        libraries = {
            name: os.environ.get(f"TI_TEST_{name.upper()}_LIBRARY_PATH") for name in ("nvrtc", "nvjitlink")
        }
        if not all(libraries.values()):
            pytest.skip("explicit NVRTC and nvJitLink required")
    else:
        monkeypatch.setattr(
            _fft_lto, "_library", lambda *_args, **_kwargs: pytest.fail("plain FFT loaded a JIT compiler")
        )
    dimensions, batch, scale, shapes = case_info(case)
    real = case == "real"
    width = dimensions[-1]
    axis = len(shapes["y"]) - 2
    forward = ti.linalg.record_fft(
        dimensions,
        batch_count=batch,
        transform="r2c" if real else "c2c",
        direction="inverse" if case.endswith("inverse") else "forward",
        output_scale=scale,
        input="x",
        output="y",
        absolute_tolerance=5e-5,
        relative_tolerance=5e-5,
    )
    backward = ti.linalg.record_fft(
        dimensions,
        batch_count=batch,
        transform="c2r" if real else "c2c",
        direction="forward" if case.endswith("inverse") else "inverse",
        output_scale=scale,
        input="adjoint",
        output="dx",
        absolute_tolerance=5e-5,
        relative_tolerance=5e-5,
    )
    options = (
        dict(lto_callbacks=True, nvrtc_library=libraries["nvrtc"], nvjitlink_library=libraries["nvjitlink"])
        if callback
        else {}
    )
    forward.prepare(**options)
    backward.prepare(**options)

    @ti.func
    def weight(column):
        result = 1.0
        if ti.static(real):
            if column != 0 and 2 * column != width:
                result = 2.0
        return result

    @ti.kernel
    def seed(dy: ti.types.ndarray(ti.f32), loss: ti.types.ndarray(ti.f32, 1), dloss: ti.types.ndarray(ti.f32, 1)):
        for i in ti.grouped(dy):
            dy[i] = 0
        loss[0], dloss[0] = 0, 1

    @ti.kernel
    def energy(y: ti.types.ndarray(ti.f32, needs_grad=True), loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in ti.grouped(y):
            loss[0] += 0.5 * weight(i[axis]) * y[i] ** 2

    @ti.kernel
    def explicit_energy(y: ti.types.ndarray(ti.f32), dy: ti.types.ndarray(ti.f32), loss: ti.types.ndarray(ti.f32, 1)):
        for i in ti.grouped(y):
            dy[i] = weight(i[axis]) * y[i]
            loss[0] += 0.5 * weight(i[axis]) * y[i] ** 2

    @ti.kernel
    def prepare_adjoint(dy: ti.types.ndarray(ti.f32), adjoint: ti.types.ndarray(ti.f32)):
        for i in ti.grouped(dy):
            # C2R reconstructs both interior halves; preserve its weighted norm.
            adjoint[i] = dy[i] / weight(i[axis])

    args = {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=len(shape)) for name, shape in shapes.items()
    }

    def build(autodiff):
        builder = ti.graph.GraphBuilder()
        builder.dispatch(seed, args["dy"], args["loss"], args["dloss"])
        builder.append_native(forward)
        if autodiff:
            builder.dispatch(energy, args["y"], args["loss"])
            builder.dispatch(energy.grad, args["y"], args["loss"])
        else:
            builder.dispatch(explicit_energy, args["y"], args["dy"], args["loss"])
        builder.dispatch(prepare_adjoint, args["dy"], args["adjoint"])
        builder.append_native(backward)
        return builder.freeze()

    definition, aot_definition = build(False), build(True)
    providers = (*ti.graph.default_recipe_providers(), ti.hardware.fft.FftRecipeProvider())

    def recipes(definition):
        catalog = definition.recipe_catalog(providers=providers)
        result = {} if callback else {"whole_transform": catalog.baseline.recipe}
        choices = (
            ("whole_transform_store_scale",)
            if callback
            else ("row_batch_column_inplace", "row_batch_cross_batch_columns")
        )
        for choice in choices:
            fragments = [
                f.fragment_id
                for f in catalog.fragments
                if f.provider_namespace.endswith(".fft")
                and f.provider_metadata["family_selection"]["materialization_choice"] == choice
            ]
            if fragments:
                assert len(fragments) == 2
                result[choice] = catalog.compose(fragments, stage="aot-fft").recipe
        return result

    selected, aot_selected = recipes(definition), recipes(aot_definition)
    assert len(selected) == (1 if real or callback else 3)
    module = ti.aot.Module()
    for strategy, recipe in selected.items():
        with aot_definition.materialize(aot_selected[strategy], providers=providers) as product:
            plan = module.add_execution_plan(strategy, product.executor, gradients=dict(y="dy", loss="dloss"))
            document = json.loads(plan.to_json())
            assert len(document["resources"]) == (3 if callback else 2)
            for resource in document["resources"]:
                if resource["kind"] == "host_blob":
                    assert callback and plan.host_data_bytes == resource["byte_size"] > 0
                    assert plan.data_entries == (resource["data"],)
                    continue
                params = resource["parameters"]
                if callback:
                    assert params["store_callback"]["format"] == "cufft-store-scale-lto-v1"
                    assert "path" not in json.dumps(params)
                else:
                    assert params.get("strategy", "whole_transform") == strategy
                assert "expected_workspace_bytes" in params
            assert sum(op["kind"] == "native" for op in document["operations"]) == 2
        with definition.materialize(recipe, providers=providers) as product:
            graph = product.executor
            frames = [{name: ti.ndarray(ti.f32, shape) for name, shape in shapes.items()} for _ in range(2)]
            for factor in (1, -0.5, 0):
                for index, arrays in enumerate(frames):
                    values = [factor * (index + 1) * (i % 13 - 6) / 32 for i in range(math.prod(shapes["x"]))]
                    arrays["x"].from_numpy(np.array(values, np.float32).reshape(shapes["x"]))
                    graph.submit(arrays).wait()
                    for name, expected in reference(case, values).items():
                        np.testing.assert_allclose(arrays[name].to_numpy().ravel(), expected, atol=5e-5, rtol=5e-5)
    artifact = tmp_path / "fft-recipes.tcm"
    module.archive(str(artifact))
    del product, graph, definition, aot_definition, selected, aot_selected, recipe
    gc.collect()
    forward.close()
    backward.close()
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_fft_recipe_probe.py")),
                mode,
                str(artifact),
                case,
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "plan destruction returned status" not in result.stdout + result.stderr
    if dll is not None:
        dll.close()
