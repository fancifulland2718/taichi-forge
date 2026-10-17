"""Selected CUDA binding recipes retain their shared-executable contract."""

from contextlib import contextmanager
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
from tests.python.test_graph_binding_frame_recipe import _catalog, _texture_definition


@contextmanager
def selected(definition):
    catalog = _catalog(definition)
    recipe = next(entry.recipe for entry in catalog.entries() if entry.recipe.fragments)
    with definition.materialization_context(provider_set=catalog.provider_set) as context:
        with context.materialize(recipe) as materialized:
            yield materialized.executor


def consume(module, tmp_path, case):
    artifact = tmp_path / (case + ".tcm")
    module.archive(str(artifact))
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_binding_family_probe.py")),
                mode,
                str(artifact),
                case,
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Unable to release execution resource" not in result.stdout + result.stderr


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_aot_selected_cuda_binding_recipe_and_explicit_adjoint(tmp_path, monkeypatch):
    @ti.kernel
    def stage(x: ti.types.ndarray(ti.f32, 1), scratch: ti.types.ndarray(ti.f32, 1), gain: ti.f32):
        for i in x:
            scratch[i] = gain * x[i] * x[i]

    @ti.kernel
    def finish(scratch: ti.types.ndarray(ti.f32, 1), y: ti.types.ndarray(ti.f32, 1)):
        for i in y:
            y[i] = scratch[i] + 7

    @ti.kernel
    def backward(
        x: ti.types.ndarray(ti.f32, 1),
        dy: ti.types.ndarray(ti.f32, 1),
        dx: ti.types.ndarray(ti.f32, 1),
        gain: ti.f32,
    ):
        for i in x:
            dx[i] = 2 * gain * x[i] * dy[i]

    names = ("x", "y", "dy", "dx")
    args = {name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1) for name in names}
    gain = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "gain", ti.f32)
    builder = ti.graph.GraphBuilder()
    scratch = builder.private_ndarray("scratch", ti.f32, 17)
    builder.dispatch(stage, args["x"], scratch, gain)
    builder.dispatch(finish, scratch, args["y"])
    builder.dispatch(backward, args["x"], args["dy"], args["dx"], gain)
    module = ti.aot.Module()
    values = np.linspace(-1, 1, 17, dtype=np.float32)
    with selected(builder.freeze()) as graph:
        frame = {name: ti.ndarray(ti.f32, 17) for name in names}
        frame["x"].from_numpy(values)
        frame["dy"].fill(0.5)
        graph.submit(graph.bind(dict(frame, gain=3.0))).wait()
        np.testing.assert_allclose(frame["y"].to_numpy(), 3 * values**2 + 7)
        np.testing.assert_allclose(frame["dx"].to_numpy(), 3 * values)
        plan = module.add_execution_plan("step", graph)
        description = json.loads(plan.to_json())
        assert description["binding_strategy"] == "cuda_immutable_argument_frames"
        assert description["version"] == 10
        assert len(description["resources"]) == 1

        class UnqualifiedExecutor(graph._spec._binding_executor_factory):
            pass

        rejected = ti.aot.Module()
        with monkeypatch.context() as patch:
            patch.setattr(graph._spec, "_binding_executor_factory", UnqualifiedExecutor)
            with pytest.raises(ti.TaichiRuntimeError, match="selected storage/binding recipe"):
                rejected.add_execution_plan("step", graph)
        assert not rejected._exported_kernels
    del builder, graph, frame
    gc.collect()
    consume(module, tmp_path, "selected")


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_aot_selected_cuda_binding_fft_recipe(tmp_path):
    directory = os.environ.get("TI_TEST_CUFFT_DIRECTORY")
    dll = os.add_dll_directory(directory) if directory and os.name == "nt" else None
    if not ti.hardware.fft.is_available():
        pytest.skip("external cuFFT required")
    builder = ti.graph.GraphBuilder()
    tolerance = dict(absolute_tolerance=5e-5, relative_tolerance=5e-5)
    builder.append_native(ti.linalg.record_fft((1, 17), input="x", output="spectrum", **tolerance))
    builder.append_native(
        ti.linalg.record_fft((1, 17), input="spectrum", output="y", direction="inverse", **tolerance)
    )
    module = ti.aot.Module()
    with selected(builder.freeze()) as graph:
        plan = module.add_execution_plan("step", graph)
        assert plan.binding_strategy == "cuda_immutable_argument_frames"
    del builder, graph
    gc.collect()
    consume(module, tmp_path, "fft_selected")
    if dll is not None:
        dll.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_aot_selected_cuda_binding_texture_rejects_before_export():
    with selected(_texture_definition()) as graph:
        module = ti.aot.Module()
        with pytest.raises(ti.TaichiRuntimeError, match="source_bindings"):
            module.add_execution_plan("step", graph)
        assert not module._exported_kernels and not module._execution_plans
