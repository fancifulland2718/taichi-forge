"""Shared native FFT plans survive export independently of their JIT owner."""

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


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cufft_creation_recording_and_export_share_frozen_definition(tmp_path, monkeypatch):
    from taichi_forge.graph._ir import GraphAccess
    from taichi_forge.graph._native_definition import NativeDefinition, NativeOperationBinding
    from taichi_forge_runtime.aot import Runtime

    fft = ti.hardware.fft
    if not fft.is_available():
        pytest.skip("external cuFFT required")
    plan = fft.CufftPlan1D(8, batch_count=2)
    assert not hasattr(plan, "_aot_plan_parameters")
    # Reassignment remains allowed; these attributes do not reconfigure cuFFT.
    plan.dimensions, plan.batch_count, plan.transform = (99,), 5, "c2r"
    plan.input_layout = fft.CufftLayout((99,), stride=3, batch_distance=300)
    assert plan.input_shape == plan.output_shape == (2, 8, 2)
    assert plan.inverse_scale == 1 / 8
    recording = plan.record(input="signal", output="spectrum")
    contract = recording._graph_source_contract()
    assert contract["workload"]["dimensions"] == [2, 8]
    assert contract["workload"]["logical_items"] == 16  # Not 32 packed float lanes.
    assert contract["workload"]["scheduling_owner"] == "provider"
    assert contract["workload"]["transformation_permissions"] == []
    assert recording.direction == "forward"
    assert recording.binding_names == ("signal", "spectrum")
    assert tuple((effect.resource, effect.access) for effect in recording.resource_effects) == (
        ("signal", GraphAccess.READ),
        ("spectrum", GraphAccess.WRITE),
    )
    values = np.arange(32, dtype=np.float32).reshape(2, 8, 2) / 16
    source, output = (ti.ndarray(ti.f32, (2, 8, 2)) for _ in range(2))
    source.from_numpy(values)
    expected = np.fft.fft(values[..., 0] + 1j * values[..., 1], axis=1)

    def no_description_conversion(*args, **kwargs):
        raise AssertionError("Ordinary execution entered description lowering")

    before = fft.cache_statistics()
    with monkeypatch.context() as patch:
        patch.setattr(NativeDefinition, "parameter_dict", no_description_conversion)
        patch.setattr(NativeOperationBinding, "lower", no_description_conversion)
        patch.setattr(NativeOperationBinding, "source_contract", no_description_conversion)
        plan.execute(source, output)
        recording.execute(dict(signal=source, spectrum=output))
    assert fft.cache_statistics() == before
    actual = output.to_numpy()
    np.testing.assert_allclose(actual[..., 0] + 1j * actual[..., 1], expected, rtol=2e-5, atol=2e-5)
    builder = ti.graph.GraphBuilder()
    builder.append_native(recording)
    graph = builder.compile()
    sources = [json.loads(action.source_contract_json)
               for stage in graph.definition._runtime_spec.pipeline_definition
               for action in stage.get("native_actions", ()) if action.source_contract_json]
    assert sources == [contract]
    graph.run(dict(signal=source, spectrum=output))
    module = ti.aot.Module()
    exported = module.add_execution_plan("fft", graph)
    document = json.loads(exported.to_json())
    resource = document["resources"][0]
    assert resource["parameters"]["dimensions"] == [8]
    assert resource["parameters"]["batch_count"] == 2
    assert resource["parameters"]["transform"] == "c2c"
    assert resource["parameters"] == contract["resources"]["plan"]["parameters"]
    assert document["operations"][0]["bindings"] == dict(plan=resource["name"], input="signal", output="spectrum")
    artifact = tmp_path / "fft.tcm"
    module.archive(str(artifact))
    graph.close()
    plan.close()
    with Runtime("cuda") as runtime:
        loaded = runtime.load_module(artifact)
        x, y = (runtime.ndarray("f32", (2, 8, 2)) for _ in range(2))
        x.write(values.tobytes())
        prepared = loaded.prepare_execution("fft", dict(signal=x, spectrum=y))
        prepared()
        actual = np.frombuffer(y.read(), np.float32).reshape(2, 8, 2)
        np.testing.assert_allclose(actual[..., 0] + 1j * actual[..., 1], expected, rtol=2e-5, atol=2e-5)
        prepared.close()


@pytest.mark.parametrize("case", ("complex", "padded", "real", "workspace"))
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_fft_recording_export(case, tmp_path):
    from taichi_forge_runtime.aot import Runtime

    directory = os.environ.get("TI_TEST_CUFFT_DIRECTORY")
    dll = os.add_dll_directory(directory) if directory and os.name == "nt" else None
    with Runtime("cuda") as runtime:
        if not runtime.probe_native_operation("cufft_execute_f32")["available"]:
            pytest.skip("external cuFFT required")

    fft = ti.hardware.fft
    if case in ("complex", "workspace"):
        forward = fft.CufftPlan1D(17 if case == "complex" else 65536, batch_count=2 if case == "complex" else 1)
        inverse = forward
    elif case == "real":
        forward = fft.CufftPlanND((2, 3, 5), batch_count=2, transform="r2c")
        inverse = fft.CufftPlanND((2, 3, 5), batch_count=2, transform="c2r")
    else:
        left = fft.CufftLayout((3, 7), stride=2, batch_distance=47)
        right = fft.CufftLayout((3, 6), stride=1, batch_distance=23)
        forward = fft.CufftPlanND((3, 5), batch_count=2, input_layout=left, output_layout=right)
        inverse = fft.CufftPlanND((3, 5), batch_count=2, input_layout=right, output_layout=left)

    @ti.kernel
    def scale(x: ti.types.ndarray(ti.f32), source: ti.types.ndarray(ti.f32)):
        for i in ti.grouped(x):
            source[i] = 2 * x[i]

    @ti.kernel
    def copy_spectrum(spectrum: ti.types.ndarray(ti.f32), saved: ti.types.ndarray(ti.f32)):
        for i in ti.grouped(spectrum):
            saved[i] = spectrum[i]

    size = 1
    for extent in forward.dimensions:
        size *= extent

    @ti.kernel
    def finish(inverted: ti.types.ndarray(ti.f32), y: ti.types.ndarray(ti.f32), dx: ti.types.ndarray(ti.f32)):
        for i in ti.grouped(inverted):
            y[i] = inverted[i] / (2.0 * size)
            # Adjoint of the scaled C2C forward for 0.5*||FFT(2*x)||^2.
            # For real transforms this uses the full-spectrum weighted norm.
            dx[i] = 2.0 * inverted[i]

    shapes = {name: forward.input_shape for name in ("x", "source", "inverted", "y", "dx")}
    shapes.update({name: forward.output_shape for name in ("spectrum", "saved")})
    args = {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=len(shape)) for name, shape in shapes.items()
    }
    builder = ti.graph.GraphBuilder()
    builder.dispatch(scale, args["x"], args["source"])
    builder.append_native(forward.record(input="source", output="spectrum"))
    builder.dispatch(copy_spectrum, args["spectrum"], args["saved"])
    builder.append_native(inverse.record(input="spectrum", output="inverted", direction="inverse"))
    builder.dispatch(finish, args["inverted"], args["y"], args["dx"])
    graph = builder.compile()
    module = ti.aot.Module()
    plan = module.add_execution_plan("step", graph)
    description = json.loads(plan.to_json())
    assert len(description["resources"]) == (1 if forward is inverse else 2)
    native = [item for item in description["operations"] if item["kind"] == "native"]
    assert len(native) == 2 and len(module._exported_kernels) == 3
    assert (native[0]["bindings"]["plan"] == native[1]["bindings"]["plan"]) == (forward is inverse)
    workspace = forward._workspace_bytes + (inverse._workspace_bytes if inverse is not forward else 0)
    graph.close()
    del graph, builder
    gc.collect()
    forward.close()
    if inverse is not forward:
        inverse.close()

    artifact = tmp_path / "fft-recording.tcm"
    module.archive(str(artifact))
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_fft_recording_probe.py")),
                mode,
                str(artifact),
                case,
                str(workspace),
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


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_fft_export_rejects_unqualified_plan(tmp_path):
    fft = ti.hardware.fft
    if not fft.is_available():
        pytest.skip("external cuFFT required")

    class CallbackVariant(fft.CufftPlan1D):
        pass

    plan = CallbackVariant(7)
    builder = ti.graph.GraphBuilder()
    builder.append_native(plan.record())
    graph = builder.compile()
    module = ti.aot.Module()
    with pytest.raises(ti.TaichiRuntimeError, match="plan variants"):
        module.add_execution_plan("step", graph)
    assert not module._content and not module._execution_plans
    graph.close()
    del graph, builder
    gc.collect()
    plan.close()
