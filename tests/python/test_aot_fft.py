import os
from pathlib import Path
import subprocess
import sys

import pytest
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.vulkan])
def test_optional_fft_is_not_advertised_on_unregistered_backends():
    from taichi_forge_runtime.aot import Runtime

    arch = ti.lang.impl.current_cfg().arch
    with Runtime("vulkan" if arch == ti.vulkan else "cpu") as runtime:
        assert runtime.probe_native_operation("fft_c2c_f32") == {
            "available": False,
            "capabilities": frozenset(),
            "library_version": None,
        }
        assert runtime.probe_native_operation("compact")["available"]
        assert not runtime.probe_native_operation("compact", version=2)["available"]


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_aot_cufft_mixed_execution_and_explicit_adjoint(tmp_path):
    from taichi_forge_runtime.aot import Runtime

    dll_directory = os.environ.get("TI_TEST_CUFFT_DIRECTORY")
    directory = os.add_dll_directory(dll_directory) if dll_directory and os.name == "nt" else None
    with Runtime("cuda") as runtime:
        if not runtime.probe_native_operation("fft_c2c_f32")["available"]:
            pytest.skip("Compatible external cuFFT library is unavailable")

    @ti.kernel
    def scale(
        x: ti.types.ndarray(dtype=ti.math.vec2, ndim=1, needs_grad=True),
        y: ti.types.ndarray(dtype=ti.math.vec2, ndim=1, needs_grad=True),
    ):
        for i in x:
            y[i] = 2.0 * x[i]

    @ti.kernel
    def energy(
        frequency: ti.types.ndarray(dtype=ti.math.vec2, ndim=1, needs_grad=True),
        loss: ti.types.ndarray(dtype=ti.f32, ndim=1, needs_grad=True),
    ):
        for i in frequency:
            loss[0] += 0.5 * frequency[i].dot(frequency[i])

    steps = [{"kind": "fill32", "destination": name, "value": 0} for name in ("loss", "dx", "dfrequency")] + [
        {"kind": "fill32", "destination": "dloss", "value": 1065353216},
        {"kind": "kernel", "entry": "scale", "arguments": ["x", "temporary"]},
        {
            "kind": "fft_c2c_f32",
            "input": "temporary",
            "output": "frequency",
            "dimensions": [257],
            "batch_count": 2,
            "direction": "forward",
        },
        {"kind": "kernel", "entry": "energy", "arguments": ["frequency", "loss"]},
        {"kind": "kernel", "entry": "energy_grad", "arguments": ["frequency", "loss"]},
        {
            "kind": "fft_c2c_f32",
            "input": "dfrequency",
            "output": "dtemporary",
            "dimensions": [257],
            "batch_count": 2,
            "direction": "inverse",
        },
        {"kind": "kernel", "entry": "scale_grad", "arguments": ["x", "temporary"]},
    ]
    resources = [
        {"kind": "ndarray", "name": name, "dtype": "f32", "element_shape": [2], "shape": [{"array": "x", "axis": 0}]}
        for name in ("temporary", "dtemporary", "dfrequency")
    ] + [{"kind": "ndarray", "name": "dloss", "dtype": "f32", "shape": [1]}]
    plan = ti.aot.ExecutionPlan(
        steps,
        resources=resources,
        gradients={"x": "dx", "temporary": "dtemporary", "frequency": "dfrequency", "loss": "dloss"},
    )
    module = ti.aot.Module()
    for entry, name in ((scale, "scale"), (scale.grad, "scale_grad"), (energy, "energy"), (energy.grad, "energy_grad")):
        module.add_kernel(entry, name=name)
    module.add_execution_plan("spectral_energy", plan)
    artifact = tmp_path / "fft.tcm"
    module.archive(str(artifact))
    for mode in ("standalone", "borrowed"):
        command = [sys.executable]
        if mode == "standalone":
            command.append("-S")
        result = subprocess.run(
            [*command, str(Path(__file__).with_name("aot_fft_probe.py")), mode, str(artifact)],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "plan destruction returned status" not in result.stdout + result.stderr
    if directory is not None:
        directory.close()
