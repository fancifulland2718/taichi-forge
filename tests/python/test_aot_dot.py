import os
from pathlib import Path
import subprocess
import sys

import pytest
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.vulkan])
def test_optional_dot_is_not_advertised_on_unregistered_backends():
    from taichi_forge_runtime.aot import Runtime

    arch = ti.lang.impl.current_cfg().arch
    with Runtime("vulkan" if arch == ti.vulkan else "cpu") as runtime:
        assert not runtime.probe_native_operation("dot_f32")["available"]


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_aot_cublas_device_dot_and_explicit_adjoint(tmp_path):
    from taichi_forge_runtime.aot import Runtime

    with Runtime("cuda") as runtime:
        if not runtime.probe_native_operation("dot_f32")["available"]:
            pytest.skip("Compatible external cuBLAS dot provider is unavailable")

    @ti.kernel
    def energy(
        dot: ti.types.ndarray(dtype=ti.f32, ndim=1, needs_grad=True),
        loss: ti.types.ndarray(dtype=ti.f32, ndim=1, needs_grad=True),
    ):
        loss[0] = 0.5 * dot[0] * dot[0]

    @ti.kernel
    def dot_adjoint(
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
        ddot: ti.types.ndarray(dtype=ti.f32, ndim=1),
        dx: ti.types.ndarray(dtype=ti.f32, ndim=1),
        dy: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in x:
            dx[i] = ddot[0] * y[i]
            dy[i] = ddot[0] * x[i]

    plan = ti.aot.ExecutionPlan(
        [
            {"kind": "fill32", "destination": "ddot", "value": 0},
            {"kind": "fill32", "destination": "dloss", "value": 1065353216},
            {
                "kind": "native",
                "provider": "cublas",
                "operation": "dot_f32",
                "version": 1,
                "parameters": {"size": 37},
                "bindings": {"x": "x", "y": "y", "output": "dot", "context": "blas"},
            },
            {"kind": "kernel", "entry": "energy", "arguments": ["dot", "loss"]},
            {"kind": "kernel", "entry": "energy_grad", "arguments": ["dot", "loss"]},
            {"kind": "kernel", "entry": "dot_adjoint", "arguments": ["x", "y", "ddot", "dx", "dy"]},
        ],
        resources=[
            {
                "kind": "native",
                "name": "blas",
                "provider": "cublas",
                "operation": "context",
                "version": 1,
                "parameters": {},
                "bindings": {},
            },
            {"kind": "ndarray", "name": "ddot", "dtype": "f32", "shape": [1]},
            {"kind": "ndarray", "name": "dloss", "dtype": "f32", "shape": [1]},
        ],
        gradients={"dot": "ddot", "loss": "dloss"},
    )
    module = ti.aot.Module()
    module.add_kernel(energy, name="energy")
    module.add_kernel(energy.grad, name="energy_grad")
    module.add_kernel(dot_adjoint, name="dot_adjoint")
    module.add_execution_plan("dot_energy", plan)
    artifact = tmp_path / "dot.tcm"
    module.archive(str(artifact))
    for mode in ("standalone", "borrowed"):
        command = [sys.executable]
        if mode == "standalone":
            command.append("-S")
        result = subprocess.run(
            [*command, str(Path(__file__).with_name("aot_dot_probe.py")), mode, str(artifact)],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "handle destruction returned status" not in result.stdout + result.stderr
