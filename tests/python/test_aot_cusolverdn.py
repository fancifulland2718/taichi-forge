"""Loaded cuSOLVER plans preserve device factors and root submission order."""

import os
from pathlib import Path
import subprocess
import sys

import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.aot_cusolverdn_probe import operation, resource


@test_utils.test(arch=[ti.cpu, ti.vulkan])
def test_cusolverdn_aot_does_not_probe_on_other_backends():
    with ti.aot.borrow_runtime() as runtime:
        runtime.configure_provider("cusolverdn", adapter_path="unused-adapter", library_path="unused-vendor")
        assert runtime.probe_native_operation("cusolverdn_cholesky") == {
            "available": False,
            "capabilities": frozenset(),
            "library_version": None,
        }


@pytest.mark.parametrize("dtype", (ti.f32, ti.f64))
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cusolverdn_aot_resources_real_gradient_and_explicit_adjoint(tmp_path, dtype):
    adapter = os.environ.get("TI_CUSOLVERDN_TEST_ADAPTER")
    vendor = os.environ.get("TI_FORGE_TEST_CUSOLVERDN_LIBRARY_PATH")
    if not adapter or not vendor:
        pytest.skip("set real cuSOLVER adapter/vendor paths")

    @ti.kernel
    def energy(x: ti.types.ndarray(dtype, 2, needs_grad=True), loss: ti.types.ndarray(dtype, 1, needs_grad=True)):
        for i, j in x:
            loss[0] += 0.5 * x[i, j] * x[i, j]

    @ti.kernel
    def initialize(dx: ti.types.ndarray(dtype, 2), loss: ti.types.ndarray(dtype, 1, needs_grad=True)):
        loss[0] = 0
        loss.grad[0] = 1
        for i, j in dx:
            dx[i, j] = 0

    resources = [
        resource(str(dtype)),
        dict(kind="ndarray", name="dx", dtype=str(dtype), shape=[3, 17]),
        dict(kind="ndarray", name="dloss", dtype=str(dtype), shape=[1]),
    ]
    operations = [
        dict(kind="kernel", entry="initialize", arguments=["dx", "loss"]),
        operation(),
        dict(kind="kernel", entry="energy", arguments=["x", "loss"]),
        dict(kind="kernel", entry="energy_grad", arguments=["x", "loss"]),
        operation("solve", "dx", "db"),
    ]
    module = ti.aot.Module()
    module.add_kernel(initialize)
    module.add_kernel(energy)
    module.add_kernel(energy.grad, name="energy_grad")
    module.add_execution_plan(
        "energy", ti.aot.ExecutionPlan(operations, resources=resources, gradients={"x": "dx", "loss": "dloss"})
    )
    artifact = tmp_path / "cusolverdn.tcm"
    module.archive(str(artifact))
    for mode in ("standalone", "borrowed"):
        command = [sys.executable] + (["-S"] if mode == "standalone" else [])
        result = subprocess.run(
            [
                *command,
                str(Path(__file__).with_name("aot_cusolverdn_probe.py")),
                mode,
                str(artifact),
                adapter,
                vendor,
                str(dtype),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr
