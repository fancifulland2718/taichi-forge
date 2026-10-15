import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.aot_cudss_probe import plan_resource, solve_group


@test_utils.test(arch=[ti.cpu, ti.vulkan])
def test_cudss_aot_is_not_advertised_on_other_backends():
    with ti.aot.borrow_runtime() as runtime:
        runtime.configure_provider("cudss", adapter_path="unused-adapter", library_path="unused-vendor")
        assert runtime.probe_native_operation("cudss_solve_group_f32") == {
            "available": False, "capabilities": frozenset(), "library_version": None,
        }


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cudss_aot_reconstruction_and_explicit_adjoint(tmp_path):
    adapter = os.environ.get("TI_CUDSS_TEST_ADAPTER")
    vendor = os.environ.get("TI_CUDSS_TEST_LIBRARY")
    if not adapter or not vendor:
        pytest.skip("set TI_CUDSS_TEST_ADAPTER and TI_CUDSS_TEST_LIBRARY for real cuDSS tests")

    @ti.kernel
    def energy(x: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True),
               loss: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True)):
        for i in x:
            loss[0] += 0.5 * x[i] * x[i]

    resources = [plan_resource(),
                 dict(kind="ndarray", name="dx", dtype="f32", shape=[4]),
                 dict(kind="ndarray", name="dloss", dtype="f32", shape=[1])]
    operations = [dict(kind="fill32", destination="dx", value=0),
                  dict(kind="fill32", destination="loss", value=0),
                  dict(kind="fill32", destination="dloss", value=1065353216),
                  solve_group([("b", "x"), ("b2", "x2")]),
                  dict(kind="kernel", entry="energy", arguments=["x", "loss"]),
                  dict(kind="kernel", entry="energy_grad", arguments=["x", "loss"]),
                  solve_group([("dx", "db")])]
    module = ti.aot.Module()
    module.add_kernel(energy)
    module.add_kernel(energy.grad, name="energy_grad")
    module.add_execution_plan("solve_energy", ti.aot.ExecutionPlan(
        operations, resources=resources, gradients={"x": "dx", "loss": "dloss"}))
    artifact = tmp_path / "cudss.tcm"
    module.archive(str(artifact))
    with zipfile.ZipFile(artifact) as archive:
        for name in archive.namelist():
            data = archive.read(name)
            assert adapter.encode() not in data and vendor.encode() not in data
    for mode in ("standalone", "borrowed"):
        command = [sys.executable] + (["-S"] if mode == "standalone" else [])
        result = subprocess.run(
            [*command, str(Path(__file__).with_name("aot_cudss_probe.py")), mode, str(artifact), adapter, vendor],
            env=os.environ.copy(), capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr
