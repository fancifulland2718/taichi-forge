import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.aot_cutensor_probe import native_plan, contract


@test_utils.test(arch=[ti.cpu, ti.vulkan])
def test_cutensor_is_not_advertised_on_other_backends():
    with ti.aot.borrow_runtime() as runtime:
        runtime.configure_provider("cutensor", adapter_path="unused-adapter", library_path="unused-vendor")
        assert runtime.probe_native_operation("cutensor_contract_f32") == {
            "available": False,
            "capabilities": frozenset(),
            "library_version": None,
        }
        runtime.configure_provider("cutensor", adapter_path=None)
        for invalid in ("", "bad\0path"):
            with pytest.raises(ValueError):
                runtime.configure_provider("cutensor", adapter_path=invalid)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cutensor_aot_reconstruction_and_explicit_adjoint(tmp_path):
    vendor = os.environ.get("TI_FORGE_TEST_CUTENSOR_LIBRARY_PATH")
    if not vendor:
        pytest.skip("set TI_FORGE_TEST_CUTENSOR_LIBRARY_PATH for real cuTENSOR tests")
    from taichi_forge.hardware._bundled_runtime_provider import _bundled_provider_candidates
    from taichi_forge.hardware._cutensor import DEFINITION

    candidates = _bundled_provider_candidates(DEFINITION)
    assert candidates, "The requested cuTENSOR adapter was not built/staged"
    adapter = candidates[0]

    @ti.kernel
    def energy(
        c: ti.types.ndarray(ti.f32, ndim=2, needs_grad=True), loss: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True)
    ):
        for i, j in c:
            loss[0] += 0.5 * c[i, j] * c[i, j]

    resources = [
        native_plan("forward", [(5, 7), (7, 3), (5, 3), (5, 3)], ["ik", "kj", "ij", "ij"]),
        native_plan("adjoint_a", [(5, 3), (7, 3), (5, 7), (5, 7)], ["ij", "kj", "ik", "ik"]),
        native_plan("adjoint_b", [(5, 7), (5, 3), (7, 3), (7, 3)], ["ik", "ij", "kj", "kj"]),
        {"kind": "ndarray", "name": "dc", "dtype": "f32", "shape": [5, 3]},
        {"kind": "ndarray", "name": "dloss", "dtype": "f32", "shape": [1]},
        dict(
            kind="native",
            name="unused",
            provider="unused.provider",
            operation="unknown",
            version=1,
            parameters={},
            bindings={},
        ),
    ]
    operations = [
        {"kind": "fill32", "destination": "loss", "value": 0},
        {"kind": "fill32", "destination": "dc", "value": 0},
        {"kind": "fill32", "destination": "dloss", "value": 1065353216},
        {"kind": "copy", "source": "seed", "destination": "c"},
        contract("forward", "a", "b", "c", "c", alpha=1.25, beta=0.25),
        {"kind": "kernel", "entry": "energy", "arguments": ["c", "loss"]},
        {"kind": "kernel", "entry": "energy_grad", "arguments": ["c", "loss"]},
        contract("adjoint_a", "dc", "b", "da", "da", alpha=1.25),
        contract("adjoint_b", "a", "dc", "db", "db", alpha=1.25),
    ]
    module = ti.aot.Module()
    module.add_kernel(energy)
    module.add_kernel(energy.grad, name="energy_grad")
    module.add_execution_plan(
        "contraction_energy",
        ti.aot.ExecutionPlan(operations, resources=resources, gradients={"c": "dc", "loss": "dloss"}),
    )
    artifact = tmp_path / "cutensor.tcm"
    module.archive(str(artifact))
    with zipfile.ZipFile(artifact) as archive:
        for name in archive.namelist():
            data = archive.read(name)
            assert adapter.encode() not in data and vendor.encode() not in data
    for mode in ("standalone", "borrowed"):
        command = [sys.executable] + (["-S"] if mode == "standalone" else [])
        result = subprocess.run(
            [*command, str(Path(__file__).with_name("aot_cutensor_probe.py")), mode, str(artifact), adapter, vendor],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr
