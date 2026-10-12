import os
from pathlib import Path
import subprocess
import sys

import pytest
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.vulkan])
def test_optional_gemm_is_not_advertised_on_unregistered_backends():
    from taichi_forge_runtime.aot import Runtime

    arch = ti.lang.impl.current_cfg().arch
    with Runtime("vulkan" if arch == ti.vulkan else "cpu") as runtime:
        assert runtime.probe_native_operation("gemm_f32") == {
            "available": False,
            "capabilities": frozenset(),
            "library_version": None,
        }


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_aot_cublas_mixed_execution_and_explicit_adjoint(tmp_path):
    from taichi_forge_runtime.aot import Runtime

    with Runtime("cuda") as runtime:
        if not runtime.probe_native_operation("gemm_f32")["available"]:
            pytest.skip("Compatible external cuBLAS library is unavailable")

    @ti.kernel
    def energy(
        c: ti.types.ndarray(dtype=ti.f32, ndim=2, needs_grad=True),
        loss: ti.types.ndarray(dtype=ti.f32, ndim=1, needs_grad=True),
    ):
        for i, j in c:
            loss[0] += 0.5 * c[i, j] * c[i, j]

    steps = [{"kind": "fill32", "destination": name, "value": 0} for name in ("loss", "dc")] + [
        {"kind": "fill32", "destination": "dloss", "value": 1065353216},
        {"kind": "copy", "source": "seed", "destination": "c"},
        {
            "kind": "gemm_f32",
            "a": "a",
            "b": "b",
            "output": "c",
            "rows": 17,
            "columns": 9,
            "inner": 13,
            "alpha": 1.25,
            "beta": 0.25,
        },
        {"kind": "kernel", "entry": "energy", "arguments": ["c", "loss"]},
        {"kind": "kernel", "entry": "energy_grad", "arguments": ["c", "loss"]},
        {
            "kind": "gemm_f32",
            "a": "dc",
            "b": "b",
            "output": "da",
            "rows": 17,
            "columns": 13,
            "inner": 9,
            "alpha": 1.25,
            "transpose_b": True,
        },
        {
            "kind": "gemm_f32",
            "a": "a",
            "b": "dc",
            "output": "db",
            "rows": 13,
            "columns": 9,
            "inner": 17,
            "alpha": 1.25,
            "transpose_a": True,
        },
    ]
    # Export the new common envelope. The consumer's direct transpose/alias
    # cases below retain legacy syntax, exercising both through one factory.
    steps = [
        (
            dict(
                kind="native",
                provider="cublas",
                operation="gemm_f32",
                version=1,
                bindings={**{name: step[name] for name in ("a", "b", "output")}, "context": "blas"},
                parameters={key: value for key, value in step.items() if key not in ("kind", "a", "b", "output")},
            )
            if step["kind"] == "gemm_f32"
            else step
        )
        for step in steps
    ]
    plan = ti.aot.ExecutionPlan(
        steps,
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
            {"kind": "ndarray", "name": "dc", "dtype": "f32", "shape": [17, 9]},
            {"kind": "ndarray", "name": "dloss", "dtype": "f32", "shape": [1]},
        ],
        gradients={"c": "dc", "loss": "dloss"},
    )
    module = ti.aot.Module()
    module.add_kernel(energy, name="energy")
    module.add_kernel(energy.grad, name="energy_grad")
    module.add_execution_plan("matrix_energy", plan)
    artifact = tmp_path / "gemm.tcm"
    module.archive(str(artifact))
    for mode in ("standalone", "borrowed"):
        command = [sys.executable]
        if mode == "standalone":
            command.append("-S")
        result = subprocess.run(
            [*command, str(Path(__file__).with_name("aot_gemm_probe.py")), mode, str(artifact)],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "handle destruction returned status" not in result.stdout + result.stderr
