"""Shared cuBLAS vector descriptors retain device coefficients in captured AOT."""

import os
from pathlib import Path
import subprocess
import sys

import pytest
import taichi_forge as ti
from tests import test_utils


def native(operation, bindings, **parameters):
    return dict(
        kind="native",
        provider="cublas",
        operation=operation,
        version=1,
        parameters=parameters,
        bindings=dict(bindings, context="blas"),
    )


@test_utils.test(arch=[ti.cpu, ti.vulkan])
def test_blas_vectors_are_not_advertised_on_other_backends():
    from taichi_forge_runtime.aot import Runtime

    with Runtime("cpu" if ti.lang.impl.current_cfg().arch == ti.cpu else "vulkan") as runtime:
        for name in ("axpy_f32", "scal_f32", "nrm2_f32"):
            assert not runtime.probe_native_operation(name)["available"]


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_aot_blas_vectors_device_coefficients_and_explicit_adjoint(tmp_path):
    from taichi_forge_runtime.aot import Runtime

    with Runtime("cuda") as runtime:
        for name in ("axpy_f32", "scal_f32", "nrm2_f32"):
            if not runtime.probe_native_operation(name)["available"]:
                pytest.skip(f"Compatible external cuBLAS {name} is unavailable")

    @ti.kernel
    def energy(
        norm: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True), loss: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True)
    ):
        loss[0] = 0.5 * norm[0] * norm[0]

    @ti.kernel
    def adjoint(
        x: ti.types.ndarray(ti.f32, ndim=1),
        y0: ti.types.ndarray(ti.f32, ndim=1),
        y: ti.types.ndarray(ti.f32, ndim=1),
        norm: ti.types.ndarray(ti.f32, ndim=1),
        dnorm: ti.types.ndarray(ti.f32, ndim=1),
        alpha: ti.types.ndarray(ti.f32, ndim=0),
        beta: ti.types.ndarray(ti.f32, ndim=1),
        dx: ti.types.ndarray(ti.f32, ndim=1),
        dy: ti.types.ndarray(ti.f32, ndim=1),
        da: ti.types.ndarray(ti.f32, ndim=1),
        db: ti.types.ndarray(ti.f32, ndim=1),
    ):
        for i in x:
            gradient = 0.0
            if norm[0] > 0:
                gradient = dnorm[0] * y[i] / norm[0]
            dx[i] = gradient * -0.5 * beta[0] * alpha[None]
            dy[i] = gradient * -0.25 * beta[0]
            da[0] += gradient * -0.5 * beta[0] * x[i]
            db[0] += gradient * -0.5 * (alpha[None] * x[i] + 0.5 * y0[i])

    operations = [dict(kind="fill32", destination=name, value=0) for name in ("dnorm", "da", "db")]
    operations += [
        dict(kind="fill32", destination="dloss", value=0x3F800000),
        dict(kind="copy", source="y0", destination="y"),
        native("scal_f32", {"x": "y"}, size=37, alpha=0.5),
        native("axpy_f32", {"x": "x", "y": "y", "alpha": "alpha"}, size=37),
        native("scal_f32", {"x": "y", "alpha": "beta"}, size=37),
        native("scal_f32", {"x": "y"}, size=37, alpha=-0.5),
        native("nrm2_f32", {"x": "y", "output": "norm"}, size=37),
        dict(kind="kernel", entry="energy", arguments=["norm", "loss"]),
        dict(kind="kernel", entry="energy_grad", arguments=["norm", "loss"]),
        dict(
            kind="kernel",
            entry="adjoint",
            arguments=["x", "y0", "y", "norm", "dnorm", "alpha", "beta", "dx", "dy", "da", "db"],
        ),
    ]
    resources = [
        dict(kind="native", name="blas", provider="cublas", operation="context", version=1, parameters={}, bindings={})
    ]
    resources += [dict(kind="ndarray", name=name, dtype="f32", shape=[1]) for name in ("dnorm", "dloss")]
    module = ti.aot.Module()
    module.add_kernel(energy)
    module.add_kernel(energy.grad, name="energy_grad")
    module.add_kernel(adjoint)
    module.add_execution_plan(
        "vectors", ti.aot.ExecutionPlan(operations, resources=resources, gradients={"norm": "dnorm", "loss": "dloss"})
    )
    artifact = tmp_path / "vectors.tcm"
    module.archive(str(artifact))
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_blas_vectors_probe.py")),
                mode,
                str(artifact),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "handle destruction returned status" not in result.stdout + result.stderr
