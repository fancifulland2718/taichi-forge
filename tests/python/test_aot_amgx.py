"""Real-vendor AOT reconstruction and mixed execution, on small scalar CSR systems."""

import os
from pathlib import Path
import subprocess
import sys

import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.aot_amgx_probe import N, plans


@test_utils.test(arch=[ti.cpu, ti.vulkan])
def test_amgx_aot_does_not_advertise_other_backends():
    with ti.aot.borrow_runtime() as runtime:
        runtime.configure_provider("amgx", adapter_path="unused-adapter")
        for kind in ("amgx_setup_csr", "amgx_replace_coefficients", "amgx_solve"):
            assert not runtime.native_operation_capabilities(kind)
            assert not runtime.probe_native_operation(kind)["available"]


def export(module):
    for dtype in (ti.f32, ti.f64):

        @ti.kernel
        def rhs(seed: ti.types.ndarray(dtype, ndim=1), output: ti.types.ndarray(dtype, ndim=1)):
            for i in seed:
                output[i] = 2 * seed[i] + 0.125

        @ti.kernel
        def publish(solution: ti.types.ndarray(dtype, ndim=1), output: ti.types.ndarray(dtype, ndim=1)):
            for i in solution:
                output[i] = 2 * solution[i] + 1

        @ti.kernel
        def coefficients(diagonal: ti.types.ndarray(dtype, ndim=1), values: ti.types.ndarray(dtype, ndim=1)):
            for i in range(N):
                center = 3 * i
                values[center] = diagonal[0]
                if i > 0:
                    values[center - 1] = -1
                if i + 1 < N:
                    values[center + 1] = -1

        @ti.kernel
        def gradient(
            solution: ti.types.ndarray(dtype, ndim=1),
            adjoint: ti.types.ndarray(dtype, ndim=1),
            values: ti.types.ndarray(dtype, ndim=1),
        ):
            # Explicit adjoint of A x = rhs for this symmetric tridiagonal fixture.
            for i in range(N):
                center = 3 * i
                values[center] = -adjoint[i] * solution[i]
                if i > 0:
                    values[center - 1] = -adjoint[i] * solution[i - 1]
                if i + 1 < N:
                    values[center + 1] = -adjoint[i] * solution[i + 1]

        for name, kernel in (
            ("rhs", rhs),
            ("publish", publish),
            ("coefficients", coefficients),
            ("gradient", gradient),
        ):
            module.add_kernel(kernel, name=f"{name}_{dtype}")
        for name, plan in plans(str(dtype)).items():
            module.add_execution_plan(name, plan)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_amgx_aot_reconstruction_and_explicit_adjoint(tmp_path):
    vendor = os.environ.get("TI_FORGE_TEST_AMGX_LIBRARY_PATH")
    if not vendor:
        pytest.skip("set TI_FORGE_TEST_AMGX_LIBRARY_PATH for real AmgX tests")
    from taichi_forge.hardware._bundled_runtime_provider import _bundled_provider_candidates
    from taichi_forge.hardware._amgx import DEFINITION

    candidates = _bundled_provider_candidates(DEFINITION)
    assert candidates, "The requested AmgX adapter was not built/staged"
    module = ti.aot.Module()
    export(module)
    artifact = tmp_path / "amgx.tcm"
    module.archive(str(artifact))
    for mode in ("standalone", "borrowed"):
        command = [sys.executable] + (["-S"] if mode == "standalone" else [])
        result = subprocess.run(
            [*command, str(Path(__file__).with_name("aot_amgx_probe.py")), mode, str(artifact), candidates[0], vendor],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr
