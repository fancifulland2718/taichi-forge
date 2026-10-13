"""Reconstruct cuSPARSELt resources without a JIT provider in the consumer."""

import os
from pathlib import Path
import subprocess
import sys

import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.aot_cusparselt_probe import plan_resource, compress, multiply


@test_utils.test(arch=[ti.cpu, ti.vulkan])
def test_cusparselt_aot_is_not_advertised_on_other_backends():
    with ti.aot.borrow_runtime() as runtime:
        runtime.configure_provider("cusparselt", adapter_path="unused-adapter")
        for kind in ("cusparselt_compress_f16", "cusparselt_matmul_f16"):
            assert not runtime.native_operation_capabilities(kind)
            assert not runtime.probe_native_operation(kind)["available"]


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cusparselt_aot_shared_compression_and_explicit_adjoint(tmp_path):
    vendor = os.environ.get("TI_FORGE_TEST_CUSPARSELT_LIBRARY_PATH")
    if not vendor:
        pytest.skip("set TI_FORGE_TEST_CUSPARSELT_LIBRARY_PATH for real cuSPARSELt tests")
    from taichi_forge.hardware._bundled_runtime_provider import _bundled_provider_candidates
    from taichi_forge.hardware._cusparselt import DEFINITION

    candidates = _bundled_provider_candidates(DEFINITION)
    assert candidates, "The requested cuSPARSELt adapter was not built/staged"

    @ti.kernel
    def energy(
        d: ti.types.ndarray(ti.f16, ndim=2), e: ti.types.ndarray(ti.f16, ndim=2), loss: ti.types.ndarray(ti.f32, ndim=1)
    ):
        for i, j in d:
            loss[0] += 0.5 * (ti.cast(d[i, j], ti.f32) ** 2 + ti.cast(e[i, j], ti.f32) ** 2)

    @ti.kernel
    def adjoint(
        a: ti.types.ndarray(ti.f16, ndim=2),
        b: ti.types.ndarray(ti.f16, ndim=2),
        d: ti.types.ndarray(ti.f16, ndim=2),
        e: ti.types.ndarray(ti.f16, ndim=2),
        da: ti.types.ndarray(ti.f32, ndim=2),
        db: ti.types.ndarray(ti.f32, ndim=2),
    ):
        # Explicit adjoint with the current sparsity pattern held fixed.
        # Arithmetic is f32; this does not differentiate FP16 rounding/pruning.
        for i, k in da:
            value = 0.0
            if a[i, k] != 0:
                for j in range(b.shape[0]):
                    value += (0.75 * ti.cast(d[i, j], ti.f32) - 0.5 * ti.cast(e[i, j], ti.f32)) * ti.cast(
                        b[j, k], ti.f32
                    )
            da[i, k] = value
        for j, k in db:
            value = 0.0
            for i in range(a.shape[0]):
                value += (0.75 * ti.cast(d[i, j], ti.f32) - 0.5 * ti.cast(e[i, j], ti.f32)) * ti.cast(a[i, k], ti.f32)
            db[j, k] = value

    operations = [
        {"kind": "fill32", "destination": "loss", "value": 0},
        compress("a"),
        multiply("b", "c", "d", alpha=0.75, beta=0.25),
        multiply("b", "c", "e", alpha=-0.5, beta=0.5),
        {"kind": "kernel", "entry": "energy", "arguments": ["d", "e", "loss"]},
        {"kind": "kernel", "entry": "adjoint", "arguments": ["a", "b", "d", "e", "da", "db"]},
    ]
    module = ti.aot.Module()
    module.add_kernel(energy)
    module.add_kernel(adjoint)
    module.add_execution_plan("shared_a", ti.aot.ExecutionPlan(operations, resources=[plan_resource()]))
    from tests.python.aot_cusparselt_snapshot_probe import snapshot_plans

    for name, plan in snapshot_plans().items():
        module.add_execution_plan(name, plan)
    artifact = tmp_path / "cusparselt.tcm"
    module.archive(str(artifact))
    for mode in ("standalone", "borrowed"):
        command = [sys.executable] + (["-S"] if mode == "standalone" else [])
        result = subprocess.run(
            [
                *command,
                str(Path(__file__).with_name("aot_cusparselt_probe.py")),
                mode,
                str(artifact),
                candidates[0],
                vendor,
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr
