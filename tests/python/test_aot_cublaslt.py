"""Frozen cuBLASLt plans are independently consumable without the DSL."""

import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.aot_cublaslt_probe import matmul, native_plan


@test_utils.test(arch=[ti.cpu, ti.vulkan])
def test_cublaslt_aot_does_not_probe_on_other_backends():
    with ti.aot.borrow_runtime() as runtime:
        runtime.configure_provider("cublaslt", adapter_path="unused-adapter", library_path="unused-vendor")
        assert runtime.probe_native_operation("cublaslt_matmul_f32") == {
            "available": False,
            "capabilities": frozenset(),
            "library_version": None,
        }


@pytest.mark.parametrize(
    "batch,layout,epilogue", [(1, "row_major", "identity"), (2, "transposed_column_major", "relu")]
)
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cublaslt_aot_frozen_plan_and_real_gradient(tmp_path, batch, layout, epilogue):
    from taichi_forge.hardware._aot_cublaslt import frozen_parameters
    from taichi_forge.hardware._cublaslt import CublasLtProvider
    from taichi_forge.hardware._cublaslt_algorithms import _MatmulRecipePlan

    adapter = os.environ.get("TI_CUBLASLT_TEST_ADAPTER")
    vendor = os.environ.get("TI_FORGE_TEST_CUBLASLT_LIBRARY_PATH")
    if not adapter or not vendor:
        pytest.skip("set real cuBLASLt adapter/vendor paths")
    m, n, k = (17, 9, 13) if batch == 1 else (65, 49, 97)
    dimensions = 2 if batch == 1 else 3

    @ti.kernel
    def energy(
        c: ti.types.ndarray(ti.f32, dimensions, needs_grad=True), loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)
    ):
        for index in ti.grouped(c):
            loss[0] += 0.5 * c[index] * c[index]

    provider = CublasLtProvider(vendor)
    semantics = dict(
        m=m,
        n=n,
        k=k,
        batch_count=batch,
        transpose_a=False,
        transpose_b=False,
        alpha=0.75,
        beta=0.25,
        a="a",
        b="b",
        output="c",
    )
    forward = _MatmulRecipePlan(
        provider,
        semantics,
        layout=layout,
        epilogue=epilogue,
        workspace_limit_bytes=32 << 20,
    )
    plans = {
        "forward": forward,
        "adjoint_a": provider.plan(m, k, n, batch_count=batch, transpose_b=True, alpha=0.75),
        "adjoint_b": provider.plan(k, n, m, batch_count=batch, transpose_a=True, alpha=0.75),
        "square": provider.plan(4, 4, 4, alpha=0.75, beta=0.25),
    }
    descriptions = {name: native_plan(name, frozen_parameters(plan)) for name, plan in plans.items()}
    resources = [descriptions[name] for name in ("forward", "adjoint_a", "adjoint_b")]
    resources.extend(
        [
            dict(kind="ndarray", name="dc", dtype="f32", shape=list(forward.output_shape)),
            dict(kind="ndarray", name="dloss", dtype="f32", shape=[1]),
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
    )
    operations = [
        dict(kind="fill32", destination="loss", value=0),
        dict(kind="fill32", destination="dc", value=0),
        dict(kind="fill32", destination="dloss", value=1065353216),
        dict(kind="copy", source="seed", destination="c"),
        matmul("forward", "a", "b", "c"),
        dict(kind="kernel", entry="energy", arguments=["c", "loss"]),
        dict(kind="kernel", entry="energy_grad", arguments=["c", "loss"]),
        matmul("adjoint_a", "dc", "b", "da"),
        matmul("adjoint_b", "a", "dc", "db"),
    ]
    module = ti.aot.Module()
    module.add_kernel(energy)
    module.add_kernel(energy.grad, name="energy_grad")
    module.add_execution_plan(
        "energy", ti.aot.ExecutionPlan(operations, resources=resources, gradients={"c": "dc", "loss": "dloss"})
    )
    module.add_execution_plan(
        "square", ti.aot.ExecutionPlan([matmul("square", "a", "b", "c")], resources=[descriptions["square"]])
    )
    artifact = tmp_path / "cublaslt.tcm"
    module.archive(str(artifact))
    with zipfile.ZipFile(artifact) as archive:
        for name in archive.namelist():
            data = archive.read(name)
            assert adapter.encode() not in data and vendor.encode() not in data
    for plan in plans.values():
        plan.close()
    provider.close()
    ti.reset()
    for mode in ("standalone", "borrowed"):
        command = [sys.executable] + (["-S"] if mode == "standalone" else [])
        result = subprocess.run(
            [*command, str(Path(__file__).with_name("aot_cublaslt_probe.py")), mode, str(artifact), adapter, vendor],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr
