"""Fixed CSR topology uses the common native registry and source-free runtime."""

import array
import os
from pathlib import Path
import subprocess
import sys

import pytest
import taichi_forge as ti
from tests import test_utils


def native(operation, bindings, **parameters):
    return dict(
        kind="native", provider="cusparse", operation=operation, version=1, parameters=parameters, bindings=bindings
    )


@test_utils.test(arch=[ti.cpu, ti.vulkan])
def test_csr_is_not_advertised_on_other_backends():
    from taichi_forge_runtime.aot import Runtime

    with Runtime("cpu" if ti.lang.impl.current_cfg().arch == ti.cpu else "vulkan") as runtime:
        assert not runtime.probe_native_operation("cusparse_spmv_f32")["available"]


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_aot_csr_source_free_forward_adjoint_and_retirement(tmp_path):
    from taichi_forge_runtime.aot import Runtime

    with Runtime("cuda") as runtime:
        if not runtime.probe_native_operation("cusparse_spmv_f32")["available"]:
            pytest.skip("Compatible external cuSPARSE is unavailable")

    @ti.kernel
    def energy(
        y: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True), loss: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True)
    ):
        for i in y:
            loss[0] += 0.5 * y[i] * y[i]

    module = ti.aot.Module()
    module.add_kernel(energy)
    module.add_kernel(energy.grad, name="energy_grad")
    for dtype, code in (("i32", "i"), ("i64", "q")):
        resources = [
            dict(
                kind="ndarray",
                name=name,
                dtype=dtype,
                shape=[len(values)],
                initial_data=module.add_data(array.array(code, values)),
            )
            for name, values in (("offsets", [0, 2, 2, 5]), ("columns", [2, 0, 3, 0, 1]))
        ]
        resources += [
            dict(
                native(
                    "csr_f32", dict(row_offsets="offsets", column_indices="columns", values="values"), rows=3, columns=4
                ),
                name="matrix",
            )
        ]
        resources += [
            dict(kind="ndarray", name=name, dtype="f32", shape=[size]) for name, size in (("dy", 3), ("dloss", 1))
        ]
        operations = [dict(kind="fill32", destination=name, value=0) for name in ("dy", "loss")]
        operations += [
            dict(kind="fill32", destination="dloss", value=0x3F800000),
            dict(kind="copy", source="y0", destination="y"),
            native("spmv_f32", dict(matrix="matrix", input="x", output="y"), alpha=1.5, beta=-0.25),
            dict(kind="kernel", entry="energy", arguments=["y", "loss"]),
            dict(kind="kernel", entry="energy_grad", arguments=["y", "loss"]),
            native("spmv_f32", dict(matrix="matrix", input="dy", output="dx"), transpose=True, alpha=1.5),
        ]
        module.add_execution_plan(
            dtype, ti.aot.ExecutionPlan(operations, resources=resources, gradients={"y": "dy", "loss": "dloss"})
        )
    artifact = tmp_path / "csr.tcm"
    module.archive(str(artifact))
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_csr_probe.py")),
                mode,
                str(artifact),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "returned status" not in result.stdout + result.stderr
