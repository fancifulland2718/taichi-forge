"""CSR SpMM uses the common native registry, resources and adjoint kernels."""

import array
import os
from pathlib import Path
import subprocess
import sys

import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.test_aot_csr import native


@test_utils.test(arch=[ti.cpu, ti.vulkan])
def test_spmm_is_not_advertised_on_other_backends():
    from taichi_forge_runtime.aot import Runtime

    with Runtime("cpu" if ti.lang.impl.current_cfg().arch == ti.cpu else "vulkan") as runtime:
        assert not runtime.probe_native_operation("cusparse_spmm_f32")["available"]


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_aot_spmm_fixed_pattern_updates_and_adjoint(tmp_path):
    if not ti.hardware.linalg.cusparse_spmm_is_available():
        pytest.skip("External cuSPARSE SpMM is unavailable")

    @ti.kernel
    def energy(y: ti.types.ndarray(ti.f32, 2, needs_grad=True), loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i, j in y:
            loss[0] += 0.5 * y[i, j] * y[i, j]

    module = ti.aot.Module()
    module.add_kernel(energy)
    module.add_kernel(energy.grad, name="backward")
    for name, algorithm, preprocess, dtype, rhs in (
        ("row", "csr_alg2", "none", "i32", 3),
        ("direct", "csr_alg3", "none", "i64", 3),
        ("optional", "csr_alg3", "optional", "i32", 3),
        ("prepared", "csr_alg3", "required", "i64", 3),
        ("single", "csr_alg2", "none", "i64", 1),
    ):
        resources = []
        for key, rows, cols, offsets, indices, values in (
            ("matrix", 3, 4, [0, 2, 2, 4], [0, 2, 1, 3], "values"),
            ("transpose", 4, 3, [0, 1, 2, 3, 4], [0, 2, 0, 2], "transpose_values"),
        ):
            for suffix, data in (("offsets", offsets), ("indices", indices)):
                resources.append(
                    dict(
                        kind="ndarray",
                        name=key + suffix,
                        dtype=dtype,
                        shape=[len(data)],
                        initial_data=module.add_data(array.array("i" if dtype == "i32" else "q", data)),
                    )
                )
            resources.append(
                dict(
                    native(
                        "csr_f32",
                        dict(row_offsets=key + "offsets", column_indices=key + "indices", values=values),
                        rows=rows,
                        columns=cols,
                    ),
                    name=key,
                )
            )
        resources += [
            dict(kind="ndarray", name="dy", dtype="f32", shape=[3, rhs]),
            dict(kind="ndarray", name="dloss", dtype="f32", shape=[1]),
        ]
        operations = [
            dict(kind="copy", source="y0", destination="y"),
            native(
                "spmm_f32",
                dict(matrix="matrix", input="x", output="y"),
                rhs_count=rhs,
                algorithm=algorithm,
                preprocess=preprocess,
                alpha=1.5,
                beta=-0.25,
            ),
        ]
        operations += [dict(kind="fill32", destination=n, value=0) for n in ("dy", "loss")]
        operations += [
            dict(kind="fill32", destination="dloss", value=0x3F800000),
            dict(kind="kernel", entry="energy", arguments=["y", "loss"]),
            dict(kind="kernel", entry="backward", arguments=["y", "loss"]),
            native(
                "spmm_f32",
                dict(matrix="transpose", input="dy", output="dx"),
                rhs_count=rhs,
                algorithm=algorithm,
                preprocess=preprocess,
                alpha=1.5,
            ),
        ]
        module.add_execution_plan(
            name, ti.aot.ExecutionPlan(operations, resources=resources, gradients=dict(y="dy", loss="dloss"))
        )
    artifact = tmp_path / "spmm.tcm"
    module.archive(str(artifact))
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_spmm_probe.py")),
                mode,
                str(artifact),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr
