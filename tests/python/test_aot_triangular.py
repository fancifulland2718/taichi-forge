"""Retained triangular analysis and explicit refresh use the common AOT registry."""

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
def test_triangular_provider_is_not_advertised_on_other_backends():
    from taichi_forge_runtime.aot import Runtime

    arch = "cpu" if ti.lang.impl.current_cfg().arch == ti.cpu else "vulkan"
    with Runtime(arch) as runtime:
        for suffix in ("solve", "update"):
            assert not runtime.probe_native_operation(f"cusparse_triangular_{suffix}_f32")["available"]


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_aot_triangular_updates_adjoint_and_independent_bindings(tmp_path):
    if not (ti.hardware.linalg.cusparse_spsv_is_available() and ti.hardware.linalg.cusparse_spsm_is_available()):
        pytest.skip("External cuSPARSE triangular solve is unavailable")
    module = ti.aot.Module()
    for name, rhs, dtype, fill, unit, transpose in (
        ("vector", 1, "i32", "lower", False, False),
        ("matrix", 2, "i64", "lower", False, False),
        ("upper", 1, "i64", "upper", False, True),
        ("unit_transpose", 2, "i32", "upper", True, True),
    ):
        resources = []
        for port, data in (("offsets", [0, 3, 6, 9]), ("columns", [0, 1, 2] * 3)):
            resources.append(
                dict(
                    kind="ndarray",
                    name=port,
                    dtype=dtype,
                    shape=[len(data)],
                    initial_data=module.add_data(array.array("i" if dtype == "i32" else "q", data)),
                )
            )
        resources.append(
            dict(
                native(
                    "csr_f32", dict(row_offsets="offsets", column_indices="columns", values="values"), rows=3, columns=3
                ),
                name="matrix",
            )
        )
        for plan, source, output, transposed in (
            ("forward", "x", "y", transpose),
            ("adjoint", "y", "dx", not transpose),
        ):
            resources.append(
                dict(
                    native(
                        "triangular_plan_f32",
                        dict(matrix="matrix", input=source, output=output),
                        rhs_count=rhs,
                        fill_mode=fill,
                        unit_diagonal=unit,
                        transpose=transposed,
                        mutable_values=True,
                    ),
                    name=plan,
                )
            )
        operations = [dict(kind="copy", source="coefficients", destination="values")]
        operations += [native("triangular_update_f32", dict(plan=p)) for p in ("forward", "adjoint")]
        operations += [native("triangular_solve_f32", dict(plan=p)) for p in ("forward", "adjoint")]
        module.add_execution_plan(name, ti.aot.ExecutionPlan(operations, resources=resources))
    artifact = tmp_path / "triangular.tcm"
    module.archive(str(artifact))
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_triangular_probe.py")),
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
