import os
from pathlib import Path
import subprocess
import sys

import pytest
import taichi_forge as ti
from taichi_forge._lib import core
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_borrowed_aot_ndarray_import(tmp_path):
    if not hasattr(core, "_borrow_aot_runtime"):
        pytest.skip("Requires a split runtime containing the C API")
    arch = str(ti.lang.impl.current_cfg().arch).split(".")[-1]
    if arch in ("x64", "arm64"):
        arch = "cpu"

    @ti.kernel
    def forward(
        x: ti.types.ndarray(dtype=ti.f32, ndim=1, needs_grad=True),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1, needs_grad=True),
    ):
        for i in x:
            y[i] = x[i] * x[i] + 2 * x[i]

    module = ti.aot.Module()
    module.add_kernel(forward)
    module.add_kernel(forward.grad, name="backward")
    module.add_execution_plan(
        "differentiate",
        ti.aot.ExecutionPlan(
            [
                {"kind": "fill32", "destination": "dx", "value": 0},
                {"kind": "fill32", "destination": "dy", "value": 1065353216},
                {"kind": "kernel", "entry": "forward", "arguments": ["x", "y"]},
                {"kind": "kernel", "entry": "backward", "arguments": ["x", "y"]},
            ],
            resources=[{"kind": "ndarray", "name": "dy", "dtype": "f32", "shape": [{"array": "x", "axis": 0}]}],
            gradients={"x": "dx", "y": "dy"},
        ),
    )
    artifact = tmp_path / "imported.tcm"
    module.archive(str(artifact))
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("aot_ndarray_import_probe.py")), arch, str(artifact)],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ndarray import OK" in result.stdout
    assert "Unable to release execution resource" not in result.stdout + result.stderr
