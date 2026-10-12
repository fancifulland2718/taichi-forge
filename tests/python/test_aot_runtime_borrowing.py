import os
from pathlib import Path
import subprocess
import sys

import pytest
import taichi_forge as ti
from taichi_forge._lib import core
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_borrowed_aot_owner_and_retirement(tmp_path):
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

    counter = ti.field(ti.i32, shape=())

    @ti.kernel
    def next_count(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        counter[None] += 1
        out[0] = counter[None]

    module = ti.aot.Module()
    module.add_field("counter", counter)
    module.add_kernel(forward)
    module.add_kernel(forward.grad, name="backward")
    module.add_kernel(next_count)
    graph = ti.graph.GraphBuilder()
    x = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "x", ti.f32, ndim=1)
    y = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "y", ti.f32, ndim=1)
    graph.dispatch(forward, x, y)
    graph.dispatch(forward.grad, x, y)
    module.add_graph("differentiate", graph.compile())
    module.add_execution_plan(
        "private_gradient",
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
    artifact = tmp_path / "borrowed.tcm"
    module.archive(str(artifact))
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("aot_borrowing_probe.py")), arch, str(artifact)],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Unable to release execution resource" not in result.stdout + result.stderr
