"""Conditional tensor adjoints retain conservative component aliasing."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_conditional_tensor_adjoint_jit_and_aot(tmp_path):
    @ti.kernel
    def energy(x: ti.types.ndarray(ti.math.vec4, 1, needs_grad=True), y: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in x:
            if x[i][0] > 0:
                y[0] += 0.5 * x[i][0] * x[i][0]

    start = time.perf_counter()
    module = ti.aot.Module()
    module.add_kernel(energy)
    module.add_kernel(energy.grad, name="backward")
    module.add_execution_plan(
        "step",
        ti.aot.ExecutionPlan(
            [
                {"kind": "fill32", "destination": "y", "value": 0},
                {"kind": "fill32", "destination": "dx", "value": 0},
                {"kind": "fill32", "destination": "dy", "value": 1065353216},
                {"kind": "kernel", "entry": "energy", "arguments": ["x", "y"]},
                {"kind": "kernel", "entry": "backward", "arguments": ["x", "y"]},
            ],
            gradients=dict(x="dx", y="dy"),
        ),
    )
    costs = dict(build_ms=(time.perf_counter() - start) * 1000)
    artifact = tmp_path / "conditional.tcm"
    start = time.perf_counter()
    module.archive(str(artifact))
    costs.update(archive_ms=(time.perf_counter() - start) * 1000, bytes=artifact.stat().st_size)
    x = ti.ndarray(ti.math.vec4, 4, needs_grad=True)
    y = ti.ndarray(ti.f32, 1, needs_grad=True)
    for scale in (1, -2, 0):
        values = np.array([[v * scale, 9, 8, 7] for v in (1, -2, 0, 3)], dtype=np.float32)
        x.from_numpy(values)
        x.grad.fill(0)
        y.fill(0)
        y.grad.fill(1)
        energy(x, y)
        energy.grad(x, y)
        expected = np.zeros_like(values)
        expected[:, 0] = np.maximum(values[:, 0], 0)
        np.testing.assert_array_equal(x.grad.to_numpy(), expected)
        np.testing.assert_array_equal(y.to_numpy(), [0.5 * np.sum(expected[:, 0] ** 2)])
    arch = {ti.cpu: "cpu", ti.cuda: "cuda", ti.vulkan: "vulkan"}[ti.cfg.arch]
    ti.reset()
    result = subprocess.run(
        [sys.executable, "-S", str(Path(__file__).with_name("aot_tensor_alias_grad_probe.py")), arch, str(artifact)],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    costs["consumer"] = json.loads(next(line for line in result.stdout.splitlines() if line.startswith('{"load_ms"')))
    (tmp_path / "costs.json").write_text(json.dumps(costs, indent=2))
