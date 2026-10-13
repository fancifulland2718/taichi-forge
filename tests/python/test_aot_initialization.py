"""One-time initialization in source-free and borrowed execution owners."""

import os
from pathlib import Path
import subprocess
import sys

import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_aot_initialization_and_replay_retirement(tmp_path):
    arch = str(ti.lang.impl.current_cfg().arch).split(".")[-1]
    if arch in ("x64", "arm64"):
        arch = "cpu"

    @ti.kernel
    def initialize(x: ti.types.ndarray(ti.f32, ndim=1), snapshot: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True)):
        for i in x:
            snapshot[i] = x[i] * 2.0

    @ti.kernel
    def evaluate(
        snapshot: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True),
        y: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True),
    ):
        for i in y:
            y[i] = snapshot[i] * snapshot[i]

    module = ti.aot.Module()
    module.add_kernel(initialize)
    module.add_kernel(evaluate)
    module.add_kernel(evaluate.grad, name="backward")
    module.add_execution_plan(
        "snapshot",
        ti.aot.ExecutionPlan(
            [
                dict(kind="fill32", destination="ds", value=0),
                dict(kind="fill32", destination="dy", value=1065353216),
                dict(kind="kernel", entry="evaluate", arguments=["snapshot", "y"]),
                dict(kind="kernel", entry="backward", arguments=["snapshot", "y"]),
                dict(kind="copy", source="ds", destination="gradient"),
            ],
            initialization=[
                dict(kind="copy", source="x", destination="temporary"),
                dict(kind="kernel", entry="initialize", arguments=["temporary", "snapshot"]),
            ],
            resources=[
                dict(kind="ndarray", name=name, dtype="f32", shape=[17])
                for name in ("temporary", "snapshot", "ds", "dy")
            ],
            gradients={"snapshot": "ds", "y": "dy"},
        ),
    )
    artifact = tmp_path / "initialization.tcm"
    module.archive(str(artifact))
    for mode in ("standalone", "borrowed"):
        command = [sys.executable] + (["-S"] if mode == "standalone" else [])
        result = subprocess.run(
            [*command, str(Path(__file__).with_name("aot_initialization_probe.py")), mode, arch, str(artifact)],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr
