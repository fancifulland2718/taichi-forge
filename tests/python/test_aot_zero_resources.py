"""Private storage starts at zero once, with no extra kernel in its artifact."""

import os
from pathlib import Path
import subprocess
import sys

import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_aot_zero_resources_and_explicit_gradient(tmp_path):
    @ti.kernel
    def advance(x: ti.types.ndarray(ti.f32, ndim=1), state: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True)):
        for i in x:
            state[i] = state[i] + x[i]

    @ti.kernel
    def square(
        state: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True),
        y: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True),
    ):
        for i in state:
            y[i] = state[i] * state[i]

    @ti.kernel
    def copy_bytes(source: ti.types.ndarray(ti.u8, ndim=1), output: ti.types.ndarray(ti.u8, ndim=1)):
        for i in output:
            output[i] = source[i]

    caps = (
        [ti.DeviceCapability.spirv_has_int8, "spirv_has_int8_storage"]
        if ti.lang.impl.current_cfg().arch == ti.vulkan
        else []
    )
    module = ti.aot.Module(caps=caps)
    module.add_kernel(advance)
    module.add_kernel(square)
    module.add_kernel(square.grad, name="backward")
    module.add_kernel(copy_bytes)
    module.add_execution_plan(
        "state",
        ti.aot.ExecutionPlan(
            [
                dict(kind="kernel", entry="advance", arguments=["x", "state"]),
                dict(kind="kernel", entry="square", arguments=["state", "y"]),
                dict(kind="fill32", destination="ds", value=0),
                dict(kind="fill32", destination="dy", value=1065353216),
                dict(kind="kernel", entry="backward", arguments=["state", "y"]),
                dict(kind="copy", source="ds", destination="dx"),
            ],
            resources=[
                dict(kind="ndarray", name=key, dtype="f32", shape=[17], zero_initialize=True) for key in ("state", "ds")
            ],
            gradients={"state": "ds", "y": "dy"},
        ),
    )
    artifact = tmp_path / "zero-resources.tcm"
    module.archive(str(artifact))
    arch = "cpu" if ti.lang.impl.current_cfg().arch == ti.cpu else str(ti.lang.impl.current_cfg().arch).split(".")[-1]
    for mode in ("standalone", "borrowed"):
        command = [sys.executable] + (["-S"] if mode == "standalone" else [])
        result = subprocess.run(
            [*command, str(Path(__file__).with_name("aot_zero_resources_probe.py")), mode, arch, str(artifact)],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr
        assert "Unable to release execution resource" not in result.stdout + result.stderr
