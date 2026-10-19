"""Independent CPU AOT storage survives another owner's reset or close."""

import os
from pathlib import Path
import subprocess
import sys

import pytest
import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("reset_arch", ["cpu", "cuda", "vulkan"])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_cpu_aot_owner_survives_jit_reset(tmp_path, reset_arch):
    if not ti.lang.misc.is_arch_supported(getattr(ti, reset_arch)):
        pytest.skip(f"{reset_arch} is unavailable")

    @ti.kernel
    def forward(
        x: ti.types.ndarray(ti.f32, 1, needs_grad=True),
        y: ti.types.ndarray(ti.f32, 1, needs_grad=True),
    ):
        for i in x:
            y[i] = x[i] * x[i] + 2 * x[i]

    sparse = ti.field(ti.i32)
    ti.root.pointer(ti.i, 4).dense(ti.i, 2).place(sparse)

    @ti.kernel
    def touch(index: ti.i32, out: ti.types.ndarray(ti.i32, 1)):
        sparse[index] += 1
        out[0] = 0
        for i in sparse:
            out[0] += sparse[i]

    module = ti.aot.Module()
    module.add_kernel(forward)
    module.add_kernel(forward.grad, name="backward")
    module.add_kernel(touch)
    artifact = tmp_path / "cpu-owner.tcm"
    module.archive(str(artifact))
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("aot_cpu_owner_probe.py")), reset_arch, str(artifact)],
        env={**os.environ, "TI_SKIP_VERSION_CHECK": "ON"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CPU AOT owner isolation passed" in result.stdout
    assert "Unable to release execution resource" not in result.stdout + result.stderr
    assert "VUID-" not in result.stdout + result.stderr
