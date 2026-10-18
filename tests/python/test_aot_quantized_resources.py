"""Packed integer storage is independent across source-free AOT modules."""

import os
from pathlib import Path
import subprocess
import sys

import pytest
import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("layout", ["bitpacked", "quant_array"])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, vulkan_quant_experimental=True)
def test_aot_quantized_module_isolation(tmp_path, layout):
    unused = ti.field(ti.i32, shape=1)
    unused[0] = 17
    x = ti.field(ti.types.quant.int(7, signed=True))
    y = ti.field(ti.types.quant.int(9, signed=False)) if layout == "bitpacked" else None
    fields = ti.FieldsBuilder()
    if layout == "bitpacked":
        packed = ti.BitpackedFields(max_num_bits=32)
        packed.place(x, y)
        fields.dense(ti.i, 8).place(packed)
    else:
        fields.dense(ti.i, 2).quant_array(ti.i, 4, max_num_bits=32).place(x)
    fields.finalize()

    @ti.kernel
    def seed(offset: ti.i32):
        for i in range(8):
            x[i] = offset + i
            if ti.static(layout == "bitpacked"):
                y[i] = i * 2 + 7

    @ti.kernel
    def gather(out: ti.types.ndarray(ti.i32, 1)):
        for i in x:
            out[0] += x[i]
            if ti.static(layout == "bitpacked"):
                out[1] += y[i]

    @ti.kernel
    def energy(
        weight: ti.types.ndarray(ti.f32, 1, needs_grad=True),
        loss: ti.types.ndarray(ti.f32, 1, needs_grad=True),
    ):
        for i in x:
            loss[0] += ti.cast(x[i], ti.f32) * weight[0]

    arch = {ti.cpu: "cpu", ti.cuda: "cuda", ti.vulkan: "vulkan"}[ti.lang.impl.current_cfg().arch]
    module = ti.aot.Module(**({"artifact_format": "ptx"} if arch == "cuda" else {}))
    for kernel in (seed, gather, energy):
        module.add_kernel(kernel)
    module.add_kernel(energy.grad, name="backward")
    artifact = tmp_path / "packed.tcm"
    module.archive(str(artifact))
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_quantized_resources_probe.py")),
                arch,
                mode,
                str(artifact),
                layout,
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Failed to retire" not in result.stdout + result.stderr
