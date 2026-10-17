"""AOT borrows existing Program textures without copying or replacing samplers."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_aot_borrowed_texture_lifetime_and_updates(tmp_path):
    @ti.kernel
    def sample(
        image: ti.types.texture(num_dimensions=2),
        alias: ti.types.texture(num_dimensions=2),
        output: ti.types.ndarray(ti.f32, 1),
    ):
        output[0] = image.fetch(ti.Vector([0, 4]), 0).x
        output[1] = image.fetch(ti.Vector([2, 1]), 0).x
        output[2] = image.sample_lod(ti.Vector([0.5, 0.25]), 0.0).x
        output[3] = image.sample_lod(ti.Vector([1.1, 0.9]), 0.0).x
        output[4] = alias.fetch(ti.Vector([1, 2]), 0).x

    @ti.kernel
    def energy(
        output: ti.types.ndarray(ti.f32, 1, needs_grad=True),
        loss: ti.types.ndarray(ti.f32, 1, needs_grad=True),
    ):
        for i in output:
            loss[0] += 0.5 * output[i] * output[i]

    start = time.perf_counter()
    module = ti.aot.Module()
    module.add_kernel(sample)
    module.add_kernel(energy)
    module.add_kernel(energy.grad, name="backward")
    module.add_execution_plan(
        "step",
        ti.aot.ExecutionPlan(
            [
                dict(
                    kind="kernel",
                    entry="sample",
                    arguments=["image", "alias", "output"],
                ),
                dict(kind="fill32", destination="doutput", value=0),
                dict(kind="fill32", destination="loss", value=0),
                dict(kind="fill32", destination="dloss", value=1065353216),
                dict(kind="kernel", entry="energy", arguments=["output", "loss"]),
                dict(kind="kernel", entry="backward", arguments=["output", "loss"]),
            ],
            gradients=dict(output="doutput", loss="dloss"),
        ),
    )
    costs = dict(compile_ms=(time.perf_counter() - start) * 1000)
    artifact = tmp_path / "borrowed-texture.tcm"
    module.archive(str(artifact))
    costs["bytes"] = artifact.stat().st_size
    arch = ti.cfg.arch.name
    ti.reset()
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("aot_texture_borrowing_probe.py")),
            arch,
            str(artifact),
            str(tmp_path / "consumer.json"),
        ],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    (tmp_path / "consumer.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Validation Error" not in result.stdout + result.stderr
    assert "cleanup failed" not in result.stdout + result.stderr
    (tmp_path / "producer.json").write_text(json.dumps(costs, indent=2))
