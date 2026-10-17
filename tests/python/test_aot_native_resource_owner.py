"""Public native owners share factories, execution and retirement across plans."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_aot_public_native_resource_ownership(tmp_path):
    arch = "cpu" if ti.cfg.arch in (ti.cpu, ti.x64, ti.arm64) else ti.cfg.arch.name
    module = ti.aot.Module()

    @ti.kernel
    def energy(
        output: ti.types.ndarray(ti.f32, 1, needs_grad=True),
        loss: ti.types.ndarray(ti.f32, 1, needs_grad=True),
    ):
        for i in output:
            loss[0] += 0.5 * output[i] ** 2

    @ti.kernel
    def sample(
        image: ti.types.texture(num_dimensions=2),
        alias: ti.types.texture(num_dimensions=2),
        output: ti.types.ndarray(ti.f32, 1),
    ):
        output[0] = image.fetch(ti.Vector([0, 4]), 0).x
        output[1] = image.fetch(ti.Vector([2, 1]), 0).x
        output[2] = alias.fetch(ti.Vector([1, 2]), 0).x
        output[3] = alias.sample_lod(ti.Vector([0.5, 0.5]), 0.0).x

    start = time.perf_counter()
    module.add_kernel(energy)
    module.add_kernel(energy.grad, name="backward")
    operations = []
    if arch != "cpu":
        module.add_kernel(sample)
        operations.append(
            dict(kind="kernel", entry="sample", arguments=["image", "alias", "output"])
        )
    operations += [
        dict(kind="fill32", destination="doutput", value=0),
        dict(kind="fill32", destination="loss", value=0),
        dict(kind="fill32", destination="dloss", value=1065353216),
        dict(kind="kernel", entry="energy", arguments=["output", "loss"]),
        dict(kind="kernel", entry="backward", arguments=["output", "loss"]),
    ]
    module.add_execution_plan(
        "step",
        ti.aot.ExecutionPlan(
            operations, gradients=dict(output="doutput", loss="dloss")
        ),
    )
    costs = dict(compile_ms=(time.perf_counter() - start) * 1000)
    artifact = tmp_path / "resources.tcm"
    module.archive(str(artifact))
    costs["bytes"] = artifact.stat().st_size
    ti.reset()
    adapter = (
        os.environ.get("TAICHI_FORGE_TEST_OPTIX_PROVIDER", "") if arch == "cuda" else ""
    )
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_native_resource_owner_probe.py")),
                mode,
                arch,
                str(artifact),
                adapter,
                str(tmp_path / f"{mode}.json"),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Validation Error" not in result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr
    (tmp_path / "producer.json").write_text(json.dumps(costs, indent=2))
