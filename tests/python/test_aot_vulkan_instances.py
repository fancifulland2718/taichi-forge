"""AOT Vulkan independent geometry and instance resources."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.aot_vulkan_instances_probe import plan_parts


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_instances_aot_reconstruction(tmp_path):
    if not ti.hardware.ray.is_available():
        pytest.skip("Vulkan ray query unavailable")

    @ti.kernel
    def move(
        seed: ti.types.ndarray(ti.f32, 2),
        poses: ti.types.ndarray(ti.f32, 2),
        control: ti.types.ndarray(ti.f32, 1),
        vertices: ti.types.ndarray(ti.f32, 2),
        transforms: ti.types.ndarray(ti.f32, 2),
    ):
        for i, j in seed:
            vertices[i, j] = seed[i, j] + (control[0] if j == 2 else 0.0)
        for i, j in poses:
            transforms[i, j] = poses[i, j] + (control[1] if j == 3 else 0.0)

    @ti.kernel
    def energy(hits: ti.types.ndarray(ti.f32, 2, needs_grad=True), loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in range(hits.shape[0]):
            if hits[i, 0] > 0:
                loss[0] += 0.5 * hits[i, 0] * hits[i, 0]

    start = time.perf_counter()
    module = ti.aot.Module()
    module.add_kernel(move)
    module.add_kernel(energy)
    module.add_kernel(energy.grad, name="energy_grad")
    for private in (False, True):
        operations, resources = plan_parts(private)
        module.add_execution_plan(
            "private" if private else "public",
            ti.aot.ExecutionPlan(operations, resources=resources, gradients=dict(hits="dhits", loss="dloss")),
        )
    compile_ms = (time.perf_counter() - start) * 1000
    artifact = tmp_path / "vulkan_instances.tcm"
    module.archive(str(artifact))
    (tmp_path / "producer.json").write_text(
        json.dumps(dict(compile_ms=compile_ms, artifact_bytes=artifact.stat().st_size))
    )
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_vulkan_instances_probe.py")),
                mode,
                str(artifact),
                str(tmp_path / f"{mode}.json"),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Validation Error" not in result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr
