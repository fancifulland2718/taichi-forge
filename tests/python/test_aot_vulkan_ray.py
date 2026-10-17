"""Native Vulkan ray resources in runtime-only and JIT-borrowed consumers."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.aot_vulkan_ray_probe import plan_parts


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_vulkan_ray_not_registered_on_other_backends():
    with ti.aot.borrow_runtime() as runtime:
        for kind in ("vulkan_ray_query", "vulkan_ray_refit"):
            assert not runtime.probe_native_operation(kind)["available"]


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_ray_aot_reconstruction_and_gradients(tmp_path):
    if not ti.hardware.ray.is_available():
        pytest.skip("Vulkan ray query unavailable")

    @ti.kernel
    def move(
        seed: ti.types.ndarray(ti.f32, 2), shift: ti.types.ndarray(ti.f32, 1), vertices: ti.types.ndarray(ti.f32, 2)
    ):
        for i, j in seed:
            vertices[i, j] = seed[i, j] + (shift[0] if j == 2 else 0.0)

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
    artifact = tmp_path / "vulkan_ray.tcm"
    module.archive(str(artifact))
    (tmp_path / "producer.json").write_text(
        json.dumps(dict(compile_ms=compile_ms, artifact_bytes=artifact.stat().st_size))
    )
    ti.reset()
    for mode in ("standalone", "borrowed"):
        command = [sys.executable] + (["-S"] if mode == "standalone" else [])
        result = subprocess.run(
            [
                *command,
                str(Path(__file__).with_name("aot_vulkan_ray_probe.py")),
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
