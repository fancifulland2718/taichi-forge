"""AOT kernel parameters bind the same managed TLAS as native operations."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.aot_kernel_as_probe import plan_parts


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_aot_kernel_acceleration_structure(tmp_path):
    if not ti.hardware.ray.is_available():
        pytest.skip("Vulkan ray query unavailable")

    @ti.func
    def accept(hit: ti.template(), gates: ti.template()):
        return gates[hit.instance_id] != 0

    @ti.kernel
    def query(
        scene: ti.types.acceleration_structure(),
        rays: ti.types.ndarray(ti.f32, 2),
        gates: ti.types.ndarray(ti.i32, 1),
        hits: ti.types.ndarray(ti.f32, 2),
        ids: ti.types.ndarray(ti.i32, 2),
        occluded: ti.types.ndarray(ti.i32, 1),
    ):
        for i in range(rays.shape[0]):
            origin = ti.Vector([rays[i, 0], rays[i, 1], rays[i, 2]])
            direction = ti.Vector([rays[i, 4], rays[i, 5], rays[i, 6]])
            h = scene.trace_closest_filtered(
                origin, direction, accept, args=(gates,), t_min=rays[i, 3], t_max=rays[i, 7], respect_opacity=True
            )
            any_hit = scene.trace_any_filtered(
                origin, direction, accept, args=(gates,), t_min=rays[i, 3], t_max=rays[i, 7], respect_opacity=True
            )
            hits[i, 0] = h.t
            hits[i, 1] = h.barycentric_u
            hits[i, 2] = h.barycentric_v
            hits[i, 3] = 0
            ids[i, 0] = ti.cast(h.primitive_index, ti.i32)
            ids[i, 1] = ti.cast(h.instance_id, ti.i32)
            ids[i, 2] = ti.cast(h.instance_custom_index, ti.i32)
            ids[i, 3] = h.hit
            occluded[i] = any_hit.hit

    @ti.kernel
    def energy(hits: ti.types.ndarray(ti.f32, 2, needs_grad=True), loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in range(hits.shape[0]):
            if hits[i, 0] > 0:
                loss[0] += 0.5 * hits[i, 0] * hits[i, 0]

    start = time.perf_counter()
    module = ti.aot.Module(caps=["spirv_has_ray_query"])
    module.add_kernel(query)
    module.add_kernel(energy)
    module.add_kernel(energy.grad, name="backward")
    for private in (True, False):
        module.add_execution_plan("private" if private else "public", ti.aot.ExecutionPlan(**plan_parts(private)))
    costs = dict(compile_ms=(time.perf_counter() - start) * 1000)
    artifact = tmp_path / "kernel-as.tcm"
    module.archive(str(artifact))
    costs["artifact_bytes"] = artifact.stat().st_size
    (tmp_path / "producer.json").write_text(json.dumps(costs, indent=2))
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_kernel_as_probe.py")),
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
