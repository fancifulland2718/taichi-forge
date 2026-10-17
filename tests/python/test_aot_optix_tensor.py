"""AOT OptiX shares the JIT packed vector and transform layouts."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.aot_optix_tensor_probe import plan_parts


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_optix_tensor_inputs_outputs_refit_and_gradients(tmp_path):
    adapter = os.environ.get("TAICHI_FORGE_TEST_OPTIX_PROVIDER")
    if not adapter:
        pytest.skip("explicit common OptiX adapter required")

    vec3, vec4, mat34 = ti.types.vector(3, ti.f32), ti.types.vector(4, ti.f32), ti.types.matrix(3, 4, ti.f32)

    @ti.kernel
    def animate(
        seed: ti.types.ndarray(vec3, 1),
        shift: ti.types.ndarray(ti.f32, 1),
        vertices: ti.types.ndarray(vec3, 1),
        requested: ti.types.ndarray(mat34, 1),
        transforms: ti.types.ndarray(mat34, 1),
    ):
        for i in seed:
            vertices[i] = seed[i] + ti.Vector([shift[0], shift[1], shift[2]])
        for i in requested:
            transforms[i] = requested[i]

    @ti.kernel
    def energy(hits: ti.types.ndarray(vec4, 1, needs_grad=True), loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in hits:
            if hits[i][0] > 0:
                loss[0] += 0.5 * hits[i][0] * hits[i][0]

    start = time.perf_counter()
    module = ti.aot.Module()
    module.add_kernel(animate)
    module.add_kernel(energy)
    module.add_kernel(energy.grad, name="energy_grad")
    for policy in ("tensor", "scalar_transforms"):
        operations, resources = plan_parts(policy)
        module.add_execution_plan(
            policy, ti.aot.ExecutionPlan(operations, resources=resources, gradients=dict(hits="dhits", loss="dloss"))
        )
    build_ms = (time.perf_counter() - start) * 1000
    artifact = tmp_path / "tensor.tcm"
    start = time.perf_counter()
    module.archive(str(artifact))
    archive_ms = (time.perf_counter() - start) * 1000
    assert len(module._exported_kernels) == 3
    assert not module._data_payloads
    (tmp_path / "producer.json").write_text(
        json.dumps(dict(build_ms=build_ms, archive_ms=archive_ms, bytes=artifact.stat().st_size))
    )
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_optix_tensor_probe.py")),
                mode,
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
        assert "cleanup failed" not in result.stdout + result.stderr
