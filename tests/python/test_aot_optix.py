"""OptiX resources and ordered work, without producer ray objects at consumption."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.aot_optix_probe import plan_parts


@test_utils.test(arch=[ti.cpu, ti.vulkan])
def test_optix_does_not_advertise_non_cuda_execution():
    with ti.aot.borrow_runtime() as runtime:
        assert not runtime.probe_native_operation("optix_query")["available"]
        assert not runtime.probe_native_operation("optix_refit")["available"]


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_optix_aot_resources_queries_refit_and_post_query_gradients(tmp_path):
    adapter = os.environ.get("TAICHI_FORGE_TEST_OPTIX_PROVIDER")
    if not adapter:
        pytest.skip("explicit common OptiX adapter required")

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
    for policy in ("two_sided", "front_only", "table"):
        operations, resources = plan_parts(policy)
        module.add_execution_plan(
            policy, ti.aot.ExecutionPlan(operations, resources=resources, gradients=dict(hits="dhits", loss="dloss"))
        )
    build_ms = (time.perf_counter() - start) * 1000
    artifact = tmp_path / "optix.tcm"
    start = time.perf_counter()
    module.archive(str(artifact))
    archive_ms = (time.perf_counter() - start) * 1000
    assert len(module._exported_kernels) == 3
    assert not module._data_payloads
    with zipfile.ZipFile(artifact) as archive:
        assert all(adapter.encode() not in archive.read(name) for name in archive.namelist())
    (tmp_path / "producer.json").write_text(
        json.dumps(dict(build_ms=build_ms, archive_ms=archive_ms, bytes=artifact.stat().st_size))
    )
    ti.reset()
    for mode in ("standalone", "borrowed"):
        command = [sys.executable] + (["-S"] if mode == "standalone" else [])
        result = subprocess.run(
            [
                *command,
                str(Path(__file__).with_name("aot_optix_probe.py")),
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
