"""Live per-instance OptiX face tables are produced before each AOT query."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.aot_optix_live_faces_probe import POLICIES, plan_parts


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_optix_live_face_tables_change_without_repreparing(tmp_path):
    adapter = os.environ.get("TAICHI_FORGE_TEST_OPTIX_PROVIDER")
    if not adapter:
        pytest.skip("explicit common OptiX adapter required")

    @ti.kernel
    def update_rules(
        epoch: ti.types.ndarray(ti.i32, 1), first: ti.types.ndarray(ti.u32, 1), second: ti.types.ndarray(ti.u32, 1)
    ):
        first[0], first[1] = epoch[0], 1
        second[0], second[1] = 1 - epoch[0], 1

    @ti.kernel
    def energy(hits: ti.types.ndarray(ti.f32, 2, needs_grad=True), loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in range(hits.shape[0]):
            if hits[i, 0] > 0:
                loss[0] += 0.5 * hits[i, 0] * hits[i, 0]

    start = time.perf_counter()
    module = ti.aot.Module()
    module.add_kernel(update_rules)
    module.add_kernel(energy)
    module.add_kernel(energy.grad, name="energy_grad")
    for policy in POLICIES:
        operations, resources = plan_parts(policy)
        module.add_execution_plan(
            policy, ti.aot.ExecutionPlan(operations, resources=resources, gradients=dict(hits="dhits", loss="dloss"))
        )
    build_ms = (time.perf_counter() - start) * 1000
    artifact = tmp_path / "live_faces.tcm"
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
                str(Path(__file__).with_name("aot_optix_live_faces_probe.py")),
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
