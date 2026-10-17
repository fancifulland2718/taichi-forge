"""Native SPIR-V compute reconstruction shares binding and execution contracts."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import taichi_forge as ti
from tests import test_utils
from tests.python.aot_vulkan_compute_probe import operations


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_aot_compute_backend_isolation():
    with ti.aot.borrow_runtime() as runtime:
        assert not runtime.native_operation_capabilities("vulkan_compute_dispatch")
        assert not runtime.native_resource_capabilities("vulkan_compute_pipeline")


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_aot_vulkan_compute_reconstruction(tmp_path):
    @ti.kernel
    def energy(
        values: ti.types.ndarray(ti.f32, 1, needs_grad=True), loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)
    ):
        for i in values:
            loss[0] += 0.5 * values[i] ** 2

    start = time.perf_counter()
    module = ti.aot.Module()
    module.add_kernel(energy)
    module.add_kernel(energy.grad, name="backward")
    source = Path(__file__).with_name("assets") / "aot_compute.comp.spv"
    code = source.read_bytes()
    contract = hashlib.sha256(code).hexdigest()
    pipeline = dict(
        kind="native",
        name="pipeline",
        provider="vulkan_compute",
        operation="pipeline",
        version=1,
        parameters=dict(expected_shader=contract),
        bindings=dict(shader="shader"),
    )
    scratch = [dict(kind="ndarray", name="temp", dtype="f32", shape=[16])]
    for name in ("private", "public"):
        resources = scratch.copy()
        if name == "private":
            resources += [
                dict(kind="host_blob", name="shader", data=module.add_data(code), byte_size=len(code)),
                pipeline,
            ]
        module.add_execution_plan(
            name,
            ti.aot.ExecutionPlan(
                operations(contract), resources=resources, gradients=dict(output="doutput", loss="dloss")
            ),
        )
    artifact = tmp_path / "compute.tcm"
    module.archive(str(artifact))
    (tmp_path / "producer.json").write_text(
        json.dumps(dict(compile_ms=(time.perf_counter() - start) * 1000, bytes=artifact.stat().st_size))
    )
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_vulkan_compute_probe.py")),
                mode,
                str(artifact),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Validation Error" not in result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr
