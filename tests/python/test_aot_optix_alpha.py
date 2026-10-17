"""Explicit AOT images, GPU uploads and combined alpha/face traversal."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.vulkan])
def test_managed_alpha_does_not_advertise_unqualified_backends():
    with ti.aot.borrow_runtime() as runtime:
        assert not runtime.probe_native_operation("optix_query", version=3)["available"]
        assert not runtime.probe_native_operation("sampled_image")["available"]
        assert not runtime.probe_native_operation("image_upload")["available"]


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_aot_managed_alpha_uploads_and_post_query_gradients(tmp_path):
    adapter = os.environ.get("TAICHI_FORGE_TEST_OPTIX_PROVIDER")
    if not adapter:
        pytest.skip("explicit common OptiX adapter required")
    from tests.python.aot_optix_alpha_probe import plan_parts

    @ti.func
    def update(
        config: ti.template(),
        uva: ti.template(),
        uvb: ti.template(),
        faces: ti.template(),
    ):
        for i in range(3):
            uva[i, 0] = 0.75 if config[3] != 0 else 0.25
            uva[i, 1] = 0.875 if config[3] != 0 else 0.125
            uvb[i, 0] = 0.25 if config[3] != 0 else 0.75
            uvb[i, 1] = 0.125 if config[3] != 0 else 0.875
        faces[0] = ti.cast(config[2], ti.i32)

    @ti.kernel
    def paint_1(
        config: ti.types.ndarray(ti.f32, 1),
        pixels: ti.types.ndarray(ti.f32, 2),
        uva: ti.types.ndarray(ti.f32, 2),
        uvb: ti.types.ndarray(ti.f32, 2),
        faces: ti.types.ndarray(ti.i32, 1),
    ):
        for i, j in pixels:
            pixels[i, j] = (
                config[0]
                if i == 0 and j == 0
                else config[1] if i == 1 and j == 3 else 0.0
            )
        update(config, uva, uvb, faces)

    @ti.kernel
    def paint_4(
        config: ti.types.ndarray(ti.f32, 1),
        pixels: ti.types.ndarray(ti.f32, 3),
        uva: ti.types.ndarray(ti.f32, 2),
        uvb: ti.types.ndarray(ti.f32, 2),
        faces: ti.types.ndarray(ti.i32, 1),
    ):
        for i, j, k in pixels:
            pixels[i, j, k] = (
                (
                    config[0]
                    if i == 0 and j == 0
                    else config[1] if i == 1 and j == 3 else 0.0
                )
                if k == 3
                else 1.0
            )
        update(config, uva, uvb, faces)

    @ti.kernel
    def energy(
        hits: ti.types.ndarray(ti.f32, 2, needs_grad=True),
        loss: ti.types.ndarray(ti.f32, 1, needs_grad=True),
    ):
        for i in range(hits.shape[0]):
            if hits[i, 0] > 0:
                loss[0] += 0.5 * hits[i, 0] * hits[i, 0]

    start = time.perf_counter()
    module = ti.aot.Module()
    for kernel in (paint_1, paint_4, energy):
        module.add_kernel(kernel)
    module.add_kernel(energy.grad, name="energy_grad")
    for channels in (1, 4):
        # Keep the scalar case on v1; RGBA exercises the legacy alpha/face SBT
        # compatibility path with two different nonzero instance offsets.
        operations, resources = plan_parts(channels, sbt_offsets=[2, 4] if channels == 4 else None)
        module.add_execution_plan(
            f"alpha_{channels}",
            ti.aot.ExecutionPlan(
                operations,
                resources=resources,
                gradients=dict(hits="dhits", loss="dloss"),
            ),
        )
    build_ms = (time.perf_counter() - start) * 1000
    artifact = tmp_path / "alpha.tcm"
    module.archive(str(artifact))
    (tmp_path / "producer.json").write_text(
        json.dumps(dict(build_ms=build_ms, bytes=artifact.stat().st_size))
    )
    ti.reset()
    for mode in ("standalone", "borrowed"):
        command = [sys.executable] + (["-S"] if mode == "standalone" else [])
        result = subprocess.run(
            [
                *command,
                str(Path(__file__).with_name("aot_optix_alpha_probe.py")),
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
