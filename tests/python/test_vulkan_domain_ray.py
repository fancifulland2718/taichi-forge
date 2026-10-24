"""Private AS initialization, refit and typed queries on independent queues."""

import os
from pathlib import Path
import subprocess
import sys

import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.aot_vulkan_ray_probe import native


def build_fixture(folder):
    @ti.kernel
    def move(seed: ti.types.ndarray(ti.f32, 2), vertices: ti.types.ndarray(ti.f32, 2),
             transforms: ti.types.ndarray(ti.f32, 3), phase: ti.u64):
        for i, j in seed:
            vertices[i, j] = seed[i, j] + (ti.cast(phase & 7, ti.f32) if j == 2 else 0.0)
        for i, r, c in transforms:
            value = ti.cast(r == c, ti.f32)
            if c == 3:
                value = 4.0 * i if r == 0 else (2.0 * i if r == 2 else 0.0)
            transforms[i, r, c] = value

    @ti.kernel
    def energy(hits: ti.types.ndarray(ti.f32, 2, needs_grad=True),
               loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in range(hits.shape[0]):
            if hits[i, 0] > 0:
                loss[0] += 0.5 * hits[i, 0] * hits[i, 0]

    module = ti.aot.Module(caps=["spirv_has_int64", "spirv_has_ray_query"])
    module.add_kernel(move)
    module.add_kernel(energy)
    module.add_kernel(energy.grad, name="backward")
    for instanced in (False, True):
        name = "instances" if instanced else "scene"
        resources = [
            dict(kind="ndarray", name="vertices", dtype="f32", shape=[6, 3]),
            dict(kind="ndarray", name="transforms", dtype="f32", shape=[2, 3, 4]),
            dict(kind="ndarray", name="dloss", dtype="f32", shape=[1]),
        ]
        geometry = "blas" if instanced else "scene"
        resources.append(dict(native("triangle_blas" if instanced else "triangle_scene",
                                     dict(vertices="seed", indices="indices"),
                                     vertex_count=6, triangle_count=2), name=geometry))
        if instanced:
            resources.append(dict(native("instance_tlas", dict(blas_0="blas", blas_1="blas",
                                                               transforms="initial_transforms"),
                                         instance_count=2, custom_indices=[17, 21]), name="scene"))
        operations = [
            dict(kind="kernel", entry="move", arguments=["seed", "vertices", "transforms", "phase"]),
        ]
        if instanced:
            # Exercise rebuild and metadata update as well as the usual refits.
            operations += [
                native("blas_build", dict(scene="blas", vertices="vertices", indices="indices")),
                native("instance_update", dict(scene="scene"),
                       transforms=[[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                                   [1, 0, 0, 4, 0, 1, 0, 0, 0, 0, 1, 2]],
                       custom_indices=[19, 23], visibility_masks=[255, 255], update=True),
            ]
        operations.append(native("refit", dict(scene=geometry, vertices="vertices")))
        if instanced:
            operations.append(native("instance_refit", dict(scene="scene", transforms="transforms")))
        operations += [
            native("query", dict(scene="scene", rays="rays", hits="hits", hit_indices="ids"), ray_count=6),
            dict(kind="fill32", destination="dhits", value=0),
            dict(kind="fill32", destination="loss", value=0),
            dict(kind="fill32", destination="dloss", value=1065353216),
            dict(kind="kernel", entry="energy", arguments=["hits", "loss"]),
            dict(kind="kernel", entry="backward", arguments=["hits", "loss"]),
        ]
        plan = ti.aot.ExecutionPlan(operations, resources=resources, gradients=dict(hits="dhits", loss="dloss"))
        module.add_execution_plan(name, plan)
        folder.mkdir(exist_ok=True)
        (folder / f"{name}.json").write_text(plan.to_json(), encoding="utf8")
    module.save(folder)


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_private_ray_domain_initialization_refit_and_adjoint(tmp_path):
    if not ti.hardware.ray.is_available():
        pytest.skip("Vulkan ray query unavailable")
    folder = tmp_path / "module"
    build_fixture(folder)
    ti.reset()
    env = os.environ.copy()
    for mode in ("standalone", "borrowed"):
        command = [sys.executable] + (["-S"] if mode == "standalone" else [])
        result = subprocess.run(
            [*command, str(Path(__file__).with_name("aot_domain_ray_probe.py")), str(folder), mode],
            env=env, capture_output=True, text=True, timeout=90)
        (tmp_path / f"consumer-{mode}.log").write_text(result.stdout + result.stderr, encoding="utf8")
        assert result.returncode == 0, result.stdout + result.stderr
        assert "VUID-" not in result.stdout + result.stderr
        assert "SYNC-HAZARD" not in result.stdout + result.stderr
        assert "private_ray_domain_ok" in result.stdout
    executable = env.get("TI_TEST_DOMAIN_EXECUTABLE")
    if executable:
        env["TAICHI_VULKAN_RAY_DOMAIN_AOT_DIR"] = str(folder)
        result = subprocess.run(
            [executable, "--gtest_filter=VulkanExecutionOwner.PrivateRayInitializationAndIndependentRefit"],
            env=env, capture_output=True, text=True, timeout=60)
        (tmp_path / "native.log").write_text(result.stdout + result.stderr, encoding="utf8")
        assert result.returncode == 0, result.stdout + result.stderr
        assert "[  PASSED  ] 1 test." in result.stdout
        assert "[  SKIPPED ]" not in result.stdout
        assert "VUID-" not in result.stdout + result.stderr
        assert "SYNC-HAZARD" not in result.stdout + result.stderr
