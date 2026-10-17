"""BLAS rebuild and captured TLAS metadata/build export use the shared cores."""

from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.aot_vulkan_build_probe import RAYS, TRANSFORMS, VERTICES, oracle, values


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_build_recordings_export(tmp_path):
    if not ti.hardware.ray.is_available():
        pytest.skip("Vulkan ray query unavailable")
    vertices, indices = ti.ndarray(ti.f32, (3, 3)), ti.ndarray(ti.i32, (1, 3))
    vertices.from_numpy(np.array(VERTICES, np.float32).reshape(3, 3))
    indices.from_numpy(np.array([[0, 1, 2]], np.int32))
    costs = {}
    with ExitStack() as stack:
        blas = stack.enter_context(ti.hardware.ray.TriangleBLAS(vertices, indices))

        def instances(custom, masked=False, dz=0):
            result = []
            for i in range(2):
                transform = TRANSFORMS[i * 12 : (i + 1) * 12]
                transform[11] += dz
                result.append(
                    ti.hardware.ray.RayInstance(
                        blas, transform=tuple(transform), custom_index=custom[i], mask=0 if masked and i == 1 else 255
                    )
                )
            return result

        scene = stack.enter_context(ti.hardware.ray.InstanceTLAS(instances((17, 21))))
        builder = ti.graph.GraphBuilder()
        builder.append_native(blas.record_build(), admission="explicit")
        builder.append_native(scene.record_refit_transforms(transforms="before"), admission="explicit")
        builder.append_native(scene.record_refit(instances((31, 37), masked=True, dz=0.5)), admission="explicit")
        builder.append_native(
            scene.record_typed(len(RAYS), hits="hits_host", hit_indices="ids_host"), admission="explicit"
        )
        builder.append_native(scene.record_build(instances((41, 43))), admission="explicit")
        builder.append_native(
            scene.record_typed(len(RAYS), hits="hits_build", hit_indices="ids_build"), admission="explicit"
        )
        builder.append_native(scene.record_refit_transforms(transforms="after"), admission="explicit")
        builder.append_native(
            scene.record_typed(len(RAYS), hits="hits_device", hit_indices="ids_device"), admission="explicit"
        )
        start = time.perf_counter()
        graph = builder.compile()
        costs["graph_ms"] = (time.perf_counter() - start) * 1000
        frame = dict(
            vertices=vertices,
            indices=indices,
            before=ti.ndarray(ti.f32, (2, 12)),
            after=ti.ndarray(ti.f32, (2, 12)),
            rays=ti.ndarray(ti.f32, (len(RAYS), 8)),
        )
        frame.update(
            {
                f"{port}_{stage}": ti.ndarray(ti.i32 if port == "ids" else ti.f32, (len(RAYS), 4))
                for stage in ("host", "build", "device")
                for port in ("hits", "ids")
            }
        )
        frame["rays"].from_numpy(np.array(RAYS, np.float32))
        for dz, flipped, dx in ((0, False, 0), (0.25, True, 0), (0, False, 8)):
            for key, value in values(dz, flipped, dx).items():
                frame[key].from_numpy(
                    np.array(value, np.int32 if key == "indices" else np.float32).reshape(frame[key].shape)
                )
            graph.run(frame)
            oracle(lambda key: frame[key].to_numpy().reshape(-1).tolist(), dz, flipped, dx)
        # Changing the source owner's current metadata must not replace captured records.
        scene.refit(instances((99, 101)))
        module = ti.aot.Module()
        for private in (True, False):
            start = time.perf_counter()
            plan = module.add_execution_plan(
                "private" if private else "public",
                graph,
                resource_bindings=(
                    {blas: dict(vertices="seed", indices="seed_indices"), scene: dict(transforms="seed_transforms")}
                    if private
                    else {blas: "blas", scene: "scene"}
                ),
            )
            costs[("private" if private else "public") + "_export_ms"] = (time.perf_counter() - start) * 1000
            operations = json.loads(plan.to_json())["operations"]
            updates = [op for op in operations if op.get("operation") == "instance_update"]
            assert [op["parameters"]["update"] for op in updates] == [True, False]
            assert [op["parameters"]["custom_indices"] for op in updates] == [[31, 37], [41, 43]]
        graph.close()
        assert not module._exported_kernels
        artifact = tmp_path / "vulkan_builds.tcm"
        module.archive(str(artifact))
        costs["artifact_bytes"] = artifact.stat().st_size
    (tmp_path / "producer.json").write_text(json.dumps(costs, indent=2))
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_vulkan_build_probe.py")),
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
