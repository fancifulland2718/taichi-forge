"""Vulkan shared BLAS/TLAS Graph export preserves owner relationships."""

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
from tests.python.aot_vulkan_instances_probe import RAYS, TRANSFORMS, VERTICES


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_instance_recording_export(tmp_path):
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
    def clear(
        dhits: ti.types.ndarray(ti.f32, 2), loss: ti.types.ndarray(ti.f32, 1), dloss: ti.types.ndarray(ti.f32, 1)
    ):
        for i, j in dhits:
            dhits[i, j] = 0
        loss[0], dloss[0] = 0.0, 1.0

    @ti.kernel
    def energy(hits: ti.types.ndarray(ti.f32, 2, needs_grad=True), loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in range(hits.shape[0]):
            if hits[i, 0] > 0:
                loss[0] += 0.5 * hits[i, 0] * hits[i, 0]

    vertices, indices = ti.ndarray(ti.f32, (3, 3)), ti.ndarray(ti.i32, (1, 3))
    vertices.from_numpy(np.array(VERTICES, np.float32).reshape(3, 3))
    indices.from_numpy(np.array([[0, 1, 2]], np.int32))
    symbol = lambda name, ndim: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=ndim)
    module, costs = ti.aot.Module(), {}
    with ExitStack() as stack:
        blas = stack.enter_context(ti.hardware.ray.TriangleBLAS(vertices, indices, opaque=False))
        scene = stack.enter_context(
            ti.hardware.ray.InstanceTLAS(
                [
                    ti.hardware.ray.RayInstance(
                        blas, transform=tuple(TRANSFORMS[:12]), custom_index=17, sbt_record_offset=3
                    ),
                    ti.hardware.ray.RayInstance(
                        blas, transform=tuple(TRANSFORMS[12:]), custom_index=21, sbt_record_offset=0xFFFFFF
                    ),
                ]
            )
        )
        poses = ti.ndarray(ti.f32, (2, 12))
        moved = np.array(TRANSFORMS, np.float32).reshape(2, 12)
        moved[:, 3] += 30
        poses.from_numpy(moved)
        scene.refit_transforms(poses)
        shifted = np.array(VERTICES, np.float32).reshape(3, 3)
        shifted[:, 2] += 20
        vertices.from_numpy(shifted)
        blas.refit(vertices)
        ti.sync()
        vertices.fill(99)
        indices.fill(99)
        poses.fill(99)  # Explicit consumer seeds define the fresh AS, not these arrays.
        seeds = {blas: dict(vertices="seed", indices="indices"), scene: dict(transforms="poses")}
        for query_only in (False, True):
            if query_only:
                blas.close()  # Live TLAS retains geometry after the source handle closes.
            builder = ti.graph.GraphBuilder()
            if not query_only:
                out_vertices = builder.private_ndarray("vertices", ti.f32, (3, 3))
                out_transforms = builder.private_ndarray("transforms", ti.f32, (2, 12))
                builder.dispatch(
                    move, symbol("seed", 2), symbol("poses", 2), symbol("control", 1), out_vertices, out_transforms
                )
                builder.append_native(blas.record_refit(), admission="explicit")
                builder.append_native(scene.record_refit_transforms(), admission="explicit")
            builder.append_native(scene.record_typed(len(RAYS), hit_indices="ids"), admission="explicit")
            builder.append_native(scene.record(len(RAYS), hits="legacy"), admission="explicit")
            dloss = builder.private_ndarray("dloss", ti.f32, 1)
            builder.dispatch(clear, symbol("dhits", 2), symbol("loss", 1), dloss)
            builder.dispatch(energy, symbol("hits", 2), symbol("loss", 1))
            builder.dispatch(energy.grad, symbol("hits", 2), symbol("loss", 1))
            start = time.perf_counter()
            graph = builder.compile()
            suffix = "query" if query_only else "step"
            costs[suffix + "_graph_ms"] = (time.perf_counter() - start) * 1000
            if not query_only:
                with pytest.raises(ti.TaichiRuntimeError, match="resource_bindings"):
                    module.add_execution_plan("missing", graph, resource_bindings={blas: seeds[blas]})
                with pytest.raises(ValueError, match="exactly"):
                    module.add_execution_plan(
                        "ports", graph, resource_bindings={blas: seeds[blas], scene: dict(vertices="poses")}
                    )
                with pytest.raises(ValueError, match="rebuilt privately"):
                    module.add_execution_plan("split", graph, resource_bindings={blas: seeds[blas], scene: "scene"})
                assert not module._exported_kernels and not module._graph_export_failed
            start = time.perf_counter()
            private = module.add_execution_plan(
                "private_" + suffix, graph, resource_bindings=seeds, gradients=dict(hits="dhits", loss="dloss")
            )
            costs[suffix + "_private_export_ms"] = (time.perf_counter() - start) * 1000
            native = [r for r in json.loads(private.to_json())["resources"] if r["kind"] == "native"]
            assert len(native) == 2
            tlas = next(r for r in native if r["operation"] == "instance_tlas")
            assert tlas["bindings"]["blas_0"] == tlas["bindings"]["blas_1"]
            assert tlas["parameters"]["custom_indices"] == [17, 21]
            assert tlas["parameters"]["sbt_record_offsets"] == [3, 0xFFFFFF]
            assert next(r for r in native if r["operation"] == "triangle_blas")["parameters"]["opaque"] is False
            public_map = {scene: "scene"} if query_only else {blas: "blas", scene: "scene"}
            public = module.add_execution_plan(
                "public_" + suffix, graph, resource_bindings=public_map, gradients=dict(hits="dhits", loss="dloss")
            )
            document = json.loads(public.to_json())
            assert document["version"] == 17
            assert any(c["kind"] == "resource_alias" for c in document["constraints"])
            assert not [r for r in document.get("resources", ()) if r["kind"] == "native"]
            if not query_only:
                mixed = module.add_execution_plan(
                    "mixed",
                    graph,
                    resource_bindings={blas: "blas", scene: seeds[scene]},
                    gradients=dict(hits="dhits", loss="dloss"),
                )
                assert len([r for r in json.loads(mixed.to_json())["resources"] if r["kind"] == "native"]) == 1
            graph.close()
    assert len(module._exported_kernels) == 4 and not module._data_payloads
    artifact = tmp_path / "vulkan_instances_export.tcm"
    module.archive(str(artifact))
    costs["artifact_bytes"] = artifact.stat().st_size
    (tmp_path / "producer.json").write_text(json.dumps(costs, indent=2))
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_vulkan_instances_export_probe.py")),
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
