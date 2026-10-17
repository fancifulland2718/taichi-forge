"""Mixed mesh/instance/particle/line recordings export without a consumer UI frontend."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge._lib import core
from tests import test_utils


@pytest.mark.skipif(not core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_aot_raster_mesh_particle_recordings(tmp_path, monkeypatch):
    shape = (32, 24)
    total = shape[0] * shape[1] * 5
    vertices = ti.Vector.field(3, ti.f32, 3)
    indices = ti.field(ti.i32, 3)
    vertices.from_numpy(np.array([[-0.7, -0.5, 0], [-0.1, -0.5, 0], [-0.4, 0.2, 0]], np.float32))
    indices.from_numpy(np.array([0, 1, 2], np.int32))
    transforms = ti.Matrix.ndarray(4, 4, ti.f32, 2)
    centers = ti.Vector.ndarray(3, ti.f32, 2)
    camera = ti.ui.Camera()
    camera.position(0, 0, 2)
    camera.lookat(0, 0, 0)
    camera.fov(60)

    @ti.kernel
    def sample(color: ti.types.texture(2), depth: ti.types.texture(2), output: ti.types.ndarray(ti.f32, 1)):
        for i in range(shape[0] * shape[1]):
            point = ti.Vector([i % shape[0], i // shape[0]])
            value = color.fetch(point, 0)
            for c in ti.static(range(4)):
                output[i * 5 + c] = value[c]
            output[i * 5 + 4] = depth.fetch(point, 0).x

    @ti.kernel
    def seed(dout: ti.types.ndarray(ti.f32, 1), loss: ti.types.ndarray(ti.f32, 1), dloss: ti.types.ndarray(ti.f32, 1)):
        for i in dout:
            dout[i] = 0
        loss[0], dloss[0] = 0, 1

    @ti.kernel
    def energy(
        output: ti.types.ndarray(ti.f32, 1, needs_grad=True), loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)
    ):
        for i in output:
            loss[0] += 0.5 * output[i] ** 2

    with ti.hardware.raster.RasterPass(shape) as raster:
        raster.set_camera(camera).ambient_light((0.4, 0.4, 0.4)).point_light((1, 2, 3), (0.5, 0.4, 0.3))
        raster.mesh(vertices, color=(1, 0, 0), two_sided=True)
        raster.mesh_instance(
            vertices,
            indices=indices,
            transforms=transforms,
            instance_count=5,
            instance_offset=1,
            color=(0, 1, 0),
            two_sided=True,
        )
        raster.particles(centers, radius=0.15, color=(0, 0, 1), index_count=1, index_offset=1)
        raster.lines(vertices, width=1, color=(1, 1, 0), vertex_count=3)
        start = time.perf_counter()
        recording = raster.record()
        record_ms = (time.perf_counter() - start) * 1000
        result = ti.ndarray(ti.f32, total)
        references = []
        for offset in (-0.2, 0.2, 5):
            matrices = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
            matrices[1, 0, 3] = 0.8 + offset
            transforms.from_numpy(matrices)
            centers.from_numpy(np.array([[9, 9, 0], [offset, 0.45, 0]], np.float32))
            recording.execute()
            sample(raster.color_texture, raster.depth_texture, result)
            references.append(result.to_numpy().tolist())
        assert references[0] != references[1] != references[2]

        def graph_for(selected, repeats=1):
            builder = ti.graph.GraphBuilder()
            for _ in range(repeats):
                builder.append_native(selected, admission="explicit")
            arg = lambda name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1)
            builder.dispatch(
                sample,
                ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "color_image", ndim=2),
                ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "depth_image", ndim=2),
                arg("output"),
            )
            builder.dispatch(seed, arg("dout"), arg("loss"), arg("dloss"))
            builder.dispatch(energy, arg("output"), arg("loss"))
            builder.dispatch(energy.grad, arg("output"), arg("loss"))
            return builder.compile()

        module = ti.aot.Module()
        sources = dict(color_image=raster.color_texture, depth_image=raster.depth_texture)
        options = dict(
            source_bindings=dict(sources, centers=centers, transforms=transforms),
            gradients=dict(output="dout", loss="dloss"),
        )
        graph = graph_for(recording, 2)
        with pytest.raises(ti.TaichiRuntimeError, match="snapshot_state"):
            module.add_execution_plan("invalid", graph, **options)
        assert not module._exported_kernels and not module._graph_export_failed
        import taichi_forge.hardware._graphics as graphics

        with monkeypatch.context() as patch:
            patch.setattr(
                graphics.VulkanGraphicsPipeline, "__init__", lambda *a, **k: pytest.fail("export built a GPU pipeline")
            )
            patch.setattr(recording._prepared_draws[0], "run", lambda: pytest.fail("export executed producer geometry"))
            start = time.perf_counter()
            plan = module.add_execution_plan("private", graph, snapshot_state=True, **options)
            export_ms = (time.perf_counter() - start) * 1000
            kernels = len(module._exported_kernels)
            module.add_execution_plan(
                "public",
                graph,
                snapshot_state=True,
                **options,
                resource_bindings={raster.color_texture: "color", raster.depth_texture: "depth"},
            )
            assert len(module._exported_kernels) == kernels
        document = json.loads(plan.to_json())
        assert len([r for r in document["resources"] if r.get("operation") == "pipeline"]) == 4
        assert len(document["initialization"]) == 4
        assert len([op for op in document["operations"] if op.get("provider") == "vulkan_compute"]) == 2
        assert len([op for op in document["operations"] if op.get("operation") == "pass_direct"]) == 2
        graph.close()

        raster.clear().mesh(vertices, vertex_count=0)
        empty_recording = raster.record()
        empty = graph_for(empty_recording)
        empty_plan = module.add_execution_plan(
            "empty", empty, snapshot_state=True, source_bindings=sources, gradients=dict(output="dout", loss="dloss")
        )
        empty_document = json.loads(empty_plan.to_json())
        assert all(r.get("provider") == "image" for r in empty_document["resources"])
        assert not empty_document.get("initialization")
        assert len(module._exported_kernels) == kernels
        empty.close()
        # Mutating the producer camera after export cannot change any artifact.
        camera.position(4, 3, 2)
    artifact = tmp_path / "raster.tcm"
    module.archive(str(artifact))
    (tmp_path / "references.json").write_text(json.dumps(references))
    (tmp_path / "producer.json").write_text(
        json.dumps(dict(record_ms=record_ms, export_ms=export_ms, kernels=kernels, bytes=artifact.stat().st_size))
    )
    ti.reset()
    for mode in ("standalone", "borrowed"):
        process = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_raster_export_probe.py")),
                mode,
                str(artifact),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert process.returncode == 0, process.stdout + process.stderr
        assert "Validation Error" not in process.stdout + process.stderr
