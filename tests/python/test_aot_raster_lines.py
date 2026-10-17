"""Line expansion exports through native compute and the shared graphics pass."""

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
from tests.python.aot_raster_lines_probe import geometry


@pytest.mark.skipif(not core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_aot_raster_lines_ranges_updates_and_retirement(tmp_path, monkeypatch):
    vertices, colors = (ti.Vector.ndarray(3, ti.f32, 8) for _ in range(2))
    indices = ti.ndarray(ti.i32, 7)
    camera = ti.ui.Camera()
    camera.position(0, 0, 2)
    camera.lookat(0, 0, 0)
    camera.fov(60)
    output = ti.ndarray(ti.f32, 32 * 24 * 5)

    @ti.kernel
    def sample(color: ti.types.texture(2), depth: ti.types.texture(2), output: ti.types.ndarray(ti.f32, 1)):
        for i in range(32 * 24):
            pixel = ti.Vector([i % 32, i // 32])
            value = color.fetch(pixel, 0)
            for j in ti.static(range(4)):
                output[i * 5 + j] = value[j]
            output[i * 5 + 4] = depth.fetch(pixel, 0).x

    module = ti.aot.Module()
    references, costs = {}, {}
    with ti.hardware.raster.RasterPass((32, 24)) as raster:
        raster.set_camera(camera)
        sources = dict(color_image=raster.color_texture, depth_image=raster.depth_texture)
        for name in ("plain", "indexed", "empty"):
            options = dict(vertex_offset=2, vertex_count=5, per_vertex_color=colors)
            if name != "plain":
                options.update(indices=indices, index_offset=2, index_count=1 if name == "empty" else 5)
            raster.clear().lines(vertices, width=3, **options)
            start = time.perf_counter()
            recording = raster.record()
            record_ms = (time.perf_counter() - start) * 1000
            references[name] = []
            for variant in range(3):
                points, color_values, index_values = geometry(variant)
                vertices.from_numpy(np.array(points, np.float32).reshape(8, 3))
                colors.from_numpy(np.array(color_values, np.float32).reshape(8, 3))
                indices.from_numpy(np.array(index_values, np.int32))
                recording.execute()
                sample(raster.color_texture, raster.depth_texture, output)
                references[name].append(output.to_numpy().tolist())
            if name != "empty":
                assert references[name][0] != references[name][1] != references[name][2]
            assert references[name][2] == [0, 0, 0, 1, 0] * (32 * 24)
            builder = ti.graph.GraphBuilder()
            builder.append_native(recording, admission="explicit")
            builder.append_native(recording, admission="explicit")
            builder.dispatch(
                sample,
                ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "color_image", ndim=2),
                ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "depth_image", ndim=2),
                ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1),
            )
            graph = builder.compile()
            inputs = dict(sources)
            if name != "empty":
                inputs.update(vertices=vertices, colors=colors)
                if name == "indexed":
                    inputs["indices"] = indices
            with monkeypatch.context() as patch:
                patch.setattr(recording._prepared_draws[0], "run", lambda: pytest.fail("export executed geometry"))
                start = time.perf_counter()
                plan = module.add_execution_plan(name, graph, snapshot_state=True, source_bindings=inputs)
                export_ms = (time.perf_counter() - start) * 1000
            document = json.loads(plan.to_json())
            if name == "empty":
                assert all(r.get("provider") == "image" for r in document["resources"])
                assert not document.get("initialization")
            else:
                assert len(document["initialization"]) == 1
                assert len([r for r in document["resources"] if r.get("operation") == "pipeline"]) == 2
                assert len([op for op in document["operations"] if op.get("provider") == "vulkan_compute"]) == 2
                # Two recordings reuse one pair of expansion arrays; no initial-data payload.
                translated = [
                    r
                    for r in document["resources"]
                    if r.get("shape") in ([8], [12])
                    and r.get("dtype") in ("f32", "u32")
                    and r.get("element_shape", []) in ([8], [])
                    and r.get("usage") in (["storage", "index"], ["storage", "vertex"])
                ]
                assert len(translated) == 2
                assert all("initial_data" not in r for r in translated)
            costs[name] = dict(record_ms=record_ms, export_ms=export_ms)
            graph.close()
        assert references["plain"][0] == references["indexed"][0]
        camera.position(4, 3, 2)
    artifact = tmp_path / "lines.tcm"
    module.archive(str(artifact))
    costs.update(bytes=artifact.stat().st_size, kernels=len(module._exported_kernels))
    (tmp_path / "producer.json").write_text(json.dumps(costs, indent=2))
    (tmp_path / "references.json").write_text(json.dumps(references))
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_raster_lines_probe.py")),
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
