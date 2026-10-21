"""Vulkan Graph export uses explicit geometry seeds or a public native scene."""

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
from tests.python.aot_vulkan_ray_probe import VERTICES


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_ray_recording_export(tmp_path, monkeypatch):
    if not ti.hardware.ray.is_available():
        pytest.skip("Vulkan ray query unavailable")

    from taichi_forge.graph._native_definition import NativeResourceBinding

    lowered = []
    original_lower = NativeResourceBinding.lower

    def lower(binding, context, **kwargs):
        if hasattr(binding.owner, "_definition"):
            assert binding.definition is binding.owner._definition
            lowered.append(binding.owner)
        return original_lower(binding, context, **kwargs)

    monkeypatch.setattr(NativeResourceBinding, "lower", lower)

    @ti.kernel
    def move(
        seed: ti.types.ndarray(ti.f32, 2), shift: ti.types.ndarray(ti.f32, 1), vertices: ti.types.ndarray(ti.f32, 2)
    ):
        for i, j in seed:
            vertices[i, j] = seed[i, j] + (shift[0] if j == 2 else 0.0)

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
    module, costs = ti.aot.Module(), {}
    symbol = lambda name, ndim: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=ndim)
    with ti.hardware.ray.TriangleScene(vertices, indices) as scene:
        shifted = np.array(VERTICES, np.float32).reshape(3, 3)
        shifted[:, 2] += 20
        vertices.from_numpy(shifted)
        scene.refit(vertices)
        ti.sync()
        vertices.fill(99)
        indices.fill(99)  # Neither live AS nor original arrays is an export snapshot.
        for query_only in (False, True):
            builder = ti.graph.GraphBuilder()
            if not query_only:
                builder.dispatch(move, symbol("seed", 2), symbol("shift", 1), symbol("vertices", 2))
                builder.append_native(scene.record_refit(), admission="explicit")
            builder.append_native(scene.record_typed(4, hit_indices="ids"), admission="explicit")
            builder.append_native(scene.record(4, hits="legacy"), admission="explicit")
            builder.dispatch(clear, symbol("dhits", 2), symbol("loss", 1), symbol("dloss", 1))
            builder.dispatch(energy, symbol("hits", 2), symbol("loss", 1))
            builder.dispatch(energy.grad, symbol("hits", 2), symbol("loss", 1))
            start = time.perf_counter()
            graph = builder.compile()
            costs["query_graph_ms" if query_only else "step_graph_ms"] = (time.perf_counter() - start) * 1000
            if not query_only:
                with pytest.raises(ti.TaichiRuntimeError, match="resource_bindings"):
                    module.add_execution_plan("missing", graph)
                with pytest.raises(ValueError, match="exactly"):
                    module.add_execution_plan("ports", graph, resource_bindings={scene: dict(vertices="seed")})
                assert not module._exported_kernels and not module._graph_export_failed
                start = time.perf_counter()
                private = module.add_execution_plan(
                    "private",
                    graph,
                    gradients=dict(hits="dhits", loss="dloss"),
                    resource_bindings={scene: dict(vertices="seed", indices="indices")},
                )
                costs["private_export_ms"] = (time.perf_counter() - start) * 1000
                resources = json.loads(private.to_json())["resources"]
                assert len(resources) == 1 and resources[0]["operation"] == "triangle_scene"
                assert "indices" in private.argument_names
            start = time.perf_counter()
            public = module.add_execution_plan(
                "query" if query_only else "public",
                graph,
                gradients=dict(hits="dhits", loss="dloss"),
                resource_bindings={scene: "scene"},
            )
            costs["query_export_ms" if query_only else "public_export_ms"] = (time.perf_counter() - start) * 1000
            assert not json.loads(public.to_json()).get("resources")
            assert "scene" in public.argument_names and "indices" not in public.argument_names
            graph.close()
    assert len(module._exported_kernels) == 4 and not module._data_payloads
    assert scene in lowered
    artifact = tmp_path / "vulkan_export.tcm"
    module.archive(str(artifact))
    costs["artifact_bytes"] = artifact.stat().st_size
    (tmp_path / "producer.json").write_text(json.dumps(costs, indent=2))
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_vulkan_ray_export_probe.py")),
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
