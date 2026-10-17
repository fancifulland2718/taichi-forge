"""OptiX recording export uses explicit fresh-scene inputs, never stale AS data."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.aot._graph_resource_bindings import ResourceInputBindings
from tests import test_utils
from tests.python.aot_optix_probe import VERTICES


def test_resource_input_bindings_are_explicit_and_transactional():
    owner, unused = object(), object()
    ports = {"vertices": "seed", "indices": "topology"}
    inputs = ResourceInputBindings({owner: ports}, ())
    ports["vertices"] = "changed"
    with pytest.raises(ti.TaichiRuntimeError, match="explicit resource_bindings"):
        inputs.require(unused, ("vertices", "indices"))
    with pytest.raises(ValueError, match="exactly"):
        inputs.require(owner, ("vertices",))
    with pytest.raises(ValueError, match="unused"):
        inputs.validate_used()
    assert inputs.require(owner, ("vertices", "indices")) == dict(vertices="seed", indices="topology")
    inputs.validate_used()
    assert inputs.used_names == {"seed", "topology"}
    with pytest.raises(ValueError, match="private"):
        ResourceInputBindings({owner: {"vertices": "scratch"}}, ("scratch",))
    with pytest.raises(ValueError, match="NUL"):
        ResourceInputBindings({owner: {"vertices": "bad\0name"}}, ())


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_optix_recording_export_and_consumer_scene_reconstruction(tmp_path):
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
    def clear(
        dhits: ti.types.ndarray(ti.f32, 2), loss: ti.types.ndarray(ti.f32, 1), dloss: ti.types.ndarray(ti.f32, 1)
    ):
        for i, j in dhits:
            dhits[i, j] = 0
        loss[0] = 0
        dloss[0] = 1

    @ti.kernel
    def energy(hits: ti.types.ndarray(ti.f32, 2, needs_grad=True), loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in range(hits.shape[0]):
            if hits[i, 0] > 0:
                loss[0] += 0.5 * hits[i, 0] * hits[i, 0]

    vertices, indices = ti.ndarray(ti.f32, (6, 3)), ti.ndarray(ti.i32, (2, 3))
    vertices.from_numpy(np.array(VERTICES, np.float32).reshape(6, 3))
    indices.from_numpy(np.arange(6, dtype=np.int32).reshape(2, 3))
    with ti.hardware.ray.load_optix_provider(provider_path=adapter) as provider:
        with provider.triangle_scene(vertices, indices) as scene:
            # The latest AS and original arrays no longer describe the initial
            # geometry. Export must use only the explicit consumer seed policy.
            moved = np.array(VERTICES, np.float32).reshape(6, 3)
            moved[:, 2] += 20
            vertices.from_numpy(moved)
            scene.record_refit().execute(dict(vertices=vertices))
            ti.sync()
            vertices.fill(99)
            indices.fill(99)
            module = ti.aot.Module()
            timings = {}
            symbol = lambda name, rank: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=rank)
            for policy in ("two_sided", "front_only", "table", "query_only"):
                rules = (ti.hardware.ray.OptixFaceRuleTable("faces"),) if policy == "table" else ("front_only",)
                if policy == "two_sided":
                    rules = None
                start = time.perf_counter()
                builder = ti.graph.GraphBuilder()
                if policy != "query_only":
                    builder.dispatch(move, symbol("seed", 2), symbol("shift", 1), symbol("vertices", 2))
                    builder.append_native(scene.record_refit(), admission="explicit")
                builder.append_native(scene.record_typed(4, hit_indices="ids", face_rules=rules), admission="explicit")
                builder.append_native(scene.record_occlusion(4, face_rules=rules), admission="explicit")
                builder.dispatch(clear, symbol("dhits", 2), symbol("loss", 1), symbol("dloss", 1))
                builder.dispatch(energy, symbol("hits", 2), symbol("loss", 1))
                builder.dispatch(energy.grad, symbol("hits", 2), symbol("loss", 1))
                graph = builder.compile()
                timings[policy + "_graph_ms"] = (time.perf_counter() - start) * 1000
                if policy == "two_sided":
                    with pytest.raises(ti.TaichiRuntimeError, match="resource_bindings"):
                        module.add_execution_plan("missing", graph)
                    with pytest.raises(ValueError, match="exactly"):
                        module.add_execution_plan("ports", graph, resource_bindings={scene: {"vertices": "seed"}})
                    with pytest.raises(ValueError, match="unused"):
                        module.add_execution_plan(
                            "unused",
                            graph,
                            resource_bindings={scene: {"vertices": "seed", "indices": "indices"}, object(): {"x": "y"}},
                        )
                    assert not module._exported_kernels and not module._graph_export_failed
                start = time.perf_counter()
                plan = module.add_execution_plan(
                    policy,
                    graph,
                    gradients=dict(hits="dhits", loss="dloss"),
                    resource_bindings={scene: {"vertices": "seed", "indices": "indices"}},
                )
                timings[policy + "_export_ms"] = (time.perf_counter() - start) * 1000
                resources = json.loads(plan.to_json())["resources"]
                assert len(resources) == 2
                assert {r["operation"] for r in resources} == {"context", "triangle_scene"}
                assert "indices" in plan.argument_names
                graph.close()
            legacy = ti.graph.GraphBuilder()
            legacy.append_native(scene.record(4), admission="explicit")
            graph = legacy.compile()
            with pytest.raises(ti.TaichiRuntimeError, match="typed or compact"):
                module.add_execution_plan("legacy", graph)
            graph.close()
            alpha = ti.graph.GraphBuilder()
            alpha.append_native(
                scene.record_typed(4, alpha_masks=(ti.hardware.ray.OptixAlphaMask("uvs", "texture"),)),
                admission="explicit",
            )
            graph = alpha.compile()
            with pytest.raises(ti.TaichiRuntimeError, match="source_bindings"):
                module.add_execution_plan(
                    "alpha", graph, resource_bindings={scene: {"vertices": "seed", "indices": "indices"}}
                )
            graph.close()
            assert len(module._exported_kernels) == 4 and not module._data_payloads
            artifact = tmp_path / "export.tcm"
            start = time.perf_counter()
            module.archive(str(artifact))
            timings.update(archive_ms=(time.perf_counter() - start) * 1000, bytes=artifact.stat().st_size)
            with zipfile.ZipFile(artifact) as archive:
                assert all(adapter.encode() not in archive.read(name) for name in archive.namelist())
    (tmp_path / "producer.json").write_text(json.dumps(timings, indent=2))
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_optix_export_probe.py")),
                mode,
                str(artifact),
                adapter,
                str(tmp_path / f"{mode}.json"),
            ],
            capture_output=True,
            text=True,
            env=os.environ.copy(),
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr
