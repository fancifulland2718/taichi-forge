"""Baked OMM resources in ordinary Graph exports and independent consumers."""

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
from tests.python.aot_vulkan_micromap_probe import asset, geometry, POSES


@test_utils.test(arch=[ti.cpu, ti.vulkan])
def test_optix_micromap_factory_is_cuda_only():
    with ti.aot.borrow_runtime() as runtime:
        assert not runtime.native_resource_capabilities("optix_opacity_micromap")
        assert not runtime.probe_native_resource("optix_triangle_gas", version=2)["available"]


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_optix_micromap_export_refit_queries_and_retirement(tmp_path):
    adapter = os.environ.get("TAICHI_FORGE_TEST_OPTIX_PROVIDER")
    if not adapter:
        pytest.skip("explicit common OptiX adapter required")

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

    module, costs = ti.aot.Module(), []
    ray = ti.hardware.ray
    for case in ("two", "indexed", "predefined"):
        vertices, coords, states = geometry(case)
        data, descriptors, mapping, _ = asset(case)
        baked = ray.OptixOpacityMicromap(data, descriptors, mapping)
        triangles, count = len(vertices) // 9, len(states)
        v, ix = ti.ndarray(ti.f32, (triangles * 3, 3)), ti.ndarray(ti.i32, (triangles, 3))
        v.from_numpy(np.array(vertices, np.float32).reshape(-1, 3))
        ix.from_numpy(np.arange(triangles * 3, dtype=np.int32).reshape(-1, 3))
        with ExitStack() as stack:
            provider = stack.enter_context(ray.OptixProvider(provider_path=adapter))
            front = stack.enter_context(provider.triangle_gas(v, ix, opacity_micromap=baked))
            back = stack.enter_context(provider.triangle_gas(v, ix))
            scene = stack.enter_context(
                provider.instance_scene(
                    (
                        ray.OptixRayInstance(front, transform=tuple(POSES[:12]), custom_index=11, sbt_record_offset=3),
                        ray.OptixRayInstance(back, transform=tuple(POSES[12:]), custom_index=22, opaque=True),
                    )
                )
            )
            texture = ti.Texture(ti.Format.r32f, (2, 2))
            pixels = ti.ndarray(ti.f32, (2, 2))
            pixels.fill(1)
            texture.from_ndarray(pixels)
            start = time.perf_counter()
            builder = ti.graph.GraphBuilder()
            builder.append_native(front.record_refit(), admission="explicit")
            builder.append_native(scene.record_refit_transforms(transforms="poses"), admission="explicit")
            masks = (ray.OptixAlphaMask("uvs", "alpha", channel=0), None)
            for prefix, policy in (("", masks), ("plain_", None)):
                builder.append_native(
                    scene.record_typed(count, hits=prefix + "hits", hit_indices=prefix + "ids", alpha_masks=policy),
                    admission="explicit",
                )
                builder.append_native(
                    scene.record_occlusion(count, occluded=prefix + "occluded", alpha_masks=policy),
                    admission="explicit",
                )
            builder.append_native(
                scene.record_typed(count, hits="first_hits", hit_indices="first_ids", alpha_masks=masks, any_hit=True),
                admission="explicit",
            )
            arg = lambda name, rank: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=rank)
            builder.dispatch(clear, arg("dhits", 2), arg("loss", 1), arg("dloss", 1))
            builder.dispatch(energy, arg("hits", 2), arg("loss", 1))
            builder.dispatch(energy.grad, arg("hits", 2), arg("loss", 1))
            graph = builder.compile()
            graph_ms = (time.perf_counter() - start) * 1000
            seeds = {
                front: dict(vertices="vertices", indices="indices", opacity_micromap=baked),
                back: dict(vertices="vertices", indices="indices"),
                scene: dict(transforms="poses"),
                texture: "alpha_image",
            }
            if case == "two":
                for replacement in (None, ray.OptixOpacityMicromap(b"\0", descriptors)):
                    bad = dict(seeds)
                    bad[front] = dict(vertices="vertices", indices="indices")
                    if replacement is not None:
                        bad[front]["opacity_micromap"] = replacement
                    with pytest.raises((ValueError, ti.TaichiRuntimeError), match="reconstruction ports|fingerprint"):
                        module.add_execution_plan(
                            "bad", graph, source_bindings=dict(alpha=texture), resource_bindings=bad
                        )
                    assert not module._exported_kernels and not module._graph_export_failed
            if case == "indexed":
                seeds[baked] = "public_omm"
            start = time.perf_counter()
            plan = module.add_execution_plan(
                case,
                graph,
                source_bindings=dict(alpha=texture),
                resource_bindings=seeds,
                gradients=dict(hits="dhits", loss="dloss"),
            )
            costs.append(dict(case=case, graph_ms=graph_ms, export_ms=(time.perf_counter() - start) * 1000))
            description = json.loads(plan.to_json())
            gas = [r for r in description["resources"] if r.get("operation") == "triangle_gas"]
            assert sorted(r["version"] for r in gas) == [1, 2]
            assert next(r for r in gas if r["version"] == 2)["parameters"]["expected_micromap"] == baked.fingerprint
            graph.close()
            texture._delete_runtime_texture()
    artifact = tmp_path / "optix-omm.tcm"
    module.archive(str(artifact))
    (tmp_path / "producer.json").write_text(json.dumps(dict(cases=costs, bytes=artifact.stat().st_size), indent=2))
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_optix_micromap_probe.py")),
                mode,
                str(artifact),
                adapter,
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr
