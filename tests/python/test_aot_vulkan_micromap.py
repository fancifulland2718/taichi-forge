"""Managed baked opacity micromaps use the shared AS kernel path."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.aot_vulkan_micromap_probe import POSES, asset, geometry, resources, plan_operations


@pytest.mark.parametrize("recipe", ["explicit", "graph"])
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_aot_vulkan_micromap_factories_and_gradients(tmp_path, recipe):
    if not ti.hardware.ray.is_opacity_micromap_available():
        pytest.skip("Vulkan opacity micromap unavailable")

    @ti.func
    def accept(hit: ti.template(), gate: ti.template()):
        return hit.instance_id == 1 or gate[0] != 0

    @ti.kernel
    def query(
        scene: ti.types.acceleration_structure(),
        coords: ti.types.ndarray(ti.f32, 2),
        result: ti.types.ndarray(ti.f32, 2),
        gate: ti.types.ndarray(ti.i32, 1),
        limit: ti.types.ndarray(ti.f32, 1),
    ):
        for i in range(coords.shape[0]):
            origin, direction = ti.Vector([coords[i, 0], coords[i, 1], 3.0]), ti.Vector([0.0, 0.0, -1.0])
            hit = scene.trace_closest_filtered(
                origin, direction, accept, args=(gate,), t_max=limit[0], respect_opacity=True
            )
            any_hit = scene.trace_any_filtered(
                origin, direction, accept, args=(gate,), t_max=limit[0], respect_opacity=True
            )
            result[i, 0], result[i, 1] = ti.cast(hit.instance_custom_index, ti.i32) if hit.hit else -1, hit.t
            result[i, 2], result[i, 3] = hit.hit, any_hit.hit

    @ti.kernel
    def energy(
        result: ti.types.ndarray(ti.f32, 2, needs_grad=True), loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)
    ):
        for i in range(result.shape[0]):
            if result[i, 1] > 0:
                loss[0] += 0.5 * result[i, 1] * result[i, 1]

    @ti.kernel
    def seed(
        result: ti.types.ndarray(ti.f32, 2), loss: ti.types.ndarray(ti.f32, 1), dloss: ti.types.ndarray(ti.f32, 1)
    ):
        for i, j in result:
            result[i, j] = 0
        loss[0], dloss[0] = 0.0, 1.0

    start = time.perf_counter()
    module = ti.aot.Module(caps=["spirv_has_ray_query"])
    module.add_kernel(query)
    module.add_kernel(energy)
    module.add_kernel(energy.grad, name="backward")
    gradient = dict(result="dresult", loss="dloss")
    for case in ("two", "indexed", "predefined"):
        if recipe == "graph":
            _export_graph(module, case, query, energy, seed)
            continue
        blobs = []
        for name, value in zip(("data", "descriptors", "mapping"), asset(case)[:3]):
            if value is not None:
                blobs.append(dict(kind="host_blob", name=name, data=module.add_data(value), byte_size=len(value)))
        module.add_execution_plan(
            case,
            ti.aot.ExecutionPlan(
                plan_operations(),
                resources=blobs
                + resources(case, [r["name"] for r in blobs])
                + [dict(kind="ndarray", name="dloss", dtype="f32", shape=[1])],
                gradients=gradient,
            ),
        )
    if recipe == "explicit":
        module.add_execution_plan(
            "public",
            ti.aot.ExecutionPlan(
                plan_operations(),
                resources=[dict(kind="ndarray", name="dloss", dtype="f32", shape=[1])],
                gradients=gradient,
            ),
        )
    costs = dict(compile_ms=(time.perf_counter() - start) * 1000)
    artifact = tmp_path / "micromap.tcm"
    module.archive(str(artifact))
    costs["artifact_bytes"] = artifact.stat().st_size
    (tmp_path / "producer.json").write_text(json.dumps(costs, indent=2))
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_vulkan_micromap_probe.py")),
                mode,
                str(artifact),
                str(tmp_path / f"{mode}.json"),
                recipe,
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Validation Error" not in result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr


def _export_graph(module, case, query, energy, seed):
    import numpy as np

    vertices, _, _ = geometry(case)
    data, descriptors, indices, _ = asset(case)
    baked = ti.hardware.ray.VulkanOpacityMicromap(data, descriptors, indices)
    v = ti.ndarray(ti.f32, (len(vertices) // 3, 3))
    ix = ti.ndarray(ti.i32, (len(vertices) // 9, 3))
    v.from_numpy(np.array(vertices, np.float32).reshape(-1, 3))
    ix.from_numpy(np.arange(len(vertices) // 3, dtype=np.int32).reshape(-1, 3))
    with ti.hardware.ray.TriangleBLAS(v, ix, opacity_micromap=baked) as front:
        with ti.hardware.ray.TriangleBLAS(v, ix) as back:
            with ti.hardware.ray.InstanceTLAS(
                [
                    ti.hardware.ray.RayInstance(front, transform=tuple(POSES[:12]), custom_index=11),
                    ti.hardware.ray.RayInstance(back, transform=tuple(POSES[12:]), custom_index=22),
                ]
            ) as scene:
                builder = ti.graph.GraphBuilder()

                def arg(name, dtype=ti.f32, ndim=2):
                    return ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, dtype, ndim=ndim)

                builder.append_native(front.record_refit(), admission="explicit")
                builder.append_native(scene.record_refit_transforms(transforms="poses"), admission="explicit")
                builder.dispatch(
                    query,
                    ti.graph.Arg(ti.graph.ArgKind.ACCELERATION_STRUCTURE, "scene"),
                    arg("coords"),
                    arg("result"),
                    arg("gate", ti.i32, 1),
                    arg("limit", ndim=1),
                )
                dloss = builder.private_ndarray("dloss", ti.f32, 1)
                builder.dispatch(seed, arg("dresult"), arg("loss", ndim=1), dloss)
                builder.dispatch(energy, arg("result"), arg("loss", ndim=1))
                builder.dispatch(energy.grad, arg("result"), arg("loss", ndim=1))
                graph = builder.compile()
                mappings = {
                    front: dict(vertices="vertices", indices="indices", opacity_micromap=baked),
                    back: dict(vertices="vertices", indices="indices"),
                    scene: dict(transforms="poses"),
                }
                options = dict(source_bindings=dict(scene=scene), gradients=dict(result="dresult", loss="dloss"))
                if case == "two":
                    before = len(module._exported_kernels)
                    wrong = ti.hardware.ray.VulkanOpacityMicromap(bytes([0]), descriptors, indices)
                    with pytest.raises(ti.TaichiRuntimeError, match="fingerprint"):
                        module.add_execution_plan(
                            "wrong",
                            graph,
                            **options,
                            resource_bindings=mappings | {front: dict(mappings[front], opacity_micromap=wrong)},
                        )
                    with pytest.raises(ValueError, match="symbolic"):
                        module.add_execution_plan(
                            "wrong_port",
                            graph,
                            **options,
                            resource_bindings=mappings | {front: dict(mappings[front], vertices=baked)},
                        )
                    assert len(module._exported_kernels) == before and not module._graph_export_failed
                    module.add_execution_plan(
                        "public", graph, **options, resource_bindings={front: "front", scene: "scene"}
                    )
                private = module.add_execution_plan(case, graph, **options, resource_bindings=mappings)
                description = json.loads(private.to_json())
                assert len([r for r in description["resources"] if r.get("operation") == "opacity_micromap"]) == 1
                shared = module.add_execution_plan(
                    case + "_shared_omm",
                    graph,
                    **options,
                    resource_bindings=mappings | {front: dict(mappings[front], opacity_micromap="omm")},
                )
                assert "omm" in shared.argument_names
                assert not any(
                    r.get("operation") == "opacity_micromap" for r in json.loads(shared.to_json())["resources"]
                )
                graph.close()
