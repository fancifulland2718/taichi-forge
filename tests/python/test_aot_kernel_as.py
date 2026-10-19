"""AOT kernel parameters bind the same managed TLAS as native operations."""

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
from tests.python.aot_kernel_as_probe import POSES, VERTICES, plan_parts


@pytest.mark.parametrize("recipe", ["explicit", "graph"])
@pytest.mark.parametrize("warm", [False, True])
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_aot_kernel_acceleration_structure(tmp_path, recipe, warm):
    if not ti.hardware.ray.is_available():
        pytest.skip("Vulkan ray query unavailable")

    @ti.func
    def accept(hit: ti.template(), gates: ti.template()):
        return gates[hit.instance_id] != 0

    @ti.kernel
    def query(
        scene: ti.types.acceleration_structure(),
        rays: ti.types.ndarray(ti.f32, 2),
        gates: ti.types.ndarray(ti.i32, 1),
        hits: ti.types.ndarray(ti.f32, 2),
        ids: ti.types.ndarray(ti.i32, 2),
        occluded: ti.types.ndarray(ti.i32, 1),
    ):
        for i in range(rays.shape[0]):
            origin = ti.Vector([rays[i, 0], rays[i, 1], rays[i, 2]])
            direction = ti.Vector([rays[i, 4], rays[i, 5], rays[i, 6]])
            h = scene.trace_closest_filtered(
                origin, direction, accept, args=(gates,), t_min=rays[i, 3], t_max=rays[i, 7], respect_opacity=True
            )
            any_hit = scene.trace_any_filtered(
                origin, direction, accept, args=(gates,), t_min=rays[i, 3], t_max=rays[i, 7], respect_opacity=True
            )
            hits[i, 0] = h.t
            hits[i, 1] = h.barycentric_u
            hits[i, 2] = h.barycentric_v
            hits[i, 3] = 0
            ids[i, 0] = ti.cast(h.primitive_index, ti.i32)
            ids[i, 1] = ti.cast(h.instance_id, ti.i32)
            ids[i, 2] = ti.cast(h.instance_custom_index, ti.i32)
            ids[i, 3] = h.hit
            occluded[i] = any_hit.hit

    @ti.kernel
    def energy(hits: ti.types.ndarray(ti.f32, 2, needs_grad=True), loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in range(hits.shape[0]):
            if hits[i, 0] > 0:
                loss[0] += 0.5 * hits[i, 0] * hits[i, 0]

    if warm:
        # The same frontend query expressions must lower again for the AOT
        # target after JIT has consumed and destroyed its temporary lowered IR.
        vertices = ti.ndarray(ti.f32, (3, 3))
        indices = ti.ndarray(ti.i32, (1, 3))
        vertices.from_numpy(np.array(VERTICES, np.float32).reshape(3, 3))
        indices.from_numpy(np.array([[0, 1, 2]], np.int32))
        rays = ti.ndarray(ti.f32, (1, 8))
        rays.from_numpy(np.array([[0, 0, 0, 0.01, 0, 0, 1, 3]], np.float32))
        gates = ti.ndarray(ti.i32, 1)
        gates.fill(1)
        hits, ids = ti.ndarray(ti.f32, (1, 4)), ti.ndarray(ti.i32, (1, 4))
        occluded = ti.ndarray(ti.i32, 1)
        with ti.hardware.ray.TriangleBLAS(vertices, indices) as blas:
            with ti.hardware.ray.InstanceTLAS([ti.hardware.ray.RayInstance(blas)]) as scene:
                query(scene, rays, gates, hits, ids, occluded)
                assert ids.to_numpy()[0, 3] == 1
                assert occluded.to_numpy()[0] == 1

    start = time.perf_counter()
    module = ti.aot.Module(caps=["spirv_has_ray_query"])
    module.add_kernel(query)
    module.add_kernel(energy)
    module.add_kernel(energy.grad, name="backward")
    if recipe == "graph":
        _export_graph(module, query, energy)
    else:
        for private in (True, False):
            module.add_execution_plan("private" if private else "public", ti.aot.ExecutionPlan(**plan_parts(private)))
    costs = dict(compile_ms=(time.perf_counter() - start) * 1000)
    artifact = tmp_path / "kernel-as.tcm"
    module.archive(str(artifact))
    costs["artifact_bytes"] = artifact.stat().st_size
    (tmp_path / "producer.json").write_text(json.dumps(costs, indent=2))
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_kernel_as_probe.py")),
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


def _export_graph(module, query, energy):
    import numpy as np

    @ti.kernel
    def seed(dhits: ti.types.ndarray(ti.f32, 2), loss: ti.types.ndarray(ti.f32, 1), dloss: ti.types.ndarray(ti.f32, 1)):
        for i, j in dhits:
            dhits[i, j] = 0
        loss[0], dloss[0] = 0.0, 1.0

    def array(name, dtype=ti.f32, ndim=2):
        return ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, dtype, ndim=ndim)

    vertices, indices = ti.ndarray(ti.f32, (3, 3)), ti.ndarray(ti.i32, (1, 3))
    vertices.from_numpy(np.array(VERTICES, np.float32).reshape(3, 3))
    indices.from_numpy(np.array([[0, 1, 2]], np.int32))
    with ti.hardware.ray.TriangleBLAS(vertices, indices, opaque=False) as blas:
        with ti.hardware.ray.InstanceTLAS([
            ti.hardware.ray.RayInstance(blas, transform=tuple(POSES[:12]), custom_index=17),
            ti.hardware.ray.RayInstance(blas, transform=tuple(POSES[12:]), custom_index=21),
        ]) as scene:
            builder = ti.graph.GraphBuilder()
            symbolic = ti.graph.Arg(ti.graph.ArgKind.ACCELERATION_STRUCTURE, "scene")
            alias = ti.graph.Arg(ti.graph.ArgKind.ACCELERATION_STRUCTURE, "alias")
            builder._bind_internal_runtime_arg(alias, scene)
            builder.append_native(blas.record_refit(), admission="explicit")
            builder.append_native(scene.record_refit_transforms(transforms="poses"), admission="explicit")
            arguments = [array("rays"), array("gates", ti.i32, 1), array("hits"), array("ids", ti.i32), array("occluded", ti.i32, 1)]
            for acceleration in (symbolic, alias):
                builder.dispatch(query, acceleration, *arguments)
            dloss = builder.private_ndarray("dloss", ti.f32, 1)
            builder.dispatch(seed, array("dhits"), array("loss", ndim=1), dloss)
            builder.dispatch(energy, array("hits"), array("loss", ndim=1))
            builder.dispatch(energy.grad, array("hits"), array("loss", ndim=1))
            graph = builder.compile()
            mappings = {blas: dict(vertices="vertices", indices="indices"), scene: dict(transforms="poses")}
            sources = dict(scene=scene)
            count = len(module._exported_kernels)
            with pytest.raises(ti.TaichiRuntimeError, match="source_bindings"):
                module.add_execution_plan("missing", graph, resource_bindings=mappings)
            with pytest.raises(ValueError, match="unused"):
                module.add_execution_plan("unused", graph, source_bindings=dict(sources, unused=scene), resource_bindings=mappings)
            with pytest.raises(ValueError, match="conflict"):
                module.add_execution_plan("conflict", graph, source_bindings=sources, resource_bindings={blas: "blas", scene: "hits"})
            assert len(module._exported_kernels) == count and not module._graph_export_failed
            for private in (True, False):
                plan = module.add_execution_plan(
                    "private" if private else "public", graph,
                    source_bindings=sources,
                    resource_bindings=mappings if private else {blas: "blas", scene: "scene"},
                    gradients=dict(hits="dhits", loss="dloss"),
                )
                document = json.loads(plan.to_json())
                resources = [r for r in document.get("resources", ()) if r["kind"] == "native"]
                assert len(resources) == (2 if private else 0)
                calls = [op for op in document["operations"] if op["kind"] == "kernel"]
                assert calls[0]["arguments"][0] == calls[1]["arguments"][0]
                assert "alias" not in plan.argument_names
                assert any(c["kind"] == "resource_alias" for c in document.get("constraints", ())) == (not private)
            graph.close()
    assert not module._data_payloads  # No producer geometry or opaque handles are snapshotted.
