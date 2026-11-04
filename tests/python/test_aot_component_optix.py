"""Original OptiX recordings retain consumer scene and whole-array query extents."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils


@ti.data_oriented
class OptixComponent:
    def __init__(self, count, scene):
        self.count = count
        builder = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=builder)
        scope.output = scope.field(
            ti.f32, shape=scope.dim("count"), initialization="zero"
        )
        self.tree = builder.finalize()
        self.refit = scene.record_refit_transforms(
            transforms="transforms", scene="scene"
        )
        self.query = scene.record_typed(count, scene="scene", graph_fixed_query=False)
        self.occlusion = scene.record_occlusion(
            count, scene="scene", graph_fixed_query=False
        )

    @ti.kernel
    def generate(
        self,
        transforms: ti.types.ndarray(ti.f32, ndim=3),
        shift: ti.types.ndarray(ti.f32, ndim=1),
        rays: ti.types.ndarray(ti.f32, ndim=2),
    ):
        for i in range(transforms.shape[0]):
            for r, c in ti.static(ti.ndrange(3, 4)):
                transforms[i, r, c] = ti.cast(r == c, ti.f32)
            transforms[i, 0, 3] = 4.0 * i
            transforms[i, 2, 3] = shift[0] + 0.125 * i
        for i in range(rays.shape[0]):
            for j in ti.static(range(8)):
                rays[i, j] = 0
            rays[i, 0] = 4.0 * (i % transforms.shape[0])
            rays[i, 2] = 2.0
            rays[i, 3] = 0.001
            rays[i, 6] = -1.0
            rays[i, 7] = 5.0

    @ti.kernel
    def publish(
        self,
        hits: ti.types.ndarray(ti.f32, ndim=2),
        hit_indices: ti.types.ndarray(ti.i32, ndim=2),
        occluded: ti.types.ndarray(ti.i32, ndim=1),
    ):
        for i in self.output:
            self.output[i] = (
                hits[i, 0] if hit_indices[i, 0] >= 0 and occluded[i] == 1 else -1.0
            )

    def record(self, builder):
        a, k = ti.graph.Arg, ti.graph.ArgKind
        builder.dispatch(
            self.generate,
            a(k.NDARRAY, "transforms", ti.f32, ndim=3),
            a(k.NDARRAY, "shift", ti.f32, ndim=1),
            a(k.NDARRAY, "rays", ti.f32, ndim=2),
            template_args={"self": self},
        )
        builder.append_native(self.refit, admission="auto")
        builder.append_native(self.query, admission="auto")
        builder.append_native(self.occlusion, admission="auto")
        builder.dispatch(
            self.publish,
            a(k.NDARRAY, "hits", ti.f32, ndim=2),
            a(k.NDARRAY, "hit_indices", ti.i32, ndim=2),
            a(k.NDARRAY, "occluded", ti.i32, ndim=1),
            template_args={"self": self},
        )


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_optix_component_new_geometry_extents_and_retirement(tmp_path):
    adapter = os.environ.get("TAICHI_FORGE_TEST_OPTIX_PROVIDER")
    if not adapter:
        pytest.skip("explicit OptiX adapter required")
    from taichi_forge.hardware._optix import load_provider

    with load_provider(provider_path=adapter) as provider:
        vertices, indices = ti.ndarray(ti.f32, (3, 3)), ti.ndarray(ti.i32, (1, 3))
        vertices.from_numpy(np.array([[-2, -2, 0], [2, -2, 0], [0, 2, 0]], np.float32))
        indices.from_numpy(np.array([[0, 1, 2]], np.int32))
        with provider.triangle_gas(vertices, indices) as gas:
            with provider.instance_scene(
                [ti.hardware.ray.OptixRayInstance(gas)]
            ) as scene:
                model = OptixComponent(2, scene)
                arrays = {
                    name: ti.ndarray(dtype, shape)
                    for name, dtype, shape in (
                        ("transforms", ti.f32, (1, 3, 4)),
                        ("shift", ti.f32, 1),
                        ("rays", ti.f32, (2, 8)),
                        ("hits", ti.f32, (2, 4)),
                        ("hit_indices", ti.i32, (2, 4)),
                        ("occluded", ti.i32, 2),
                    )
                }
                arrays["shift"].fill(0.25)
                builder = ti.graph.GraphBuilder()
                model.record(builder)
                graph = builder.compile()
                graph.run(dict(scene=scene, **arrays))
                np.testing.assert_array_equal(model.output.to_numpy(), [1.75, 1.75])
                with pytest.raises(ti.TaichiRuntimeError, match="recording owner"):
                    model.query.validate_graph_bindings(dict(scene=gas, **arrays))
                root = ti.aot.Component.from_owner(model).compile_graph(model.record)
                module = ti.aot.Module()
                plan = root.export(module, "optix")
                document = json.loads(plan.to_json())
                assert not document.get("resources") and not module._data_payloads
                queries = [
                    op
                    for op in document["operations"]
                    if op.get("operation") == "query"
                ]
                assert len(queries) == 2 and all(
                    op["version"] == 4 and "ray_count" not in op["parameters"]
                    for op in queries
                )
                artifact = tmp_path / "optix.tcm"
                module.archive(str(artifact))
                graph.close()
                model.tree.destroy()
    ti.reset()
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(Path(__file__).with_name("aot_component_optix_probe.py")),
            str(artifact),
        ],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "COMPONENT_OPTIX_OK" in result.stdout
