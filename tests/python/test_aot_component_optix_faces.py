"""One stable face-table port covers new IAS sizes and heterogeneous GAS counts."""

from contextlib import ExitStack, closing
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.test_aot_component_optix import OptixComponent


@ti.data_oriented
class FaceComponent(OptixComponent):
    def __init__(self, count, scene, fixed):
        super().__init__(count, scene)
        table = ti.hardware.ray.OptixFaceRuleTable("faces")
        self.query = scene.record_typed(
            count, scene="scene", face_rules=table, graph_fixed_query=fixed
        )
        self.occlusion = scene.record_occlusion(
            count, scene="scene", face_rules=table, graph_fixed_query=fixed
        )

    def record(self, builder):
        builder.append_native(self.query, admission="auto")
        builder.append_native(self.occlusion, admission="auto")
        a, k = ti.graph.Arg, ti.graph.ArgKind
        builder.dispatch(
            self.publish,
            a(k.NDARRAY, "hits", ti.f32, ndim=2),
            a(k.NDARRAY, "hit_indices", ti.i32, ndim=2),
            a(k.NDARRAY, "occluded", ti.i32, ndim=1),
            template_args={"self": self},
        )


@test_utils.test(arch=ti.cuda, offline_cache=False)
@pytest.mark.parametrize("fixed", [False, True])
def test_optix_component_flat_face_partitions(tmp_path, fixed):
    adapter = os.environ.get("TAICHI_FORGE_TEST_OPTIX_PROVIDER")
    if not adapter:
        pytest.skip("explicit OptiX adapter required")
    from taichi_forge.hardware._optix import load_provider

    # Keep the runtime-only helper independent of test package imports.
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from aot_component_optix_faces_probe import geometry
    finally:
        sys.path.pop(0)
    vertices, indices, transforms, rays, faces, expected = geometry(2, 2, 0)

    def allocate(dtype, data, shape):
        value = ti.ndarray(dtype, shape)
        value.from_numpy(
            np.asarray(data, dtype=np.float32 if dtype == ti.f32 else np.int32).reshape(
                shape
            )
        )
        return value

    with ExitStack() as life:
        provider = life.enter_context(load_provider(provider_path=adapter))
        gas = []
        for n in (1, 2):
            vd = allocate(ti.f32, vertices[: 9 * n], (3 * n, 3))
            ix = allocate(ti.i32, indices[: 3 * n], (n, 3))
            gas.append(life.enter_context(provider.triangle_gas(vd, ix)))
        scene = life.enter_context(
            provider.instance_scene(
                [
                    ti.hardware.ray.OptixRayInstance(
                        gas[i],
                        transform=np.asarray(transforms[12 * i : 12 * (i + 1)]).reshape(
                            3, 4
                        ),
                    )
                    for i in range(2)
                ]
            )
        )
        n = len(faces)
        model = FaceComponent(n, scene, fixed)
        repeated = scene.record_typed(n, face_rules=(ti.hardware.ray.OptixFaceRuleTable("faces"),) * 2)
        assert model.query._face_identity != repeated._face_identity
        life.callback(model.tree.destroy)
        args = dict(
            scene=scene,
            rays=allocate(ti.f32, rays, (n, 8)),
            hits=ti.ndarray(ti.f32, (n, 4)),
            hit_indices=ti.ndarray(ti.i32, (n, 4)),
            occluded=ti.ndarray(ti.i32, n),
            faces=allocate(ti.i32, faces, (n,)),
        )
        builder = ti.graph.GraphBuilder()
        model.record(builder)
        graph = life.enter_context(closing(builder.compile()))
        graph.run(args)
        np.testing.assert_array_equal(model.output.to_numpy(), expected)
        # Legacy direct path also uses the same instance partition descriptors.
        for recording in (model.query, model.occlusion):
            recording.execute({name: args[name] for name in recording.binding_names})
        model.publish(args["hits"], args["hit_indices"], args["occluded"])
        np.testing.assert_array_equal(model.output.to_numpy(), expected)
        with pytest.raises(ti.TaichiRuntimeError, match="all instance primitives"):
            model.query.validate_graph_bindings(
                dict(args, faces=ti.ndarray(ti.i32, n + 1))
            )
        root = ti.aot.Component.from_owner(model).compile_graph(model.record)
        module = ti.aot.Module()
        plan = root.export(module, "faces")
        doc = json.loads(plan.to_json())
        queries = [op for op in doc["operations"] if op.get("operation") == "query"]
        assert len(queries) == 2 and all(
            op["version"] == 5
            and set(op["parameters"]) == {"occlusion", "any_hit", "graph_fixed_query"}
            for op in queries
        )
        assert not doc.get("resources") and not module._data_payloads
        artifact = tmp_path / "faces.tcm"
        module.archive(str(artifact))
    ti.reset()
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(Path(__file__).with_name("aot_component_optix_faces_probe.py")),
            str(artifact),
        ],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "COMPONENT_OPTIX_FACES_OK" in result.stdout
