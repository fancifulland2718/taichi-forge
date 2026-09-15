import os

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.hardware import _optix
from tests import test_utils


@pytest.mark.parametrize("instanced", [False, True])
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_compact_occlusion_prepared_graph_updates_and_lifetime(instanced, monkeypatch):
    path = os.environ.get("TAICHI_FORGE_TEST_OPTIX_PROVIDER")
    if not path and _optix.probe_provider()["discovery"] != "present":
        pytest.skip("OptiX runtime unavailable")
    provider = ti.hardware.ray.load_optix_provider(provider_path=path)
    scene = gas = graph = None
    try:
        vertices, indices = ti.ndarray(ti.f32, (3, 3)), ti.ndarray(ti.i32, (1, 3))
        vertices.from_numpy(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32))
        indices.from_numpy(np.array([[0, 1, 2]], np.int32))
        if instanced:
            gas = provider.triangle_gas(vertices, indices)
            scene = provider.instance_scene((ti.hardware.ray.OptixRayInstance(gas, sbt_record_offset=3),))
        else:
            scene = provider.triangle_scene(vertices, indices)
        rays = ti.ndarray(ti.f32, (4, 8))
        rays.from_numpy(
            np.array(
                [
                    [0.2, 0.2, 2, 0, 0, 0, -2, 2],
                    [-1, 0.2, 2, 0, 0, 0, -2, 2],
                    [0.2, 0.2, 2, 0, 0, 0, -2, 0.5],
                    [0.2, 0.2, 2, 1.5, 0, 0, -2, 2],
                ],
                np.float32,
            )
        )
        flag_owner = ti.field(ti.u32, shape=4) if instanced else ti.ndarray(ti.u32, 4)
        flags = ti.experimental.ndarray_view(flag_owner) if instanced else flag_owner
        result = ti.ndarray(ti.u32, 4)

        @ti.kernel
        def consume(src: ti.types.ndarray(ti.u32, ndim=1), dst: ti.types.ndarray(ti.u32, ndim=1)):
            for i in dst:
                dst[i] = src[i] * 7

        recording = scene.record_occlusion(4)
        assert recording.binding_names == ("rays", "occluded")
        assert recording.memory_report().components[-1].requested_bytes == 48
        builder = ti.graph.GraphBuilder()
        builder.append_native(recording, admission="explicit")
        builder.dispatch(
            consume,
            ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "occluded", ti.u32, ndim=1),
            ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "result", ti.u32, ndim=1),
        )
        graph = builder.compile()
        bound = graph.bind(dict(rays=rays, occluded=flags, result=result))
        assert bound.fast_path_qualified

        def no_prepare(*args, **kwargs):
            raise AssertionError("compact query rebuilt storage during replay")

        with monkeypatch.context() as patched:
            patched.setattr(_optix, "_ray_storage", no_prepare)
            for _ in range(3):
                graph.submit(bound).wait()
        np.testing.assert_array_equal(result.to_numpy(), [7, 0, 0, 0])
        replacement = ti.ndarray(ti.u32, 4)
        bound.update(occluded=replacement)
        graph.submit(bound).wait()
        np.testing.assert_array_equal(replacement.to_numpy(), [1, 0, 0, 0])
        with pytest.raises(RuntimeError, match="dtype"):
            recording.prepare_graph_execute(dict(rays=rays, occluded=ti.ndarray(ti.f32, 4)))
        with pytest.raises(RuntimeError, match="ray count"):
            recording.prepare_graph_execute(dict(rays=rays, occluded=ti.ndarray(ti.i32, 3)))
        prepared = recording.prepare_graph_execute(dict(rays=rays, occluded=ti.ndarray(ti.i32, (4, 1))))
        prepared()
        ti.sync()
        graph.close()
        graph = None
        scene.close()
        with pytest.raises(RuntimeError, match="closed"):
            prepared()
    finally:
        if graph is not None:
            graph.close()
        if scene is not None:
            scene.close()
        if gas is not None:
            gas.close()
        provider.close()
