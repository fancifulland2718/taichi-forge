"""Opaque first-accepted queries must not require an alpha-mask table."""

from contextlib import ExitStack
import os

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.hardware import _optix as optix
from tests import test_utils
from tests.python.aot_optix_probe import RAYS
from tests.python.test_hardware_optix_faces import _TRIANGLES, _reference


@pytest.mark.parametrize("instanced", (False, True))
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_opaque_first_hit_direct_and_graph_refit(instanced):
    adapter = os.environ.get("TAICHI_FORGE_TEST_OPTIX_PROVIDER")
    if not adapter:
        pytest.skip("explicit OptiX adapter required")
    vertices, indices = ti.ndarray(ti.f32, (6, 3)), ti.ndarray(ti.i32, (2, 3))
    vertices.from_numpy(_TRIANGLES.reshape(6, 3))
    indices.from_numpy(np.arange(6, dtype=np.int32).reshape(2, 3))
    rays_np = np.asarray(RAYS, np.float32).copy()
    rays_np[:, 7] = (
        1.5  # One accepted candidate at most, regardless of traversal order.
    )
    rays = ti.ndarray(ti.f32, rays_np.shape)
    rays.from_numpy(rays_np)
    hits, ids = ti.ndarray(ti.f32, (len(RAYS), 4)), ti.ndarray(ti.i32, (len(RAYS), 4))
    with ExitStack() as stack:
        provider = stack.enter_context(optix.OptixProvider(provider_path=adapter))
        transforms = [np.eye(4, dtype=np.float32)[:3]]
        customs = [0]
        if instanced:
            gas = stack.enter_context(provider.triangle_gas(vertices, indices))
            transforms.append(transforms[0].copy())
            transforms[1][0, 3] = 4
            customs = [17, 29]
            scene = stack.enter_context(
                provider.instance_scene(
                    tuple(
                        optix.OptixRayInstance(
                            gas,
                            transform=tuple(t.ravel()),
                            custom_index=custom,
                            opaque=bool(i),
                            sbt_record_offset=2 * i,
                        )
                        for i, (t, custom) in enumerate(zip(transforms, customs))
                    )
                )
            )
        else:
            scene = stack.enter_context(provider.triangle_scene(vertices, indices))
        recording = scene.record_typed(len(RAYS), any_hit=True)
        bindings = dict(rays=rays, hits=hits, hit_indices=ids)
        recording.execute(bindings)
        builder = ti.graph.GraphBuilder()
        builder.append_native(recording, admission="explicit")
        graph = builder.compile()
        stack.callback(graph.close)
        bound = graph.bind(bindings)
        for shift in (0, 4, 0.25, 0):
            moved = _TRIANGLES.copy()
            moved[:, :, 2] += shift
            vertices.from_numpy(moved.reshape(6, 3))
            if instanced:
                gas.refit(vertices)
                scene.refit()
            else:
                scene.refit(vertices)
            graph.submit(bound).wait()
            expected_hits, expected_ids = _reference(
                rays_np,
                [(moved, [0, 0], t, custom) for t, custom in zip(transforms, customs)],
            )
            np.testing.assert_allclose(hits.to_numpy(), expected_hits, atol=1e-5)
            np.testing.assert_array_equal(ids.to_numpy(), expected_ids)
