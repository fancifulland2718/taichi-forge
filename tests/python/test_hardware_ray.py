import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.graph._ir import GraphAccess
from tests import test_utils


@test_utils.test(arch=ti.cpu)
def test_vulkan_triangle_ray_contract_rejects_non_vulkan_runtime():
    vertices = ti.ndarray(ti.f32, shape=(3, 3))
    indices = ti.ndarray(ti.i32, shape=(1, 3))

    assert not ti.hardware.ray.is_available()
    with pytest.raises(RuntimeError, match="requires the Vulkan backend"):
        ti.hardware.ray.TriangleScene(vertices, indices)

    descriptor = ti.hardware.capability("ray.query.batch.vulkan")
    assert descriptor.implementation_status == "existing_public"
    assert descriptor.scopes == ("python", "graph")
    assert descriptor.graph_support == "recordable"
    assert descriptor.execution_kind == "native_command"
    assert descriptor.workspace_ownership == "provider_owned"


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_triangle_ray_query_executes_directly_and_through_graph():
    if not ti.hardware.ray.is_available():
        pytest.skip(
            "Vulkan ray query requires buffer-device-address, acceleration-"
            "structure, and ray-query features"
        )

    resolved = next(
        operation
        for operation in ti.hardware.report().operations
        if operation.descriptor.operation_id == "ray.query.batch.vulkan"
    )
    assert resolved.discovery == "available"
    assert resolved.enablement == "enabled"
    assert resolved.selection == "eligible"

    vertices = ti.ndarray(ti.f32, shape=(3, 3))
    indices = ti.ndarray(ti.i32, shape=(1, 3))
    rays = ti.ndarray(ti.f32, shape=(3, 8))
    hits = ti.ndarray(ti.f32, shape=(3, 4))
    vertices.from_numpy(
        np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], dtype=np.float32)
    )
    indices.from_numpy(np.array([[0, 1, 2]], dtype=np.int32))
    rays.from_numpy(
        np.array(
            [
                [0, 0, 1, 0.001, 0, 0, -1, 100],
                [2, 0, 1, 0.001, 0, 0, -1, 100],
                [0, 0, -1, 0.001, 0, 0, 1, 100],
            ],
            dtype=np.float32,
        )
    )
    expected = np.array(
        [[1, 0, 0, 1], [-1, -1, -1, 0], [1, 0, 0, 1]],
        dtype=np.float32,
    )

    scene = ti.hardware.ray.TriangleScene(vertices, indices)
    scene.trace(rays, hits)
    ti.sync()
    np.testing.assert_allclose(hits.to_numpy(), expected, rtol=0, atol=1e-5)

    hits.fill(0)
    recording = scene.record(3)
    assert tuple(
        (effect.resource, effect.access) for effect in recording.resource_effects
    ) == (("rays", GraphAccess.READ), ("hits", GraphAccess.WRITE))
    builder = ti.graph.GraphBuilder()
    builder.append_native(recording, admission="auto")
    graph = builder.compile()
    graph.run({"rays": rays, "hits": hits})
    ti.sync()
    np.testing.assert_allclose(hits.to_numpy(), expected, rtol=0, atol=1e-5)
    assert graph._debug_info["optimization"]["backend_command_nodes"] == 1

    scene.close()
    with pytest.raises(RuntimeError, match="closed"):
        graph.run({"rays": rays, "hits": hits})


@test_utils.test(arch=ti.vulkan)
def test_vulkan_triangle_ray_layout_and_recording_validation():
    if not ti.hardware.ray.is_available():
        pytest.skip("Vulkan ray query features are unavailable")

    vertices = ti.ndarray(ti.f32, shape=(3, 2))
    indices = ti.ndarray(ti.i32, shape=(1, 3))
    with pytest.raises(RuntimeError, match=r"shape \(N, 3\)"):
        ti.hardware.ray.TriangleScene(vertices, indices)

    with pytest.raises(TypeError, match="TriangleScene"):
        ti.hardware.ray.VulkanRayQueryRecording(object(), 1)
    uninitialized_scene = object.__new__(ti.hardware.ray.TriangleScene)
    with pytest.raises(ValueError, match="count"):
        ti.hardware.ray.VulkanRayQueryRecording(uninitialized_scene, 0)
