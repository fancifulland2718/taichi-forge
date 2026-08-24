import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.graph._ir import GraphAccess
from taichi_forge.lang import impl
from tests import test_utils
from tests.python.hardware_provider_lifecycle_qualification import (
    stress_iterations,
)


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
    assert descriptor.graph_integration == "root_ordered"
    assert descriptor.execution_kind == "native_command"
    assert descriptor.workspace_ownership == "provider_owned"

    refit = ti.hardware.capability("ray.as_refit.vulkan")
    assert refit.implementation_status == "existing_public"
    assert refit.scopes == ("python", "graph")
    assert refit.graph_integration == "root_ordered"
    assert refit.hardware_acceleration == "implementation_defined"
    assert refit.workspace_ownership == "provider_owned"

    build = ti.hardware.capability("ray.as_build.vulkan")
    assert build.scopes == ("python", "graph")
    assert "TriangleBLAS" in build.public_api


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
        (effect.resource, effect.access, effect.runtime_bound)
        for effect in recording.resource_effects
    ) == (
        ("rays", GraphAccess.READ, True),
        ("hits", GraphAccess.WRITE, True),
        (scene._effect_name, GraphAccess.READ, False),
    )
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


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_triangle_ray_refit_executes_directly_and_through_graph():
    if not ti.hardware.ray.is_available():
        pytest.skip("Vulkan ray query features are unavailable")

    vertices = ti.ndarray(ti.f32, shape=(3, 3))
    updated = ti.ndarray(ti.f32, shape=(3, 3))
    indices = ti.ndarray(ti.i32, shape=(1, 3))
    rays = ti.ndarray(ti.f32, shape=(1, 8))
    hits = ti.ndarray(ti.f32, shape=(1, 4))
    vertices.from_numpy(
        np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], dtype=np.float32)
    )
    indices.from_numpy(np.array([[0, 1, 2]], dtype=np.int32))
    rays.from_numpy(
        np.array([[0, 0, 1, 0.001, 0, 0, -1, 100]], dtype=np.float32)
    )

    scene = ti.hardware.ray.TriangleScene(vertices, indices)
    updated.from_numpy(
        np.array([[-1, -1, -2], [1, -1, -2], [0, 1, -2]], dtype=np.float32)
    )
    assert scene.refit(updated) is scene
    scene.trace(rays, hits)
    ti.sync()
    np.testing.assert_allclose(
        hits.to_numpy(), np.array([[3, 0, 0, 1]], dtype=np.float32), atol=1e-5
    )

    updated.from_numpy(
        np.array([[-1, -1, 0.5], [1, -1, 0.5], [0, 1, 0.5]], dtype=np.float32)
    )
    recording = scene.record_refit(vertices="positions")
    assert tuple(
        (effect.resource, effect.access, effect.runtime_bound)
        for effect in recording.resource_effects
    ) == (
        ("positions", GraphAccess.READ, True),
        (scene._effect_name, GraphAccess.WRITE, False),
    )
    builder = ti.graph.GraphBuilder()
    builder.append_native(recording, admission="auto")
    builder.append_native(scene.record(1), admission="auto")
    graph = builder.compile()
    graph.run({"positions": updated, "rays": rays, "hits": hits})
    ti.sync()
    np.testing.assert_allclose(
        hits.to_numpy(), np.array([[0.5, 0, 0, 1]], dtype=np.float32), atol=1e-5
    )
    assert graph._debug_info["optimization"]["backend_command_nodes"] == 2

    wrong_vertices = ti.ndarray(ti.f32, shape=(4, 3))
    with pytest.raises(RuntimeError, match="wrong vertex count"):
        scene.refit(wrong_vertices)
    scene.close()
    with pytest.raises(RuntimeError, match="closed"):
        graph.run({"positions": updated, "rays": rays, "hits": hits})


@test_utils.test(arch=ti.vulkan, offline_cache=False, debug=True)
def test_vulkan_independent_blas_tlas_transform_metadata_and_graph_updates():
    if not ti.hardware.ray.is_available():
        pytest.skip("Vulkan ray query features are unavailable")

    vertices = ti.ndarray(ti.f32, shape=(3, 3))
    lowered = ti.ndarray(ti.f32, shape=(3, 3))
    indices = ti.ndarray(ti.i32, shape=(1, 3))
    rays = ti.ndarray(ti.f32, shape=(3, 8))
    hits = ti.ndarray(ti.f32, shape=(3, 4))
    vertices.from_numpy(
        np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], dtype=np.float32)
    )
    lowered.from_numpy(
        np.array([[-1, -1, -1], [1, -1, -1], [0, 1, -1]], dtype=np.float32)
    )
    indices.from_numpy(np.array([[0, 1, 2]], dtype=np.int32))
    rays.from_numpy(
        np.array(
            [
                [2, 0, 1, 0.001, 0, 0, -1, 100],
                [0, 0, 1, 0.001, 0, 0, -1, 100],
                [-2, 0, 1, 0.001, 0, 0, -1, 100],
            ],
            dtype=np.float32,
        )
    )

    blas = ti.hardware.ray.TriangleBLAS(vertices, indices)
    translated = (
        1,
        0,
        0,
        2,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
    )
    masked_translation = (
        1,
        0,
        0,
        -2,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
    )
    tlas = ti.hardware.ray.InstanceTLAS(
        [
            ti.hardware.ray.RayInstance(
                blas, transform=translated, custom_index=7
            ),
            ti.hardware.ray.RayInstance(
                blas,
                transform=masked_translation,
                mask=0,
                custom_index=9,
            ),
        ]
    )
    tlas.trace(rays, hits)
    ti.sync()
    np.testing.assert_allclose(
        hits.to_numpy(),
        np.array(
            [[1, 0, 7, 1], [-1, -1, -1, 0], [-1, -1, -1, 0]],
            dtype=np.float32,
        ),
        rtol=0,
        atol=1e-5,
    )

    builder = ti.graph.GraphBuilder()
    builder.append_native(
        blas.record_build(vertices="updated_vertices", indices="triangles"),
        admission="auto",
    )
    builder.append_native(tlas.record(3), admission="auto")
    graph = builder.compile()
    graph.run(
        {
            "updated_vertices": lowered,
            "triangles": indices,
            "rays": rays,
            "hits": hits,
        }
    )
    ti.sync()
    np.testing.assert_allclose(
        hits.to_numpy(),
        np.array(
            [[2, 0, 7, 1], [-1, -1, -1, 0], [-1, -1, -1, 0]],
            dtype=np.float32,
        ),
        rtol=0,
        atol=1e-5,
    )

    identity_instances = [
        ti.hardware.ray.RayInstance(blas, custom_index=11),
        ti.hardware.ray.RayInstance(blas, mask=0, custom_index=12),
    ]
    with pytest.raises(RuntimeError, match="preserve BLAS count and order"):
        tlas.refit(identity_instances[:1])
    builder = ti.graph.GraphBuilder()
    builder.append_native(
        blas.record_refit(vertices="restored_vertices"), admission="auto"
    )
    builder.append_native(tlas.record_refit(identity_instances), admission="auto")
    builder.append_native(tlas.record(3), admission="auto")
    graph = builder.compile()
    graph.run({"restored_vertices": vertices, "rays": rays, "hits": hits})
    ti.sync()
    np.testing.assert_allclose(
        hits.to_numpy(),
        np.array(
            [[-1, -1, -1, 0], [1, 0, 11, 1], [-1, -1, -1, 0]],
            dtype=np.float32,
        ),
        rtol=0,
        atol=1e-5,
    )
    tlas.close()
    blas.close()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_independent_tlas_retains_closed_blas_and_defers_close():
    if not ti.hardware.ray.is_available():
        pytest.skip("Vulkan ray query features are unavailable")

    vertices = ti.ndarray(ti.f32, shape=(3, 3))
    indices = ti.ndarray(ti.i32, shape=(1, 3))
    rays = ti.ndarray(ti.f32, shape=(1, 8))
    hits = ti.ndarray(ti.f32, shape=(1, 4))
    vertices.from_numpy(
        np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], dtype=np.float32)
    )
    indices.from_numpy(np.array([[0, 1, 2]], dtype=np.int32))
    rays.from_numpy(
        np.array([[0, 0, 1, 0.001, 0, 0, -1, 100]], dtype=np.float32)
    )
    ti.sync()
    program = impl.get_runtime().prog
    baseline = dict(program._debug_vulkan_ray_resource_stats())

    blas = ti.hardware.ray.TriangleBLAS(vertices, indices)
    tlas = ti.hardware.ray.InstanceTLAS([ti.hardware.ray.RayInstance(blas)])
    blas.close()
    tlas.trace(rays, hits)
    ti.sync()
    np.testing.assert_allclose(
        hits.to_numpy(), np.array([[1, 0, 0, 1]], dtype=np.float32), atol=1e-5
    )

    tlas.trace(rays, hits)
    waits_before = program._runtime_statistics_snapshot()["synchronization"][
        "backend_waits"
    ]
    tlas.close()
    waits_after = program._runtime_statistics_snapshot()["synchronization"][
        "backend_waits"
    ]
    retiring = dict(program._debug_vulkan_ray_resource_stats())
    assert waits_after == waits_before
    assert retiring["independent_live"] == baseline["independent_live"]
    assert retiring["independent_retiring"] >= baseline[
        "independent_retiring"
    ] + 1
    assert retiring["independent_completion_retained"] >= 1

    ti.sync()
    completed = dict(program._debug_vulkan_ray_resource_stats())
    assert completed["independent_live"] == baseline["independent_live"]
    assert completed["independent_retiring"] == baseline["independent_retiring"]


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_triangle_ray_close_defers_inflight_scene_without_waiting():
    if not ti.hardware.ray.is_available():
        pytest.skip("Vulkan ray query features are unavailable")

    vertices = ti.ndarray(ti.f32, shape=(3, 3))
    indices = ti.ndarray(ti.i32, shape=(1, 3))
    rays = ti.ndarray(ti.f32, shape=(1, 8))
    hits = ti.ndarray(ti.f32, shape=(1, 4))
    vertices.from_numpy(
        np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], dtype=np.float32)
    )
    indices.from_numpy(np.array([[0, 1, 2]], dtype=np.int32))
    rays.from_numpy(np.array([[0, 0, 1, 0.001, 0, 0, -1, 100]], dtype=np.float32))
    ti.sync()
    program = impl.get_runtime().prog
    baseline = dict(program._debug_vulkan_ray_resource_stats())

    scene = ti.hardware.ray.TriangleScene(vertices, indices)
    scene.trace(rays, hits)
    waits_before = program._runtime_statistics_snapshot()["synchronization"][
        "backend_waits"
    ]
    scene.close()

    waits_after = program._runtime_statistics_snapshot()["synchronization"][
        "backend_waits"
    ]
    retiring = dict(program._debug_vulkan_ray_resource_stats())
    assert waits_after == waits_before
    assert retiring["live"] == baseline["live"]
    assert retiring["retiring"] == baseline["retiring"] + 1
    assert retiring["completion_retained"] >= 1

    ti.sync()
    completed = dict(program._debug_vulkan_ray_resource_stats())
    assert completed["live"] == baseline["live"]
    assert completed["retiring"] == baseline["retiring"]


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
    with pytest.raises(TypeError, match="TriangleScene"):
        ti.hardware.ray.VulkanRayRefitRecording(object())
    uninitialized_scene = object.__new__(ti.hardware.ray.TriangleScene)
    with pytest.raises(ValueError, match="count"):
        ti.hardware.ray.VulkanRayQueryRecording(uninitialized_scene, 0)

    uninitialized_blas = object.__new__(ti.hardware.ray.TriangleBLAS)
    instance = ti.hardware.ray.RayInstance(
        uninitialized_blas,
        transform=((1, 0, 0, 2), (0, 1, 0, 3), (0, 0, 1, 4)),
    )
    assert instance.transform[3::4] == (2.0, 3.0, 4.0)
    with pytest.raises(ValueError, match="3x4"):
        ti.hardware.ray.RayInstance(uninitialized_blas, transform=(1, 2, 3))
    with pytest.raises(ValueError, match="finite"):
        ti.hardware.ray.RayInstance(
            uninitialized_blas,
            transform=(float("nan"),) + (0,) * 11,
        )
    with pytest.raises(ValueError, match="mask"):
        ti.hardware.ray.RayInstance(uninitialized_blas, mask=256)
    with pytest.raises(ValueError, match="custom_index"):
        ti.hardware.ray.RayInstance(uninitialized_blas, custom_index=0x1000000)


@pytest.mark.run_in_serial
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_ray_serial_churn_releases_all_generations():
    if not ti.hardware.ray.is_available():
        pytest.skip("Vulkan ray query features are unavailable")

    iterations = stress_iterations(8)
    vertices = ti.ndarray(ti.f32, shape=(3, 3))
    indices = ti.ndarray(ti.i32, shape=(1, 3))
    rays = ti.ndarray(ti.f32, shape=(1, 8))
    hits = ti.ndarray(ti.f32, shape=(1, 4))
    vertices.from_numpy(
        np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], dtype=np.float32)
    )
    indices.from_numpy(np.array([[0, 1, 2]], dtype=np.int32))
    rays.from_numpy(np.array([[0, 0, 1, 0.001, 0, 0, -1, 100]], dtype=np.float32))
    ti.sync()
    program = impl.get_runtime().prog
    baseline = dict(program._debug_vulkan_ray_resource_stats())
    midpoint = None

    for iteration in range(iterations):
        if iteration % 2 == 0:
            owner = ti.hardware.ray.TriangleScene(vertices, indices)
            owner.trace(rays, hits)
            owner.close()
        else:
            blas = ti.hardware.ray.TriangleBLAS(vertices, indices)
            owner = ti.hardware.ray.InstanceTLAS([ti.hardware.ray.RayInstance(blas)])
            owner.trace(rays, hits)
            owner.close()
            blas.close()
        if (iteration + 1) % 32 == 0:
            ti.sync()
        if iteration + 1 == max(1, iterations // 2):
            ti.sync()
            midpoint = dict(program._debug_vulkan_ray_resource_stats())
    ti.sync()

    final = dict(program._debug_vulkan_ray_resource_stats())
    for key in (
        "live",
        "retiring",
        "completion_retained",
        "independent_live",
        "independent_retiring",
        "independent_completion_retained",
    ):
        assert midpoint[key] == baseline[key]
        assert final[key] == baseline[key]
    np.testing.assert_allclose(
        hits.to_numpy(), np.array([[1, 0, 0, 1]], dtype=np.float32), atol=1e-5
    )


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_ray_plan_and_graph_fail_closed_after_runtime_reset():
    if not ti.hardware.ray.is_available():
        pytest.skip("Vulkan ray query features are unavailable")

    vertices = ti.ndarray(ti.f32, shape=(3, 3))
    indices = ti.ndarray(ti.i32, shape=(1, 3))
    rays = ti.ndarray(ti.f32, shape=(1, 8))
    hits = ti.ndarray(ti.f32, shape=(1, 4))
    vertices.from_numpy(
        np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], dtype=np.float32)
    )
    indices.from_numpy(np.array([[0, 1, 2]], dtype=np.int32))
    scene = ti.hardware.ray.TriangleScene(vertices, indices)
    builder = ti.graph.GraphBuilder()
    builder.append_native(scene.record(1), admission="auto")
    graph = builder.compile()

    ti.reset()

    with pytest.raises(RuntimeError, match="previous Taichi runtime"):
        scene.record(1)
    with pytest.raises(RuntimeError, match="compiled before ti.reset"):
        graph.run({"rays": rays, "hits": hits})
    scene.close()
