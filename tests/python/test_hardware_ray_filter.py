"""Device-side candidate filtering, resource visibility, and scoped rejection."""

import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


def _layers():
    if not ti.hardware.ray.is_available():
        pytest.skip("Vulkan ray query is unavailable")
    vertices = ti.ndarray(ti.f32, (3, 3))
    indices = ti.ndarray(ti.i32, (1, 3))
    vertices.from_numpy(np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], np.float32))
    indices.from_numpy(np.array([[0, 1, 2]], np.int32))
    blas = ti.hardware.ray.TriangleBLAS(vertices, indices)
    scene = ti.hardware.ray.InstanceTLAS(
        [
            ti.hardware.ray.RayInstance(
                blas,
                transform=(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, z),
                custom_index=custom,
            )
            for z, custom in ((2, 11), (1, 22), (0, 33))
        ]
    )
    return blas, scene


@ti.func
def _accept_material(
    hit: ti.template(), materials: ti.template(), alpha: ti.template()
):
    # 0: absent, 1: alpha mask, 2: opaque material. Barycentrics are caller UVs
    # for this triangle; no material interpretation lives in the query intrinsic.
    mode = materials[hit.instance_id]
    uv = ti.Vector([hit.barycentric_u, hit.barycentric_v])
    return mode == 2 or (mode == 1 and alpha.sample_lod(uv, 0.0).x >= 0.5)


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_filtered_ray_nearest_any_mask_updates_and_graph_lifetime():
    blas, scene = _layers()
    materials = ti.ndarray(ti.i32, 3)
    ids = ti.ndarray(ti.i32, (129, 4))
    values = ti.ndarray(ti.f32, (129, 3))
    alpha = ti.Texture(ti.Format.r32f, (4, 2))
    upload = ti.ndarray(ti.f32, 8)

    @ti.kernel
    def update(
        materials: ti.types.ndarray(ti.i32, ndim=1), front: ti.i32, back: ti.i32
    ):
        materials[0] = front
        materials[1] = back
        materials[2] = 0

    @ti.kernel
    def trace(
        scene: ti.types.acceleration_structure(),
        materials: ti.types.ndarray(ti.i32, ndim=1),
        alpha: ti.types.texture(num_dimensions=2),
        ids: ti.types.ndarray(ti.i32, ndim=2),
        values: ti.types.ndarray(ti.f32, ndim=2),
    ):
        for i in range(ids.shape[0]):
            origin = ti.Vector([0.5 + 10.0 * ti.cast(i % 3 == 2, ti.f32), 0.5, 3.0])
            direction = ti.Vector([0.0, 0.0, -1.0])
            nearest = scene.trace_closest_filtered(
                origin, direction, _accept_material, args=(materials, alpha)
            )
            any_hit = scene.trace_any_filtered(
                origin, direction, _accept_material, args=(materials, alpha)
            )
            ids[i, 0] = nearest.hit
            ids[i, 1] = ti.cast(nearest.instance_custom_index, ti.i32)
            ids[i, 2] = ti.cast(nearest.primitive_index, ti.i32)
            ids[i, 3] = any_hit.hit
            values[i, 0] = nearest.t
            values[i, 1] = nearest.barycentric_u
            values[i, 2] = nearest.barycentric_v

    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        trace,
        ti.graph.Arg(ti.graph.ArgKind.ACCELERATION_STRUCTURE, "scene"),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "materials", ti.i32, ndim=1),
        ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "alpha", ndim=2),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "ids", ti.i32, ndim=2),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "values", ti.f32, ndim=2),
    )
    graph = builder.compile()
    bound = graph.bind(
        dict(scene=scene, materials=materials, alpha=alpha, ids=ids, values=values)
    )
    for front, back, opacity, custom, distance in (
        (1, 2, 0.0, 22, 2.0),  # reject front mask, keep opaque back
        (
            1,
            2,
            1.0,
            11,
            1.0,
        ),  # accepted front must be nearest, independent of traversal order
        (1, 0, 0.0, -1, -1.0),  # all rejected
        (0, 1, 1.0, 22, 2.0),  # device-updated material table, same binding
    ):
        update(materials, front, back)
        upload.fill(opacity)
        ti.hardware.image.VulkanBufferToImageRecording().execute(
            dict(source=upload, destination=alpha)
        )
        graph.run(bound)
        ti.sync()
        valid = np.arange(129) % 3 != 2
        if custom == -1:
            valid[:] = False
        expected = np.tile([0, -1, -1, 0], (129, 1)).astype(np.int32)
        expected[valid] = [1, custom, 0, 1]
        np.testing.assert_array_equal(ids.to_numpy(), expected)
        expected_values = np.tile([-1, 0, 0], (129, 1)).astype(np.float32)
        expected_values[valid] = [distance, 0.25, 0.25]
        np.testing.assert_allclose(values.to_numpy(), expected_values, atol=1e-6)
    graph.close()
    scene.close()
    blas.close()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_filtered_ray_rejects_side_effects_and_python_callbacks():
    blas, scene = _layers()
    output = ti.ndarray(ti.i32, 1)

    @ti.func
    def bad(hit: ti.template(), target: ti.template()):
        target[0] = 19
        return 1

    @ti.kernel
    def trace(
        scene: ti.types.acceleration_structure(),
        target: ti.types.ndarray(ti.i32, ndim=1),
    ):
        hit = scene.trace_closest_filtered(
            ti.Vector([0.5, 0.5, 3.0]), ti.Vector([0.0, 0.0, -1.0]), bad, args=(target,)
        )
        target[0] = hit.hit

    output.fill(7)
    with pytest.raises(Exception, match="read-only"):
        trace(scene, output)
    np.testing.assert_array_equal(output.to_numpy(), [7])

    @ti.kernel
    def callback(scene: ti.types.acceleration_structure()):
        hit = scene.trace_closest_filtered(
            ti.Vector([0.5, 0.5, 3.0]), ti.Vector([0.0, 0.0, -1.0]), int
        )

    with pytest.raises(Exception, match="inlined ti.func"):
        callback(scene)

    @ti.func
    def bad_local(hit: ti.template(), state: ti.template()):
        state[0] = 99
        return 1

    @ti.kernel
    def mutate_enclosing_local(scene: ti.types.acceleration_structure()) -> ti.i32:
        state = ti.Vector([7])
        hit = scene.trace_closest_filtered(
            ti.Vector([0.5, 0.5, 3.0]), ti.Vector([0.0, 0.0, -1.0]),
            bad_local, args=(state,)
        )
        return state[0] + hit.hit

    with pytest.raises(Exception, match="read-only"):
        mutate_enclosing_local(scene)

    @ti.func
    def shared_write(hit: ti.template()):
        shared = ti.simt.block.SharedArray((1,), ti.i32)
        shared[0] = 1
        return shared[0]

    @ti.kernel
    def mutate_shared(scene: ti.types.acceleration_structure()):
        for i in range(64):
            hit = scene.trace_closest_filtered(
                ti.Vector([0.5, 0.5, 3.0]), ti.Vector([0.0, 0.0, -1.0]), shared_write
            )

    with pytest.raises(Exception, match="read-only"):
        mutate_shared(scene)

    @ti.func
    def own_local(hit: ti.template()):
        total = 0
        for j in range(2):
            total += j
        return total == 1 and hit.instance_id == 1

    @ti.kernel
    def local_work(scene: ti.types.acceleration_structure()) -> ti.i32:
        hit = scene.trace_closest_filtered(
            ti.Vector([0.5, 0.5, 3.0]), ti.Vector([0.0, 0.0, -1.0]), own_local
        )
        return ti.cast(hit.instance_custom_index, ti.i32)

    assert local_work(scene) == 22
    scene.close()
    blas.close()
