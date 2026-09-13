"""Native micromap traversal, unknown filtering and retained AS ownership."""

from dataclasses import FrozenInstanceError
import struct

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils


def _require_omm():
    if not ti.hardware.ray.is_opacity_micromap_available():
        pytest.skip("Vulkan opacity micromap is unavailable")


def test_vulkan_baked_micromap_identity_and_immutable_packing():
    data = bytearray([0xE4])
    indices = np.array([0, -1, -2, -3, -4], dtype="<i4")
    asset = ti.hardware.ray.VulkanOpacityMicromap(data, [(0, 1, 2)], indices)
    same = ti.hardware.ray.VulkanOpacityMicromap(
        bytes(data), struct.pack("<IHH", 0, 1, 2), indices.tobytes()
    )
    assert asset == same and asset.triangle_count == 5
    optix = ti.hardware.ray.OptixOpacityMicromap(data, [(0, 1, 2)], indices)
    assert asset.fingerprint != optix.fingerprint
    data[0], indices[0] = 0, -1
    assert asset.data == b"\xe4" and asset == same
    with pytest.raises(FrozenInstanceError):
        asset.data = b"\0"


@pytest.mark.parametrize("format", [1, 2])
@pytest.mark.parametrize("indexed", [False, True])
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_micromap_classification_graph_refit_and_lifetime(format, indexed):
    _require_omm()
    mapping = [0, 0, -1, -2, -3, -4] if indexed else None
    triangles = len(mapping) if indexed else 1
    vertices = ti.ndarray(ti.f32, (triangles * 3, 3))
    indices = ti.ndarray(ti.i32, (triangles, 3))
    mesh = np.concatenate(
        [
            np.array([[3 * i, 0, 0], [3 * i + 1, 0, 0], [3 * i, 1, 0]], np.float32)
            for i in range(triangles)
        ]
    )
    vertices.from_numpy(mesh)
    indices.from_numpy(np.arange(3 * triangles, dtype=np.int32).reshape(-1, 3))
    asset = ti.hardware.ray.VulkanOpacityMicromap(
        bytes([0xA if format == 1 else 0xE4]), [(0, 1, format)], mapping
    )
    front = ti.hardware.ray.TriangleBLAS(vertices, indices, opacity_micromap=asset)
    back = ti.hardware.ray.TriangleBLAS(vertices, indices)
    scene = ti.hardware.ray.InstanceTLAS(
        [
            ti.hardware.ray.RayInstance(
                front, transform=(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1), custom_index=11
            ),
            ti.hardware.ray.RayInstance(back, custom_index=22),
        ]
    )
    centers = np.array(
        [[1 / 6, 1 / 6], [1 / 3, 1 / 3], [2 / 3, 1 / 6], [1 / 6, 2 / 3]], np.float32
    )
    coords = ti.ndarray(ti.f32, (triangles * 4, 2))
    coords.from_numpy(
        np.concatenate([centers + [3 * i, 0] for i in range(triangles)]).astype(
            np.float32
        )
    )
    result = ti.ndarray(ti.f32, (triangles * 4, 4))
    gate = ti.ndarray(ti.i32, 1)

    @ti.func
    def accept(hit: ti.template(), gate: ti.template()):
        return hit.instance_id == 1 or gate[0] != 0

    @ti.kernel
    def query(
        scene: ti.types.acceleration_structure(),
        coords: ti.types.ndarray(ti.f32, ndim=2),
        result: ti.types.ndarray(ti.f32, ndim=2),
        gate: ti.types.ndarray(ti.i32, ndim=1),
    ):
        for i in range(coords.shape[0]):
            origin = ti.Vector([coords[i, 0], coords[i, 1], 3.0])
            direction = ti.Vector([0.0, 0.0, -1.0])
            h = scene.trace_closest_filtered(
                origin, direction, accept, args=(gate,), respect_opacity=True
            )
            a = scene.trace_any_filtered(
                origin, direction, accept, args=(gate,), respect_opacity=True
            )
            filtered = scene.trace_closest_filtered(
                origin, direction, accept, args=(gate,)
            )
            result[i, 0] = h.instance_custom_index
            result[i, 1] = h.t
            result[i, 2] = a.hit
            result[i, 3] = filtered.instance_custom_index

    builder = ti.graph.GraphBuilder()
    a, k = ti.graph.Arg, ti.graph.ArgKind
    builder.dispatch(
        query,
        a(k.ACCELERATION_STRUCTURE, "scene"),
        a(k.NDARRAY, "coords", ti.f32, ndim=2),
        a(k.NDARRAY, "result", ti.f32, ndim=2),
        a(k.NDARRAY, "gate", ti.i32, ndim=1),
    )
    graph = builder.compile()
    bound = graph.bind(dict(scene=scene, coords=coords, result=result, gate=gate))
    try:
        states = [0, 1, 0, 1] if format == 1 else [0, 1, 2, 3]
        states = np.array(
            [
                state
                for index in (mapping or [0])
                for state in (states if index >= 0 else [-index - 1] * 4)
            ]
        )
        for moved in (False, True):
            if moved:
                mesh[:, 2] = 0.25
                vertices.from_numpy(mesh)
                front.refit(vertices)
                scene.refit()
            for accepted in (0, 1):
                gate.fill(accepted)
                graph.run(bound)
                ti.sync()
                expected_front = (states == 1) | ((states >= 2) & bool(accepted))
                expected = np.empty((triangles * 4, 4), np.float32)
                expected[:, 0] = np.where(expected_front, 11, 22)
                expected[:, 1] = np.where(expected_front, 1.75 if moved else 2, 3)
                expected[:, 2] = 1
                expected[:, 3] = np.where((states != 0) & bool(accepted), 11, 22)
                np.testing.assert_allclose(result.to_numpy(), expected, atol=1e-6)
        memory = front.memory_report().to_dict()
        components = {item["name"]: item for item in memory["components"]}
        assert components["opacity_micromap_storage"]["requested_bytes"] > 0
        assert not components["opacity_micromap_import_temporary"]["resident"]
        assert components["opacity_micromap_indices"]["requested_bytes"] == (
            4 * triangles if indexed else 0
        )
        front.close()  # TLAS and recorded command references keep native OMM alive.
        graph.run(bound)
        ti.sync()
        np.testing.assert_allclose(result.to_numpy(), expected, atol=1e-6)
        if indexed and format == 2:
            ti.reset()  # A live TLAS owns OMM while the runtime is torn down.
        else:
            scene.close()
        with pytest.raises(Exception, match="closed|invalid|released|runtime|generation"):
            graph.run(bound)
    finally:
        graph.close()
        scene.close()
        front.close()
        back.close()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_micromap_rejected_import_and_predefined_only():
    _require_omm()
    vertices, indices = ti.ndarray(ti.f32, (3, 3)), ti.ndarray(ti.i32, (1, 3))
    vertices.from_numpy(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32))
    indices.from_numpy(np.array([[0, 1, 2]], np.int32))
    program = impl.get_runtime().prog
    before = dict(program._debug_vulkan_ray_resource_stats())
    for asset, message in (
        (ti.hardware.ray.VulkanOpacityMicromap(b"\0", [(1, 1, 2)]), "bounds"),
        (ti.hardware.ray.VulkanOpacityMicromap(b"\0", [(0, 0, 7)]), "format"),
        (ti.hardware.ray.VulkanOpacityMicromap(b"\0", [(0, 0, 1)], [1]), "index"),
    ):
        with pytest.raises(Exception, match=message):
            ti.hardware.ray.TriangleBLAS(vertices, indices, opacity_micromap=asset)
    assert dict(program._debug_vulkan_ray_resource_stats()) == before
    front = ti.hardware.ray.TriangleBLAS(
        vertices,
        indices,
        opacity_micromap=ti.hardware.ray.VulkanOpacityMicromap(b"", [], [-1]),
    )
    scene = ti.hardware.ray.InstanceTLAS([ti.hardware.ray.RayInstance(front)])
    rays, hits = ti.ndarray(ti.f32, (1, 8)), ti.ndarray(ti.f32, (1, 4))
    rays.from_numpy(np.array([[0.2, 0.2, 1, 0, 0, 0, -1, 10]], np.float32))
    scene.trace(rays, hits)
    assert hits.to_numpy()[0, 0] == -1
    scene.close()
    front.close()

    @ti.func
    def reject(hit: ti.template()):
        return 0

    @ti.kernel
    def query(scene: ti.types.acceleration_structure()) -> ti.i32:
        h = scene.trace_closest_filtered(
            ti.Vector([0.2, 0.2, 1.0]),
            ti.Vector([0.0, 0.0, -1.0]),
            reject,
            respect_opacity=True,
        )
        return h.hit

    for opaque in (False, True):
        blas = ti.hardware.ray.TriangleBLAS(vertices, indices, opaque=opaque)
        scene = ti.hardware.ray.InstanceTLAS([ti.hardware.ray.RayInstance(blas)])
        assert blas.opaque is opaque
        assert query(scene) == int(opaque)
        scene.close()
        blas.close()
