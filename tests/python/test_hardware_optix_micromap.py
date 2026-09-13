import ctypes
import os
import struct

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.hardware import _optix
from tests import test_utils


def _provider():
    path = os.environ.get("TAICHI_FORGE_TEST_OPTIX_PROVIDER")
    if not path and _optix.probe_provider()["discovery"] != "present":
        pytest.skip("OptiX runtime unavailable")
    provider = ti.hardware.ray.load_optix_provider(provider_path=path)
    if not int(provider.identity["feature_bits"]) & _optix._OPACITY_MICROMAP_IMPORT:
        provider.close()
        pytest.skip("OMM import adapter unavailable")
    return provider


def test_baked_omm_freezes_external_buffers_and_normalizes_identity():
    data = bytearray([0xE4])
    indices = np.array([0, 0, -1, -2, -3, -4], dtype="<i4")
    packed = struct.pack("<IHH", 0, 1, 2)
    asset = ti.hardware.ray.OptixOpacityMicromap(data, packed, indices)
    same = ti.hardware.ray.OptixOpacityMicromap(
        bytes(data), [(0, 1, 2)], indices.tolist()
    )
    assert asset.fingerprint == same.fingerprint
    data[0], indices[0] = 0, -2
    assert asset.data == b"\xe4" and asset.triangle_count == 6
    native, owners = asset._native()
    assert owners and native.triangle_indices[0] == 0
    assert native.entries[0].subdivision_level == 1
    assert ctypes.sizeof(_optix._MicromapMemory) == 32
    with pytest.raises(ValueError, match="8-byte"):
        ti.hardware.ray.OptixOpacityMicromap(b"\0", b"\0")
    with pytest.raises(ValueError, match="int32"):
        ti.hardware.ray.OptixOpacityMicromap(b"\0", [(0, 0, 1)], b"\0")


@pytest.mark.parametrize("format", [1, 2])
@pytest.mark.parametrize("indexed", [False, True])
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_baked_omm_microtriangle_queries_refit_and_retained_lifetime(format, indexed):
    provider = _provider()
    front = back = scene = graph = None
    try:
        # Four level-1 centroid positions, in native OptiX micromap order.
        # The shader fallback deliberately disagrees with classified cells, so
        # an accidentally forced any-hit / ignored OMM cannot pass this oracle.
        centers = np.array(
            [[1 / 6, 1 / 6], [1 / 3, 1 / 3], [2 / 3, 1 / 6], [1 / 6, 2 / 3]], np.float32
        )
        mapping = [0, 0, -1, -2, -3, -4] if indexed else None
        triangles = 6 if indexed else 2
        count = triangles * 4
        vertices = ti.ndarray(ti.f32, (triangles * 3, 3))
        mesh = np.concatenate(
            [
                np.array([[3 * i, 0, 1], [3 * i + 1, 0, 1], [3 * i, 1, 1]], np.float32)
                for i in range(triangles)
            ]
        )
        vertices.from_numpy(mesh)
        indices = ti.ndarray(ti.i32, (triangles, 3))
        indices.from_numpy(np.arange(triangles * 3, dtype=np.int32).reshape(-1, 3))
        packed = bytes([0xA if format == 1 else 0xE4])
        asset = ti.hardware.ray.OptixOpacityMicromap(
            packed if indexed else packed * 2,
            [(0, 1, format)] if indexed else [(0, 1, format), (1, 1, format)],
            mapping,
        )
        front = provider.triangle_gas(vertices, indices, opacity_micromap=asset)
        back = provider.triangle_gas(vertices, indices)
        with pytest.raises(ValueError, match="always-opaque"):
            provider.instance_scene(
                (ti.hardware.ray.OptixRayInstance(front, opaque=True),)
            )
        scene = provider.instance_scene(
            (
                ti.hardware.ray.OptixRayInstance(front, custom_index=13),
                ti.hardware.ray.OptixRayInstance(
                    back,
                    opaque=True,
                    custom_index=29,
                    transform=(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, -1),
                ),
            )
        )
        with pytest.raises(ValueError, match="typed"):
            scene.record(count)
        rays = ti.ndarray(ti.f32, (count, 8))
        rays_host = np.tile([0, 0, 2, 0, 0, 0, -1, 10], (count, 1)).astype(np.float32)
        for i in range(triangles):
            rays_host[4 * i : 4 * i + 4, :2] = centers + [3 * i, 0]
        rays.from_numpy(rays_host)
        hits, ids = ti.ndarray(ti.f32, (count, 4)), ti.ndarray(ti.u32, (count, 4))
        uvs = ti.ndarray(ti.f32, (triangles * 3, 2))
        uvs.fill(0.25)
        pixels = ti.ndarray(ti.f32, (2, 2))
        texture = ti.Texture(ti.Format.r32f, (2, 2))
        mask = ti.hardware.ray.OptixAlphaMask("uvs", "alpha", channel=0)
        query = scene.record_typed(count, alpha_masks=(mask, None))
        bindings = dict(rays=rays, hits=hits, hit_indices=ids, uvs=uvs, alpha=texture)
        builder = ti.graph.GraphBuilder()
        builder.append_native(query, admission="explicit")
        graph = builder.compile()
        bound = graph.bind(bindings)
        any_hit = scene.record_typed(
            count, alpha_masks=(mask, None), any_hit=True
        ).prepare_graph_execute(bindings)
        per_map = [0, 1, 0, 1] if format == 1 else [0, 1, 2, 3]
        states = np.array(
            per_map * 2 + ([0] * 4 + [1] * 4 + [2] * 4 + [3] * 4 if indexed else [])
        )
        refit = front.record_refit().prepare_graph_execute(dict(vertices=vertices))
        update_ias = scene.record_refit().prepare_graph_execute({})
        for z in (1.0, 0.5):
            mesh[:, 2] = z
            vertices.from_numpy(mesh)
            refit()
            update_ias()
            for alpha in (0, 1):
                pixels.fill(alpha)
                texture.from_ndarray(pixels)
                accepted = (states == 1) | ((states >= 2) & bool(alpha))
                expected = np.where(accepted, 2 - z, 2)
                graph.run(bound)
                np.testing.assert_allclose(hits.to_numpy()[:, 0], expected)
                np.testing.assert_array_equal(
                    ids.to_numpy()[:, 2], np.where(accepted, 13, 29)
                )
                any_hit()
                np.testing.assert_array_equal(ids.to_numpy()[:, 3], 1)
                np.testing.assert_allclose(hits.to_numpy()[~accepted, 0], 2)
        memory = {item.name: item for item in front.memory_report().components}
        assert memory["opacity_micromap_array_and_indices"].requested_bytes > 0
        assert not memory["opacity_micromap_import_temporary"].resident
        # The IAS holds the native GAS, including OMM allocations, after the
        # original owner closes. New uses through that owner remain invalid.
        front.close()
        graph.run(bound)
        ti.sync()
        graph.close()
        scene.close()
        with pytest.raises(RuntimeError, match="closed"):
            any_hit()
    finally:
        if graph is not None:
            graph.close()
        if scene is not None:
            scene.close()
        if front is not None:
            front.close()
        if back is not None:
            back.close()
        provider.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_baked_omm_import_rejects_bad_metadata_without_retaining_gas():
    provider = _provider()
    try:
        vertices, indices = ti.ndarray(ti.f32, (3, 3)), ti.ndarray(ti.i32, (1, 3))
        vertices.from_numpy(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32))
        indices.from_numpy(np.array([[0, 1, 2]], np.int32))
        for entries, mapping, reason in (
            ([(0, 13, 2)], None, "level or format"),
            ([(0, 0, 3)], None, "level or format"),
            ([(1, 0, 2)], None, "exceeds"),
            ([(0, 0, 2)], [1], "triangle mapping"),
            ([(0, 0, 2)], [-5], "triangle mapping"),
        ):
            asset = ti.hardware.ray.OptixOpacityMicromap(b"\0", entries, mapping)
            with pytest.raises(RuntimeError, match=reason):
                provider.triangle_gas(vertices, indices, opacity_micromap=asset)
        api = provider._loaded.api
        feature = api.info.features
        api.info.features &= ~_optix._OPACITY_MICROMAP_IMPORT
        try:
            with pytest.raises(RuntimeError, match="does not support baked OMM"):
                provider.triangle_gas(vertices, indices, opacity_micromap=asset)
        finally:
            api.info.features = feature
        # Some bakers optimize every triangle to a uniform special index.
        empty = ti.hardware.ray.OptixOpacityMicromap(b"", b"", [-1])
        with provider.triangle_gas(vertices, indices, opacity_micromap=empty) as gas:
            with provider.instance_scene(
                (ti.hardware.ray.OptixRayInstance(gas),)
            ) as scene:
                rays, hits, ids = (
                    ti.ndarray(ti.f32, (1, 8)),
                    ti.ndarray(ti.f32, (1, 4)),
                    ti.ndarray(ti.u32, (1, 4)),
                )
                rays.from_numpy(np.array([[0.2, 0.2, 2, 0, 0, 0, -1, 10]], np.float32))
                scene.trace_typed(rays, hits, ids)
                np.testing.assert_array_equal(ids.to_numpy()[:, 3], 0)
    finally:
        provider.close()
