import ctypes
import os

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
    if not int(provider.identity["feature_bits"]) & _optix._ALPHA_MASK:
        provider.close()
        pytest.skip("OptiX alpha-mask adapter unavailable")
    return provider


@pytest.mark.parametrize("instanced", [False, True])
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_optix_alpha_device_graph_and_retained_bindings(instanced, monkeypatch):
    count = 33
    vertices = ti.ndarray(ti.f32, (3 if instanced else 6, 3))
    front = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1]], np.float32)
    vertices.from_numpy(
        front if instanced else np.concatenate((front, front - [0, 0, 1]))
    )
    indices = ti.ndarray(ti.i32, (1 if instanced else 2, 3))
    indices.from_numpy(
        np.array([[0, 1, 2]] if instanced else [[0, 1, 2], [3, 4, 5]], np.int32)
    )
    uvs = ti.ndarray(ti.f32, (vertices.shape[0], 2))
    front_uv = np.array([[0, 0], [1, 0], [0, 1]], np.float32)
    uvs.from_numpy(
        front_uv if instanced else np.concatenate((front_uv, np.full((3, 2), 0.75)))
    )
    pixels = ti.Vector.ndarray(4, ti.f32, (4, 8))
    image = np.ones((4, 8, 4), np.float32)
    image[0, :4, 3] = 0
    pixels.from_numpy(image)
    texture = ti.Texture(
        ti.Format.rgba32f,
        (4, 8),
        sampler=ti.hardware.sampling.SamplerConfig(
            min_filter="nearest", mag_filter="nearest"
        ),
    )
    texture.from_ndarray(pixels)
    rays = ti.ndarray(ti.f32, (count, 8))
    ray_data = np.tile(np.array([0.2, 0.3, 2, 0, 0, 0, -1, 10], np.float32), (count, 1))
    ray_data[1::2, 0] = -1
    rays.from_numpy(ray_data)
    hits, ids = ti.ndarray(ti.f32, (count, 4)), ti.ndarray(ti.u32, (count, 4))
    result = ti.ndarray(ti.f32, count)

    @ti.kernel
    def update_uv(uv: ti.types.ndarray(ti.f32, ndim=2), accept: ti.i32):
        for i in range(3):
            uv[i, 0] = 0.75 if accept else ti.cast(i == 1, ti.f32)
            uv[i, 1] = 0.75 if accept else ti.cast(i == 2, ti.f32)

    @ti.kernel
    def consume(
        h: ti.types.ndarray(ti.f32, ndim=2),
        index: ti.types.ndarray(ti.u32, ndim=2),
        out: ti.types.ndarray(ti.f32, ndim=1),
    ):
        for i in out:
            out[i] = h[i, 0] if index[i, 3] else -1

    provider = _provider()
    gas = scene = graph = None
    try:
        if instanced:
            gas = provider.triangle_gas(vertices, indices)
            scene = provider.instance_scene(
                (
                    ti.hardware.ray.OptixRayInstance(gas, custom_index=17),
                    ti.hardware.ray.OptixRayInstance(
                        gas,
                        custom_index=0xFFFFFE,
                        transform=(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, -1),
                    ),
                )
            )
        else:
            scene = provider.triangle_scene(vertices, indices)
        mask = ti.hardware.ray.OptixAlphaMask("uvs", "alpha")
        masks = (mask, None) if instanced else (mask,)
        record = scene.record_typed(count, alpha_masks=masks)
        assert not provider._typed_prepared
        assert record.memory_report().components[-1].requested_bytes == 48 + 32 * len(
            masks
        )
        builder = ti.graph.GraphBuilder()
        builder.dispatch(
            update_uv,
            ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "uvs", ti.f32, ndim=2),
            ti.graph.Arg(ti.graph.ArgKind.SCALAR, "accept", ti.i32),
        )
        builder.append_native(record, admission="explicit")
        builder.dispatch(
            consume,
            ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "hits", ti.f32, ndim=2),
            ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "hit_indices", ti.u32, ndim=2),
            ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "result", ti.f32, ndim=1),
        )
        graph = builder.compile()
        bindings = {
            "rays": rays,
            "hits": hits,
            "hit_indices": ids,
            "uvs": uvs,
            "alpha": texture,
            "accept": 0,
            "result": result,
        }
        bound = graph.bind(bindings)
        assert bound.fast_path_qualified
        for accept in (0, 1, 0):
            bound.update(accept=accept)

            def no_prepare(*args, **kwargs):
                raise AssertionError("alpha replay rebuilt its fixed binding packet")

            with monkeypatch.context() as patched:
                patched.setattr(_optix, "_ray_storage", no_prepare)
                patched.setattr(
                    _optix.OptixRayQueryRecording, "_mask_bindings", no_prepare
                )
                graph.submit(bound).wait()
            np.testing.assert_allclose(result.to_numpy()[::2], 1 if accept else 2)
            np.testing.assert_allclose(result.to_numpy()[1::2], -1)
            actual = ids.to_numpy()[::2]
            if instanced:
                np.testing.assert_array_equal(actual[:, 1], 0 if accept else 1)
                np.testing.assert_array_equal(actual[:, 2], 17 if accept else 0xFFFFFE)
            else:
                np.testing.assert_array_equal(actual[:, 0], 0 if accept else 1)
            np.testing.assert_allclose(
                hits.to_numpy()[::2, 1:3], np.tile([0.2, 0.3], (17, 1)), atol=2e-6
            )
        # Same CUDA Texture contents change without replacing the material table.
        pixels.fill(1)
        texture.from_ndarray(pixels)
        graph.submit(bound).wait()
        np.testing.assert_allclose(result.to_numpy()[::2], 1)
        replacement = ti.Texture(ti.Format.rgba32f, (4, 8))
        pixels.fill(0)
        replacement.from_ndarray(pixels)
        bound.update(alpha=replacement)
        graph.submit(bound).wait()
        np.testing.assert_allclose(result.to_numpy()[::2], 2 if instanced else -1)
        graph.close()
        graph = None

        # Every instance masked: no accepted hit; any-hit never stops at a
        # rejected front layer. All-opaque entries retain typed hit semantics.
        all_masks = (mask,) * len(masks)
        any_record = scene.record_typed(count, alpha_masks=all_masks, any_hit=True)
        args = {
            "rays": rays,
            "hits": hits,
            "hit_indices": ids,
            "uvs": uvs,
            "alpha": replacement,
        }
        prepared = any_record.prepare_graph_execute(args)
        prepared()
        ti.sync()
        assert not ids.to_numpy()[:, 3].any()
        pixels.fill(1)
        replacement.from_ndarray(pixels)
        prepared()
        ti.sync()
        assert ids.to_numpy()[::2, 3].all()
        assert not ids.to_numpy()[1::2, 3].any()
        with pytest.raises(ValueError, match="instance count"):
            scene.record_typed(count, alpha_masks=())
        with pytest.raises(RuntimeError, match="UV count"):
            any_record.prepare_graph_execute(
                {**args, "uvs": ti.ndarray(ti.f32, (1, 2))}
            )
        with pytest.raises(RuntimeError, match="2D"):
            any_record.prepare_graph_execute(
                {**args, "alpha": ti.Texture(ti.Format.rgba32f, (2, 2, 2))}
            )
        # Existing prefix cannot accidentally advertise or invoke the suffix.
        api = provider._loaded.api
        old_size = api.struct_size
        api.struct_size = _optix._ProviderApi.prepare_alpha.offset
        provider._alpha_prepared = False
        with pytest.raises(RuntimeError, match="does not support alpha"):
            scene.record_typed(count, alpha_masks=all_masks)
        api.struct_size = old_size
        provider._alpha_prepared = True
        # Explicit resource retirement is rejected before the provider sees it.
        replacement._delete_runtime_texture()
        with pytest.raises(RuntimeError, match="retired Texture"):
            prepared()
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


def test_optix_alpha_mask_metadata_and_suffix():
    mask = ti.hardware.ray.OptixAlphaMask("uvs", "alpha", channel=0)
    assert mask.cutoff == 0.5
    for cutoff in (float("nan"), float("inf"), -1, 2):
        with pytest.raises(ValueError, match="cutoff"):
            ti.hardware.ray.OptixAlphaMask("uvs", "alpha", cutoff=cutoff)
    with pytest.raises(ValueError, match="differ"):
        ti.hardware.ray.OptixAlphaMask("same", "same")
    with pytest.raises(ValueError, match="channel"):
        ti.hardware.ray.OptixAlphaMask("uvs", "alpha", channel=4)
    assert ctypes.sizeof(_optix._AlphaMask) == 32
    assert _optix._AlphaTraceDesc.masks.offset == ctypes.sizeof(_optix._TypedTraceDesc)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_optix_instance_opacity_preserves_mask_queries_and_device_refit():
    provider = _provider()
    gas = scene = None
    try:
        if not int(provider.identity["feature_bits"]) & _optix._INSTANCE_OPACITY:
            pytest.skip("OptiX instance opacity adapter unavailable")
        vertices = ti.ndarray(ti.f32, (3, 3))
        vertices.from_numpy(np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1]], np.float32))
        indices = ti.ndarray(ti.i32, (1, 3))
        indices.from_numpy(np.array([[0, 1, 2]], np.int32))
        gas = provider.triangle_gas(vertices, indices)
        front = ti.hardware.ray.OptixRayInstance(gas, custom_index=13)
        back = ti.hardware.ray.OptixRayInstance(
            gas,
            transform=(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, -1),
            custom_index=29,
            opaque=True,
        )
        with pytest.raises(TypeError, match="opaque"):
            ti.hardware.ray.OptixRayInstance(gas, opaque=1)
        api = provider._loaded.api
        features = api.info.features
        api.info.features &= ~_optix._INSTANCE_OPACITY
        try:
            with pytest.raises(RuntimeError, match="instance opacity"):
                provider.instance_scene((front, back))
        finally:
            api.info.features = features
        scene = provider.instance_scene((front, back))
        count = 32
        rays = ti.ndarray(ti.f32, (count, 8))
        rays.from_numpy(
            np.tile(np.array([0.2, 0.2, 2, 0, 0, 0, -1, 10], np.float32), (count, 1))
        )
        hits, ids = ti.ndarray(ti.f32, (count, 4)), ti.ndarray(ti.u32, (count, 4))
        bindings = dict(rays=rays, hits=hits, hit_indices=ids)
        opaque = scene.record_typed(count, alpha_masks=(None, None))
        assert opaque.alpha_masks is None
        assert not provider._alpha_prepared
        opaque.prepare_graph_execute(bindings)()
        np.testing.assert_array_equal(ids.to_numpy()[:, 2], 13)
        uvs = ti.ndarray(ti.f32, (3, 2))
        uvs.fill(0.25)
        pixels = ti.ndarray(ti.f32, (2, 2))
        pixels.fill(0)
        texture = ti.Texture(ti.Format.r32f, (2, 2))
        texture.from_ndarray(pixels)
        mask = ti.hardware.ray.OptixAlphaMask("uvs", "alpha", channel=0)
        with pytest.raises(ValueError, match="opaque instances"):
            scene.record_typed(count, alpha_masks=(mask, mask))
        bindings.update(uvs=uvs, alpha=texture)
        nearest = scene.record_typed(
            count, alpha_masks=(mask, None)
        ).prepare_graph_execute(bindings)
        any_hit = scene.record_typed(
            count, alpha_masks=(mask, None), any_hit=True
        ).prepare_graph_execute(bindings)
        transforms = ti.ndarray(ti.f32, (2, 12))

        @ti.kernel
        def move(t: ti.types.ndarray(ti.f32, ndim=2), offset: ti.f32):
            for i in range(2):
                for j in ti.static(range(12)):
                    t[i, j] = 1.0 if j == 0 or j == 5 or j == 10 else 0.0
                t[i, 11] = 0.0 if i == 0 else offset

        refit = scene.record_refit_transforms().prepare_graph_execute(
            dict(transforms=transforms)
        )
        for offset in (-1.0, -2.0):
            move(transforms, offset)
            refit()
            for execute in (nearest, any_hit):
                execute()
                np.testing.assert_array_equal(
                    ids.to_numpy(), np.tile([0, 1, 29, 1], (count, 1))
                )
                np.testing.assert_allclose(hits.to_numpy()[:, 0], 1.0 - offset)
        pixels.fill(1)
        texture.from_ndarray(pixels)
        nearest()
        np.testing.assert_array_equal(ids.to_numpy()[:, 2], 13)
        scene.close()
        with pytest.raises(RuntimeError, match="closed"):
            nearest()
    finally:
        if scene is not None:
            scene.close()
        if gas is not None:
            gas.close()
        provider.close()
