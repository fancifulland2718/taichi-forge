"""Fixed-batch OptiX face acceptance, independent of GeoPhys or assets."""

from contextlib import ExitStack
import os

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.hardware import _optix
from tests import test_utils

_FEATURES = ("face_filter_typed", "face_filter_occlusion", "face_filter_per_primitive", "face_filter_alpha")
_TRIANGLES = np.array(
    [
        [[-1, -1, 1], [1, -1, 1], [0, 1, 1]],
        [[-1, -1, 2], [0, 1, 2], [1, -1, 2]],
    ],
    np.float32,
)
_RAYS = np.array(
    [
        [0, 0, 0, 0.01, 0, 0, 1, 3],
        [0, 0, 0, 0.01, 0, 0, 1, 1.5],
        [0, 0, 3, 0.01, 0, 0, -1, 4],
        [0, 0, 0, 1.5, 0, 0, 1, 3],
        [4, 0, 0, 0.01, 0, 0, 1, 3],
        [0, 0, 0, 2.1, 0, 0, 1, 3],
        [0, 0, 0, 0.01, 0, 0, 2, 1.5],
    ],
    np.float32,
)


def _provider():
    path = os.environ.get("TAICHI_FORGE_TEST_OPTIX_PROVIDER")
    if not path:
        probe = _optix.probe_provider()
        if (
            probe["discovery"] != "present"
            or not probe["native_facts"].get("feature_bits", 0) & _optix._FACE_FILTER_TYPED
        ):
            pytest.skip("OptiX face-filter adapter unavailable")
    return ti.hardware.ray.load_optix_provider(provider_path=path, required_features=_FEATURES)


def _array(values, dtype):
    values = np.asarray(values, np.float32 if dtype == ti.f32 else np.uint32 if dtype == ti.u32 else np.int32)
    result = ti.ndarray(dtype, values.shape)
    result.from_numpy(values)
    return result


def _geometry(triangles):
    return _array(triangles.reshape(-1, 3), ti.f32), _array(np.arange(len(triangles) * 3).reshape(-1, 3), ti.i32)


def _reference(rays, instances):
    """Scalar geometric oracle over transformed vertices, not OptiX hit flags."""
    hits = np.zeros((len(rays), 4), np.float32)
    hits[:, 0] = -1
    ids = np.full((len(rays), 4), -1, np.int32)
    ids[:, 3] = 0
    for row, ray in enumerate(rays.astype(np.float64)):
        origin, direction = ray[:3], ray[4:7]
        nearest = ray[7] + 1
        for ordinal, (triangles, rules, transform, custom) in enumerate(instances):
            triangles = triangles.astype(np.float64) @ transform[:, :3].T + transform[:, 3]
            for primitive, (triangle, front_only) in enumerate(zip(triangles, rules)):
                a, b, c = triangle
                e1, e2 = b - a, c - a
                if front_only and np.dot(direction, np.cross(e1, e2)) >= 0:
                    continue
                p = np.cross(direction, e2)
                determinant = np.dot(e1, p)
                if abs(determinant) < 1e-12:
                    continue
                offset = origin - a
                u = np.dot(offset, p) / determinant
                q = np.cross(offset, e1)
                v = np.dot(direction, q) / determinant
                t = np.dot(e2, q) / determinant
                if u >= 0 and v >= 0 and u + v <= 1 and ray[3] <= t <= ray[7] and t < nearest:
                    nearest = t
                    hits[row] = [t, u, v, 0]
                    ids[row] = [primitive, ordinal, custom, 1]
    return hits, ids


def _check(hits, ids, occluded, expected):
    np.testing.assert_allclose(hits.to_numpy(), expected[0], rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(ids.to_numpy().astype(np.int32), expected[1])
    np.testing.assert_array_equal(occluded.to_numpy(), expected[1][:, 3])


@pytest.mark.parametrize("instanced", [False, True])
@pytest.mark.parametrize("triangle_count", [1, 2])
@pytest.mark.parametrize("policy", ["front_only", "mixed", "two_sided"])
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_face_candidates_continue_and_clear_misses(instanced, triangle_count, policy, monkeypatch):
    triangles = _TRIANGLES[:triangle_count]
    with ExitStack() as stack:
        provider = stack.enter_context(_provider())
        vertices, indices = _geometry(triangles)
        custom = 37 if instanced else 0
        if instanced:
            gas = stack.enter_context(provider.triangle_gas(vertices, indices))
            scene = stack.enter_context(
                provider.instance_scene(
                    (ti.hardware.ray.OptixRayInstance(gas, custom_index=custom, opaque=True, sbt_record_offset=3),)
                )
            )
        else:
            scene = stack.enter_context(provider.triangle_scene(vertices, indices))
        rules = np.ones(triangle_count, np.int32)
        bindings = {}
        if policy == "mixed":
            rules[0] = 0
            face_rules = (ti.hardware.ray.OptixFaceRuleTable("faces"),)
            bindings["faces"] = _array(rules, ti.u32 if instanced else ti.i32)
        else:
            rules[:] = int(policy == "front_only")
            face_rules = (policy,)
        rays = _array(_RAYS, ti.f32)
        hits, ids = ti.ndarray(ti.f32, (len(_RAYS), 4)), ti.ndarray(ti.i32, (len(_RAYS), 4))
        flags = ti.ndarray(ti.i32, len(_RAYS))
        typed = scene.record_typed(len(_RAYS), face_rules=face_rules)
        compact = scene.record_occlusion(len(_RAYS), face_rules=face_rules)
        if policy == "two_sided":
            assert typed.face_rules is None and compact.face_rules is None
        main = typed.prepare_graph_execute(dict(rays=rays, hits=hits, hit_indices=ids, **bindings))
        shadow = compact.prepare_graph_execute(dict(rays=rays, occluded=flags, **bindings))
        identity = np.eye(4)[:3]
        expected = _reference(_RAYS, [(triangles, rules, identity, custom)])
        if policy == "front_only" and triangle_count == 2:
            assert expected[1][0, 0] == 1 and expected[0][0, 0] == 2
            assert expected[1][1, 3] == 0
        if policy == "mixed":
            assert expected[1][0, 0] == 0 and expected[0][0, 0] == 1

        def no_prepare(*args, **kwargs):
            raise AssertionError("prepared query re-entered cold validation")

        with monkeypatch.context() as patched:
            patched.setattr(_optix, "_ray_storage", no_prepare)
            main()
            shadow()
            ti.sync()
        _check(hits, ids, flags, expected)
        # Reuse every output buffer after a hit; misses must overwrite all lanes.
        misses = _RAYS.copy()
        misses[:, 0] = 4
        rays.from_numpy(misses)
        main()
        shadow()
        ti.sync()
        _check(hits, ids, flags, _reference(misses, [(triangles, rules, identity, custom)]))


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_shared_gas_faces_graph_refit_rebind_and_retirement(monkeypatch):
    with ExitStack() as stack:
        provider = stack.enter_context(_provider())
        vertices, indices = _geometry(_TRIANGLES)
        gas = stack.enter_context(provider.triangle_gas(vertices, indices))
        scene = stack.enter_context(
            provider.instance_scene(
                tuple(
                    ti.hardware.ray.OptixRayInstance(
                        gas,
                        custom_index=17 + 12 * i,
                        opaque=True,
                        sbt_record_offset=3 + i,
                        transform=((1, 0, 0, 4 * i), (0, 1, 0, 0), (0, 0, 1, 0)),
                    )
                    for i in range(2)
                )
            )
        )
        rays_np = np.tile(_RAYS[0], (2, 1))
        rays_np[1, 0] = 4
        rays = _array(rays_np, ti.f32)
        hits, ids, flags = ti.ndarray(ti.f32, (2, 4)), ti.ndarray(ti.i32, (2, 4)), ti.ndarray(ti.i32, 2)
        transforms = ti.ndarray(ti.f32, (2, 12))
        table = _array([0, 1], ti.i32)
        face_rules = ("front_only", ti.hardware.ray.OptixFaceRuleTable("faces"))
        main = scene.record_typed(2, face_rules=face_rules)
        shadow = scene.record_occlusion(2, face_rules=face_rules)
        builder = ti.graph.GraphBuilder()
        builder.append_native(scene.record_refit_transforms(), admission="explicit")
        builder.append_native(main, admission="explicit")
        builder.append_native(shadow, admission="explicit")
        graph = builder.compile()
        stack.callback(graph.close)
        bound = graph.bind(
            dict(transforms=transforms, rays=rays, hits=hits, hit_indices=ids, occluded=flags, faces=table)
        )
        assert bound.fast_path_qualified
        # IAS owns the native GAS and its primitive numbering after owner close.
        gas.close()
        variants = [
            np.eye(4)[:3],
            np.diag([-1, 1, 1, 1])[:3],
            np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 3]]),
            np.array([[2, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1.5, 0.2]]),
            np.array([[1, 0, 0, 2], [0, 1, 0, 0], [0, 0, 1, 0]]),
            np.eye(4)[:3],
        ]

        def no_cold_work(*args, **kwargs):
            raise AssertionError("Graph replay re-entered cold face/refit validation")

        with monkeypatch.context() as patched:
            patched.setattr(_optix, "_ray_storage", no_cold_work)
            patched.setattr(_optix, "_instance_transform_storage", no_cold_work)
            patched.setattr(_optix, "_query_provider", no_cold_work)
            for transform in variants:
                matrices = np.tile(transform, (2, 1, 1))
                matrices[1, 0, 3] += 4
                transforms.from_numpy(matrices.astype(np.float32).reshape(2, 12))
                graph.submit(bound).wait()
                expected = _reference(
                    rays_np,
                    [
                        (_TRIANGLES, [1, 1], matrices[0], 17),
                        (_TRIANGLES, [0, 1], matrices[1], 29),
                    ],
                )
                _check(hits, ids, flags, expected)
        # Rebinding replaces the immutable table at a cold boundary.
        replacement = _array([1, 1], ti.i32)
        bound.update(faces=replacement)
        ticket = graph.submit(bound)
        graph.close()
        ticket.wait()
        expected = _reference(
            rays_np,
            [
                (_TRIANGLES, [1, 1], matrices[0], 17),
                (_TRIANGLES, [1, 1], matrices[1], 29),
            ],
        )
        _check(hits, ids, flags, expected)
        # Resize by recording a new fixed batch after retiring the old Graph.
        small_rays = _array(_RAYS[:1], ti.f32)
        small_hits, small_ids, small_flags = (
            ti.ndarray(ti.f32, (1, 4)),
            ti.ndarray(ti.i32, (1, 4)),
            ti.ndarray(ti.i32, 1),
        )
        resized = ti.graph.GraphBuilder()
        resized.append_native(scene.record_typed(1, face_rules=face_rules), admission="explicit")
        resized.append_native(scene.record_occlusion(1, face_rules=face_rules), admission="explicit")
        small_graph = resized.compile()
        stack.callback(small_graph.close)
        small_bound = small_graph.bind(
            dict(rays=small_rays, hits=small_hits, hit_indices=small_ids, occluded=small_flags, faces=replacement)
        )
        small_graph.submit(small_bound).wait()
        np.testing.assert_array_equal(small_ids.to_numpy(), [[1, 0, 17, 1]])
        small_graph.close()
        scene.close()
        graph.close()
        with pytest.raises(RuntimeError, match="closed"):
            main.prepare_graph_execute(dict(rays=rays, hits=hits, hit_indices=ids, faces=replacement))


@pytest.mark.parametrize(
    "front_only,alpha_a,alpha_b,expected_primitive",
    [
        (True, 1, 1, 1),
        (True, 1, 0, -1),
        (False, 0, 1, 1),
        (False, 1, 0, 0),
        (False, 0, 0, -1),
    ],
)
@pytest.mark.parametrize("instanced", [False, True])
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_face_and_alpha_acceptance(front_only, alpha_a, alpha_b, expected_primitive, instanced):
    with ExitStack() as stack:
        provider = stack.enter_context(_provider())
        vertices, indices = _geometry(_TRIANGLES)
        count = 2 if instanced else 1
        if instanced:
            gas = stack.enter_context(provider.triangle_gas(vertices, indices))
            scene = stack.enter_context(
                provider.instance_scene(
                    tuple(
                        ti.hardware.ray.OptixRayInstance(
                            gas,
                            custom_index=17 + i,
                            sbt_record_offset=3 + i,
                            transform=((1, 0, 0, 4 * i), (0, 1, 0, 0), (0, 0, 1, 0)),
                        )
                        for i in range(count)
                    )
                )
            )
        else:
            scene = stack.enter_context(provider.triangle_scene(vertices, indices))
        uvs = _array([[0.25, 0.25]] * 3 + [[0.75, 0.75]] * 3, ti.f32)
        pixels = _array([[alpha_a, 0], [0, alpha_b]], ti.f32)
        texture = ti.Texture(
            ti.Format.r32f,
            (2, 2),
            sampler=ti.hardware.sampling.SamplerConfig(
                min_filter="nearest",
                mag_filter="nearest",
            ),
        )
        texture.from_ndarray(pixels)
        rays_np = np.tile(_RAYS[:1], (count, 1))
        rays_np[:, 0] = np.arange(count) * 4
        rays = _array(rays_np, ti.f32)
        hits, ids, flags = ti.ndarray(ti.f32, (count, 4)), ti.ndarray(ti.i32, (count, 4)), ti.ndarray(ti.i32, count)
        faces = _array([int(front_only), 1], ti.i32)
        options = dict(
            face_rules=(ti.hardware.ray.OptixFaceRuleTable("faces"),) * count,
            alpha_masks=(ti.hardware.ray.OptixAlphaMask("uvs", "texture", channel=0),) * count,
        )
        bindings = dict(rays=rays, faces=faces, uvs=uvs, texture=texture)
        scene.record_typed(count, **options).execute(dict(hits=hits, hit_indices=ids, **bindings))
        scene.record_occlusion(count, **options).execute(dict(occluded=flags, **bindings))
        ti.sync()
        np.testing.assert_array_equal(ids.to_numpy()[:, 0], np.full(count, expected_primitive))
        np.testing.assert_array_equal(flags.to_numpy(), np.full(count, int(expected_primitive >= 0)))
        if expected_primitive < 0:
            np.testing.assert_array_equal(hits.to_numpy(), np.tile([-1, 0, 0, 0], (count, 1)))
            np.testing.assert_array_equal(ids.to_numpy(), np.tile([-1, -1, -1, 0], (count, 1)))
        else:
            np.testing.assert_allclose(hits.to_numpy()[:, 0], np.full(count, expected_primitive + 1))
            np.testing.assert_array_equal(ids.to_numpy()[:, 1], np.arange(count))
            np.testing.assert_array_equal(ids.to_numpy()[:, 2], np.arange(count) + (17 if instanced else 0))


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_face_contract_rejects_unsupported_rules_before_preparation(monkeypatch):
    with ExitStack() as stack:
        provider = stack.enter_context(_provider())
        vertices, indices = _geometry(_TRIANGLES)
        scene = stack.enter_context(provider.triangle_scene(vertices, indices))
        assert set(_FEATURES) <= provider.features
        with pytest.raises(TypeError, match="sequence"):
            scene.record_typed(1, face_rules="front_only")
        with pytest.raises(ValueError, match="instance count"):
            scene.record_occlusion(1, face_rules=())
        with pytest.raises(ValueError, match="face rules"):
            scene.record_typed(1, face_rules=("back_only",))
        with pytest.raises(ValueError, match="nonempty"):
            ti.hardware.ray.OptixFaceRuleTable("")
        with pytest.raises(ValueError, match="reuse"):
            scene.record_typed(1, face_rules=(ti.hardware.ray.OptixFaceRuleTable("hits"),))
        rays = _array(_RAYS[:1], ti.f32)
        hits, ids = ti.ndarray(ti.f32, (1, 4)), ti.ndarray(ti.i32, (1, 4))
        query = scene.record_typed(1, face_rules=(ti.hardware.ray.OptixFaceRuleTable("faces"),))
        with pytest.raises(RuntimeError, match="dtype"):
            query.prepare_graph_execute(dict(rays=rays, hits=hits, hit_indices=ids, faces=ti.ndarray(ti.f32, 2)))
        with pytest.raises(RuntimeError, match="GAS primitives"):
            query.prepare_graph_execute(dict(rays=rays, hits=hits, hit_indices=ids, faces=ti.ndarray(ti.i32, 1)))

        def no_prepare(*args, **kwargs):
            raise AssertionError("unsupported capability reached pipeline preparation")

        with monkeypatch.context() as patched:
            patched.setattr(provider, "_prepare_faces", no_prepare)
            info = provider._loaded.api.info
            patched.setattr(info, "features", int(info.features) & ~_optix._FACE_FILTER_OCCLUSION)
            with pytest.raises(RuntimeError, match="requested face filtering"):
                scene.record_occlusion(1, face_rules=("front_only",))
        with monkeypatch.context() as patched:
            patched.setattr(provider, "_prepare_faces", no_prepare)
            info = provider._loaded.api.info
            patched.setattr(info, "features", int(info.features) & ~_optix._FACE_FILTER_PER_PRIMITIVE)
            with pytest.raises(RuntimeError, match="requested face filtering"):
                scene.record_typed(1, face_rules=(ti.hardware.ray.OptixFaceRuleTable("faces"),))
        with monkeypatch.context() as patched:
            patched.setattr(provider, "_prepare_faces", no_prepare)
            info = provider._loaded.api.info
            patched.setattr(info, "features", int(info.features) & ~_optix._FACE_FILTER_ALPHA)
            with pytest.raises(RuntimeError, match="combined with alpha"):
                scene.record_typed(
                    1, face_rules=("front_only",), alpha_masks=(ti.hardware.ray.OptixAlphaMask("uvs", "alpha"),)
                )
        # any_hit means first accepted candidate; rejecting A must still reach B.
        scene.record_typed(1, face_rules=("front_only",), any_hit=True).execute(
            dict(rays=rays, hits=hits, hit_indices=ids)
        )
        ti.sync()
        assert ids.to_numpy()[0, 0] == 1 and hits.to_numpy()[0, 0] == 2
        # Opacity micromaps can bypass any-hit and are explicitly unsupported.
        asset = ti.hardware.ray.OptixOpacityMicromap(bytes([0xAA]), [(0, 1, 2)], triangle_indices=[0, 0])
        gas = stack.enter_context(provider.triangle_gas(vertices, indices, opacity_micromap=asset))
        omm_scene = stack.enter_context(provider.instance_scene((ti.hardware.ray.OptixRayInstance(gas),)))
        for entry in (omm_scene.record_typed, omm_scene.record_occlusion):
            with pytest.raises(ValueError, match="opacity micromaps"):
                entry(1, face_rules=("front_only",))


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_face_required_features_reject_legacy_before_context(monkeypatch):
    path = os.environ.get("TAICHI_FORGE_TEST_OPTIX_PROVIDER")
    if not path:
        pytest.skip("explicit adapter required for capability negotiation test")
    loaded = _optix._query_provider(path)
    info = loaded.api.info
    old_features = int(info.features) & ~(
        _optix._FACE_FILTER_TYPED
        | _optix._FACE_FILTER_OCCLUSION
        | _optix._FACE_FILTER_PER_PRIMITIVE
        | _optix._FACE_FILTER_ALPHA
    )
    contexts = []

    @_optix._CreateContext
    def no_context(desc, result):
        contexts.append(True)
        return 1

    with monkeypatch.context() as patched:
        patched.setattr(info, "features", old_features)
        patched.setattr(loaded.api, "create_context", no_context)
        patched.setattr(_optix, "_query_provider", lambda candidate: loaded)
        with pytest.raises(RuntimeError, match="missing required features"):
            ti.hardware.ray.load_optix_provider(provider_path=path, required_features=("face_filter_typed",))
        assert not contexts
    with monkeypatch.context() as patched:
        patched.setattr(loaded.api, "struct_size", _optix._ProviderApi.prepare_faces.offset)
        with pytest.raises(RuntimeError, match="face-filter feature table"):
            _optix._check_api(loaded.api)
