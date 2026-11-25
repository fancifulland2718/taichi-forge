"""Backend selection is cold; native scene/Graph ownership remains unchanged."""

import os
from pathlib import Path

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.hardware import _optix
from tests import test_utils
from tests.python.test_hardware_optix import _FakeOptixLibrary


def _resolved(operation_id):
    return next(item for item in ti.hardware.report().operations if item.descriptor.operation_id == operation_id)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_optix_required_features_filter_before_context_creation(monkeypatch):
    old, new = _FakeOptixLibrary(optix_abi=118), _FakeOptixLibrary(optix_abi=105)
    new._api.info.features |= 1 << 14
    libraries = {"old.dll": old, "program.dll": new}
    monkeypatch.delenv("TAICHI_FORGE_OPTIX_LIBRARY", raising=False)
    monkeypatch.setattr(_optix, "_bundled_provider_candidates", lambda: tuple(libraries))
    monkeypatch.setattr(_optix, "_load_library", lambda path: libraries[Path(path).name])
    with ti.hardware.ray.load_optix_provider(required_features=("program",)) as provider:
        assert provider.identity["optix_abi_version"] == 105
        assert provider.required_features == ("program",)
        assert _resolved("ray.program.optix").selection == "eligible"
    assert old.calls["create_context"] == 0
    assert new.calls["create_context"] == new.calls["destroy_context"] == 1
    with ti.hardware.ray.OptixProvider() as legacy:
        assert legacy.identity["optix_abi_version"] == 118
        assert _resolved("ray.program.optix").discovery == "incompatible"
        assert _resolved("ray.query.batch.optix").selection == "eligible"
    assert old.calls["create_context"] == old.calls["destroy_context"] == 1
    monkeypatch.setattr(_optix, "_provider_candidates_for_load", lambda *_args: (("old.dll",), None, "explicit"))
    with pytest.raises(RuntimeError, match="missing required features"):
        ti.hardware.ray.OptixProvider(provider_path="old.dll", required_features=("program",))
    with pytest.raises(ValueError, match="unknown"):
        ti.hardware.ray.OptixProvider(required_features=("imaginary_feature",))
    assert old.calls["create_context"] == 1
    wrong_vertices = ti.ndarray(ti.i32, (3, 3))
    indices = ti.ndarray(ti.i32, (1, 3))
    with pytest.raises(RuntimeError, match="dtype"):
        ti.hardware.ray.triangle_scene(wrong_vertices, indices)
    assert old.calls["create_context"] == old.calls["destroy_context"] == 2


@test_utils.test(arch=[ti.vulkan, ti.cuda], offline_cache=False)
def test_native_triangle_factory_direct_graph_refit_and_owner_close(monkeypatch):
    ray = ti.hardware.ray
    if ti.cfg.arch == ti.vulkan:
        if not ray.is_available():
            pytest.skip("Vulkan ray query unavailable")

        def no_optix(*args, **kwargs):
            raise AssertionError("Vulkan factory must not load OptiX")

        monkeypatch.setattr(_optix, "OptixProvider", no_optix)
    else:
        path = os.environ.get("TAICHI_FORGE_TEST_OPTIX_PROVIDER")
        if path:
            monkeypatch.setattr(_optix, "_bundled_provider_candidates", lambda: (path,))
        elif not _optix._bundled_provider_candidates():
            pytest.skip("OptiX adapter unavailable")
    vertices, indices = ti.ndarray(ti.f32, (3, 3)), ti.ndarray(ti.i32, (1, 3))
    host = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    vertices.from_numpy(host)
    indices.from_numpy(np.array([[0, 1, 2]], np.int32))
    rays, hits, ids = ti.ndarray(ti.f32, (2, 8)), ti.ndarray(ti.f32, (2, 4)), ti.ndarray(ti.i32, (2, 4))
    rays.from_numpy(np.array([[0.2, 0.3, 1, 0, 0, 0, -1, 10], [4, 0.3, 1, 0, 0, 0, -1, 10]], np.float32))
    with ray.triangle_scene(vertices, indices) as scene:
        assert scene.backend == ("vulkan" if ti.cfg.arch == ti.vulkan else "cuda")
        assert scene.trace_typed.__self__ is scene.native_scene
        provider = None if scene.backend == "vulkan" else scene.native_scene.provider
        scene.trace_typed(rays, hits, ids)
        np.testing.assert_allclose(hits.to_numpy(), [[1, 0.2, 0.3, 0], [-1, 0, 0, 0]], atol=1e-6)
        np.testing.assert_array_equal(ids.to_numpy(), [[0, 0, 0, 1], [-1, -1, -1, 0]])
        builder = ti.graph.GraphBuilder()
        builder.append_native(scene.record_refit(), admission="auto")
        builder.append_native(scene.record_typed(2), admission="auto")
        graph = builder.compile()
        try:
            bound = graph.bind(dict(vertices=vertices, rays=rays, hits=hits, hit_indices=ids))
            for height in (0.25, 0.5):
                host[:, 2] = height
                vertices.from_numpy(host)
                graph.run(bound)
                np.testing.assert_allclose(hits.to_numpy()[0, :3], [1 - height, 0.2, 0.3], atol=1e-6)
        finally:
            graph.close()
    assert scene.closed and (provider is None or provider.closed)
    scene.close()
    with pytest.raises(RuntimeError, match="switch"):
        ray.triangle_scene(vertices, indices, backend="cuda" if ti.cfg.arch == ti.vulkan else "vulkan")
    if ti.cfg.arch == ti.cuda:
        with ray.OptixProvider() as borrowed:
            with ray.triangle_scene(vertices, indices, provider=borrowed) as second:
                second.trace_typed(rays, hits, ids)
            assert not borrowed.closed
        owned = ray.triangle_scene(vertices, indices)
        ti.reset()
        owned.close()
        assert owned.closed
