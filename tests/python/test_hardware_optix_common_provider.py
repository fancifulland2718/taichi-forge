"""Common sessions and legacy OptiX GPU owners share one vendor loader."""

from contextlib import ExitStack
import ctypes
import os

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.hardware import _bundled_runtime_provider as common
from tests import test_utils
from tests.python.test_hardware_optix_faces import _array, _geometry, _RAYS, _TRIANGLES


@pytest.mark.parametrize("retire_common_first", [False, True])
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_common_and_legacy_contexts_share_loader(retire_common_first):
    path = os.environ.get("TAICHI_FORGE_TEST_OPTIX_PROVIDER")
    if not path:
        pytest.skip("explicit common OptiX adapter required")
    library = ctypes.CDLL(path)
    query = library.taichi_forge_optix_runtime_provider_query
    query.argtypes = [ctypes.c_uint32, ctypes.c_size_t, ctypes.POINTER(common._ProviderApi)]
    query.restype = ctypes.c_int
    api = common._ProviderApi()
    assert query(common.PROVIDER_ABI_VERSION, ctypes.sizeof(api), ctypes.byref(api)) == 0
    facts = common._RuntimeInfo()
    facts.struct_size = ctypes.sizeof(facts)
    handle = ctypes.c_void_p()
    assert api.create_runtime(None, ctypes.byref(handle), ctypes.byref(facts)) == 0

    def release_common():
        if handle.value:
            assert api.destroy_runtime(handle) == 0
            handle.value = None

    with ExitStack() as stack:
        stack.callback(release_common)
        provider = stack.enter_context(ti.hardware.ray.load_optix_provider(provider_path=path))
        vertices, indices = _geometry(_TRIANGLES)
        scene = stack.enter_context(provider.triangle_scene(vertices, indices))
        rays = _array(_RAYS[:1], ti.f32)
        hits, ids = ti.ndarray(ti.f32, (1, 4)), ti.ndarray(ti.i32, (1, 4))
        if retire_common_first:
            release_common()
        recording = scene.record_typed(1, face_rules=("front_only",))
        recording.execute({"rays": rays, "hits": hits, "hit_indices": ids})
        ti.sync()
        assert hits.to_numpy()[0, 0] == pytest.approx(2.0)
        np.testing.assert_array_equal(ids.to_numpy()[0], [1, 0, 0, 1])
        assert api.probe_runtime(b"missing-optix-runtime", ctypes.byref(facts)) == 8
        scene.close()
        provider.close()
        if not retire_common_first:
            assert api.probe_runtime(None, ctypes.byref(facts)) == 0
            assert api.probe_runtime(b"missing-optix-runtime", ctypes.byref(facts)) == 8
            release_common()
        assert api.probe_runtime(b"missing-optix-runtime", ctypes.byref(facts)) == 3
