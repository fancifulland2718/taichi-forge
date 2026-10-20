"""Python-owned providers must execute in Forge's context and restore callers."""

from contextlib import contextmanager, ExitStack
import ctypes as C
from pathlib import Path
import sys

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.hardware import _optix
from taichi_forge.hardware._external_cuda_submission import external_cuda_submission
from taichi_forge.lang import impl
from tests import test_utils
from tests.python.test_hardware_optix import _FakeOptixLibrary
from tests.python.test_hardware_optix_faces import _array, _geometry, _provider, _RAYS, _TRIANGLES


@contextmanager
def _ambient_context(mode):
    driver = C.WinDLL("nvcuda.dll") if sys.platform == "win32" else C.CDLL("libcuda.so.1")
    driver.cuCtxGetCurrent.argtypes = [C.POINTER(C.c_void_p)]
    driver.cuCtxSetCurrent.argtypes = [C.c_void_p]
    driver.cuCtxGetDevice.argtypes = [C.POINTER(C.c_int)]
    driver.cuCtxCreate_v2.argtypes = [C.POINTER(C.c_void_p), C.c_uint, C.c_int]
    driver.cuCtxDestroy_v2.argtypes = [C.c_void_p]

    def current():
        result = C.c_void_p()
        assert driver.cuCtxGetCurrent(C.byref(result)) == 0
        return result.value

    previous = current()
    foreign = C.c_void_p()
    try:
        scope = impl.get_runtime().prog._begin_external_cuda_submission()
        try:
            owned = current()
            assert owned is not None
            device = C.c_int()
            assert driver.cuCtxGetDevice(C.byref(device)) == 0
        finally:
            del scope
        assert current() == previous
        if mode == "foreign":
            assert driver.cuCtxCreate_v2(C.byref(foreign), 0, device.value) == 0
            assert foreign.value != owned
        assert driver.cuCtxSetCurrent(foreign) == 0
        yield current, foreign.value, owned
    finally:
        assert driver.cuCtxSetCurrent(previous) == 0
        if foreign.value:
            assert driver.cuCtxDestroy_v2(foreign) == 0


@pytest.mark.parametrize("mode", ["empty", "foreign"])
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_external_cuda_scopes_restore_context_on_success_and_failure(mode):
    program = impl.get_runtime().prog
    packet = program._prepare_external_cuda_storage((), ())
    with _ambient_context(mode) as (current, ambient, owned):
        def check():
            assert current() == owned
            with external_cuda_submission(program, ()) as nested:
                nested.invoke(lambda: None)
            assert current() == owned
            return 37

        with external_cuda_submission(program, ()) as submission:
            assert submission.invoke(check) == 37
        assert current() == ambient
        program._invoke_external_cuda_prepared(packet, check)
        assert current() == ambient

        def fail():
            assert current() == owned
            raise ValueError("injected adapter failure")

        with pytest.raises(ValueError, match="injected adapter failure"):
            with external_cuda_submission(program, ()) as submission:
                submission.invoke(fail)
        assert current() == ambient
        with pytest.raises(ValueError, match="injected adapter failure"):
            program._invoke_external_cuda_prepared(packet, fail)
        assert current() == ambient
        program._invoke_external_cuda_prepared(packet, check)
        assert current() == ambient


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_optix_context_failure_and_fallback_restore_context(monkeypatch):
    libraries = {
        "new.dll": _FakeOptixLibrary(optix_abi=118, context_result=_optix._OPTIX_UNAVAILABLE),
        "fallback.dll": _FakeOptixLibrary(optix_abi=105),
    }
    monkeypatch.setattr(_optix, "_bundled_provider_candidates", lambda: tuple(libraries))
    monkeypatch.setattr(_optix, "_load_library", lambda path: libraries[Path(path).name])
    with _ambient_context("foreign") as (current, ambient, _):
        provider = ti.hardware.ray.load_optix_provider()
        assert provider.identity["optix_abi_version"] == 105
        assert current() == ambient
        provider.close()
        assert current() == ambient
        libraries["new.dll"] = _FakeOptixLibrary(context_result=2)
        with pytest.raises(RuntimeError, match="fake OptiX runtime unavailable"):
            ti.hardware.ray.load_optix_provider()
        assert current() == ambient
        # A failed cold construction does not invalidate later provider use.
        libraries.pop("new.dll")
        with ti.hardware.ray.load_optix_provider():
            assert current() == ambient
        assert current() == ambient


@pytest.mark.parametrize("mode", ["empty", "foreign"])
@pytest.mark.parametrize("instanced", [False, True])
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_optix_cold_graph_and_retirement_restore_context(mode, instanced):
    with _ambient_context(mode) as (current, ambient, _), ExitStack() as stack:
        provider = stack.enter_context(_provider())
        assert current() == ambient
        vertices, indices = _geometry(_TRIANGLES)
        if instanced:
            gas = stack.enter_context(provider.triangle_gas(vertices, indices))
            scene = stack.enter_context(provider.instance_scene([ti.hardware.ray.OptixRayInstance(gas)]))
        else:
            scene = stack.enter_context(provider.triangle_scene(vertices, indices))
        assert current() == ambient
        rays = _array(_RAYS[:1], ti.f32)
        hits, ids = ti.ndarray(ti.f32, (1, 4)), ti.ndarray(ti.i32, (1, 4))
        occluded = ti.ndarray(ti.i32, 1)
        # Prepare both the ordinary typed pipeline and the face-filter variant.
        scene.record_typed(1).execute(dict(rays=rays, hits=hits, hit_indices=ids))
        assert current() == ambient
        assert hits.to_numpy()[0, 0] == pytest.approx(1)
        builder = ti.graph.GraphBuilder()
        builder.append_native(scene.record_typed(1, face_rules=("front_only",)), admission="explicit")
        builder.append_native(scene.record_occlusion(1, face_rules=("front_only",)), admission="explicit")
        graph = builder.compile()
        stack.callback(graph.close)
        args = dict(rays=rays, hits=hits, hit_indices=ids, occluded=occluded)
        ticket = graph.submit(args)
        assert current() == ambient
        ticket.wait()
        assert current() == ambient
        assert hits.to_numpy()[0, 0] == pytest.approx(2)
        np.testing.assert_array_equal(ids.to_numpy()[0], [1, 0, 0, 1])
        np.testing.assert_array_equal(occluded.to_numpy(), [1])
        rays.from_numpy(_RAYS[1:2])
        graph.run(args)
        ti.sync()
        assert hits.to_numpy()[0, 0] == -1
        np.testing.assert_array_equal(ids.to_numpy()[0], [-1, -1, -1, 0])
        np.testing.assert_array_equal(occluded.to_numpy(), [0])
        assert current() == ambient
        stack.close()
        assert current() == ambient
