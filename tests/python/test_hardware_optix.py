import ctypes
from pathlib import Path

import pytest

import taichi_forge as ti
from taichi_forge.hardware import _optix
from tests import test_utils


class _FakeOptixLibrary:
    def __init__(self, *, optix_abi=93, query_result=0):
        self.calls = {
            "query": 0,
            "create_context": 0,
            "destroy_context": 0,
            "create_scene": 0,
            "update_scene": 0,
            "trace": 0,
            "destroy_scene": 0,
        }
        self._last_error = b"fake provider failure"

        @_optix._CreateContext
        def create_context(_desc, out_context):
            self.calls["create_context"] += 1
            out_context[0] = ctypes.c_void_p(0x1234)
            return 0

        @_optix._DestroyContext
        def destroy_context(_context):
            self.calls["destroy_context"] += 1
            return 0

        @_optix._CreateScene
        def create_scene(_context, _desc, out_scene):
            self.calls["create_scene"] += 1
            out_scene[0] = ctypes.c_void_p(0x5678)
            return 0

        @_optix._UpdateScene
        def update_scene(_scene, _desc):
            self.calls["update_scene"] += 1
            return 0

        @_optix._Trace
        def trace(_scene, _desc):
            self.calls["trace"] += 1
            return 0

        @_optix._GetSceneMemory
        def get_scene_memory(_scene, out_memory):
            memory = out_memory.contents
            memory.gas_bytes = 4096
            memory.ias_bytes = 2048
            memory.build_update_scratch_bytes = 1024
            memory.instance_bytes = 80
            memory.launch_params_bytes = 24
            memory.shared_pipeline_sbt_bytes = 192
            return 0

        @_optix._DestroyScene
        def destroy_scene(_scene):
            self.calls["destroy_scene"] += 1
            return 0

        @_optix._GetLastError
        def get_last_error(destination, destination_size):
            required = len(self._last_error) + 1
            if destination and destination_size:
                copied = min(len(self._last_error), int(destination_size) - 1)
                ctypes.memmove(destination, self._last_error, copied)
                destination[copied] = b"\0"
            return required

        self._callbacks = (
            create_context,
            destroy_context,
            create_scene,
            update_scene,
            trace,
            get_scene_memory,
            destroy_scene,
            get_last_error,
        )
        api = _optix._ProviderApi()
        api.struct_size = ctypes.sizeof(_optix._ProviderApi)
        api.provider_abi_version = _optix.PROVIDER_ABI_VERSION
        api.info.struct_size = ctypes.sizeof(_optix._ProviderInfo)
        api.info.provider_abi_version = _optix.PROVIDER_ABI_VERSION
        api.info.optix_abi_version = optix_abi
        api.info.optix_version = 80100 if optix_abi == 93 else 90000
        api.info.features = _optix._REQUIRED_FEATURES | (1 << 5)
        api.info.provider_name = b"fake-optix"
        api.info.build_identity = b"fake-provider-abi1"
        api.create_context = create_context
        api.destroy_context = destroy_context
        api.create_triangle_scene = create_scene
        api.update_triangle_scene = update_scene
        api.trace = trace
        api.get_scene_memory = get_scene_memory
        api.destroy_triangle_scene = destroy_scene
        api.get_last_error = get_last_error
        self._api = api

        query_type = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.c_uint32,
            ctypes.c_size_t,
            ctypes.POINTER(_optix._ProviderApi),
        )

        @query_type
        def query(requested_abi, api_size, out_api):
            self.calls["query"] += 1
            if query_result:
                return query_result
            if requested_abi != _optix.PROVIDER_ABI_VERSION:
                return 2
            ctypes.memmove(
                out_api,
                ctypes.byref(self._api),
                min(int(api_size), ctypes.sizeof(self._api)),
            )
            return 0

        self.taichi_forge_optix_provider_query = query
        self._query = query


def test_optix_probe_is_explicit_query_only_and_abi_versioned(monkeypatch):
    fake = _FakeOptixLibrary(optix_abi=93)
    monkeypatch.setattr(_optix, "_load_library", lambda _path: fake)

    result = _optix.probe_provider("fake-provider.dll")

    assert result["discovery"] == "present"
    assert result["unavailable_reason"] == "execution_not_qualified"
    assert result["provider_abi"] == _optix.PROVIDER_ABI_NAME
    assert result["provider_version"] == "8.1.0"
    assert result["native_facts"]["optix_abi_version"] == 93
    assert result["native_facts"]["query_only"] is True
    assert fake.calls["query"] == 1
    assert fake.calls["create_context"] == 0


def test_optix_probe_rejects_unqualified_sdk_abi_without_initializing(monkeypatch):
    fake = _FakeOptixLibrary(optix_abi=106)
    monkeypatch.setattr(_optix, "_load_library", lambda _path: fake)

    result = _optix.probe_provider("future-provider.dll")

    assert result["discovery"] == "incompatible"
    assert result["unavailable_reason"] == "provider_abi_query_failed"
    assert "outside Forge's qualified source range" in result["last_error"]
    assert result["failure_scope"] == "provider"
    assert fake.calls["create_context"] == 0


def test_optix_probe_handles_query_failure_before_error_callback_exists(monkeypatch):
    fake = _FakeOptixLibrary(query_result=2)
    monkeypatch.setattr(_optix, "_load_library", lambda _path: fake)

    result = _optix.probe_provider("abi-mismatch-provider.dll")

    assert result["discovery"] == "incompatible"
    assert result["unavailable_reason"] == "provider_abi_query_failed"
    assert "provider query failed with result 2" in result["last_error"]
    assert fake.calls["create_context"] == 0


def test_optix_passive_status_and_missing_probe_do_not_load_a_library(monkeypatch):
    def unexpected_load(_path):
        raise AssertionError("passive OptiX reporting must not load a plugin")

    monkeypatch.delenv("TAICHI_FORGE_OPTIX_PROVIDER", raising=False)
    monkeypatch.setattr(_optix, "_load_library", unexpected_load)

    assert _optix.passive_status()["library_loaded"] is False
    result = _optix.probe_provider()
    assert result["external_component_probed"] is False
    assert result["unavailable_reason"] == "provider_library_path_required"


def test_optional_optix_cmake_target_has_no_wheel_install_or_sdk_download():
    root = Path(__file__).resolve().parents[2]
    cmake = (root / "cmake" / "TaichiOptixProvider.cmake").read_text(encoding="utf-8")
    abi = (root / "taichi" / "optix" / "forge_optix_provider.h").read_text(
        encoding="utf-8"
    )

    assert '"Build the user-SDK OptiX provider outside the official wheel" OFF' in cmake
    assert "Forge does not download or package the SDK" in cmake
    assert "install(TARGETS taichi_forge_optix_provider" not in cmake
    assert "TI_FORGE_OPTIX_PROVIDER_ABI_VERSION 1u" in abi
    assert "#include <optix" not in abi.lower()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_fake_optix_provider_scene_graph_lifetime_and_memory(monkeypatch):
    fake = _FakeOptixLibrary(optix_abi=93)
    monkeypatch.setattr(_optix, "_load_library", lambda _path: fake)
    vertices = ti.ndarray(ti.f32, shape=(3, 3))
    indices = ti.ndarray(ti.i32, shape=(1, 3))
    rays = ti.ndarray(ti.f32, shape=(2, 8))
    hits = ti.ndarray(ti.f32, shape=(2, 4))

    provider = ti.hardware.ray.load_optix_provider("fake-provider.dll")
    scene = provider.triangle_scene(vertices, indices)
    recording = scene.record(2)
    builder = ti.graph.GraphBuilder()
    builder.append_native(recording, admission="explicit")
    graph = builder.compile()
    graph.run({"rays": rays, "hits": hits})
    scene.refit(vertices)

    report = scene.memory_report()
    assert report.known_resident_requested_bytes == 4096 + 2048 + 1024 + 80 + 24 + 192
    assert report.opaque_component_count == 1
    assert fake.calls["trace"] == 1
    assert fake.calls["update_scene"] == 1
    with pytest.raises(RuntimeError, match="while triangle scenes are live"):
        provider.close()
    scene.close()
    with pytest.raises(RuntimeError, match="OptixTriangleScene has been closed"):
        graph.run({"rays": rays, "hits": hits})
    provider.close()
    assert fake.calls["destroy_scene"] == 1
    assert fake.calls["destroy_context"] == 1
