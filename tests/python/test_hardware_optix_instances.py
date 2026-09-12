import ctypes
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.hardware import _optix
from tests import test_utils


def _instance_capable_api():
    api = _optix._ProviderApi()
    api.struct_size = ctypes.sizeof(api)
    api.info.features = _optix._INSTANCE_FEATURES
    callbacks = []
    for name, callback_type in _optix._ProviderApi._fields_:
        if name in (
            "prepare_typed",
            "trace_typed",
            "create_triangle_gas",
            "update_triangle_gas",
            "get_triangle_gas_memory",
            "destroy_triangle_gas",
            "create_instance_scene",
            "update_instance_scene",
            "trace_instance_scene",
            "trace_instance_scene_typed",
            "get_instance_scene_memory",
            "destroy_instance_scene",
        ):
            callback = callback_type(lambda *_args: 0)
            setattr(api, name, callback)
            callbacks.append(callback)
    return api, callbacks


def test_optix_instance_api_is_an_optional_abi1_suffix():
    api, callbacks = _instance_capable_api()
    _optix._require_instance_api(api)

    api.struct_size = _optix._ProviderApi.create_triangle_gas.offset
    assert _optix._api_has(api, "trace_typed")
    assert not _optix._api_has(api, "create_triangle_gas")
    with pytest.raises(RuntimeError, match="truncated or incomplete"):
        _optix._require_instance_api(api)
    api.struct_size = ctypes.sizeof(api)
    api.info.features &= ~_optix._DEVICE_INSTANCE_TRANSFORM_UPDATE
    with pytest.raises(RuntimeError, match="does not support shared GAS"):
        _optix._require_instance_api(api)
    assert callbacks


def test_explicit_adapter_path_stays_distinct_from_vendor_runtime(monkeypatch):
    monkeypatch.setattr(
        _optix, "_bundled_provider_candidates", lambda: ("bundled-adapter.dll",)
    )
    candidates, runtime, source = _optix._provider_candidates_for_load(
        "vendor/nvoptix.dll", "local/provider.dll"
    )
    assert candidates == (str(Path("local/provider.dll").resolve()),)
    assert runtime == str(Path("vendor/nvoptix.dll").resolve())
    assert source == "explicit_adapter_path"


def test_optix_instance_metadata_rejects_nonfinite_and_singular_transforms():
    gas = object.__new__(_optix.OptixTriangleGAS)
    nested = ((1, 0, 0, 3), (0, 2, 0, 4), (0, 0, 1, 5))
    instance = _optix.OptixRayInstance(
        gas, transform=nested, mask=7, custom_index=0xFFFFFE
    )
    assert instance.transform == (1, 0, 0, 3, 0, 2, 0, 4, 0, 0, 1, 5)

    with pytest.raises(ValueError, match="finite"):
        _optix.OptixRayInstance(
            gas,
            transform=(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, float("nan"), 0),
        )
    with pytest.raises(ValueError, match="finite in f32"):
        _optix.OptixRayInstance(
            gas,
            transform=(1e100, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
        )
    with pytest.raises(ValueError, match="invertible"):
        _optix.OptixRayInstance(
            gas,
            transform=(1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0),
        )
    with pytest.raises(ValueError, match="16777215"):
        _optix.OptixRayInstance(gas, custom_index=0x1000000)
    with pytest.raises(ValueError, match=r"\[0, 255\]"):
        _optix.OptixRayInstance(gas, mask=256)




def test_optix_instance_ctypes_layout_keeps_fixed_width_metadata():
    assert _optix._InstanceDesc.gas.offset == 8
    assert _optix._InstanceDesc.transform.offset == 8 + ctypes.sizeof(ctypes.c_void_p)
    assert _optix._InstanceDesc.custom_index.offset == (
        _optix._InstanceDesc.transform.offset + 12 * ctypes.sizeof(ctypes.c_float)
    )
    assert _optix._InstanceUpdateDesc.transforms.offset == 8
    assert _optix._InstanceUpdateDesc.cuda_stream.offset == 16


def test_optix_close_keeps_live_handle_when_synchronization_fails(monkeypatch):
    class FailingRuntime:
        def synchronize(self):
            raise RuntimeError("injected synchronize failure")

    runtime = FailingRuntime()
    monkeypatch.setattr(_optix, "runtime_generation_matches", lambda _owner: True)

    provider = object.__new__(_optix.OptixProvider)
    provider._context = ctypes.c_void_p(11)
    provider._runtime_prog = runtime
    provider._scenes = set()
    provider._gases = set()
    with pytest.raises(RuntimeError, match="injected synchronize failure"):
        provider.close()
    assert not provider.closed

    owner = SimpleNamespace(_loaded=SimpleNamespace(api=object()))
    for resource_type, attribute in (
        (_optix.OptixTriangleGAS, "_gas"),
        (_optix.OptixInstanceScene, "_scene"),
        (_optix.OptixTriangleScene, "_scene"),
    ):
        resource = object.__new__(resource_type)
        setattr(resource, attribute, ctypes.c_void_p(17))
        resource._runtime_prog = runtime
        resource.provider = owner
        with pytest.raises(RuntimeError, match="injected synchronize failure"):
            resource.close()
        assert getattr(resource, attribute).value == 17


def test_optix_stale_generation_close_drops_views_without_native_calls(monkeypatch):
    class UnexpectedRuntime:
        def synchronize(self):
            raise AssertionError("stale generation attempted synchronization")

    class UnexpectedApi:
        def __getattr__(self, _name):
            raise AssertionError("stale generation attempted native teardown")

    runtime = UnexpectedRuntime()
    monkeypatch.setattr(_optix, "runtime_generation_matches", lambda _owner: False)

    provider = object.__new__(_optix.OptixProvider)
    provider._context = ctypes.c_void_p(21)
    provider._runtime_prog = runtime
    provider._scenes = set()
    provider._gases = set()
    provider.close()
    assert provider.closed

    owner = SimpleNamespace(_loaded=SimpleNamespace(api=UnexpectedApi()))
    for resource_type, attribute in (
        (_optix.OptixTriangleGAS, "_gas"),
        (_optix.OptixInstanceScene, "_scene"),
        (_optix.OptixTriangleScene, "_scene"),
    ):
        resource = object.__new__(resource_type)
        setattr(resource, attribute, ctypes.c_void_p(23))
        resource._runtime_prog = runtime
        resource.provider = owner
        resource.close()
        assert getattr(resource, attribute) is None


def _load_test_provider():
    path = os.environ.get("TAICHI_FORGE_TEST_OPTIX_PROVIDER")
    try:
        return ti.hardware.ray.load_optix_provider(provider_path=path)
    except RuntimeError as exc:
        pytest.skip(f"OptiX provider unavailable: {exc}")


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_optix_explicit_adapter_preserves_legacy_triangle_scene(monkeypatch):
    vertices = ti.ndarray(ti.f32, (3, 3))
    indices = ti.ndarray(ti.i32, (1, 3))
    rays = ti.ndarray(ti.f32, (2, 8))
    hits = ti.ndarray(ti.f32, (2, 4))
    vertices.from_numpy(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32))
    indices.from_numpy(np.array([[0, 1, 2]], np.int32))
    rays.from_numpy(
        np.array(
            [[0.2, 0.3, 1, 0, 0, 0, -1, 10], [8.2, 0.3, 1, 0, 0, 0, -1, 10]],
            np.float32,
        )
    )

    @ti.kernel
    def relocate(v: ti.types.ndarray(ti.f32, ndim=2), offset: ti.f32):
        for i in range(3):
            v[i, 0] = offset + ti.cast(i == 1, ti.f32)

    with _load_test_provider() as provider:
        with provider.triangle_scene(vertices, indices, allow_update=True) as scene:
            builder = ti.graph.GraphBuilder()
            builder.dispatch(
                relocate,
                ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "vertices", ti.f32, ndim=2),
                ti.graph.Arg(ti.graph.ArgKind.SCALAR, "offset", ti.f32),
            )
            builder.append_native(scene.record_refit(), admission="explicit")
            builder.append_native(scene.record(2), admission="explicit")
            graph = builder.compile()
            binding = graph.bind(
                {"vertices": vertices, "offset": 8.0, "rays": rays, "hits": hits}
            )
            before_bytes = scene.memory_report().known_resident_requested_bytes
            try:
                for offset in (8.0, 0.0, 8.0):
                    binding.update(offset=offset)

                    def forbidden(*_args, **_kwargs):
                        raise AssertionError(
                            "Fixed refit re-entered storage preparation"
                        )

                    with monkeypatch.context() as patched:
                        patched.setattr(_optix, "_ray_storage", forbidden)
                        ticket = graph.submit(binding)
                    ticket.wait()
                    actual = hits.to_numpy()
                    selected = int(offset != 0)
                    assert actual[selected].tolist() == [1.0, 0.0, 0.0, 1.0]
                    assert actual[1 - selected].tolist() == [-1.0, -1.0, -1.0, 0.0]
                assert (
                    scene.memory_report().known_resident_requested_bytes
                    == before_bytes
                )
            finally:
                graph.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_optix_shared_gas_instances_device_graph_and_close_order(monkeypatch):
    count = 5

    def geometry(z):
        vertices = ti.ndarray(ti.f32, (3, 3))
        indices = ti.ndarray(ti.i32, (1, 3))
        vertices.from_numpy(
            np.array([[0, 0, z], [2, 0, z], [0, 2, z]], np.float32)
        )
        indices.from_numpy(np.array([[0, 1, 2]], np.int32))
        return vertices, indices

    provider = _load_test_provider()
    if provider.identity["feature_bits"] & _optix._INSTANCE_FEATURES != (
        _optix._INSTANCE_FEATURES
    ):
        provider.close()
        pytest.skip("installed OptiX adapter predates shared GAS instances")
    vertices_a, indices_a = geometry(0)
    vertices_b, indices_b = geometry(0.25)
    gas_a = provider.triangle_gas(vertices_a, indices_a)
    gas_b = provider.triangle_gas(vertices_b, indices_b)
    topology = (gas_a, gas_a, gas_b, gas_a, gas_b)
    instances = tuple(
        ti.hardware.ray.OptixRayInstance(
            gas,
            transform=(1, 0, 0, 4 * i, 0, 1, 0, 0, 0, 0, 1, 0),
            mask=0 if i == 3 else 0xFF,
            custom_index=100 + i,
        )
        for i, gas in enumerate(topology)
    )
    scene = provider.instance_scene(instances)
    assert scene.instance_count == count
    assert gas_a.memory_report().known_resident_requested_bytes > 0
    scene_memory = scene.memory_report()
    assert scene_memory.known_resident_requested_bytes >= count * 80

    transforms = ti.ndarray(ti.f32, (count, 12))
    rays = ti.ndarray(ti.f32, (count, 8))
    hits = ti.ndarray(ti.f32, (count, 4))
    ids = ti.ndarray(ti.i32, (count, 4))
    selected = ti.ndarray(ti.i32, (count,))
    offset_value = ti.ndarray(ti.f32, (1,))
    ray_values = np.tile(
        np.array([0.5, 0.5, 2, 0.001, 0, 0, -1, 100], np.float32),
        (count, 1),
    )
    ray_values[:, 0] += 4 * np.arange(count)
    rays.from_numpy(ray_values)

    @ti.kernel
    def produce(
        values: ti.types.ndarray(ti.f32, ndim=2),
        offset: ti.types.ndarray(ti.f32, ndim=1),
    ):
        for i in range(values.shape[0]):
            for word in ti.static(range(12)):
                values[i, word] = 0
            values[i, 0] = 1
            values[i, 5] = 1
            values[i, 10] = 1
            values[i, 3] = 4 * i + offset[0]

    @ti.kernel
    def consume(
        hit_ids: ti.types.ndarray(ti.i32, ndim=2),
        output: ti.types.ndarray(ti.i32, ndim=1),
    ):
        for i in output:
            output[i] = hit_ids[i, 2]

    transform_recording = scene.record_refit_transforms()
    query_recording = scene.record_typed(count)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        produce,
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "transforms", ti.f32, ndim=2),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "offset", ti.f32, ndim=1),
    )
    builder.append_native(transform_recording, admission="explicit")
    builder.append_native(query_recording, admission="explicit")
    builder.dispatch(
        consume,
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "hit_indices", ti.i32, ndim=2),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "selected", ti.i32, ndim=1),
    )
    graph = builder.compile()
    binding = graph.bind(
        {
            "transforms": transforms,
            "offset": offset_value,
            "rays": rays,
            "hits": hits,
            "hit_indices": ids,
            "selected": selected,
        }
    )
    assert binding.fast_path_qualified, binding.statistics()

    # Native IAS references outlive the independent Python GAS owners.
    gas_a.close()
    gas_b.close()
    with pytest.raises(RuntimeError, match="OptixTriangleGAS has been closed"):
        gas_a.refit(vertices_a)
    with pytest.raises(RuntimeError, match="triangle scenes are live"):
        provider.close()

    def unexpected_cold_work(*_args, **_kwargs):
        raise AssertionError("OptiX storage validation or instance packing re-entered")

    expected_custom = np.arange(100, 100 + count, dtype=np.int32)
    expected_custom[3] = -1
    with monkeypatch.context() as patched:
        patched.setattr(_optix, "_instance_transform_storage", unexpected_cold_work)
        patched.setattr(
            ti.hardware.ray.OptixRayInstance, "__post_init__", unexpected_cold_work
        )
        for offset in (0.0, 2.0, 0.0):
            offset_value.fill(offset)
            graph.run(binding)
            expected = expected_custom if offset == 0 else np.full(count, -1)
            np.testing.assert_array_equal(selected.to_numpy(), expected)
            if offset == 0:
                visible = np.arange(count) != 3
                actual_ids = ids.to_numpy()
                np.testing.assert_array_equal(
                    actual_ids[visible, 0], np.zeros(visible.sum(), np.int32)
                )
                np.testing.assert_array_equal(
                    actual_ids[visible, 1], np.arange(count)[visible]
                )
                np.testing.assert_array_equal(
                    actual_ids[visible, 2], expected_custom[visible]
                )
                np.testing.assert_array_equal(actual_ids[visible, 3], 1)
                expected_t = np.array([2, 2, 1.75, 2, 1.75], np.float32)
                np.testing.assert_allclose(
                    hits.to_numpy()[visible, 0], expected_t[visible]
                )

    prepared = transform_recording.prepare_graph_execute({"transforms": transforms})
    scene.close()
    with pytest.raises(RuntimeError, match="OptixInstanceScene has been closed"):
        prepared()
    with pytest.raises(RuntimeError, match="OptixInstanceScene has been closed"):
        graph.run(binding)
    graph.close()
    provider.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_new_shim_loads_old_typed_provider_prefix():
    old_path = os.environ.get("TAICHI_FORGE_TEST_OLD_OPTIX_PROVIDER")
    if not old_path:
        pytest.skip("old ABI-1 provider path was not supplied")
    vertices = ti.ndarray(ti.f32, (3, 3))
    indices = ti.ndarray(ti.i32, (1, 3))
    rays = ti.ndarray(ti.f32, (1, 8))
    hits = ti.ndarray(ti.f32, (1, 4))
    ids = ti.ndarray(ti.i32, (1, 4))
    vertices.from_numpy(np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], np.float32))
    indices.from_numpy(np.array([[0, 1, 2]], np.int32))
    rays.from_numpy(np.array([[0.5, 0.5, 2, 0.001, 0, 0, -1, 100]], np.float32))

    with ti.hardware.ray.load_optix_provider(provider_path=old_path) as provider:
        assert provider.identity["feature_bits"] & _optix._INSTANCE_FEATURES == 0
        with pytest.raises(RuntimeError, match="does not support shared GAS"):
            provider.triangle_gas(vertices, indices)
        with provider.triangle_scene(vertices, indices) as scene:
            scene.trace_typed(rays, hits, ids)
            np.testing.assert_array_equal(ids.to_numpy(), [[0, 0, 0, 1]])
