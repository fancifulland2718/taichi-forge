"""Optional programmable ABI: native dispatch, record data, and owner leases."""

import ctypes as c
import os
from pathlib import Path
import struct

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.hardware import _optix as optix
from taichi_forge.hardware import _optix_program_abi as a
from taichi_forge.hardware._shader_artifact import PtxModule
from taichi_forge.lang._storage_view import describe_storage
from tests import test_utils


@pytest.mark.parametrize("raygen", ["render", "query"])
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_native_optix_program_sbt_and_launch_lifetime(raygen):
    path = os.environ.get("TAICHI_FORGE_TEST_OPTIX_PROVIDER")
    try:
        provider = optix.OptixProvider(provider_path=path)
    except RuntimeError as error:
        if path:
            raise
        pytest.skip(f"OptiX provider unavailable: {error}")
    if not int(provider._loaded.api.info.features) & (1 << 14):
        provider.close()
        pytest.skip("installed OptiX adapter predates programmable pipelines")
    api = a.load_program_api(provider)
    vertices = ti.ndarray(ti.f32, (3, 3))
    indices = ti.ndarray(ti.i32, (1, 3))
    vertices.from_numpy(np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], np.float32))
    indices.from_numpy(np.array([[0, 1, 2]], np.int32))
    gas = provider.triangle_gas(vertices, indices)
    scene = provider.instance_scene([optix.OptixRayInstance(gas, custom_index=17)])
    binding = a.Scene(scene._scene, 1)
    info = a.SceneInfo(c.sizeof(a.SceneInfo))
    optix._invoke_checked(provider._loaded.api, api.scene_info, provider._context, c.byref(binding), c.byref(info))
    code = PtxModule.from_file(Path(__file__).parent / "assets/hardware_optix_program.ptx").code
    modules = (a.Module * 1)(a.Module(code, len(code)))
    groups = (a.Group * 3)(
        a.Group(0, a.Entry(0, f"__raygen__{raygen}".encode()), a.Entry()),
        a.Group(1, a.Entry(0, b"__miss__value"), a.Entry()),
        a.Group(2, a.Entry(0, b"__closesthit__value"), a.Entry(0, b"__anyhit__mask")),
    )
    desc = a.ProgramDesc(c.sizeof(a.ProgramDesc), 1, modules, 3, groups, b"params", 24, 1, 2, 1, 0)
    program, launch = c.c_void_p(), c.c_void_p()
    try:
        # A failing compiler/group operation must not publish a partial owner.
        groups[0].entry.name = b"__raygen__missing"
        assert int(api.create(provider._context, c.byref(desc), c.byref(program))) != 0
        assert not program.value
        groups[0].entry.name = f"__raygen__{raygen}".encode()
        optix._invoke_checked(provider._loaded.api, api.create, provider._context, c.byref(desc), c.byref(program))
        output = ti.ndarray(ti.u32, 8)
        storage = provider._runtime_prog._prepare_external_cuda_storage(
            (describe_storage(output).descriptor,), (True,)
        )
        params = c.create_string_buffer(struct.pack("<QQII", info.traversable, storage.pointers[0], 5, 8))
        miss_payload = c.create_string_buffer(struct.pack("<I", 7))
        hit0 = c.create_string_buffer(struct.pack("<II", 100, 1))
        hit1 = c.create_string_buffer(struct.pack("<II", 200, 0))
        miss = (a.Record * 2)(a.Record(1, 4, c.addressof(miss_payload)), a.Record(1, 4, c.addressof(miss_payload)))
        hit = (a.Record * 2)(a.Record(2, 8, c.addressof(hit0)), a.Record(2, 8, c.addressof(hit1)))
        scenes = (a.Scene * 1)(binding)
        launch_desc = a.LaunchDesc(
            c.sizeof(a.LaunchDesc), 8, 1, 1, c.addressof(params), 24, a.Record(0, 0, None), miss, 2, hit, 2, scenes, 1
        )
        optix._invoke_checked(provider._loaded.api, api.prepare, program, c.byref(launch_desc), c.byref(launch))
        assert int(api.launch(launch, 0)) == 8  # requires explicit initialization
        assert int(api.destroy(program)) == 8  # live launch lease
        assert int(provider._loaded.api.destroy_instance_scene(scene._scene)) == 8
        invoke = provider._runtime_prog._invoke_external_cuda_prepared
        invoke(storage, lambda: optix._invoke_checked(provider._loaded.api, api.initialize, launch, 0))
        # Host payload storage may change after preparation; native snapshots do not.
        hit0.raw = bytes(len(hit0))
        for _ in range(3):
            invoke(storage, lambda: optix._invoke_checked(provider._loaded.api, api.launch, launch, 0))
        expected = [122, 12, 12, 12, 122, 12, 12, 12] if raygen == "render" else [5] * 8
        np.testing.assert_array_equal(output.to_numpy(), expected)
        memory = a.Memory(c.sizeof(a.Memory))
        optix._invoke_checked(provider._loaded.api, api.memory, launch, c.byref(memory))
        assert memory.device_data_bytes == memory.pinned_host_bytes == 256
        assert memory.parameter_bytes == 24 and memory.hit_stride == memory.miss_stride == 48
    finally:
        if launch.value:
            optix._invoke_checked(provider._loaded.api, api.destroy_launch, launch)
        if program.value:
            optix._invoke_checked(provider._loaded.api, api.destroy, program)
        scene.close()
        gas.close()
        provider.close()
