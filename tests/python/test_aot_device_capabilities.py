import array
from contextlib import closing
import json
import struct
from zipfile import ZipFile

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_aot_device_capabilities_snapshot_and_retirement():
    runtime = ti.aot.borrow_runtime()
    capabilities = runtime.device_capabilities()
    assert all(isinstance(k, str) and isinstance(v, int) for k, v in capabilities.items())
    if ti.cfg.arch == ti.vulkan:
        assert capabilities["spirv_version"] >= 0x10000
        assert capabilities.get("spirv_has_ray_query", 0) in (0, 1)
        assert (
            bool(capabilities.get("vulkan_has_opacity_micromap", 0)) == ti.hardware.ray.is_opacity_micromap_available()
        )
        assert bool(capabilities.get("vulkan_has_ray_tracing_pipeline", 0)) == ti.hardware.ray.is_program_available()
    else:
        assert "spirv_has_ray_query" not in capabilities
        assert "vulkan_has_opacity_micromap" not in capabilities
        assert "vulkan_has_ray_tracing_pipeline" not in capabilities
    if ti.cfg.arch == ti.cuda:
        with pytest.raises(RuntimeError, match="storage images require Vulkan"):
            runtime.native_resource("image", "sampled", parameters=dict(shape=[2, 2], format="rgba32f", storage=True))
    capabilities["spirv_version"] = 0
    assert runtime.device_capabilities().get("spirv_version") != 0
    runtime.close()
    with pytest.raises(RuntimeError, match="closed"):
        runtime.device_capabilities()


@test_utils.test(arch=[ti.vulkan], offline_cache=False)
def test_aot_float_atomic_target_preserves_fallback_and_native_codegen(tmp_path):
    @ti.kernel
    def scatter(values: ti.types.ndarray(ti.f32, 1), sums: ti.types.ndarray(ti.f32, 1)):
        for i in range(256):
            ti.atomic_add(sums[i % 4], values[i])

    source = np.asarray([(i % 7 - 3) * 0.25 for i in range(256)], dtype=np.float32)
    expected = source.reshape(-1, 4).sum(axis=0)
    values, sums = ti.ndarray(ti.f32, 256), ti.ndarray(ti.f32, 4)
    values.from_numpy(source)
    sums.fill(0)
    scatter(values, sums)  # The JIT first compiles against physical device capabilities.
    np.testing.assert_array_equal(sums.to_numpy(), expected)

    capability = ti.DeviceCapability.spirv_has_atomic_float_add
    with ti.aot.borrow_runtime() as runtime:
        native = bool(runtime.device_capabilities().get(capability, 0))
        # Switch back to the baseline after a native export to exercise cache identity.
        profiles = (False, True, False) if native else (False,)
        for index, use_native in enumerate(profiles):
            module = ti.aot.Module(caps=[capability] if use_native else [])
            module.add_kernel(scatter)
            artifact = tmp_path / f"scatter-{index}.tcm"
            module.archive(str(artifact))
            with ZipFile(artifact) as archive:
                metadata = json.loads(archive.read("metadata.json"))
                required = {item["key"]: item["value"] for item in metadata["required_caps"]}
                assert bool(required.get(capability, 0)) == use_native
                opcodes = []
                for name in archive.namelist():
                    if not name.endswith(".spv"):
                        continue
                    data = archive.read(name)
                    words = struct.unpack(f"<{len(data) // 4}I", data)
                    offset = 5  # SPIR-V module header.
                    while offset < len(words):
                        size, opcode = words[offset] >> 16, words[offset] & 0xFFFF
                        assert size and offset + size <= len(words)
                        opcodes.append(opcode)
                        offset += size
                # OpAtomicFAddEXT=6035; OpAtomicCompareExchange=230.
                assert (6035 in opcodes) == use_native
                assert (230 in opcodes) == (not use_native)
            with runtime.load_module(artifact) as loaded, closing(runtime.import_ndarray(values)) as inp, closing(
                runtime.import_ndarray(sums)
            ) as out:
                with closing(loaded.kernel("scatter").prepare(inp, out)) as replay:
                    for factor in (1, 0, -1):
                        inp.write((source * factor).tobytes())
                        out.write(bytes(4 * 4))
                        replay()
                        result = array.array("f")
                        result.frombytes(out.read())
                        np.testing.assert_array_equal(result, expected * factor)
