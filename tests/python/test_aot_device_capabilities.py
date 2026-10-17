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
