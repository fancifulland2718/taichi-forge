"""Resource discovery uses the same backend catalog without creating a resource."""

import pytest
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_aot_resource_discovery_kind_and_backend_boundaries():
    arch = ti.cfg.arch
    with ti.aot.borrow_runtime() as runtime:
        image_modes = (
            frozenset({"cuda_capture" if arch == ti.cuda else "vulkan_inline"}) if arch != ti.cpu else frozenset()
        )
        for kind, modes in [
            ("sampled_image", image_modes),
            ("vulkan_graphics_pipeline", frozenset({"vulkan_inline"}) if arch == ti.vulkan else frozenset()),
        ]:
            assert runtime.native_resource_capabilities(kind) == modes
            assert runtime.probe_native_resource(kind) == dict(
                available=bool(modes), capabilities=modes, library_version=None
            )
            assert not runtime.native_operation_capabilities(kind)
            assert not runtime.probe_native_operation(kind)["available"]
        assert not runtime.native_resource_capabilities("compact")
        assert not runtime.probe_native_resource("compact")["available"]
        assert runtime.native_operation_capabilities("compact")
        assert runtime.probe_native_operation("compact")["available"]
        for kind, version in [("unknown_resource", 1), ("sampled_image", 0xFFFFFFFF)]:
            assert not runtime.native_resource_capabilities(kind, version=version)
            assert runtime.probe_native_resource(kind, version=version) == dict(
                available=False, capabilities=frozenset(), library_version=None
            )
        for query in (runtime.native_resource_capabilities, runtime.probe_native_resource):
            for version in (0, -1, True, 1.5, 0x100000000):
                with pytest.raises(ValueError, match="version"):
                    query("sampled_image", version=version)
            for kind in ("", "image\0sampled"):
                with pytest.raises(ValueError):
                    query(kind)
