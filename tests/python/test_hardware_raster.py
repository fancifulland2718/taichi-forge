import numpy as np
import pytest
from taichi_forge._lib import core as _ti_core

import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=ti.cpu)
def test_raster_pass_rejects_non_vulkan_runtime():
    with pytest.raises(RuntimeError, match="requires the Vulkan backend"):
        ti.hardware.raster.RasterPass((16, 16))


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_raster_pass_executes_hardware_graphics_pipeline():
    vertices = ti.Vector.field(3, dtype=ti.f32, shape=3)
    normals = ti.Vector.field(3, dtype=ti.f32, shape=3)

    @ti.kernel
    def initialize():
        vertices[0] = ti.Vector([-0.8, -0.8, 0.0])
        vertices[1] = ti.Vector([0.8, -0.8, 0.0])
        vertices[2] = ti.Vector([0.0, 0.8, 0.0])
        for i in normals:
            normals[i] = ti.Vector([0.0, 0.0, 1.0])

    initialize()
    camera = ti.ui.Camera()
    camera.position(0.0, 0.0, 2.0)
    camera.lookat(0.0, 0.0, 0.0)

    raster_pass = ti.hardware.raster.RasterPass(
        (64, 64), background_color=(0.0, 0.0, 0.0)
    )
    try:
        raster_pass.set_camera(camera)
        raster_pass.ambient_light((1.0, 1.0, 1.0))
        raster_pass.point_light((0.0, 0.0, 2.0), (1.0, 1.0, 1.0))
        raster_pass.mesh(
            vertices,
            normals=normals,
            color=(1.0, 0.0, 0.0),
            two_sided=True,
        )

        recording = raster_pass.record()
        contract = recording.to_dict()
        assert contract["backend"] == "vulkan"
        assert contract["queue"] == "graphics"
        assert contract["stream_binding"] == "runtime_ordered"
        assert contract["workspace_ownership"] == "provider_generation"
        assert contract["no_host_readback"]
        assert len(recording.resource_effects) == 2
        assert all(
            effect.access.value == "read"
            for effect in recording.resource_effects
        )

        with pytest.raises(RuntimeError, match="Only DSL-defined native graph nodes"):
            ti.graph.GraphBuilder().append_native(recording, admission="auto")

        recording.execute()
        color = raster_pass.color_numpy()
        recording.execute()
        depth = raster_pass.depth_numpy()
        assert color.shape[:2] == (64, 64)
        assert depth.shape == (64, 64)
        assert float(np.max(color[..., 0])) > 0.25
        assert float(np.max(color[..., 0])) > float(np.max(color[..., 1]))
        assert float(np.min(depth)) < 1.0
    finally:
        raster_pass.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=ti.vulkan)
def test_vulkan_raster_recording_lifetime_and_empty_draw_fail_closed():
    raster_pass = ti.hardware.raster.RasterPass((16, 16))
    camera = ti.ui.Camera()
    raster_pass.set_camera(camera)
    with pytest.raises(RuntimeError, match="requires a new execute"):
        raster_pass.color_numpy()
    with pytest.raises(ValueError, match="at least one draw"):
        raster_pass.record()
    raster_pass.destroy()
    with pytest.raises(RuntimeError, match="destroyed"):
        raster_pass.color_numpy()
