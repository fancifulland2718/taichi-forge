"""Frozen common shader uniforms reproduce the live JIT raster output."""

from pathlib import Path
import struct

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge._lib import core
from taichi_forge.hardware._raster_readback import color_numpy, depth_numpy
from tests import test_utils


@pytest.mark.skipif(not core.GGUI_AVAILABLE, reason="GGUI Not Available")
@pytest.mark.parametrize("kind", ["mesh", "particles"])
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_frozen_raster_uniforms_match_live_scene(kind):
    graphics = ti.hardware.graphics
    shaders = Path(__file__).parents[2] / "python/taichi_forge/shaders"
    vertices = ti.Vector.field(3, ti.f32, 3)
    vertices.from_numpy(np.array([[-0.5, -0.4, 0], [0.5, -0.4, 0], [0, 0.4, 0]], np.float32))
    camera = ti.ui.Camera()
    camera.position(0.1, 0.2, 2)
    camera.lookat(0, 0, 0)
    camera.fov(60)
    if kind == "mesh":
        camera.projection_mode(ti.ui.ProjectionMode.Orthogonal)
    ambient, light_pos, light_color, color = (0.2, 0.3, 0.4), (1, 2, 3), (0.5, 0.4, 0.3), (0.8, 0.5, 0.25)
    with ti.hardware.raster.RasterPass((32, 24)) as raster:
        raster.set_camera(camera).ambient_light(ambient).point_light(light_pos, light_color)
        if kind == "mesh":
            raster.mesh(vertices, color=color, two_sided=True)
        else:
            raster.particles(vertices, radius=0.12, color=color)
        recording = raster.record()
        recording.execute()
        expected_color, expected_depth = color_numpy(raster.color_texture), depth_numpy(raster.depth_texture)
        assert expected_color[..., :3].max() > 0.1 and expected_depth.max() > 0
        scene_bytes = camera.ptr._raster_scene_uniforms(32, 24, ambient, 1)
        assert len(scene_bytes) == 176 and scene_bytes[12:16] == bytes(4)
        assert struct.unpack_from("<i", scene_bytes, 156) == (1,)
        camera.position(4, 3, 2)
        assert scene_bytes != camera.ptr._raster_scene_uniforms(32, 24, ambient, 1)
        with pytest.raises(ValueError):
            camera.ptr._raster_scene_uniforms(0, 24, ambient, 1)
        with pytest.raises(ValueError):
            camera.ptr._raster_scene_uniforms(32, 24, ambient, -1)

        def upload(data):
            array = ti.ndarray(ti.u32, len(data) // 4)
            array.from_numpy(np.frombuffer(data, np.uint32).copy())
            return array

        uniform = (
            core._raster_mesh_uniforms(color, False, True, False)
            if kind == "mesh"
            else core._raster_particle_uniforms(color, False, False, 0.12)
        )
        assert len(uniform) == 32 and uniform[24:] == bytes(8)
        bindings = dict(
            vertices=recording._prepared_draws[0]._buffers["vertices"],
            scene=upload(scene_bytes),
            uniform=upload(uniform),
            lights=upload(core._raster_point_light(light_pos, light_color)),
        )
        if kind == "mesh":
            bindings["transforms"] = upload(bytes(64))
        name = "Mesh" if kind == "mesh" else "Particles"
        with graphics.VulkanGraphicsPipeline(
            (shaders / f"{name}_vk_vert.spv").read_bytes(),
            (shaders / f"{name}_vk_frag.spv").read_bytes(),
            vertex_bindings=(graphics.VertexBinding(0, 48, instance=kind == "particles"),),
            vertex_attributes=(
                graphics.VertexAttribute(0, 0, ti.Format.rgb32f, 0),
                graphics.VertexAttribute(1, 0, ti.Format.rgb32f, 12),
                graphics.VertexAttribute(2, 0, ti.Format.rg32f, 24),
                graphics.VertexAttribute(3, 0, ti.Format.rgba32f, 32),
            ),
            shader_buffer_bindings=tuple(
                graphics.ShaderBufferBinding(0, i, "uniform" if i < 2 else "storage", "read")
                for i in range(4 if kind == "mesh" else 3)
            ),
            depth_test=True,
            depth_write=True,
            blending=True,
        ) as pipeline:
            draw = pipeline.pass_draw(
                graphics.Draw(3) if kind == "mesh" else graphics.Draw(6, instance_count=3),
                vertex_buffers={0: "vertices"},
                shader_buffers={
                    (0, i): name
                    for i, name in enumerate(
                        ("uniform", "scene", "lights", "transforms")
                        if kind == "mesh"
                        else ("uniform", "scene", "lights")
                    )
                },
            )
            target = ti.Texture(ti.Format.rgba8, (32, 24))
            depth = ti.Texture(ti.Format.depth32f, (32, 24))
            pipeline.record_pass((draw,), depth="depth").execute(dict(bindings, color=target, depth=depth))
            np.testing.assert_array_equal(color_numpy(target), expected_color)
            np.testing.assert_array_equal(depth_numpy(depth), expected_depth)
