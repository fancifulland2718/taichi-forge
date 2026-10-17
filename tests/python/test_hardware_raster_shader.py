"""The shared raster shaders must not read disabled optional inputs."""

from pathlib import Path

import numpy as np
import taichi_forge as ti
from tests import test_utils
from tests.python.test_hardware_graphics import _texture_rgb


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_mesh_shader_ignores_disabled_instance_transform():
    graphics = ti.hardware.graphics
    shaders = Path(__file__).parents[2] / "python/taichi_forge/shaders"
    vertices = ti.ndarray(ti.f32, 36)
    data = np.zeros((3, 12), np.float32)
    data[:, :3] = ((-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0, 0.5, 0.5))
    data[:, 5] = 1
    vertices.from_numpy(data.ravel())
    scene = ti.ndarray(ti.f32, 44)
    values = np.zeros(44, np.float32)
    values[4:20] = values[20:36] = np.eye(4).ravel()
    values[36:39] = 1  # Ambient-only shading; zero point lights.
    scene.from_numpy(values)
    uniform, lights, transforms = (ti.ndarray(ti.f32, n) for n in (8, 8, 16))
    bindings = dict(vertices=vertices, scene=scene, uniform=uniform, lights=lights, transforms=transforms)
    with graphics.VulkanGraphicsPipeline(
        (shaders / "Mesh_vk_vert.spv").read_bytes(),
        (shaders / "Mesh_vk_frag.spv").read_bytes(),
        vertex_bindings=(graphics.VertexBinding(0, 48),),
        vertex_attributes=(
            graphics.VertexAttribute(0, 0, ti.Format.rgb32f, 0),
            graphics.VertexAttribute(1, 0, ti.Format.rgb32f, 12),
            graphics.VertexAttribute(2, 0, ti.Format.rg32f, 24),
            graphics.VertexAttribute(3, 0, ti.Format.rgba32f, 32),
        ),
        shader_buffer_bindings=tuple(
            graphics.ShaderBufferBinding(0, slot, "uniform" if slot < 2 else "storage", "read") for slot in range(4)
        ),
    ) as pipeline:

        def render(matrix, *, enabled=False, first_instance=0):
            values = np.zeros(8, np.float32)
            values[0] = 1
            values[5] = enabled
            uniform.from_numpy(values)
            transforms.from_numpy(np.asarray(matrix, np.float32).ravel())
            image = ti.Texture(ti.Format.rgba8, (32, 32))
            draw = pipeline.pass_draw(
                graphics.Draw(3, first_instance=first_instance),
                vertex_buffers={0: "vertices"},
                shader_buffers={(0, i): name for i, name in enumerate(("uniform", "scene", "lights", "transforms"))},
            )
            pipeline.record_pass((draw,)).execute(dict(bindings, color=image))
            return _texture_rgb(image)

        identity = np.eye(4, dtype=np.float32)
        expected = render(identity)
        assert expected[..., 0].max() > 200
        np.testing.assert_array_equal(render(np.full((4, 4), np.nan)), expected)
        # No transform is enabled, so this instance index must never dereference
        # the one-matrix descriptor even though the draw itself is valid.
        np.testing.assert_array_equal(render(np.full((4, 4), np.nan), first_instance=7), expected)
        np.testing.assert_array_equal(render(identity, enabled=True), expected)
        translated = identity.copy()
        translated[0, 3] = 0.5
        assert not np.array_equal(render(translated, enabled=True), expected)
