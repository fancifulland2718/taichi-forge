import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.graph._ir import GraphAccess
from tests import test_utils


@test_utils.test(arch=ti.cpu)
def test_vulkan_image_copy_rejects_non_vulkan_runtime():
    with pytest.raises(RuntimeError, match="requires the Vulkan backend"):
        ti.hardware.image.VulkanImageCopyRecording()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_image_copy_direct_graph_ordering_and_lifetime():
    source = ti.Texture(ti.Format.r32f, (4, 4))
    destination = ti.Texture(ti.Format.r32f, (4, 4))
    output = ti.ndarray(ti.f32, shape=(4, 4))

    @ti.kernel
    def write(texture: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.r32f, lod=0)):
        for i, j in ti.ndrange(4, 4):
            texture.store(
                ti.Vector([i, j]),
                ti.Vector([i * 10.0 + j, 0.0, 0.0, 0.0]),
            )

    @ti.kernel
    def read(
        texture: ti.types.texture(num_dimensions=2),
        result: ti.types.ndarray(dtype=ti.f32, ndim=2),
    ):
        for i, j in ti.ndrange(4, 4):
            result[i, j] = texture.fetch(ti.Vector([i, j]), 0).x

    expected = np.fromfunction(
        lambda i, j: i * 10 + j, (4, 4), dtype=np.float32
    )
    for _ in range(16):
        write(source)
        assert ti.hardware.image.copy(destination, source) is destination
        read(destination, output)
        np.testing.assert_allclose(output.to_numpy(), expected)

    recording = ti.hardware.image.VulkanImageCopyRecording(
        source="input", destination="output"
    )
    assert tuple(
        (effect.resource, effect.access) for effect in recording.resource_effects
    ) == (("input", GraphAccess.READ), ("output", GraphAccess.WRITE))
    builder = ti.graph.GraphBuilder()
    builder.append_native(recording, admission="auto")
    graph = builder.compile()
    graph.run({"input": source, "output": destination})
    read(destination, output)
    np.testing.assert_allclose(output.to_numpy()[3, 2], 32.0)
    assert graph._debug_info["optimization"]["backend_command_nodes"] == 1

    ti.reset()
    with pytest.raises(RuntimeError, match="previous Taichi runtime"):
        recording.validate_graph_lifetime()
    with pytest.raises(RuntimeError, match="compiled before ti.reset"):
        graph.run({"input": source, "output": destination})


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_image_copy_validation():
    source = ti.Texture(ti.Format.r32f, (4, 4))
    different_extent = ti.Texture(ti.Format.r32f, (2, 2))
    different_format = ti.Texture(ti.Format.rgba8, (4, 4))
    recording = ti.hardware.image.VulkanImageCopyRecording()

    with pytest.raises(RuntimeError, match="extents"):
        recording.execute(
            {"source": source, "destination": different_extent}
        )
    with pytest.raises(RuntimeError, match="formats"):
        recording.execute(
            {"source": source, "destination": different_format}
        )
    with pytest.raises(RuntimeError, match="must not alias"):
        recording.execute({"source": source, "destination": source})
