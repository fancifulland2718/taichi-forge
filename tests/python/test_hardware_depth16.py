"""Normalized depth16 sampling and absolute buffer-aspect alignment."""

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_depth16_uploads_and_view_alignment():
    texture = ti.Texture(ti.Format.depth16, (2, 2))
    pixels = ti.ndarray(ti.u16, (2, 2))
    field = ti.field(ti.u16, shape=(2, 2))
    output = ti.ndarray(ti.f32, 4)
    values = np.array([0, 65535, 16384, 32768], dtype=np.uint16)

    @ti.kernel
    def sample(image: ti.types.texture(2), result: ti.types.ndarray(ti.f32, 1)):
        for i in result:
            result[i] = image.fetch(ti.Vector([i % 2, i // 2]), 0).x

    for upload, storage in ((texture.from_ndarray, pixels), (texture.from_field, field)):
        storage.from_numpy(values.reshape(2, 2))
        upload(storage)
        sample(texture, output)
        np.testing.assert_allclose(output.to_numpy(), values.astype(np.float32) / 65535, atol=1e-6)

    source, destination = [ti.ndarray(ti.u16, 6) for _ in range(2)]
    source.from_numpy(np.concatenate((np.array([123, 456], np.uint16), values)))
    destination.fill(99)
    source_view, target_view = [ti.experimental.ndarray_view(v, slices=(slice(1, 6),)) for v in (source, destination)]
    image = ti.hardware.image
    with pytest.raises(RuntimeError, match="texel/aspect"):
        image.VulkanBufferToImageRecording().execute(dict(source=source_view, destination=texture))
    with pytest.raises(RuntimeError, match="texel/aspect"):
        image.VulkanImageToBufferRecording().execute(dict(source=texture, destination=target_view))
    # Both offsets are two bytes; only the composed VkBuffer offset needs 4B.
    layout = image.VulkanBufferImageLayout(byte_offset=2)
    upload = image.VulkanBufferToImageRecording(source="input", destination="image", buffer_layout=layout)
    download = image.VulkanImageToBufferRecording(source="image", destination="output", buffer_layout=layout)
    builder = ti.graph.GraphBuilder()
    builder.append_native(upload, admission="auto")
    builder.append_native(download, admission="auto")
    graph = builder.compile()
    graph.run(dict(input=source_view, image=texture, output=target_view))
    np.testing.assert_array_equal(destination.to_numpy(), np.concatenate((np.array([99, 99], np.uint16), values)))
    graph.close()
