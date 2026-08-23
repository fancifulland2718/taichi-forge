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
    def write(
        texture: ti.types.rw_texture(
            num_dimensions=2, fmt=ti.Format.r32f, lod=0
        ),
        base: ti.f32,
    ):
        for i, j in ti.ndrange(4, 4):
            texture.store(
                ti.Vector([i, j]),
                ti.Vector([base + i * 10.0 + j, 0.0, 0.0, 0.0]),
            )

    @ti.kernel
    def read(
        texture: ti.types.texture(num_dimensions=2),
        result: ti.types.ndarray(dtype=ti.f32, ndim=2),
    ):
        for i, j in ti.ndrange(4, 4):
            result[i, j] = texture.fetch(ti.Vector([i, j]), 0).x

    pattern = np.fromfunction(lambda i, j: i * 10 + j, (4, 4), dtype=np.float32)
    for iteration in range(16):
        base = float(iteration * 100)
        write(source, base)
        write(destination, -1000.0 - base)
        assert ti.hardware.image.copy(destination, source) is destination
        read(destination, output)
        np.testing.assert_allclose(output.to_numpy(), pattern + base)

    recording = ti.hardware.image.VulkanImageCopyRecording(
        source="input", destination="output"
    )
    assert tuple(
        (effect.resource, effect.access) for effect in recording.resource_effects
    ) == (("input", GraphAccess.READ), ("output", GraphAccess.WRITE))
    builder = ti.graph.GraphBuilder()
    builder.append_native(recording, admission="auto")
    graph = builder.compile()
    write(source, 2000.0)
    write(destination, -2000.0)
    graph.run({"input": source, "output": destination})
    read(destination, output)
    np.testing.assert_allclose(output.to_numpy(), pattern + 2000.0)
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


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_image_region_copy_preserves_pixels_outside_region():
    source = ti.Texture(ti.Format.r32f, (6, 6))
    destination = ti.Texture(ti.Format.r32f, (6, 6))
    output = ti.ndarray(ti.f32, shape=(6, 6))

    @ti.kernel
    def write_pattern(
        texture: ti.types.rw_texture(
            num_dimensions=2, fmt=ti.Format.r32f, lod=0
        ),
        base: ti.f32,
    ):
        for i, j in ti.ndrange(6, 6):
            texture.store(
                ti.Vector([i, j]),
                ti.Vector([base + i * 10.0 + j, 0.0, 0.0, 0.0]),
            )

    @ti.kernel
    def read(
        texture: ti.types.texture(num_dimensions=2),
        result: ti.types.ndarray(dtype=ti.f32, ndim=2),
    ):
        for i, j in ti.ndrange(6, 6):
            result[i, j] = texture.fetch(ti.Vector([i, j]), 0).x

    write_pattern(source, 0.0)
    write_pattern(destination, -1000.0)
    source_region = ti.hardware.image.VulkanImageRegion(
        offset=(1, 2), extent=(3, 2)
    )
    destination_region = ti.hardware.image.VulkanImageRegion(offset=(2, 1))
    recording = ti.hardware.image.VulkanImageCopyRecording(
        source_region=source_region,
        destination_region=destination_region,
    )
    assert recording.resource_effects[0].subresource == (
        "image",
        0,
        0,
        1,
        (1, 2, 0),
        (3, 2, 1),
    )
    assert recording.resource_effects[0].to_dict()["subresource"] == (
        "image",
        0,
        0,
        1,
        (1, 2, 0),
        (3, 2, 1),
    )
    assert recording.resource_effects[1].subresource[-1] == (3, 2, 1)

    builder = ti.graph.GraphBuilder()
    builder.append_native(recording, admission="auto")
    graph = builder.compile()
    graph.run({"source": source, "destination": destination})
    read(destination, output)
    actual = output.to_numpy()
    expected = np.fromfunction(
        lambda i, j: -1000.0 + i * 10.0 + j,
        (6, 6),
        dtype=np.float32,
    )
    expected[2:5, 1:3] = np.fromfunction(
        lambda i, j: (i + 1) * 10.0 + (j + 2),
        (3, 2),
        dtype=np.float32,
    )
    np.testing.assert_allclose(actual, expected)

    unsupported = ti.hardware.image.VulkanImageCopyRecording(
        source_region=ti.hardware.image.VulkanImageRegion(
            extent=(1, 1), mip_level=1
        ),
        destination_region=ti.hardware.image.VulkanImageRegion(extent=(1, 1)),
    )
    with pytest.raises(RuntimeError, match="mip level"):
        unsupported.execute({"source": source, "destination": destination})


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_buffer_image_region_round_trip_with_pitch_and_offset():
    texture = ti.Texture(ti.Format.r32f, (5, 4))
    image_region = ti.hardware.image.VulkanImageRegion(
        offset=(1, 1), extent=(3, 2)
    )
    layout = ti.hardware.image.VulkanBufferImageLayout(
        byte_offset=8, row_length=5, image_height=4
    )
    source_values = np.full(20, -99.0, dtype=np.float32)
    source_values[2:5] = (10.0, 11.0, 12.0)
    source_values[7:10] = (20.0, 21.0, 22.0)
    source = ti.ndarray(ti.f32, shape=20)
    destination = ti.ndarray(ti.f32, shape=20)
    source.from_numpy(source_values)
    destination.from_numpy(np.full(20, -77.0, dtype=np.float32))

    upload = ti.hardware.image.VulkanBufferToImageRecording(
        source="input_buffer",
        destination="image",
        buffer_layout=layout,
        image_region=image_region,
    )
    download = ti.hardware.image.VulkanImageToBufferRecording(
        source="image",
        destination="output_buffer",
        buffer_layout=layout,
        image_region=image_region,
    )
    assert upload.resource_effects[0].access == GraphAccess.READ
    assert upload.resource_effects[1].subresource[-1] == (3, 2, 1)

    builder = ti.graph.GraphBuilder()
    builder.append_native(upload, admission="auto")
    builder.append_native(download, admission="auto")
    graph = builder.compile()
    graph.run(
        {
            "input_buffer": source,
            "image": texture,
            "output_buffer": destination,
        }
    )
    ti.sync()
    actual = destination.to_numpy()
    np.testing.assert_allclose(actual[2:5], source_values[2:5])
    np.testing.assert_allclose(actual[7:10], source_values[7:10])
    untouched = np.ones(20, dtype=bool)
    untouched[2:5] = False
    untouched[7:10] = False
    np.testing.assert_allclose(actual[untouched], -77.0)

    too_small = ti.ndarray(ti.f32, shape=4)
    with pytest.raises(RuntimeError, match="too small"):
        upload.execute({"input_buffer": too_small, "image": texture})
    misaligned = ti.hardware.image.VulkanBufferToImageRecording(
        buffer_layout=ti.hardware.image.VulkanBufferImageLayout(byte_offset=2),
        image_region=image_region,
    )
    with pytest.raises(RuntimeError, match="texel block size"):
        misaligned.execute({"source": source, "destination": texture})

    excessive_pitch = ti.hardware.image.VulkanBufferToImageRecording(
        buffer_layout=ti.hardware.image.VulkanBufferImageLayout(
            row_length=0x7FFFFFFF
        ),
        image_region=image_region,
    )
    with pytest.raises(RuntimeError, match="row pitch"):
        excessive_pitch.execute({"source": source, "destination": texture})



@test_utils.test(arch=ti.vulkan, offline_cache=False, debug=True)
def test_vulkan_byte_aligned_buffer_image_copy_passes_validation():
    texture = ti.Texture(ti.Format.r8, (1, 1))
    source = ti.ndarray(ti.u8, shape=4)
    destination = ti.ndarray(ti.u8, shape=4)
    source.from_numpy(np.array([0, 123, 0, 0], dtype=np.uint8))
    destination.from_numpy(np.zeros(4, dtype=np.uint8))
    layout = ti.hardware.image.VulkanBufferImageLayout(byte_offset=1)
    region = ti.hardware.image.VulkanImageRegion(extent=(1, 1))
    ti.hardware.image.copy_buffer_to_image(
        texture, source, buffer_layout=layout, image_region=region
    )
    ti.hardware.image.copy_image_to_buffer(
        destination, texture, buffer_layout=layout, image_region=region
    )
    np.testing.assert_array_equal(
        destination.to_numpy(), np.array([0, 123, 0, 0], dtype=np.uint8)
    )


@test_utils.test(arch=ti.vulkan, offline_cache=False, debug=True)
def test_vulkan_image_blit_is_feature_gated_and_scales():
    source = ti.Texture(ti.Format.r32f, (4, 4))
    destination = ti.Texture(ti.Format.r32f, (2, 2))
    output = ti.ndarray(ti.f32, shape=(2, 2))

    @ti.kernel
    def fill(
        texture: ti.types.rw_texture(
            num_dimensions=2, fmt=ti.Format.r32f, lod=0
        )
    ):
        for i, j in ti.ndrange(4, 4):
            texture.store(ti.Vector([i, j]), ti.Vector([7.0, 0.0, 0.0, 0.0]))

    @ti.kernel
    def read(
        texture: ti.types.texture(num_dimensions=2),
        result: ti.types.ndarray(dtype=ti.f32, ndim=2),
    ):
        for i, j in ti.ndrange(2, 2):
            result[i, j] = texture.fetch(ti.Vector([i, j]), 0).x

    fill(source)
    recording = ti.hardware.image.VulkanImageBlitRecording(filter="linear")
    assert recording.queue == "graphics"
    try:
        recording.execute({"source": source, "destination": destination})
    except RuntimeError as exc:
        if "unsupported for the selected format and filter" in str(exc):
            pytest.skip("active Vulkan device does not support linear r32f blit")
        raise
    read(destination, output)
    np.testing.assert_allclose(output.to_numpy(), 7.0)

    with pytest.raises(ValueError, match="filter"):
        ti.hardware.image.VulkanImageBlitRecording(filter="cubic")
