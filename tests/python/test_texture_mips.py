import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_managed_mip_extents_transfers_and_base_upload_isolation():
    image = ti.Texture(ti.Format.r32f, (7, 5), mip_levels=3)
    assert image.mip_levels == 3
    assert [image.mip_shape(i) for i in range(3)] == [(7, 5), (3, 2), (1, 1)]
    regions = [ti.hardware.image.VulkanImageRegion(mip_level=i) for i in range(3)]
    assert regions[1].resolved_extent(image) == (3, 2, 1)
    values = [
        np.arange(w * h, dtype=np.float32) + i * 100
        for i, (w, h) in enumerate(image.mip_shape(level) for level in range(3))
    ]
    sources = [ti.ndarray(ti.f32, shape=data.size) for data in values]
    results = [ti.ndarray(ti.f32, shape=data.size) for data in values]
    builder = ti.graph.GraphBuilder()
    bindings = {"image": image}
    for i, region in enumerate(regions):
        sources[i].from_numpy(values[i])
        builder.append_native(
            ti.hardware.image.VulkanBufferToImageRecording(
                source=f"source{i}", destination="image", image_region=region
            )
        )
        builder.append_native(
            ti.hardware.image.VulkanImageToBufferRecording(
                source="image", destination=f"result{i}", image_region=region
            )
        )
        bindings[f"source{i}"] = sources[i]
        bindings[f"result{i}"] = results[i]
    graph = builder.compile()
    graph.run(bindings)
    for result, expected in zip(results, values):
        np.testing.assert_array_equal(result.to_numpy(), expected)

    # Copy between differently sized allocations at their matching mip extents.
    small = ti.Texture(ti.Format.r32f, (3, 2))
    ti.hardware.image.copy(small, image, source_region=regions[1], destination_region=regions[0])
    ti.hardware.image.VulkanImageToBufferRecording().execute({"source": small, "destination": results[1]})
    np.testing.assert_array_equal(results[1].to_numpy(), values[1])

    # Convenience uploads replace only mip zero; initialized higher levels survive.
    base = ti.ndarray(ti.f32, shape=(7, 5))
    field = ti.field(ti.f32, shape=(7, 5))
    base.fill(17)
    field.fill(29)
    for upload, source, expected_base in ((image.from_ndarray, base, 17), (image.from_field, field, 29)):
        upload(source)
        for i, region in enumerate(regions):
            ti.hardware.image.VulkanImageToBufferRecording(image_region=region).execute(
                {"source": image, "destination": results[i]}
            )
            np.testing.assert_array_equal(results[i].to_numpy(), expected_base if i == 0 else values[i])


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_mip_sample_lod_graph_replay_observes_in_place_uploads():
    image = ti.Texture(ti.Format.r32f, (8, 4), mip_levels=4)
    output = ti.ndarray(ti.f32, shape=4)
    inputs = [ti.ndarray(ti.f32, shape=max(1, 8 >> i) * max(1, 4 >> i)) for i in range(4)]
    uploads = [
        ti.hardware.image.VulkanBufferToImageRecording(image_region=ti.hardware.image.VulkanImageRegion(mip_level=i))
        for i in range(4)
    ]

    @ti.kernel
    def sample(texture: ti.types.texture(num_dimensions=2), result: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for i in range(4):
            result[i] = texture.sample_lod(ti.Vector([0.5, 0.5]), ti.cast(i, ti.f32)).x

    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        sample,
        ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "image", ndim=2),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "result", ti.f32, ndim=1),
    )
    graph = builder.compile()
    bound = graph.bind({"image": image, "result": output})
    _ = graph._instance.debug_graph_stats
    for base in (10, 20, 30):
        for i, (source, upload) in enumerate(zip(inputs, uploads)):
            source.fill(base + i)
            upload.execute({"source": source, "destination": image})
        graph.run(bound)
        np.testing.assert_allclose(output.to_numpy(), np.arange(4, dtype=np.float32) + base)
    stats = graph.execution_stats()
    assert stats.execution_path != "ordinary"
    graph.close()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_mip_contract_rejects_out_of_chain_or_out_of_level_extent():
    with pytest.raises(ValueError, match="complete mip chain"):
        ti.Texture(ti.Format.r32f, (7, 5), mip_levels=4)
    with pytest.raises(ValueError, match="2D Vulkan"):
        ti.Texture(ti.Format.r32f, (8,), mip_levels=2)
    image = ti.Texture(ti.Format.r32f, (7, 5), mip_levels=3)
    buffer = ti.ndarray(ti.f32, shape=35)
    with pytest.raises(ValueError, match="mip level"):
        image.mip_shape(3)
    with pytest.raises(RuntimeError, match="region exceeds"):
        ti.hardware.image.VulkanBufferToImageRecording(
            image_region=ti.hardware.image.VulkanImageRegion(mip_level=1, extent=(4, 2))
        ).execute({"source": buffer, "destination": image})
    with pytest.raises(RuntimeError, match="mip level"):
        ti.hardware.image.VulkanBufferToImageRecording(
            image_region=ti.hardware.image.VulkanImageRegion(mip_level=3, extent=(1, 1))
        ).execute({"source": buffer, "destination": image})
