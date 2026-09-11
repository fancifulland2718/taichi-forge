import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge._lib import core
from taichi_forge.graph._recipes.binding_frames import GraphBindingFrameRecipeProvider
from taichi_forge.graph._recipes.families import GraphRuntimeAssemblyProvider
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


def _mip_reduction(level):
    @ti.kernel
    def reduce(
        source: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.r32f, lod=level),
        destination: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.r32f, lod=level + 1),
    ):
        # Iteration and .shape must refer to the bound mip, not the allocation.
        for x, y in destination:
            total = 0.0
            for dx, dy in ti.static(ti.ndrange(2, 2)):
                total += source.load(ti.Vector([x * 2 + dx, y * 2 + dy])).x
            destination.store(ti.Vector([x, y]), ti.Vector([total * 0.25, 0.0, 0.0, 0.0]))

    return reduce


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_storage_mip_views_compose_and_replay_without_new_allocations(monkeypatch):
    from taichi_forge.lang import impl

    image = ti.Texture(ti.Format.r32f, (15, 9), mip_levels=4)
    source = ti.ndarray(ti.f32, shape=15 * 9)
    upload = ti.hardware.image.VulkanBufferToImageRecording()
    output = ti.ndarray(ti.f32, shape=4)
    reductions = [_mip_reduction(i) for i in range(3)]

    @ti.kernel
    def sample(texture: ti.types.texture(num_dimensions=2), result: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for i in range(4):
            result[i] = texture.fetch(ti.Vector([0, 0]), i).x

    builder = ti.graph.GraphBuilder()
    args = [ti.graph.Arg(ti.graph.ArgKind.RWTEXTURE, f"mip{i}", fmt=ti.Format.r32f, ndim=2) for i in range(4)]
    for i, reduction in enumerate(reductions):
        builder.dispatch(reduction, args[i], args[i + 1])
    builder.dispatch(
        sample,
        ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "sampled", ndim=2),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1),
    )
    definition = builder.freeze()
    providers = (GraphRuntimeAssemblyProvider(), GraphBindingFrameRecipeProvider())
    catalog = definition.recipe_catalog(providers=providers)
    recipe = next(entry.recipe for entry in catalog.entries() if entry.recipe.fragments)
    values = {f"mip{i}": image for i in range(4)}
    values.update(sampled=image, output=output)

    def reference(host):
        expected = [host[0, 0]]
        for _ in range(3):
            height, width = host.shape
            host = (
                sum(host[dy : height // 2 * 2 : 2, dx : width // 2 * 2 : 2] for dx in range(2) for dy in range(2))
                * 0.25
            )
            expected.append(host[0, 0])
        return expected

    with definition.materialization_context(provider_set=catalog.provider_set) as context:
        with context.materialize(recipe) as materialized:
            graph = materialized.executor
            binding = graph.bind(values)
            frame = binding._version.execution_frame
            assert frame.uses_secondary_commands()
            argument_bytes = frame.argument_bytes()
            too_short = ti.Texture(ti.Format.r32f, (15, 9), mip_levels=3)
            old_version = binding._version
            with pytest.raises(RuntimeError, match="mip level"):
                binding.update(mip3=too_short)
            assert binding._version is old_version
            too_short._delete_runtime_texture()

            def unexpected(*args, **kwargs):
                raise AssertionError("Mip replay must not reconstruct its storage views")

            monkeypatch.setattr(core, "_prepare_vulkan_graph_recording", unexpected)
            for offset in (0, 100, 200):
                host = np.arange(15 * 9, dtype=np.float32).reshape(9, 15) + offset
                source.from_numpy(host.ravel())
                upload.execute({"source": source, "destination": image})
                graph.run(binding)
                np.testing.assert_allclose(output.to_numpy(), reference(host))
                assert frame.argument_bytes() == argument_bytes
            # A prepared frame owns the one allocation, not one owner per mip.
            before = dict(impl.get_runtime().prog._debug_texture_resource_stats())
            image._delete_runtime_texture()
            graph.run(binding)
        np.testing.assert_allclose(output.to_numpy(), reference(host))
    after = dict(impl.get_runtime().prog._debug_texture_resource_stats())
    assert after["released_total"] == before["released_total"] + 1
    assert after["release_errors"] == after["retiring"] == after["inflight"] == 0


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_storage_view_lod_is_part_of_compilation_identity_and_bound_extent():
    @ti.kernel
    def write(image: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.r32u, lod=1)):
        for x, y in image:
            image.store(ti.Vector([x, y]), ti.Vector([ti.cast(image.shape[0] * 100 + x * 10 + y, ti.u32), 0, 0, 0]))

    image = ti.Texture(ti.Format.r32u, (7, 5), mip_levels=3)
    write(image)
    output = ti.ndarray(ti.u32, shape=6)
    ti.hardware.image.VulkanImageToBufferRecording(
        image_region=ti.hardware.image.VulkanImageRegion(mip_level=1)
    ).execute({"source": image, "destination": output})
    np.testing.assert_array_equal(output.to_numpy(), [300, 310, 320, 301, 311, 321])
    too_short = ti.Texture(ti.Format.r32u, (7, 5))
    with pytest.raises(RuntimeError, match="mip level"):
        write(too_short)

    def definition(level):
        builder = ti.graph.GraphBuilder()
        builder.dispatch(
            _mip_reduction(level),
            ti.graph.Arg(ti.graph.ArgKind.RWTEXTURE, "source", fmt=ti.Format.r32f, ndim=2),
            ti.graph.Arg(ti.graph.ArgKind.RWTEXTURE, "destination", fmt=ti.Format.r32f, ndim=2),
        )
        return builder.freeze()

    first, different, equivalent = definition(0), definition(1), definition(0)
    assert first.semantic_graph_id != different.semantic_graph_id
    assert first.semantic_graph_id == equivalent.semantic_graph_id
