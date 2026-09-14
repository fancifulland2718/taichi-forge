"""Public programmable RT: fixed resources, shared AS and explicit initialization."""

from pathlib import Path
import re
import struct

import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


def _shader(name, stage):
    source = Path(__file__).parents[1] / "cpp/aot/vulkan/shaders" / (name + ".spv.h")
    words = [int(value, 16) for value in re.findall(r"0x[0-9a-fA-F]+", source.read_text())]
    return ti.hardware.ray.SpirvShader(struct.pack(f"<{len(words)}I", *words), stage)


def _pipeline():
    ray = ti.hardware.ray
    return ray.VulkanRayTracingPipeline(
        raygen={"primary": _shader("instance_mapping.rgen", "raygen")},
        miss={"background": _shader("instance_mapping.rmiss", "miss")},
        hit_groups={
            "opaque": ray.VulkanHitGroup(_shader("instance_mapping.rchit", "closest_hit")),
            "surface": ray.VulkanHitGroup(
                _shader("instance_mapping.rchit", "closest_hit"), _shader("instance_mapping.rahit", "any_hit")
            ),
        },
    )


def _instances(blas, z=0):
    ray = ti.hardware.ray
    return [
        ray.RayInstance(
            blas, transform=(1, 0, 0, x, 0, 1, 0, 0, 0, 0, 1, z), custom_index=custom, sbt_record_offset=offset
        )
        for x, custom, offset in ((0, 7, 1), (2, 11, 4))
    ]


def _graph(builder, fixed):
    if not fixed:
        return builder.compile(), None, None
    from taichi_forge.graph._recipes.binding_frames import GraphBindingFrameRecipeProvider
    from taichi_forge.graph._recipes.families import GraphRuntimeAssemblyProvider

    definition = builder.freeze()
    catalog = definition.recipe_catalog(providers=(GraphRuntimeAssemblyProvider(), GraphBindingFrameRecipeProvider()))
    recipe = next(entry.recipe for entry in catalog.entries() if entry.recipe.fragments)
    context = definition.materialization_context(provider_set=catalog.provider_set)
    materialized = context.materialize(recipe)
    return materialized.executor, context, materialized


@pytest.mark.parametrize("fixed_graph", [False, True])
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_public_vulkan_program_direct_graph_shared_as_and_close(monkeypatch, fixed_graph):
    ray = ti.hardware.ray
    if not ray.is_program_available():
        pytest.skip("Vulkan RT pipeline unavailable")
    vertices, indices = ti.ndarray(ti.f32, (6, 3)), ti.ndarray(ti.i32, (2, 3))
    vertices.from_numpy(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]], np.float32))
    indices.from_numpy(np.array([[0, 1, 2], [3, 4, 5]], np.int32))
    blas = ray.TriangleBLAS(vertices, indices, opaque=False)
    scene = ray.InstanceTLAS(_instances(blas))
    output = ti.ndarray(ti.u32, (5, 8))
    pipeline = _pipeline()
    recording = pipeline.record(
        5,
        raygen="primary",
        bindings={
            "scene": ray.VulkanRayBinding(0, 0, "scene"),
            "output": ray.VulkanRayBinding(0, 1, access="write"),
        },
        miss=[ray.VulkanSbtRecord("background", struct.pack("<I", 200 + i)) for i in range(2)],
        hit=[
            ray.VulkanSbtRecord("opaque" if i in (2, 5) else "surface", struct.pack("<I", 10 * i + 3))
            for i in range(6)
        ],
    )
    launch = recording.prepare(dict(scene=scene, output=output))
    before = launch.preparation_info()
    assert before["initialized"] == 0 and before["upload_requested_bytes"] > 0
    with pytest.raises(RuntimeError, match="initialize"):
        launch.run()
    launch.initialize()
    after = launch.preparation_info()
    assert after["sbt_requested_bytes"] == before["sbt_requested_bytes"]
    assert after["initialized"] == 1 and after["upload_requested_bytes"] == 0
    assert launch.memory_report().to_dict()["provider"] == "vulkan_ray_program"
    launch.run()
    values = output.to_numpy()
    np.testing.assert_array_equal(values[:4, :4], [[1, 0, 7, 13], [0, 0, 7, 23], [1, 1, 11, 43], [0, 1, 11, 53]])
    expected_geometry = np.tile([2, 0.25, 0.25, 1], (4, 1))
    expected_geometry[1::2, 0] = 1
    np.testing.assert_array_equal(values[:4, 4:].view(np.float32), expected_geometry)
    np.testing.assert_array_equal(values[4, :4], [0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 200])

    if ray.is_available():
        inline_result = ti.ndarray(ti.f32, 2)

        @ti.kernel
        def inline_query(world: ti.types.acceleration_structure(), result: ti.types.ndarray(ti.f32, ndim=1)):
            for i in result:
                hit = world.trace_closest(
                    ti.Vector([0.25 + 2.0 * i, 0.25, -1.0]), ti.Vector([0.0, 0.0, 1.0]), ray_flags=1
                )
                result[i] = hit.t

        # Explicit ForceOpaque query uses the same AS, ignoring shader-record
        # mappings and the programmable any-hit filter.
        inline_query(scene, inline_result)
        np.testing.assert_array_equal(inline_result.to_numpy(), [1, 1])

    builder = ti.graph.GraphBuilder()
    builder.append_native(launch.graph_recording(), admission="auto")
    graph, context, materialized = _graph(builder, fixed_graph)
    bound = graph.bind(dict(output=output))
    if fixed_graph:
        assert bound._version.execution_frame.uses_secondary_commands()

    def unexpected(*args, **kwargs):
        raise AssertionError("program/descriptor/SBT preparation was repeated during run")

    monkeypatch.setattr(ray.VulkanProgramRecording, "prepare", unexpected)
    monkeypatch.setattr(ray.VulkanPreparedLaunch, "initialize", unexpected)
    scene.refit(_instances(blas, 1))
    graph.run(bound)
    values = output.to_numpy()
    np.testing.assert_array_equal(values[:4, 4:].view(np.float32)[:, 0], [3, 2, 3, 2])
    graph.close()
    if context is not None:
        materialized.close()
        context.close()

    transforms, shift = ti.ndarray(ti.f32, (2, 12)), ti.ndarray(ti.f32, 1)
    distances = ti.ndarray(ti.f32, 4)

    @ti.kernel
    def transform(values: ti.types.ndarray(ti.f32, ndim=2), z: ti.types.ndarray(ti.f32, ndim=1)):
        for i, j in values:
            values[i, j] = 0.0
            if j == 0 or j == 5 or j == 10:
                values[i, j] = 1.0
            if j == 3:
                values[i, j] = 2.0 * i
            if j == 11:
                values[i, j] = z[0]

    @ti.kernel
    def consume(values: ti.types.ndarray(ti.u32, ndim=2), result: ti.types.ndarray(ti.f32, ndim=1)):
        for i in result:
            result[i] = ti.bit_cast(values[i, 4], ti.f32) * 2.0

    dynamic = ti.graph.GraphBuilder()
    dynamic.dispatch(
        transform,
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "transforms", ti.f32, ndim=2),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "shift", ti.f32, ndim=1),
    )
    dynamic.append_native(scene.record_refit_transforms(), admission="auto")
    dynamic.append_native(launch.graph_recording(), admission="auto")
    dynamic.dispatch(
        consume,
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.u32, ndim=2),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "distances", ti.f32, ndim=1),
    )
    graph, context, materialized = _graph(dynamic, fixed_graph)
    bound = graph.bind(dict(transforms=transforms, shift=shift, output=output, distances=distances))
    if fixed_graph:
        assert graph._instance.physical_submission_mode == "vulkan_secondary_frames_with_ordered_native_published"
    for z in (0, 1, 0):
        shift.fill(z)
        graph.run(bound)
        np.testing.assert_array_equal(distances.to_numpy(), np.array([4, 2, 4, 2], np.float32) + 2 * z)
    scene.close()
    if fixed_graph:
        with pytest.raises(RuntimeError, match="closed|retired|invalid"):
            graph.run(bound)
    graph.close()
    if context is not None:
        materialized.close()
        context.close()
    with pytest.raises(RuntimeError, match="closed|retired"):
        launch.run()
    launch.close()
    pipeline.close()
    blas.close()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_program_failed_binding_is_transactional_and_reset_invalidates():
    ray = ti.hardware.ray
    if not ray.is_program_available():
        pytest.skip("Vulkan RT pipeline unavailable")
    shader = _shader("sbt_record.rgen", "raygen")
    pipeline = ray.VulkanRayTracingPipeline(raygen={"entry": shader})
    output = ti.ndarray(ti.u32, 257)
    output.from_numpy(np.arange(257, dtype=np.uint32))
    sbt = ray.VulkanSbtRecord("entry", struct.pack("<I", 17))
    bad = pipeline.record(257, raygen=sbt, bindings={"output": ray.VulkanRayBinding(0, 1, access="read_write")})
    with pytest.raises(RuntimeError, match="descriptor"):
        bad.prepare(dict(output=output))
    good = pipeline.record(257, raygen=sbt, bindings={"output": ray.VulkanRayBinding(0, 0, access="read_write")})
    launch = good.prepare(dict(output=output)).initialize()
    launch.run()
    launch.run()
    np.testing.assert_array_equal(output.to_numpy(), 9 * np.arange(257, dtype=np.uint32) + 68)
    ti.reset()
    with pytest.raises(RuntimeError, match="runtime|retired|closed"):
        launch.run()
    launch.close()
    pipeline.close()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_program_image_mip_and_dense_field_in_complete_graph(monkeypatch):
    ray = ti.hardware.ray
    if not ray.is_program_available():
        pytest.skip("Vulkan RT pipeline unavailable")
    from taichi_forge._lib import core
    from taichi_forge.graph._recipes.binding_frames import GraphBindingFrameRecipeProvider
    from taichi_forge.graph._recipes.families import GraphRuntimeAssemblyProvider

    source = ti.Texture(ti.Format.rgba32f, (13, 7))
    target = ti.Texture(ti.Format.rgba32f, (26, 14), mip_levels=2)
    gain = ti.field(ti.f32, shape=1)
    result = ti.ndarray(ti.f32, (13, 7, 4))
    pipeline = ray.VulkanRayTracingPipeline(raygen={"image": _shader("program_images.rgen", "raygen")})
    recording = pipeline.record(
        (13, 7),
        raygen="image",
        push_constants=struct.pack("<4f", 0.5, 0, 0, 0),
        bindings={
            "source": ray.VulkanRayBinding(0, 0, "sampled_image"),
            "target": ray.VulkanRayBinding(0, 1, "storage_image", "write", mip_level=1),
            "gain": ray.VulkanRayBinding(0, 2),
        },
    )
    launch = recording.prepare(dict(source=source, target=target, gain=gain)).initialize()

    @ti.kernel
    def produce(image: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.rgba32f)):
        for x, y in image:
            image.store(ti.Vector([x, y]), ti.Vector([x, y, x + y, 1.0]))

    @ti.kernel
    def consume(image: ti.types.texture(num_dimensions=2), output: ti.types.ndarray(dtype=ti.f32, ndim=3)):
        for x, y in ti.ndrange(13, 7):
            value = image.fetch(ti.Vector([x, y]), 1)
            for c in ti.static(range(4)):
                output[x, y, c] = value[c]

    builder = ti.graph.GraphBuilder()
    builder.dispatch(produce, ti.graph.Arg(ti.graph.ArgKind.RWTEXTURE, "source", ndim=2, fmt=ti.Format.rgba32f))
    builder.append_native(launch.graph_recording(), admission="auto")
    builder.dispatch(
        consume,
        ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "target", ndim=2),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "result", ti.f32, ndim=3),
    )
    definition = builder.freeze()
    catalog = definition.recipe_catalog(providers=(GraphRuntimeAssemblyProvider(), GraphBindingFrameRecipeProvider()))
    recipe = next(entry.recipe for entry in catalog.entries() if entry.recipe.fragments)
    with definition.materialization_context(provider_set=catalog.provider_set) as context:
        with context.materialize(recipe) as materialized:
            graph = materialized.executor
            bound = graph.bind(dict(source=source, target=target, gain=gain, result=result))
            assert bound._version.execution_frame.uses_secondary_commands()

            def unexpected(*args, **kwargs):
                raise AssertionError("fixed RT replay rebuilt commands or initialized resources")

            monkeypatch.setattr(core, "_prepare_vulkan_graph_recording", unexpected)
            monkeypatch.setattr(ray.VulkanPreparedLaunch, "initialize", unexpected)
            x, y = np.meshgrid(np.arange(13), np.arange(7), indexing="ij")
            expected = np.stack((x, y, x + y, np.ones_like(x)), axis=-1).astype(np.float32)
            for value in (2, 3, 2):
                gain.fill(value)
                graph.run(bound)
                np.testing.assert_array_equal(result.to_numpy(), expected * value + 0.5)
            # Closing a program must first invalidate its published Graph frames.
            graph.run(bound)
            pipeline.close()
            ti.sync()
            np.testing.assert_array_equal(result.to_numpy(), expected * 2 + 0.5)
            with pytest.raises(RuntimeError, match="closed|retired|invalid"):
                graph.run(bound)
