from pathlib import Path

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.graph._ir import GraphAccess
from taichi_forge.lang import impl
from tests import test_utils


def _sampled_pipeline():
    graphics = ti.hardware.graphics
    shaders = Path(__file__).parents[2] / "python" / "taichi_forge" / "shaders"
    return graphics.VulkanGraphicsPipeline(
        (shaders / "SetImage_vk_vert.spv").read_bytes(),
        (shaders / "SetImage_vk_frag.spv").read_bytes(),
        vertex_bindings=(graphics.VertexBinding(0, 48),),
        vertex_attributes=(
            graphics.VertexAttribute(0, 0, ti.Format.rgb32f, 0),
            graphics.VertexAttribute(1, 0, ti.Format.rgb32f, 12),
            graphics.VertexAttribute(2, 0, ti.Format.rg32f, 24),
            graphics.VertexAttribute(3, 0, ti.Format.rgba32f, 32),
        ),
        shader_buffer_bindings=(graphics.ShaderBufferBinding(0, 1, "uniform", "read"),),
        shader_image_bindings=(graphics.ShaderImageBinding(0, 0),),
    )


def _quad_resources():
    data = np.zeros((6, 12), dtype=np.float32)
    data[:, :2] = [(-1, -1), (1, -1), (1, 1), (-1, -1), (1, 1), (-1, 1)]
    data[:, 6:8] = (data[:, :2] + 1) * 0.5
    vertices = ti.ndarray(ti.f32, shape=data.size)
    vertices.from_numpy(data.reshape(-1))
    uniform = ti.ndarray(ti.f32, shape=8)
    # SetImage UBO: lower, upper, x/y scale, transpose (int zero), padding.
    uniform.from_numpy(np.array([0, 0, 1, 1, 1, 1, 0, 0], dtype=np.float32))
    return vertices, uniform


def _recording(pipeline):
    draw = pipeline.pass_draw(
        ti.hardware.graphics.Draw(6),
        vertex_buffers={0: "vertices"},
        shader_buffers={(0, 1): "uniform"},
        shader_images={(0, 0): "source"},
    )
    return pipeline.record_pass((draw,), color="target")


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_sampled_graphics_device_producer_consumer_and_prepared_rebind(monkeypatch):
    if not ti.hardware.graphics.is_available():
        pytest.skip("Vulkan graphics commands are unavailable")

    @ti.kernel
    def produce(
        image: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.rgba8),
        value: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for x, y in image:
            image.store(ti.Vector([x, y]), ti.Vector([value[0], 0.25, 0.5, 1.0]))

    @ti.kernel
    def consume(
        image: ti.types.texture(num_dimensions=2),
        result: ti.types.ndarray(dtype=ti.f32, ndim=3),
    ):
        for x, y in ti.ndrange(result.shape[0], result.shape[1]):
            sampled = image.fetch(ti.Vector([x, y]), 0)
            for channel in ti.static(range(4)):
                result[x, y, channel] = sampled[channel]

    with _sampled_pipeline() as pipeline:
        vertices, uniform = _quad_resources()
        recording = _recording(pipeline)
        assert ("source", GraphAccess.READ) in tuple(
            (effect.resource, effect.access) for effect in recording.resource_effects
        )
        builder = ti.graph.GraphBuilder()
        builder.dispatch(
            produce,
            ti.graph.Arg(
                ti.graph.ArgKind.RWTEXTURE, "source", ndim=2, fmt=ti.Format.rgba8
            ),
            ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "value", ti.f32, ndim=1),
        )
        builder.append_native(recording, admission="auto")
        builder.dispatch(
            consume,
            ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "target", ndim=2),
            ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "result", ti.f32, ndim=3),
        )
        graph = builder.compile()
        source = ti.Texture(ti.Format.rgba8, (64, 32))
        target = ti.Texture(ti.Format.rgba8, (32, 16))
        value = ti.ndarray(ti.f32, shape=1)
        result = ti.ndarray(ti.f32, shape=(32, 16, 4))
        bindings = graph.bind(
            dict(
                source=source,
                target=target,
                vertices=vertices,
                uniform=uniform,
                value=value,
                result=result,
            )
        )
        prepare = recording._prepare_packet

        def unexpected_prepare(*args, **kwargs):
            raise AssertionError("Bound execution rebuilt graphics descriptors")

        monkeypatch.setattr(recording, "_prepare_packet", unexpected_prepare)
        for red in (0.2, 0.8):
            value.fill(red)
            graph.run(bindings)
            expected = np.broadcast_to([red, 0.25, 0.5, 1.0], (32, 16, 4))
            np.testing.assert_allclose(result.to_numpy(), expected, atol=1 / 255)

        monkeypatch.setattr(recording, "_prepare_packet", prepare)
        revision = bindings.revision
        with pytest.raises(
            (RuntimeError, ti.TaichiRuntimeError), match="alias|attachment"
        ):
            bindings.update(source=target)
        assert bindings.revision == revision
        replacement = ti.Texture(ti.Format.rgba8, (48, 24))
        bindings.update(source=replacement)
        monkeypatch.setattr(recording, "_prepare_packet", unexpected_prepare)
        graph.run(bindings)
        np.testing.assert_allclose(result.to_numpy(), expected, atol=1 / 255)
        graph.close()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_sampled_graphics_prepared_packet_lifetime_and_reset():
    if not ti.hardware.graphics.is_available():
        pytest.skip("Vulkan graphics commands are unavailable")
    pipeline = _sampled_pipeline()
    vertices, uniform = _quad_resources()
    source = ti.Texture(ti.Format.rgba8, (16, 8))
    target = ti.Texture(ti.Format.rgba8, (16, 8))
    recording = _recording(pipeline)
    program = impl.get_runtime().prog
    before = dict(program._debug_vulkan_graphics_resource_stats())
    queue_before = dict(program._debug_vulkan_queue_submission_stats())
    execute = recording.prepare_graph_execute(
        dict(source=source, target=target, vertices=vertices, uniform=uniform)
    )
    queue_prepared = dict(program._debug_vulkan_queue_submission_stats())
    assert (
        queue_prepared["queue_submit_calls"] == queue_before["queue_submit_calls"]
    )
    assert (
        queue_prepared["submitted_command_buffers"]
        == queue_before["submitted_command_buffers"]
    )
    prepared = dict(program._debug_vulkan_graphics_resource_stats())
    assert (
        prepared["prepared_resource_leases"]
        == before["prepared_resource_leases"] + 1
    )
    assert (
        prepared["prepared_draw_resources"]
        == before["prepared_draw_resources"] + 1
    )
    assert (
        prepared["prepared_descriptor_sets"]
        == before["prepared_descriptor_sets"] + 1
    )
    assert (
        prepared["prepared_raster_resources"]
        == before["prepared_raster_resources"] + 1
    )
    execute()
    execute()
    replayed = dict(program._debug_vulkan_graphics_resource_stats())
    for key in (
        "prepared_resource_leases",
        "prepared_draw_resources",
        "prepared_descriptor_sets",
        "prepared_raster_resources",
    ):
        assert replayed[key] == prepared[key]
    pipeline.close()
    closed = dict(program._debug_vulkan_graphics_resource_stats())
    assert closed["prepared_resource_leases"] == before["prepared_resource_leases"]
    assert closed["prepared_draw_resources"] == before["prepared_draw_resources"]
    assert closed["prepared_descriptor_sets"] == before["prepared_descriptor_sets"]
    assert closed["prepared_raster_resources"] == before["prepared_raster_resources"]
    ti.sync()
    with pytest.raises((RuntimeError, ti.TaichiRuntimeError), match="closed|stale"):
        execute()
    stats = dict(program._debug_vulkan_graphics_resource_stats())
    assert stats["retiring"] == 0
    # Retained host packets must not own a device pipeline beyond reset.
    ti.reset()
    with pytest.raises(
        (RuntimeError, ti.TaichiRuntimeError), match="runtime|generation"
    ):
        execute()
