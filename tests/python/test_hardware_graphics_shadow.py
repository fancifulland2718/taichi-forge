"""Depth-only producer -> graphics/compute comparison consumers, on device."""

from pathlib import Path
import re
import struct

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils

ASSETS = Path(__file__).parent / "assets"


def _pipeline(fragment, **kwargs):
    words = [
        int(x, 16)
        for x in re.findall(
            r"0x[0-9a-fA-F]+",
            (ASSETS / "hardware_graphics_depth.vert.spv.h").read_text(),
        )
    ]
    gfx = ti.hardware.graphics
    return gfx.VulkanGraphicsPipeline(
        struct.pack(f"<{len(words)}I", *words),
        (ASSETS / fragment).read_bytes(),
        vertex_bindings=(gfx.VertexBinding(0, 24),),
        vertex_attributes=(
            gfx.VertexAttribute(0, 0, ti.Format.rgb32f, 0),
            gfx.VertexAttribute(1, 0, ti.Format.rgb32f, 12),
        ),
        **kwargs,
    )


def _vertices():
    data = np.zeros((6, 6), np.float32)
    data[:, :2] = [(-1, -1), (0, -1), (0, 1), (-1, -1), (0, 1), (-1, 1)]
    data[:, 2] = 0.25
    vertices = ti.ndarray(ti.f32, data.size)
    vertices.from_numpy(data.reshape(-1))
    return vertices, data


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_shadow_depth_only_graph_consumers_rebind_and_comparison_filter(monkeypatch):
    gfx = ti.hardware.graphics
    vertices, data = _vertices()
    config = ti.hardware.sampling.SamplerConfig(
        compare_op="less_equal",
        address_mode_u="clamp_to_edge",
        address_mode_v="clamp_to_edge",
    )
    depth = ti.Texture(ti.Format.depth32f, (32, 24), sampler=config)
    color = ti.Texture(ti.Format.rgba32f, (32, 24))
    result = ti.ndarray(ti.f32, 5)
    graphics_result = ti.Vector.ndarray(4, ti.f32, (32, 24))

    @ti.kernel
    def consume(
        shadow: ti.types.texture(2),
        image: ti.types.texture(2),
        out: ti.types.ndarray(ti.f32, ndim=1),
        pixels: ti.types.ndarray(),
    ):
        out[0] = shadow.sample_compare(ti.Vector([0.25, 0.5]), 0.5)
        out[1] = shadow.sample_compare(ti.Vector([0.75, 0.5]), 0.5)
        out[2] = shadow.sample_compare(ti.Vector([0.5, 0.5]), 0.5)
        out[3] = shadow.fetch(ti.Vector([8, 12]), 0).x
        out[4] = shadow.fetch(ti.Vector([24, 12]), 0).x
        for i, j in pixels:
            pixels[i, j] = image.fetch(ti.Vector([i, j]), 0)

    writer = _pipeline(
        "hardware_graphics_depth_only.frag.spv",
        depth_test=True,
        depth_write=True,
        depth_compare="less",
    )
    reader = _pipeline(
        "hardware_graphics_shadow.frag.spv",
        shader_image_bindings=(gfx.ShaderImageBinding(0, 0),),
    )
    write = writer.record_pass(
        (writer.pass_draw(gfx.Draw(6), vertex_buffers={0: "vertices"}),),
        colors=(),
        depth="shadow",
        clear_depth=1.0,
    )
    sample = reader.record_pass(
        (
            reader.pass_draw(
                gfx.Draw(6),
                vertex_buffers={0: "vertices"},
                shader_images={(0, 0): "shadow"},
            ),
        ),
        color="color",
    )
    builder = ti.graph.GraphBuilder()
    builder.append_native(write, admission="auto")
    builder.append_native(sample, admission="auto")
    arg, kind = ti.graph.Arg, ti.graph.ArgKind
    builder.dispatch(
        consume,
        arg(kind.TEXTURE, "shadow", ndim=2),
        arg(kind.TEXTURE, "color", ndim=2),
        arg(kind.NDARRAY, "result", ti.f32, ndim=1),
        arg(kind.NDARRAY, "pixels", ti.types.vector(4, ti.f32), ndim=2),
    )
    graph = builder.compile()
    bindings = dict(
        vertices=vertices,
        shadow=depth,
        color=color,
        result=result,
        pixels=graphics_result,
    )
    bound = graph.bind(bindings)
    prepare = write._prepare_packet

    def unexpected_prepare(*args, **kwargs):
        raise AssertionError("steady bound replay must not prepare graphics")

    for z, expected in ((0.25, (0, 1, 0.5, 0.25, 1)), (0.75, (1, 1, 1, 0.75, 1))):
        data[:, 2] = z
        vertices.from_numpy(data.reshape(-1))
        monkeypatch.setattr(write, "_prepare_packet", unexpected_prepare)
        for _ in range(2):
            graph.submit(bound).wait()
            np.testing.assert_allclose(result.to_numpy(), expected, atol=1e-5)
            image = graphics_result.to_numpy()
            drawn = image[:, :, 1] > 0.5
            assert drawn.any() and (~drawn).any()
            np.testing.assert_allclose(
                image[drawn], np.tile((*expected[:3], 1), (drawn.sum(), 1)), atol=1e-5
            )
        monkeypatch.setattr(write, "_prepare_packet", prepare)
        replacement = ti.Texture(ti.Format.depth32f, (32, 24), sampler=config)
        bound.update(shadow=replacement)
    graph.close()
    writer.close()
    reader.close()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_shadow_cold_rejection_preserves_prepared_graph():
    gfx = ti.hardware.graphics
    vertices, _ = _vertices()
    writer = _pipeline(
        "hardware_graphics_depth_only.frag.spv", depth_test=True, depth_write=True
    )
    draw = writer.pass_draw(gfx.Draw(6), vertex_buffers={0: "vertices"})
    with pytest.raises(ValueError, match="depth binding"):
        writer.record_pass((draw,), colors=())
    with pytest.raises(RuntimeError, match="2D depth32f"):
        ti.Texture(
            ti.Format.r32f,
            (32, 24),
            sampler=ti.hardware.sampling.SamplerConfig(compare_op="less"),
        )

    output = ti.ndarray(ti.f32, 1)

    @ti.kernel
    def compare(shadow: ti.types.texture(2), out: ti.types.ndarray()):
        out[0] = shadow.sample_compare(ti.Vector([0.25, 0.5]), 0.5)

    recording = writer.record_pass((draw,), colors=(), depth="shadow", clear_depth=1)
    builder = ti.graph.GraphBuilder()
    builder.append_native(recording, admission="auto")
    builder.dispatch(
        compare,
        ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "shadow", ndim=2),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.f32, ndim=1),
    )
    from taichi_forge.graph._recipes.binding_frames import (
        GraphBindingFrameRecipeProvider,
    )
    from taichi_forge.graph._recipes.families import GraphRuntimeAssemblyProvider

    definition = builder.freeze()
    catalog = definition.recipe_catalog(
        providers=(GraphRuntimeAssemblyProvider(), GraphBindingFrameRecipeProvider())
    )
    recipe = next(entry.recipe for entry in catalog.entries() if entry.recipe.fragments)
    context = definition.materialization_context(provider_set=catalog.provider_set)
    materialized = context.materialize(recipe)
    graph = materialized.executor
    depth = ti.Texture(
        ti.Format.depth32f,
        (32, 24),
        sampler=ti.hardware.sampling.SamplerConfig(compare_op="less"),
    )
    bound = graph.bind(dict(shadow=depth, vertices=vertices, out=output))
    revision = bound.revision
    with pytest.raises(RuntimeError, match="compare_op"):
        bound.update(shadow=ti.Texture(ti.Format.depth32f, (32, 24)))
    assert bound.revision == revision
    graph.submit(bound).wait()
    np.testing.assert_array_equal(output.to_numpy(), [1])
    graph.close()
    materialized.close()
    context.close()
    writer.close()
    ti.reset()
    with pytest.raises(RuntimeError, match="closed|runtime|generation"):
        graph.run(bound)


@test_utils.test(arch=ti.vulkan, offline_cache=True)
def test_depth_comparison_ops_and_fetch_only_sampler_independence():
    gfx = ti.hardware.graphics
    vertices, _ = _vertices()
    writer = _pipeline(
        "hardware_graphics_depth_only.frag.spv", depth_test=True, depth_write=True
    )
    recording = writer.record_pass(
        (writer.pass_draw(gfx.Draw(6), vertex_buffers={0: "vertices"}),),
        colors=(),
        depth="shadow",
        clear_depth=0.5,
    )
    output = ti.ndarray(ti.f32, 3)

    @ti.kernel
    def compare(shadow: ti.types.texture(2), out: ti.types.ndarray()):
        out[0] = shadow.sample_compare(ti.Vector([0.25, 0.5]), 0.25)
        out[1] = shadow.sample_compare(ti.Vector([0.25, 0.5]), 0.5)
        out[2] = shadow.sample_compare(ti.Vector([0.25, 0.5]), 0.75)

    @ti.kernel
    def fetch(shadow: ti.types.texture(2), out: ti.types.ndarray()):
        for i in out:
            out[i] = shadow.fetch(ti.Vector([8, 12]), 0).x

    for op, expected in (
        ("never", (0, 0, 0)),
        ("less", (1, 0, 0)),
        ("equal", (0, 1, 0)),
        ("less_equal", (1, 1, 0)),
        ("greater", (0, 0, 1)),
        ("not_equal", (1, 0, 1)),
        ("greater_equal", (0, 1, 1)),
        ("always", (1, 1, 1)),
    ):
        shadow = ti.Texture(
            ti.Format.depth32f,
            (32, 24),
            sampler=ti.hardware.sampling.SamplerConfig(
                compare_op=op, min_filter="nearest", mag_filter="nearest"
            ),
        )
        prepared = recording.prepare_graph_execute(
            dict(shadow=shadow, vertices=vertices)
        )
        assert prepared() is shadow
        compare(shadow, output)
        np.testing.assert_array_equal(output.to_numpy(), expected)
        fetch(shadow, output)
        np.testing.assert_array_equal(output.to_numpy(), [0.5, 0.5, 0.5])
    writer.close()
