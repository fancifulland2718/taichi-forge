"""Typed MRT producer/consumer composition and cold attachment contracts."""

from pathlib import Path

import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils
from tests.python.test_hardware_graphics import _spirv_header


def _pipeline(**kwargs):
    return ti.hardware.graphics.VulkanGraphicsPipeline(
        _spirv_header("2_triangle.vert.spv.h"),
        (Path(__file__).parent / "assets/hardware_graphics_mrt.frag.spv").read_bytes(),
        vertex_bindings=(ti.hardware.graphics.VertexBinding(0, 20),),
        vertex_attributes=(
            ti.hardware.graphics.VertexAttribute(0, 0, ti.Format.rg32f, 0),
            ti.hardware.graphics.VertexAttribute(1, 0, ti.Format.rgb32f, 8),
        ),
        **kwargs,
    )


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_mrt_float_integer_outputs_graph_rebind_and_lifetime(monkeypatch):
    monkeypatch.setenv("TI_VULKAN_GRAPHICS_RETAINED_REPLAY_PROOF", "1")
    gfx = ti.hardware.graphics
    vertices = ti.ndarray(ti.f32, 15)
    image = ti.Texture(ti.Format.rgba32f, (32, 24))
    unsigned = ti.Texture(ti.Format.r32u, (32, 24))
    signed = ti.Texture(ti.Format.r32i, (32, 24))
    ids = ti.ndarray(ti.u32, (32, 24))
    primitives = ti.ndarray(ti.i32, (32, 24))
    colors = ti.Vector.ndarray(4, ti.f32, (32, 24))

    @ti.kernel
    def produce(v: ti.types.ndarray(ti.f32, ndim=1)):
        # One triangle, with constant HDR red/alpha supplied to the fragment stage.
        for i in range(3):
            v[i * 5] = -0.8
            v[i * 5 + 1] = -0.8
            if i == 1:
                v[i * 5] = 0.8
            if i == 2:
                v[i * 5 + 1] = 0.8
            v[i * 5 + 2] = 1.0
            v[i * 5 + 3] = 0.0
            v[i * 5 + 4] = 0.0

    @ti.kernel
    def consume(
        color: ti.types.texture(2),
        unsigned: ti.types.rw_texture(2, fmt=ti.Format.r32u, lod=0),
        signed: ti.types.rw_texture(2, fmt=ti.Format.r32i, lod=0),
        ids: ti.types.ndarray(ti.u32, ndim=2),
        primitives: ti.types.ndarray(ti.i32, ndim=2),
        colors: ti.types.ndarray(ti.f32, ndim=2, element_shape=(4,)),
    ):
        for i, j in ids:
            xy = ti.Vector([i, j])
            ids[i, j] = unsigned.load(xy).x
            primitives[i, j] = signed.load(xy).x
            colors[i, j] = color.fetch(xy, 0)

    pipeline = _pipeline(
        color_targets=(gfx.ColorTarget(), gfx.ColorTarget(), gfx.ColorTarget())
    )
    recording = pipeline.record_pass(
        (pipeline.pass_draw(gfx.Draw(3), vertex_buffers={0: "vertices"}),),
        colors=(
            gfx.ColorAttachment("color", clear_value=(0.25, 0.5, 0.75, 1)),
            gfx.ColorAttachment("unsigned", clear_value=(0xFEDCBA98, 0, 0, 0)),
            gfx.ColorAttachment("signed", clear_value=(-987654321, 0, 0, 0)),
        ),
    )
    arg = ti.graph.Arg
    kind = ti.graph.ArgKind
    builder = ti.graph.GraphBuilder()
    builder.dispatch(produce, arg(kind.NDARRAY, "vertices", ti.f32, ndim=1))
    builder.append_native(recording, admission="auto")
    builder.dispatch(
        consume,
        arg(kind.TEXTURE, "color", ndim=2),
        arg(kind.RWTEXTURE, "unsigned", ndim=2, fmt=ti.Format.r32u),
        arg(kind.RWTEXTURE, "signed", ndim=2, fmt=ti.Format.r32i),
        arg(kind.NDARRAY, "ids", ti.u32, ndim=2),
        arg(kind.NDARRAY, "primitives", ti.i32, ndim=2),
        arg(kind.NDARRAY, "colors", ti.types.vector(4, ti.f32), ndim=2),
    )
    graph = builder.compile()
    bindings = dict(
        vertices=vertices,
        color=image,
        unsigned=unsigned,
        signed=signed,
        ids=ids,
        primitives=primitives,
        colors=colors,
    )
    bound = graph.bind(bindings)
    for replace in (False, True):
        if replace:
            bindings["unsigned"] = ti.Texture(ti.Format.r32u, (32, 24))
            bound = graph.bind(bindings)
        graph.run(bound)
        graph.run(bound)
        ti.sync()
        actual = ids.to_numpy()
        drawn = actual == 0xF1234567
        assert drawn.any() and (~drawn).any()
        np.testing.assert_array_equal(actual[~drawn], 0xFEDCBA98)
        np.testing.assert_array_equal(primitives.to_numpy()[drawn], -123456789)
        np.testing.assert_array_equal(primitives.to_numpy()[~drawn], -987654321)
        np.testing.assert_allclose(
            colors.to_numpy()[drawn], np.tile([4, 0, 0, 0.5], (drawn.sum(), 1))
        )
        np.testing.assert_allclose(
            colors.to_numpy()[~drawn],
            np.tile([0.25, 0.5, 0.75, 1], ((~drawn).sum(), 1)),
        )
    graph.close()
    pipeline.close()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_mrt_attachment_types_aliases_blending_and_typed_clear_rejected_cold():
    gfx = ti.hardware.graphics
    vertices = ti.ndarray(ti.f32, 15)
    bindings = dict(
        vertices=vertices,
        color=ti.Texture(ti.Format.rgba32f, (4, 4)),
        unsigned=ti.Texture(ti.Format.r32u, (4, 4)),
        signed=ti.Texture(ti.Format.r32i, (4, 4)),
    )
    with _pipeline() as pipeline:
        draw = pipeline.pass_draw(gfx.Draw(3), vertex_buffers={0: "vertices"})
        attachments = (
            gfx.ColorAttachment("color"),
            gfx.ColorAttachment("unsigned"),
            gfx.ColorAttachment("signed"),
        )
        recording = pipeline.record_pass((draw,), colors=attachments)
        for replacement, message in (
            (bindings["color"], "alias"),
            (ti.Texture(ti.Format.r32f, (4, 4)), "numeric types"),
            (ti.Texture(ti.Format.r32u, (2, 4)), "shape"),
        ):
            with pytest.raises(RuntimeError, match=message):
                recording.prepare_graph_execute(dict(bindings, unsigned=replacement))
        for value in (0.5, 1 << 32, -1):
            bad = pipeline.record_pass(
                (draw,),
                colors=(
                    attachments[0],
                    gfx.ColorAttachment("unsigned", clear_value=(value, 0, 0, 0)),
                    attachments[2],
                ),
            )
            with pytest.raises(RuntimeError, match="[Ii]nteger"):
                bad.prepare_graph_execute(bindings)
        with pytest.raises(ValueError, match="distinct"):
            pipeline.record_pass((draw,), colors=(attachments[0], attachments[0]))
    with _pipeline(
        color_targets=(gfx.ColorTarget(), gfx.ColorTarget(True), gfx.ColorTarget())
    ) as pipeline:
        recording = pipeline.record_pass(
            (pipeline.pass_draw(gfx.Draw(3), vertex_buffers={0: "vertices"}),),
            colors=attachments,
        )
        with pytest.raises(RuntimeError, match="Blending"):
            recording.prepare_graph_execute(bindings)


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_mrt_per_target_write_mask_blend_and_load():
    gfx = ti.hardware.graphics
    vertices = ti.ndarray(ti.f32, 15)
    vertices.from_numpy(
        np.array(
            [-0.8, -0.8, 1, 0, 0, 0.8, -0.8, 1, 0, 0, -0.8, 0.8, 1, 0, 0], np.float32
        )
    )
    bindings = dict(
        vertices=vertices,
        color=ti.Texture(ti.Format.rgba32f, (32, 24)),
        unsigned=ti.Texture(ti.Format.r32u, (32, 24)),
        signed=ti.Texture(ti.Format.r32i, (32, 24)),
    )
    color_values = ti.Vector.ndarray(4, ti.f32, (32, 24))
    ids = ti.ndarray(ti.u32, (32, 24))

    @ti.kernel
    def read(
        color: ti.types.texture(2),
        unsigned: ti.types.rw_texture(2, fmt=ti.Format.r32u, lod=0),
        values: ti.types.ndarray(ti.f32, ndim=2, element_shape=(4,)),
        ids: ti.types.ndarray(ti.u32, ndim=2),
    ):
        for i, j in ids:
            values[i, j] = color.fetch(ti.Vector([i, j]), 0)
            ids[i, j] = unsigned.load(ti.Vector([i, j])).x

    with _pipeline(
        color_targets=(
            gfx.ColorTarget(True, 1),
            gfx.ColorTarget(False, 0),
            gfx.ColorTarget(),
        )
    ) as pipeline:
        draw = pipeline.pass_draw(gfx.Draw(3), vertex_buffers={0: "vertices"})
        for load, expected_red in (("clear", 2.125), ("load", 3.0625)):
            recording = pipeline.record_pass(
                (draw,),
                colors=(
                    gfx.ColorAttachment(
                        "color", load_op=load, clear_value=(0.25, 0.5, 0.75, 1)
                    ),
                    gfx.ColorAttachment(
                        "unsigned", load_op=load, clear_value=(0xFEDCBA98, 0, 0, 0)
                    ),
                    gfx.ColorAttachment("signed", load_op=load),
                ),
            )
            prepared = recording.prepare_graph_execute(bindings)
            prepared()
            read(bindings["color"], bindings["unsigned"], color_values, ids)
            ti.sync()
            image = color_values.to_numpy()
            drawn = image[:, :, 0] > 1
            assert drawn.any()
            np.testing.assert_allclose(
                image[drawn], np.tile([expected_red, 0.5, 0.75, 1], (drawn.sum(), 1))
            )
            np.testing.assert_array_equal(ids.to_numpy(), 0xFEDCBA98)
