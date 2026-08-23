import gc
import re
import struct
from pathlib import Path

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.graph._ir import GraphAccess
from taichi_forge.lang import impl
from tests import test_utils


def _spirv_header(name):
    path = (
        Path(__file__).parents[2]
        / "cpp_examples"
        / "rhi_examples"
        / "shaders"
        / name
    )
    words = [int(value, 16) for value in re.findall(r"0x[0-9a-fA-F]+", path.read_text())]
    return struct.pack(f"<{len(words)}I", *words)


def _triangle_pipeline():
    return ti.hardware.graphics.VulkanGraphicsPipeline(
        _spirv_header("2_triangle.vert.spv.h"),
        _spirv_header("2_triangle.frag.spv.h"),
        vertex_bindings=(ti.hardware.graphics.VertexBinding(0, 20),),
        vertex_attributes=(
            ti.hardware.graphics.VertexAttribute(0, 0, ti.Format.rg32f, 0),
            ti.hardware.graphics.VertexAttribute(1, 0, ti.Format.rgb32f, 8),
        ),
    )


def _depth_triangle_pipeline(*, enabled):
    vertex_path = (
        Path(__file__).parent
        / "assets"
        / "hardware_graphics_depth.vert.spv.h"
    )
    words = [
        int(value, 16)
        for value in re.findall(r"0x[0-9a-fA-F]+", vertex_path.read_text())
    ]
    vertex_spirv = struct.pack(f"<{len(words)}I", *words)
    return ti.hardware.graphics.VulkanGraphicsPipeline(
        vertex_spirv,
        _spirv_header("2_triangle.frag.spv.h"),
        vertex_bindings=(ti.hardware.graphics.VertexBinding(0, 24),),
        vertex_attributes=(
            ti.hardware.graphics.VertexAttribute(0, 0, ti.Format.rgb32f, 0),
            ti.hardware.graphics.VertexAttribute(1, 0, ti.Format.rgb32f, 12),
        ),
        depth_test=enabled,
        depth_write=enabled,
    )


def _triangle_vertices():
    vertices = ti.ndarray(ti.f32, shape=(15,))
    vertices.from_numpy(
        np.array(
            [
                0.0,
                0.5,
                1.0,
                0.0,
                0.0,
                0.5,
                -0.5,
                0.0,
                1.0,
                0.0,
                -0.5,
                -0.5,
                0.0,
                0.0,
                1.0,
            ],
            dtype=np.float32,
        )
    )
    return vertices


def _overlapping_depth_vertices():
    vertices = ti.ndarray(ti.f32, shape=(36,))
    vertices.from_numpy(
        np.array(
            [
                -0.6,
                -0.6,
                0.75,
                0.0,
                1.0,
                0.0,
                0.6,
                -0.6,
                0.75,
                0.0,
                1.0,
                0.0,
                0.0,
                0.6,
                0.75,
                0.0,
                1.0,
                0.0,
                -0.6,
                -0.6,
                0.25,
                1.0,
                0.0,
                0.0,
                0.6,
                -0.6,
                0.25,
                1.0,
                0.0,
                0.0,
                0.0,
                0.6,
                0.25,
                1.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )
    )
    return vertices


@test_utils.test(arch=ti.cpu)
def test_vulkan_graphics_contract_rejects_non_vulkan_runtime():
    assert not ti.hardware.graphics.is_available()
    with pytest.raises(RuntimeError, match="requires the Vulkan backend"):
        _triangle_pipeline()
    with pytest.raises(ValueError, match="positive uint32"):
        ti.hardware.graphics.Draw(0)
    with pytest.raises(ValueError, match="minimum must not exceed maximum"):
        ti.hardware.graphics.Draw(3, index_bounds=(2, 1))
    with pytest.raises(ValueError, match="signed int32"):
        ti.hardware.graphics.Draw(3, vertex_offset=1 << 31)
    with pytest.raises(TypeError, match="ti.Format"):
        ti.hardware.graphics.VertexAttribute(0, 0, "rg32f")


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_graphics_draw_executes_directly_and_through_graph():
    if not ti.hardware.graphics.is_available():
        pytest.skip("Vulkan graphics commands are unavailable")

    pipeline = _triangle_pipeline()
    vertices = _triangle_vertices()
    color = ti.Texture(ti.Format.rgba8, (64, 64))
    draw = ti.hardware.graphics.Draw(3)

    assert pipeline.draw(color, {0: vertices}, draw=draw) is color
    ti.sync()
    image = np.asarray(color.to_image())
    assert image[32, 32].max() > 32
    assert image[2, 2].max() == 0

    recording = pipeline.record(
        draw,
        color="target",
        vertex_buffers={0: "vertices"},
        clear_color=(0.0, 0.0, 0.0, 1.0),
    )
    assert tuple(
        (effect.resource, effect.access) for effect in recording.resource_effects
    ) == (
        ("target", GraphAccess.WRITE),
        ("vertices", GraphAccess.READ),
    )
    builder = ti.graph.GraphBuilder()
    builder.append_native(recording, admission="auto")
    graph = builder.compile()
    graph.run({"target": color, "vertices": vertices})
    ti.sync()
    assert np.asarray(color.to_image())[32, 32].max() > 32
    assert graph._debug_info["optimization"]["backend_command_nodes"] == 1

    report = pipeline.memory_report()
    assert report.known_resident_requested_bytes == 0
    assert report.opaque_component_count == 1

    pipeline.close()
    with pytest.raises(RuntimeError, match="closed"):
        graph.run({"target": color, "vertices": vertices})


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_graphics_viewport_uses_offset_and_extent():
    if not ti.hardware.graphics.is_available():
        pytest.skip("Vulkan graphics commands are unavailable")

    with _triangle_pipeline() as pipeline:
        vertices = _triangle_vertices()
        color = ti.Texture(ti.Format.rgba8, (64, 64))
        draw = ti.hardware.graphics.Draw(3)

        pipeline.draw(
            color,
            {0: vertices},
            draw=draw,
            viewport=(16, 16, 32, 32),
        )
        ti.sync()
        image = np.asarray(color.to_image())
        assert image[32, 32].max() > 32
        assert image[8, 8].max() == 0

        with pytest.raises(RuntimeError, match="inside the color attachment"):
            pipeline.draw(
                color,
                {0: vertices},
                draw=draw,
                viewport=(48, 48, 32, 32),
            )


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_graphics_indexed_draw_uses_declared_bounds():
    if not ti.hardware.graphics.is_available():
        pytest.skip("Vulkan graphics commands are unavailable")

    with _triangle_pipeline() as pipeline:
        vertices = _triangle_vertices()
        indices = ti.ndarray(ti.u32, shape=(3,))
        indices.from_numpy(np.array([1, 2, 3], dtype=np.uint32))
        color = ti.Texture(ti.Format.rgba8, (64, 64))

        draw = ti.hardware.graphics.Draw(
            3,
            vertex_offset=-1,
            index_bounds=(1, 3),
        )
        pipeline.draw(color, {0: vertices}, index_buffer=indices, draw=draw)
        ti.sync()
        assert np.asarray(color.to_image())[32, 32].max() > 32

        with pytest.raises(RuntimeError, match="vertex binding 0 is too small"):
            pipeline.draw(
                color,
                {0: vertices},
                index_buffer=indices,
                draw=ti.hardware.graphics.Draw(3, index_bounds=(0, 99)),
            )

        signed_indices = ti.ndarray(ti.i32, shape=(3,))
        signed_indices.from_numpy(np.array([0, 1, 2], dtype=np.int32))
        with pytest.raises(RuntimeError, match="index buffer must use u32"):
            pipeline.draw(
                color,
                {0: vertices},
                index_buffer=signed_indices,
                draw=ti.hardware.graphics.Draw(3, index_bounds=(0, 2)),
            )

        with pytest.raises(ValueError, match="require declared index_bounds"):
            pipeline.record(
                ti.hardware.graphics.Draw(3),
                vertex_buffers={0: "vertices"},
                index_buffer="indices",
            )
        with pytest.raises(ValueError, match="index_bounds require"):
            pipeline.record(
                ti.hardware.graphics.Draw(3, index_bounds=(0, 2)),
                vertex_buffers={0: "vertices"},
            )


@test_utils.test(arch=ti.vulkan, offline_cache=False, debug=True)
def test_vulkan_graphics_depth_attachment_controls_visibility_and_lifetime():
    if not ti.hardware.graphics.is_available():
        pytest.skip("Vulkan graphics commands are unavailable")

    vertices = _overlapping_depth_vertices()
    draw = ti.hardware.graphics.Draw(6)

    with _depth_triangle_pipeline(enabled=False) as pipeline:
        color_without_depth = ti.Texture(ti.Format.rgba8, (64, 64))
        pipeline.draw(color_without_depth, {0: vertices}, draw=draw)
        ti.sync()
        without_depth = np.asarray(color_without_depth.to_image())[32, 32]
        assert without_depth[0] > without_depth[1]

    with _depth_triangle_pipeline(enabled=True) as pipeline:
        color_with_depth = ti.Texture(ti.Format.rgba8, (64, 64))
        depth = ti.Texture(ti.Format.depth32f, (64, 64))
        pipeline.draw(color_with_depth, {0: vertices}, depth=depth, draw=draw)
        del depth
        gc.collect()
        ti.sync()
        with_depth = np.asarray(color_with_depth.to_image())[32, 32]
        assert with_depth[1] > with_depth[0]


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_graphics_draw_validates_bindings_and_runtime_generation():
    if not ti.hardware.graphics.is_available():
        pytest.skip("Vulkan graphics commands are unavailable")

    pipeline = _triangle_pipeline()
    with pytest.raises(ValueError, match="exactly"):
        pipeline.record(
            ti.hardware.graphics.Draw(3),
            vertex_buffers={1: "vertices"},
        )
    recording = pipeline.record(
        ti.hardware.graphics.Draw(3), vertex_buffers={0: "vertices"}
    )
    with pytest.raises(RuntimeError, match="missing color"):
        recording.execute({"vertices": _triangle_vertices()})

    ti.reset()
    with pytest.raises(RuntimeError, match="previous Taichi runtime"):
        pipeline.record(ti.hardware.graphics.Draw(3), vertex_buffers={0: "vertices"})


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_graphics_pipeline_close_releases_program_resources():
    if not ti.hardware.graphics.is_available():
        pytest.skip("Vulkan graphics commands are unavailable")

    program = impl.get_runtime().prog
    baseline = program._debug_vulkan_graphics_pipeline_count()
    for _ in range(16):
        pipeline = _triangle_pipeline()
        assert program._debug_vulkan_graphics_pipeline_count() == baseline + 1
        pipeline.close()
        assert program._debug_vulkan_graphics_pipeline_count() == baseline
