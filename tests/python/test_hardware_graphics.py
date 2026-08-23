import re
import struct
from pathlib import Path

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.graph._ir import GraphAccess
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


@test_utils.test(arch=ti.cpu)
def test_vulkan_graphics_contract_rejects_non_vulkan_runtime():
    assert not ti.hardware.graphics.is_available()
    with pytest.raises(RuntimeError, match="requires the Vulkan backend"):
        _triangle_pipeline()
    with pytest.raises(ValueError, match="positive uint32"):
        ti.hardware.graphics.Draw(0)
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
