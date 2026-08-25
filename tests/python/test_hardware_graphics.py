import gc
import re
import struct
from pathlib import Path

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.graph._ir import GraphAccess, ResourceEffect
from taichi_forge.lang import impl
from tests import test_utils
from tests.python.hardware_provider_lifecycle_qualification import (
    stress_iterations,
)
from tests.python.hardware_process_memory import ProcessMemoryPlateau


def _texture_rgb(texture):
    """Read an rgba8 texture without binding the hardware oracle to Pillow."""
    from taichi_forge._kernels import save_texture_to_numpy

    image = np.zeros(texture.shape + (3,), dtype=np.uint8)
    save_texture_to_numpy(texture, image)
    return np.rot90(image, 3)


def _spirv_header(name):
    path = (
        Path(__file__).parents[2] / "cpp_examples" / "rhi_examples" / "shaders" / name
    )
    words = [
        int(value, 16) for value in re.findall(r"0x[0-9a-fA-F]+", path.read_text())
    ]
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
        Path(__file__).parent / "assets" / "hardware_graphics_depth.vert.spv.h"
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


def _instanced_triangle_pipeline():
    vertex_path = (
        Path(__file__).parent / "assets" / "hardware_graphics_instanced.vert.spv.h"
    )
    words = [
        int(value, 16)
        for value in re.findall(r"0x[0-9a-fA-F]+", vertex_path.read_text())
    ]
    vertex_spirv = struct.pack(f"<{len(words)}I", *words)
    return ti.hardware.graphics.VulkanGraphicsPipeline(
        vertex_spirv,
        _spirv_header("2_triangle.frag.spv.h"),
        vertex_bindings=(
            ti.hardware.graphics.VertexBinding(0, 20),
            ti.hardware.graphics.VertexBinding(1, 8, instance=True),
        ),
        vertex_attributes=(
            ti.hardware.graphics.VertexAttribute(0, 0, ti.Format.rg32f, 0),
            ti.hardware.graphics.VertexAttribute(1, 0, ti.Format.rgb32f, 8),
            ti.hardware.graphics.VertexAttribute(2, 1, ti.Format.rg32f, 0),
        ),
    )


def _uniform_triangle_pipeline():
    shader_dir = Path(__file__).parents[2] / "python" / "taichi_forge" / "shaders"
    return ti.hardware.graphics.VulkanGraphicsPipeline(
        (shader_dir / "Triangles_vk_vert.spv").read_bytes(),
        (shader_dir / "Triangles_vk_frag.spv").read_bytes(),
        vertex_bindings=(ti.hardware.graphics.VertexBinding(0, 48),),
        vertex_attributes=(
            ti.hardware.graphics.VertexAttribute(0, 0, ti.Format.rgb32f, 0),
            ti.hardware.graphics.VertexAttribute(1, 0, ti.Format.rgb32f, 12),
            ti.hardware.graphics.VertexAttribute(2, 0, ti.Format.rg32f, 24),
            ti.hardware.graphics.VertexAttribute(3, 0, ti.Format.rgba32f, 32),
        ),
        shader_buffer_bindings=(
            ti.hardware.graphics.ShaderBufferBinding(0, 0, "uniform", "read"),
        ),
    )


def _storage_image_pipeline():
    shader_dir = Path(__file__).parents[2] / "python" / "taichi_forge" / "shaders"
    return ti.hardware.graphics.VulkanGraphicsPipeline(
        (shader_dir / "SetImage_vk_vert.spv").read_bytes(),
        (shader_dir / "SetImageBuffer_vk_frag.spv").read_bytes(),
        vertex_bindings=(ti.hardware.graphics.VertexBinding(0, 48),),
        vertex_attributes=(
            ti.hardware.graphics.VertexAttribute(0, 0, ti.Format.rgb32f, 0),
            ti.hardware.graphics.VertexAttribute(1, 0, ti.Format.rgb32f, 12),
            ti.hardware.graphics.VertexAttribute(2, 0, ti.Format.rg32f, 24),
            ti.hardware.graphics.VertexAttribute(3, 0, ti.Format.rgba32f, 32),
        ),
        shader_buffer_bindings=(
            ti.hardware.graphics.ShaderBufferBinding(0, 0, "storage", "read"),
            ti.hardware.graphics.ShaderBufferBinding(0, 1, "uniform", "read"),
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


def _two_triangle_vertices():
    vertices = ti.ndarray(ti.f32, shape=(30,))
    vertices.from_numpy(
        np.array(
            [
                -0.8,
                0.5,
                1.0,
                0.0,
                0.0,
                -0.1,
                -0.5,
                1.0,
                0.0,
                0.0,
                -0.8,
                -0.5,
                1.0,
                0.0,
                0.0,
                0.1,
                0.5,
                0.0,
                1.0,
                0.0,
                0.8,
                -0.5,
                0.0,
                1.0,
                0.0,
                0.1,
                -0.5,
                0.0,
                1.0,
                0.0,
            ],
            dtype=np.float32,
        )
    )
    return vertices


def _uniform_triangle_vertices():
    vertices = ti.ndarray(ti.f32, shape=(36,))
    vertices.from_numpy(
        np.array(
            [
                0.2,
                0.2,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.8,
                0.2,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.5,
                0.8,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            dtype=np.float32,
        )
    )
    return vertices


def _fullscreen_triangle_vertices():
    vertices = ti.ndarray(ti.f32, shape=(36,))
    values = np.zeros((3, 12), dtype=np.float32)
    values[:, :3] = np.array(
        [(-1.0, -1.0, 0.0), (3.0, -1.0, 0.0), (-1.0, 3.0, 0.0)],
        dtype=np.float32,
    )
    values[:, 6:8] = np.array([(0.0, 0.0), (2.0, 0.0), (0.0, 2.0)], dtype=np.float32)
    values[:, 8:12] = 1.0
    vertices.from_numpy(values.reshape(-1))
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
    with pytest.raises(ValueError, match="four-byte aligned"):
        ti.hardware.graphics.IndirectDraw(1, 3, command_offset=2)
    with pytest.raises(ValueError, match="four-byte aligned"):
        ti.hardware.graphics.IndirectDraw(1, 3, count_offset=2)
    with pytest.raises(ValueError, match="positive uint32"):
        ti.hardware.graphics.IndirectDraw(0, 3)


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
    image = _texture_rgb(color)
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
    graph_color = ti.Texture(ti.Format.rgba8, (64, 64))
    pipeline.draw(
        graph_color,
        {0: vertices},
        draw=draw,
        clear_color=(0.0, 0.0, 1.0, 1.0),
    )
    ti.sync()
    assert _texture_rgb(graph_color)[2, 2].max() > 32

    graph.run({"target": graph_color, "vertices": vertices})
    ti.sync()
    graph_image = _texture_rgb(graph_color)
    assert graph_image[32, 32].max() > 32
    assert graph_image[2, 2].max() == 0
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
        image = _texture_rgb(color)
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

    descriptor = ti.hardware.capability("raster.draw.vulkan")
    assert "index:u32" in descriptor.dtypes
    assert all("index:i32" not in dtype for dtype in descriptor.dtypes)

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
        assert _texture_rgb(color)[32, 32].max() > 32

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


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_graphics_instanced_draw_uses_all_vertex_bindings():
    if not ti.hardware.graphics.is_available():
        pytest.skip("Vulkan graphics commands are unavailable")

    with _instanced_triangle_pipeline() as pipeline:
        vertices = _triangle_vertices()
        offsets = ti.ndarray(ti.f32, shape=(6,))
        offsets.from_numpy(np.array([8.0, 8.0, -0.5, 0.0, 0.5, 0.0], dtype=np.float32))
        color = ti.Texture(ti.Format.rgba8, (64, 64))
        pipeline.draw(
            color,
            {0: vertices, 1: offsets},
            draw=ti.hardware.graphics.Draw(
                3,
                instance_count=2,
                first_instance=1,
            ),
        )
        ti.sync()
        image = _texture_rgb(color)
        assert image[32, 16].max() > 32
        assert image[32, 48].max() > 32
        assert image[2, 2].max() == 0


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
        without_depth = _texture_rgb(color_without_depth)[32, 32]
        assert without_depth[0] > without_depth[1]

    with _depth_triangle_pipeline(enabled=True) as pipeline:
        color_with_depth = ti.Texture(ti.Format.rgba8, (64, 64))
        depth = ti.Texture(ti.Format.depth32f, (64, 64))
        pipeline.draw(color_with_depth, {0: vertices}, depth=depth, draw=draw)
        del depth
        gc.collect()
        ti.sync()
        with_depth = _texture_rgb(color_with_depth)[32, 32]
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
    iterations = stress_iterations(16)
    process_memory = ProcessMemoryPlateau(
        "vulkan-graphics-pipeline-churn", ("vulkan-graphics",)
    )
    process_memory.capture("before")
    for iteration in range(iterations):
        pipeline = _triangle_pipeline()
        assert program._debug_vulkan_graphics_pipeline_count() == baseline + 1
        pipeline.close()
        assert program._debug_vulkan_graphics_pipeline_count() == baseline
        if iteration + 1 == max(1, iterations // 2):
            ti.sync()
            process_memory.capture("midpoint")
    ti.sync()
    process_memory.capture("after")
    process_memory.finish(iterations)


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_graphics_close_defers_inflight_resource_without_waiting():
    if not ti.hardware.graphics.is_available():
        pytest.skip("Vulkan graphics commands are unavailable")

    program = impl.get_runtime().prog
    ti.sync()
    baseline = dict(program._debug_vulkan_graphics_resource_stats())
    pipeline = _triangle_pipeline()
    color = ti.Texture(ti.Format.rgba8, (64, 64))
    vertices = _triangle_vertices()
    pipeline.draw(color, {0: vertices}, draw=ti.hardware.graphics.Draw(3))
    memory_before_close = program._runtime_statistics_snapshot()["memory"]
    waits_before = program._runtime_statistics_snapshot()["synchronization"][
        "backend_waits"
    ]

    pipeline.close()

    waits_after = program._runtime_statistics_snapshot()["synchronization"][
        "backend_waits"
    ]
    retiring = dict(program._debug_vulkan_graphics_resource_stats())
    hardware = ti.hardware.telemetry()
    runtime_memory = program._runtime_statistics_snapshot()["memory"]
    assert waits_after == waits_before
    assert retiring["live"] == baseline["live"]
    assert retiring["retiring"] == baseline["retiring"] + 1
    assert retiring["completion_retained"] >= 1
    assert runtime_memory["live_resources"] == (
        memory_before_close["live_resources"] - 1
    )
    assert runtime_memory["retiring_resources"] >= (
        memory_before_close["retiring_resources"] + 1
    )
    assert runtime_memory["inflight_resources"] >= (
        memory_before_close["inflight_resources"] + 1
    )
    assert hardware.resources["vulkan_graphics_pipeline"]["retiring"] >= 1
    assert hardware.runtime["physical_queue_counts_exact"]
    operation = hardware.operations["raster.draw.vulkan"]
    assert operation.recordings >= 1
    assert operation.executed >= 1
    assert operation.executed_backend_commands >= 1

    ti.sync()
    completed = dict(program._debug_vulkan_graphics_resource_stats())
    assert completed["live"] == baseline["live"]
    assert completed["retiring"] == baseline["retiring"]


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_graphics_pass_batches_multiple_draws_and_graph_nodes():
    if not ti.hardware.graphics.is_available():
        pytest.skip("Vulkan graphics commands are unavailable")

    with _triangle_pipeline() as pipeline:
        vertices = _two_triangle_vertices()
        indices = ti.ndarray(ti.u32, shape=(3,))
        indices.from_numpy(np.array([3, 4, 5], dtype=np.uint32))
        pass_draws = (
            pipeline.pass_draw(
                ti.hardware.graphics.Draw(3, first_vertex=0),
                vertex_buffers={0: "vertices"},
            ),
            pipeline.pass_draw(
                ti.hardware.graphics.Draw(3, index_bounds=(3, 5)),
                vertex_buffers={0: "vertices"},
                index_buffer="indices",
            ),
        )
        recording = pipeline.record_pass(
            pass_draws,
            color="target",
            clear_color=(0.0, 0.0, 0.0, 1.0),
        )
        assert recording.command_count == 1
        assert tuple(
            (effect.resource, effect.access) for effect in recording.resource_effects
        ) == (
            ("target", GraphAccess.WRITE),
            ("vertices", GraphAccess.READ),
            ("indices", GraphAccess.READ),
        )

        color = ti.Texture(ti.Format.rgba8, (64, 64))
        program = impl.get_runtime().prog
        ti.sync()
        queue_before = dict(program._debug_vulkan_queue_submission_stats())
        assert (
            recording.execute(
                {"target": color, "vertices": vertices, "indices": indices}
            )
            is color
        )
        ti.sync()
        queue_after = dict(program._debug_vulkan_queue_submission_stats())
        assert (
            queue_after["queue_submit_calls"] - queue_before["queue_submit_calls"] <= 3
        )
        assert (
            queue_after["submitted_command_buffers"]
            - queue_before["submitted_command_buffers"]
            <= 3
        )
        image = _texture_rgb(color)
        assert image[32, 52, 0] > 32
        assert image[32, 20, 1] > 32
        assert image[2, 2].max() == 0

        builder = ti.graph.GraphBuilder()
        builder.append_native(recording, admission="auto")
        graph = builder.compile()
        graph_color = ti.Texture(ti.Format.rgba8, (64, 64))
        graph.run({"target": graph_color, "vertices": vertices, "indices": indices})
        ti.sync()
        graph_image = _texture_rgb(graph_color)
        assert graph_image[32, 52, 0] > 32
        assert graph_image[32, 20, 1] > 32
        assert graph._debug_info["optimization"]["backend_command_nodes"] == 1


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_graphics_indirect_draw_abis_and_gpu_count_graph():
    if not ti.hardware.graphics.is_indirect_available():
        pytest.skip("Vulkan indirect graphics commands are unavailable")

    capabilities = dict(ti.hardware.graphics.indirect_capabilities())
    assert capabilities["fixed_count"]
    assert capabilities["max_draw_count"] >= 1

    with _triangle_pipeline() as pipeline:
        vertices = _triangle_vertices()
        command = ti.ndarray(ti.u32, shape=4)
        command.from_numpy(np.array([3, 1, 0, 0], dtype=np.uint32))
        draw = pipeline.pass_draw(
            ti.hardware.graphics.IndirectDraw(1, vertex_record_limit=3),
            vertex_buffers={0: "vertices"},
            indirect_buffer="commands",
        )
        recording = pipeline.record_pass((draw,), color="target")
        assert tuple(
            (effect.resource, effect.access) for effect in recording.resource_effects
        ) == (
            ("target", GraphAccess.WRITE),
            ("vertices", GraphAccess.READ),
            ("commands", GraphAccess.READ),
        )
        with pytest.raises(ValueError, match="smaller than the Vulkan"):
            pipeline.pass_draw(
                ti.hardware.graphics.IndirectDraw(
                    1,
                    vertex_record_limit=3,
                    command_stride=12,
                ),
                vertex_buffers={0: "vertices"},
                indirect_buffer="commands",
            )
        with pytest.raises(ValueError, match="indirect-count binding"):
            pipeline.pass_draw(
                ti.hardware.graphics.IndirectDraw(
                    1,
                    vertex_record_limit=3,
                    count_offset=0,
                ),
                vertex_buffers={0: "vertices"},
                indirect_buffer="commands",
            )
        color = ti.Texture(ti.Format.rgba8, (64, 64))
        with pytest.raises(RuntimeError, match="must use u32"):
            recording.execute(
                {
                    "target": color,
                    "vertices": vertices,
                    "commands": ti.ndarray(ti.f32, shape=4),
                }
            )
        with pytest.raises(RuntimeError, match="too small"):
            recording.execute(
                {
                    "target": color,
                    "vertices": vertices,
                    "commands": ti.ndarray(ti.u32, shape=3),
                }
            )
        with pytest.raises(RuntimeError, match="vertex binding 0 is too small"):
            recording.execute(
                {
                    "target": color,
                    "vertices": ti.ndarray(ti.f32, shape=10),
                    "commands": command,
                }
            )
        recording.execute({"target": color, "vertices": vertices, "commands": command})
        ti.sync()
        center = _texture_rgb(color)[32, 32]
        assert center.max() > 32

        indices = ti.ndarray(ti.u32, shape=3)
        indices.from_numpy(np.array([0, 1, 2], dtype=np.uint32))
        indexed_command = ti.ndarray(ti.u32, shape=5)
        indexed_command.from_numpy(np.array([3, 1, 0, 0, 0], dtype=np.uint32))
        indexed_draw = pipeline.pass_draw(
            ti.hardware.graphics.IndirectDraw(
                1,
                vertex_record_limit=3,
                index_element_limit=3,
            ),
            vertex_buffers={0: "vertices"},
            index_buffer="indices",
            indirect_buffer="commands",
        )
        indexed_recording = pipeline.record_pass((indexed_draw,), color="target")
        indexed_color = ti.Texture(ti.Format.rgba8, (64, 64))
        indexed_recording.execute(
            {
                "target": indexed_color,
                "vertices": vertices,
                "indices": indices,
                "commands": indexed_command,
            }
        )
        ti.sync()
        assert _texture_rgb(indexed_color)[32, 32].max() > 32

    if not ti.hardware.graphics.is_indirect_available(count_buffer=True):
        return

    @ti.kernel
    def publish_visible_draw(
        commands: ti.types.ndarray(dtype=ti.u32, ndim=1),
        count: ti.types.ndarray(dtype=ti.u32, ndim=1),
    ):
        commands[0] = 3
        commands[1] = 1
        commands[2] = 3
        commands[3] = 0
        count[0] = 1

    with _triangle_pipeline() as pipeline:
        vertices = _two_triangle_vertices()
        command = ti.ndarray(ti.u32, shape=4)
        count = ti.ndarray(ti.u32, shape=1)
        target = ti.Texture(ti.Format.rgba8, (64, 64))
        draw = pipeline.pass_draw(
            ti.hardware.graphics.IndirectDraw(
                1,
                vertex_record_limit=6,
                count_offset=0,
            ),
            vertex_buffers={0: "vertices"},
            indirect_buffer="commands",
            count_buffer="count",
        )
        recording = pipeline.record_pass((draw,), color="target")
        command_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "commands", ti.u32, ndim=1)
        count_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "count", ti.u32, ndim=1)
        builder = ti.graph.GraphBuilder()
        builder.dispatch(publish_visible_draw, command_arg, count_arg)
        builder.append_native(recording, admission="auto")
        graph = builder.compile()
        graph.run(
            {
                "target": target,
                "vertices": vertices,
                "commands": command,
                "count": count,
            }
        )
        ti.sync()
        image = _texture_rgb(target)
        assert image[32, 20, 1] > 32
        assert image[32, 52].max() == 0
        assert graph._debug_info["optimization"]["backend_command_nodes"] == 1


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_graphics_pass_load_and_depth_semantics():
    if not ti.hardware.graphics.is_available():
        pytest.skip("Vulkan graphics commands are unavailable")

    vertices = _two_triangle_vertices()
    with _triangle_pipeline() as pipeline:
        color = ti.Texture(ti.Format.rgba8, (64, 64))
        pipeline.draw(
            color,
            {0: vertices},
            draw=ti.hardware.graphics.Draw(3, first_vertex=0),
            clear_color=(0.0, 0.0, 1.0, 1.0),
        )
        load_recording = pipeline.record_pass(
            (
                pipeline.pass_draw(
                    ti.hardware.graphics.Draw(3, first_vertex=3),
                    vertex_buffers={0: "vertices"},
                ),
            ),
            color="target",
            color_load_op="load",
        )
        assert load_recording.resource_effects[0] == ResourceEffect(
            "target", GraphAccess.READ_WRITE
        )
        load_recording.execute({"target": color, "vertices": vertices})
        ti.sync()
        image = _texture_rgb(color)
        assert image[32, 52, 0] > 32
        assert image[32, 20, 1] > 32
        assert image[2, 2, 2] > 32

    depth_vertices = _overlapping_depth_vertices()
    with _depth_triangle_pipeline(enabled=True) as pipeline:
        color = ti.Texture(ti.Format.rgba8, (64, 64))
        depth = ti.Texture(ti.Format.depth32f, (64, 64))
        recording = pipeline.record_pass(
            (
                pipeline.pass_draw(
                    ti.hardware.graphics.Draw(3, first_vertex=0),
                    vertex_buffers={0: "vertices"},
                ),
                pipeline.pass_draw(
                    ti.hardware.graphics.Draw(3, first_vertex=3),
                    vertex_buffers={0: "vertices"},
                ),
            ),
            color="target",
            depth="depth",
        )
        recording.execute({"target": color, "depth": depth, "vertices": depth_vertices})
        ti.sync()
        center = _texture_rgb(color)[32, 32]
        assert center[1] > center[0]


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_graphics_pass_binds_real_uniform_descriptor():
    if not ti.hardware.graphics.is_available():
        pytest.skip("Vulkan graphics commands are unavailable")

    with _uniform_triangle_pipeline() as pipeline:
        vertices = _uniform_triangle_vertices()
        parameters = ti.ndarray(ti.f32, shape=(4,))
        parameters.from_numpy(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        color = ti.Texture(ti.Format.rgba8, (64, 64))
        draw = pipeline.pass_draw(
            ti.hardware.graphics.Draw(3),
            vertex_buffers={0: "vertices"},
            shader_buffers={(0, 0): "parameters"},
        )
        recording = pipeline.record_pass((draw,), color="target")
        assert tuple(
            (effect.resource, effect.access) for effect in recording.resource_effects
        ) == (
            ("target", GraphAccess.WRITE),
            ("vertices", GraphAccess.READ),
            ("parameters", GraphAccess.READ),
        )
        recording.execute(
            {"target": color, "vertices": vertices, "parameters": parameters}
        )
        ti.sync()
        center = _texture_rgb(color)[32, 32]
        assert center[0] > 32
        assert center[1] < 8
        assert center[2] < 8


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_graphics_pass_binds_real_storage_descriptor():
    if not ti.hardware.graphics.is_available():
        pytest.skip("Vulkan graphics commands are unavailable")

    with _storage_image_pipeline() as pipeline:
        vertices = _fullscreen_triangle_vertices()
        pixels = ti.ndarray(ti.u32, shape=(4,))
        pixels.from_numpy(
            np.array([0x000000FF, 0x0000FF00, 0x00FF0000, 0x000000FF], np.uint32)
        )
        parameters = ti.ndarray(ti.u32, shape=(12,))
        parameter_words = np.zeros(12, dtype=np.uint32)
        parameter_words[0:2] = np.array([0.0, 0.0], np.float32).view(np.uint32)
        parameter_words[2:4] = np.array([1.0, 1.0], np.float32).view(np.uint32)
        parameter_words[4:6] = np.array([1.0, 1.0], np.float32).view(np.uint32)
        parameter_words[6:9] = (0, 2, 2)
        parameters.from_numpy(parameter_words)
        color = ti.Texture(ti.Format.rgba8, (64, 64))
        draw = pipeline.pass_draw(
            ti.hardware.graphics.Draw(3),
            vertex_buffers={0: "vertices"},
            shader_buffers={(0, 0): "pixels", (0, 1): "parameters"},
        )
        recording = pipeline.record_pass((draw,), color="target")
        recording.execute(
            {
                "target": color,
                "vertices": vertices,
                "pixels": pixels,
                "parameters": parameters,
            }
        )
        ti.sync()
        center = _texture_rgb(color)[32, 32]
        assert center[0] < 8
        assert center[1] < 8
        assert center[2] > 32


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_graphics_pass_shader_buffer_effects_are_explicit():
    if not ti.hardware.graphics.is_available():
        pytest.skip("Vulkan graphics commands are unavailable")

    with ti.hardware.graphics.VulkanGraphicsPipeline(
        _spirv_header("2_triangle.vert.spv.h"),
        _spirv_header("2_triangle.frag.spv.h"),
        vertex_bindings=(ti.hardware.graphics.VertexBinding(0, 20),),
        vertex_attributes=(
            ti.hardware.graphics.VertexAttribute(0, 0, ti.Format.rg32f, 0),
            ti.hardware.graphics.VertexAttribute(1, 0, ti.Format.rgb32f, 8),
        ),
        shader_buffer_bindings=(
            ti.hardware.graphics.ShaderBufferBinding(0, 0, "uniform", "read"),
            ti.hardware.graphics.ShaderBufferBinding(0, 1, "storage", "read_write"),
        ),
    ) as pipeline:
        draw = pipeline.pass_draw(
            ti.hardware.graphics.Draw(3),
            vertex_buffers={0: "vertices"},
            shader_buffers={(0, 0): "parameters", (0, 1): "state"},
        )
        recording = pipeline.record_pass((draw,), color="target", color_load_op="load")
        assert tuple(
            (effect.resource, effect.access) for effect in recording.resource_effects
        ) == (
            ("target", GraphAccess.READ_WRITE),
            ("vertices", GraphAccess.READ),
            ("parameters", GraphAccess.READ),
            ("state", GraphAccess.READ_WRITE),
        )
        with pytest.raises(ValueError, match="only supports color store"):
            pipeline.record_pass((draw,), color_store_op="discard")
        with pytest.raises(ValueError, match="exactly"):
            pipeline.pass_draw(
                ti.hardware.graphics.Draw(3),
                vertex_buffers={0: "vertices"},
                shader_buffers={(0, 0): "parameters"},
            )
