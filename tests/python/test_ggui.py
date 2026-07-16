import gc
import os
import platform
import pathlib
import subprocess
import sys
import textwrap
import threading
import time
import weakref

import numpy as np
import pytest
from taichi_forge._lib import core as _ti_core

import taichi_forge as ti
from taichi_forge.lang import impl
from taichi_forge.lang.misc import is_arch_supported
from taichi_forge.ui.scene import (
    get_normals_field,
    normals_field_cache,
)
from taichi_forge.ui.staging_buffer import (
    _NUMPY_IMAGE_FIELD_CACHE_MAX_ENTRIES,
    _image_object_field_cache,
    clear_staging_caches,
    image_field_cache,
    image_packed_ndarray_cache,
    image_texture_cache,
    to_rgba8,
    to_rgba8_packed_ndarray,
)
from taichi_forge.ui.utils import get_field_info
from tests import test_utils
from tests.test_utils import verify_image

# FIXME: render(); get_image_buffer_as_numpy() loop does not actually redraw
RENDER_REPEAT = 5
# FIXME: enable ggui tests on ti.cpu backend. It's blocked by macos10.15
supported_archs = [ti.vulkan, ti.cuda, ti.metal]


@pytest.mark.skipif(
    os.environ.get("TI_RUN_HEADED_GGUI_TESTS") != "1",
    reason="requires an opt-in headed desktop session",
)
@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=[ti.vulkan])
def test_vulkan_async_kernel_and_present_queue_concurrency():
    width, height = 640, 360
    state = ti.field(dtype=ti.f32, shape=1 << 18)
    image = np.zeros((height, width, 4), dtype=np.uint8)

    @ti.kernel
    def producer_step():
        for i in state:
            state[i] = state[i] * 0.999 + 0.001

    producer_step()
    ti.sync()

    stop = threading.Event()
    worker_errors = []

    def producer():
        try:
            while not stop.is_set():
                producer_step()
        except BaseException as exc:
            worker_errors.append(exc)
            stop.set()

    worker = threading.Thread(target=producer, daemon=True)
    worker.start()
    window = None
    frame = 0
    try:
        window = ti.ui.Window(
            "Vulkan queue concurrency test",
            (width, height),
            vsync=False,
            fps_limit=65535,
            show_window=True,
        )
        canvas = window.get_canvas()
        duration = float(os.environ.get("TI_HEADED_GGUI_STRESS_SECONDS", "5"))
        deadline = time.perf_counter() + duration
        while window.running and not stop.is_set() and time.perf_counter() < deadline:
            image[..., 0] = frame & 0xFF
            image[..., 1] = (frame * 3) & 0xFF
            image[..., 2] = 96
            image[..., 3] = 255
            canvas.set_image(image)
            window.show()
            frame += 1
    finally:
        stop.set()
        worker.join(timeout=10)
        if window is not None:
            window.destroy()

    assert not worker.is_alive()
    assert frame > 0
    if worker_errors:
        raise worker_errors[0]
    ti.sync()


def _pack_rgba8_numpy_reference(src):
    if src.dtype == np.uint8:
        values = src
    else:
        values = np.clip(src, 0.0, 1.0) * 255.0
    if src.ndim == 2:
        c = values.astype(np.uint32)
        return c | (c << 8) | (c << 16) | np.uint32(0xFF000000)
    px = values.astype(np.uint32)
    channels = src.shape[2]
    r = px[..., 0]
    g = px[..., 1] if channels > 1 else 0
    b = px[..., 2] if channels > 2 else 0
    a = px[..., 3] if channels > 3 else np.uint32(0xFF)
    return r | (g << 8) | (b << 16) | (a << 24)


def _make_solid_packed_frame(width, height, rgba):
    image = ti.ndarray(ti.u32, shape=(width, height))
    r, g, b, a = rgba

    @ti.kernel
    def fill(img: ti.types.ndarray()):
        packed = (
            ti.cast(r, ti.u32)
            | (ti.cast(g, ti.u32) << 8)
            | (ti.cast(b, ti.u32) << 16)
            | (ti.cast(a, ti.u32) << 24)
        )
        for i, j in ti.ndrange(width, height):
            img[i, j] = packed

    fill(image)
    return ti.ui.DisplayFrame.from_packed_u32_ndarray(image)


def _assert_solid_rgba(rendered, rgba):
    expected = np.empty_like(rendered)
    expected[...] = np.array(rgba, dtype=np.float32) / 255.0
    np.testing.assert_allclose(rendered, expected, atol=1.0 / 255.0 + 1e-5)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_to_rgba8_numpy_no_helper_compile():
    image_field_cache.clear()
    cases = [
        np.array([[0, 127], [255, 3]], dtype=np.uint8),
        np.array(
            [[[255, 0, 8], [1, 2, 3]], [[4, 5, 6], [7, 8, 9]]],
            dtype=np.uint8,
        ),
        np.array(
            [[[-1.0, 0.5, 2.0, 0.25], [0.1, 0.2, 0.3, 1.0]]],
            dtype=np.float32,
        ),
    ]
    for image in cases:
        compiled = impl.get_runtime().get_num_compiled_functions()
        out = to_rgba8(image)
        assert impl.get_runtime().get_num_compiled_functions() == compiled
        np.testing.assert_array_equal(out, _pack_rgba8_numpy_reference(image))

    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[:, :, 0] = 7
    image[:, :, 1] = 13
    image[:, :, 2] = 29
    compiled = impl.get_runtime().get_num_compiled_functions()
    out = to_rgba8(image)
    assert impl.get_runtime().get_num_compiled_functions() == compiled
    np.testing.assert_array_equal(out, _pack_rgba8_numpy_reference(image))

    image = np.zeros((512, 512, 4), dtype=np.float32)
    image[:, :, 0] = 0.25
    image[:, :, 1] = 0.5
    image[:, :, 2] = 0.75
    image[:, :, 3] = 1.0
    compiled = impl.get_runtime().get_num_compiled_functions()
    out = to_rgba8(image)
    assert impl.get_runtime().get_num_compiled_functions() == compiled
    np.testing.assert_array_equal(out, _pack_rgba8_numpy_reference(image))


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_to_rgba8_small_taichi_image_no_helper_compile():
    image_field_cache.clear()

    scalar_np = np.array([[0.0, 0.25], [0.5, 1.25]], dtype=np.float32)
    scalar = ti.field(ti.f32, shape=scalar_np.shape)
    scalar.from_numpy(scalar_np)
    compiled = impl.get_runtime().get_num_compiled_functions()
    out = to_rgba8(scalar)
    assert impl.get_runtime().get_num_compiled_functions() == compiled
    np.testing.assert_array_equal(out, _pack_rgba8_numpy_reference(scalar_np))

    vector_np = np.array(
        [[[0.0, 0.5, 1.0], [1.2, -0.5, 0.25]]],
        dtype=np.float32,
    )
    vector = ti.Vector.field(3, ti.f32, shape=vector_np.shape[:2])
    vector.from_numpy(vector_np)
    compiled = impl.get_runtime().get_num_compiled_functions()
    out = to_rgba8(vector)
    assert impl.get_runtime().get_num_compiled_functions() == compiled
    np.testing.assert_array_equal(out, _pack_rgba8_numpy_reference(vector_np))

    ndarray_np = np.array(
        [[[1, 2, 3, 4], [255, 128, 64, 32]]],
        dtype=np.uint8,
    )
    arr = ti.Vector.ndarray(4, ti.u8, shape=ndarray_np.shape[:2])
    arr.from_numpy(ndarray_np)
    compiled = impl.get_runtime().get_num_compiled_functions()
    out = to_rgba8(arr)
    assert impl.get_runtime().get_num_compiled_functions() == compiled
    np.testing.assert_array_equal(out, _pack_rgba8_numpy_reference(ndarray_np))


@test_utils.test(arch=[ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_to_rgba8_packed_ndarray_taichi_image():
    image_packed_ndarray_cache.clear()

    vector_np = np.array(
        [
            [[0.0, 0.5, 1.0, 1.0], [1.2, -0.5, 0.25, 0.75]],
            [[0.1, 0.2, 0.3, 0.4], [0.9, 0.8, 0.7, 0.6]],
        ],
        dtype=np.float32,
    )
    vector = ti.Vector.field(4, ti.f32, shape=vector_np.shape[:2])
    vector.from_numpy(vector_np)

    packed = to_rgba8_packed_ndarray(vector)
    assert packed.dtype == ti.u32
    assert packed.shape == vector_np.shape[:2]
    np.testing.assert_array_equal(packed.to_numpy(), _pack_rgba8_numpy_reference(vector_np))
    assert to_rgba8_packed_ndarray(vector) is packed

    scalar_np = np.array([[0, 127], [255, 3]], dtype=np.uint8)
    scalar = ti.ndarray(ti.u8, shape=scalar_np.shape)
    scalar.from_numpy(scalar_np)
    packed_scalar = to_rgba8_packed_ndarray(scalar)
    np.testing.assert_array_equal(packed_scalar.to_numpy(), _pack_rgba8_numpy_reference(scalar_np))


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_geometry_2d():
    window = ti.ui.Window("test", (640, 480), show_window=False)
    canvas = window.get_canvas()

    # simple circles
    n_circles_0 = 10
    circle_positions_0 = ti.Vector.field(2, ti.f32, shape=n_circles_0)
    for i in range(n_circles_0):
        circle_positions_0[i] = ti.Vector([0.1, i * 0.1])

    # circles with per vertex colors
    n_circles_1 = 10
    circle_positions_1 = ti.Vector.field(2, ti.f32, shape=n_circles_1)
    circle_colors_1 = ti.Vector.field(3, ti.f32, shape=n_circles_1)
    for i in range(n_circles_0):
        circle_positions_1[i] = ti.Vector([0.2, i * 0.1])
        circle_colors_1[i] = ti.Vector([i * 0.1, 1.0 - i * 0.1, 0.5])

    # simple triangles
    n_triangles_0 = 10
    triangles_positions_0 = ti.Vector.field(2, ti.f32, shape=3 * n_triangles_0)
    for i in range(n_triangles_0):
        triangles_positions_0[3 * i] = ti.Vector([0.3, i * 0.1])
        triangles_positions_0[3 * i + 1] = ti.Vector([0.35, i * 0.1])
        triangles_positions_0[3 * i + 2] = ti.Vector([0.35, i * 0.1 + 0.05])

    # triangles with per vertex colors and indices
    triangles_positions_1 = ti.Vector.field(2, ti.f32, shape=4)
    triangles_colors_1 = ti.Vector.field(3, ti.f32, shape=4)
    triangles_positions_1[0] = ti.Vector([0.4, 0])
    triangles_positions_1[1] = ti.Vector([0.4, 1])
    triangles_positions_1[2] = ti.Vector([0.45, 0])
    triangles_positions_1[3] = ti.Vector([0.45, 1])
    triangles_colors_1[0] = ti.Vector([0, 0, 0])
    triangles_colors_1[1] = ti.Vector([1, 0, 0])
    triangles_colors_1[2] = ti.Vector([0, 1, 0])
    triangles_colors_1[3] = ti.Vector([1, 1, 0])
    triangle_indices_1 = ti.Vector.field(3, ti.i32, shape=2)
    triangle_indices_1[0] = ti.Vector([0, 1, 3])
    triangle_indices_1[1] = ti.Vector([0, 2, 3])

    # simple lines
    n_lines_0 = 10
    lines_positions_0 = ti.Vector.field(2, ti.f32, shape=2 * n_lines_0)
    for i in range(n_lines_0):
        lines_positions_0[2 * i] = ti.Vector([0.5, i * 0.1])
        lines_positions_0[2 * i + 1] = ti.Vector([0.5, i * 0.1 + 0.05])

    # lines with per vertex colors and indices
    lines_positions_1 = ti.Vector.field(2, ti.f32, shape=4)
    lines_colors_1 = ti.Vector.field(3, ti.f32, shape=4)
    lines_positions_1[0] = ti.Vector([0.6, 0])
    lines_positions_1[1] = ti.Vector([0.6, 1])
    lines_positions_1[2] = ti.Vector([0.65, 0])
    lines_positions_1[3] = ti.Vector([0.65, 1])
    lines_colors_1[0] = ti.Vector([0, 0, 0])
    lines_colors_1[1] = ti.Vector([1, 0, 0])
    lines_colors_1[2] = ti.Vector([0, 1, 0])
    lines_colors_1[3] = ti.Vector([1, 1, 0])
    lines_indices_1 = ti.Vector.field(2, ti.i32, shape=6)
    line_id = 0
    for i in range(4):
        for j in range(i + 1, 4):
            lines_indices_1[line_id] = ti.Vector([i, j])
            line_id += 1

    # circles with per vertex radius
    n_circles_2 = 10
    circle_positions_2 = ti.Vector.field(2, ti.f32, shape=n_circles_2)
    circle_radii_2 = ti.field(ti.f32, shape=n_circles_2)
    for i in range(n_circles_2):
        circle_positions_2[i] = ti.Vector([0.75, i * 0.1])
        circle_radii_2[i] = (i + 1) / n_circles_2 * 0.05

    def render():
        canvas.circles(circle_positions_0, radius=0.05, color=(1, 0, 0))

        canvas.circles(circle_positions_1, radius=0.05, per_vertex_color=circle_colors_1)

        canvas.triangles(triangles_positions_0, color=(0, 0, 1))

        canvas.triangles(
            triangles_positions_1,
            per_vertex_color=triangles_colors_1,
            indices=triangle_indices_1,
        )

        canvas.lines(lines_positions_0, width=0.01, color=(0, 1, 0))

        canvas.lines(
            lines_positions_1,
            width=0.01,
            per_vertex_color=lines_colors_1,
            indices=lines_indices_1,
        )

        canvas.circles(circle_positions_2, radius=0.05, color=(0, 0, 1), per_vertex_radius=circle_radii_2)

    # Render in off-line mode to check if there are errors
    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()
    verify_image(window.get_image_buffer_as_numpy(), "test_geometry_2d", tolerance=0.05)
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_geometry_3d():
    window = ti.ui.Window("test", (640, 480), show_window=False)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0.0, 0.0, 1.5)
    camera.lookat(0.0, 0.0, 0)
    scene.set_camera(camera)

    # simple particles
    num_per_dim = 32
    num_particles_0 = int(num_per_dim**3)
    particles_positions_0 = ti.Vector.field(3, ti.f32, shape=num_particles_0)

    @ti.kernel
    def init_particles_0():
        for x, y, z in ti.ndrange(num_per_dim, num_per_dim, num_per_dim):
            i = x * (num_per_dim**2) + y * num_per_dim + z
            gap = 0.01
            particles_positions_0[i] = ti.Vector([-0.4, 0, 0.0], dt=ti.f32) + ti.Vector([x, y, z], dt=ti.f32) * gap

    init_particles_0()

    # particles with individual colors
    num_per_dim = 32
    num_particles_1 = int(num_per_dim**3)
    particles_positions_1 = ti.Vector.field(3, ti.f32, shape=num_particles_1)
    particles_colors_1 = ti.Vector.field(3, ti.f32, shape=num_particles_1)

    @ti.kernel
    def init_particles_1():
        for x, y, z in ti.ndrange(num_per_dim, num_per_dim, num_per_dim):
            i = x * (num_per_dim**2) + y * num_per_dim + z
            gap = 0.01
            particles_positions_1[i] = ti.Vector([0.2, 0, 0.0], dt=ti.f32) + ti.Vector([x, y, z], dt=ti.f32) * gap
            particles_colors_1[i] = ti.Vector([x, y, z], dt=ti.f32) / num_per_dim

    init_particles_1()

    # mesh
    vertices = ti.Vector.field(3, ti.f32, shape=8)
    colors = ti.Vector.field(3, ti.f32, shape=8)

    @ti.kernel
    def init_mesh():
        for i, j, k in ti.ndrange(2, 2, 2):
            index = i * 4 + j * 2 + k
            vertices[index] = ti.Vector([-0.1, -0.3, 0.0], dt=ti.f32) + ti.Vector([i, j, k], dt=ti.f32) * 0.25
            colors[index] = ti.Vector([i, j, k], dt=ti.f32)

    init_mesh()
    indices = ti.field(ti.i32, shape=36)
    indices_np = np.array(
        [
            0,
            1,
            2,
            3,
            1,
            2,
            4,
            5,
            6,
            7,
            5,
            6,
            0,
            1,
            4,
            5,
            1,
            4,
            2,
            3,
            6,
            7,
            3,
            6,
            0,
            2,
            4,
            6,
            2,
            4,
            1,
            3,
            5,
            7,
            3,
            5,
        ],
        dtype=np.int32,
    )
    indices.from_numpy(indices_np)

    def render():
        scene.point_light(pos=(2, 2, 2), color=(1, 1, 1))

        scene.particles(particles_positions_0, radius=0.01, color=(0.5, 0, 0))

        scene.particles(particles_positions_1, radius=0.01, per_vertex_color=particles_colors_1)

        scene.mesh(vertices, per_vertex_color=colors, indices=indices, two_sided=True)

        canvas.scene(scene)

    # Render in off-line mode to check if there are errors
    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()
    verify_image(window.get_image_buffer_as_numpy(), "test_geometry_3d")
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_geometry_3d_old():
    window = ti.ui.Window("test", (640, 480), show_window=False)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(0.0, 0.0, 1.5)
    camera.lookat(0.0, 0.0, 0)
    scene.set_camera(camera)

    # simple particles
    num_per_dim = 32
    num_particles_0 = int(num_per_dim**3)
    particles_positions_0 = ti.Vector.field(3, ti.f32, shape=num_particles_0)

    @ti.kernel
    def init_particles_0():
        for x, y, z in ti.ndrange(num_per_dim, num_per_dim, num_per_dim):
            i = x * (num_per_dim**2) + y * num_per_dim + z
            gap = 0.01
            particles_positions_0[i] = ti.Vector([-0.4, 0, 0.0], dt=ti.f32) + ti.Vector([x, y, z], dt=ti.f32) * gap

    init_particles_0()

    # particles with individual colors
    num_per_dim = 32
    num_particles_1 = int(num_per_dim**3)
    particles_positions_1 = ti.Vector.field(3, ti.f32, shape=num_particles_1)
    particles_colors_1 = ti.Vector.field(3, ti.f32, shape=num_particles_1)

    @ti.kernel
    def init_particles_1():
        for x, y, z in ti.ndrange(num_per_dim, num_per_dim, num_per_dim):
            i = x * (num_per_dim**2) + y * num_per_dim + z
            gap = 0.01
            particles_positions_1[i] = ti.Vector([0.2, 0, 0.0], dt=ti.f32) + ti.Vector([x, y, z], dt=ti.f32) * gap
            particles_colors_1[i] = ti.Vector([x, y, z], dt=ti.f32) / num_per_dim

    init_particles_1()

    # mesh
    vertices = ti.Vector.field(3, ti.f32, shape=8)
    colors = ti.Vector.field(3, ti.f32, shape=8)

    @ti.kernel
    def init_mesh():
        for i, j, k in ti.ndrange(2, 2, 2):
            index = i * 4 + j * 2 + k
            vertices[index] = ti.Vector([-0.1, -0.3, 0.0], dt=ti.f32) + ti.Vector([i, j, k], dt=ti.f32) * 0.25
            colors[index] = ti.Vector([i, j, k], dt=ti.f32)

    init_mesh()
    indices = ti.field(ti.i32, shape=36)
    indices_np = np.array(
        [
            0,
            1,
            2,
            3,
            1,
            2,
            4,
            5,
            6,
            7,
            5,
            6,
            0,
            1,
            4,
            5,
            1,
            4,
            2,
            3,
            6,
            7,
            3,
            6,
            0,
            2,
            4,
            6,
            2,
            4,
            1,
            3,
            5,
            7,
            3,
            5,
        ],
        dtype=np.int32,
    )
    indices.from_numpy(indices_np)

    def render():
        scene.point_light(pos=(2, 2, 2), color=(1, 1, 1))

        scene.particles(particles_positions_0, radius=0.01, color=(0.5, 0, 0))

        scene.particles(particles_positions_1, radius=0.01, per_vertex_color=particles_colors_1)

        scene.mesh(vertices, per_vertex_color=colors, indices=indices, two_sided=True)

        canvas.scene(scene)

    # Render in off-line mode to check if there are errors
    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()
    verify_image(window.get_image_buffer_as_numpy(), "test_geometry_3d")
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_draw_3d_lines():
    N = 4
    CAMERA_POS = ti.Vector([-3, 2.5, -4])

    coor_pos = ti.Vector.field(3, dtype=ti.f32, shape=N * N * N)
    colors = ti.Vector.field(3, dtype=ti.f32, shape=N * N * N)
    coor_idx = ti.ndarray(dtype=ti.i32, shape=N * N * N * 2)  # not works

    # Inverse of Part1By2 - "delete" all bits not at positions divisible by 3
    @ti.func
    def compact_1by2(input: ti.u32):
        x = input & 0x09249249  # x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
        x = (x ^ (x >> 2)) & 0x030C30C3  # x = ---- --98 ---- 76-- --54 ---- 32-- --10
        x = (x ^ (x >> 4)) & 0x0300F00F  # x = ---- --98 ---- ---- 7654 ---- ---- 3210
        x = (x ^ (x >> 8)) & 0x7F0000FF  # x = ---- --98 ---- ---- ---- ---- 7654 3210
        x = (x ^ (x >> 16)) & 0x000003FF  # x = ---- ---- ---- ---- ---- --98 7654 3210
        return x

    @ti.func
    def decode_morton(code: ti.u32):
        return ti.Vector([compact_1by2(code >> 0), compact_1by2(code >> 1), compact_1by2(code >> 2)])

    @ti.kernel
    def init_coordinates(coor_idx: ti.types.ndarray()):
        for i, j, k in ti.ndrange(N, N, N):
            idx = i * N * N + j * N + k
            coor_pos[idx] = ti.Vector([i, j, k])
            fpos = ti.cast(ti.Vector([i, j, k]), ti.f32) + 0.01
            colors[idx] = fpos.normalized() * (0.1 + 2.0 / (ti.math.distance(fpos, CAMERA_POS) - 3.0))
        for i in ti.ndrange(N * N * N - 1):
            ipos0 = decode_morton(i)
            lindex0 = ipos0.x * N * N + ipos0.y * N + ipos0.z
            ipos1 = decode_morton(i + 1)
            lindex1 = ipos1.x * N * N + ipos1.y * N + ipos1.z
            coor_idx[i * 2 + 0] = lindex0
            coor_idx[i * 2 + 1] = lindex1

    window = ti.ui.Window("Test for Drawing 3d-lines", (512, 512), show_window=False)
    canvas = window.get_canvas()
    scene = window.get_scene()

    def render():
        init_coordinates(coor_idx)
        camera = ti.ui.Camera()
        camera.position(CAMERA_POS.x, CAMERA_POS.y, CAMERA_POS.z)
        camera.lookat(2, 1, 2)
        scene.set_camera(camera)
        scene.lines(coor_pos, indices=coor_idx, per_vertex_color=colors, width=3.0)
        canvas.scene(scene)

    render()
    verify_image(window.get_image_buffer_as_numpy(), "test_draw_3d_lines")
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_draw_3d_lines_old():
    N = 4
    CAMERA_POS = ti.Vector([-3, 2.5, -4])

    coor_pos = ti.Vector.field(3, dtype=ti.f32, shape=N * N * N)
    colors = ti.Vector.field(3, dtype=ti.f32, shape=N * N * N)
    coor_idx = ti.ndarray(dtype=ti.i32, shape=N * N * N * 2)  # not works

    # Inverse of Part1By2 - "delete" all bits not at positions divisible by 3
    @ti.func
    def compact_1by2(input: ti.u32):
        x = input & 0x09249249  # x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
        x = (x ^ (x >> 2)) & 0x030C30C3  # x = ---- --98 ---- 76-- --54 ---- 32-- --10
        x = (x ^ (x >> 4)) & 0x0300F00F  # x = ---- --98 ---- ---- 7654 ---- ---- 3210
        x = (x ^ (x >> 8)) & 0x7F0000FF  # x = ---- --98 ---- ---- ---- ---- 7654 3210
        x = (x ^ (x >> 16)) & 0x000003FF  # x = ---- ---- ---- ---- ---- --98 7654 3210
        return x

    @ti.func
    def decode_morton(code: ti.u32):
        return ti.Vector([compact_1by2(code >> 0), compact_1by2(code >> 1), compact_1by2(code >> 2)])

    @ti.kernel
    def init_coordinates(coor_idx: ti.types.ndarray()):
        for i, j, k in ti.ndrange(N, N, N):
            idx = i * N * N + j * N + k
            coor_pos[idx] = ti.Vector([i, j, k])
            fpos = ti.cast(ti.Vector([i, j, k]), ti.f32) + 0.01
            colors[idx] = fpos.normalized() * (0.1 + 2.0 / (ti.math.distance(fpos, CAMERA_POS) - 3.0))
        for i in ti.ndrange(N * N * N - 1):
            ipos0 = decode_morton(i)
            lindex0 = ipos0.x * N * N + ipos0.y * N + ipos0.z
            ipos1 = decode_morton(i + 1)
            lindex1 = ipos1.x * N * N + ipos1.y * N + ipos1.z
            coor_idx[i * 2 + 0] = lindex0
            coor_idx[i * 2 + 1] = lindex1

    window = ti.ui.Window("Test for Drawing 3d-lines", (512, 512), show_window=False)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()

    def render():
        init_coordinates(coor_idx)
        camera = ti.ui.Camera()
        camera.position(CAMERA_POS.x, CAMERA_POS.y, CAMERA_POS.z)
        camera.lookat(2, 1, 2)
        scene.set_camera(camera)
        scene.lines(coor_pos, indices=coor_idx, per_vertex_color=colors, width=3.0)
        canvas.scene(scene)

    render()
    verify_image(window.get_image_buffer_as_numpy(), "test_draw_3d_lines")
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_set_image():
    window = ti.ui.Window("test", (640, 480), show_window=False)
    canvas = window.get_canvas()

    img = ti.Vector.field(4, ti.f32, (512, 512))

    @ti.kernel
    def init_img():
        for i, j in img:
            img[i, j] = ti.Vector([i, j, 0, 512], dt=ti.f32) / 512

    init_img()

    def render():
        canvas.set_image(img)

    # Render in off-line mode to check if there are errors
    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()

    verify_image(window.get_image_buffer_as_numpy(), "test_set_image")
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=[ti.vulkan])
def test_hidden_window_show_after_set_image():
    window = ti.ui.Window("test", (64, 64), show_window=False)
    canvas = window.get_canvas()

    img = ti.Vector.field(4, ti.f32, (64, 64))

    @ti.kernel
    def init_img():
        for i, j in img:
            img[i, j] = ti.Vector([i, j, 0, 64], dt=ti.f32) / 64

    init_img()
    canvas.set_image(img)
    window.show()
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(
    arch=supported_archs + [ti.cpu],
    exclude=[(ti.cpu, "Darwin")],
)
def test_hidden_window_display_stats_after_set_image():
    window = ti.ui.Window("test", (64, 64), show_window=False)
    canvas = window.get_canvas()
    prog = impl.get_runtime().prog

    img = ti.Vector.field(4, ti.f32, (64, 64))

    @ti.kernel
    def init_img():
        for i, j in img:
            img[i, j] = ti.Vector([i, j, 0, 64], dt=ti.f32) / 64

    init_img()
    window.reset_display_stats()
    runtime_before = prog._runtime_statistics_snapshot()["display"]
    assert window.is_headless_display() is True
    assert canvas.set_image(img) is True
    stats = window.get_display_stats()
    runtime_after_accept = prog._runtime_statistics_snapshot()["display"]
    assert stats["display_mode"] == "offscreen"
    assert stats["headless"] is True
    assert stats["accepted_frames"] == 1
    assert stats["submitted_frames"] == 0
    assert stats["window_submitted_frames"] == 0
    assert stats["offscreen_submitted_frames"] == 0
    assert stats["last_accepted"] is True
    assert (
        runtime_after_accept["accepted_frames"]
        - runtime_before["accepted_frames"]
        == stats["accepted_frames"]
    )
    assert (
        runtime_after_accept["submitted_frames"]
        - runtime_before["submitted_frames"]
        == stats["submitted_frames"]
    )
    assert (
        runtime_after_accept["accepted_frame_bytes"]
        - runtime_before["accepted_frame_bytes"]
        == 64 * 64 * 4
    )

    # Headless windows always accept another frame, so exercise the same
    # dropped-frame callback Canvas uses under headed backpressure directly.
    window.window.record_display_frame_dropped()
    stats = window.get_display_stats()
    runtime_after_drop = prog._runtime_statistics_snapshot()["display"]
    assert stats["dropped_frames"] == 1
    assert (
        runtime_after_drop["dropped_frames"]
        - runtime_before["dropped_frames"]
        == stats["dropped_frames"]
    )

    assert window.show() is True
    stats = window.get_display_stats()
    runtime_after_show = prog._runtime_statistics_snapshot()["display"]
    assert stats["accepted_frames"] == 1
    assert stats["submitted_frames"] == 1
    assert stats["window_submitted_frames"] == 0
    assert stats["offscreen_submitted_frames"] == 1
    assert stats["dropped_frames"] == 1
    assert stats["last_submitted"] is True
    assert stats["last_window_submitted"] is False
    assert stats["last_offscreen_submitted"] is True
    assert (
        runtime_after_show["accepted_frames"]
        - runtime_before["accepted_frames"]
        == stats["accepted_frames"]
    )
    assert (
        runtime_after_show["submitted_frames"]
        - runtime_before["submitted_frames"]
        == stats["submitted_frames"]
    )
    assert (
        runtime_after_show["dropped_frames"]
        - runtime_before["dropped_frames"]
        == stats["dropped_frames"]
    )
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=[ti.cuda, ti.vulkan])
def test_display_frame_ndarray_survives_gc_until_frame_completion():
    window = ti.ui.Window("test", (16, 16), show_window=False)
    canvas = window.get_canvas()
    prog = impl.get_runtime().prog
    baseline = prog._debug_ndarray_resource_stats()

    image = ti.ndarray(ti.u32, shape=(16, 16))
    image.fill(0xFF332211)
    frame = ti.ui.DisplayFrame.from_packed_u32_ndarray(image)
    assert canvas.submit_frame(frame) is True

    submitted = prog._debug_ndarray_resource_stats()
    assert submitted["views"] == baseline["views"] + 1
    assert submitted["leases"] == baseline["leases"] + 2

    image_ref = weakref.ref(image)
    del frame, image
    gc.collect()
    assert image_ref() is None
    retired = prog._debug_ndarray_resource_stats()
    assert retired["views"] == baseline["views"]
    assert retired["retiring"] == baseline["retiring"] + 1

    assert window.show() is True
    window.destroy()
    ti.sync()
    completed = prog._debug_ndarray_resource_stats()
    for key in ("live", "retiring", "leases", "views", "inflight"):
        assert completed[key] == baseline[key]


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=[ti.vulkan])
def test_ggui_rejects_mismatched_ndarray_identity_and_allocation():
    window = ti.ui.Window("test", (4, 4), show_window=False)
    first = ti.ndarray(ti.u32, shape=(4, 4))
    second = ti.ndarray(ti.u32, shape=(4, 4))
    info = get_field_info(first)
    info.dev_alloc = second.arr.device_allocation()

    with pytest.raises(
        RuntimeError,
        match="GGUI Ndarray identity does not match its DeviceAllocation",
    ):
        window.get_canvas().canvas.set_image(info)
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=[ti.vulkan])
def test_display_frame_texture_survives_gc_until_frame_completion():
    window = ti.ui.Window("test", (16, 16), show_window=False)
    canvas = window.get_canvas()
    prog = impl.get_runtime().prog
    baseline = prog._debug_texture_resource_stats()

    texture = ti.Texture(ti.Format.rgba8, (16, 16))
    frame = ti.ui.DisplayFrame.from_texture(texture)
    assert canvas.submit_frame(frame) is True

    submitted = prog._debug_texture_resource_stats()
    assert submitted["views"] == baseline["views"] + 1
    assert submitted["leases"] == baseline["leases"] + 2

    texture_ref = weakref.ref(texture)
    del frame, texture
    gc.collect()
    assert texture_ref() is None
    retired = prog._debug_texture_resource_stats()
    assert retired["views"] == baseline["views"]
    assert retired["retiring"] == baseline["retiring"] + 1

    assert window.show() is True
    window.destroy()
    ti.sync()
    completed = prog._debug_texture_resource_stats()
    for key in ("live", "retiring", "leases", "views", "inflight"):
        assert completed[key] == baseline[key]


@test_utils.test(arch=[ti.vulkan])
def test_staging_cache_uses_weak_source_ownership():
    clear_staging_caches()
    image = ti.Vector.ndarray(4, ti.f32, shape=(2, 2))
    image.fill([0.0, 0.25, 0.5, 1.0])
    assert to_rgba8_packed_ndarray(image) is not None
    assert len(image_packed_ndarray_cache) == 1

    image_ref = weakref.ref(image)
    del image
    gc.collect()
    assert image_ref() is None
    assert len(image_packed_ndarray_cache) == 0


@test_utils.test(arch=[ti.cpu])
def test_generated_normals_cache_uses_weak_vertex_ownership():
    normals_field_cache.clear()
    vertices = ti.Vector.field(3, ti.f32, shape=3)
    normals, weights = get_normals_field(vertices)
    assert normals.shape == (3,)
    assert weights.shape == (3,)
    assert len(normals_field_cache) == 1

    vertices_ref = weakref.ref(vertices)
    del vertices, normals, weights
    # Materialization consumes PyTaichi's temporary field-validation refs.
    impl.get_runtime().materialize()
    gc.collect()
    assert vertices_ref() is None
    assert len(normals_field_cache) == 0


def test_numpy_image_staging_cache_is_bounded_lru():
    image_field_cache.clear()
    for width in range(1, _NUMPY_IMAGE_FIELD_CACHE_MAX_ENTRIES + 2):
        to_rgba8(np.zeros((width, 2, 4), dtype=np.uint8))

    assert len(image_field_cache) == _NUMPY_IMAGE_FIELD_CACHE_MAX_ENTRIES
    assert (1, 2) not in image_field_cache
    assert (_NUMPY_IMAGE_FIELD_CACHE_MAX_ENTRIES + 1, 2) in image_field_cache


@test_utils.test(arch=[ti.vulkan])
def test_reset_clears_all_staging_cache_generations():
    clear_staging_caches()
    image = ti.Vector.ndarray(4, ti.f32, shape=(2, 2))
    vertices = ti.Vector.field(3, ti.f32, shape=3)
    get_normals_field(vertices)
    image.fill([0.0, 0.25, 0.5, 1.0])
    assert to_rgba8_packed_ndarray(image) is not None
    assert len(image_packed_ndarray_cache) == 1
    assert len(normals_field_cache) == 1

    ti.reset()
    assert not image_field_cache
    assert not _image_object_field_cache
    assert not image_texture_cache
    assert not image_packed_ndarray_cache
    assert not normals_field_cache


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_set_image_numpy_rgba8_direct():
    window = ti.ui.Window("test", (64, 64), show_window=False)
    canvas = window.get_canvas()

    x = np.arange(64, dtype=np.uint8)[:, None]
    y = np.arange(64, dtype=np.uint8)[None, :]
    image = np.empty((64, 64, 4), dtype=np.uint8)
    image[..., 0] = x
    image[..., 1] = y
    image[..., 2] = np.uint8(0)
    image[..., 3] = np.uint8(255)

    assert canvas.set_image(image) is True
    rendered = window.get_image_buffer_as_numpy()
    expected = image.astype(np.float32) / 255.0
    np.testing.assert_allclose(rendered, expected, atol=1.0 / 255.0 + 1e-5)
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_set_image_display_frame_host_rgba8():
    window = ti.ui.Window("test", (64, 64), show_window=False)
    canvas = window.get_canvas()

    x = np.arange(64, dtype=np.uint8)[:, None]
    y = np.arange(64, dtype=np.uint8)[None, :]
    image = np.empty((64, 64, 4), dtype=np.uint8)
    image[..., 0] = x
    image[..., 1] = y
    image[..., 2] = np.uint8(0)
    image[..., 3] = np.uint8(255)

    frame = ti.ui.DisplayFrame.from_numpy_rgba8(image)
    window.reset_display_stats()
    assert canvas.submit_frame(frame) is True
    stats = window.get_display_stats()
    assert stats["accepted_frames"] == 1
    assert stats["last_accepted"] is True

    rendered = window.get_image_buffer_as_numpy()
    expected = image.astype(np.float32) / 255.0
    np.testing.assert_allclose(rendered, expected, atol=1.0 / 255.0 + 1e-5)
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_set_image_display_frame_packed_u32_ndarray():
    window = ti.ui.Window("test", (64, 64), show_window=False)
    canvas = window.get_canvas()
    image = ti.ndarray(ti.u32, shape=(64, 64))

    @ti.kernel
    def init_img(img: ti.types.ndarray()):
        for i, j in ti.ndrange(64, 64):
            img[i, j] = ti.cast(i, ti.u32) | (ti.cast(j, ti.u32) << 8) | (ti.u32(255) << 24)

    init_img(image)
    frame = ti.ui.DisplayFrame.from_packed_u32_ndarray(image)
    assert canvas.submit_frame(frame) is True

    rendered = window.get_image_buffer_as_numpy()
    expected = np.empty((64, 64, 4), dtype=np.float32)
    x = np.arange(64, dtype=np.float32)[:, None]
    y = np.arange(64, dtype=np.float32)[None, :]
    expected[..., 0] = x / 255.0
    expected[..., 1] = y / 255.0
    expected[..., 2] = 0.0
    expected[..., 3] = 1.0
    np.testing.assert_allclose(rendered, expected, atol=1.0 / 255.0 + 1e-5)
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_set_image_display_frame_packed_u32_ndarray_non_square():
    width, height = 32, 48
    window = ti.ui.Window("test", (width, height), show_window=False)
    canvas = window.get_canvas()
    image = ti.ndarray(ti.u32, shape=(width, height))

    @ti.kernel
    def init_img(img: ti.types.ndarray()):
        for i, j in ti.ndrange(width, height):
            red = ti.cast(i * 3, ti.u32)
            green = ti.cast(j * 5, ti.u32) << 8
            alpha = ti.u32(255) << 24
            img[i, j] = red | green | alpha

    init_img(image)
    frame = ti.ui.DisplayFrame.from_packed_u32_ndarray(image)
    assert canvas.submit_frame(frame) is True

    rendered = window.get_image_buffer_as_numpy()
    expected = np.empty((width, height, 4), dtype=np.float32)
    x = np.arange(width, dtype=np.float32)[:, None]
    y = np.arange(height, dtype=np.float32)[None, :]
    expected[..., 0] = (x * 3) / 255.0
    expected[..., 1] = (y * 5) / 255.0
    expected[..., 2] = 0.0
    expected[..., 3] = 1.0
    np.testing.assert_allclose(rendered, expected, atol=1.0 / 255.0 + 1e-5)
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_set_image_display_frame_packed_u32_ndarray_resize_sequence():
    window = ti.ui.Window("test", (64, 64), show_window=False)
    canvas = window.get_canvas()

    assert (
        canvas.submit_frame(_make_solid_packed_frame(64, 64, (255, 0, 0, 255)))
        is True
    )
    _assert_solid_rgba(window.get_image_buffer_as_numpy(), (255, 0, 0, 255))

    assert (
        canvas.submit_frame(_make_solid_packed_frame(32, 48, (0, 255, 0, 255)))
        is True
    )
    _assert_solid_rgba(window.get_image_buffer_as_numpy(), (0, 255, 0, 255))
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_set_image_display_frame_switch_packed_to_host_rgba8_resize():
    window = ti.ui.Window("test", (64, 64), show_window=False)
    canvas = window.get_canvas()

    assert (
        canvas.submit_frame(_make_solid_packed_frame(64, 64, (255, 0, 0, 255)))
        is True
    )
    _assert_solid_rgba(window.get_image_buffer_as_numpy(), (255, 0, 0, 255))

    image = np.zeros((16, 24, 4), dtype=np.uint8)
    image[..., 1] = np.uint8(255)
    image[..., 3] = np.uint8(255)
    assert canvas.submit_frame(ti.ui.DisplayFrame.from_numpy_rgba8(image)) is True
    _assert_solid_rgba(window.get_image_buffer_as_numpy(), (0, 255, 0, 255))
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
def test_hidden_window_show_after_set_image_process_exit(tmp_path):
    if not is_arch_supported(ti.vulkan):
        pytest.skip("Vulkan is not supported")

    script = tmp_path / "hidden_window_show_repro.py"
    script.write_text(
        textwrap.dedent(
            """
            import taichi_forge as ti

            ti.init(
                arch=ti.vulkan,
                offline_cache=False,
                vulkan_sparse_experimental=False,
            )

            img = ti.Vector.field(4, ti.f32, shape=(64, 64))

            @ti.kernel
            def init_img():
                for i, j in img:
                    img[i, j] = ti.Vector([i, j, 0, 64], dt=ti.f32) / 64

            init_img()
            window = ti.ui.Window("test", (64, 64), show_window=False)
            canvas = window.get_canvas()
            canvas.set_image(img)
            window.show()
            ti.sync()
            print("hidden window show completed", flush=True)
            """
        ),
        encoding="utf-8",
    )

    repo_python = pathlib.Path(__file__).parents[2] / "python"
    env = os.environ.copy()
    env["TI_SKIP_VERSION_CHECK"] = "ON"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_python), env.get("PYTHONPATH", "")]
    )

    result = subprocess.run(
        [sys.executable, "-u", str(script)],
        cwd=str(pathlib.Path(__file__).parents[2]),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, (
        f"subprocess failed with {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_set_image_flat_field():
    window = ti.ui.Window("test", (640, 480), show_window=False)
    canvas = window.get_canvas()

    img = ti.field(ti.f32, (512, 512, 4))

    @ti.kernel
    def init_img():
        for i, j in ti.ndrange(img.shape[0], img.shape[1]):
            img[i, j, 0] = i / 512
            img[i, j, 1] = j / 512
            img[i, j, 2] = 0
            img[i, j, 3] = 1.0

    init_img()

    def render():
        canvas.set_image(img)

    # Render in off-line mode to check if there are errors
    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()

    verify_image(window.get_image_buffer_as_numpy(), "test_set_image")
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=[ti.vulkan, ti.metal])
def test_set_image_with_texture():
    window = ti.ui.Window("test", (640, 480), show_window=False)
    canvas = window.get_canvas()

    img = ti.Texture(ti.Format.rgba8, (512, 512))

    @ti.kernel
    def init_img(img: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.rgba8, lod=0)):
        for i, j in ti.ndrange(512, 512):
            img.store(ti.Vector([i, j]), ti.Vector([i, j, 0, 512], dt=ti.f32) / 512)

    init_img(img)

    def render():
        canvas.set_image(img)

    # Render in off-line mode to check if there are errors
    for _ in range(3):
        render()
        window.get_image_buffer_as_numpy()

    render()

    # Relaxed error because texture sampler differences
    # Note: the error is measured from a 0..255 range image
    verify_image(window.get_image_buffer_as_numpy(), "test_set_image", 0.3)
    window.destroy()


# NOTE: Cannot automate the test for the case of ImGui scaling on HiDPI displays. So that needs to be tested manually.
@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_imgui():
    window = ti.ui.Window("test", (640, 480), show_window=False)
    gui = window.get_gui()

    def render():
        with gui.sub_window("window 0", 0.1, 0.1, 0.8, 0.2) as w:
            w.text("Hello Taichi!")
            w.text("Hello Again!")
        with gui.sub_window("window 1", 0.1, 0.4, 0.8, 0.2) as w:
            w.button("Press to unlease creativity")
            w.slider_float("creativity level", 100.0, 0.0, 100.0)
        with gui.sub_window("window 2", 0.1, 0.7, 0.8, 0.2) as w:
            w.color_edit_3("Heyy", (0, 0, 1))

    # Render in off-line mode to check if there are errors
    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()
    verify_image(window.get_image_buffer_as_numpy(), "test_imgui")
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_imgui_empty_frame_after_widget_frame():
    w, h = 320, 240
    image = np.zeros((w, h, 3), dtype=np.float32)
    image[..., 0] = 0.2
    image[..., 1] = 0.3
    image[..., 2] = 0.4

    window = ti.ui.Window("test", (w, h), show_window=False, fps_limit=65535)
    canvas = window.get_canvas()
    gui = window.get_gui()

    canvas.set_image(image)
    with gui.sub_window("Panel", 0.02, 0.02, 0.2, 0.12) as panel:
        panel.text("first")
    assert window.show()

    canvas.set_image(image)
    assert window.show()

    canvas.set_image(image)
    assert window.show()
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_exit_without_showing():
    window = ti.ui.Window("Taichi", (256, 256), show_window=False)


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_get_camera_view_and_projection_matrix():
    window = ti.ui.Window("Taichi", (256, 256), show_window=False)
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0, 0, 3)
    camera.lookat(0, 0, 0)

    scene.set_camera(camera)

    view_matrix = camera.get_view_matrix()
    projection_matrix = camera.get_projection_matrix(1080 / 720)

    for i in range(4):
        assert abs(view_matrix[i, i] - 1) <= 1e-5
    assert abs(view_matrix[3, 2] + 3) <= 1e-5

    assert abs(projection_matrix[0, 0] - 1.6094756) <= 1e-5
    assert abs(projection_matrix[1, 1] - 2.4142134) <= 1e-5
    assert abs(projection_matrix[2, 2] - 1.0001000e-4) <= 1e-5
    assert abs(projection_matrix[2, 3] + 1.0000000) <= 1e-5
    assert abs(projection_matrix[3, 2] - 1.0001000e-1) <= 1e-5
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_get_camera_view_and_projection_matrix_old():
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(0, 0, 3)
    camera.lookat(0, 0, 0)

    scene.set_camera(camera)

    view_matrix = camera.get_view_matrix()
    projection_matrix = camera.get_projection_matrix(1080 / 720)

    for i in range(4):
        assert abs(view_matrix[i, i] - 1) <= 1e-5
    assert abs(view_matrix[3, 2] + 3) <= 1e-5

    assert abs(projection_matrix[0, 0] - 1.6094756) <= 1e-5
    assert abs(projection_matrix[1, 1] - 2.4142134) <= 1e-5
    assert abs(projection_matrix[2, 2] - 1.0001000e-4) <= 1e-5
    assert abs(projection_matrix[2, 3] + 1.0000000) <= 1e-5
    assert abs(projection_matrix[3, 2] - 1.0001000e-1) <= 1e-5


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_fetching_color_attachment():
    window = ti.ui.Window("test", (640, 480), show_window=False)
    canvas = window.get_canvas()

    img = ti.Vector.field(4, ti.f32, (512, 512))

    @ti.kernel
    def init_img():
        for i, j in img:
            img[i, j] = ti.Vector([i, j, 0, 512], dt=ti.f32) / 512

    init_img()

    def render():
        canvas.set_image(img)

    # Render in off-line mode to check if there are errors
    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()
    verify_image(window.get_image_buffer_as_numpy(), "test_set_image")
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_fetching_depth_attachment():
    window = ti.ui.Window("test", (512, 512), vsync=True, show_window=False)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()

    ball_center = ti.Vector.field(3, dtype=float, shape=(1,))
    ball_center[0] = ti.math.vec3(0, 0, 0.5)

    def render():
        camera.position(0.0, 0.0, 1)
        camera.lookat(0.0, 0.0, 0)
        scene.set_camera(camera)
        scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.particles(ball_center, radius=0.05, color=(0.5, 0.42, 0.8))
        canvas.scene(scene)

    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()
    verify_image(window.get_depth_buffer_as_numpy(), "test_depth")
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_fetching_depth_attachment_old():
    window = ti.ui.Window("test", (512, 512), vsync=True, show_window=False)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()

    ball_center = ti.Vector.field(3, dtype=float, shape=(1,))
    ball_center[0] = ti.math.vec3(0, 0, 0.5)

    def render():
        camera.position(0.0, 0.0, 1)
        camera.lookat(0.0, 0.0, 0)
        scene.set_camera(camera)
        scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.particles(ball_center, radius=0.05, color=(0.5, 0.42, 0.8))
        canvas.scene(scene)

    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()
    verify_image(window.get_depth_buffer_as_numpy(), "test_depth")
    window.destroy()


@pytest.mark.parametrize("offset", [None, (0, 0), (-256, -256), (256, -256), (-256, 256), (256, 256), (23333, 233333)])
@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_get_depth_buffer_with_offset(offset):
    window = ti.ui.Window("test", (512, 512), vsync=True, show_window=False)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()

    ball_center = ti.Vector.field(3, dtype=float, shape=(1,))
    ball_center[0] = ti.math.vec3(0, 0, 0.5)

    def render():
        camera.position(0.0, 0.0, 1)
        camera.lookat(0.0, 0.0, 0)
        scene.set_camera(camera)
        scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.particles(ball_center, radius=0.05, color=(0.5, 0.42, 0.8))
        canvas.scene(scene)

    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()

    depth_buffer_field = ti.field(dtype=ti.f32, shape=(512, 512), offset=offset)
    window.get_depth_buffer(depth_buffer_field)
    verify_image(depth_buffer_field, "test_depth")
    window.destroy()


@pytest.mark.parametrize("offset", [None, (0, 0), (-256, -256), (256, -256), (-256, 256), (256, 256), (23333, 233333)])
@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_get_depth_buffer_with_offset_old(offset):
    window = ti.ui.Window("test", (512, 512), vsync=True, show_window=False)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()

    ball_center = ti.Vector.field(3, dtype=float, shape=(1,))
    ball_center[0] = ti.math.vec3(0, 0, 0.5)

    def render():
        camera.position(0.0, 0.0, 1)
        camera.lookat(0.0, 0.0, 0)
        scene.set_camera(camera)
        scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.particles(ball_center, radius=0.05, color=(0.5, 0.42, 0.8))
        canvas.scene(scene)

    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()

    depth_buffer_field = ti.field(dtype=ti.f32, shape=(512, 512), offset=offset)
    window.get_depth_buffer(depth_buffer_field)
    verify_image(depth_buffer_field, "test_depth")
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_draw_lines():
    N = 10
    particles_pos = ti.Vector.field(3, dtype=ti.f32, shape=N)
    points_pos = ti.Vector.field(3, dtype=ti.f32, shape=N)

    @ti.kernel
    def init_points_pos(points: ti.template()):
        for i in range(points.shape[0]):
            points[i] = [i for j in ti.static(range(3))]

    init_points_pos(particles_pos)
    init_points_pos(points_pos)

    window = ti.ui.Window("Test for Drawing 3d-lines", (768, 768), show_window=False)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0, 5, -10)
    camera.lookat(3, 3, 1)

    def render():
        scene.set_camera(camera)
        scene.ambient_light((0.8, 0.8, 0.8))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

        scene.particles(particles_pos, color=(0.68, 0.26, 0.19), radius=0.5)
        scene.lines(points_pos, color=(0.28, 0.68, 0.99), width=5.0)
        canvas.scene(scene)

    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()
    verify_image(window.get_image_buffer_as_numpy(), "test_draw_lines")
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_draw_lines_old():
    N = 10
    particles_pos = ti.Vector.field(3, dtype=ti.f32, shape=N)
    points_pos = ti.Vector.field(3, dtype=ti.f32, shape=N)

    @ti.kernel
    def init_points_pos(points: ti.template()):
        for i in range(points.shape[0]):
            points[i] = [i for j in ti.static(range(3))]

    init_points_pos(particles_pos)
    init_points_pos(points_pos)

    window = ti.ui.Window("Test for Drawing 3d-lines", (768, 768), show_window=False)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(0, 5, -10)
    camera.lookat(3, 3, 1)

    def render():
        scene.set_camera(camera)
        scene.ambient_light((0.8, 0.8, 0.8))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

        scene.particles(particles_pos, color=(0.68, 0.26, 0.19), radius=0.5)
        scene.lines(points_pos, color=(0.28, 0.68, 0.99), width=5.0)
        canvas.scene(scene)

    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()
    verify_image(window.get_image_buffer_as_numpy(), "test_draw_lines")
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_draw_part_of_particles():
    N = 10
    particles_pos = ti.Vector.field(3, dtype=ti.f32, shape=N)
    points_pos = ti.Vector.field(3, dtype=ti.f32, shape=N)

    @ti.kernel
    def init_points_pos(points: ti.template()):
        for i in range(points.shape[0]):
            points[i] = [i for j in ti.static(range(3))]

    init_points_pos(particles_pos)
    init_points_pos(points_pos)

    window = ti.ui.Window("Test", (768, 768), show_window=False)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0, 5, -10)
    camera.lookat(3, 3, 1)

    def render():
        scene.set_camera(camera)
        scene.ambient_light((0.8, 0.8, 0.8))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

        scene.particles(
            particles_pos,
            color=(0.68, 0.26, 0.19),
            radius=0.5,
            index_offset=2,
            index_count=6,
        )
        canvas.scene(scene)

    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()
    verify_image(window.get_image_buffer_as_numpy(), "test_draw_part_of_particles")
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_draw_part_of_particles_old():
    N = 10
    particles_pos = ti.Vector.field(3, dtype=ti.f32, shape=N)
    points_pos = ti.Vector.field(3, dtype=ti.f32, shape=N)

    @ti.kernel
    def init_points_pos(points: ti.template()):
        for i in range(points.shape[0]):
            points[i] = [i for j in ti.static(range(3))]

    init_points_pos(particles_pos)
    init_points_pos(points_pos)

    window = ti.ui.Window("Test", (768, 768), show_window=False)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(0, 5, -10)
    camera.lookat(3, 3, 1)

    def render():
        scene.set_camera(camera)
        scene.ambient_light((0.8, 0.8, 0.8))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

        scene.particles(
            particles_pos,
            color=(0.68, 0.26, 0.19),
            radius=0.5,
            index_offset=2,
            index_count=6,
        )
        canvas.scene(scene)

    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()
    verify_image(window.get_image_buffer_as_numpy(), "test_draw_part_of_particles")
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_draw_part_of_particles_per_vertex_rad_and_col():
    N = 10
    particles_pos = ti.Vector.field(3, dtype=ti.f32, shape=N)
    particles_col = ti.Vector.field(3, dtype=ti.f32, shape=N)
    particles_radii = ti.field(dtype=ti.f32, shape=N)

    @ti.kernel
    def init_points_pos(points: ti.template()):
        for i in range(points.shape[0]):
            points[i] = [i for j in ti.static(range(3))]

    @ti.kernel
    def init_points_col(points: ti.template()):
        for i in range(points.shape[0]):
            points[i] = [(i + 1) / N, 0.5, (i + 1) / N]

    @ti.kernel
    def init_points_radii(radii: ti.template()):
        for i in range(radii.shape[0]):
            radii[i] = (i + 1) * 0.05

    init_points_pos(particles_pos)
    init_points_radii(particles_radii)
    init_points_col(particles_col)

    window = ti.ui.Window("Test", (768, 768), show_window=False)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0, 5, -10)
    camera.lookat(3, 3, 1)

    def render():
        scene.set_camera(camera)
        scene.ambient_light((0.8, 0.8, 0.8))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

        scene.particles(
            particles_pos,
            color=(0.68, 0.26, 0.19),
            radius=0.5,
            per_vertex_color=particles_col,
            per_vertex_radius=particles_radii,
            index_offset=2,
            index_count=6,
        )
        canvas.scene(scene)

    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()
    verify_image(window.get_image_buffer_as_numpy(), "test_draw_part_of_particles_per_vertex_rad_and_col")
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_draw_part_of_particles_per_vertex_rad_and_col_old():
    N = 10
    particles_pos = ti.Vector.field(3, dtype=ti.f32, shape=N)
    particles_col = ti.Vector.field(3, dtype=ti.f32, shape=N)
    particles_radii = ti.field(dtype=ti.f32, shape=N)

    @ti.kernel
    def init_points_pos(points: ti.template()):
        for i in range(points.shape[0]):
            points[i] = [i for j in ti.static(range(3))]

    @ti.kernel
    def init_points_col(points: ti.template()):
        for i in range(points.shape[0]):
            points[i] = [(i + 1) / N, 0.5, (i + 1) / N]

    @ti.kernel
    def init_points_radii(radii: ti.template()):
        for i in range(radii.shape[0]):
            radii[i] = (i + 1) * 0.05

    init_points_pos(particles_pos)
    init_points_radii(particles_radii)
    init_points_col(particles_col)

    window = ti.ui.Window("Test", (768, 768), show_window=False)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(0, 5, -10)
    camera.lookat(3, 3, 1)

    def render():
        scene.set_camera(camera)
        scene.ambient_light((0.8, 0.8, 0.8))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

        scene.particles(
            particles_pos,
            color=(0.68, 0.26, 0.19),
            radius=0.5,
            per_vertex_color=particles_col,
            per_vertex_radius=particles_radii,
            index_offset=2,
            index_count=6,
        )
        canvas.scene(scene)

    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()
    verify_image(window.get_image_buffer_as_numpy(), "test_draw_part_of_particles_per_vertex_rad_and_col")
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_draw_part_of_mesh():
    N = 10
    NV = (N + 1) ** 2
    NT = 2 * N**2
    NE = 2 * N * (N + 1) + N**2
    pos = ti.Vector.field(3, ti.f32, shape=NV)
    tri = ti.field(ti.i32, shape=3 * NT)
    edge = ti.Vector.field(2, ti.i32, shape=NE)

    @ti.kernel
    def init_pos():
        for i, j in ti.ndrange(N + 1, N + 1):
            idx = i * (N + 1) + j
            pos[idx] = ti.Vector([i / N, 1.0 - j / N, 0.5])

    @ti.kernel
    def init_tri():
        for i, j in ti.ndrange(N, N):
            tri_idx = 6 * (i * N + j)
            pos_idx = i * (N + 1) + j
            if (i + j) % 2 == 0:
                tri[tri_idx + 0] = pos_idx
                tri[tri_idx + 1] = pos_idx + N + 2
                tri[tri_idx + 2] = pos_idx + 1
                tri[tri_idx + 3] = pos_idx
                tri[tri_idx + 4] = pos_idx + N + 1
                tri[tri_idx + 5] = pos_idx + N + 2
            else:
                tri[tri_idx + 0] = pos_idx
                tri[tri_idx + 1] = pos_idx + N + 1
                tri[tri_idx + 2] = pos_idx + 1
                tri[tri_idx + 3] = pos_idx + 1
                tri[tri_idx + 4] = pos_idx + N + 1
                tri[tri_idx + 5] = pos_idx + N + 2

    @ti.kernel
    def init_edge():
        for i, j in ti.ndrange(N + 1, N):
            edge_idx = i * N + j
            pos_idx = i * (N + 1) + j
            edge[edge_idx] = ti.Vector([pos_idx, pos_idx + 1])
        start = N * (N + 1)
        for i, j in ti.ndrange(N, N + 1):
            edge_idx = start + j * N + i
            pos_idx = i * (N + 1) + j
            edge[edge_idx] = ti.Vector([pos_idx, pos_idx + N + 1])
        start = 2 * N * (N + 1)
        for i, j in ti.ndrange(N, N):
            edge_idx = start + i * N + j
            pos_idx = i * (N + 1) + j
            if (i + j) % 2 == 0:
                edge[edge_idx] = ti.Vector([pos_idx, pos_idx + N + 2])
            else:
                edge[edge_idx] = ti.Vector([pos_idx + 1, pos_idx + N + 1])

    init_pos()
    init_tri()
    init_edge()

    window = ti.ui.Window("test", (1024, 1024), vsync=True, show_window=False)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(1.5, 1, -1)
    camera.lookat(1, 0.5, 0)
    camera.fov(90)

    def render():
        scene.set_camera(camera)
        scene.point_light(pos=(0.5, 1, 2), color=(1, 1, 1))

        scene.mesh(
            pos,
            tri,
            color=(39 / 255, 123 / 255, 192 / 255),
            two_sided=True,
            index_count=2 * NT,
            index_offset=9,
        )
        canvas.scene(scene)

    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()
    verify_image(window.get_image_buffer_as_numpy(), "test_draw_part_of_mesh")
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_draw_part_of_mesh_old():
    N = 10
    NV = (N + 1) ** 2
    NT = 2 * N**2
    NE = 2 * N * (N + 1) + N**2
    pos = ti.Vector.field(3, ti.f32, shape=NV)
    tri = ti.field(ti.i32, shape=3 * NT)
    edge = ti.Vector.field(2, ti.i32, shape=NE)

    @ti.kernel
    def init_pos():
        for i, j in ti.ndrange(N + 1, N + 1):
            idx = i * (N + 1) + j
            pos[idx] = ti.Vector([i / N, 1.0 - j / N, 0.5])

    @ti.kernel
    def init_tri():
        for i, j in ti.ndrange(N, N):
            tri_idx = 6 * (i * N + j)
            pos_idx = i * (N + 1) + j
            if (i + j) % 2 == 0:
                tri[tri_idx + 0] = pos_idx
                tri[tri_idx + 1] = pos_idx + N + 2
                tri[tri_idx + 2] = pos_idx + 1
                tri[tri_idx + 3] = pos_idx
                tri[tri_idx + 4] = pos_idx + N + 1
                tri[tri_idx + 5] = pos_idx + N + 2
            else:
                tri[tri_idx + 0] = pos_idx
                tri[tri_idx + 1] = pos_idx + N + 1
                tri[tri_idx + 2] = pos_idx + 1
                tri[tri_idx + 3] = pos_idx + 1
                tri[tri_idx + 4] = pos_idx + N + 1
                tri[tri_idx + 5] = pos_idx + N + 2

    @ti.kernel
    def init_edge():
        for i, j in ti.ndrange(N + 1, N):
            edge_idx = i * N + j
            pos_idx = i * (N + 1) + j
            edge[edge_idx] = ti.Vector([pos_idx, pos_idx + 1])
        start = N * (N + 1)
        for i, j in ti.ndrange(N, N + 1):
            edge_idx = start + j * N + i
            pos_idx = i * (N + 1) + j
            edge[edge_idx] = ti.Vector([pos_idx, pos_idx + N + 1])
        start = 2 * N * (N + 1)
        for i, j in ti.ndrange(N, N):
            edge_idx = start + i * N + j
            pos_idx = i * (N + 1) + j
            if (i + j) % 2 == 0:
                edge[edge_idx] = ti.Vector([pos_idx, pos_idx + N + 2])
            else:
                edge[edge_idx] = ti.Vector([pos_idx + 1, pos_idx + N + 1])

    init_pos()
    init_tri()
    init_edge()

    window = ti.ui.Window("test", (1024, 1024), vsync=True, show_window=False)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(1.5, 1, -1)
    camera.lookat(1, 0.5, 0)
    camera.fov(90)

    def render():
        scene.set_camera(camera)
        scene.point_light(pos=(0.5, 1, 2), color=(1, 1, 1))

        scene.mesh(
            pos,
            tri,
            color=(39 / 255, 123 / 255, 192 / 255),
            two_sided=True,
            index_count=2 * NT,
            index_offset=9,
        )
        canvas.scene(scene)

    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()
    verify_image(window.get_image_buffer_as_numpy(), "test_draw_part_of_mesh")
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_draw_part_of_lines():
    N = 10
    particles_pos = ti.Vector.field(3, dtype=ti.f32, shape=N)
    points_pos = ti.Vector.field(3, dtype=ti.f32, shape=N)

    @ti.kernel
    def init_points_pos(points: ti.template()):
        for i in range(points.shape[0]):
            points[i] = [i for j in ti.static(range(3))]

    init_points_pos(particles_pos)
    init_points_pos(points_pos)

    window = ti.ui.Window("Test", (768, 768), show_window=False)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0, 5, -10)
    camera.lookat(3, 3, 1)

    def render():
        scene.set_camera(camera)
        scene.ambient_light((0.8, 0.8, 0.8))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

        scene.particles(particles_pos, color=(0.68, 0.26, 0.19), radius=0.5)
        scene.lines(
            points_pos,
            color=(0.28, 0.68, 0.99),
            width=5.0,
            vertex_count=6,
            vertex_offset=2,
        )
        canvas.scene(scene)

    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()
    verify_image(window.get_image_buffer_as_numpy(), "test_draw_part_of_lines", 0.3)
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_draw_part_of_lines_old():
    N = 10
    particles_pos = ti.Vector.field(3, dtype=ti.f32, shape=N)
    points_pos = ti.Vector.field(3, dtype=ti.f32, shape=N)

    @ti.kernel
    def init_points_pos(points: ti.template()):
        for i in range(points.shape[0]):
            points[i] = [i for j in ti.static(range(3))]

    init_points_pos(particles_pos)
    init_points_pos(points_pos)

    window = ti.ui.Window("Test", (768, 768), show_window=False)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(0, 5, -10)
    camera.lookat(3, 3, 1)

    def render():
        scene.set_camera(camera)
        scene.ambient_light((0.8, 0.8, 0.8))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

        scene.particles(particles_pos, color=(0.68, 0.26, 0.19), radius=0.5)
        scene.lines(
            points_pos,
            color=(0.28, 0.68, 0.99),
            width=5.0,
            vertex_count=6,
            vertex_offset=2,
        )
        canvas.scene(scene)

    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()
    verify_image(window.get_image_buffer_as_numpy(), "test_draw_part_of_lines", 0.3)
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_draw_mesh_instances():
    N = 10
    NV = (N + 1) ** 2
    NT = 2 * N**2
    NE = 2 * N * (N + 1) + N**2
    pos = ti.Vector.field(3, ti.f32, shape=NV)
    tri = ti.field(ti.i32, shape=3 * NT)
    edge = ti.Vector.field(2, ti.i32, shape=NE)

    # Instance Attribute Information
    NInstanceRows = 100
    NInstanceCols = 100
    NInstance = NInstanceRows * NInstanceCols
    instances_transforms = ti.Matrix.field(4, 4, ti.f32, shape=(NInstance,))

    @ti.kernel
    def init_transforms_of_instances():
        identity = ti.Matrix.identity(ti.f32, 4)
        for i in range(NInstanceRows):
            for j in range(NInstanceCols):
                index = i * NInstanceCols + j
                instances_transforms[index] = identity
                translate_matrix = ti.math.translate(1.2 * j, 0, -1.2 * i)
                instances_transforms[index] = translate_matrix @ instances_transforms[index]

    @ti.kernel
    def init_pos():
        for i, j in ti.ndrange(N + 1, N + 1):
            idx = i * (N + 1) + j
            pos[idx] = ti.Vector([i / N, 1.0 - j / N, 0.5])

    @ti.kernel
    def init_tri():
        for i, j in ti.ndrange(N, N):
            tri_idx = 6 * (i * N + j)
            pos_idx = i * (N + 1) + j
            if (i + j) % 2 == 0:
                tri[tri_idx + 0] = pos_idx
                tri[tri_idx + 1] = pos_idx + N + 2
                tri[tri_idx + 2] = pos_idx + 1
                tri[tri_idx + 3] = pos_idx
                tri[tri_idx + 4] = pos_idx + N + 1
                tri[tri_idx + 5] = pos_idx + N + 2
            else:
                tri[tri_idx + 0] = pos_idx
                tri[tri_idx + 1] = pos_idx + N + 1
                tri[tri_idx + 2] = pos_idx + 1
                tri[tri_idx + 3] = pos_idx + 1
                tri[tri_idx + 4] = pos_idx + N + 1
                tri[tri_idx + 5] = pos_idx + N + 2

    @ti.kernel
    def init_edge():
        for i, j in ti.ndrange(N + 1, N):
            edge_idx = i * N + j
            pos_idx = i * (N + 1) + j
            edge[edge_idx] = ti.Vector([pos_idx, pos_idx + 1])
        start = N * (N + 1)
        for i, j in ti.ndrange(N, N + 1):
            edge_idx = start + j * N + i
            pos_idx = i * (N + 1) + j
            edge[edge_idx] = ti.Vector([pos_idx, pos_idx + N + 1])
        start = 2 * N * (N + 1)
        for i, j in ti.ndrange(N, N):
            edge_idx = start + i * N + j
            pos_idx = i * (N + 1) + j
            if (i + j) % 2 == 0:
                edge[edge_idx] = ti.Vector([pos_idx, pos_idx + N + 2])
            else:
                edge[edge_idx] = ti.Vector([pos_idx + 1, pos_idx + N + 1])

    @ti.kernel
    def update_transform(t: ti.f32):
        for i in range(NInstance):
            rotation_matrix = ti.math.rot_by_axis(ti.math.vec3(0, 1, 0), 0.01 * ti.math.sin(t))
            instances_transforms[i] = instances_transforms[i] @ rotation_matrix

    init_transforms_of_instances()

    init_pos()
    init_tri()
    init_edge()

    window = ti.ui.Window("test", (1024, 1024), vsync=True, show_window=False)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(-1.82731234, 2.26492691, 2.27800684)
    camera.lookat(-1.13230401, 2.11502124, 1.57480579)
    camera.fov(90)

    def render():
        scene.set_camera(camera)
        scene.point_light(pos=(0.5, 1, 2), color=(1, 1, 1))

        scene.mesh_instance(
            pos,
            tri,
            color=(39 / 255, 123 / 255, 192 / 255),
            two_sided=True,
            transforms=instances_transforms,
        )
        canvas.scene(scene)

    for i in range(30):
        update_transform(30)
        render()
        window.get_image_buffer_as_numpy()

    render()
    verify_image(window.get_image_buffer_as_numpy(), "test_draw_mesh_instances")
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_draw_mesh_instances_old():
    N = 10
    NV = (N + 1) ** 2
    NT = 2 * N**2
    NE = 2 * N * (N + 1) + N**2
    pos = ti.Vector.field(3, ti.f32, shape=NV)
    tri = ti.field(ti.i32, shape=3 * NT)
    edge = ti.Vector.field(2, ti.i32, shape=NE)

    # Instance Attribute Information
    NInstanceRows = 100
    NInstanceCols = 100
    NInstance = NInstanceRows * NInstanceCols
    instances_transforms = ti.Matrix.field(4, 4, ti.f32, shape=(NInstance,))

    @ti.kernel
    def init_transforms_of_instances():
        identity = ti.Matrix.identity(ti.f32, 4)
        for i in range(NInstanceRows):
            for j in range(NInstanceCols):
                index = i * NInstanceCols + j
                instances_transforms[index] = identity
                translate_matrix = ti.math.translate(1.2 * j, 0, -1.2 * i)
                instances_transforms[index] = translate_matrix @ instances_transforms[index]

    @ti.kernel
    def init_pos():
        for i, j in ti.ndrange(N + 1, N + 1):
            idx = i * (N + 1) + j
            pos[idx] = ti.Vector([i / N, 1.0 - j / N, 0.5])

    @ti.kernel
    def init_tri():
        for i, j in ti.ndrange(N, N):
            tri_idx = 6 * (i * N + j)
            pos_idx = i * (N + 1) + j
            if (i + j) % 2 == 0:
                tri[tri_idx + 0] = pos_idx
                tri[tri_idx + 1] = pos_idx + N + 2
                tri[tri_idx + 2] = pos_idx + 1
                tri[tri_idx + 3] = pos_idx
                tri[tri_idx + 4] = pos_idx + N + 1
                tri[tri_idx + 5] = pos_idx + N + 2
            else:
                tri[tri_idx + 0] = pos_idx
                tri[tri_idx + 1] = pos_idx + N + 1
                tri[tri_idx + 2] = pos_idx + 1
                tri[tri_idx + 3] = pos_idx + 1
                tri[tri_idx + 4] = pos_idx + N + 1
                tri[tri_idx + 5] = pos_idx + N + 2

    @ti.kernel
    def init_edge():
        for i, j in ti.ndrange(N + 1, N):
            edge_idx = i * N + j
            pos_idx = i * (N + 1) + j
            edge[edge_idx] = ti.Vector([pos_idx, pos_idx + 1])
        start = N * (N + 1)
        for i, j in ti.ndrange(N, N + 1):
            edge_idx = start + j * N + i
            pos_idx = i * (N + 1) + j
            edge[edge_idx] = ti.Vector([pos_idx, pos_idx + N + 1])
        start = 2 * N * (N + 1)
        for i, j in ti.ndrange(N, N):
            edge_idx = start + i * N + j
            pos_idx = i * (N + 1) + j
            if (i + j) % 2 == 0:
                edge[edge_idx] = ti.Vector([pos_idx, pos_idx + N + 2])
            else:
                edge[edge_idx] = ti.Vector([pos_idx + 1, pos_idx + N + 1])

    @ti.kernel
    def update_transform(t: ti.f32):
        for i in range(NInstance):
            rotation_matrix = ti.math.rot_by_axis(ti.math.vec3(0, 1, 0), 0.01 * ti.math.sin(t))
            instances_transforms[i] = instances_transforms[i] @ rotation_matrix

    init_transforms_of_instances()

    init_pos()
    init_tri()
    init_edge()

    window = ti.ui.Window("test", (1024, 1024), vsync=True, show_window=False)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(-1.82731234, 2.26492691, 2.27800684)
    camera.lookat(-1.13230401, 2.11502124, 1.57480579)
    camera.fov(90)

    def render():
        scene.set_camera(camera)
        scene.point_light(pos=(0.5, 1, 2), color=(1, 1, 1))

        scene.mesh_instance(
            pos,
            tri,
            color=(39 / 255, 123 / 255, 192 / 255),
            two_sided=True,
            transforms=instances_transforms,
        )
        canvas.scene(scene)

    for i in range(30):
        update_transform(30)
        render()
        window.get_image_buffer_as_numpy()

    render()
    verify_image(window.get_image_buffer_as_numpy(), "test_draw_mesh_instances")
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_draw_part_of_mesh_instances():
    N = 10
    NV = (N + 1) ** 2
    NT = 2 * N**2
    NE = 2 * N * (N + 1) + N**2
    pos = ti.Vector.field(3, ti.f32, shape=NV)
    tri = ti.field(ti.i32, shape=3 * NT)
    edge = ti.Vector.field(2, ti.i32, shape=NE)

    # Instance Attribute Information
    NInstanceRows = 10
    NInstanceCols = 10
    NInstance = NInstanceRows * NInstanceCols
    instances_transforms = ti.Matrix.field(4, 4, ti.f32, shape=(NInstance,))

    @ti.kernel
    def init_transforms_of_instances():
        identity = ti.Matrix.identity(ti.f32, 4)
        for i in range(NInstanceRows):
            for j in range(NInstanceCols):
                index = i * NInstanceCols + j
                instances_transforms[index] = identity
                translate_matrix = ti.math.translate(1.2 * j, 0, -1.2 * i)
                instances_transforms[index] = translate_matrix @ instances_transforms[index]

    @ti.kernel
    def init_pos():
        for i, j in ti.ndrange(N + 1, N + 1):
            idx = i * (N + 1) + j
            pos[idx] = ti.Vector([i / N, 1.0 - j / N, 0.5])

    @ti.kernel
    def init_tri():
        for i, j in ti.ndrange(N, N):
            tri_idx = 6 * (i * N + j)
            pos_idx = i * (N + 1) + j
            if (i + j) % 2 == 0:
                tri[tri_idx + 0] = pos_idx
                tri[tri_idx + 1] = pos_idx + N + 2
                tri[tri_idx + 2] = pos_idx + 1
                tri[tri_idx + 3] = pos_idx
                tri[tri_idx + 4] = pos_idx + N + 1
                tri[tri_idx + 5] = pos_idx + N + 2
            else:
                tri[tri_idx + 0] = pos_idx
                tri[tri_idx + 1] = pos_idx + N + 1
                tri[tri_idx + 2] = pos_idx + 1
                tri[tri_idx + 3] = pos_idx + 1
                tri[tri_idx + 4] = pos_idx + N + 1
                tri[tri_idx + 5] = pos_idx + N + 2

    @ti.kernel
    def init_edge():
        for i, j in ti.ndrange(N + 1, N):
            edge_idx = i * N + j
            pos_idx = i * (N + 1) + j
            edge[edge_idx] = ti.Vector([pos_idx, pos_idx + 1])
        start = N * (N + 1)
        for i, j in ti.ndrange(N, N + 1):
            edge_idx = start + j * N + i
            pos_idx = i * (N + 1) + j
            edge[edge_idx] = ti.Vector([pos_idx, pos_idx + N + 1])
        start = 2 * N * (N + 1)
        for i, j in ti.ndrange(N, N):
            edge_idx = start + i * N + j
            pos_idx = i * (N + 1) + j
            if (i + j) % 2 == 0:
                edge[edge_idx] = ti.Vector([pos_idx, pos_idx + N + 2])
            else:
                edge[edge_idx] = ti.Vector([pos_idx + 1, pos_idx + N + 1])

    init_transforms_of_instances()
    init_pos()
    init_tri()
    init_edge()

    window = ti.ui.Window("test", (1024, 1024), vsync=True, show_window=False)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(-1.82731234, 2.26492691, 2.27800684)
    camera.lookat(-1.13230401, 2.11502124, 1.57480579)
    camera.fov(90)

    def render():
        scene.set_camera(camera)
        scene.point_light(pos=(0.5, 1, 2), color=(1, 1, 1))

        scene.mesh_instance(
            pos,
            tri,
            color=(39 / 255, 123 / 255, 192 / 255),
            two_sided=True,
            transforms=instances_transforms,
            instance_count=10,
            instance_offset=2,
        )
        canvas.scene(scene)

    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()
    verify_image(window.get_image_buffer_as_numpy(), "test_draw_part_of_mesh_instances")
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_draw_part_of_mesh_instances_old():
    N = 10
    NV = (N + 1) ** 2
    NT = 2 * N**2
    NE = 2 * N * (N + 1) + N**2
    pos = ti.Vector.field(3, ti.f32, shape=NV)
    tri = ti.field(ti.i32, shape=3 * NT)
    edge = ti.Vector.field(2, ti.i32, shape=NE)

    # Instance Attribute Information
    NInstanceRows = 10
    NInstanceCols = 10
    NInstance = NInstanceRows * NInstanceCols
    instances_transforms = ti.Matrix.field(4, 4, ti.f32, shape=(NInstance,))

    @ti.kernel
    def init_transforms_of_instances():
        identity = ti.Matrix.identity(ti.f32, 4)
        for i in range(NInstanceRows):
            for j in range(NInstanceCols):
                index = i * NInstanceCols + j
                instances_transforms[index] = identity
                translate_matrix = ti.math.translate(1.2 * j, 0, -1.2 * i)
                instances_transforms[index] = translate_matrix @ instances_transforms[index]

    @ti.kernel
    def init_pos():
        for i, j in ti.ndrange(N + 1, N + 1):
            idx = i * (N + 1) + j
            pos[idx] = ti.Vector([i / N, 1.0 - j / N, 0.5])

    @ti.kernel
    def init_tri():
        for i, j in ti.ndrange(N, N):
            tri_idx = 6 * (i * N + j)
            pos_idx = i * (N + 1) + j
            if (i + j) % 2 == 0:
                tri[tri_idx + 0] = pos_idx
                tri[tri_idx + 1] = pos_idx + N + 2
                tri[tri_idx + 2] = pos_idx + 1
                tri[tri_idx + 3] = pos_idx
                tri[tri_idx + 4] = pos_idx + N + 1
                tri[tri_idx + 5] = pos_idx + N + 2
            else:
                tri[tri_idx + 0] = pos_idx
                tri[tri_idx + 1] = pos_idx + N + 1
                tri[tri_idx + 2] = pos_idx + 1
                tri[tri_idx + 3] = pos_idx + 1
                tri[tri_idx + 4] = pos_idx + N + 1
                tri[tri_idx + 5] = pos_idx + N + 2

    @ti.kernel
    def init_edge():
        for i, j in ti.ndrange(N + 1, N):
            edge_idx = i * N + j
            pos_idx = i * (N + 1) + j
            edge[edge_idx] = ti.Vector([pos_idx, pos_idx + 1])
        start = N * (N + 1)
        for i, j in ti.ndrange(N, N + 1):
            edge_idx = start + j * N + i
            pos_idx = i * (N + 1) + j
            edge[edge_idx] = ti.Vector([pos_idx, pos_idx + N + 1])
        start = 2 * N * (N + 1)
        for i, j in ti.ndrange(N, N):
            edge_idx = start + i * N + j
            pos_idx = i * (N + 1) + j
            if (i + j) % 2 == 0:
                edge[edge_idx] = ti.Vector([pos_idx, pos_idx + N + 2])
            else:
                edge[edge_idx] = ti.Vector([pos_idx + 1, pos_idx + N + 1])

    init_transforms_of_instances()
    init_pos()
    init_tri()
    init_edge()

    window = ti.ui.Window("test", (1024, 1024), vsync=True, show_window=False)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(-1.82731234, 2.26492691, 2.27800684)
    camera.lookat(-1.13230401, 2.11502124, 1.57480579)
    camera.fov(90)

    def render():
        scene.set_camera(camera)
        scene.point_light(pos=(0.5, 1, 2), color=(1, 1, 1))

        scene.mesh_instance(
            pos,
            tri,
            color=(39 / 255, 123 / 255, 192 / 255),
            two_sided=True,
            transforms=instances_transforms,
            instance_count=10,
            instance_offset=2,
        )
        canvas.scene(scene)

    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()
    verify_image(window.get_image_buffer_as_numpy(), "test_draw_part_of_mesh_instances")
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_wireframe_mode():
    N = 10
    NV = (N + 1) ** 2
    NT = 2 * N**2
    NE = 2 * N * (N + 1) + N**2
    pos = ti.Vector.field(3, ti.f32, shape=NV)
    tri = ti.field(ti.i32, shape=3 * NT)
    edge = ti.Vector.field(2, ti.i32, shape=NE)

    @ti.kernel
    def init_pos():
        for i, j in ti.ndrange(N + 1, N + 1):
            idx = i * (N + 1) + j
            pos[idx] = ti.Vector([i / N, 1.0 - j / N, 0.5])

    @ti.kernel
    def init_tri():
        for i, j in ti.ndrange(N, N):
            tri_idx = 6 * (i * N + j)
            pos_idx = i * (N + 1) + j
            if (i + j) % 2 == 0:
                tri[tri_idx + 0] = pos_idx
                tri[tri_idx + 1] = pos_idx + N + 2
                tri[tri_idx + 2] = pos_idx + 1
                tri[tri_idx + 3] = pos_idx
                tri[tri_idx + 4] = pos_idx + N + 1
                tri[tri_idx + 5] = pos_idx + N + 2
            else:
                tri[tri_idx + 0] = pos_idx
                tri[tri_idx + 1] = pos_idx + N + 1
                tri[tri_idx + 2] = pos_idx + 1
                tri[tri_idx + 3] = pos_idx + 1
                tri[tri_idx + 4] = pos_idx + N + 1
                tri[tri_idx + 5] = pos_idx + N + 2

    @ti.kernel
    def init_edge():
        for i, j in ti.ndrange(N + 1, N):
            edge_idx = i * N + j
            pos_idx = i * (N + 1) + j
            edge[edge_idx] = ti.Vector([pos_idx, pos_idx + 1])
        start = N * (N + 1)
        for i, j in ti.ndrange(N, N + 1):
            edge_idx = start + j * N + i
            pos_idx = i * (N + 1) + j
            edge[edge_idx] = ti.Vector([pos_idx, pos_idx + N + 1])
        start = 2 * N * (N + 1)
        for i, j in ti.ndrange(N, N):
            edge_idx = start + i * N + j
            pos_idx = i * (N + 1) + j
            if (i + j) % 2 == 0:
                edge[edge_idx] = ti.Vector([pos_idx, pos_idx + N + 2])
            else:
                edge[edge_idx] = ti.Vector([pos_idx + 1, pos_idx + N + 1])

    init_pos()
    init_tri()
    init_edge()

    window = ti.ui.Window("test", (1024, 1024), vsync=True, show_window=False)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(1.5, 1, -1)
    camera.lookat(1, 0.5, 0)
    camera.fov(90)

    def render():
        scene.set_camera(camera)
        scene.point_light(pos=(0.5, 1, 2), color=(1, 1, 1))

        scene.mesh(
            pos,
            tri,
            color=(39 / 255, 123 / 255, 192 / 255),
            two_sided=True,
            show_wireframe=True,
        )
        canvas.scene(scene)

    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()
    verify_image(window.get_image_buffer_as_numpy(), "test_wireframe_mode")
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_wireframe_mode_old():
    N = 10
    NV = (N + 1) ** 2
    NT = 2 * N**2
    NE = 2 * N * (N + 1) + N**2
    pos = ti.Vector.field(3, ti.f32, shape=NV)
    tri = ti.field(ti.i32, shape=3 * NT)
    edge = ti.Vector.field(2, ti.i32, shape=NE)

    @ti.kernel
    def init_pos():
        for i, j in ti.ndrange(N + 1, N + 1):
            idx = i * (N + 1) + j
            pos[idx] = ti.Vector([i / N, 1.0 - j / N, 0.5])

    @ti.kernel
    def init_tri():
        for i, j in ti.ndrange(N, N):
            tri_idx = 6 * (i * N + j)
            pos_idx = i * (N + 1) + j
            if (i + j) % 2 == 0:
                tri[tri_idx + 0] = pos_idx
                tri[tri_idx + 1] = pos_idx + N + 2
                tri[tri_idx + 2] = pos_idx + 1
                tri[tri_idx + 3] = pos_idx
                tri[tri_idx + 4] = pos_idx + N + 1
                tri[tri_idx + 5] = pos_idx + N + 2
            else:
                tri[tri_idx + 0] = pos_idx
                tri[tri_idx + 1] = pos_idx + N + 1
                tri[tri_idx + 2] = pos_idx + 1
                tri[tri_idx + 3] = pos_idx + 1
                tri[tri_idx + 4] = pos_idx + N + 1
                tri[tri_idx + 5] = pos_idx + N + 2

    @ti.kernel
    def init_edge():
        for i, j in ti.ndrange(N + 1, N):
            edge_idx = i * N + j
            pos_idx = i * (N + 1) + j
            edge[edge_idx] = ti.Vector([pos_idx, pos_idx + 1])
        start = N * (N + 1)
        for i, j in ti.ndrange(N, N + 1):
            edge_idx = start + j * N + i
            pos_idx = i * (N + 1) + j
            edge[edge_idx] = ti.Vector([pos_idx, pos_idx + N + 1])
        start = 2 * N * (N + 1)
        for i, j in ti.ndrange(N, N):
            edge_idx = start + i * N + j
            pos_idx = i * (N + 1) + j
            if (i + j) % 2 == 0:
                edge[edge_idx] = ti.Vector([pos_idx, pos_idx + N + 2])
            else:
                edge[edge_idx] = ti.Vector([pos_idx + 1, pos_idx + N + 1])

    init_pos()
    init_tri()
    init_edge()

    window = ti.ui.Window("test", (1024, 1024), vsync=True, show_window=False)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(1.5, 1, -1)
    camera.lookat(1, 0.5, 0)
    camera.fov(90)

    def render():
        scene.set_camera(camera)
        scene.point_light(pos=(0.5, 1, 2), color=(1, 1, 1))

        scene.mesh(
            pos,
            tri,
            color=(39 / 255, 123 / 255, 192 / 255),
            two_sided=True,
            show_wireframe=True,
        )
        canvas.scene(scene)

    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()
    verify_image(window.get_image_buffer_as_numpy(), "test_wireframe_mode")
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=[ti.vulkan])
def test_multi_windows():
    window = ti.ui.Window("x", (128, 128), vsync=True, show_window=False)
    window2 = ti.ui.Window("x2", (128, 128), vsync=True, show_window=False)
