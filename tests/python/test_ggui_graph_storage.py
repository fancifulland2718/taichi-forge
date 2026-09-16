import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge._lib import core as _ti_core
from taichi_forge.lang import impl
from tests import test_utils


def _vulkan_graph_canvas_ordering(kind):
    ti.init(arch=ti.vulkan, offline_cache=False)

    @ti.kernel
    def produce_buffer(image: ti.types.ndarray(), value: ti.i32):
        for i, j in image:
            if ti.static(kind == "packed"):
                image[i, j] = ti.u32(0xFF000000) | (ti.u32(value) * 255)
            else:
                image[i, j] = ti.Vector([ti.cast(value, ti.f32), 0.0, 0.0, 1.0])

    @ti.kernel
    def produce_texture(image: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.rgba8), value: ti.i32):
        for i, j in image:
            image.store(ti.Vector([i, j]), ti.Vector([ti.cast(value, ti.f32), 0.0, 0.0, 1.0]))

    @ti.kernel
    def advance(state: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in state:
            state[i] += 1

    builder = ti.graph.GraphBuilder()
    value = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "value", ti.i32)
    if kind == "texture":
        target = ti.graph.Arg(ti.graph.ArgKind.RWTEXTURE, "image", ndim=2, fmt=ti.Format.rgba8)
        builder.dispatch(produce_texture, target, value)
    else:
        dtype = ti.u32 if kind == "packed" else ti.types.vector(4, ti.f32)
        target = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "image", dtype, ndim=2)
        builder.dispatch(produce_buffer, target, value)
    graph = builder.compile()
    sim_builder = ti.graph.GraphBuilder()
    sim_builder.dispatch(advance, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "state", ti.i32, ndim=1))
    sim_graph = sim_builder.compile()
    state = ti.ndarray(ti.i32, 128)
    state.fill(0)
    sim_bound = sim_graph.bind({"state": state})
    window = ti.ui.Window("Graph Canvas ordering", (16, 12), show_window=False)
    canvas = window.get_canvas()
    stop = threading.Event()
    errors = []
    iterations = []
    # Match the application's public host-launch contract. This serializes
    # only invocation admission, not GPU execution or whole rendering frames.
    pacer = ti.graph.SubmissionPacer(2, max_in_flight_per_lane=1, max_queued=8)

    def simulate():
        try:
            while not stop.is_set():
                sim_graph.submit(sim_bound, pacer=pacer, lane="simulation")
                iterations.append(1)
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=simulate, daemon=True)
    worker.start()
    tickets = []
    try:
        for shape in ((13, 9), (7, 11)):
            image = ti.Texture(ti.Format.rgba8, shape) if kind == "texture" else ti.ndarray(dtype, shape)
            bindings = [graph.bind({"image": image, "value": value}) for value in (0, 1)]
            for frame in range(12):
                value = frame % 2
                tickets.append(graph.submit(bindings[value], pacer=pacer, lane="render"))
                # No producer wait/sync. Conversion/copy and graphics consume
                # the existing Program.flush semaphore, even for cached Graphs.
                if kind == "packed":
                    assert canvas.submit_frame(ti.ui.DisplayFrame.from_packed_u32_ndarray(image))
                else:
                    assert canvas.set_image(image)
                observed = window.get_image_buffer_as_numpy()
                np.testing.assert_allclose(observed, np.broadcast_to([value, 0, 0, 1], observed.shape), atol=1 / 255)
                # Readback is the display-consumer completion boundary. Only
                # now reuse/replace this source; a producer ticket is not enough.
    finally:
        stop.set()
        worker.join(5)
        assert not worker.is_alive()
        graph.close()
        sim_graph.close()
        window.destroy()
    assert not errors, errors
    assert iterations
    assert all(ticket.done() for ticket in tickets)
    np.testing.assert_array_equal(state.to_numpy(), np.full(128, len(iterations), np.int32))
    ti.reset()


@pytest.mark.parametrize("kind", ["packed", "float", "texture"])
@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=ti.vulkan)
def test_cached_graph_canvas_ordering_with_paced_simulation(kind):
    # A native concurrency regression must not hang the test runner.
    code = (
        "import sys, types\n"
        f"sys.path[:] = {json.dumps(sys.path)}\n"
        "package = types.ModuleType('taichi_forge._lib.core')\n"
        f"package.__path__ = [{json.dumps(str(Path(_ti_core.__file__).parent))}]\n"
        "sys.modules[package.__name__] = package\n"
        "from tests.python.test_ggui_graph_storage import _vulkan_graph_canvas_ordering\n"
        f"_vulkan_graph_canvas_ordering({kind!r})\n"
    )
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=[ti.cuda])
def test_graph_borrowed_display_storage_retention_and_retirement():
    window = ti.ui.Window("graph display", (12, 20), show_window=False)
    canvas = window.get_canvas()
    program = impl.get_runtime().prog
    baseline = program._debug_external_dense_storage_stats()

    @ti.kernel
    def paint(target: ti.types.ndarray(dtype=ti.u32, ndim=2), value: ti.u32):
        for i, j in target:
            target[i, j] = value

    builder = ti.graph.GraphBuilder()
    target = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "target", ti.u32, ndim=2)
    value_arg = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "value", ti.u32)
    builder.dispatch(paint, target, value_arg)
    builder.dispatch(paint, target, value_arg)
    graph = builder.compile()
    saved = None
    try:
        for value in (0xFF0000FF, 0xFF00FF00, 0xFF332211):
            with canvas.acquire_frame(12, 20) as frame:
                bound = graph.bind(dict(target=frame.pixels, value=value))
                # Both dispatches share one external producer epoch; no global
                # inflight lease may prevent display-consumer retirement.
                graph.run(bound)
                assert canvas.submit_frame(frame, track_source=True)
            frame.source_completion.wait()
            assert window.show()
            frame.completion.wait()
            saved = bound
        canvas.repeat_frame()
        image = window.get_image_buffer_as_numpy()
        np.testing.assert_allclose(
            image, np.broadcast_to(np.array([17, 34, 51, 255]) / 255, image.shape), atol=1 / 255 + 1e-5
        )
        # A cancelled epoch also has to release the Graph's local leases.
        with canvas.acquire_frame(12, 20) as cancelled:
            graph.run(graph.bind(dict(target=cancelled.pixels, value=0)))
        assert cancelled.completion.status == "cancelled"
        window.destroy()
        with pytest.raises(RuntimeError, match="retired|stale|closed|invalid"):
            graph.run(saved)
    finally:
        graph.close()
        window.destroy()
    ti.sync()
    final = program._debug_external_dense_storage_stats()
    for key in ("live", "retiring", "leases"):
        assert final[key] == baseline[key]


def _continuous_display_ring(kind, *, visible=False, window_action=None):
    """Reuse application storage only after the corresponding display read."""
    shape = (24, 16)

    @ti.kernel
    def paint_buffer(image: ti.types.ndarray(dtype=ti.u32, ndim=2), value: ti.i32):
        for i, j in image:
            image[i, j] = ti.u32(0xFF000000) | ti.u32(value)

    @ti.kernel
    def paint_texture(image: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.rgba8), value: ti.i32):
        for i, j in image:
            image.store(ti.Vector([i, j]), ti.Vector([ti.cast(value, ti.f32) / 255, 0.0, 0.0, 1.0]))

    builder = ti.graph.GraphBuilder()
    value = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "value", ti.i32)
    if kind == "texture":
        arg = ti.graph.Arg(ti.graph.ArgKind.RWTEXTURE, "image", ndim=2, fmt=ti.Format.rgba8)
        builder.dispatch(paint_texture, arg, value)
        images = [ti.Texture(ti.Format.rgba8, shape) for _ in range(3)]
        frames = [ti.ui.DisplayFrame.from_texture(image) for image in images]
    else:
        arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "image", ti.u32, ndim=2)
        builder.dispatch(paint_buffer, arg, value)
        images = [ti.ndarray(ti.u32, shape) for _ in range(3)]
        frames = [ti.ui.DisplayFrame.from_packed_u32_ndarray(image) for image in images]
    graph = builder.compile()
    window = ti.ui.Window("Forge continuous display lifecycle", (320, 240) if visible else shape,
                          show_window=visible, vsync=False, fps_limit=65535)
    canvas = window.get_canvas()
    previous = [None] * 3
    saved = []
    try:
        for index in range(96):
            slot = index % len(images)
            completion = previous[slot]
            if completion is not None and completion.status != "cancelled":
                # No producer wait, global sync or readback in the steady loop.
                if not completion.done():
                    completion.wait()
                assert completion.done()
            if window_action is not None:
                window_action(index, window)
            graph.submit({"image": images[slot], "value": index})
            completion = canvas.submit_frame(frames[slot], track_completion=True)
            previous[slot] = completion
            if completion is not None:
                assert isinstance(completion, ti.ui.DisplayCompletion)
                assert completion.status == "pending"
                if index == 0:
                    with pytest.raises(RuntimeError, match="not been submitted"):
                        completion.wait()
                saved.append(completion)
            shown = window.show()
            if completion is not None and not shown:
                assert completion.status == "cancelled"
        for completion in previous:
            if completion is not None and completion.status != "cancelled":
                completion.wait()
        # Readback is only an end-of-run pixel oracle, never the reuse boundary.
        if not visible:
            # show() has consumed the pending image; explicitly draw again for
            # the oracle rather than reading a new, empty offscreen frame.
            assert canvas.submit_frame(frames[2]) is True
            observed = window.get_image_buffer_as_numpy()
            np.testing.assert_allclose(observed, np.broadcast_to([95 / 255, 0, 0, 1], observed.shape), atol=1 / 255)
            superseded = canvas.submit_frame(frames[0], track_completion=True)
            assert canvas.submit_frame(frames[1]) is True
            assert superseded.status == "cancelled"
            with pytest.raises(RuntimeError, match="cancelled"):
                superseded.wait()
            assert window.show()
        # Keep a submitted ticket alive across teardown; no explicit wait.
        final = None
        for _ in range(100):
            completion = canvas.submit_frame(frames[2], track_completion=True)
            if window.show() and completion is not None:
                final = completion
                break
            if completion is not None:
                assert completion.status == "cancelled"
            time.sleep(0.001)
        assert final is not None, "Window failed to accept a frame after restoration"
        if window_action is not None:
            window_action(96, window)
        pending = canvas.submit_frame(frames[0], track_completion=True)
        if window_action is not None:
            assert not window.show()
            assert not window.running
        window.destroy()
        assert final.done()
        if pending is not None:
            assert pending.status == "cancelled"
        assert all(ticket.status == "cancelled" or ticket.done() for ticket in saved)
        with pytest.raises(RuntimeError, match="closed"):
            canvas.submit_frame(frames[0], track_completion=True)
    finally:
        window.destroy()
        graph.close()


@pytest.mark.parametrize("kind", ["packed", "texture"])
@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=ti.vulkan)
def test_display_completion_drives_continuous_graph_storage_reuse(kind):
    code = (
        "import sys, types\n"
        f"sys.path[:] = {json.dumps(sys.path)}\n"
        "package = types.ModuleType('taichi_forge._lib.core')\n"
        f"package.__path__ = [{json.dumps(str(Path(_ti_core.__file__).parent))}]\n"
        "sys.modules[package.__name__] = package\n"
        "import taichi_forge as ti\n"
        "ti.init(arch=ti.vulkan, offline_cache=False)\n"
        "from tests.python.test_ggui_graph_storage import _continuous_display_ring\n"
        f"_continuous_display_ring({kind!r})\n"
    )
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr


def _windows_display_lifecycle(kind):
    import ctypes
    from ctypes import wintypes

    user = ctypes.WinDLL("user32", use_last_error=True)
    user.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
    user.FindWindowW.restype = wintypes.HWND
    user.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, ctypes.c_int, wintypes.UINT]
    user.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user.IsIconic.argtypes = [wintypes.HWND]
    user.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    handle = None

    def action(index, window):
        nonlocal handle
        if handle is None:
            handle = user.FindWindowW(None, "Forge continuous display lifecycle")
            assert handle
        pid = wintypes.DWORD()
        user.GetWindowThreadProcessId(handle, ctypes.byref(pid))
        assert pid.value == os.getpid(), "Only manipulate the test-owned window"
        if index in (8, 48):
            width, height = (460, 310) if index == 8 else (640, 420)
            assert user.SetWindowPos(handle, None, 100, 100, width, height, 0x0014)
        elif index == 24:
            user.ShowWindow(handle, 6)
            assert user.IsIconic(handle)
            assert not window.window.can_render_frame()
        elif index == 32:
            user.ShowWindow(handle, 9)
            assert not user.IsIconic(handle)
        elif index == 96:
            # Closing a minimized window must not require restoring it first.
            user.ShowWindow(handle, 6)
            assert user.PostMessageW(handle, 0x0010, 0, 0)
        time.sleep(0.01)

    _continuous_display_ring(kind, visible=True, window_action=action)


@pytest.mark.skipif(os.name != "nt" or os.environ.get("TI_VISIBLE_DISPLAY_TESTS") != "1",
                    reason="Requires an explicitly enabled Windows desktop session")
@pytest.mark.parametrize("kind", ["packed", "texture"])
@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=ti.vulkan)
def test_visible_display_resize_minimize_restore_close(kind):
    code = (
        "import sys, types, faulthandler\n"
        "faulthandler.dump_traceback_later(20, exit=True)\n"
        f"sys.path[:] = {json.dumps(sys.path)}\n"
        "package = types.ModuleType('taichi_forge._lib.core')\n"
        f"package.__path__ = [{json.dumps(str(Path(_ti_core.__file__).parent))}]\n"
        "sys.modules[package.__name__] = package\n"
        "import taichi_forge as ti\n"
        "ti.init(arch=ti.vulkan, offline_cache=False)\n"
        "from tests.python.test_ggui_graph_storage import _windows_display_lifecycle\n"
        f"_windows_display_lifecycle({kind!r})\n"
        "ti.reset()\n"
        "faulthandler.cancel_dump_traceback_later()\n"
    )
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
