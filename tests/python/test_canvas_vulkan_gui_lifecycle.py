"""GUI context, managed-target and opt-in owned Win32 window qualifications."""
import ctypes as C
from ctypes import wintypes as W
import os
import time
import uuid

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge._lib import core
from tests import test_utils
from tests.python.test_canvas_vulkan_execution_domain import _domains


BACKGROUND = np.array([17, 34, 51, 255], dtype=np.float32) / 255


def _widgets(gui, color, name="fixed"):
    with gui.sub_window(name, .04, .05, .45, .75) as panel:
        panel.text("MMMMMMMMMMMM", color=color)


def _assert_text_and_background(image, channel):
    others = [index for index in range(3) if index != channel]
    colored = (image[..., channel] > .65)
    for index in others:
        colored &= image[..., index] < .25
    assert np.count_nonzero(colored) > 20
    width, height = image.shape[:2]
    crop = image[3 * width // 4:7 * width // 8, height // 4:3 * height // 4]
    np.testing.assert_allclose(crop, np.broadcast_to(BACKGROUND, crop.shape), atol=1/255)


class SelectedWindow:
    def __init__(self, domain, *, visible=False, name="GUI context"):
        self.window = ti.ui.Window(name, (320, 240), show_window=visible, vsync=False)
        self.canvas = self.window.get_canvas()
        self.canvas.set_execution_domain(domain)
        self.pixels = ti.ndarray(ti.u32, (320, 240))
        self.pixels.fill(0xFF332211)
        ti.sync()  # Cold, before any pending/gated qualification.
        self.frame = ti.ui.DisplayFrame.from_packed_u32_ndarray(self.pixels)
        self.canvas.prepare_frame_sources([self.frame], capacity=3)
        self.source = domain.latest_completion()
        self.gui = self.window.get_gui()
        self.first = True

    def capture(self, *, visible=False, widgets=None):
        if self.first:
            ticket = self.canvas.submit_frame(self.frame, source_completion=self.source,
                                             track_source=True, track_completion=True)
            self.first = False
        else:
            ticket = self.canvas.repeat_frame()
        assert ticket is not None
        if widgets is not None:
            # Visible Canvas admission requires an empty pending frame. Queue
            # its retained image before adding GUI commands to that frame.
            widgets()
        image = self.window.get_image_buffer_as_numpy()
        if visible:
            assert self.window.show()  # Presents the already-read current frame.
        assert ticket.done()
        return image

    def close(self):
        self.window.destroy()
        self.source.close()


@pytest.mark.skipif(not core.GGUI_AVAILABLE, reason="GGUI not available")
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_selected_gui_two_contexts_interleaved_widgets_and_survivor():
    domain, = _domains(1, capacity=4, instance_capacity=2)
    windows = []
    try:
        left = SelectedWindow(domain, name="left GUI context")
        windows.append(left)
        right = SelectedWindow(domain, name="right GUI context")
        windows.append(right)
        left.gui.set_font_scale(1.0)
        right.gui.set_font_scale(1.5)
        # Both frames have live context-local Begin/End stacks. Retained Gui
        # wrappers deliberately alternate calls without another get_gui().
        for index in range(3):
            with left.gui.sub_window("same-name", .04, .05, .45, .75) as a:
                with right.gui.sub_window("same-name", .04, .05, .45, .75) as b:
                    a.text("MMMMMMMMMMMM", color=(1, 0, 0))
                    b.text("MMMMMMMMMMMM", color=(0, 1, 0))
                    a.text("left context", color=(1, 0, 0))
                a.text("left still open", color=(1, 0, 0))
            # Reverse render order every time, independently of widget order.
            order = [(right, 1), (left, 0)] if index % 2 == 0 else [(left, 0), (right, 1)]
            for fixture, channel in order:
                _assert_text_and_background(fixture.capture(), channel)
            assert left.gui.get_font_scale() == pytest.approx(1.0)
            assert right.gui.get_font_scale() == pytest.approx(1.5)
        right.close()
        windows.remove(right)
        _widgets(left.gui, (0, 0, 1), name="survivor")
        _assert_text_and_background(left.capture(), 2)
    finally:
        for fixture in reversed(windows):
            fixture.close()
        domain.close()


@pytest.mark.parametrize("gui_before_binding", [False, True])
@pytest.mark.skipif(not core.GGUI_AVAILABLE, reason="GGUI not available")
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_ordinary_runtime_offscreen_gui_uses_managed_targets(gui_before_binding):
    from taichi_forge.hardware._raster_readback import color_numpy

    shape = (320, 240)
    color = ti.Texture(ti.Format.rgba8, shape)
    depth = ti.Texture(ti.Format.depth32f, shape)
    window = ti.ui.Window("ordinary target GUI", shape, show_window=False)
    try:
        gui = window.get_gui() if gui_before_binding else None
        # Same native cold binding used by the existing RasterPass, without
        # adding GUI to RasterPass's public API or selecting an output domain.
        window.window._set_offscreen_targets(color.tex, depth.tex)
        if gui is None:
            gui = window.get_gui()
        canvas = window.get_canvas()
        canvas.set_background_color(tuple(float(x) for x in BACKGROUND[:3]))
        for channel in range(3):
            window.window._begin_offscreen_frame()
            rgb = tuple(int(index == channel) for index in range(3))
            _widgets(gui, rgb)
            assert window.window._render_offscreen_frame()
            image = color_numpy(color)
            _assert_text_and_background(image, channel)
        window.destroy()
        # Closing GUI/renderer retires readers; it does not delete caller-owned
        # Program textures. Do not use Window screenshot after target binding.
        np.testing.assert_array_equal(color_numpy(color), image)
    finally:
        window.destroy()
        color._delete_runtime_texture()
        depth._delete_runtime_texture()


class OwnWin32Window:
    def __init__(self, title):
        self.user32 = C.WinDLL("user32", use_last_error=True)
        self.user32.FindWindowW.argtypes = [W.LPCWSTR, W.LPCWSTR]
        self.user32.FindWindowW.restype = W.HWND
        self.user32.GetWindowThreadProcessId.argtypes = [W.HWND, C.POINTER(W.DWORD)]
        self.user32.GetWindowThreadProcessId.restype = W.DWORD
        self.user32.GetClientRect.argtypes = [W.HWND, C.POINTER(W.RECT)]
        self.user32.GetClientRect.restype = W.BOOL
        self.user32.GetWindowRect.argtypes = [W.HWND, C.POINTER(W.RECT)]
        self.user32.GetWindowRect.restype = W.BOOL
        self.user32.SetWindowPos.argtypes = [W.HWND, W.HWND, C.c_int, C.c_int, C.c_int, C.c_int, W.UINT]
        self.user32.SetWindowPos.restype = W.BOOL
        self.user32.ShowWindow.argtypes = [W.HWND, C.c_int]
        self.user32.ShowWindow.restype = W.BOOL
        self.user32.IsIconic.argtypes = [W.HWND]
        self.user32.IsIconic.restype = W.BOOL
        self.hwnd = self.user32.FindWindowW(None, title)
        assert self.hwnd, "fixture's unique native window not found before first show"
        pid = W.DWORD()
        self.user32.GetWindowThreadProcessId(self.hwnd, C.byref(pid))
        assert pid.value == os.getpid(), "refuse to operate another process's window"

    def resize_client(self, width, height):
        client, outer = W.RECT(), W.RECT()
        assert self.user32.GetClientRect(self.hwnd, C.byref(client))
        assert self.user32.GetWindowRect(self.hwnd, C.byref(outer))
        width += (outer.right - outer.left) - (client.right - client.left)
        height += (outer.bottom - outer.top) - (client.bottom - client.top)
        # No move, no z-order change, no activation; only this known HWND.
        assert self.user32.SetWindowPos(self.hwnd, None, 0, 0, width, height, 0x16)

    def minimize(self):
        self.user32.ShowWindow(self.hwnd, 6)

    def restore(self):
        self.user32.ShowWindow(self.hwnd, 9)

    def iconic(self):
        return bool(self.user32.IsIconic(self.hwnd))


def _pump_until(window, predicate, description):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        window.poll_events()
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError(description)


@pytest.mark.skipif(os.name != "nt" or os.environ.get("TI_TEST_VISIBLE_GUI") != "1",
                    reason="explicit opt-in Win32 visible qualification")
@pytest.mark.skipif(not core.GGUI_AVAILABLE, reason="GGUI not available")
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_selected_gui_win32_resize_minimize_restore():
    domain, = _domains(1, capacity=4, instance_capacity=1)
    fixture = None
    try:
        title = f"Forge GUI resize {os.getpid()} {uuid.uuid4().hex}"
        fixture = SelectedWindow(domain, visible=True, name=title)
        native = OwnWin32Window(title)  # Cache before FPS suffix can change it.
        _assert_text_and_background(fixture.capture(
            visible=True, widgets=lambda: _widgets(fixture.gui, (1, 0, 0))), 0)
        sequence = domain.latest_completion().info["sequence"]
        for shape in [(384, 288), (320, 240)]:
            native.resize_client(*shape)
            _pump_until(fixture.window, lambda: tuple(fixture.window.get_window_shape()) == shape,
                        f"framebuffer resize did not reach {shape}")
            image = fixture.capture(
                visible=True, widgets=lambda: _widgets(fixture.gui, (0, 1, 0)))
            assert image.shape[:2] == shape
            _assert_text_and_background(image, 1)

        # Retained image + populated GUI CPU frame must be dropped while
        # minimized, without consuming another source/copy prefix.
        dropped = fixture.canvas.repeat_frame()
        assert dropped is not None
        _widgets(fixture.gui, (1, 0, 0))
        before = fixture.window.get_display_stats()["dropped_frames"]
        native.minimize()
        _pump_until(fixture.window, native.iconic, "fixture did not minimize")
        assert not fixture.window.show()
        assert dropped.status == "cancelled"
        assert fixture.window.get_display_stats()["dropped_frames"] == before + 1
        native.restore()
        _pump_until(fixture.window, lambda: not native.iconic()
                    and tuple(fixture.window.get_window_shape()) == (320, 240),
                    "fixture did not restore its framebuffer")
        _assert_text_and_background(fixture.capture(
            visible=True, widgets=lambda: _widgets(fixture.gui, (0, 0, 1))), 2)
        assert domain.latest_completion().info["sequence"] == sequence
    finally:
        if fixture is not None:
            fixture.close()
        domain.close()
