import os
import sys

import pytest

import taichi_forge as ti
from taichi_forge._lib import core
from tests import test_utils


@pytest.mark.skipif(not core.GGUI_AVAILABLE, reason="GGUI not available")
@test_utils.test(arch=[ti.vulkan])
def test_scroll_headless_and_closed_window():
    window = ti.ui.Window("scroll lifecycle", (64, 64), show_window=False)
    native = window.window
    try:
        assert window.get_scroll_delta() == (0.0, 0.0)
        assert native.get_scroll_delta() == (0.0, 0.0)
        assert window.get_scroll_delta() == (0.0, 0.0)
    finally:
        window.destroy()
    with pytest.raises(RuntimeError, match="closed"):
        window.get_scroll_delta()
    with pytest.raises(RuntimeError, match="closed"):
        native.get_scroll_delta()


@pytest.mark.skipif(
    sys.platform != "win32" or os.environ.get("TI_RUN_HEADED_GGUI_TESTS") != "1",
    reason="requires an opt-in Windows desktop session",
)
@pytest.mark.skipif(not core.GGUI_AVAILABLE, reason="GGUI not available")
@test_utils.test(arch=[ti.vulkan])
def test_scroll_windows_event_pump_and_window_isolation():
    import ctypes
    import time
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows.argtypes = (enum_proc, wintypes.LPARAM)
    user32.GetWindowTextW.argtypes = (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)
    user32.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
    user32.PostMessageW.argtypes = (wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
    user32.PostMessageW.restype = wintypes.BOOL
    user32.ClientToScreen.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.POINT))
    user32.GetCursorPos.argtypes = (ctypes.POINTER(wintypes.POINT),)
    user32.SetCursorPos.argtypes = (ctypes.c_int, ctypes.c_int)
    user32.SetForegroundWindow.argtypes = (wintypes.HWND,)
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.SetWindowPos.argtypes = (
        wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, wintypes.UINT,
    )
    old_position = wintypes.POINT()
    assert user32.GetCursorPos(ctypes.byref(old_position))
    old_foreground = user32.GetForegroundWindow()

    def handle(title):
        matches = []

        @enum_proc
        def visit(hwnd, _):
            buffer = ctypes.create_unicode_buffer(256)
            pid = wintypes.DWORD()
            user32.GetWindowTextW(hwnd, buffer, len(buffer))
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if pid.value == os.getpid() and buffer.value == title:
                matches.append(hwnd)
            return True

        user32.EnumWindows(visit, 0)
        assert len(matches) == 1
        return matches[0]

    def post(hwnd, message, delta):
        assert user32.PostMessageW(hwnd, message, (delta & 0xFFFF) << 16, 0)

    first = second = headless = None
    try:
        first = ti.ui.Window("Forge scroll first", (256, 192), pos=(100, 100))
        first_handle = handle("Forge scroll first")
        headless = ti.ui.Window("Forge scroll headless", (64, 64), show_window=False)
        second = ti.ui.Window("Forge scroll second", (256, 192), pos=(400, 100))
        second_handle = handle("Forge scroll second")
        headless.get_gui().text("Temporarily activate a context without a GLFW backend")
        first.poll_events()
        first.get_scroll_delta()
        second.get_scroll_delta()
        post(first_handle, 0x020A, 15)
        post(first_handle, 0x020A, -60)
        post(first_handle, 0x020E, -30)
        post(second_handle, 0x020A, 120)
        # Reading never pumps the pending OS event queue.
        assert first.get_scroll_delta() == (0.0, 0.0)
        assert second.get_scroll_delta() == (0.0, 0.0)
        second.poll_events()
        assert first.get_scroll_delta() == (0.25, -0.375)
        assert second.window.get_scroll_delta() == (0.0, 1.0)
        assert first.get_scroll_delta() == second.get_scroll_delta() == (0.0, 0.0)
        assert not first.is_pressed(ti.ui.MMB)
        assert not second.is_pressed(ti.ui.MMB)
        headless.destroy()
        first.get_gui().text("Close this context while the other window survives")
        first.destroy()
        post(second_handle, 0x020A, -30)
        second.poll_events()
        assert second.get_scroll_delta() == (0.0, -0.25)
        second.show()

        gui = second.get_gui()

        def draw_panel():
            with gui.sub_window("Scrollable UI", 0.05, 0.05, 0.55, 0.75):
                for row in range(30):
                    gui.text(f"Row {row}")
            second.show()

        def move_cursor(x, y):
            point = wintypes.POINT(x, y)
            assert user32.ClientToScreen(second_handle, ctypes.byref(point))
            assert user32.SetCursorPos(point.x, point.y)

        user32.SetForegroundWindow(second_handle)
        move_cursor(60, 80)
        for _ in range(4):
            draw_panel()
        assert not second.is_render_input_available()
        post(second_handle, 0x020A, -120)
        draw_panel()
        assert second.get_scroll_delta() == (0.0, -1.0)
        assert not second.is_render_input_available()
        # Discard captured input; leaving the UI must not replay it to a camera.
        move_cursor(230, 170)
        # SetCursorPos and the GLFW/ImGui event queue need not settle within a
        # fixed number of unpaced frames. Wait for input delivery, without a GPU
        # synchronization or any additional read of the scroll accumulator.
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            draw_panel()
            if second.is_render_input_available():
                break
            time.sleep(0.005)
        assert second.is_render_input_available(), second.get_cursor_pos()
        assert second.get_scroll_delta() == (0.0, 0.0)
        # Resize immediately after presentation. Validation must accept retiring
        # both render work and present waits, then closing the rebuilt swapchain.
        assert user32.SetWindowPos(second_handle, None, 0, 0, 300, 230, 0x0016)
        second.poll_events()
        for _ in range(3):
            draw_panel()
    finally:
        if first is not None:
            first.destroy()
        if second is not None:
            second.destroy()
        if headless is not None:
            headless.destroy()
        user32.SetCursorPos(old_position.x, old_position.y)
        if old_foreground:
            user32.SetForegroundWindow(old_foreground)
