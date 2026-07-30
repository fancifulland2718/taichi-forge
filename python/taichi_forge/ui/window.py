import pathlib
import weakref

import numpy
from taichi_forge._kernels import (
    arr_vulkan_layout_to_arr_normal_layout,
    arr_vulkan_layout_to_field_normal_layout,
)
from taichi_forge._lib import core as _ti_core
from taichi_forge.lang._ndarray import Ndarray
from taichi_forge.lang.impl import Field, default_cfg, get_runtime
from taichi_forge.ui.staging_buffer import get_depth_ndarray, remove_window_staging_cache

from taichi_forge import f32

from .canvas import Canvas
from .scene import SceneV2
from .constants import PRESS, RELEASE
from .imgui import Gui
from .layout import EdgeRegion, WindowLayout
from .utils import check_ggui_availability


_active_windows = weakref.WeakSet()


def _destroy_all_windows():
    for window in tuple(_active_windows):
        window.destroy()


class Window:
    """The window class.

    Args:
        name (str): Window title.
        res (tuple[int]): resolution (width, height) of the window, in pixels.
        vsync (bool): whether or not vertical sync should be enabled.
        show_window (bool): where or not display the window after initialization.
        pos (tuple[int]): position (left to right, up to bottom) of the window which origins from the left-top of your main screen, in pixels.
    """

    def __init__(self, name, res, vsync=False, show_window=True, fps_limit=65535, pos=(100, 100)):
        check_ggui_availability()
        package_path = str(pathlib.Path(__file__).parent.parent)
        ti_arch = default_cfg().arch
        self.window = _ti_core.PyWindow(
            get_runtime().prog,
            name,
            res,
            pos,
            vsync,
            show_window,
            fps_limit,
            package_path,
            ti_arch,
        )
        self._layout = WindowLayout(self)
        _active_windows.add(self)

    @property
    def running(self):
        """Check whether this window is running or not."""
        return self.window.is_running()

    @running.setter
    def running(self, value):
        """Set the running status of this window.

        Example::

            >>> window.running = False
        """
        self.window.set_is_running(value)

    @property
    def event(self):
        """Get the current unprocessed event."""
        return self.window.get_current_event()

    @event.setter
    def event(self, value):
        """Set the current unprocessed event."""
        self.window.set_current_event(value)

    def poll_events(self):
        """Poll window events without drawing or presenting a frame.

        ``show()`` already polls events once per frame. This method is only
        needed by custom loops that want an explicit event-pump point.
        """
        return self.window.poll_events()

    def get_events(self, tag=None, poll=True):
        """Get the current list of unprocessed events.

        Args:
            tag (str): A tag used for filtering events. \
                If it is None, then all events are returned.
            poll (bool): Whether to poll GLFW events before draining the queue.
                Set this to ``False`` in high-throughput render loops that use
                ``show()`` as the single event-pump point.
        """
        if tag is None:
            return self.window.get_events(_ti_core.EventType.Any, poll)
        if tag is PRESS:
            return self.window.get_events(_ti_core.EventType.Press, poll)
        if tag is RELEASE:
            return self.window.get_events(_ti_core.EventType.Release, poll)
        raise Exception("unrecognized event tag")

    def get_event(self, tag=None, poll=True):
        """Returns whether or not a event that matches tag has occurred.

        If tag is None, then no filters are applied. If this function
        returns `True`, the `event` property of the window will be set
        to the corresponding event.

        Args:
            tag (str): A tag used for filtering events.
            poll (bool): Whether to poll GLFW events before draining the queue.
                Set this to ``False`` in high-throughput render loops that use
                ``show()`` as the single event-pump point.
        """
        if tag is None:
            return self.window.get_event(_ti_core.EventType.Any, poll)
        if tag is PRESS:
            return self.window.get_event(_ti_core.EventType.Press, poll)
        if tag is RELEASE:
            return self.window.get_event(_ti_core.EventType.Release, poll)
        raise Exception("unrecognized event tag")

    def is_pressed(self, *keys):
        """Checks if any of a set of specified keys is pressed.

        Args:
            keys (list[:mod:`~taichi_forge.ui.constants`]): The keys to be matched.

        Returns:
            bool: `True` if any key among `keys` is pressed, else `False`.
        """
        for k in keys:
            if self.window.is_pressed(k):
                return True
        return False

    def get_canvas(self):
        """Returns a canvas handle. See :class`~taichi_forge.ui.canvas.Canvas`"""
        return Canvas(self.window.get_canvas(), self.window)

    def get_scene(self):
        """Returns a scene handle. See :class`~taichi_forge.ui.scene.SceneV2`"""
        return SceneV2(self.window.get_scene())

    @property
    def GUI(self):
        """Returns a IMGUI handle. See :class`~taichi_forge.ui.ui.Gui` This is an
        deprecated interface, please use `~taichi_forge.ui.Window.get_gui` instead.
        """
        return self.get_gui()

    def get_gui(self):
        """Returns a IMGUI handle. See :class`~taichi_forge.ui.ui.Gui`"""
        return Gui(self.window.GUI())

    def configure_layout(
        self,
        *,
        top=None,
        bottom=None,
        left=None,
        right=None,
        minimum_render_size=(1, 1),
    ):
        """Configure optional Window-owned edge regions.

        Each argument is an :class:`EdgeRegion`, a compatible ``dict``, or
        ``None`` to disable that edge. The central render viewport is the
        remaining rectangle after top/bottom and left/right regions.
        """

        for edge, config in (
            ("top", top),
            ("bottom", bottom),
            ("left", left),
            ("right", right),
        ):
            self._layout.configure_region(edge, config)
        self._layout.set_minimum_render_size(*minimum_render_size)
        return self._layout

    def get_layout(self):
        """Return this Window's persistent root-layout facade."""

        return self._layout

    def get_cursor_pos(self):
        """Get current cursor position, in the range `[0, 1] x [0, 1]`."""
        return self.window.get_cursor_pos()

    def get_render_cursor_pos(self, clamp=False):
        """Return cursor coordinates normalized to the render viewport."""

        return self.window.get_render_cursor_pos(clamp)

    def is_cursor_in_render_viewport(self):
        """Return whether the cursor lies inside the central render viewport."""

        return self.window.is_cursor_in_render_viewport()

    def is_render_input_available(self):
        """Return whether render interaction is not captured by edge UI."""

        return self.window.is_render_input_available()

    def show(self):
        """Display this window."""
        return self.window.show()

    def is_headless_display(self):
        """Return whether this window uses the offscreen display sink."""
        return self.window.is_headless_display()

    def get_display_stats(self):
        """Return display submission statistics for set_image/show."""
        return self.window.get_display_stats()

    def reset_display_stats(self):
        """Reset display submission statistics."""
        return self.window.reset_display_stats()

    def get_window_shape(self):
        """Return the shape of window.
        Return:
            tuple : (width, height)
        """
        return self.window.get_window_shape()

    def save_image(self, filename):
        """Save the window content to an image file.

        Args:
            filename (str): output filename.
        """
        return self.window.write_image(filename)

    def get_depth_buffer(self, depth):
        """fetch the depth information of current scene to ti.ndarray/ti.field
           (support copy from vulkan to cuda/cpu which is a faster version)
        Args:
            depth(ti.ndarray/ti.field): [window_width, window_height] carries depth information.
        """
        if not (len(depth.shape) == 2 and depth.dtype == f32):
            raise Exception("Only Support 2d-shape and ti.f32 data format.")
        if not isinstance(depth, (Ndarray, Field)):
            raise Exception("Only Support Ndarray and Field data type.")
        tmp_depth = get_depth_ndarray(self.window)
        self.window.copy_depth_buffer_to_ndarray(tmp_depth.arr)
        if isinstance(depth, Ndarray):
            arr_vulkan_layout_to_arr_normal_layout(tmp_depth, depth)
        else:
            arr_vulkan_layout_to_field_normal_layout(tmp_depth, depth)

    def get_depth_buffer_as_numpy(self):
        """Get the depth information of current scene to numpy array.

        Returns:
            2d numpy array: [width, height] with (0.0~1.0) float-format.
        """
        tmp_depth = get_depth_ndarray(self.window)
        self.window.copy_depth_buffer_to_ndarray(tmp_depth.arr)
        w, h = self.get_window_shape()
        return numpy.ascontiguousarray(tmp_depth.to_numpy().reshape(h, w)[::-1, :].T)

    def get_image_buffer_as_numpy(self):
        """Get the window content to numpy array.

        Returns:
            3d numpy array: [width, height, channels] with (0.0~1.0) float-format color.
        """
        return self.window.get_image_buffer_as_numpy()

    def destroy(self):
        """Destroy this window. The window will be unavailable then."""
        if self.window is None:
            return None
        window = self.window
        self.window = None
        _active_windows.discard(self)
        remove_window_staging_cache(window)
        return window.destroy()
