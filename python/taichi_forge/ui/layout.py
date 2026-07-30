from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass


_EDGES = ("top", "bottom", "left", "right")


def _normalize_edge(edge):
    edge = str(edge).lower()
    if edge not in _EDGES:
        raise ValueError("edge must be 'top', 'bottom', 'left', or 'right'")
    return edge


@dataclass(frozen=True)
class EdgeRegion:
    """Configuration for one Window-owned edge region.

    Sizes use logical pixels, so HiDPI framebuffer scaling does not change the
    perceived width or height.
    """

    size: float = 240.0
    minimum_size: float = 120.0
    maximum_fraction: float = 0.5
    resizable: bool = True
    collapsible: bool = True
    collapsed: bool = False


class WindowLayout:
    """Facade for a Window's top/bottom/left/right root layout."""

    __slots__ = ("_window",)

    def __init__(self, window):
        self._window = window

    def configure_region(self, edge, config):
        edge = _normalize_edge(edge)
        if config is None:
            self._window.window.disable_edge_region(edge)
            return self
        if isinstance(config, dict):
            config = EdgeRegion(**config)
        if not isinstance(config, EdgeRegion):
            raise TypeError("edge region must be EdgeRegion, dict, or None")
        self._window.window.configure_edge_region(
            edge,
            True,
            config.size,
            config.minimum_size,
            config.maximum_fraction,
            config.resizable,
            config.collapsible,
            config.collapsed,
        )
        return self

    def disable(self, edge):
        self._window.window.disable_edge_region(_normalize_edge(edge))
        return self

    def set_collapsed(self, edge, collapsed=True):
        self._window.window.set_edge_region_collapsed(
            _normalize_edge(edge), bool(collapsed)
        )
        return self

    def toggle(self, edge):
        self._window.window.toggle_edge_region(_normalize_edge(edge))
        return self

    def set_minimum_render_size(self, width, height):
        self._window.window.set_minimum_render_size(width, height)
        return self

    @property
    def state(self):
        """Return the current logical and framebuffer layout snapshot."""

        return self._window.window.get_window_layout()

    @contextmanager
    def region(self, edge, name=None):
        """Open an edge-owned ImGui region for the current frame.

        The yielded object is the Window's :class:`Gui` while expanded and
        ``None`` while disabled or collapsed.
        """

        edge = _normalize_edge(edge)
        if name is None:
            name = edge.capitalize()
        gui = self._window.get_gui()
        opened = gui.gui.begin_edge_region(name, edge)
        try:
            yield gui if opened else None
        finally:
            if opened:
                gui.gui.end_edge_region(edge)


__all__ = ["EdgeRegion", "WindowLayout"]
