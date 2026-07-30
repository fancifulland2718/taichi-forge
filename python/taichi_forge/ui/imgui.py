from contextlib import contextmanager


class Gui:
    """For declaring IMGUI components in a :class:`taichi_forge.ui.Window`
    created by the GGUI system.

    Args:
        gui: reference to a `PyGui`.
    """

    def __init__(self, gui) -> None:
        self.gui = gui

    def set_font_scale(self, scale):
        """Use a fixed global font scale for subsequent GGUI frames.

        This disables automatic scaling by window height. The new scale takes
        effect at the next frame boundary.

        Args:
            scale (float): A finite scale greater than zero.
        """
        self.gui.set_font_scale(scale)

    def set_font_scale_from_window_height(
        self, reference_height, reference_scale=1.0
    ):
        """Continuously scale fonts from the window's logical height.

        For each subsequent GGUI frame, the effective scale is
        ``reference_scale * logical_height / reference_height``. Logical
        height excludes the framebuffer pixel multiplier on HiDPI displays.
        A minimized zero-height window keeps its last valid scale.

        Args:
            reference_height (float): Logical window height at which
                ``reference_scale`` is used. Must be finite and greater than
                zero.
            reference_scale (float): Font scale at ``reference_height``.
                Must be finite and greater than zero.
        """
        self.gui.set_font_scale_from_window_height(
            reference_height, reference_scale
        )

    def get_font_scale(self):
        """Return the effective font scale prepared for the current frame."""
        return self.gui.get_font_scale()

    def set_font_size(self, size):
        """Use a fixed logical-pixel font size for subsequent GGUI frames.

        The size is converted to a scale from the actual default font, so this
        remains meaningful if the font atlas uses a different base size.

        Args:
            size (float): A finite logical-pixel size greater than zero.
        """
        self.gui.set_font_size(size)

    def set_font_size_from_window_height(
        self,
        reference_height,
        reference_size=16.0,
        minimum_size=12.0,
        maximum_size=24.0,
    ):
        """Continuously choose a readable logical-pixel font size.

        The unclamped size is ``reference_size * logical_height /
        reference_height``. The default 12-to-24 logical-pixel bounds avoid
        unreadably small text and excessive control panels.

        Args:
            reference_height (float): Logical window height at which
                ``reference_size`` is used.
            reference_size (float): Font size at ``reference_height``.
            minimum_size (float): Smallest effective logical font size.
            maximum_size (float): Largest effective logical font size.
        """
        self.gui.set_font_size_from_window_height(
            reference_height,
            reference_size,
            minimum_size,
            maximum_size,
        )

    def get_font_size(self):
        """Return the effective logical-pixel font size."""
        return self.gui.get_font_size()

    @contextmanager
    def sub_window(self, name, x, y, width, height=None):
        """Creating a context manager for subwindow.

        Note:
            All args of this method should align with `begin`.

        Args:
            x (float): The x-coordinate (between 0 and 1) of the top-left \
                corner of the subwindow, relative to the full window.
            y (float): The y-coordinate (between 0 and 1) of the top-left \
                corner of the subwindow, relative to the full window.
            width (float): The width of the subwindow relative to the full window.
            height (float | None): The height relative to the full window.
                If ``None``, the height follows the currently visible content.

        Example::

            >>> with gui.sub_window(name, x, y, width, height=None) as g:
            >>>     g.text("Hello, World!")
        """
        self.begin(name, x, y, width, height)
        try:
            yield self
        finally:
            self.end()

    def begin(self, name, x, y, width, height=None):
        """Creates a subwindow that holds imgui widgets.

        All widget function calls (e.g. `text`, `button`) after the `begin`
        and before the next `end` will describe the widgets within this subwindow.

        Args:
            x (float): The x-coordinate (between 0 and 1) of the top-left \
                corner of the subwindow, relative to the full window.
            y (float): The y-coordinate (between 0 and 1) of the top-left \
                corner of the subwindow, relative to the full window.
            width (float): The width of the subwindow relative to the full window.
            height (float | None): The height relative to the full window.
                If ``None``, the height follows the visible content.
        """
        if height is None:
            self.gui.begin_auto(name, x, y, width)
        else:
            self.gui.begin(name, x, y, width, height)

    @contextmanager
    def collapsible_section(self, name, default_open=True):
        """Create an independently collapsible section in a subwindow.

        The yielded value is this GUI object while the section is expanded,
        otherwise it is ``None``. Widget declarations should therefore be
        guarded with ``if section``. Open sections indent their contents and
        isolate widget IDs from sibling sections.

        Example::

            >>> with gui.sub_window("Controls", 0.02, 0.02, 0.3) as panel:
            >>>     with panel.collapsible_section("Solver") as section:
            >>>         if section:
            >>>             section.text("PCG")

        Args:
            name (str): Visible section label and persistent ImGui identity.
            default_open (bool): Initial state before the user toggles it.
        """
        expanded = self.gui.begin_collapsible_section(name, default_open)
        try:
            yield self if expanded else None
        finally:
            if expanded:
                self.gui.end_collapsible_section()

    def end(self):
        """End the description of the current subwindow."""
        self.gui.end()

    def text(self, text, color=None):
        """Declares a line of text."""
        if color is None:
            self.gui.text(text)
        else:
            self.gui.text_colored(text, color)

    def checkbox(self, text, old_value):
        """Declares a checkbox, and returns whether or not it has been checked.

        Args:
            text (str): a line of text to be shown next to the checkbox.
            old_value (bool): whether the checkbox is currently checked.
        """
        return self.gui.checkbox(text, old_value)

    def slider_int(self, text, old_value, minimum, maximum):
        """Declares a slider, and returns its newest value.

        Args:
            text (str): a line of text to be shown next to the slider
            old_value (int) : the current value of the slider.
            minimum (int): the minimum value of the slider.
            maximum (int): the maximum value of the slider.

        Returns:
            int: the updated value of the slider.
        """
        return self.gui.slider_int(text, old_value, minimum, maximum)

    def slider_float(self, text, old_value, minimum, maximum):
        """Declares a slider, and returns its newest value.

        Args:
            text (str): a line of text to be shown next to the slider
            old_value (float): the current value of the slider.
            minimum (float): the minimum value of the slider.
            maximum (float): the maximum value of the slider.
        """
        return self.gui.slider_float(text, old_value, minimum, maximum)

    def color_edit_3(self, text, old_value):
        """Declares a color edit palate.

        Args:
            text (str): a line of text to be shown next to the palate.
            old_value (Tuple[float]): the current value of the color, this \
                should be a tuple of floats in [0,1] that indicates RGB values.
        """
        return self.gui.color_edit_3(text, old_value)

    def button(self, text):
        """Declares a button, and returns whether or not it had just been clicked.

        Args:
            text (str): a line of text to be shown next to the button.
        """
        return self.gui.button(text)
