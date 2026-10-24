"""Real retained images, first-frame widgets and exact pending GUI readers."""
import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge._lib import core
from tests import test_utils
from tests.python.test_canvas_vulkan_execution_domain import _domains
from tests.python.test_canvas_vulkan_submission import _ready
from tests.python.vulkan_queue_gate import VulkanQueueGate


class GuiFixture:
    def __init__(self):
        self.domain, = _domains(1, capacity=4, instance_capacity=1)
        self.window = ti.ui.Window("GUI reader ledger", (320, 240), show_window=False)
        self.canvas = self.window.get_canvas()
        self.canvas.set_execution_domain(self.domain)
        self.pixels = ti.ndarray(ti.u32, (320, 240))
        self.pixels.fill(0xFF332211)
        ti.sync()  # Cold initialization only, before every controlled gate.
        self.frame = ti.ui.DisplayFrame.from_packed_u32_ndarray(self.pixels)
        self.canvas.prepare_frame_sources([self.frame], capacity=3)
        self.producer = self.domain.latest_completion()
        self.gui = self.window.get_gui()  # Must prepare fonts before gate.
        self.gates = []

    def widgets(self, color=(1, 0, 0), rows=1):
        with self.gui.sub_window("fixed", 0.04, 0.05, 0.45, 0.8) as panel:
            for _ in range(rows):
                panel.text("MMMMMMMMMMMM", color=color)

    def initial(self):
        return self.canvas.submit_frame(self.frame, source_completion=self.producer,
                                       track_source=True, track_completion=True)

    def close(self):
        for gate in self.gates:
            gate.release()
        for gate in self.gates:
            gate.close()
        self.window.destroy()
        self.producer.close()
        self.domain.close()


@pytest.mark.skipif(not core.GGUI_AVAILABLE, reason="GGUI not available")
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_selected_gui_first_frame_and_repeat_pixels():
    f = GuiFixture()
    try:
        f.widgets()
        assert f.initial() is not None
        first = f.window.get_image_buffer_as_numpy()
        background = np.array([17, 34, 51, 255]) / 255
        np.testing.assert_allclose(first[240:300, 40:180],
                                   np.broadcast_to(background, (60, 140, 4)), atol=1/255)
        red = (first[..., 0] > .65) & (first[..., 1] < .25) & (first[..., 2] < .25)
        assert np.count_nonzero(red) > 20, "first requested frame lost its GUI text"
        sequence = f.domain.latest_completion().info["sequence"]
        # Calling get_gui twice inside a populated frame must be idempotent.
        f.widgets((0, 1, 0))
        f.window.get_gui()
        assert f.canvas.repeat_frame() is not None
        second = f.window.get_image_buffer_as_numpy()
        green = (second[..., 1] > .65) & (second[..., 0] < .25) & (second[..., 2] < .25)
        assert np.count_nonzero(green) > 20
        assert not np.array_equal(first, second)
        assert f.domain.latest_completion().info["sequence"] == sequence
    finally:
        f.close()


@pytest.mark.skipif(not core.GGUI_AVAILABLE, reason="GGUI not available")
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_selected_gui_two_pending_readers_busy_before_backend_write():
    f = GuiFixture()
    try:
        f.widgets()
        assert f.initial() is not None
        f.window.get_image_buffer_as_numpy()  # Warm pipeline/buffers before gate.
        sequence = f.domain.latest_completion().info["sequence"]
        gate = VulkanQueueGate(f.domain, graphics=True)
        f.gates.append(gate)
        accepted = []
        for color in [(1, 0, 0), (0, 1, 0)]:
            f.widgets(color)
            ticket = f.canvas.repeat_frame()
            assert ticket is not None
            assert f.window.show()
            accepted.append(ticket)
        assert all(not t.done() for t in accepted)
        # The third record would grow/map a backend slot in the old implicit
        # ring. No synchronous readback may run before observing this refusal.
        f.widgets((0, 0, 1), rows=500)
        refused = f.canvas.repeat_frame()
        assert refused is not None
        assert not f.window.show()
        assert refused.status == "cancelled"
        assert f.domain.latest_completion().info["sequence"] == sequence
        assert not gate.timed_out
        gate.release()
        for ticket in accepted:
            _ready(ticket)
        f.widgets((0, 0, 1))
        assert f.canvas.repeat_frame() is not None
        image = f.window.get_image_buffer_as_numpy()
        blue = (image[..., 2] > .65) & (image[..., 0] < .25) & (image[..., 1] < .25)
        assert np.count_nonzero(blue) > 20
        assert not gate.timed_out
    finally:
        f.close()
