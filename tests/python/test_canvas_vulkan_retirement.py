"""Exact selected-image teardown, with real tracked default-queue work."""

import threading

import pytest
import taichi_forge as ti
from taichi_forge._lib import core
from tests import test_utils
from tests.python.test_canvas_vulkan_submission import _close, _ready, _setup
from tests.python.vulkan_queue_gate import VulkanQueueGate


def _ordinary_call(graph):
    # Warm before blocking: compilation/resource initialization may synchronize.
    marker = ti.ndarray(ti.u32, (1, 1))
    arguments = dict(x=marker, value=19)
    graph.submit(arguments).wait()
    return arguments


@pytest.mark.skipif(not core.GGUI_AVAILABLE, reason="GGUI not available")
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_display_close_excludes_unrelated_tracked_queue_tails():
    resources = _setup()
    output, producer, window, canvas, frame, graph, writer, tick = resources
    gates, tickets = [], []
    try:
        arguments = _ordinary_call(graph)
        source = writer.submit(values={"value": 0xFF332211})
        tickets.append(source)
        display = canvas.submit_frame(frame, source_completion=source,
                                      track_source=True, track_completion=True)
        assert display is not None and window.show()
        _ready(display)
        repeat = canvas.repeat_frame()
        assert repeat is not None and window.show()
        _ready(repeat)

        default = VulkanQueueGate(output, queue_index=0)
        gates.append(default)
        # Unlike a raw queue gate, this tail is visible to device.wait_idle().
        ordinary = graph.submit(arguments)
        late = VulkanQueueGate(output)
        gates.append(late)
        tail = tick.submit()
        tickets.append(tail)
        assert not ordinary.done() and not tail.done()
        window.destroy()
        assert not default.released and not late.released
        assert not ordinary.done() and not tail.done()
        assert frame.source_completion.done()
        default.release()
        late.release()
        ordinary.wait()
        _ready(tail)
        # Closing the Canvas retires its own instance, not the selected domain.
        after = tick.submit()
        tickets.append(after)
        _ready(after)
        assert all(not gate.timed_out for gate in gates)
    finally:
        _close(resources, gates, tickets)


@pytest.mark.skipif(not core.GGUI_AVAILABLE, reason="GGUI not available")
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_display_close_waits_cancelled_copy_but_not_later_work():
    resources = _setup()
    output, producer, window, canvas, frame, graph, writer, tick = resources
    gates, tickets = [], []
    release = None
    try:
        arguments = _ordinary_call(graph)
        source_gate = VulkanQueueGate(producer)
        gates.append(source_gate)
        source = writer.submit(values={"value": 0xFF332211})
        tickets.append(source)
        cancelled = canvas.submit_frame(frame, source_completion=source,
                                        track_source=True, track_completion=True)
        assert cancelled is not None and not frame.source_completion.done()
        canvas.canvas._cancel_display_frame()
        assert cancelled.status == "cancelled"
        late = VulkanQueueGate(output)
        gates.append(late)
        tail = tick.submit()
        tickets.append(tail)
        default = VulkanQueueGate(output, queue_index=0)
        gates.append(default)
        ordinary = graph.submit(arguments)
        release = threading.Timer(0.05, source_gate.release)
        release.start()
        window.destroy()
        assert source_gate.released and frame.source_completion.done()
        assert not late.released and not default.released
        assert not tail.done() and not ordinary.done()
        late.release()
        default.release()
        _ready(tail)
        ordinary.wait()
        assert all(not gate.timed_out for gate in gates)
    finally:
        if release is not None:
            release.join()
        _close(resources, gates, tickets)


@pytest.mark.skipif(not core.GGUI_AVAILABLE, reason="GGUI not available")
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_display_with_general_history_keeps_conservative_retirement():
    resources = _setup(general_history=True)
    output, _, window, canvas, frame, graph, writer, _ = resources
    gates, tickets = [], []
    release = None
    try:
        arguments = _ordinary_call(graph)
        source = writer.submit(values={"value": 0xFF332211})
        tickets.append(source)
        display = canvas.submit_frame(frame, source_completion=source,
                                      track_source=True, track_completion=True)
        assert display is not None and window.show()
        _ready(display)
        gate = VulkanQueueGate(output, queue_index=0)
        gates.append(gate)
        ordinary = graph.submit(arguments)
        assert not ordinary.done()
        release = threading.Timer(0.05, gate.release)
        release.start()
        window.destroy()
        assert gate.released and ordinary.done()
        assert not gate.timed_out
    finally:
        if release is not None:
            release.join()
        _close(resources, gates, tickets)
