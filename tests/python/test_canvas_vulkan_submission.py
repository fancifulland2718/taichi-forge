"""Controlled queue gates for selected-domain display, using tiny real kernels."""

import time

import pytest
import taichi_forge as ti
from taichi_forge._lib import core
from tests import test_utils
from tests.python.test_canvas_vulkan_execution_domain import _domains, _expect_rgba
from tests.python.vulkan_queue_gate import VulkanQueueGate


def _ready(ticket):
    deadline = time.monotonic() + 2
    while not ticket.done() and time.monotonic() < deadline:
        time.sleep(0.001)
    assert ticket.done(), "exact prefix did not complete while unrelated gate was held"


def _setup():
    output, producer = _domains(2, capacity=3, instance_capacity=2)
    window = ti.ui.Window("Vulkan gated display", (6, 4), show_window=False)
    canvas = window.get_canvas()
    canvas.set_execution_domain(output)
    pixels, marker = ti.ndarray(ti.u32, (6, 4)), ti.ndarray(ti.u32, (1, 1))

    @ti.kernel
    def paint(x: ti.types.ndarray(ti.u32, 2), value: ti.u64):
        for i, j in x:
            x[i, j] = ti.u32(value)

    builder = ti.graph.GraphBuilder()
    builder.dispatch(paint,
                     ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "x", ti.u32, ndim=2),
                     ti.graph.Arg(ti.graph.ArgKind.SCALAR, "value", ti.u64))
    graph = builder.compile()
    writer = graph.prepare(dict(x=pixels, value=0xFF332211), execution_domain=producer,
                           mutable_parameters=("value",))
    tick = graph.prepare(dict(x=marker, value=7), execution_domain=output)
    writer.initialize()
    tick.initialize()
    frame = ti.ui.DisplayFrame.from_packed_u32_ndarray(pixels)
    canvas.prepare_frame_sources([frame], capacity=2)
    return output, producer, window, canvas, frame, graph, writer, tick


def _close(resources, gates, tickets):
    # Release all prefixes first: a gate fence may be behind another gate.
    for gate in gates:
        gate.release()
    for gate in gates:
        gate.close()
    output, producer, window, _, _, graph, writer, tick = resources
    window.destroy()
    writer.close()
    tick.close()
    output.close()
    producer.close()
    graph.close()
    for ticket in tickets:
        ticket.close()


@pytest.mark.skipif(not core.GGUI_AVAILABLE, reason="GGUI not available")
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_cancelled_copy_stays_busy_until_its_exact_producer():
    resources = _setup()
    output, producer, window, canvas, frame, _, writer, tick = resources
    gates, tickets = [], []
    try:
        first = writer.submit(values={"value": 0xFF332211})
        tickets.append(first)
        display = canvas.submit_frame(frame, source_completion=first,
                                      track_source=True, track_completion=True)
        assert display is not None
        frame.source_completion.wait()
        _expect_rgba(window.get_image_buffer_as_numpy(), [17, 34, 51, 255])
        # Slot one remains pinned as the last displayed frame.
        blocked = VulkanQueueGate(producer)
        gates.append(blocked)
        second = writer.submit(values={"value": 0xFF665544})
        tickets.append(second)
        late = VulkanQueueGate(producer)
        gates.append(late)
        tail = writer.submit(values={"value": 0xFF998877})
        tickets.append(tail)
        assert not second.done()
        independent = tick.submit()
        tickets.append(independent)
        _ready(independent)  # Output progresses while producer is blocked.
        cancelled = canvas.submit_frame(frame, source_completion=second,
                                        track_source=True, track_completion=True)
        assert cancelled is not None
        copy = frame.source_completion
        assert not copy.done()
        canvas.canvas._cancel_display_frame()
        assert cancelled.status == "cancelled"
        before = output.latest_completion().info["sequence"]
        assert canvas.submit_frame(frame, source_completion=second,
                                   track_source=True, track_completion=True) is None
        assert output.latest_completion().info["sequence"] == before
        assert frame.source_completion is copy
        blocked.release()
        _ready(copy)
        assert not tail.done() and not late.timed_out
        accepted = canvas.submit_frame(frame, source_completion=second,
                                       track_source=True, track_completion=True)
        assert accepted is not None
        _ready(frame.source_completion)
        assert window.show()
        _ready(accepted)  # Graphics does not wait the producer's later prefix.
        assert not tail.done()
        late.release()
        _ready(tail)
        assert canvas.repeat_frame() is not None
        _expect_rgba(window.get_image_buffer_as_numpy(), [68, 85, 102, 255])
        assert all(not gate.timed_out for gate in gates)
    finally:
        _close(resources, gates, tickets)


@pytest.mark.skipif(not core.GGUI_AVAILABLE, reason="GGUI not available")
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_copy_avoids_default_queue_and_graphics_ignores_later_output():
    resources = _setup()
    output, producer, window, canvas, frame, _, writer, tick = resources
    gates, tickets = [], []
    try:
        default = VulkanQueueGate(output, queue_index=0)
        gates.append(default)
        source = writer.submit(values={"value": 0xFF332211})
        tickets.append(source)
        display = canvas.submit_frame(frame, source_completion=source,
                                      track_source=True, track_completion=True)
        assert display is not None
        _ready(frame.source_completion)  # No default queue upload/copy/flush.
        assert not default.released
        late = VulkanQueueGate(output)
        gates.append(late)
        tail = tick.submit()
        tickets.append(tail)
        assert not tail.done()
        assert window.show()  # Host submission must not wait default or tail.
        assert not default.timed_out and not late.timed_out
        shares_graphics = default.queue.value == default.info.graphics_queue
        print(f"DEFAULT_QUEUE_SHARES_GRAPHICS={shares_graphics}")
        if shares_graphics:
            assert not display.done()  # Real physical queue ordering remains.
        default.release()
        _ready(display)
        assert not tail.done()  # Graphics waited copy, not latest output work.
        assert canvas.repeat_frame() is not None
        assert window.show()
        late.release()
        _ready(tail)
        assert canvas.repeat_frame() is not None
        _expect_rgba(window.get_image_buffer_as_numpy(), [17, 34, 51, 255])
        assert all(not gate.timed_out for gate in gates)
    finally:
        _close(resources, gates, tickets)
