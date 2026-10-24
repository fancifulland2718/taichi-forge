"""Same-Program retained Vulkan packed copies; no writable owner is implied.

These basic tests are deliberately separate from controlled timeline-gate
qualification of pending cancellation, exact late tails and queue saturation.
"""

import gc

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge._lib import core
from tests import test_utils


def _domains(count, **options):
    domains = []
    try:
        for _ in range(count):
            domains.append(ti.graph.execution_domain(**options))
    except RuntimeError as error:
        for domain in domains:
            domain.close()
        if ("No independent Vulkan compute queue is available" in str(error)
                or "timeline support" in str(error)):
            pytest.skip(str(error))
        raise
    return domains


def _expect_rgba(image, rgba):
    np.testing.assert_allclose(
        image, np.broadcast_to(np.array(rgba) / 255, image.shape), atol=1 / 255
    )


@pytest.mark.skipif(not core.GGUI_AVAILABLE, reason="GGUI not available")
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_canvas_accepts_peer_producer_and_repeats_exact_copy():
    output, peer = _domains(2, capacity=2, instance_capacity=2)
    window = ti.ui.Window("Vulkan packed peer", (12, 20), show_window=False)
    canvas = window.get_canvas()
    canvas.set_execution_domain(output)
    pixels = ti.ndarray(ti.u32, (12, 20))
    frame = ti.ui.DisplayFrame.from_packed_u32_ndarray(pixels)

    @ti.kernel
    def paint(x: ti.types.ndarray(ti.u32, 2)):
        for i, j in x:
            x[i, j] = ti.u32(i * 3) | (ti.u32(j * 5) << 8) | ti.u32(0xFF330000)

    expected = np.empty((12, 20, 4), dtype=np.float32)
    expected[..., 0] = np.arange(12)[:, None] * 3 / 255
    expected[..., 1] = np.arange(20)[None, :] * 5 / 255
    expected[..., 2] = 51 / 255
    expected[..., 3] = 1

    builder = ti.graph.GraphBuilder()
    builder.dispatch(paint, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "x", ti.u32, ndim=2))
    graph = builder.compile()
    writer = graph.prepare(dict(x=pixels), execution_domain=peer)
    writer.initialize()
    try:
        # Caller-chosen capacity, greater than the hidden renderer's flight count.
        assert canvas.prepare_frame_sources([frame], capacity=7) is None
        for _ in range(4):
            producer = writer.submit()
            assert producer is not None
            display = canvas.submit_frame(
                frame, source_completion=producer, track_source=True, track_completion=True
            )
            assert display is not None
            producer.close()  # Captured exact dependency survives host handle release.
            frame.source_completion.wait()
            copy_sequence = frame.source_completion.sequence
            np.testing.assert_allclose(window.get_image_buffer_as_numpy(), expected, atol=1 / 255)
            assert display.done()
            repeated = canvas.repeat_frame()
            assert repeated is not None
            np.testing.assert_allclose(window.get_image_buffer_as_numpy(), expected, atol=1 / 255)
            assert repeated.done()
            assert output.latest_completion().info["sequence"] == copy_sequence
        assert window.get_display_stats()["zero_copy_render_submissions"] == 0
        window.destroy()
        # Closing the pool retires its owned instance, not unrelated domain work.
        assert not output.info["closed"]
        assert writer.submit() is not None
    finally:
        window.destroy()
        writer.close()
        output.close()
        peer.close()
        graph.close()


@pytest.mark.skipif(not core.GGUI_AVAILABLE, reason="GGUI not available")
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_canvas_cancel_retains_source_and_copy_ticket():
    (output,) = _domains(1, capacity=2, instance_capacity=1)
    window = ti.ui.Window("Vulkan packed cancellation", (9, 13), show_window=False)
    canvas = window.get_canvas()
    canvas.set_execution_domain(output)
    pixels = ti.ndarray(ti.u32, (9, 13))
    pixels.fill(0xFF665544)
    ti.sync()  # Cold initialization, not an independence assertion.
    frame = ti.ui.DisplayFrame.from_packed_u32_ndarray(pixels)
    canvas.prepare_frame_sources([frame], capacity=2)
    producer = output.latest_completion()  # Same-owner completed cold prefix.
    try:
        display = canvas.submit_frame(
            frame, source_completion=producer, track_source=True, track_completion=True
        )
        copy = frame.source_completion
        canvas.canvas._cancel_display_frame()
        assert display.status == "cancelled"
        with pytest.raises(RuntimeError, match="cancelled"):
            display.done()
        # No Python ndarray reference remains; native source generation is leased.
        frame._packed_u32 = None
        del pixels
        gc.collect()
        copy.wait()
        assert copy.done()
        again = canvas.submit_frame(
            frame, source_completion=producer, track_source=True, track_completion=True
        )
        assert again is not None
        frame.source_completion.wait()
        _expect_rgba(window.get_image_buffer_as_numpy(), [68, 85, 102, 255])
        assert again.done()
    finally:
        producer.close()
        window.destroy()
        output.close()


@pytest.mark.skipif(not core.GGUI_AVAILABLE, reason="GGUI not available")
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_canvas_source_validation_is_cold_and_recoverable():
    (output,) = _domains(1, capacity=2, instance_capacity=1)
    window = ti.ui.Window("Vulkan packed validation", (8, 11), show_window=False)
    canvas = window.get_canvas()
    canvas.set_execution_domain(output)
    pixels = ti.ndarray(ti.u32, (8, 11))
    other = ti.ndarray(ti.u32, (7, 11))
    pixels.fill(0xFF332211)
    ti.sync()
    frame = ti.ui.DisplayFrame.from_packed_u32_ndarray(pixels)
    wrong_shape = ti.ui.DisplayFrame.from_packed_u32_ndarray(other)
    producer = output.latest_completion()
    try:
        before = producer.info["sequence"]
        with pytest.raises(RuntimeError, match="same shape"):
            canvas.prepare_frame_sources([frame, wrong_shape])
        assert output.latest_completion().info["sequence"] == before
        # A failed cold preparation did not consume the sole instance budget.
        canvas.prepare_frame_sources([frame], capacity=2)
        with pytest.raises(ValueError, match="source_completion"):
            canvas.submit_frame(frame)
        with pytest.raises(RuntimeError, match="prepared binding|not cold-prepared"):
            canvas.submit_frame(wrong_shape, source_completion=producer)
        with pytest.raises(ValueError, match="prepare_frame_pool"):
            canvas.acquire_frame(8, 11)
        with pytest.raises(RuntimeError, match="previous display domain"):
            canvas.prepare_frame_pool(8, 11)
        # Reject before Python geometry helpers can create/stage any buffers.
        with pytest.raises(ValueError, match="image-only"):
            canvas.circles(None, radius=0.1)
        assert output.latest_completion().info["sequence"] == before
        assert canvas.submit_frame(frame, source_completion=producer, track_source=True)
        frame.source_completion.wait()
        _expect_rgba(window.get_image_buffer_as_numpy(), [17, 34, 51, 255])
        with pytest.raises(RuntimeError, match="previous display domain"):
            canvas.prepare_frame_sources([wrong_shape], capacity=2)
        # Rejected reconfiguration leaves the previous pool and last image intact.
        assert canvas.repeat_frame() is not None
        _expect_rgba(window.get_image_buffer_as_numpy(), [17, 34, 51, 255])
    finally:
        producer.close()
        window.destroy()
        output.close()


@pytest.mark.skipif(not core.GGUI_AVAILABLE, reason="GGUI not available")
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_canvas_reprepare_and_reset_retire_copy_owner():
    (output,) = _domains(1, capacity=2, instance_capacity=1)
    window = ti.ui.Window("Vulkan packed resize", (8, 8), show_window=False)
    canvas = window.get_canvas()
    pixels = ti.ndarray(ti.u32, (8, 8))
    pixels.fill(0xFF332211)
    ti.sync()
    frame = ti.ui.DisplayFrame.from_packed_u32_ndarray(pixels)
    canvas.set_execution_domain(output)
    canvas.prepare_frame_sources([frame], capacity=2)
    initial = output.latest_completion()
    replacement = None
    try:
        assert canvas.submit_frame(frame, source_completion=initial, track_source=True)
        frame.source_completion.wait()
        window.get_image_buffer_as_numpy()
        output.close()
        (replacement,) = _domains(1, capacity=2, instance_capacity=1)
        canvas.set_execution_domain(replacement)
        larger = ti.ndarray(ti.u32, (16, 10))
        larger.fill(0xFF665544)
        ti.sync()
        resized = ti.ui.DisplayFrame.from_packed_u32_ndarray(larger)
        canvas.prepare_frame_sources([resized], capacity=3)
        assert canvas.submit_frame(
            resized, source_completion=replacement.latest_completion(), track_source=True
        )
        resized.source_completion.wait()
        _expect_rgba(window.get_image_buffer_as_numpy(), [68, 85, 102, 255])
        ti.reset()
        with pytest.raises(RuntimeError, match="invalidated"):
            canvas.prepare_frame_sources([resized], capacity=2)
    finally:
        initial.close()
        window.destroy()
        output.close()
        if replacement is not None:
            replacement.close()
