"""Prepared local writers and actual graphics readers share one finite pool."""

import gc

import pytest
import taichi_forge as ti
from taichi_forge._lib import core
from taichi_forge.lang import impl
from tests import test_utils
from tests.python.test_canvas_vulkan_execution_domain import _domains, _expect_rgba
from tests.python.test_canvas_vulkan_submission import _ready
from tests.python.vulkan_queue_gate import VulkanQueueGate


class WritableFixture:
    def __init__(self, tmp_path, kind="graph", *, slots=2, capacity=4):
        from taichi_forge_runtime.aot import ExecutionPlan

        @ti.kernel
        def paint(left: ti.types.ndarray(ti.u32, 2), right: ti.types.ndarray(ti.u32, 2)):
            for i, j in left:
                left[i, j] = ti.u32(0xFF332211)
            for i, j in right:
                right[i, j] = ti.u32(0xFF332211)

        self.domain, = _domains(1, capacity=capacity, instance_capacity=slots + 1)
        self.window = ti.ui.Window("Vulkan writable", (6, 4), show_window=False)
        self.canvas = self.window.get_canvas()
        self.canvas.set_execution_domain(self.domain)
        self.views = self.canvas.prepare_frame_pool(6, 4, capacity=slots)
        self.loans, self.calls, self.tickets, self.gates = [], {}, [], []
        self.module = None
        builder = ti.graph.GraphBuilder()
        left = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "left", ti.u32, ndim=2)
        right = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "right", ti.u32, ndim=2)
        builder.dispatch(paint, left, right)
        builder.dispatch(paint, left, right)
        self.graph = builder.compile()
        if kind == "aot":
            artifact = ti.aot.Module()
            artifact.add_kernel(paint)
            path = tmp_path / "writable.tcm"
            artifact.archive(str(path))
            runtime = self.domain._runtime
            self.module = runtime.load_module(path)
            plan = ExecutionPlan([
                dict(kind="kernel", entry="paint", arguments=["left", "right"]),
                dict(kind="kernel", entry="paint", arguments=["left", "right"]),
            ])
        for view in self.views:
            if kind == "graph":
                call = self.graph.prepare(dict(left=view, right=view), execution_domain=self.domain)
            else:
                # Distinct Program loans of one owner must deduplicate once for
                # the complete recipe, including both native dispatches.
                loans = [runtime.import_dense_view(view), runtime.import_dense_view(view)]
                self.loans.extend(loans)
                call = self.module.prepare_execution(
                    plan, dict(left=loans[0], right=loans[1]), execution_domain=self.domain
                )
            call.initialize()
            self.calls[id(view.descriptor)] = call
        scratch = ti.ndarray(ti.u32, (1, 1))
        self.tick = self.graph.prepare(dict(left=scratch, right=scratch), execution_domain=self.domain)
        self.tick.initialize()

    def writer(self, frame):
        return self.calls[id(frame.pixels.descriptor)]

    def write(self, frame):
        ticket = self.writer(frame).submit()
        assert ticket is not None
        self.tickets.append(ticket)
        return ticket

    def gate(self, **kwargs):
        result = VulkanQueueGate(self.domain, **kwargs)
        self.gates.append(result)
        return result

    def close(self):
        for gate in self.gates:
            gate.release()
        for gate in self.gates:
            gate.close()
        self.window.destroy()
        for call in self.calls.values():
            call.close()
        self.tick.close()
        for loan in self.loans:
            loan.close()
        if self.module is not None:
            self.module.close()
        self.domain.close()
        self.graph.close()
        for ticket in self.tickets:
            ticket.close()


@pytest.mark.parametrize("kind", ["graph", "aot"])
@pytest.mark.skipif(not core.GGUI_AVAILABLE, reason="GGUI not available")
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_writable_prepared_multiview_exact_and_retirement(tmp_path, kind):
    # Exercise caller capacity beyond both renderer flight count and the old
    # opportunistic Python view-cache size, without any hot allocation.
    fixture = WritableFixture(tmp_path, kind, slots=17)
    canvas, window, domain = fixture.canvas, fixture.window, fixture.domain
    try:
        assert len(fixture.views) == 17
        for view in fixture.views:
            assert canvas._shared_view(view._source) is view
        empty = canvas.acquire_frame(6, 4)
        assert canvas.submit_frame(empty, track_source=True, track_completion=True) is not None
        _expect_rgba(window.get_image_buffer_as_numpy(), [0, 0, 0, 0])
        for _ in range(3):
            frame = canvas.acquire_frame(6, 4)
            writer = fixture.writer(frame)
            ticket = fixture.write(frame)
            with pytest.raises(RuntimeError, match="exact producer"):
                canvas.submit_frame(frame)  # Required even without tracking.
            late = fixture.tick.submit()
            assert late is not None
            fixture.tickets.append(late)
            with pytest.raises((RuntimeError, ValueError), match="exact producer"):
                canvas.submit_frame(frame, source_completion=late)
            # Both failed seals leave the host lease writable; a second actual
            # prepared submission is rejected before consuming another epoch.
            assert frame.pixels is not None
            with pytest.raises(RuntimeError, match="one complete prepared writer"):
                writer.submit()
            assert canvas.submit_frame(frame, source_completion=ticket,
                                       track_source=True, track_completion=True) is not None
            assert frame.source_completion.sequence == ticket.info["sequence"]
            ticket.close()
            _expect_rgba(window.get_image_buffer_as_numpy(), [17, 34, 51, 255])
            before = domain.latest_completion().info["sequence"]
            assert canvas.repeat_frame() is not None
            _expect_rgba(window.get_image_buffer_as_numpy(), [17, 34, 51, 255])
            assert domain.latest_completion().info["sequence"] == before
        cancelled = canvas.acquire_frame(6, 4)
        before = domain.latest_completion().info["sequence"]
        cancelled.cancel()
        assert domain.latest_completion().info["sequence"] == before
        with pytest.raises(RuntimeError, match="previous display domain"):
            canvas.prepare_frame_pool(7, 5, capacity=2)
        assert canvas.repeat_frame() is not None
        _expect_rgba(window.get_image_buffer_as_numpy(), [17, 34, 51, 255])
        window.destroy()
        assert not domain.info["closed"]
        alive = fixture.tick.submit()
        fixture.tickets.append(alive)
        alive.wait()
        with pytest.raises(RuntimeError):
            writer.submit()  # Only storage-related instances were revoked.
        # Public source proof and loans outlive physical storage retirement.
        assert frame.source_completion.done()
        ti.reset()
        assert frame.source_completion.done()
        del frame, empty
        gc.collect()
    finally:
        fixture.close()


@pytest.mark.skipif(not core.GGUI_AVAILABLE, reason="GGUI not available")
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_writable_cancel_pending_and_empty_busy_are_non_consuming(tmp_path):
    fixture = WritableFixture(tmp_path, capacity=1)
    canvas, window, domain = fixture.canvas, fixture.window, fixture.domain
    try:
        first = canvas.acquire_frame(6, 4)
        ticket = fixture.write(first)
        assert canvas.submit_frame(first, source_completion=ticket)
        _expect_rgba(window.get_image_buffer_as_numpy(), [17, 34, 51, 255])
        pending = canvas.acquire_frame(6, 4)
        descriptor = pending.pixels.descriptor
        gate = fixture.gate()
        ticket = fixture.write(pending)
        assert not ticket.done()
        pending.cancel()
        before = domain.latest_completion().info["sequence"]
        assert canvas.acquire_frame(6, 4) is None
        assert domain.latest_completion().info["sequence"] == before
        gate.release()
        _ready(ticket)
        empty = canvas.acquire_frame(6, 4)
        assert empty.pixels.descriptor is descriptor
        full = fixture.gate()
        tail = fixture.tick.submit()
        fixture.tickets.append(tail)
        before = domain.latest_completion().info["sequence"]
        assert canvas.submit_frame(empty, track_source=True, track_completion=True) is None
        assert empty.pixels.descriptor is descriptor
        assert domain.latest_completion().info["sequence"] == before
        full.release()
        _ready(tail)
        assert canvas.submit_frame(empty, track_source=True, track_completion=True) is not None
        _expect_rgba(window.get_image_buffer_as_numpy(), [17, 34, 51, 255])
        assert all(not gate.timed_out for gate in fixture.gates)
    finally:
        fixture.close()


@pytest.mark.skipif(not core.GGUI_AVAILABLE, reason="GGUI not available")
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_writable_graphics_readers_and_exact_late_tail(tmp_path):
    fixture = WritableFixture(tmp_path)
    canvas, window, domain = fixture.canvas, fixture.window, fixture.domain
    try:
        default = fixture.gate(queue_index=0)
        frame = canvas.acquire_frame(6, 4)
        ticket = fixture.write(frame)
        _ready(ticket)  # Prepared writer did not submit through the default queue.
        assert canvas.submit_frame(frame, source_completion=ticket, track_source=True)
        tail_gate = fixture.gate()
        tail = fixture.tick.submit()
        fixture.tickets.append(tail)
        assert window.show()
        default.release()  # Graphics may physically share queue zero.
        _ready(frame.completion)
        assert not tail.done() and not tail_gate.timed_out
        before = domain.latest_completion().info["sequence"]
        repeated = canvas.repeat_frame()
        assert repeated is not None
        assert window.show()
        _ready(repeated)
        assert domain.latest_completion().info["sequence"] == before
        tail_gate.release()
        _ready(tail)
        frame.completion.wait()
        graphics = fixture.gate(graphics=True)
        baseline = dict(impl.get_runtime().prog._debug_external_dense_storage_stats())
        frames = []
        for _ in range(2):
            frame = canvas.acquire_frame(6, 4)
            assert frame is not None
            ticket = fixture.write(frame)
            _ready(ticket)
            assert canvas.submit_frame(frame, source_completion=ticket, track_completion=True)
            assert window.show()
            frames.append(frame)
        assert all(not frame.completion.done() for frame in frames)
        before = domain.latest_completion().info["sequence"]
        assert canvas.acquire_frame(6, 4) is None
        assert domain.latest_completion().info["sequence"] == before
        final = dict(impl.get_runtime().prog._debug_external_dense_storage_stats())
        assert final["created_total"] == baseline["created_total"]
        assert final["released_total"] == baseline["released_total"]
        graphics.release()
        for frame in frames:
            _ready(frame.completion)
        assert canvas.repeat_frame() is not None
        _expect_rgba(window.get_image_buffer_as_numpy(), [17, 34, 51, 255])
        assert all(not gate.timed_out for gate in fixture.gates)
    finally:
        fixture.close()


@pytest.mark.skipif(not core.GGUI_AVAILABLE, reason="GGUI not available")
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_writable_resize_reset_retains_proof_shells(tmp_path):
    fixture = WritableFixture(tmp_path)
    canvas, window = fixture.canvas, fixture.window
    replacement = None
    try:
        original = canvas.acquire_frame(6, 4)
        ticket = fixture.write(original)
        assert canvas.submit_frame(original, source_completion=ticket, track_source=True)
        _expect_rgba(window.get_image_buffer_as_numpy(), [17, 34, 51, 255])
        fixture.domain.close()
        replacement, = _domains(1, capacity=2, instance_capacity=1)
        canvas.set_execution_domain(replacement)
        before = replacement.latest_completion().info["sequence"]
        with pytest.raises(RuntimeError, match="signed shader"):
            canvas.prepare_frame_pool(2**31 - 1, 2, capacity=2)
        assert replacement.latest_completion().info["sequence"] == before
        views = canvas.prepare_frame_pool(8, 5, capacity=2)
        assert tuple(fixture.views[0].descriptor.index_shape) == (6, 4)
        resized = canvas.acquire_frame(8, 5)
        assert canvas.submit_frame(resized, track_source=True, track_completion=True)
        _expect_rgba(window.get_image_buffer_as_numpy(), [0, 0, 0, 0])
        pending = canvas.acquire_frame(8, 5)
        ti.reset()  # Window, views, writer shells and exact tickets remain held.
        assert original.source_completion.done() and resized.source_completion.done()
        with pytest.raises(RuntimeError, match="no longer writable"):
            _ = pending.pixels
        assert tuple(views[0].descriptor.index_shape) == (8, 5)
        del pending, resized, original
        gc.collect()
    finally:
        fixture.close()
        if replacement is not None:
            replacement.close()
