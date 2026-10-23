"""Selected display copies, empty writes and source tickets preserve ownership."""

import numpy as np
import os
import time
import pytest
import taichi_forge as ti
from taichi_forge._lib import core
from tests import test_utils


@pytest.mark.skipif(not core.GGUI_AVAILABLE, reason="GGUI not available")
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_pending_canvas_writer_does_not_block_physics(tmp_path):
    from taichi_forge_runtime.aot import ExecutionPlan, Runtime
    from tests.python.aot_external_memory_probe import Producer, ReadyGate

    @ti.kernel
    def paint(x: ti.types.ndarray(ti.u32, 2)):
        for i, j in x:
            x[i, j] = ti.u32(0xFF332211)

    @ti.kernel
    def step(x: ti.types.ndarray(ti.i32, 1)):
        for i in x:
            x[i] = 314159 + i

    exported = ti.aot.Module()
    exported.add_kernel(paint)
    artifact = tmp_path / "pending-display.tcm"
    exported.archive(str(artifact))
    window = ti.ui.Window("pending writer", (12, 20), show_window=False)
    canvas = window.get_canvas()
    output = ti.graph.execution_domain(capacity=2, instance_capacity=8)
    physics = ti.graph.execution_domain(capacity=2, instance_capacity=4)
    canvas.set_execution_domain(output)
    views = canvas.prepare_frame_pool(12, 20, capacity=2)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(paint, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "x", ti.u32, ndim=2))
    graph = builder.compile()
    calls = {
        id(view.descriptor): graph.prepare(dict(x=view), execution_domain=output)
        for view in views
    }
    for call in calls.values():
        call.initialize()
    state = ti.ndarray(ti.i32, 64)
    state.fill(0)
    state.to_numpy()  # Warm host readback before either controlled gate.
    builder = ti.graph.GraphBuilder()
    builder.dispatch(step, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "x", ti.i32, ndim=1))
    physical_call = builder.compile().prepare(dict(x=state), execution_domain=physics)
    physical_call.initialize()
    runtime = output._runtime
    module = runtime.load_module(artifact)
    with Runtime(
        "vulkan", device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", "0"))
    ) as vk:
        producer = Producer(vk)
        try:
            external = runtime.import_cuda_external_memory(
                memory_handle=producer.handles[0],
                ready_for_cuda=producer.handles[1],
                ready_for_external=producer.handles[2],
                allocation_size=producer.allocation_size,
                device_uuid=producer.uuid,
                dtype="f32",
                shape=64,
                byte_offset=4,
            )
            gate_call = module.prepare_execution(
                ExecutionPlan([dict(kind="fill32", destination="external", value=0)]),
                dict(external=external),
                execution_domain=output,
            )
            gate_call.initialize()
            frame = canvas.acquire_frame(12, 20)
            assert frame is not None
            gate = ReadyGate(producer)
            try:
                blocked = gate_call.submit()
                assert blocked is not None and not blocked.done()
                writer = calls[id(frame.pixels.descriptor)].submit()
                assert writer is not None and not writer.done()
                assert canvas.submit_frame(
                    frame, source_completion=writer, track_source=True
                )
                assert not frame.source_completion.done()
                sentinel = physical_call.submit()
                assert sentinel is not None
                deadline = time.monotonic() + 2
                while not sentinel.done() and time.monotonic() < deadline:
                    time.sleep(0.001)
                assert sentinel.done()
                assert not frame.source_completion.done()
                assert not blocked.done() and not gate.timed_out
            finally:
                gate.close()
            producer.submit(wait=True)
            writer.wait()
            assert frame.source_completion.done()
            # Ordinary ndarray host readback is an explicit synchronization
            # boundary. The exact physics ticket above proves progress while
            # the Canvas writer remains pending; validate its unchanged values
            # after releasing the controlled external wait.
            np.testing.assert_array_equal(state.to_numpy(), np.arange(64) + 314159)
            image = window.get_image_buffer_as_numpy()
            np.testing.assert_allclose(
                image,
                np.broadcast_to(np.array([17, 34, 51, 255]) / 255, image.shape),
                atol=1 / 255,
            )
            producer.runtime.wait()
            window.destroy()
            external.close()
            gate_call.close()
        finally:
            window.destroy()
            for call in calls.values():
                call.close()
            physical_call.close()
            output.close()
            physics.close()
            producer.close()


@pytest.mark.skipif(not core.GGUI_AVAILABLE, reason="GGUI not available")
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_selected_display_copy_cancel_and_empty_source():
    window = ti.ui.Window("selected copy", (12, 20), show_window=False)
    canvas = window.get_canvas()
    domain = ti.graph.execution_domain(instance_capacity=16)
    canvas.set_execution_domain(domain)
    pixels = ti.ndarray(ti.u32, (12, 20))
    pixels.fill(0xFF332211)
    ti.sync()  # Cold source initialization.
    try:
        with pytest.raises(ValueError, match="packed DisplayFrame"):
            canvas.set_image(pixels)
        for _ in range(8):
            empty = canvas.acquire_frame(12, 20)
            assert empty is not None
            # The common storage interface must preserve pybind downcasting
            # and the cached ndarray view for the existing CUDA implementation.
            assert isinstance(empty.pixels._source, core._DisplayStorage)
            assert isinstance(empty.pixels._source, core._SharedCudaVulkanImage)
            assert empty.pixels._source.identity > 0
            assert (empty.pixels._source.width, empty.pixels._source.height) == (12, 20)
            assert canvas._shared_view(empty.pixels._source) is empty.pixels
            before = domain.latest_completion().info["sequence"]
            empty.cancel()
            assert domain.latest_completion().info["sequence"] == before
            frame = ti.ui.DisplayFrame.from_packed_u32_ndarray(pixels)
            assert (
                canvas.submit_frame(frame, track_source=True, track_completion=True)
                is not None
            )
            assert frame.source_completion.sequence > before
            frame.source_completion.wait()
            image = window.get_image_buffer_as_numpy()
            np.testing.assert_allclose(
                image,
                np.broadcast_to(np.array([17, 34, 51, 255]) / 255, image.shape),
                atol=1 / 255,
            )
            assert canvas.repeat_frame() is not None
            np.testing.assert_allclose(
                window.get_image_buffer_as_numpy(), image, atol=1 / 255
            )
        empty = canvas.acquire_frame(12, 20)
        assert canvas.submit_frame(empty, track_source=True)
        empty.source_completion.wait()
        window.get_image_buffer_as_numpy()
        window.destroy()
        assert domain.info["closed"]
    finally:
        window.destroy()
        domain.close()


@pytest.mark.skipif(not core.GGUI_AVAILABLE, reason="GGUI not available")
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_selected_display_rejects_other_stream_producer_before_sealing():
    window = ti.ui.Window("selected producer", (8, 8), show_window=False)
    canvas = window.get_canvas()
    selected, peer = ti.graph.execution_domain(), ti.graph.execution_domain()
    canvas.set_execution_domain(selected)

    @ti.kernel
    def paint(x: ti.types.ndarray(ti.u32, 2)):
        for i, j in x:
            x[i, j] = ti.u32(0xFF332211)

    builder = ti.graph.GraphBuilder()
    builder.dispatch(paint, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "x", ti.u32, ndim=2))
    graph = builder.compile()
    scratch = ti.ndarray(ti.u32, (8, 8))
    foreign = graph.prepare(dict(x=scratch), execution_domain=peer)
    foreign.initialize()
    frame = canvas.acquire_frame(8, 8)
    writer = graph.prepare(dict(x=frame.pixels), execution_domain=selected)
    writer.initialize()
    try:
        ticket = writer.submit()
        other = foreign.submit()
        other.wait()  # Origin remains checkable after event recycling.
        with pytest.raises(RuntimeError, match="another CUDA stream"):
            canvas.submit_frame(frame, source_completion=other, track_source=True)
        assert frame.pixels is not None
        assert canvas.submit_frame(frame, source_completion=ticket, track_source=True)
        frame.source_completion.wait()
        window.get_image_buffer_as_numpy()
    finally:
        window.destroy()
        writer.close()
        foreign.close()
        selected.close()
        peer.close()
        graph.close()


@pytest.mark.skipif(not core.GGUI_AVAILABLE, reason="GGUI not available")
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_selected_display_source_excludes_later_external_wait(tmp_path):
    from taichi_forge_runtime.aot import Runtime, ExecutionPlan
    from tests.python.aot_external_memory_probe import Producer, ReadyGate

    @ti.kernel
    def paint(x: ti.types.ndarray(ti.u32, 2)):
        for i, j in x:
            x[i, j] = ti.u32(0xFF332211)

    exported = ti.aot.Module()
    exported.add_kernel(paint)
    artifact = tmp_path / "display-gate.tcm"
    exported.archive(str(artifact))
    window = ti.ui.Window("exact display source", (12, 20), show_window=False)
    canvas = window.get_canvas()
    domain = ti.graph.execution_domain(capacity=1, instance_capacity=16)
    canvas.set_execution_domain(domain)
    runtime = domain._runtime
    module = runtime.load_module(artifact)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(paint, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "x", ti.u32, ndim=2))
    graph = builder.compile()
    calls = {}
    with Runtime(
        "vulkan", device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", "0"))
    ) as vk:
        producer = Producer(vk)
        try:
            external = runtime.import_cuda_external_memory(
                memory_handle=producer.handles[0],
                ready_for_cuda=producer.handles[1],
                ready_for_external=producer.handles[2],
                allocation_size=producer.allocation_size,
                device_uuid=producer.uuid,
                dtype="f32",
                shape=64,
                byte_offset=4,
            )
            gate_call = module.prepare_execution(
                ExecutionPlan([dict(kind="fill32", destination="external", value=0)]),
                dict(external=external),
                execution_domain=domain,
            )
            gate_call.initialize()
            # Materialize all image bindings while no domain has pending waits.
            for _ in range(6):
                frame = canvas.acquire_frame(12, 20)
                key = id(frame.pixels.descriptor)
                if key not in calls:
                    calls[key] = graph.prepare(
                        dict(x=frame.pixels), execution_domain=domain
                    )
                    calls[key].initialize()
                ticket = calls[key].submit()
                assert ticket is not None
                assert canvas.submit_frame(
                    frame, source_completion=ticket, track_source=True
                )
                window.get_image_buffer_as_numpy()
                ticket.wait()
            frame = canvas.acquire_frame(12, 20)
            ticket = calls[id(frame.pixels.descriptor)].submit()
            ticket.wait()  # Establish space for the deliberately pending tail.
            gate = ReadyGate(producer)
            try:
                pending = gate_call.submit()
                assert pending is not None and not pending.done()
                assert canvas.submit_frame(
                    frame, source_completion=ticket, track_source=True
                )
                deadline = time.monotonic() + 2
                while (
                    not frame.source_completion.done() and time.monotonic() < deadline
                ):
                    time.sleep(0.001)
                assert frame.source_completion.done()
                assert frame.source_completion.sequence == ticket.info["sequence"]
                # Actual Vulkan consumption must also exclude the later CUDA wait.
                image = window.get_image_buffer_as_numpy()
                np.testing.assert_allclose(
                    image,
                    np.broadcast_to(np.array([17, 34, 51, 255]) / 255, image.shape),
                    atol=1 / 255,
                )
                assert not pending.done() and not gate.timed_out
                assert (
                    canvas.repeat_frame() is None
                )  # O remains full; no epoch consumed.
                empty = canvas.acquire_frame(12, 20)
                assert empty is not None
                assert not canvas.submit_frame(empty, track_source=True)
                assert empty.pixels is not None  # Busy does not seal the lease.
                empty.cancel()
                assert not pending.done() and not gate.timed_out
            finally:
                gate.close()
            producer.submit(wait=True)
            pending.wait()
            producer.runtime.wait()
            window.destroy()
            external.close()
            gate_call.close()
        finally:
            window.destroy()
            for call in calls.values():
                call.close()
            domain.close()
            module.close()
            graph.close()
            producer.close()


@pytest.mark.skipif(not core.GGUI_AVAILABLE, reason="GGUI not available")
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_prepared_display_pool_returns_busy_while_readers_are_pending(tmp_path):
    from taichi_forge_runtime.aot import Runtime, ExecutionPlan
    from tests.python.aot_external_memory_probe import Producer, ReadyGate

    @ti.kernel
    def paint(x: ti.types.ndarray(ti.u32, 2)):
        for i, j in x:
            x[i, j] = ti.u32(0xFF332211)

    exported = ti.aot.Module()
    exported.add_kernel(paint)
    artifact = tmp_path / "pool-gate.tcm"
    exported.archive(str(artifact))
    window = ti.ui.Window("bounded pending pool", (12, 20), show_window=False)
    canvas = window.get_canvas()
    domain = ti.graph.execution_domain(capacity=4, instance_capacity=8)
    canvas.set_execution_domain(domain)
    views = canvas.prepare_frame_pool(12, 20, capacity=2)
    runtime = domain._runtime
    module = runtime.load_module(artifact)
    loans = {id(view.descriptor): runtime.import_dense_view(view) for view in views}
    calls = {
        id(view.descriptor): module.prepare_execution(
            ExecutionPlan([dict(kind="kernel", entry="paint", arguments=["image"])]),
            dict(image=loans[id(view.descriptor)]),
            execution_domain=domain,
        )
        for view in views
    }
    for call in calls.values():
        call.initialize()
    with Runtime(
        "vulkan", device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", "0"))
    ) as vk:
        producer = Producer(vk)
        try:
            external = runtime.import_cuda_external_memory(
                memory_handle=producer.handles[0],
                ready_for_cuda=producer.handles[1],
                ready_for_external=producer.handles[2],
                allocation_size=producer.allocation_size,
                device_uuid=producer.uuid,
                dtype="f32",
                shape=64,
                byte_offset=4,
            )
            gate_call = module.prepare_execution(
                ExecutionPlan([dict(kind="fill32", destination="external", value=0)]),
                dict(external=external),
                execution_domain=domain,
            )
            gate_call.initialize()
            baseline = dict(
                ti.lang.impl.get_runtime().prog._debug_external_dense_storage_stats()
            )
            gate = ReadyGate(producer)
            frames = []
            try:
                pending = gate_call.submit()
                assert pending is not None and not pending.done()
                for _ in range(2):
                    frame = canvas.acquire_frame(12, 20)
                    assert frame is not None
                    ticket = calls[id(frame.pixels.descriptor)].submit()
                    assert ticket is not None
                    assert canvas.submit_frame(
                        frame, source_completion=ticket, track_source=True
                    )
                    assert window.show()
                    frames.append(frame)
                # Both actual Vulkan readers are queued behind the gated CUDA
                # producer. Busy must not allocate a third physical image.
                assert canvas.acquire_frame(12, 20) is None
                assert not pending.done() and not gate.timed_out
                assert all(not frame.completion.done() for frame in frames)
                final = dict(
                    ti.lang.impl.get_runtime().prog._debug_external_dense_storage_stats()
                )
                assert final["created_total"] == baseline["created_total"]
                assert final["released_total"] == baseline["released_total"]
            finally:
                gate.close()
            producer.submit(wait=True)
            for frame in frames:
                frame.completion.wait()
            assert canvas.repeat_frame() is not None
            image = window.get_image_buffer_as_numpy()
            np.testing.assert_allclose(
                image,
                np.broadcast_to(np.array([17, 34, 51, 255]) / 255, image.shape),
                atol=1 / 255,
            )
            window.destroy()
            gate_call.close()
            external.close()
            for loan in loans.values():
                loan.close()
        finally:
            window.destroy()
            for call in calls.values():
                call.close()
            domain.close()
            module.close()
            producer.close()


@pytest.mark.skipif(not core.GGUI_AVAILABLE, reason="GGUI not available")
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_selected_display_resize_and_reset_are_cold_boundaries():
    window = ti.ui.Window("selected resize", (12, 20), show_window=False)
    canvas = window.get_canvas()
    domain = ti.graph.execution_domain()
    canvas.set_execution_domain(domain)
    old_views = canvas.prepare_frame_pool(12, 20)
    for _ in range(4):
        frame = canvas.acquire_frame(12, 20)
        assert canvas.submit_frame(frame, track_source=True)
        frame.source_completion.wait()
        window.get_image_buffer_as_numpy()
    with pytest.raises(RuntimeError, match="cold resize boundary"):
        canvas.acquire_frame(16, 16)
    with pytest.raises(RuntimeError, match="previous display domain"):
        canvas.prepare_frame_pool(16, 16)
    assert domain.info["accepting"]
    domain.close()
    replacement = ti.graph.execution_domain()
    canvas.set_execution_domain(replacement)
    canvas.prepare_frame_pool(16, 16)
    # Descriptors are immutable values even after their storage is retired.
    assert tuple(old_views[0].descriptor.index_shape) == (12, 20)
    for _ in range(4):
        frame = canvas.acquire_frame(16, 16)
        assert canvas.submit_frame(frame, track_source=True)
        frame.source_completion.wait()
        window.get_image_buffer_as_numpy()
    pending = canvas.acquire_frame(16, 16)
    ti.reset()
    with pytest.raises(RuntimeError, match="invalidated|no longer writable"):
        _ = pending.pixels
    window.destroy()
    replacement.close()


@pytest.mark.skipif(not core.GGUI_AVAILABLE, reason="GGUI not available")
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_selected_display_fault_retains_unsafe_graphics_closure():
    from taichi_forge.lang import impl

    window = ti.ui.Window("selected fault", (8, 8), show_window=False)
    canvas = window.get_canvas()
    domain = ti.graph.execution_domain()
    canvas.set_execution_domain(domain)
    frame = canvas.acquire_frame(8, 8)
    assert canvas.submit_frame(frame, track_source=True)
    frame.source_completion.wait()
    window.get_image_buffer_as_numpy()
    program = impl.get_runtime().prog
    program._debug_inject_runtime_fault(
        700, "selected_display_fault", "injected fatal CUDA fault"
    )
    with pytest.raises(RuntimeError, match="selected_display_fault"):
        canvas.repeat_frame()
    # The graphics device cannot be freed when a fatal CUDA owner forbids the
    # external retirement proof. This uses synthetic fault state, not a claim
    # that a driver/device reset was induced or recovered in place.
    window.destroy()
    assert program._debug_runtime_fault_state()["operation"] == "selected_display_fault"
    with pytest.raises(RuntimeError, match="fatal backend error"):
        domain.close()
    domain.release()
    ti.reset()
