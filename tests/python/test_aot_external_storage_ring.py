"""Three real external slots retain versions across a delayed consumer return."""

import array
from contextlib import ExitStack, nullcontext
import json
import os
import time

import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_three_slot_aot_ring_delayed_consumer_1000_epochs(tmp_path):
    from taichi_forge_runtime.aot import ExecutionPlan, Runtime
    from tests.python.aot_external_memory_probe import Producer, ReadyGate

    if os.environ.get("F3_RING_TRACE") == "1":
        from nvtx import annotate
    else:

        def annotate(_):
            return nullcontext()

    @ti.kernel
    def snapshot(
        counter: ti.types.ndarray(ti.i32, 1), slot: ti.types.ndarray(ti.i32, 1)
    ):
        counter[0] += 1
        slot[0] = counter[0]

    @ti.kernel
    def render(slot: ti.types.ndarray(ti.i32, 1), image: ti.types.ndarray(ti.f32, 1)):
        for i in image:
            image[i] = slot[0] * 1000 + i

    @ti.kernel
    def progress(image: ti.types.ndarray(ti.f32, 1)):
        for i in image:
            image[i] = 314159 + i

    exported = ti.aot.Module()
    for kernel in (snapshot, render, progress):
        exported.add_kernel(kernel)
    for name, args in (
        ("snapshot", ["counter", "slot"]),
        ("render", ["slot", "image"]),
        ("progress", ["image"]),
    ):
        exported.add_execution_plan(
            name,
            ti.aot.ExecutionPlan([dict(kind="kernel", entry=name, arguments=args)]),
        )
    artifact = tmp_path / "external-ring.tcm"
    exported.archive(str(artifact))

    def verify(producer, base):
        values = array.array("f")
        values.frombytes(producer.array.read())
        assert values.tolist() == [0] + [base + i for i in range(64)] + [0]

    with ExitStack() as scope:
        index = int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", "0"))
        producers = [
            Producer(scope.enter_context(Runtime("vulkan", device_index=index)))
            for _ in range(3)
        ]
        for producer in producers:
            scope.callback(producer.close)
        runtime = scope.enter_context(Runtime("cuda"))
        module = runtime.load_module(artifact)
        physics = runtime.execution_domain(capacity=2, instance_capacity=8)
        output = runtime.execution_domain(capacity=2, instance_capacity=8)
        counter = runtime.ndarray("i32", 1)
        counter.write(bytes(4))
        slots = [runtime.ndarray("i32", 1) for _ in range(3)]
        for slot in slots:
            slot.write(bytes(4))
        images = [
            runtime.import_cuda_external_memory(
                memory_handle=producer.handles[0],
                ready_for_cuda=producer.handles[1],
                ready_for_external=producer.handles[2],
                allocation_size=producer.allocation_size,
                device_uuid=producer.uuid,
                dtype="f32",
                shape=64,
                byte_offset=4,
            )
            for producer in producers
        ]
        snapshots = [
            module.prepare_execution(
                "snapshot", dict(counter=counter, slot=slot), execution_domain=physics
            )
            for slot in slots
        ]
        renders = [
            module.prepare_execution(
                "render", dict(slot=slot, image=image), execution_domain=output
            )
            for slot, image in zip(slots, images)
        ]
        sentinel = module.prepare_execution(
            "progress", dict(image=images[1]), execution_domain=physics
        )
        for call in (*snapshots, *renders, sentinel):
            call.initialize()
        for producer in producers:
            producer.submit(signal=True)
        prior = [None] * 3
        counts = [0] * 3
        gate = None
        isolation = {}
        domains_before = [physics.info, output.info]
        try:
            for epoch in range(1000):
                slot = epoch % 3
                dependencies = () if prior[slot] is None else (prior[slot],)
                with annotate("F3/ring/physics"):
                    copied = snapshots[slot].submit(wait_for=dependencies)
                assert copied is not None
                with annotate("F3/ring/render"):
                    written = renders[slot].submit(wait_for=(copied,))
                assert written is not None
                copied.close()
                if prior[slot] is not None:
                    prior[slot].close()
                prior[slot] = written
                if epoch == 3:
                    # Slot 0's consumer retained ownership after reading version
                    # 1. Its return signal is genuinely queued behind a host
                    # event; attempted reuse of the slot must remain pending.
                    assert gate is not None and not written.done()
                    with annotate("F3/ring/sentinel"):
                        independent = sentinel.submit()
                    assert independent is not None
                    deadline = time.monotonic() + 2
                    while not independent.done() and time.monotonic() < deadline:
                        time.sleep(0.001)
                    assert independent.done()
                    producers[1].submit(wait=True)
                    verify(producers[1], 314159)
                    isolation["sentinel_done_ns"] = time.perf_counter_ns()
                    assert not written.done() and not gate.timed_out
                    producers[1].submit(signal=True)
                    independent.close()
                    isolation["blocker_release_ns"] = time.perf_counter_ns()
                    with annotate("F3/ring/release_consumer"):
                        gate.close()
                    gate = None
                producers[slot].submit(wait=True)
                verify(producers[slot], (epoch + 1) * 1000)
                assert written.done()
                counts[slot] += 1
                if epoch == 0:
                    # Delay the consumer's return of slot 0, while slots 1 and
                    # 2 continue normally. Creation enqueues its signal before
                    # the next CUDA wait; no invalid unrecorded event is used.
                    gate = ReadyGate(producers[0])
                else:
                    producers[slot].submit(signal=True)
                assert physics.info["pending"] <= 2 and output.info["pending"] <= 2
            assert counts == [334, 333, 333]
            assert isolation["sentinel_done_ns"] < isolation["blocker_release_ns"]
            assert array.array("i", counter.read()).tolist() == [1000]
            assert [physics.info["stream"], output.info["stream"]] == [
                info["stream"] for info in domains_before
            ]
            assert domains_before[0]["stream"] != domains_before[1]["stream"]
            for producer in producers:
                producer.assert_handles_live()
            (tmp_path / "external-ring-evidence.json").write_text(
                json.dumps(
                    dict(
                        epochs=1000,
                        slot_epochs=counts,
                        imported_external_allocations=3,
                        snapshot_instances=3,
                        render_instances=3,
                        probe_instances=1,
                        isolation=isolation,
                        domains=domains_before,
                        versions_and_guards_verified=1000,
                        readback_is_qualification_only=True,
                        application_fps_qualified=False,
                    ),
                    indent=2,
                )
                + "\n"
            )
        finally:
            if gate is not None:
                gate.close()
            for ticket in prior:
                if ticket is not None:
                    ticket.close()
            physics.close()
            output.close()
            for call in (*snapshots, *renders, sentinel):
                call.close()
            for image in images:
                image.close()
            module.close()
