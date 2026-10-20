"""No-DSL AOT check: an unready display input must not hold independent work.

The native fixture uses a real Vulkan fence after the producer release signal.
An artificial host timeline holds that producer to make the isolation oracle
deterministic. Completion polling duration is not a GPU kernel timing metric.
"""

import array
import ctypes as C
import json
import os
from pathlib import Path
import statistics
import sys
import time

from aot_external_memory_probe import NoFrontend, VkSubmit, vk_check
from cuda_cig_raw_producer import RawProducer
from cuda_cig_test_owner import NativeCigOwner

sys.meta_path.insert(0, NoFrontend())
from taichi_forge_runtime.aot import (
    AotError, CudaExternalReadyGate, Runtime, Scalar, configure_cuda_context, cuda_context_info,
)

_CAPSULE_NAME = b"taichi.cuda_external_ready_gate.v1"


class NativeReadyGate:
    def __init__(self, owner):
        self.library = owner.library
        create = self.library.cig_fixture_ready_create
        create.argtypes, create.restype = [C.c_void_p], C.c_void_p
        self.address = create(owner.address)
        assert self.address, self.library.cig_fixture_error()
        self.library.cig_fixture_ready_value.argtypes = [C.c_void_p, C.c_uint]
        self.library.cig_fixture_ready_value.restype = C.c_uint64
        self.library.cig_fixture_ready_action.argtypes = [C.c_void_p, C.c_uint]
        self.library.cig_fixture_ready_action.restype = C.c_int
        capsule = C.pythonapi.PyCapsule_New
        capsule.argtypes, capsule.restype = [C.c_void_p, C.c_char_p, C.c_void_p], C.py_object
        self.binding = CudaExternalReadyGate(capsule(self.address, _CAPSULE_NAME, None))

    def value(self, field):
        return self.library.cig_fixture_ready_value(self.address, field)

    def action(self, action):
        assert self.library.cig_fixture_ready_action(self.address, action) == 0, self.library.cig_fixture_error()

    def close(self):
        if self.address:
            self.action(5)
            self.address = None


class ReadyProducer(RawProducer):
    def __init__(self, owner):
        super().__init__(owner)
        self.ready = NativeReadyGate(owner)

    def submit(self, *, signal=False, wait=False):
        assert signal != wait
        if signal:
            self.wait()
            self.ready.action(0)  # Next epoch; reset only its producer fence.
        else:
            vk_check(self.reset_native(self.info.device, 1, C.byref(C.c_uint64(self.fence))))
        waits = (C.c_uint64 * 1)(self.semaphores[1])
        signals = (C.c_uint64 * 1)(self.semaphores[0])
        stages = (C.c_uint32 * 1)(0x10000)
        commands = (C.c_void_p * 1)(self.command.value if signal else self.acquire_command.value)
        info = VkSubmit(4, None, int(wait), waits if wait else None, stages if wait else None,
                        1, C.cast(commands, C.c_void_p), int(signal), signals if signal else None)
        fence = self.ready.value(0) if signal else self.fence
        vk_check(self.submit_native(self.info.compute_queue, 1, C.byref(info), fence))

    def close(self):
        self.ready.action(2)
        super().close()
        self.ready.close()


def poll(ticket, seconds=0.1):
    deadline = time.perf_counter() + seconds
    while not ticket.done():
        if time.perf_counter() >= deadline:
            return False
        time.sleep(0)
    return True


def retry(call):
    deadline = time.perf_counter() + 2
    while True:
        ticket = call.submit(ready_only=True)
        if ticket is not None:
            return ticket
        assert time.perf_counter() < deadline, "Ready input was never admitted"
        time.sleep(0)


def check(folder, *, gated=True, turns=100):
    owner = NativeCigOwner()
    mode = os.environ.get("TI_TEST_CIG_MODE", "required")
    configure_cuda_context(owner.binding if mode == "required" else None, mode=mode)
    producer = ReadyProducer(owner)
    report = dict(mode=mode, gated=gated, turns=turns)
    try:
        with Runtime("cuda") as runtime:
            module = runtime.load_module(folder)
            image = runtime.import_cuda_external_memory(
                memory_handle=producer.handles[0], ready_for_cuda=producer.handles[1],
                ready_for_external=producer.handles[2], allocation_size=producer.allocation_size,
                device_uuid=producer.uuid, dtype="f32", shape=64, byte_offset=4,
                ready_gate=producer.ready.binding if gated else None,
            )
            assert producer.ready.value(1) == (2 if gated else 1)
            domain = runtime.execution_domain(capacity=1, instance_capacity=2)
            peer = runtime.execution_domain(capacity=1, instance_capacity=2)
            private = runtime.ndarray("f32", 64)
            render = module.prepare_execution("public_stage", dict(out=image, seed=Scalar("f32", 2)),
                                              execution_domain=domain)
            physics = module.prepare_execution("public_stage", dict(out=private, seed=Scalar("f32", 3)),
                                               execution_domain=peer)
            render.initialize()
            physics.initialize()
            assert render.info["execution_kind"] == "cuda_graph"
            with domain.latest_completion() as prefix:
                sequence = prefix.info["sequence"]
            producer.ready.action(1)
            try:
                producer.submit(signal=True)
                started = time.perf_counter_ns()
                pending = render.submit(ready_only=True)
                report["render_admission_us"] = (time.perf_counter_ns() - started) / 1000
                if gated:
                    assert pending is None
                    durations = []
                    for _ in range(1000):
                        started = time.perf_counter_ns()
                        assert render.submit(ready_only=True) is None
                        durations.append((time.perf_counter_ns() - started) / 1000)
                    assert producer.ready.value(2) == 1
                    assert producer.ready.value(3) == 1001
                    assert domain.info["pending"] == 0
                    with domain.latest_completion() as prefix:
                        assert prefix.info["sequence"] == sequence
                    report["unready_attempts"] = len(durations)
                    report["unready_submit_p50_us"] = statistics.median(durations)
                    report["unready_submit_max_us"] = max(durations)
                    producer.ready.action(3)
                    try:
                        render.submit()
                    except AotError as error:
                        assert "ready gate query failed" in str(error)
                    else:
                        raise AssertionError("Native producer error was ignored")
                    producer.ready.action(4)
                    assert domain.info["pending"] == 0
                else:
                    assert pending is not None
                    if os.environ.get("TI_TEST_READY_DEP_QUERY"):
                        assert physics.submit(wait_for=(pending,), ready_only=True) is None
                independent = physics.submit()
                assert independent is not None
                started = time.perf_counter_ns()
                report["independent_completed_before_release"] = poll(independent)
                report["independent_observation_us"] = (time.perf_counter_ns() - started) / 1000
                if gated or mode == "disabled":
                    assert report["independent_completed_before_release"], report
                report["independent_replays_before_release"] = 0
                if report["independent_completed_before_release"]:
                    for _ in range(32):
                        next_physics = physics.submit(ready_only=True)
                        if next_physics is None:
                            break
                        progressed = poll(next_physics, seconds=0.05)
                        next_physics.close()
                        if not progressed:
                            break
                        report["independent_replays_before_release"] += 1
                if gated or mode == "disabled":
                    assert report["independent_replays_before_release"] == 32, report
            finally:
                producer.ready.action(2)
            if pending is None:
                pending = retry(render)
            producer.submit(wait=True)
            pending.wait()
            pending.close()
            independent.wait()
            independent.close()
            producer.wait()
            if gated:
                # The old signaled fence cannot authorize another semaphore wait.
                assert render.submit() is None
                assert domain.info["pending"] == 0

            def cycle():
                producer.submit(signal=True)
                ticket = retry(render)
                producer.submit(wait=True)
                ticket.wait()
                ticket.close()
                producer.wait()

            for _ in range(10):
                cycle()
            before = cuda_context_info()["retained_pool"]
            steady_started = time.perf_counter_ns()
            for _ in range(turns):
                cycle()
            report["serial_completed_exchange_us"] = (time.perf_counter_ns() - steady_started) / (turns * 1000)
            after = cuda_context_info()["retained_pool"]
            for name in (() if before is None else (
                    "native_allocation_calls", "native_free_calls", "reserved_current_bytes")):
                assert before[name] == after[name], (name, before, after)
            values = array.array("f")
            values.frombytes(producer.read())
            assert values.tolist() == [0] + [11 * 64 * 65 // 2 + i for i in range(64)] + [0]
            actual = array.array("f")
            actual.frombytes(private.read())
            assert actual.tolist() == [16 * 64 * 65 // 2 + i for i in range(64)]
            report.update(oracle=True, before=before, after=after, queries=producer.ready.value(3))
        assert producer.ready.value(1) == 1, "Native query owner leaked after runtime retirement"
        report["owner_released"] = True
        return report
    finally:
        producer.close()
        owner.close()


if __name__ == "__main__":
    result = check(sys.argv[1], gated=sys.argv[3] != "baseline", turns=int(sys.argv[4]))
    Path(sys.argv[2]).write_text(json.dumps(result, indent=2) + "\n")
    print("EXTERNAL_READY_OK", json.dumps(result), flush=True)
