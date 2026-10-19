"""No-DSL AOT triple-slot exchange on the actual associated Vulkan device."""

import array
import faulthandler
import json
import os
from pathlib import Path
import sys

from aot_external_memory_probe import NoFrontend
from cuda_cig_raw_producer import RawProducer
from cuda_cig_test_owner import NativeCigOwner

sys.meta_path.insert(0, NoFrontend())
from taichi_forge_runtime.aot import Runtime, Scalar, configure_cuda_context, cuda_context_info


def check(folder, turns=10000):
    faulthandler.enable()
    faulthandler.dump_traceback_later(30, repeat=True)
    print("CIG_LIFETIME creating owner", flush=True)
    owner = NativeCigOwner()
    use_cig = os.environ.get("TI_TEST_CIG_MODE", "required") != "disabled"
    configure_cuda_context(owner.binding if use_cig else None, mode="required" if use_cig else "disabled")
    print("CIG_LIFETIME configured", flush=True)
    report = dict(turns=turns, slots=3, associated_vulkan_device=True,
                  device_local_exchange=True, readback_only_after_drain=True, scenes=[])
    try:
        for scene in range(2):
            producers = []
            try:
                with Runtime("cuda") as runtime:
                    print("CIG_LIFETIME CUDA runtime", scene, flush=True)
                    module = runtime.load_module(folder)
                    domains, calls, images = [], [], []
                    for slot in range(3):
                        print("CIG_LIFETIME slot", scene, slot, flush=True)
                        producer = RawProducer(owner)
                        assert producer.memory_property_flags & 1
                        producers.append(producer)
                        image = runtime.import_cuda_external_memory(
                            memory_handle=producer.handles[0], ready_for_cuda=producer.handles[1],
                            ready_for_external=producer.handles[2], allocation_size=producer.allocation_size,
                            device_uuid=producer.uuid, dtype="f32", shape=64, byte_offset=4,
                        )
                        images.append(image)
                        domain = runtime.execution_domain(capacity=1, instance_capacity=2)
                        domains.append(domain)
                        call = module.prepare_execution("public_stage", dict(out=image, seed=Scalar("f32", slot + 2)),
                                                        execution_domain=domain)
                        call.initialize()
                        calls.append(call)
                    pending = [None] * 3

                    def exchange(count):
                        for turn in range(count):
                            slot = turn % 3
                            previous = pending[slot]
                            if previous is not None:
                                previous.wait()  # Explicit bounded admission, not a hidden submit wait.
                                previous.close()
                            producers[slot].submit(signal=True)
                            ticket = calls[slot].submit()
                            assert ticket is not None
                            producers[slot].submit(wait=True)
                            pending[slot] = ticket

                    def drain():
                        for slot, ticket in enumerate(pending):
                            if ticket is not None:
                                ticket.wait()
                                ticket.close()
                                pending[slot] = None
                        for producer in producers:
                            producer.wait()

                    exchange(30)
                    print("CIG_LIFETIME warm submitted", flush=True)
                    drain()
                    before = cuda_context_info()["retained_pool"]
                    exchange(turns if scene == 0 else 300)
                    drain()
                    after = cuda_context_info()["retained_pool"]
                    for name in (() if before is None else ("native_allocation_calls", "native_free_calls", "reserved_current_bytes",
                                 "reserved_high_bytes", "used_current_bytes", "used_high_bytes")):
                        assert before[name] == after[name], (name, before, after)
                    for slot, producer in enumerate(producers):
                        values = array.array("f")
                        values.frombytes(producer.read())
                        scale = (slot + 2) * 5 + 1
                        assert values.tolist() == [0] + [scale * 64 * 65 // 2 + i for i in range(64)] + [0]
                    report["scenes"].append(dict(before=before, after=after, oracle=True))
                    # Deliberately close with accepted work still in flight.
                    exchange(3)
                    runtime.close()
                    for ticket in pending:
                        assert ticket.done()
                        ticket.close()
                    for producer in producers:
                        producer.wait()
            finally:
                for producer in producers:
                    producer.close()
        assert owner.references == (2 if use_cig else 1)
        report["final"] = cuda_context_info()
        report["native_owner_references_before_caller_release"] = owner.references
        return report
    finally:
        owner.close()


if __name__ == "__main__":
    report = check(sys.argv[1], int(sys.argv[3]) if len(sys.argv) > 3 else 10000)
    Path(sys.argv[2]).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("CIG_AOT_TRIPLE_SLOT_OK", json.dumps(report), flush=True)
