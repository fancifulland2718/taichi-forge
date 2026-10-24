"""Runtime-only private raster instances, parameter snapshots and retirement."""

from contextlib import ExitStack, closing
import json
import os
import sys
import time

from taichi_forge_runtime.aot import ExecutionPlan, Runtime, Scalar
from aot_optix_probe import read


def main(folder):
    kept = []
    evidence = []
    with ExitStack() as scope:
        runtime = scope.enter_context(Runtime(
            "vulkan", device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", "0"))))
        module = scope.enter_context(runtime.load_module(folder))
        domains = [scope.enter_context(runtime.execution_domain(
            queue="graphics", capacity=2, instance_capacity=1, gpu_timing=True)) for _ in range(2)]
        assert domains[0].info["queue_family_index"] == domains[1].info["queue_family_index"]
        assert domains[0].info["queue_index"] != domains[1].info["queue_index"]
        frames, calls = [], []
        plan = module.execution_plan("offscreen")
        for domain in domains:
            frame = {key: scope.enter_context(closing(runtime.ndarray("f32", size)))
                     for key, size in (("output", 128), ("doutput", 128), ("loss", 1), ("dloss", 1))}
            frame["scale"] = Scalar("u64", 1)
            # LOAD cannot initialize newly reconstructed private attachments.
            bad = json.loads(plan.to_json())
            bad["operations"][0]["parameters"]["colors"][0]["clear"] = False
            try:
                ExecutionPlan.from_json(json.dumps(bad)).prepare(module, frame, execution_domain=domain)
            except RuntimeError as error:
                assert "unconditional upload" in str(error), str(error)
            else:
                raise AssertionError("Uninitialized attachment accepted")
            assert domain.info["pending"] == 0 and domain.info["accepting"]
            started = time.perf_counter()
            call = scope.enter_context(plan.prepare(module, frame, execution_domain=domain,
                                                    mutable_parameters=("scale",)))
            evidence.append(dict(prepare_ms=(time.perf_counter() - started) * 1000,
                                 domain=domain.info, instance=call.info))
            call.initialize()
            frames.append(frame)
            calls.append(call)
        def check(frame, scale):
            expected = [0.5 * scale if i % 8 < 4 else 0 for i in range(64)]
            expected += [0 if i % 8 < 4 else scale for i in range(64)]
            assert read(frame["output"]) == expected, read(frame["output"])
            assert read(frame["doutput"]) == expected
            assert read(frame["loss"]) == [sum(x * x * 0.5 for x in expected)]
        for step in range(4):
            tickets = []
            for lane, call in enumerate(calls):
                values = dict(scale=2 * step + lane + 1)
                ticket = call.submit(values=values, sample_gpu=step == 0)
                assert ticket is not None
                values["scale"] = 99
                tickets.append(ticket)
            for lane, ticket in enumerate(tickets):
                ticket.wait()
                check(frames[lane], float(2 * step + lane + 1))
                timing = ticket.gpu_timing
                assert timing["scope"] == "execution_envelope"
                assert timing["sequence"] == ticket.info["sequence"]
                assert timing["status"] == ("ready" if step == 0 else "not_sampled")
                if step == 0:
                    assert timing["duration_ns"] > 0
                    kept.append((ticket, timing))
                else:
                    assert timing["duration_ns"] is None
                    ticket.close()
        pending = calls[0].submit(values=dict(scale=9))
        calls[0].close()
        pending.wait()
        check(frames[0], 9.0)
        pending.close()
        with calls[1].submit(values=dict(scale=10)) as ticket:
            ticket.wait()
            check(frames[1], 10.0)
        # New private attachments and a different output binding reuse the slot.
        with plan.prepare(module, frames[1], execution_domain=domains[0], mutable_parameters=("scale",)) as rebound:
            rebound.initialize()
            with rebound.submit(values=dict(scale=11)) as ticket:
                ticket.wait()
                check(frames[1], 11.0)
    for ticket, expected in kept:
        assert ticket.gpu_timing == expected
        ticket.close()
    assert "taichi_forge" not in sys.modules
    print(json.dumps(dict(instances=evidence, sampled_timing=[result for _, result in kept])))
    print("offscreen_domain_ok")


if __name__ == "__main__":
    main(*sys.argv[1:])
