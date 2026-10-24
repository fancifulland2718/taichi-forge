"""Private build/refit, vertex+fragment AS reads and downstream gradients on one domain."""

from contextlib import ExitStack, closing
import copy
import json
import os
from pathlib import Path
import sys
import time

from taichi_forge_runtime import runtime_contract_manifest
from taichi_forge_runtime.aot import ExecutionPlan, Runtime, Scalar
from aot_graphics_as_probe import POSE, VERTICES
from aot_optix_probe import raises, read, write


def frame_arrays(runtime, scope, lane):
    frame = {
        name: scope.enter_context(closing(runtime.ndarray(dtype, shape)))
        for name, dtype, shape in (
            ("seed_vertices", "f32", (3, 3)),
            ("seed_indices", "i32", (1, 3)),
            ("poses", "f32", (1, 12)),
            ("output", "f32", 8),
            ("dout", "f32", 8),
            ("loss", "f32", 1),
        )
    }
    frame["vertices"] = scope.enter_context(closing(runtime.ndarray("f32", 6, usage=("vertex",))))
    write(frame["seed_vertices"], VERTICES)
    write(frame["seed_indices"], [0, 1, 2], "i")
    write(frame["poses"], POSE)
    write(frame["vertices"], [-1, -1, 3, -1, -1, 3])
    write(frame["output"], [99] * 8)
    frame["phase"] = Scalar("u64", 1 << (48 + lane))
    return frame


def check(frame, phase):
    expected = [0, -1, -1, -1] * 2 if phase & 1 else [1, 0, 17, 17, 0, -1, -1, 17]
    assert read(frame["output"]) == expected
    assert read(frame["dout"]) == expected
    assert read(frame["loss"]) == [sum(0.5 * x * x for x in expected)]


def main(mode, artifact):
    assert runtime_contract_manifest()["features"]["vulkan_execution_domain_private_graphics_as"]
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.vulkan, offline_cache=False)
        owner = ti.aot.borrow_runtime()
    else:
        owner = Runtime("vulkan", device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", "0")))
    evidence, retained, submits = [], [], []
    with ExitStack() as scope:
        runtime = scope.enter_context(owner)
        module = scope.enter_context(runtime.load_module(artifact))
        domains = [
            scope.enter_context(
                runtime.execution_domain(capacity=1, instance_capacity=1, queue="graphics", gpu_timing=True)
            )
            for _ in range(2)
        ]
        frames = [frame_arrays(runtime, scope, lane) for lane in range(2)]
        plan = module.execution_plan("private")
        document = json.loads(plan.to_json())
        # Fail after private AS creation, then reuse the same instance capacity.
        bad = copy.deepcopy(document)
        raster = next(op for op in bad["operations"] if op.get("provider") == "vulkan_graphics")
        raster["parameters"]["draws"][0]["scenes"] *= 2
        raises(
            lambda: ExecutionPlan.from_json(json.dumps(bad)).prepare(module, frames[0], execution_domain=domains[0]),
            "duplicated",
        )
        assert domains[0].info["accepting"] and domains[0].info["pending"] == 0
        calls = []
        for lane, (domain, frame) in enumerate(zip(domains, frames)):
            started = time.perf_counter()
            call = scope.enter_context(
                plan.prepare(module, frame, execution_domain=domain, mutable_parameters=("phase",))
            )
            prepare_ms = (time.perf_counter() - started) * 1000
            assert read(frame["output"]) == [99] * 8
            raises(lambda: call.submit(values=dict(phase=0)))
            started = time.perf_counter()
            call.initialize()
            initialize_ms = (time.perf_counter() - started) * 1000
            # Resource initialization must not run the first frame eagerly.
            assert read(frame["output"]) == [99] * 8
            evidence.append(dict(lane=lane, prepare_ms=prepare_ms, initialize_ms=initialize_ms, info=call.info))
            calls.append(call)
        for step in (0, 1, 0):
            tickets = []
            for lane, call in enumerate(calls):
                values = dict(phase=(1 << (48 + lane)) | ((step + lane) % 2))
                started = time.perf_counter_ns()
                ticket = call.submit(values=values, sample_gpu=step == 1)
                submits.append(dict(lane=lane, step=step, host_us=(time.perf_counter_ns() - started) / 1000))
                assert ticket is not None
                values["phase"] ^= 1
                tickets.append(ticket)
            for lane, ticket in enumerate(tickets):
                ticket.wait()
                check(frames[lane], (step + lane) % 2)
                timing = ticket.gpu_timing
                assert timing["status"] == ("ready" if step == 1 else "not_sampled")
                if step == 1:
                    assert timing["duration_ns"] > 0 and timing["scope"] == "execution_envelope"
                    assert timing["sequence"] == ticket.info["sequence"]
                    retained.append((ticket, timing))
                else:
                    ticket.close()
        pending = calls[0].submit(values=dict(phase=(1 << 48) | 1))
        calls[0].close()
        pending.wait()
        check(frames[0], 1)
        pending.close()
        with calls[1].submit(values=dict(phase=1 << 49)) as ticket:
            ticket.wait()
            check(frames[1], 0)
        calls[1].close()

        # The graphics operation must reject a shared live TLAS on its own,
        # without an earlier refit operation concealing a missing closure check.
        blas = scope.enter_context(
            closing(
                runtime.native_resource(
                    "vulkan_ray",
                    "triangle_blas",
                    parameters=dict(vertex_count=3, triangle_count=1),
                    bindings=dict(vertices=frames[0]["seed_vertices"], indices=frames[0]["seed_indices"]),
                )
            )
        )
        scene = scope.enter_context(
            closing(
                runtime.native_resource(
                    "vulkan_ray",
                    "instance_tlas",
                    parameters=dict(instance_count=1, custom_indices=[17]),
                    bindings=dict(blas_0=blas, transforms=frames[0]["poses"]),
                )
            )
        )
        public = json.loads(module.execution_plan("public").to_json())
        public["operations"] = [op for op in public["operations"] if op.get("provider") != "vulkan_ray"]
        shared = ExecutionPlan.from_json(json.dumps(public))
        available = dict(frames[0], scene=scene)
        arguments = {name: available[name] for name in shared.argument_names}
        with shared.prepare(module, arguments) as ordinary:
            ordinary.initialize()
            ordinary.run()
            check(frames[0], 0)
        raises(
            lambda: shared.prepare(module, arguments, execution_domain=domains[0]),
            "must belong to this prepared instance",
        )
        with plan.prepare(module, frames[0], execution_domain=domains[0], mutable_parameters=("phase",)) as rebound:
            rebound.initialize()
            ticket = rebound.submit(values=dict(phase=1 << 48), sample_gpu=True)
            if mode == "borrowed":
                ti.reset()
                raises(lambda: rebound.submit(values=dict(phase=0)))
            ticket.wait()
            assert ticket.done()
            if mode != "borrowed":
                check(frames[0], 0)
            retained.append((ticket, ticket.gpu_timing))
    for ticket, timing in retained:
        assert timing["status"] == "ready" and ticket.gpu_timing == timing
        ticket.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules
    Path(artifact).with_name(mode + ".json").write_text(
        json.dumps(dict(instances=evidence, submits=submits, timing=[t for _, t in retained]), indent=2)
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
