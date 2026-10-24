"""Source-free AS instances; GPU traversal is checked against plane intersections."""

from contextlib import ExitStack, closing
import json
import os
import sys
import time

from taichi_forge_runtime.aot import ExecutionPlan, Runtime, Scalar
from taichi_forge_runtime import runtime_contract_manifest
from aot_optix_probe import read, write, raises
from aot_vulkan_ray_probe import native

VERTICES = [-1, -1, 1, 1, -1, 1, 0, 1, 1, -1, -1, 2, 0, 1, 2, 1, -1, 2]
TRANSFORMS = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
              1, 0, 0, 4, 0, 1, 0, 0, 0, 0, 1, 2]
RAYS = [[0, 0, 0, .01, 0, 0, 1, 10], [4, 0, 0, .01, 0, 0, 1, 10],
        [0, 0, 0, .01, 0, 0, 1, 1.5], [0, 0, 1.5, .01, 0, 0, 1, 3],
        [8, 0, 0, .01, 0, 0, 1, 10], [0, 0, 4, .01, 0, 0, -1, 10]]


def frame_arrays(runtime, scope, instanced):
    shapes = dict(seed=(6, 3), indices=(2, 3), rays=(6, 8), hits=(6, 4), ids=(6, 4), dhits=(6, 4), loss=(1,))
    if instanced:
        shapes["initial_transforms"] = (2, 3, 4)
    frame = {key: scope.enter_context(closing(runtime.ndarray("i32" if key in ("indices", "ids") else "f32", shape)))
             for key, shape in shapes.items()}
    write(frame["seed"], VERTICES)
    write(frame["indices"], list(range(6)), "i")
    write(frame["rays"], sum(RAYS, []))
    write(frame["hits"], [99] * 24)
    if instanced:
        write(frame["initial_transforms"], TRANSFORMS)
    frame["phase"] = Scalar("u64", 1 << 48)
    return frame


def check(frame, shift, instanced):
    hits, ids, gradient, loss = [], [], [], 0.0
    for ray in RAYS:
        candidates = []
        for instance in range(2 if instanced else 1):
            if ray[0] != instance * 4:
                continue
            for primitive in range(2):
                t = (primitive + 1 + shift + instance * 2 - ray[2]) / ray[6]
                if ray[3] <= t <= ray[7]:
                    candidates.append((t, primitive, instance))
        if candidates:
            t, primitive, instance = min(candidates)
            u, v = (.25, .5) if primitive == 0 else (.5, .25)
            hits += [t, u, v, 0]
            ids += [primitive, instance, (19 + 4 * instance) if instanced else 0, 1]
            gradient += [t, 0, 0, 0]
            loss += t * t * .5
        else:
            hits += [-1, 0, 0, 0]
            ids += [-1, -1, -1, 0]
            gradient += [0, 0, 0, 0]
    for key, expected in (("hits", hits), ("dhits", gradient), ("loss", [loss])):
        actual = read(frame[key])
        assert len(actual) == len(expected) and all(abs(a - b) < 1e-5 for a, b in zip(actual, expected)), (key, actual, expected)
    assert read(frame["ids"], "i") == ids


def main(folder, mode):
    assert runtime_contract_manifest()["features"]["vulkan_execution_domain_private_ray_resources"]
    evidence = []
    kept = []
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.vulkan, offline_cache=False)
        owner = ti.aot.borrow_runtime()
    else:
        owner = Runtime("vulkan", device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", "0")))
    with ExitStack() as scope:
        runtime = scope.enter_context(owner)
        module = scope.enter_context(runtime.load_module(folder))
        domains = [scope.enter_context(runtime.execution_domain(capacity=1, instance_capacity=1, gpu_timing=True)) for _ in range(2)]
        for instanced in (False, True):
            plan = module.execution_plan("instances" if instanced else "scene")
            with ExitStack() as group:
                frames = [frame_arrays(runtime, group, instanced) for _ in domains]
                calls = []
                # A late typed-query failure rolls back already allocated AS owners.
                bad = json.loads(plan.to_json())
                query = next(op for op in bad["operations"] if op.get("operation") == "query")
                query["bindings"]["hits"] = "rays"
                raises(lambda: ExecutionPlan.from_json(json.dumps(bad)).prepare(module, frames[0], execution_domain=domains[0]))
                assert domains[0].info["pending"] == 0 and domains[0].info["accepting"]
                for domain, frame in zip(domains, frames):
                    started = time.perf_counter()
                    call = group.enter_context(plan.prepare(module, frame, execution_domain=domain, mutable_parameters=("phase",)))
                    preparation_ms = (time.perf_counter() - started) * 1000
                    assert read(frame["hits"]) == [99] * 24
                    raises(lambda: call.submit(values=dict(phase=1 << 48)))
                    started = time.perf_counter()
                    call.initialize()
                    evidence.append(dict(instanced=instanced, prepare_ms=preparation_ms,
                                         initialize_ms=(time.perf_counter() - started) * 1000,
                                         instance=call.info, domain=domain.info))
                    calls.append(call)
                for step in (0, 1, 4, 0):
                    tickets = []
                    for lane, call in enumerate(calls):
                        values = dict(phase=(1 << (48 + lane)) | ((step + lane) & 7))
                        ticket = call.submit(values=values, sample_gpu=step == 1)
                        assert ticket is not None
                        values["phase"] = 7
                        tickets.append(ticket)
                    for lane, ticket in enumerate(tickets):
                        ticket.wait()
                        check(frames[lane], (step + lane) & 7, instanced)
                        timing = ticket.gpu_timing
                        assert timing["status"] == ("ready" if step == 1 else "not_sampled")
                        assert timing["sequence"] == ticket.info["sequence"]
                        if step == 1:
                            assert timing["duration_ns"] > 0 and timing["scope"] == "execution_envelope"
                            kept.append((ticket, timing))
                        else:
                            ticket.close()
                pending = calls[0].submit(values=dict(phase=(1 << 48) | 2))
                calls[0].close()
                pending.wait()
                check(frames[0], 2, instanced)
                pending.close()
                with calls[1].submit(values=dict(phase=(1 << 49) | 3)) as ticket:
                    ticket.wait()
                    check(frames[1], 3, instanced)
                calls[1].close()
                # Shared public BLAS must not become private through a new TLAS.
                public = group.enter_context(closing(runtime.native_resource(
                    "vulkan_ray", "triangle_blas" if instanced else "triangle_scene",
                    parameters=dict(vertex_count=6, triangle_count=2),
                    bindings=dict(vertices=frames[0]["seed"], indices=frames[0]["indices"]))))
                description = json.loads(plan.to_json())
                name = "blas" if instanced else "scene"
                description["resources"] = [r for r in description["resources"] if r["name"] != name]
                public_plan = ExecutionPlan.from_json(json.dumps(description))
                public_arguments = dict(frames[0], **{name: public})
                public_arguments = {key: public_arguments[key] for key in public_plan.argument_names}
                try:
                    public_plan.prepare(module, public_arguments, execution_domain=domains[0])
                except RuntimeError as error:
                    assert "must belong to this prepared instance" in str(error), str(error)
                else:
                    raise AssertionError("Public AS accepted as a private resource")
                with plan.prepare(module, frames[1], execution_domain=domains[0], mutable_parameters=("phase",)) as rebound:
                    rebound.initialize()
                    with rebound.submit(values=dict(phase=(1 << 49) | 1)) as ticket:
                        ticket.wait()
                        check(frames[1], 1, instanced)
                # Deferred builds retain their seed binding through preparation.
                seed_call = group.enter_context(plan.prepare(module, frames[0], execution_domain=domains[0], mutable_parameters=("phase",)))
                frames[0]["indices"].close()
                raises(seed_call.initialize)
        if mode == "borrowed":
            # Program teardown retires a submitted private AS before its device.
            frame = frame_arrays(runtime, scope, True)
            call = scope.enter_context(module.execution_plan("instances").prepare(
                module, frame, execution_domain=domains[0], mutable_parameters=("phase",)))
            call.initialize()
            ticket = call.submit(values=dict(phase=(1 << 49) | 2), sample_gpu=True)
            ti.reset()
            assert ticket.done()
            ticket.wait()
            timing = ticket.gpu_timing
            assert timing["status"] == "ready"
            kept.append((ticket, timing))
            raises(lambda: call.submit(values=dict(phase=0)))
    for ticket, timing in kept:
        assert ticket.gpu_timing == timing
        ticket.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules
    print(json.dumps(dict(instances=evidence, sampled_timing=[v for _, v in kept])))
    print("private_ray_domain_ok")


if __name__ == "__main__":
    main(*sys.argv[1:])
