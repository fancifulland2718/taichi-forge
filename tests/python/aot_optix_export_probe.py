"""Consume automatically exported OptiX recordings without producer objects."""

import json
from pathlib import Path
import statistics
import sys
import time

from taichi_forge_runtime.aot import Runtime
from tests.python.aot_optix_probe import VERTICES, RAYS, check, read, write, raises


def main(mode, artifact, adapter, output):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    with runtime:
        runtime.configure_provider("optix", adapter_path=adapter)
        start = time.perf_counter()
        module = runtime.load_module(artifact)
        costs = dict(load_ms=(time.perf_counter() - start) * 1000, records=[])
        retained = []
        shapes = dict(
            seed=(6, 3),
            indices=(2, 3),
            rays=(4, 8),
            shift=(1,),
            vertices=(6, 3),
            hits=(4, 4),
            ids=(4, 4),
            occluded=(4,),
            loss=(1,),
            dloss=(1,),
            dhits=(4, 4),
            faces=(2,),
        )
        costs["public_resources"] = check_public_resources(runtime, module, artifact, shapes)
        for policy in ("two_sided", "front_only", "table", "query_only"):
            for binding in range(2):
                plan = module.execution_plan(policy)
                frame = {
                    name: runtime.ndarray(
                        "i32" if name in ("indices", "ids", "occluded", "faces") else "f32", shapes[name]
                    )
                    for name in plan.argument_names
                }
                seed = VERTICES[:]
                initial_shift = 0.25 * binding
                for i in range(2, len(seed), 3):
                    seed[i] += initial_shift
                write(frame["seed"], seed)
                write(frame["indices"], range(6), "i")
                write(frame["rays"], sum(RAYS, []))
                write(frame["hits"], [-99] * 16)
                start = time.perf_counter()
                call = plan.prepare(module, frame)
                prepare_ms = (time.perf_counter() - start) * 1000
                assert read(frame["hits"]) == [-99] * 16
                write(frame["indices"], [99] * 6, "i")
                if policy == "query_only":
                    write(frame["seed"], [99] * 18)
                for shift in (0, 4, -0.25, 0):
                    rules = [int(policy != "two_sided")] * 2
                    if policy == "table":
                        rules = [int(shift == 0), 1]
                        write(frame["faces"], rules, "i")
                    if "shift" in frame:
                        write(frame["shift"], [shift])
                    call()
                    check(frame, initial_shift + (shift if policy != "query_only" else 0), rules)
                samples = []
                for _ in range(8):
                    start = time.perf_counter_ns()
                    call()
                    runtime.wait()
                    samples.append((time.perf_counter_ns() - start) / 1000)
                costs["records"].append(
                    dict(
                        policy=policy,
                        binding=binding,
                        prepare_ms=prepare_ms,
                        warm_complete_us=statistics.median(samples),
                        info=call.info,
                    )
                )
                retained.append((frame, call, policy, initial_shift))
        for frame, call, policy, initial_shift in retained[:-1]:
            call()
            check(frame, initial_shift, [int(policy != "two_sided")] * 2)
            call.close()
            for array in frame.values():
                array.close()
        retained[-1][1]()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        raises(retained[-1][1])
        retained[-1][1].close()
        for array in retained[-1][0].values():
            array.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules
    Path(output).write_text(json.dumps(costs, indent=2))


def check_public_resources(runtime, module, artifact, shapes):
    seed = runtime.ndarray("f32", shapes["seed"])
    indices = runtime.ndarray("i32", shapes["indices"])
    write(seed, VERTICES)
    write(indices, range(6), "i")
    context = runtime.native_resource("optix", "context")
    scene = runtime.native_resource(
        "optix",
        "triangle_scene",
        parameters=dict(vertex_count=6, triangle_count=2, allow_update=True),
        bindings=dict(context=context, vertices=seed, indices=indices),
    )
    peer = runtime.load_module(artifact)
    prepared = []
    records = []
    for owner, policy in ((module, "table"), (peer, "query_only")):
        plan = owner.execution_plan(policy + "_public")
        frame = {
            name: runtime.ndarray("i32" if name in ("ids", "occluded", "faces") else "f32", shapes[name])
            for name in plan.argument_names
            if name != "shared_scene"
        }
        write(frame["rays"], sum(RAYS, []))
        write(frame["hits"], [-99] * 16)
        if policy == "table":
            write(frame["seed"], VERTICES)
            write(frame["faces"], [1, 1], "i")
        start = time.perf_counter()
        call = plan.prepare(owner, dict(frame, shared_scene=scene))
        records.append(dict(policy=policy, prepare_ms=(time.perf_counter() - start) * 1000, info=call.info))
        assert read(frame["hits"]) == [-99] * 16
        prepared.append((frame, call))
    for shift in (0, 4, -0.25, 0):
        write(prepared[0][0]["shift"], [shift])
        for frame, call in prepared:
            call()
            check(frame, shift, [1, 1])
    # Query and refit plans in different modules retain one scene. Closing its
    # context waits for in-flight work and retires both plans transitively.
    for record, (_, call) in zip(records, prepared):
        samples = []
        for _ in range(8):
            start = time.perf_counter_ns()
            call()
            runtime.wait()
            samples.append((time.perf_counter_ns() - start) / 1000)
        record["warm_complete_us"] = statistics.median(samples)
    # Compare the same table plan and array bindings in alternating order; the
    # functional test's earlier/later samples are sensitive to cold GPU clocks.
    frame, public_call = prepared[0]
    private_call = module.execution_plan("table").prepare(module, dict(frame, indices=indices))
    calls = (public_call, private_call)
    for _ in range(4):
        for call in calls:
            call()
            runtime.wait()
    samples = [[], []]
    submits = [[], []]
    for _ in range(4):
        for index in (0, 1, 1, 0):
            start = time.perf_counter_ns()
            calls[index]()
            submitted = time.perf_counter_ns()
            runtime.wait()
            samples[index].append((time.perf_counter_ns() - start) / 1000)
            submits[index].append((submitted - start) / 1000)
    check(frame, 0, [1, 1])
    records[0]["alternating"] = {
        name: dict(complete_us=statistics.median(samples[i]), submit_us=statistics.median(submits[i]))
        for i, name in enumerate(("public", "private"))
    }
    private_call.close()
    prepared[0][1]()
    prepared[1][1]()
    context.close()
    for frame, call in prepared:
        raises(call)
        call.close()
        for value in frame.values():
            value.close()
    scene.close()
    seed.close()
    indices.close()
    peer.close()
    return records


if __name__ == "__main__":
    main(*sys.argv[1:])
