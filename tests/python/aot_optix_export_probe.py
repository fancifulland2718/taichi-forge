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


if __name__ == "__main__":
    main(*sys.argv[1:])
