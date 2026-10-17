"""Rebuild exported shared geometry and IAS recordings in a fresh consumer."""

import json
from pathlib import Path
import statistics
import sys
import time

from taichi_forge_runtime.aot import Runtime
from tests.python.aot_optix_instances_probe import BASE_TRANSFORMS, RAYS, SECOND, check
from tests.python.aot_optix_probe import VERTICES, raises, read, write


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
        shapes = dict(
            seed=(6, 3),
            seed1=(3, 3),
            indices=(2, 3),
            indices1=(1, 3),
            initial_transforms=(3, 12),
            requested=(3, 12),
            shift=(3,),
            rays=(len(RAYS), 8),
            hits=(len(RAYS), 4),
            ids=(len(RAYS), 4),
            occluded=(len(RAYS),),
            loss=(1,),
            dhits=(len(RAYS), 4),
            faces0=(2,),
            faces1=(1,),
        )
        retained = []
        for policy in ("two_sided", "table", "query_only"):
            for binding in range(2):
                plan = module.execution_plan(policy)
                frame = {
                    name: runtime.ndarray(
                        "i32" if name in ("indices", "indices1", "ids", "occluded", "faces0", "faces1") else "f32",
                        shapes[name],
                    )
                    for name in plan.argument_names
                }
                write(frame["seed"], VERTICES)
                write(frame["seed1"], SECOND)
                write(frame["indices"], range(6), "i")
                write(frame["indices1"], range(3), "i")
                initial = [row[:] for row in BASE_TRANSFORMS]
                for row in initial:
                    row[11] += 0.25 * binding
                write(frame["initial_transforms"], sum(initial, []))
                write(frame["rays"], sum(RAYS, []))
                write(frame["hits"], [-99] * (4 * len(RAYS)))
                start = time.perf_counter()
                call = plan.prepare(module, frame)
                prepare_ms = (time.perf_counter() - start) * 1000
                assert read(frame["hits"]) == [-99] * (4 * len(RAYS))
                for name in ("indices", "indices1", "initial_transforms"):
                    write(
                        frame[name],
                        [99] * (6 if name == "indices" else 3 if name == "indices1" else 36),
                        "i" if name != "initial_transforms" else "f",
                    )
                if policy == "query_only":
                    write(frame["seed"], [99] * 18)
                    write(frame["seed1"], [99] * 9)
                mirrored = [row[:] for row in BASE_TRANSFORMS]
                mirrored[0][0], mirrored[0][10] = -1, 1.25
                mirrored[1][0], mirrored[1][5], mirrored[1][10] = 2, 0.5, 1.25
                cases = (
                    ([0, 0, 0], BASE_TRANSFORMS),
                    ([2, 0, 0], BASE_TRANSFORMS),
                    ([0, 0, 0.25], mirrored),
                    ([0, 0, 4], BASE_TRANSFORMS),
                    ([0, 0, 0], BASE_TRANSFORMS),
                )
                for index, (shift, transforms) in enumerate(cases):
                    rules = [0] * 5 if policy == "two_sided" else [index % 2, 1, 1, 1, (index + binding) % 2]
                    if "faces0" in frame:
                        write(frame["faces0"], rules[:2], "i")
                        write(frame["faces1"], rules[4:], "i")
                    if "requested" in frame:
                        write(frame["shift"], shift)
                        write(frame["requested"], sum(transforms, []))
                    call()
                    check(
                        frame,
                        [0, 0, 0] if policy == "query_only" else shift,
                        initial if policy == "query_only" else transforms,
                        rules,
                    )
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
                retained.append((frame, call, initial if policy == "query_only" else BASE_TRANSFORMS, rules))
        for frame, call, transforms, rules in retained[:-1]:
            call()
            check(frame, [0, 0, 0], transforms, rules)
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
