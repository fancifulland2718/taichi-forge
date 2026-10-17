"""Consume exported transfer Graphs without producer source or image snapshots."""

import json
from pathlib import Path
import statistics
import sys
import time

from taichi_forge_runtime.aot import Runtime

if __package__:
    from .aot_optix_probe import raises, read, write
else:
    from aot_optix_probe import raises, read, write


def main(mode, artifact, result):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.vulkan, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("vulkan")
    costs = []
    with runtime:
        start = time.perf_counter()
        module = runtime.load_module(artifact)
        load_ms = (time.perf_counter() - start) * 1000
        frame = {
            name: runtime.ndarray("f32", (n,))
            for name, n in dict(input=32, padded=32, copied=12, output=16, doutput=16, loss=1, dloss=1).items()
        }
        for policy in ("private", "public"):
            seed = runtime.ndarray("f32", (4, 3))
            write(seed, [-7] * 12)
            frame["seed"] = seed
            images = []
            if policy == "public":
                for name, shape in (("a", [3, 2]), ("b", [4, 3]), ("c", [4, 4])):
                    value = runtime.native_resource(
                        "image",
                        "sampled",
                        parameters=dict(shape=shape, format="r32f"),
                        bindings=dict(pixels=seed) if name == "b" else {},
                    )
                    images.append(value)
                    frame[name] = value
            plan = module.execution_plan(policy)
            start = time.perf_counter()
            call = plan.prepare(module, {name: frame[name] for name in plan.argument_names})
            prepare_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            call.initialize()
            initialize_ms = (time.perf_counter() - start) * 1000
            seed.close()  # Consumer seed is cold-only; replay retains images.
            write(frame["padded"], [-99] * 32)
            for scale in (1, 0, -2):
                source = [-88] * 32
                source[3:6], source[8:11] = [scale, 2 * scale, 3 * scale], [4 * scale, 5 * scale, 6 * scale]
                write(frame["input"], source)
                call()
                expected = [x * scale for x in [2, 2, 3, 3] * 2 + [5, 5, 6, 6] * 2]
                assert read(frame["output"]) == read(frame["doutput"]) == expected
                assert read(frame["loss"]) == [sum(x * x / 2 for x in expected)]
                padded = [-99] * 32
                for y in range(4):
                    padded[3 + y * 6 : 7 + y * 6] = expected[4 * y : 4 * y + 4]
                assert read(frame["padded"]) == padded
                assert read(frame["copied"]) == [-7] * 5 + [2 * scale, 3 * scale, -7, -7, 5 * scale, 6 * scale, -7]
            submit, complete = [], []
            for _ in range(4):
                start = time.perf_counter_ns()
                call()
                submitted = time.perf_counter_ns()
                runtime.wait()
                submit.append((submitted - start) / 1000)
                complete.append((time.perf_counter_ns() - start) / 1000)
            costs.append(
                dict(
                    policy=policy,
                    prepare_ms=prepare_ms,
                    initialize_ms=initialize_ms,
                    submit_us=statistics.median(submit),
                    complete_us=statistics.median(complete),
                    info=call.info,
                )
            )
            if policy == "public":
                call()
                images[0].close()
                raises(call)
                for value in images[1:]:
                    value.close()
            call.close()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        for value in frame.values():
            value.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules
    Path(result).write_text(json.dumps(dict(load_ms=load_ms, cases=costs), indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
