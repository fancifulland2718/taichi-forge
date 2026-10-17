"""AOT Graph image ownership, initialization, write aliases and retirement."""

import json
from pathlib import Path
import statistics
import sys
import time

from taichi_forge_runtime.aot import Runtime, Scalar
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
        module = runtime.load_module(artifact)
        retained = []
        for route in ("private", "mixed", "public"):
            a, b = runtime.ndarray("f32", (2, 2)), runtime.ndarray("f32", (2, 2))
            write(a, [2] * 4)
            write(b, [5] * 4)
            frame = {
                name: runtime.ndarray("f32", (1,) if name == "loss" else (6,))
                for name in ("before", "after", "dout", "loss")
            }
            bindings = dict(frame, gain=Scalar("f32", 3))
            owners = []
            if route == "private":
                bindings.update(seed_a=a, seed_b=b)
            else:
                second = runtime.native_resource(
                    "image",
                    "sampled",
                    parameters=dict(shape=[2, 2], format="r32f", storage=True),
                    bindings=dict(pixels=b),
                )
                bindings["public_second"] = second
                owners.append(second)
                if route == "mixed":
                    bindings["seed_a"] = a
                else:
                    first = runtime.native_resource(
                        "image", "sampled", parameters=dict(shape=[2, 2], format="r32f"), bindings=dict(pixels=a)
                    )
                    table = runtime.native_resource(
                        "image",
                        "collection",
                        parameters=dict(dimensions=2, image_count=3),
                        bindings=dict(image_0=second, image_1=first, image_2=second),
                    )
                    bindings["public_table"] = table
                    owners.extend([first, table])
                    wrong = runtime.native_resource(
                        "image",
                        "collection",
                        parameters=dict(dimensions=2, image_count=3),
                        bindings=dict(image_0=first, image_1=second, image_2=first),
                    )
                    write(frame["after"], [-99] * 6)
                    raises(lambda: module.prepare_execution(route, dict(bindings, public_table=wrong)), "alias")
                    assert read(frame["after"]) == [-99] * 6
                    wrong.close()
            start = time.perf_counter()
            call = module.prepare_execution(route, bindings)
            prepare_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            call.initialize()
            initialize_ms = (time.perf_counter() - start) * 1000
            write(a, [99] * 4)
            write(b, [99] * 4)
            a.close()
            b.close()
            call()
            assert read(frame["before"]) == [10, 7, 10] * 2
            assert read(frame["after"]) == [6, 5, 6] * 2
            assert read(frame["dout"]) == [6, 5, 6] * 2
            assert read(frame["loss"]) == [97]
            call()
            assert read(frame["before"]) == [6, 5, 6] * 2
            samples, submits = [], []
            for _ in range(4):
                start = time.perf_counter_ns()
                call()
                submitted = time.perf_counter_ns()
                runtime.wait()
                samples.append((time.perf_counter_ns() - start) / 1000)
                submits.append((submitted - start) / 1000)
            costs.append(
                dict(
                    route=route,
                    prepare_ms=prepare_ms,
                    initialize_ms=initialize_ms,
                    submit_us=statistics.median(submits),
                    complete_us=statistics.median(samples),
                    info=call.info,
                )
            )
            if route != "private":
                call()
                second.close()
                raises(call)
                call.close()
                for owner in owners:
                    owner.close()
            else:
                retained.append(call)
        retained[0]()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        raises(retained[0])
        retained[0].close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules
    Path(result).write_text(json.dumps(costs, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
