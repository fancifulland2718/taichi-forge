"""Consume exported image read/write graphs without importing producer code."""

import json
from pathlib import Path
import sys
import statistics
import time

from taichi_forge_runtime.aot import Runtime
from aot_optix_probe import raises, read, write


def main(mode, artifact):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.vulkan, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("vulkan")
    costs = []
    with runtime:
        module = runtime.load_module(artifact)
        calls, resources = [], []
        for public in (False, True):
            seeds = dict(pixels=runtime.ndarray("f32", (3, 5)), upixels=runtime.ndarray("u32", (3, 5)))
            write(seeds["pixels"], list(range(15)))
            write(seeds["upixels"], [0xF1234567 + i for i in range(15)], "I")
            frame = {
                name: runtime.ndarray(dtype, shape)
                for name, dtype, shape in [
                    ("output", "f32", 4),
                    ("dout", "f32", 4),
                    ("loss", "f32", 1),
                    ("ids", "u32", 15),
                ]
            }
            start = time.perf_counter()
            if public:
                images = dict(
                    public_image=runtime.native_resource(
                        "image",
                        "sampled",
                        parameters=dict(shape=[3, 5], format="r32f", storage=True),
                        bindings=dict(pixels=seeds["pixels"]),
                    ),
                    public_ids=runtime.native_resource(
                        "image",
                        "storage",
                        parameters=dict(shape=[3, 5], format="r32u"),
                        bindings=dict(pixels=seeds["upixels"]),
                    ),
                )
                resources += list(images.values())
                call = module.prepare_execution("public", dict(frame, **images))
            else:
                call = module.prepare_execution("step", dict(frame, **seeds))
                assert call.info["native_resource_bytes"] == 120
            prepare_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            call.initialize()
            initialize_ms = (time.perf_counter() - start) * 1000
            for value in seeds.values():
                value.close()
            host, complete = [], []
            for iteration in range(5):
                start = time.perf_counter_ns()
                call()
                submitted = time.perf_counter_ns()
                runtime.wait()
                host.append((submitted - start) / 1000)
                complete.append((time.perf_counter_ns() - start) / 1000)
                expected = [iteration, 1 + iteration, 1 + iteration, 2 + iteration]
                assert read(frame["output"]) == expected
                assert read(frame["dout"]) == expected
                assert read(frame["loss"]) == [sum(x * x * 0.5 for x in expected)]
                assert read(frame["ids"], "I") == [0xF1234567 + i + 3 * (iteration + 1) for i in range(15)]
            calls.append(call)
            resources += list(frame.values())
            costs.append(
                dict(
                    public=public,
                    prepare_ms=prepare_ms,
                    initialize_ms=initialize_ms,
                    host_us=statistics.median(host),
                    complete_us=statistics.median(complete),
                )
            )
        m0, m1 = runtime.ndarray("f32", (3, 5)), runtime.ndarray("f32", (1, 2))
        out = runtime.ndarray("f32", 4)
        write(m0, list(range(15)))
        write(m1, [7, 9])
        mip = module.prepare_execution("mip", dict(m0=m0, m1=m1, mip_output=out))
        assert mip.info["native_resource_bytes"] == 68
        mip.initialize()
        m0.close()
        m1.close()
        for i in range(2):
            mip()
            assert read(out) == [1, 2, 8 + i, 10 + i]
        calls.append(mip)
        resources.append(out)
        # Public resources and private reconstructions have independent storage.
        calls[0]()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        for call in calls:
            raises(call)
            call.close()
        for resource in resources:
            resource.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules
    Path(artifact).with_name(mode + ".json").write_text(json.dumps(costs, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
