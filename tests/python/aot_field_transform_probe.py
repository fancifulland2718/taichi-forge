"""Source-free packed Field roots, explicit suffix AD, prefix and retirement."""

import array
import json
import math
import os
from pathlib import Path
import sys

from taichi_forge_runtime.aot import Runtime
from aot_scan_probe import reject
from aot_transform_probe import check


def main(mode, arch, artifact):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False, cpu_max_num_threads=4)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime(
            arch, device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", "0")) if arch == "vulkan" else 0
        )
    case = json.loads(Path(artifact).with_suffix(".json").read_text())
    dtype, n = case["dtype"], case["size"] * math.prod(case["shape"])
    code = "f" if dtype == "f32" else "I"

    def packed(values):
        return array.array(code, values).tobytes()

    def transform(values):
        return [3 * x + 7 if dtype == "f32" else (3 * x + 7) & 0xFFFFFFFF for x in values]

    with runtime:
        modules = [runtime.load_module(artifact), runtime.load_module(artifact)]
        frames = [{name: runtime.ndarray(dtype, n) for name in ("input", "output")} for _ in modules]
        if dtype == "f32":
            for frame in frames:
                frame.update({name: runtime.ndarray("f32", n) for name in ("dx", "dy", "energy")})
        calls = [module.prepare_execution("step", frame) for module, frame in zip(modules, frames)]
        for module, frame in zip(modules, frames):
            with module.prepare_execution("read", dict(output=frame["output"])) as read:
                read()
                assert frame["output"].read() == packed([37] * n)
        for iteration in (1, 0, 2):
            expected = []
            for index, (frame, call) in enumerate(zip(frames, calls)):
                values = [iteration * (i % 13) + index for i in range(n)]
                if dtype == "u32" and iteration:
                    values[0] = 0xFFFFFFFF
                result = transform(values)
                expected.append(result)
                frame["input"].write(packed(values))
                call()
                check(frame["output"].read(), packed(result), dtype)
                assert frame["input"].read() == packed(values)
                if dtype == "f32":
                    check(frame["dx"].read(), packed(result), dtype)
                    check(frame["energy"].read(), packed([0.5 * x * x for x in result]), dtype)
            # Module 1's update must not alter module 0's root or alias view.
            with modules[0].prepare_execution("read", dict(output=frames[0]["output"])) as read:
                read()
                check(frames[0]["output"].read(), packed(expected[0]), dtype)
        prefix_count = 2 * math.prod(case["shape"])
        values = list(range(n))
        frames[0]["input"].write(packed(values))
        with modules[0].prepare_execution("prefix", {k: frames[0][k] for k in ("input", "output")}) as prefix:
            prefix()
            tail = values if case["inplace"] else expected[0]
            check(frames[0]["output"].read(), packed(transform(values[:prefix_count]) + tail[prefix_count:]), dtype)
        calls[0]()
        ticket = calls[0].submit()
        modules[0].close()
        ticket.wait()
        reject(calls[0])
        calls[1]()
        check(frames[1]["output"].read(), packed(expected[1]), dtype)
        modules[1].close()
        for call in calls:
            call.close()
        for frame in frames:
            for value in frame.values():
                value.close()
    print("Packed Field transform, downstream AD and owner contracts passed", mode, arch)


if __name__ == "__main__":
    main(*sys.argv[1:])
