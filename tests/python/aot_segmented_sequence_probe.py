"""Consume segmented sequence closures without the DSL or NumPy."""

from contextlib import closing
import json
import os
from pathlib import Path
import sys

from taichi_forge_runtime.aot import Runtime
from aot_grouped_reduce_probe import assert_payload
from aot_scan_probe import reject


def main(mode, arch, artifact):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False, cpu_max_num_threads=2)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime(arch, device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", "0")) if arch == "vulkan" else 0)
    cases = json.loads(Path(artifact).with_suffix(".json").read_text())
    with runtime, runtime.load_module(artifact) as module:
        for case in cases:
            dtype = case["dtype"]
            with closing(runtime.ndarray(dtype, case["size"])) as values, closing(runtime.ndarray(dtype, case["segments"])) as output:
                initial = bytes.fromhex(case["initial"])
                output.write(initial)
                args = dict(values=values, output=output)
                with module.prepare_execution(case["name"], args) as call:
                    assert output.read() == initial
                    for sample in case["samples"]:
                        source = bytes.fromhex(sample["source"])
                        values.write(source)
                        call()
                        assert_payload(output.read(), sample["expected"], dtype)
                        assert values.read() == source
                reject(call)
        case = next(c for c in cases if c["dtype"] == "f32" and c["size"] < 32)
        frames = []
        for _ in range(2):
            args = dict(values=runtime.ndarray("f32", case["size"]), output=runtime.ndarray("f32", case["segments"]))
            frames.append((args, module.prepare_execution(case["name"], args)))
        for iteration, sample in enumerate(case["samples"]):
            args, call = frames[iteration % 2]
            args["values"].write(bytes.fromhex(sample["source"]))
            call()
            assert_payload(args["output"].read(), sample["expected"], "f32")
        reject(lambda: module.prepare_execution(case["name"], dict(values=frames[0][0]["values"], output=frames[0][0]["values"])))
        with closing(runtime.ndarray("i32", case["size"])) as wrong:
            reject(lambda: module.prepare_execution(case["name"], dict(values=wrong, output=frames[0][0]["output"])))
        with closing(runtime.ndarray("f32", case["size"] - 1)) as short:
            reject(lambda: module.prepare_execution(case["name"], dict(values=short, output=frames[0][0]["output"])))
        frames[0][1].close()
        frames[1][1]()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        reject(frames[1][1])
        for args, call in frames:
            call.close()
            for value in args.values():
                value.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules and "numpy" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
