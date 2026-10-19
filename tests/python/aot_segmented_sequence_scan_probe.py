"""Runtime-only and borrowed scan consumers, including rebind and retirement."""

from contextlib import ExitStack, closing
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

        ti.init(arch=getattr(ti, arch), offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime(arch, device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", "0")) if arch == "vulkan" else 0)
    cases = json.loads(Path(artifact).with_suffix(".json").read_text())
    with runtime, runtime.load_module(artifact) as module:
        for case in cases:
            with ExitStack() as stack:
                values = stack.enter_context(closing(runtime.ndarray(case["dtype"], case["size"])))
                output = values if case["inplace"] else stack.enter_context(closing(runtime.ndarray(case["dtype"], case["size"])))
                args = dict(values=values) if case["inplace"] else dict(values=values, output=output)
                for name in (case["name"], case["name"] + "_warm"):
                    with module.prepare_execution(name, args) as call:
                        for sample in case["samples"]:
                            source = bytes.fromhex(sample["source"])
                            values.write(source)
                            if not case["inplace"]:
                                output.write(bytes.fromhex(sample["initial"]))
                            call()
                            actual = output.read()
                            assert_payload(actual, sample["expected"], case["dtype"])
                            width = int(case["dtype"][1:]) // 8
                            assert actual[:width] == bytes.fromhex(sample["expected"])[:width]
                            if not case["inplace"]:
                                assert values.read() == source
                    reject(call)
        case = next(c for c in cases if c["dtype"] == "f32")
        size = case["size"]
        frames = []
        for _ in range(2):
            args = dict(values=runtime.ndarray("f32", size), output=runtime.ndarray("f32", size))
            frames.append((args, module.prepare_execution(case["name"], args)))
        for frame, sample in zip(frames, case["samples"]):
            args, call = frame
            args["values"].write(bytes.fromhex(sample["source"]))
            args["output"].write(bytes.fromhex(sample["initial"]))
            call()
        for frame, sample in zip(frames, case["samples"]):
            assert_payload(frame[0]["output"].read(), sample["expected"], "f32")
        with closing(runtime.ndarray("f32", size + 1)) as parent:
            plan = module.execution_plan(case["name"])
            views = [dict(kind="ndarray_view", name=name, array="parent", offset=offset, shape=[size])
                     for name, offset in (("values", 0), ("output", 1))]
            reject(lambda: module.prepare_execution(plan.with_argument_resources(views), dict(parent=parent)))
            views[0]["offset"] = 1
            sample = case["samples"][0]
            prefix = bytes.fromhex("0000803f")
            parent.write(prefix + bytes.fromhex(sample["source"]))
            with module.prepare_execution(plan.with_argument_resources(views), dict(parent=parent)) as call:
                call()
                # Serial scan leaves the padding from the aliased input.
                expected = bytearray.fromhex(sample["expected"])
                expected[269 * 4:] = bytes.fromhex(sample["source"])[269 * 4:]
                actual = parent.read()
                assert actual[:4] == prefix
                assert_payload(actual[4:], expected.hex(), "f32")
        with closing(runtime.ndarray("i32", size)) as wrong:
            reject(lambda: module.prepare_execution(case["name"], dict(values=wrong, output=frames[0][0]["output"])))
        with closing(runtime.ndarray("f32", size - 1)) as short:
            reject(lambda: module.prepare_execution(case["name"], dict(values=short, output=frames[0][0]["output"])))
        frames[0][1]()
        frames[0][1].close()  # Retire an in-flight execution without invalidating the other owner.
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
