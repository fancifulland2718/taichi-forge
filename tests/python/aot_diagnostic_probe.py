"""Consume diagnostic scalars with no compiler/frontend and test borrowed retirement."""

from contextlib import ExitStack, closing
import json
import os
from pathlib import Path
import sys

from taichi_forge_runtime.aot import Runtime
from aot_scan_probe import reject


def main(mode, arch, artifact):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False, cpu_max_num_threads=2)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime(
            arch,
            device_index=(
                int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", "0"))
                if arch == "vulkan"
                else 0
            ),
        )
    cases = json.loads(Path(artifact).with_suffix(".json").read_text())
    with runtime, runtime.load_module(artifact) as module:
        for case in cases:
            with ExitStack() as stack:
                values = stack.enter_context(closing(runtime.ndarray(case["dtype"], 7)))
                output = stack.enter_context(
                    closing(runtime.ndarray(case["result_dtype"], 1))
                )
                frame = dict(values=values, output=output)
                if case["other"]:
                    frame["other"] = stack.enter_context(
                        closing(runtime.ndarray(case["dtype"], 7))
                    )
                initial = bytes.fromhex(case["initial"])
                output.write(initial)
                with module.prepare_execution(case["name"], frame) as call:
                    assert output.read() == initial, "Preparation executed diagnostic"
                    assert call.info["workspace_bytes"] == 0
                    for sample in case["samples"]:
                        source = bytes.fromhex(sample["values"])
                        values.write(source)
                        if case["other"]:
                            frame["other"].write(source)
                        call()
                        assert output.read() == bytes.fromhex(sample["expected"]), case[
                            "name"
                        ]
                        assert values.read() == source
                reject(call)
        case = cases[0]
        with (
            closing(runtime.ndarray(case["dtype"], 7)) as values,
            closing(runtime.ndarray("i32", 1)) as output,
            runtime.load_module(artifact) as sibling,
        ):
            frame = dict(values=values, output=output)
            first = module.prepare_execution(case["name"], frame)
            second = sibling.prepare_execution(case["name"], frame)
            source = bytes.fromhex(case["samples"][0]["values"])
            values.write(source)
            first()
            first.close()
            output.write(bytes.fromhex(case["initial"]))
            second()
            assert output.read() == bytes.fromhex(case["samples"][0]["expected"])
            second.close()
            with closing(runtime.ndarray("i32", 7)) as wrong:
                reject(
                    lambda: module.prepare_execution(
                        case["name"], dict(frame, values=wrong)
                    )
                )
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules and "numpy" not in sys.modules
    else:
        runtime = ti.aot.borrow_runtime()
        module = runtime.load_module(artifact)
        values, output = runtime.ndarray(case["dtype"], 7), runtime.ndarray("i32", 1)
        values.write(source)
        call = module.prepare_execution(
            case["name"], dict(values=values, output=output)
        )
        call()
        ti.reset()
        reject(call)
        call.close()
        module.close()
        runtime.close()


if __name__ == "__main__":
    main(*sys.argv[1:])
