"""Prepared histogram replay without the frontend, and borrowed-owner retirement."""

from contextlib import closing
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
            n, bound = case["size"], case["num_bins"]
            with (
                closing(runtime.ndarray(case["value_dtype"], n)) as values,
                closing(runtime.ndarray(case["counter_dtype"], bound)) as bins,
            ):
                frame = dict(values=values, bins=bins)
                initial = bytes.fromhex(case["initial"])
                bins.write(initial)
                with module.prepare_execution(case["name"], frame) as call:
                    assert bins.read() == initial, "Preparation cleared bins"
                    scratch = call.info["workspace_bytes"]
                    assert (scratch > 0) == (arch == "cpu" and n > 262144)
                    for sample in case["samples"]:
                        source = bytes.fromhex(sample["source"])
                        values.write(source)
                        call()
                        assert bins.read() == bytes.fromhex(sample["expected"])
                        assert values.read() == source
                        assert call.info["workspace_bytes"] == scratch
                reject(call)
        case = cases[-1]
        n, bound = case["size"], case["num_bins"]
        with (
            closing(runtime.ndarray(case["value_dtype"], n)) as values,
            closing(runtime.ndarray(case["counter_dtype"], bound)) as bins,
            runtime.load_module(artifact) as other,
        ):
            frame = dict(values=values, bins=bins)
            first = module.prepare_execution(case["name"], frame)
            second = other.prepare_execution(case["name"], frame)
            source = bytes.fromhex(case["samples"][0]["source"])
            values.write(source)
            first()
            first.close()
            bins.write(bytes.fromhex(case["initial"]))
            second()
            assert bins.read() == bytes.fromhex(case["samples"][0]["expected"])
            second.close()
            with closing(runtime.ndarray("f32", n)) as wrong:
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
        values, bins = runtime.ndarray(case["value_dtype"], n), runtime.ndarray(
            case["counter_dtype"], bound
        )
        values.write(source)
        call = module.prepare_execution(case["name"], dict(values=values, bins=bins))
        call()
        ti.reset()
        reject(call)
        call.close()
        module.close()
        runtime.close()


if __name__ == "__main__":
    main(*sys.argv[1:])
