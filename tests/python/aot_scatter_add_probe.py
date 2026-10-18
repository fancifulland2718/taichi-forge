"""Frontend-free fixed accumulation, retained partials and owner retirement."""

import array
from contextlib import closing
import json
import math
import os
from pathlib import Path
import sys

from taichi_forge_runtime.aot import Runtime
from aot_scan_probe import CODES, reject


def assert_payload(actual, expected, dtype):
    expected = bytes.fromhex(expected)
    if not dtype.startswith("f"):
        assert actual == expected
        return
    left, right = array.array(CODES[dtype]), array.array(CODES[dtype])
    left.frombytes(actual)
    right.frombytes(expected)
    assert len(left) == len(right)
    assert all(a == b or (math.isnan(a) and math.isnan(b)) for a, b in zip(left, right))


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
            dtype, n, bound = case["dtype"], case["size"], case["dst_items"]
            with (
                closing(runtime.ndarray(dtype, n)) as src,
                closing(runtime.ndarray("i32", n)) as indices,
                closing(runtime.ndarray(dtype, bound)) as dst,
            ):
                frame = dict(src=src, indices=indices, dst=dst)
                initial = bytes.fromhex(case["initial"])
                dst.write(initial)
                with module.prepare_execution(case["name"], frame) as call:
                    assert dst.read() == initial, "Preparation mutated destination"
                    scratch = call.info["workspace_bytes"]
                    assert (scratch > 0) == (arch == "cpu" and n > 262144)
                    for sample in case["samples"]:
                        source, index = bytes.fromhex(sample["source"]), bytes.fromhex(
                            sample["indices"]
                        )
                        src.write(source)
                        indices.write(index)
                        dst.write(initial)
                        call()
                        assert_payload(dst.read(), sample["expected"], dtype)
                        assert src.read() == source and indices.read() == index
                        assert call.info["workspace_bytes"] == scratch
                reject(call)
        case = next(c for c in cases if c["dtype"] == "f32")
        n, bound = case["size"], case["dst_items"]
        with (
            closing(runtime.ndarray("f32", n)) as src,
            closing(runtime.ndarray("i32", n)) as indices,
            closing(runtime.ndarray("f32", bound)) as dst,
            runtime.load_module(artifact) as other,
        ):
            frame = dict(src=src, indices=indices, dst=dst)
            first = module.prepare_execution(case["name"], frame)
            second = other.prepare_execution(case["name"], frame)
            sample = case["samples"][0]
            src.write(bytes.fromhex(sample["source"]))
            indices.write(bytes.fromhex(sample["indices"]))
            dst.write(bytes.fromhex(case["initial"]))
            first()
            first.close()
            dst.write(bytes.fromhex(case["initial"]))
            second()
            assert_payload(dst.read(), sample["expected"], "f32")
            second.close()
            for dtype, count in (("i32", n), ("f32", n - 1)):
                with closing(runtime.ndarray(dtype, count)) as wrong:
                    reject(
                        lambda: module.prepare_execution(
                            case["name"], dict(frame, src=wrong)
                        )
                    )
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules and "numpy" not in sys.modules
    else:
        runtime = ti.aot.borrow_runtime()
        module = runtime.load_module(artifact)
        src, indices, dst = (
            runtime.ndarray("f32", n),
            runtime.ndarray("i32", n),
            runtime.ndarray("f32", bound),
        )
        src.write(bytes(n * 4))
        indices.write(bytes(n * 4))
        call = module.prepare_execution(
            case["name"], dict(src=src, indices=indices, dst=dst)
        )
        call()
        ti.reset()
        reject(call)
        call.close()
        module.close()
        runtime.close()


if __name__ == "__main__":
    main(*sys.argv[1:])
