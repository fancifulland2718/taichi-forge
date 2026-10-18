"""Consume native affine transform plans without producer code or NumPy."""

import array
from contextlib import ExitStack, closing
import json
import math
from pathlib import Path
import sys

from taichi_forge_runtime.aot import Runtime
from aot_scan_probe import CODES, reject


def check(actual, expected, dtype):
    assert len(actual) == len(expected)
    if not dtype.startswith("f"):
        assert actual == expected
        return
    left, right = array.array(CODES[dtype]), array.array(CODES[dtype])
    left.frombytes(actual)
    right.frombytes(expected)
    for a, b in zip(left, right):
        if math.isnan(b):
            assert math.isnan(a)
        else:
            assert a == b
            if b == 0:
                assert math.copysign(1, a) == math.copysign(1, b)


def main(mode, arch, artifact):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False, cpu_max_num_threads=2)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime(arch)
    cases = json.loads(Path(artifact).with_suffix(".json").read_text())
    with runtime, runtime.load_module(artifact) as module:
        for case in cases:
            dtype, n = case["dtype"], case["size"]
            with ExitStack() as stack:
                src = stack.enter_context(closing(runtime.ndarray(dtype, n)))
                dst = (
                    src
                    if case["inplace"]
                    else stack.enter_context(closing(runtime.ndarray(dtype, n)))
                )
                frame = dict(src=src) if case["inplace"] else dict(src=src, dst=dst)
                sentinel = array.array(CODES[dtype], [37] * n).tobytes()
                src.write(sentinel)
                dst.write(sentinel)
                with module.prepare_execution(case["name"], frame) as call:
                    assert src.read() == sentinel and dst.read() == sentinel
                    assert call.info["workspace_bytes"] == 0
                    for sample in case["samples"]:
                        data = bytes.fromhex(sample["input"])
                        src.write(data)
                        call()
                        check(dst.read(), bytes.fromhex(sample["expected"]), dtype)
                        if not case["inplace"]:
                            assert src.read() == data
                reject(call)
        case = next(c for c in cases if c["dtype"] == "f32" and not c["inplace"])
        with (
            closing(runtime.ndarray("f32", case["size"])) as src,
            closing(runtime.ndarray("f32", case["size"])) as dst,
        ):
            frame = dict(src=src, dst=dst)
            sample = case["samples"][0]
            with runtime.load_module(artifact) as other:
                first = module.prepare_execution(case["name"], frame)
                second = other.prepare_execution(case["name"], frame)
                src.write(bytes.fromhex(sample["input"]))
                first()
                first.close()
                second()
                second.close()
                check(dst.read(), bytes.fromhex(sample["expected"]), "f32")
            for dtype, n in (("i32", case["size"]), ("f32", case["size"] - 1)):
                with closing(runtime.ndarray(dtype, n)) as wrong:
                    reject(
                        lambda: module.prepare_execution(
                            case["name"], dict(src=wrong, dst=dst)
                        )
                    )
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules and "numpy" not in sys.modules
    else:
        runtime = ti.aot.borrow_runtime()
        module = runtime.load_module(artifact)
        src, dst = runtime.ndarray("f32", case["size"]), runtime.ndarray(
            "f32", case["size"]
        )
        call = module.prepare_execution(case["name"], dict(src=src, dst=dst))
        call()
        ti.reset()
        reject(call)
        call.close()
        module.close()
        runtime.close()


if __name__ == "__main__":
    main(*sys.argv[1:])
