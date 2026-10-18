"""Consume fixed indexed-copy plans without producer code or NumPy."""

import array
from contextlib import ExitStack, closing
import json
from pathlib import Path
import sys

from taichi_forge_runtime.aot import Runtime
from aot_scan_probe import CODES, reject


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
            with ExitStack() as stack:
                src = stack.enter_context(
                    closing(
                        runtime.ndarray(
                            case["dtype"],
                            case["src_items"],
                            element_shape=case["element_shape"],
                        )
                    )
                )
                dst = stack.enter_context(
                    closing(
                        runtime.ndarray(
                            case["dtype"],
                            case["dst_items"],
                            element_shape=case["element_shape"],
                        )
                    )
                )
                indices = stack.enter_context(
                    closing(runtime.ndarray("i32", case["size"]))
                )
                frame = dict(src=src, indices=indices, dst=dst)
                source = bytes.fromhex(case["source"])
                sentinel = bytes.fromhex(case["sentinel"])
                src.write(source)
                dst.write(sentinel)
                with module.prepare_execution(case["name"], frame) as call:
                    assert src.read() == source and dst.read() == sentinel
                    scratch = call.info["workspace_bytes"]
                    assert (scratch > 0) == (arch == "cpu" and case["scatter"])
                    for sample in case["samples"]:
                        index_data = bytes.fromhex(sample["indices"])
                        indices.write(index_data)
                        dst.write(sentinel)
                        call()
                        assert dst.read() == bytes.fromhex(sample["expected"])
                        assert src.read() == source and indices.read() == index_data
                        assert call.info["workspace_bytes"] == scratch
                    if arch == "cpu" and case["scatter"]:
                        indices.write(array.array("i", [0] * case["size"]))
                        dst.write(sentinel)
                        reject(call)
                        assert dst.read() == sentinel
                        indices.write(bytes.fromhex(case["samples"][-1]["indices"]))
                        call()
                        assert dst.read() == bytes.fromhex(
                            case["samples"][-1]["expected"]
                        )
                reject(call)
        case = next(c for c in cases if c["dtype"] == "f32" and not c["element_shape"])
        with (
            closing(runtime.ndarray("f32", case["src_items"])) as src,
            closing(runtime.ndarray("f32", case["dst_items"])) as dst,
            closing(runtime.ndarray("i32", case["size"])) as indices,
            runtime.load_module(artifact) as other,
        ):
            frame = dict(src=src, indices=indices, dst=dst)
            first = module.prepare_execution(case["name"], frame)
            second = other.prepare_execution(case["name"], frame)
            src.write(bytes.fromhex(case["source"]))
            indices.write(bytes.fromhex(case["samples"][0]["indices"]))
            first()
            first.close()
            second()
            second.close()
            assert dst.read() == bytes.fromhex(case["samples"][0]["expected"])
            for dtype, size in (
                ("i32", case["src_items"]),
                ("f32", case["src_items"] - 1),
            ):
                with closing(runtime.ndarray(dtype, size)) as wrong:
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
        src = runtime.ndarray("f32", case["src_items"])
        dst = runtime.ndarray("f32", case["dst_items"])
        indices = runtime.ndarray("i32", case["size"])
        indices.write(bytes(case["size"] * 4))
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
