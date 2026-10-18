"""Consume exported consecutive runs without the compiler or producer workspace."""

import array
import json
from pathlib import Path
import sys

from taichi_forge_runtime.aot import Runtime
from aot_scan_probe import CODES, read, reject


def main(mode, arch, artifact):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime(arch)
    cases = json.loads(Path(artifact).with_suffix(".json").read_text(encoding="utf8"))
    with runtime, runtime.load_module(artifact) as module:
        for case in cases:
            dtype, capacity, size = (case[k] for k in ("dtype", "capacity", "size"))
            arrays = dict(
                keys=runtime.ndarray(dtype, capacity),
                output=runtime.ndarray(dtype, capacity + 3),
                lengths=runtime.ndarray("i32", capacity + 2),
                count=runtime.ndarray("i32", 2),
            )
            expected_keys, expected_lengths = [91] * (capacity + 3), [92] * (
                capacity + 2
            )
            arrays["output"].write(array.array(CODES[dtype], expected_keys))
            arrays["lengths"].write(array.array("i", expected_lengths))
            arrays["count"].write(array.array("i", [93, 73]))
            bound = {name: arrays[name] for name in case["arguments"]}
            with module.prepare_execution(case["name"], bound) as call:
                assert read(arrays["count"]) == [93, 73], "Prepare executed RLE"
                assert call.info["resource_bytes"] == (12 * capacity if size else 0)
                for pattern in ("unique", "same", "alternating", "groups"):
                    base = 1 << 40 if dtype in ("i64", "u64") else 100
                    host = [
                        base
                        + (
                            i
                            if pattern == "unique"
                            else (
                                0
                                if pattern == "same"
                                else i % 2 if pattern == "alternating" else i // 3
                            )
                        )
                        for i in range(capacity)
                    ]
                    data = array.array(CODES[dtype], host).tobytes()
                    arrays["keys"].write(data)
                    keys, lengths = [], []
                    for value in host[:size]:
                        if not keys or keys[-1] != value:
                            keys.append(value)
                            lengths.append(1)
                        else:
                            lengths[-1] += 1
                    expected_keys[: len(keys)], expected_lengths[: len(keys)] = (
                        keys,
                        lengths,
                    )
                    call()
                    assert read(arrays["count"]) == [len(keys), 73]
                    assert read(arrays["output"]) == expected_keys
                    assert read(arrays["lengths"]) == expected_lengths
                    assert arrays["keys"].read() == data
            reject(call)
            if size:
                before = arrays["keys"].read()
                reject(
                    lambda: module.prepare_execution(
                        case["name"], dict(bound, output=arrays["keys"])
                    )
                )
                assert arrays["keys"].read() == before
            for value in arrays.values():
                value.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules and "numpy" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
