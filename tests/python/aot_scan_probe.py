"""Consume ordinary scan recordings without producer objects or DSL imports."""

import array
from contextlib import closing
import json
from pathlib import Path
import sys

from taichi_forge_runtime.aot import Runtime

CODES = dict(i32="i", u32="I", i64="q", u64="Q", f32="f", f64="d")


def normalize(value, dtype):
    if dtype.startswith("f"):
        return value
    width = int(dtype[1:])
    value = int(value) % (1 << width)
    if dtype.startswith("i") and value >= 1 << (width - 1):
        value -= 1 << width
    return value


def values_for(dtype, capacity, iteration):
    if dtype.startswith("f"):
        return [(i % 13 - 6 + iteration) / 4 for i in range(capacity)]
    return [
        normalize((1 << (int(dtype[1:]) - 1)) - 3 + i * (iteration + 1), dtype)
        for i in range(capacity)
    ]


def expected(values, size, dtype):
    result = values.copy()
    total = 0
    for i in range(size):
        total = normalize(total + values[i], dtype)
        result[i] = total
    return result


def read(value):
    host = array.array(CODES[value.dtype])
    host.frombytes(value.read())
    return list(host)


def reject(call):
    try:
        call()
    except (ValueError, RuntimeError):
        return
    raise AssertionError("Expected an invalid binding or retired execution")


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
            name, dtype, size, capacity = (
                case[k] for k in ("name", "dtype", "size", "capacity")
            )
            with closing(runtime.ndarray(dtype, capacity)) as values:
                host = values_for(dtype, capacity, 0)
                values.write(array.array(CODES[dtype], host))
                with module.prepare_execution(name, dict(values=values)) as call:
                    assert read(values) == host, "Preparation executed the scan"
                    for iteration in (0, 1, -1):
                        host = values_for(dtype, capacity, iteration)
                        values.write(array.array(CODES[dtype], host))
                        call()
                        assert read(values) == expected(host, size, dtype), (
                            arch,
                            dtype,
                            size,
                        )
                reject(call)
        case = next(c for c in cases if c["dtype"] == "f32")
        n, name = case["size"], case["name"]
        with (
            closing(runtime.ndarray("f32", n)) as values,
            runtime.load_module(artifact) as other,
        ):
            # Independent module/prepare owns its own scratch; retiring one
            # submitted frame cannot invalidate another recording.
            first = module.prepare_execution(name, dict(values=values))
            second = other.prepare_execution(name, dict(values=values))
            values.write(array.array("f", [1] * n))
            first()
            first.close()
            second()
            assert read(values) == [((i + 1) * (i + 2)) / 2 for i in range(n)]
            for dtype, count in (("i32", n), ("f32", n - 1)):
                with closing(runtime.ndarray(dtype, count)) as wrong:
                    reject(lambda: module.prepare_execution(name, dict(values=wrong)))
            values.write(array.array("f", [0] * n))
            second()
            second.close()
            assert read(values) == [0] * n
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules and "numpy" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
