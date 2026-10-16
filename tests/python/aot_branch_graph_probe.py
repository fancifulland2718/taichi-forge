"""Consume automatically exported Graph branches after producer teardown."""

import json
import struct
import sys
import time

from taichi_forge_runtime import aot


def frame(runtime, module, name, scale=1, timings=None):
    start = time.perf_counter()
    arrays = {n: runtime.ndarray("f32", 17) for n in ("x", "y", "dx", "dy", "gradient")}
    arrays["choice"] = runtime.ndarray("i32", 1)
    arrays["state"] = runtime.ndarray("i32", 2)
    arrays["x"].write(struct.pack("<17f", *[scale * (i - 8) for i in range(17)]))
    arrays["y"].write(struct.pack("<17f", *([71] * 17)))
    arrays["choice"].write(struct.pack("<i", 0))
    bound = time.perf_counter()
    prepared = module.prepare_execution(name, arrays)
    if timings is not None:
        timings.update(binding_ms=(bound - start) * 1000, prepare_ms=(time.perf_counter() - bound) * 1000)
    assert struct.unpack("<17f", arrays["y"].read()) == (71,) * 17
    return arrays, prepared


def expect(arrays, name, choice, scale=1):
    name = name.removesuffix("_again")
    fallback = name in ("if_else", "switch_default")
    if name.startswith("if"):
        gain = 1 if choice else 2 if fallback else 0
    else:
        gain = choice + 1 if 0 <= choice < 8 else 9 if fallback else 0
    if name == "nested" and choice == 0:
        gain = 0
    values = [scale * (i - 8) for i in range(17)]
    expected = [gain * v * v for v in values] if gain else [-91] * 17
    assert list(struct.unpack("<17f", arrays["y"].read())) == expected
    assert list(struct.unpack("<17f", arrays["gradient"].read())) == [2 * gain * v for v in values]
    checks = 2 if name == "nested" and choice else 1
    assert struct.unpack("<2i", arrays["state"].read()) == (checks, -77 if gain else choice)


def check(runtime, arch, path):
    start = time.perf_counter()
    module = runtime.load_module(path)
    stats = dict(load_ms=(time.perf_counter() - start) * 1000)
    names = ["if", "if_else", "switch", "switch_default"] + (["nested"] if arch == "cpu" else [])
    for name in names:
        costs = {}
        arrays, prepared = frame(runtime, module, name, timings=costs)
        arrays["choice"].write(struct.pack("<i", 2))
        start = time.perf_counter()
        prepared()
        runtime.wait()
        costs["first_complete_ms"] = (time.perf_counter() - start) * 1000
        expect(arrays, name, 2)
        for choice in (-2147483648, -1, 0, 1, 2, 7, 8, 2147483647):
            arrays["choice"].write(struct.pack("<i", choice))
            prepared()
            expect(arrays, name, choice)
        if arch == "cuda":
            assert prepared.info["execution_kind"] == "cuda_graph"
            assert prepared.info["workspace_bytes"] == 24
        arrays["choice"].write(struct.pack("<i", 2))
        start = time.perf_counter()
        for _ in range(32):
            prepared()
        submitted = time.perf_counter()
        runtime.wait()
        complete = time.perf_counter()
        expect(arrays, name, 2)
        stats[name] = dict(
            **costs,
            host_us=(submitted - start) * 1e6 / 32,
            complete_us=(complete - start) * 1e6 / 32,
            workspace_bytes=prepared.info["workspace_bytes"],
            argument_bytes=prepared.info["argument_bytes"],
        )
        prepared.close()
        for value in arrays.values():
            value.close()
    first, one = frame(runtime, module, "switch")
    second, two = frame(runtime, module, "switch_again", scale=3)
    first["choice"].write(struct.pack("<i", 2))
    second["choice"].write(struct.pack("<i", 7))
    one()
    two()
    expect(first, "switch", 2)
    expect(second, "switch_again", 7, 3)
    first["choice"].close()
    try:
        one()
    except (RuntimeError, aot.AotError):
        pass
    else:
        raise AssertionError("Retired branch input reused")
    one.close()
    two()
    expect(second, "switch_again", 7, 3)
    module.close()
    two.close()
    for arrays in (first, second):
        for value in arrays.values():
            value.close()
    return stats


if __name__ == "__main__":
    mode, arch, path = sys.argv[1:]
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False)
        with ti.aot.borrow_runtime() as runtime:
            stats = check(runtime, arch, path)
            module = runtime.load_module(path)
            arrays, prepared = frame(runtime, module, "if_else")
            prepared()
            ti.reset()
            try:
                prepared()
            except (RuntimeError, aot.AotError):
                pass
            else:
                raise AssertionError("Owner reset did not retire exported branches")
            prepared.close()
    else:
        with aot.Runtime(arch) as runtime:
            stats = check(runtime, arch, path)
        assert "taichi_forge" not in sys.modules
    print(json.dumps(stats))
