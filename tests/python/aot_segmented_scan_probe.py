"""No-DSL consumer for retained scan and selected segmented recipes."""

import array
import json
from pathlib import Path
import sys

from taichi_forge_runtime.aot import ExecutionPlan, Runtime
from tests.python.aot_dot_probe import raises


def normalize(value, dtype):
    value &= 0xFFFFFFFF
    return value - 0x100000000 if dtype == "i32" and value >= 0x80000000 else value


def inputs(count, dtype, iteration):
    return [normalize((index * 0x12435679 + 0x7FFFFFFA) * iteration, dtype) for index in range(count)]


def write(target, values, dtype):
    target.write(array.array("i" if dtype == "i32" else "I", values))


def read(target, dtype):
    result = array.array("i" if dtype == "i32" else "I")
    result.frombytes(target.read())
    return list(result)


def expected(offsets, values, dtype, inclusive):
    result = [73] * len(values)
    for begin, end in zip(offsets, offsets[1:]):
        total = 0
        for i in range(begin, end):
            previous = total
            total = normalize(total + values[i], dtype)
            result[i] = total if inclusive else previous
    return result


def native(runtime, module):
    calls = []
    for dtype in ("i32", "u32"):
        for size in (0, 1, 31, 129, 1025, 4099):
            value = runtime.ndarray(dtype, size + 7)
            write(value, [73] * (size + 7), dtype)
            call = module.prepare_execution(f"{dtype}_{size}", dict(values=value))
            assert read(value, dtype) == [73] * (size + 7)
            for iteration in (1, 0, -1):
                host = inputs(size, dtype, iteration) + [73] * 7
                write(value, host, dtype)
                call()
                assert read(value, dtype) == expected([0, size], host, dtype, True)
            calls.append(call)
    operation = dict(
        kind="native",
        provider="cuda_driver",
        operation="retained_scan",
        version=1,
        parameters=dict(size=31, dtype="u32"),
        bindings=dict(values="v", scratch="s"),
    )
    good = dict(v=runtime.ndarray("u32", 31), s=runtime.ndarray("u32", 1))
    for args in (
        dict(good, v=runtime.ndarray("u32", 30)),
        dict(good, v=runtime.ndarray("i32", 31)),
        dict(good, s=runtime.ndarray("u32", 2)),
        dict(good, s=runtime.ndarray("i32", 1)),
        dict(good, s=good["v"]),
    ):
        raises(lambda: ExecutionPlan([operation]).prepare(module, args))
    alias = dict(kind="ndarray_view", name="s", array="v", offset=3, shape=[1])
    raises(lambda: ExecutionPlan([operation], resources=[alias]).prepare(module, dict(v=good["v"])))
    with ExecutionPlan([operation]).prepare(module, good) as call:
        write(good["v"], [1] * 31, "u32")
        call()
        assert read(good["v"], "u32") == list(range(1, 32))
    return calls


def segmented(runtime, module, artifact):
    info = json.loads(Path(artifact).with_suffix(".json").read_text(encoding="utf8"))
    dtype, capacity = info["dtype"], info["capacity"]
    calls = []
    for strategy in info["strategies"]:
        frames = []
        for _ in range(2):
            args = {name: runtime.ndarray(dtype, capacity) for name in ("input", "result")}
            write(args["result"], [73] * capacity, dtype)
            call = module.prepare_execution(strategy, args)
            assert read(args["result"], dtype) == [73] * capacity
            frames.append((args, call))
            calls.append(call)
        for iteration in (1, 0, -1):
            for index, (args, call) in enumerate(frames):
                host = inputs(capacity, dtype, iteration + index)
                write(args["input"], host, dtype)
                write(args["result"], [73] * capacity, dtype)
                call()
                assert read(args["result"], dtype) == expected(info["offsets"], host, dtype, info["inclusive"])
        raises(
            lambda: module.prepare_execution(strategy, dict(input=frames[0][0]["input"], result=frames[0][0]["input"]))
        )
        frames[0][1].close()
        frames[1][1]()
    return calls


def main(mode, artifact, kind):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    with runtime:
        assert runtime.probe_native_operation("cuda_retained_scan")["available"]
        module = runtime.load_module(artifact)
        calls = native(runtime, module) if kind == "native" else segmented(runtime, module, artifact)
        calls[-1]()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        raises(calls[-1])
        for call in calls:
            call.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
