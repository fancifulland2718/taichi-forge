"""Stable native sorting contracts, exercised inside the source-free consumer."""

import array
import math

from aot_consumer_probe import raises


def check_sort(runtime, module):
    from taichi_forge_runtime import aot

    types = {"i32": "i", "u32": "I", "f32": "f", "i64": "q", "u64": "Q", "f64": "d"}

    def write(value, data):
        value.write(array.array(types[value.dtype], data))

    def read(value):
        result = array.array(types[value.dtype])
        result.frombytes(value.read())
        return result.tolist()

    plan = module.execution_plan("stable_order")
    for dtype, code in types.items():
        if dtype[0] == "f":
            initial = [math.nan, -0.0, 3.5, math.inf, -math.inf, 0.0, -7.0, -math.nan, 3.5] * 3
        else:
            base = (1 << 51) if dtype.endswith("64") else 0
            initial = [base + (i * 17 % 11) - (5 if dtype[0] == "i" else 0) for i in range(39)]
            if dtype == "i64":
                initial[:5] = [-(1 << 63), (1 << 63) - 1, -(1 << 60), 0, -(1 << 63)]
            elif dtype == "u64":
                initial[:5] = [(1 << 64) - 1, 1 << 63, 0, (1 << 64) - 1, 0]
        n = len(initial)
        bindings = {"keys": runtime.ndarray(dtype, n), "values": runtime.ndarray("i32", n, element_shape=(2,))}
        write(bindings["keys"], initial)
        with plan.prepare(module, bindings) as prepared:
            assert bindings["keys"].read() == array.array(code, initial).tobytes()
            workspace = prepared.info["workspace_bytes"]
            for data in (initial, initial[::-1]):
                write(bindings["keys"], data)
                write(bindings["values"], [item for i in range(n) for item in (i, -i - 1)])
                prepared()
                order = sorted(range(n), key=lambda i: (math.isnan(data[i]), 0 if math.isnan(data[i]) else data[i]))
                assert read(bindings["values"]) == [item for i in order for item in (i, -i - 1)]
                raw = array.array(code, data).tobytes()
                width = bindings["keys"].nbytes // n
                assert bindings["keys"].read() == b"".join(raw[i * width : (i + 1) * width] for i in order)
                assert prepared.info["workspace_bytes"] == workspace
        for value in bindings.values():
            value.close()

    keys_only = aot.ExecutionPlan([{"kind": "sort", "keys": "keys"}])
    for n in (0, 1, 65):
        keys = runtime.ndarray("i32", n)
        data = list(range(n))[::-1]
        write(keys, data)
        with keys_only.prepare(module, {"keys": keys}) as prepared:
            assert read(keys) == data
            if n <= 1:
                assert prepared.info["workspace_bytes"] == 0
            prepared()
            assert read(keys) == sorted(data)
        keys.close()

    # A producer copies input keys and creates indices, sort carries the
    # permutation, and the exported gather/backward use those integer indices.
    mixed = aot.ExecutionPlan(
        [
            {"kind": "copy", "source": "x", "destination": "keys"},
            {"kind": "kernel", "entry": "indices", "arguments": ["selected"]},
            {"kind": "sort", "keys": "keys", "values": "selected"},
            {"kind": "fill32", "destination": "dx", "value": 0},
            {"kind": "kernel", "entry": "evaluate", "arguments": ["x", "selected", "count", "y"]},
            {"kind": "kernel", "entry": "backward", "arguments": ["x", "selected", "count", "y"]},
        ]
    )
    frames = []
    for n in (17, 4097):
        bindings = {name: runtime.ndarray("f32", n) for name in ("x", "keys", "y", "dx", "dy")}
        bindings.update(selected=runtime.ndarray("i32", n), count=runtime.ndarray("i32", 1))
        write(bindings["count"], [n])
        prepared = mixed.prepare(
            module,
            {name: value for name, value in bindings.items() if name != "dy"},
            gradients={"x": bindings["dx"], "y": bindings["dy"]},
        )
        frames.append((n, bindings, prepared))
    for iteration in range(2):
        for n, bindings, prepared in frames:
            values = [float((i * 17 + iteration) % 31 - 9) for i in range(n)]
            seeds = [float(i % 7 + 1) for i in range(n)]
            write(bindings["x"], values)
            write(bindings["dy"], seeds)
            prepared()
            order = sorted(range(n), key=lambda i: values[i])
            assert read(bindings["keys"]) == sorted(values)
            assert read(bindings["selected"]) == order
            assert read(bindings["y"]) == [values[i] ** 2 for i in order]
            gradient = [0] * n
            for rank, i in enumerate(order):
                gradient[i] = 2 * values[i] * seeds[rank]
            assert read(bindings["dx"]) == gradient

    first = frames[0][2]
    first()
    frames[0][1]["keys"].close()  # This dependency occurs only in native nodes.
    raises(aot.AotError, first)
    first.close()
    frames[1][2]()
    frames[1][2].close()  # Must retire its independent scratch while in flight.

    good = runtime.ndarray("i32", 9)
    write(good, list(range(9)))
    invalid = aot.ExecutionPlan(
        [
            {"kind": "fill32", "destination": "keys", "value": 0},
            {"kind": "sort", "keys": "keys", "values": "values"},
        ]
    )
    for payload in (good, runtime.ndarray("i32", 8), runtime.ndarray("i32", (3, 3))):
        raises(aot.AotError, lambda: invalid.prepare(module, {"keys": good, "values": payload}))
        assert read(good) == list(range(9))
    with keys_only.prepare(module, {"keys": good}) as recovered:
        recovered()
    for _, bindings, _ in frames:
        for value in bindings.values():
            value.close()


if __name__ == "__main__":
    import sys

    from aot_consumer_probe import NoFrontend

    sys.meta_path.insert(0, NoFrontend())
    from taichi_forge_runtime import aot

    with aot.Runtime(sys.argv[1]) as runtime:
        with runtime.load_module(sys.argv[2]) as module:
            check_sort(runtime, module)
