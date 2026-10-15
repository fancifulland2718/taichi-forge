"""Consume a named native sort closure with no compiler or producer objects."""

import array
import math
import sys

from taichi_forge_runtime.aot import Runtime


def write(value, data):
    value.write(array.array("i" if value.dtype == "i32" else "f", data))


def read(value):
    result = array.array("i" if value.dtype == "i32" else "f")
    result.frombytes(value.read())
    return result.tolist()


def failure(call):
    try:
        call()
    except (ValueError, RuntimeError):
        return
    raise AssertionError("Expected a rejected binding or retired owner")


def main(mode, arch, artifact):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime(arch)
    n = 17
    with runtime:
        module = runtime.load_module(artifact)
        pairs = []
        for _ in range(2):
            frame = {
                name: runtime.ndarray("f32", n) for name in ("source_keys", "keys", "source", "output", "gradient")
            }
            frame["order"] = runtime.ndarray("i32", n)
            write(frame["output"], [-99] * n)
            call = module.prepare_execution("step", frame)
            assert read(frame["output"]) == [-99] * n
            pairs.append((frame, call))
        for step in range(3):
            for index, (frame, call) in enumerate(pairs):
                keys = [math.nan, 0.0, -0.0, 3, -7, 3, math.inf, -math.inf, math.nan] + list(range(8))
                if step % 2:
                    keys.reverse()
                source = [(i - 4 + index + step) / 4 for i in range(n)]
                write(frame["source_keys"], keys)
                write(frame["source"], source)
                call()
                order = sorted(range(n), key=lambda i: (math.isnan(keys[i]), 0 if math.isnan(keys[i]) else keys[i]))
                assert read(frame["order"]) == order
                assert read(frame["output"]) == [(i + 1) * source[j] ** 2 for i, j in enumerate(order)]
                gradient = [0.0] * n
                for i, j in enumerate(order):
                    gradient[j] = 2 * (i + 1) * source[j]
                assert read(frame["gradient"]) == gradient
                raw = array.array("f", keys).tobytes()
                assert frame["keys"].read() == b"".join(raw[j * 4 : (j + 1) * 4] for j in order)
        frame = pairs[0][0]
        write(frame["keys"], keys)
        write(frame["order"], range(n))
        original = frame["keys"].read()
        with module.prepare_execution("roundtrip", dict(keys=frame["keys"], order=frame["order"])) as roundtrip:
            roundtrip()
            assert frame["keys"].read() == original
            assert read(frame["order"]) == list(range(n))
        for wrong in (runtime.ndarray("i32", n), runtime.ndarray("f32", n + 1), runtime.ndarray("f32", n - 1)):
            failure(lambda: module.prepare_execution("sort", dict(keys=wrong, order=frame["order"])))
            wrong.close()
        failure(lambda: module.prepare_execution("sort", dict(keys=frame["keys"], order=frame["keys"])))
        pairs[0][1]()
        pairs[0][1].close()
        failure(pairs[0][1])
        pairs[1][1]()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        failure(pairs[1][1])
        pairs[1][1].close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
