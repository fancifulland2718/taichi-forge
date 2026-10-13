"""Small direct/staged helper execution after producer retirement."""

import struct
import sys

from taichi_forge_runtime.aot import Runtime
from tests.python.aot_binding_constraints_probe import failure
from tests.python.aot_cutensor_probe import near


def main(mode, dtype, count, artifact):
    count = int(count)
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    code = "i" if dtype == "i32" else "f"

    def write(array, values):
        array.write(struct.pack("<" + code * len(values), *values))

    def read(array):
        return struct.unpack("<" + code * (array.nbytes // 4), array.read())

    with runtime:
        module = runtime.load_module(artifact)
        frames, calls = [], []
        for name in ("direct", "staged"):
            for suffix in ("", "_again"):
                frame = dict(values=runtime.ndarray(dtype, (count,)), output=runtime.ndarray(dtype, (1,)))
                if dtype == "f32":
                    frame["dx"] = runtime.ndarray(dtype, (count,))
                frames.append(frame)
                calls.append(module.prepare_execution(name + suffix, frame))
        assert calls[2].info["resource_bytes"] > calls[0].info["resource_bytes"]
        for step in range(3):
            for index, (frame, call) in enumerate(zip(frames, calls)):
                values = [(i % 23) - 11 + step + index for i in range(count)]
                if dtype == "f32":
                    values = [v / 8 for v in values]
                write(frame["values"], values)
                write(frame["output"], [123])
                call()
                near(read(frame["output"]), [sum(v * v for v in values) if dtype == "f32" else sum(values)])
                if dtype == "f32":
                    near(read(frame["dx"]), [2 * v for v in values])
        if dtype == "i32":
            frame = frames[0]
            short = runtime.ndarray(dtype, (count - 1,))
            empty = runtime.ndarray(dtype, (0,))
            for name in ("direct", "staged"):
                failure(lambda: module.prepare_execution(name, dict(frame, values=short)), "smaller")
                failure(lambda: module.prepare_execution(name, dict(frame, output=empty)), "smaller")
                failure(lambda: module.prepare_execution(name, dict(frame, output=frame["values"])), "disjoint")
            # Physical import alias checks are exercised by the shared three-backend fixture.
            short.close()
            empty.close()
        calls[0].close()
        calls[1]()
        calls[2]()
        calls[3]()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        for call in calls[1:]:
            failure(call)
            call.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
