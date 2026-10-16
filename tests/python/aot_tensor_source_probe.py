"""Consume retained tensor bindings without producer code or data."""

import json
import math
import statistics
import struct
import sys
import time


def main(mode, arch, artifact, element_shape):
    from taichi_forge_runtime.aot import Runtime

    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False, enable_fallback=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime(arch)
    width = math.prod(element_shape)
    with runtime:
        start = time.perf_counter()
        module = runtime.load_module(artifact)
        costs = dict(load_ms=(time.perf_counter() - start) * 1000, prepare_ms=[])
        frames = []
        for name in ("step", "again"):
            arrays = {n: runtime.ndarray("f32", 3, element_shape=element_shape) for n in ("x", "dx")}
            arrays.update({n: runtime.ndarray("f32", 3) for n in ("y", "dy")})
            arrays["y"].write(struct.pack("<3f", -99, -99, -99))
            start = time.perf_counter()
            call = module.prepare_execution(name, arrays)
            costs["prepare_ms"].append((time.perf_counter() - start) * 1000)
            costs["workspace_bytes"] = call.info["workspace_bytes"]
            costs["argument_bytes"] = call.info["argument_bytes"]
            assert struct.unpack("<3f", arrays["y"].read()) == (-99,) * 3
            frames.append((arrays, call))
        for scale in (1, -2, 0):
            for index, (arrays, call) in enumerate(frames):
                values = [(i + 1) * scale * (index + 1) / 4 for i in range(3 * width)]
                arrays["x"].write(struct.pack(f"<{len(values)}f", *values))
                call()
                expected = [sum(x * x for x in values[i * width : (i + 1) * width]) for i in range(3)]
                assert list(struct.unpack("<3f", arrays["y"].read())) == expected
                assert list(struct.unpack(f"<{len(values)}f", arrays["dx"].read())) == [2 * x for x in values]
        wrong = runtime.ndarray("f32", 3, element_shape=(width, 1))
        try:
            module.prepare_execution("step", dict(frames[0][0], x=wrong))
        except (RuntimeError, ValueError):
            pass
        else:
            raise AssertionError("Same-sized but differently shaped tensor was accepted")
        frames[0][1].close()
        samples = []
        for _ in range(8):
            start = time.perf_counter()
            frames[1][1]()
            runtime.wait()
            samples.append((time.perf_counter() - start) * 1e6)
        costs["warm_complete_us"] = statistics.median(samples)
        frames[1][1]()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        try:
            frames[1][1]()
        except RuntimeError:
            pass
        else:
            raise AssertionError("Retired tensor plan remained callable")
        frames[1][1].close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules
    print(json.dumps(costs))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], tuple(map(int, sys.argv[4:])))
