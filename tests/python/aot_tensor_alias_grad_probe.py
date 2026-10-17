"""Run the tensor alias regression without the producer or NumPy."""

import json
import statistics
import struct
import sys
import time

from taichi_forge_runtime.aot import Runtime


def main(arch, artifact):
    with Runtime(arch) as runtime:
        start = time.perf_counter()
        module = runtime.load_module(artifact)
        costs = dict(load_ms=(time.perf_counter() - start) * 1000)
        arrays = {name: runtime.ndarray("f32", 4, element_shape=(4,)) for name in ("x", "dx")}
        arrays.update({name: runtime.ndarray("f32", 1) for name in ("y", "dy")})
        start = time.perf_counter()
        call = module.prepare_execution("step", arrays)
        costs["prepare_ms"] = (time.perf_counter() - start) * 1000
        for scale in (1, -2, 0, 1):
            values = [component for value in (1, -2, 0, 3) for component in (value * scale, 9, 8, 7)]
            arrays["x"].write(struct.pack("<16f", *values))
            call()
            expected = [component for value in (1, -2, 0, 3) for component in (max(value * scale, 0), 0, 0, 0)]
            assert list(struct.unpack("<16f", arrays["dx"].read())) == expected
            assert struct.unpack("<f", arrays["y"].read())[0] == 0.5 * sum(value * value for value in expected)
        samples = []
        for _ in range(8):
            start = time.perf_counter_ns()
            call()
            runtime.wait()
            samples.append((time.perf_counter_ns() - start) / 1000)
        costs["warm_complete_us"] = statistics.median(samples)
        call.close()
        for array in arrays.values():
            array.close()
    assert "taichi_forge" not in sys.modules
    print(json.dumps(costs))


if __name__ == "__main__":
    main(*sys.argv[1:])
