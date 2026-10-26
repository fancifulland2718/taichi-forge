"""Bounded cold/warm CPU comparison; run with each isolated runtime in turn."""
import argparse
import json
import statistics
import time

import numpy as np
import taichi_forge as ti

parser = argparse.ArgumentParser()
parser.add_argument("output")
args = parser.parse_args()
ti.init(arch=ti.cpu, cpu_max_num_threads=16, offline_cache=False)
results = []
for size in (1024, 65536):
    x = ti.ndarray(ti.f32, size)
    y = ti.ndarray(ti.f32, size)
    values = np.linspace(-1, 1, size, dtype=np.float32)
    x.from_numpy(values)
    for heavy in (False, True):
        @ti.kernel
        def work(src: ti.types.ndarray(ti.f32, 1), dst: ti.types.ndarray(ti.f32, 1)):
            for i in range(size):
                value = src[i]
                if ti.static(heavy):
                    for j in range(16):
                        value = ti.sin(value) * 0.9 + 0.1
                else:
                    value = value * 2 + 1
                dst[i] = value

        started = time.perf_counter_ns()
        work(x, y)
        cold = (time.perf_counter_ns() - started) / 1e6
        reference = values.copy()
        if heavy:
            for _ in range(16):
                reference = np.sin(reference) * np.float32(.9) + np.float32(.1)
        else:
            reference = reference * 2 + 1
        np.testing.assert_allclose(y.to_numpy(), reference, atol=2e-6, rtol=2e-6)
        for _ in range(30):
            work(x, y)
        samples = []
        for _ in range(7):
            started = time.perf_counter_ns()
            for _ in range(200):
                work(x, y)
            samples.append((time.perf_counter_ns() - started) / 200 / 1e3)
        ti.profiler.set_cpu_scheduler_telemetry(True, reset=True)
        work(x, y)
        telemetry = ti.profiler.query_cpu_scheduler_telemetry()
        ti.profiler.set_cpu_scheduler_telemetry(False)
        results.append(dict(size=size, heavy=heavy, first_call_ms=cold,
                            warm_us=statistics.median(samples), samples_us=samples,
                            telemetry=telemetry))
with open(args.output, "w", encoding="utf-8") as output:
    json.dump(results, output, indent=2)
print(json.dumps(results, indent=2))
