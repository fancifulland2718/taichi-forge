"""Manual CPU/CUDA/Vulkan allocation-lifecycle stress for release artifacts.

This intentionally creates and resets independent Taichi programs repeatedly.
It exercises field allocation, host upload/readback, synchronization, and
backend clear/destruction rather than application-level data sharing.
"""

from __future__ import annotations

import argparse
import gc
import time

import numpy as np
import taichi_forge as ti


ARCHES = {
    "cpu": ti.cpu,
    "cuda": ti.cuda,
    "vulkan": ti.vulkan,
}


def run_arch(name: str, arch, iterations: int, elements: int) -> dict:
    expected = np.arange(elements, dtype=np.int32) * 3 + 7
    samples = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        ti.init(arch=arch)
        data = ti.field(dtype=ti.i32, shape=elements)
        data.from_numpy(expected)

        @ti.kernel
        def verify_and_update():
            for i in data:
                data[i] = data[i] * 2 - 1

        verify_and_update()
        ti.sync()
        result = data.to_numpy()
        np.testing.assert_array_equal(result, expected * 2 - 1)
        del data
        ti.reset()
        gc.collect()
        samples.append(time.perf_counter() - t0)

    values_ms = np.asarray(samples, dtype=np.float64) * 1e3
    return {
        "arch": name,
        "iterations": iterations,
        "median_ms": float(np.median(values_ms)),
        "p95_ms": float(np.percentile(values_ms, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--elements", type=int, default=1 << 15)
    parser.add_argument("--arches", nargs="+", default=list(ARCHES))
    args = parser.parse_args()
    if args.iterations <= 0 or args.elements <= 0:
        raise ValueError("iterations and elements must be positive")

    for name in args.arches:
        if name not in ARCHES:
            raise ValueError(f"unsupported arch: {name}")
        print(run_arch(name, ARCHES[name], args.iterations, args.elements))


if __name__ == "__main__":
    main()
