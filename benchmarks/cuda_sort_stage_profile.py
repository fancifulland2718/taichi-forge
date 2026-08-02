"""Print CUDA Driver stable-sort stage timings for optimization work."""

import argparse

import numpy as np

import taichi_forge as ti


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1 << 20)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=5)
    args = parser.parse_args()
    if min(args.size, args.rounds, args.warmups) <= 0:
        parser.error("size, rounds, and warmups must be positive")

    ti.init(arch=ti.cuda, offline_cache=False, kernel_profiler=True)
    keys = ti.ndarray(ti.i32, shape=args.size)
    values = ti.ndarray(ti.i32, shape=args.size)
    workspace = ti.algorithms.SortWorkspace()
    rng = np.random.default_rng(20260803)
    source = rng.integers(
        np.iinfo(np.int32).min,
        np.iinfo(np.int32).max,
        size=args.size,
        dtype=np.int32,
    )
    payload = np.arange(args.size, dtype=np.int32)

    for _ in range(args.warmups):
        keys.from_numpy(source)
        values.from_numpy(payload)
        ti.algorithms.sort(keys, values, method="cuda_device", workspace=workspace)
    ti.sync()
    ti.profiler.clear_kernel_profiler_info()

    for _ in range(args.rounds):
        keys.from_numpy(source)
        values.from_numpy(payload)
        ti.algorithms.sort(keys, values, method="cuda_device", workspace=workspace)
    ti.sync()
    ti.profiler.print_kernel_profiler_info("count")


if __name__ == "__main__":
    main()
