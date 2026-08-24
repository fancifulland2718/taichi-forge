#!/usr/bin/env python3
"""Measure NumPy-version-sensitive host-buffer paths without gating releases."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import taichi_forge as ti


def _measure(fn, *, warmups: int, repeats: int) -> dict[str, float | int]:
    for _ in range(warmups):
        fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - start) / 1_000.0)
    return {
        "samples": len(samples),
        "median_us": statistics.median(samples),
        "mean_us": statistics.fmean(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "cv": (
            statistics.pstdev(samples) / statistics.fmean(samples)
            if len(samples) > 1 and statistics.fmean(samples) > 0
            else 0.0
        ),
    }


def _run_size(size: int, *, warmups: int, repeats: int) -> dict:
    source = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    numpy_destination = np.empty_like(source)
    field = ti.field(dtype=ti.f32, shape=size)
    field.from_numpy(source)

    def numpy_copyto():
        np.copyto(numpy_destination, source)

    def from_numpy():
        field.from_numpy(source)
        ti.sync()

    def to_numpy():
        result = field.to_numpy()
        if result.shape != source.shape:
            raise RuntimeError("unexpected to_numpy shape")

    rows = {
        "numpy_copyto": _measure(numpy_copyto, warmups=warmups, repeats=repeats),
        "field_from_numpy": _measure(from_numpy, warmups=warmups, repeats=repeats),
        "field_to_numpy": _measure(to_numpy, warmups=warmups, repeats=repeats),
    }
    np.testing.assert_array_equal(field.to_numpy(), source)
    np.testing.assert_array_equal(numpy_destination, source)
    return {"size": size, "dtype": "float32", "operations": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=(1024, 1048576))
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--label", default="unlabeled")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats <= 0 or any(n <= 0 for n in args.sizes):
        raise ValueError("sizes/repeats must be positive and warmups non-negative")

    ti.init(arch=ti.cpu, offline_cache=False)
    try:
        payload = {
            "schema_version": 1,
            "label": args.label,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "arch": "cpu",
            "warmups": args.warmups,
            "repeats": args.repeats,
            "results": [
                _run_size(size, warmups=args.warmups, repeats=args.repeats)
                for size in args.sizes
            ],
        }
    finally:
        ti.reset()

    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print("NUMPY_ABI_BENCH " + json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
