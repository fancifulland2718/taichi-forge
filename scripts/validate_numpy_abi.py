#!/usr/bin/env python3
"""Exercise the installed wheel's NumPy/pybind buffer boundary."""

from __future__ import annotations

import argparse
import json
import platform
import sys

import numpy as np
import taichi_forge as ti


def _validate_round_trip(dtype, ti_dtype) -> None:
    shape = (17, 9)
    source = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    if np.issubdtype(dtype, np.floating):
        source = source * dtype.type(0.25) - dtype.type(3.0)

    ndarray = ti.ndarray(dtype=ti_dtype, shape=shape)
    ndarray.from_numpy(source)
    np.testing.assert_array_equal(ndarray.to_numpy(), source)

    field = ti.field(dtype=ti_dtype, shape=shape)
    field.from_numpy(source)
    np.testing.assert_array_equal(field.to_numpy(), source)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-major", type=int, choices=(1, 2))
    args = parser.parse_args()

    numpy_major = int(np.__version__.split(".", 1)[0])
    if args.expected_major is not None and numpy_major != args.expected_major:
        raise RuntimeError(
            f"Expected NumPy {args.expected_major}.x, found {np.__version__}"
        )

    ti.init(arch=ti.cpu, offline_cache=False)
    try:
        _validate_round_trip(np.dtype(np.int32), ti.i32)
        _validate_round_trip(np.dtype(np.float32), ti.f32)
        _validate_round_trip(np.dtype(np.float64), ti.f64)

        non_contiguous = np.arange(64, dtype=np.float32).reshape(8, 8)[:, ::2]
        ndarray = ti.ndarray(dtype=ti.f32, shape=non_contiguous.shape)
        ndarray.from_numpy(non_contiguous)
        np.testing.assert_array_equal(ndarray.to_numpy(), non_contiguous)
    finally:
        ti.reset()

    print(
        "NUMPY_ABI_VALIDATION "
        + json.dumps(
            {
                "numpy": np.__version__,
                "numpy_major": numpy_major,
                "python": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "python_executable": sys.executable,
                "status": "passed",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
