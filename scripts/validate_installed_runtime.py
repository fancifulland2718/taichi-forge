#!/usr/bin/env python3
"""Validate an installed taichi-forge runtime/shim wheel pair."""

import os
import platform
from pathlib import Path

import numpy as np

import taichi_forge as ti


def _validate_packaged_cuda_runtime() -> Path:
    candidate = os.environ.get("TI_CUDA_CUB_SORT_BUNDLED_CUDART_PATH", "")
    if not candidate:
        raise RuntimeError("the installed runtime did not discover bundled CUDART")
    path = Path(candidate)
    if not path.is_file():
        raise RuntimeError(f"the discovered bundled CUDART does not exist: {path}")

    system = platform.system()
    name = path.name.lower()
    if system == "Windows" and name != "cudart64_13.dll":
        raise RuntimeError(f"expected CUDA 13 CUDART on Windows, found {path.name}")
    if system == "Linux" and not (
        name.startswith("libcudart.so.13")
        or (name.startswith("libcudart-") and ".so.13" in name)
    ):
        raise RuntimeError(f"expected CUDA 13 CUDART on Linux, found {path.name}")
    return path


def _validate_cpu_native_ad() -> None:
    n = 8
    ti.init(arch=ti.cpu)
    x = ti.ndarray(ti.f32, shape=n, needs_grad=True)
    y = ti.ndarray(ti.f32, shape=n, needs_grad=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)
    x.from_numpy(np.arange(n, dtype=np.float32))

    @ti.kernel
    def sum_output(values: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for i in range(n):
            loss[None] += values[i]

    with ti.ad.Tape(loss):
        ti.algorithms.experimental_transform(
            x, y, scale=2.5, bias=1.0, method="cpu_native"
        )
        sum_output(y)

    np.testing.assert_allclose(x.grad.to_numpy(), np.full(n, 2.5, np.float32))
    ti.reset()


def main() -> None:
    cudart = _validate_packaged_cuda_runtime()
    _validate_cpu_native_ad()
    print(f"installed runtime validation passed; bundled CUDART: {cudart}")


if __name__ == "__main__":
    main()
