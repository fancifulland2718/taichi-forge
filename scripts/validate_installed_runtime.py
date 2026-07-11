#!/usr/bin/env python3
"""Validate an installed taichi-forge runtime/shim wheel pair."""

import os
import platform
import re
from pathlib import Path

import numpy as np

import taichi_forge as ti


def _packaged_cuda_runtime_major(path: Path) -> int:
    name = path.name.lower()
    if platform.system() == "Windows":
        match = re.fullmatch(r"cudart64_(\d+)\.dll", name)
    elif platform.system() == "Linux":
        match = re.fullmatch(
            r"(?:libcudart\.so\.|libcudart-[^.]+\.so\.)(\d+)(?:\.\d+)*",
            name,
        )
    else:
        raise RuntimeError(
            f"unsupported platform for bundled CUDART validation: {platform.system()}"
        )
    if match is None:
        raise RuntimeError(f"unrecognized bundled CUDART name: {path.name}")
    return int(match.group(1))


def _validate_packaged_cuda_runtime() -> tuple[Path, int]:
    candidate = os.environ.get("TI_CUDA_CUB_SORT_BUNDLED_CUDART_PATH", "")
    if not candidate:
        raise RuntimeError("the installed runtime did not discover bundled CUDART")
    path = Path(candidate)
    if not path.is_file():
        raise RuntimeError(f"the discovered bundled CUDART does not exist: {path}")

    major = _packaged_cuda_runtime_major(path)
    declared_major = os.environ.get(
        "TI_CUDA_CUB_SORT_BUNDLED_CUDART_MAJOR", ""
    )
    if declared_major:
        try:
            declared_major_value = int(declared_major)
        except ValueError as exc:
            raise RuntimeError(
                f"invalid bundled CUDART manifest major: {declared_major!r}"
            ) from exc
        if declared_major_value != major:
            raise RuntimeError(
                "bundled CUDART manifest/library mismatch: "
                f"manifest={declared_major_value}, library={path.name}"
            )
    return path, major


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
    cudart, cudart_major = _validate_packaged_cuda_runtime()
    _validate_cpu_native_ad()
    print(
        "installed runtime validation passed; "
        f"bundled CUDART major {cudart_major}: {cudart}"
    )


if __name__ == "__main__":
    main()
