#!/usr/bin/env python3
"""Validate an installed taichi-forge runtime/shim wheel pair."""

import os
import platform
import re
from importlib import metadata
from importlib.util import find_spec
from pathlib import Path

import numpy as np

import taichi_forge as ti


def _validate_distribution_versions() -> str:
    shim_version = metadata.version("taichi-forge")
    runtime_version = metadata.version("taichi-forge-runtime")
    if shim_version != runtime_version:
        raise RuntimeError(
            "installed shim/runtime version mismatch: "
            f"taichi-forge={shim_version}, "
            f"taichi-forge-runtime={runtime_version}"
        )
    return shim_version


def _runtime_package_dirs() -> list[Path]:
    spec = find_spec("taichi_forge_runtime")
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError("taichi_forge_runtime package is not importable")
    return [Path(path).resolve() for path in spec.submodule_search_locations]


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_cudart_belongs_to_runtime_package(path: Path) -> None:
    candidate_roots = []
    for package_dir in _runtime_package_dirs():
        candidate_roots.append(package_dir)
        candidate_roots.append(
            package_dir.parent / f"{package_dir.name}.libs"
        )
    resolved = path.resolve()
    if not any(
        root.is_dir() and _is_relative_to(resolved, root.resolve())
        for root in candidate_roots
    ):
        raise RuntimeError(
            "bundled CUDART was not loaded from taichi-forge-runtime: "
            f"{resolved}"
        )


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


def _packaged_cudart_candidates() -> list[Path]:
    candidates = []
    for package_dir in _runtime_package_dirs():
        roots = [package_dir, package_dir.parent / f"{package_dir.name}.libs"]
        for root in roots:
            if not root.is_dir():
                continue
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                try:
                    _packaged_cuda_runtime_major(path)
                except RuntimeError:
                    continue
                candidates.append(path.resolve())
    return sorted(set(candidates))


def _validate_packaged_cuda_runtime() -> tuple[Path | None, int | None]:
    candidate = os.environ.get("TI_CUDA_CUB_SORT_BUNDLED_CUDART_PATH", "")
    if not candidate:
        stray = _packaged_cudart_candidates()
        if stray:
            raise RuntimeError(
                "installed driver-only runtime contains undiscovered CUDART: "
                f"{stray}"
            )
        return None, None
    path = Path(candidate)
    if not path.is_file():
        raise RuntimeError(f"the discovered bundled CUDART does not exist: {path}")
    _validate_cudart_belongs_to_runtime_package(path)

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
    version = _validate_distribution_versions()
    cudart, cudart_major = _validate_packaged_cuda_runtime()
    _validate_cpu_native_ad()
    if cudart is None:
        dependency = "driver-only; bundled CUDART=none"
    else:
        dependency = f"legacy bundled CUDART major {cudart_major}: {cudart}"
    print(f"installed runtime validation passed for {version}; {dependency}")


if __name__ == "__main__":
    main()
