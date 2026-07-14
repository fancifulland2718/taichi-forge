#!/usr/bin/env python3
"""Validate that a taichi-forge shim wheel contains no runtime payload."""

from __future__ import annotations

import argparse
from email.parser import Parser
from pathlib import Path
import re
from zipfile import ZipFile


PROJECT = "taichi-forge"
RUNTIME_PROJECT = "taichi-forge-runtime"
REQUIRED_PYTHON_DEPENDENCIES = frozenset(
    {"colorama", "dill", "numpy", "rich", RUNTIME_PROJECT}
)
LINUX_LLVM_ABI_SENTINELS = (
    b"_ZN4llvm23EnableABIBreakingChecksE",
    b"_ZN4llvm24DisableABIBreakingChecksE",
)
CUDA_VARIANT = re.compile(r"(?:^|[+_.-])(?:cu|cuda)\d+", re.IGNORECASE)


def _requirement_project(requirement: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
    if match is None:
        raise RuntimeError(f"Invalid Requires-Dist entry: {requirement!r}")
    return re.sub(r"[-_.]+", "-", match.group(1)).lower()


def _wheel_platform(wheel: Path) -> str:
    name = wheel.name.lower()
    if "win_amd64" in name:
        return "windows"
    if "manylinux" in name:
        return "manylinux"
    raise RuntimeError(f"Unsupported shim wheel platform tag: {wheel.name}")


def validate_shim_wheel(wheel: Path, expected_platform: str) -> str:
    platform = _wheel_platform(wheel)
    if platform != expected_platform:
        raise RuntimeError(
            f"Expected a {expected_platform} shim wheel, found {platform}: "
            f"{wheel.name}"
        )
    if not wheel.name.startswith("taichi_forge-"):
        raise RuntimeError(f"Unexpected shim distribution name: {wheel.name}")

    with ZipFile(wheel) as zf:
        corrupt = zf.testzip()
        if corrupt is not None:
            raise RuntimeError(f"Corrupt wheel member in {wheel.name}: {corrupt}")
        names = zf.namelist()
        metadata_names = [
            name for name in names if name.endswith(".dist-info/METADATA")
        ]
        record_names = [
            name for name in names if name.endswith(".dist-info/RECORD")
        ]
        if len(metadata_names) != 1 or len(record_names) != 1:
            raise RuntimeError(
                f"Expected one METADATA and RECORD in {wheel.name}, found "
                f"metadata={metadata_names}, record={record_names}"
            )
        metadata = Parser().parsestr(
            zf.read(metadata_names[0]).decode("utf-8", errors="replace")
        )
        project = metadata.get("Name")
        version = metadata.get("Version") or ""
        if project != PROJECT:
            raise RuntimeError(f"Unexpected wheel project {project!r}: {wheel.name}")
        if CUDA_VARIANT.search(version):
            raise RuntimeError(
                f"CUDA-versioned shim wheel versions are forbidden: {version}"
            )
        requirements = metadata.get_all("Requires-Dist", [])
        requirements_by_project = {}
        for requirement in requirements:
            requirements_by_project.setdefault(
                _requirement_project(requirement), []
            ).append(requirement)
        missing_dependencies = sorted(
            REQUIRED_PYTHON_DEPENDENCIES - requirements_by_project.keys()
        )
        if missing_dependencies:
            raise RuntimeError(
                "Missing required Python dependencies in "
                f"{wheel.name}: {', '.join(missing_dependencies)}"
            )
        expected_runtime = f"{RUNTIME_PROJECT}=={version}"
        runtime_requirements = requirements_by_project[RUNTIME_PROJECT]
        if runtime_requirements != [expected_runtime]:
            raise RuntimeError(
                f"Expected runtime dependency {expected_runtime!r} in {wheel.name}, "
                f"found {runtime_requirements}"
            )

        extension_suffix = ".pyd" if platform == "windows" else ".so"
        extensions = [
            name
            for name in names
            if name.startswith("taichi_forge/_lib/core/")
            and name.endswith(extension_suffix)
        ]
        if len(extensions) != 1:
            raise RuntimeError(
                f"Expected one pybind extension in {wheel.name}, found {extensions}"
            )
        if platform == "manylinux":
            extension = zf.read(extensions[0])
            abi_sentinels = [
                symbol.decode("ascii")
                for symbol in LINUX_LLVM_ABI_SENTINELS
                if symbol in extension
            ]
            if abi_sentinels:
                raise RuntimeError(
                    "Linux shim retains LLVM ABI link sentinels despite its "
                    f"header-only LLVM boundary: {abi_sentinels}"
                )

        forbidden = []
        for name in names:
            leaf = Path(name).name.lower()
            if (
                name.startswith("taichi_forge_runtime/")
                or "taichi_runtime" in leaf
                or "cudart" in leaf
                or "slim_libdevice" in leaf
                or leaf in {"runtime_cuda.bc", "runtime_x64.bc"}
                or leaf.endswith(".lib")
            ):
                forbidden.append(name)
        if forbidden:
            raise RuntimeError(
                f"Shim wheel duplicates runtime artifacts: {forbidden}"
            )
    return version


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel-dir", type=Path, required=True)
    parser.add_argument(
        "--platform", choices=["windows", "manylinux"], required=True
    )
    args = parser.parse_args()

    wheels = sorted(args.wheel_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(
            f"Expected one shim wheel in {args.wheel_dir}, "
            f"found {[wheel.name for wheel in wheels]}"
        )
    try:
        version = validate_shim_wheel(wheels[0], args.platform)
    except (OSError, RuntimeError, UnicodeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        f"Validated {wheels[0].name}: platform={args.platform}, "
        f"version={version}, runtime payload duplicates=0"
    )


if __name__ == "__main__":
    main()
