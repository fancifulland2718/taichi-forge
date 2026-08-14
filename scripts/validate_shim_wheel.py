#!/usr/bin/env python3
"""Validate that a taichi-forge shim wheel contains no runtime payload."""

from __future__ import annotations

import argparse
from email.parser import Parser
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
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
LINUX_SHARED_CPP_RUNTIME_LIBRARIES = (
    b"libstdc++.so.6",
    b"libgcc_s.so.1",
)
LINUX_RUNTIME_DEPENDENCY = b"libtaichi_runtime.so"
LINUX_RUNTIME_RPATH = b"taichi_forge_runtime/_lib/runtime_native"
MACOS_RUNTIME_DEPENDENCY = b"libtaichi_runtime.dylib"
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
    if "macosx" in name:
        return "macos"
    raise RuntimeError(f"Unsupported shim wheel platform tag: {wheel.name}")


def _strict_dynamic_contract(
    zf: ZipFile, extension_member: str, platform: str
) -> None:
    with tempfile.TemporaryDirectory(prefix="taichi-shim-dynamic-audit-") as td:
        extension = Path(td) / Path(extension_member).name
        extension.write_bytes(zf.read(extension_member))
        if platform == "manylinux":
            tool = shutil.which("readelf")
            if tool is None:
                raise RuntimeError("strict Linux shim audit requires readelf")
            command = [tool, "-d", str(extension)]
            required_dependency = "libtaichi_runtime.so"
            required_path = "taichi_forge_runtime/_lib/runtime_native"
        elif platform == "windows":
            tool = shutil.which("dumpbin")
            if tool is None:
                raise RuntimeError("strict Windows shim audit requires dumpbin")
            command = [tool, "/nologo", "/dependents", str(extension)]
            required_dependency = "taichi_runtime.dll"
            required_path = None
        else:
            tool = shutil.which("otool")
            if tool is None:
                raise RuntimeError("strict macOS shim audit requires otool")
            command = [tool, "-l", str(extension)]
            required_dependency = "libtaichi_runtime.dylib"
            required_path = "taichi_forge_runtime/_lib/runtime_native"
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "strict shim dynamic-table audit failed: "
                f"{completed.stdout.strip()}"
            )
        if required_dependency not in completed.stdout:
            raise RuntimeError(
                "final shim binary lacks its direct runtime dependency: "
                f"{required_dependency}"
            )
        if required_path is not None and required_path not in completed.stdout:
            raise RuntimeError(
                "final shim binary lacks its package-relative runtime search path"
            )


def validate_shim_wheel(
    wheel: Path, expected_platform: str, strict_binary: bool = False
) -> str:
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
        if strict_binary:
            _strict_dynamic_contract(zf, extensions[0], platform)
        if platform == "manylinux":
            extension = zf.read(extensions[0])
            if LINUX_RUNTIME_DEPENDENCY not in extension:
                raise RuntimeError(
                    "Linux split shim has no DT_NEEDED marker for "
                    "libtaichi_runtime.so"
                )
            if LINUX_RUNTIME_RPATH not in extension:
                raise RuntimeError(
                    "Linux split shim has no package-relative runtime RUNPATH"
                )
            missing_cpp_runtime_libraries = [
                library.decode("ascii")
                for library in LINUX_SHARED_CPP_RUNTIME_LIBRARIES
                if library not in extension
            ]
            if missing_cpp_runtime_libraries:
                raise RuntimeError(
                    "Linux split shim must share the C++ runtime used by "
                    "libtaichi_runtime.so; missing dynamic dependency "
                    f"markers: {missing_cpp_runtime_libraries}"
                )
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
        elif platform == "macos":
            extension = zf.read(extensions[0])
            if MACOS_RUNTIME_DEPENDENCY not in extension:
                raise RuntimeError(
                    "macOS split shim has no load-command marker for "
                    "libtaichi_runtime.dylib"
                )
            if LINUX_RUNTIME_RPATH not in extension:
                raise RuntimeError(
                    "macOS split shim has no package-relative runtime rpath"
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
        "--platform", choices=["windows", "manylinux", "macos"], required=True
    )
    parser.add_argument(
        "--strict-binary",
        action="store_true",
        help="Inspect the final extension's dynamic dependency table",
    )
    args = parser.parse_args()

    wheels = sorted(args.wheel_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(
            f"Expected one shim wheel in {args.wheel_dir}, "
            f"found {[wheel.name for wheel in wheels]}"
        )
    try:
        version = validate_shim_wheel(
            wheels[0], args.platform, strict_binary=args.strict_binary
        )
    except (OSError, RuntimeError, UnicodeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        f"Validated {wheels[0].name}: platform={args.platform}, "
        f"version={version}, runtime payload duplicates=0"
    )


if __name__ == "__main__":
    main()
