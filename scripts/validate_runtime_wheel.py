#!/usr/bin/env python3
"""Validate taichi-forge-runtime wheel identity and bundled native artifacts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from email.parser import Parser
import hashlib
import json
from pathlib import Path
import re
from zipfile import ZipFile


PROJECT = "taichi-forge-runtime"
PACKAGE = "taichi_forge_runtime"
MANIFEST = f"{PACKAGE}/_lib/runtime_native/cuda_runtime_major.txt"
WINDOWS_EXPORT_MANIFEST = (
    f"{PACKAGE}/_lib/runtime_native/taichi_runtime.exports.json"
)
CUDA_VARIANT = re.compile(r"(?:^|[+_.-])(?:cu|cuda)\d+", re.IGNORECASE)


@dataclass(frozen=True)
class RuntimeWheelInfo:
    path: Path
    platform: str
    version: str
    dependency_class: str
    cuda_major: int | None


def _wheel_platform(wheel: Path) -> str:
    name = wheel.name.lower()
    if "win_amd64" in name:
        return "windows"
    if "manylinux" in name:
        return "manylinux"
    if "linux_x86_64" in name:
        return "linux"
    raise RuntimeError(f"Unsupported runtime wheel platform tag: {wheel.name}")


def _cudart_major(platform: str, name: str) -> int | None:
    if platform == "windows":
        match = re.fullmatch(r"cudart64_(\d+)\.dll", name, re.IGNORECASE)
    else:
        match = re.fullmatch(
            r"libcudart(?:-[^.]+)?\.so\.(\d+)(?:\.\d+)*",
            name,
            re.IGNORECASE,
        )
    return int(match.group(1)) if match else None


def _export_digest(symbols: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update("\n".join(symbols).encode("utf-8"))
    return digest.hexdigest()


def _validate_windows_export_manifest(zf: ZipFile, names: list[str]) -> None:
    manifests = [name for name in names if name == WINDOWS_EXPORT_MANIFEST]
    if len(manifests) != 1:
        raise RuntimeError(
            f"Expected one {WINDOWS_EXPORT_MANIFEST}, found {manifests}"
        )
    try:
        payload = json.loads(zf.read(manifests[0]).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Invalid Windows runtime export manifest") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RuntimeError("Unsupported Windows runtime export manifest schema")
    if payload.get("dll_audited") is not True:
        raise RuntimeError("Windows runtime export manifest was not DLL-audited")

    requested = payload.get("exports")
    actual = payload.get("actual_exports")
    if (
        not isinstance(requested, list)
        or not isinstance(actual, list)
        or not all(isinstance(symbol, str) and symbol for symbol in requested)
        or not all(isinstance(symbol, str) and symbol for symbol in actual)
        or requested != sorted(set(requested))
        or actual != sorted(set(actual))
    ):
        raise RuntimeError("Windows runtime export sets are not canonical")

    requested_set = set(requested)
    actual_set = set(actual)
    required_count = payload.get("shim_required_runtime_symbol_count")
    raw_count = payload.get("raw_defined_symbol_count")
    limit = payload.get("configured_export_limit")
    counts = (
        required_count,
        raw_count,
        limit,
        payload.get("exported_symbol_count"),
        payload.get("actual_exported_symbol_count"),
        payload.get("implicit_exported_symbol_count"),
        payload.get("dropped_raw_symbol_count"),
    )
    if not all(isinstance(value, int) and value >= 0 for value in counts):
        raise RuntimeError("Windows runtime export counts are invalid")
    if required_count <= 0 or raw_count < required_count:
        raise RuntimeError("Windows runtime export closure is empty or inconsistent")
    if payload["exported_symbol_count"] != len(requested):
        raise RuntimeError("Windows requested export count is inconsistent")
    if payload["actual_exported_symbol_count"] != len(actual):
        raise RuntimeError("Windows actual export count is inconsistent")
    if payload["implicit_exported_symbol_count"] != len(
        actual_set - requested_set
    ):
        raise RuntimeError("Windows implicit export count is inconsistent")
    if payload["dropped_raw_symbol_count"] != raw_count - required_count:
        raise RuntimeError("Windows dropped export count is inconsistent")
    if not requested_set.issubset(actual_set):
        raise RuntimeError("Windows runtime DLL is missing requested exports")
    if "taichi_runtime_anchor" not in requested_set:
        raise RuntimeError("Windows runtime export manifest is missing its ABI anchor")
    if limit <= 0 or limit > 65_535 or len(actual) > limit:
        raise RuntimeError("Windows runtime export set exceeds its safety limit")
    if payload.get("export_set_sha256") != _export_digest(requested):
        raise RuntimeError("Windows requested export digest is inconsistent")
    if payload.get("actual_export_set_sha256") != _export_digest(actual):
        raise RuntimeError("Windows actual export digest is inconsistent")


def inspect_runtime_wheel(
    wheel: Path,
    expected_cuda_major: int | None = None,
    expected_dependency_class: str = "either",
) -> RuntimeWheelInfo:
    if expected_dependency_class not in {
        "driver-only",
        "toolkit-reference",
        "either",
    }:
        raise ValueError(
            "expected_dependency_class must be driver-only, "
            "toolkit-reference, or either"
        )
    platform = _wheel_platform(wheel)
    if not wheel.name.startswith("taichi_forge_runtime-"):
        raise RuntimeError(f"Unexpected runtime distribution name: {wheel.name}")

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
                f"CUDA-versioned runtime wheel versions are forbidden: {version}"
            )

        manifests = [name for name in names if name == MANIFEST]
        if len(manifests) > 1:
            raise RuntimeError(
                f"Expected at most one CUDA runtime manifest in {wheel.name}, "
                f"found {manifests}"
            )

        cudarts = []
        for name in names:
            major = _cudart_major(platform, Path(name).name)
            if major is not None:
                cudarts.append((name, major))
        runtime_native_prefix = f"{PACKAGE}/_lib/runtime_native/"
        auditwheel_prefix = f"{PACKAGE}.libs/"
        cuda_major = None
        if manifests:
            major_text = zf.read(manifests[0]).decode("ascii").strip()
            if not major_text.isdigit() or int(major_text) <= 0:
                raise RuntimeError(
                    f"Invalid CUDA runtime manifest in {wheel.name}: "
                    f"{major_text!r}"
                )
            cuda_major = int(major_text)
            if len(cudarts) != 1 or cudarts[0][1] != cuda_major:
                raise RuntimeError(
                    f"Expected one CUDART matching manifest major {cuda_major} "
                    f"in {wheel.name}, found {cudarts}"
                )
            cudart_path = cudarts[0][0]
            if not (
                cudart_path.startswith(runtime_native_prefix)
                or (
                    platform == "manylinux"
                    and cudart_path.startswith(auditwheel_prefix)
                )
            ):
                raise RuntimeError(
                    f"CUDART is outside the runtime package in {wheel.name}: "
                    f"{cudart_path}"
                )
            dependency_class = "toolkit-reference"
        else:
            if cudarts:
                raise RuntimeError(
                    f"Driver-only runtime wheel {wheel.name} contains CUDART "
                    f"without a manifest: {cudarts}"
                )
            dependency_class = "driver-only"

        if (
            expected_dependency_class != "either"
            and dependency_class != expected_dependency_class
        ):
            raise RuntimeError(
                f"Runtime dependency class mismatch in {wheel.name}: "
                f"expected={expected_dependency_class}, actual={dependency_class}"
            )
        if expected_cuda_major is not None and cuda_major != expected_cuda_major:
            raise RuntimeError(
                f"CUDA runtime major mismatch in {wheel.name}: "
                f"expected={expected_cuda_major}, manifest={cuda_major}"
            )

        if platform == "windows":
            native_runtimes = [
                name
                for name in names
                if name == f"{PACKAGE}/_lib/runtime_native/taichi_runtime.dll"
            ]
        else:
            native_runtimes = [
                name
                for name in names
                if name == f"{PACKAGE}/_lib/runtime_native/libtaichi_runtime.so"
                or (
                    platform == "manylinux"
                    and (
                        name.startswith(runtime_native_prefix)
                        or name.startswith(auditwheel_prefix)
                    )
                    and re.fullmatch(
                        r"libtaichi_runtime-[^.]+\.so", Path(name).name
                    )
                )
            ]
        if len(native_runtimes) != 1:
            raise RuntimeError(
                f"Expected one platform native runtime in {wheel.name}, "
                f"found {native_runtimes}"
            )
        if platform == "windows":
            import_library = f"{PACKAGE}/_lib/runtime_native/taichi_runtime.lib"
            if names.count(import_library) != 1:
                raise RuntimeError(
                    f"Expected one {import_library} in {wheel.name}"
                )
            _validate_windows_export_manifest(zf, names)

        wrong_entries = [
            name for name in names if name.startswith("taichi_forge/_lib/")
        ]
        if wrong_entries:
            raise RuntimeError(
                f"Runtime artifacts were installed into the shim package: "
                f"{wrong_entries}"
            )

    return RuntimeWheelInfo(
        wheel, platform, version, dependency_class, cuda_major
    )


def validate_runtime_wheels(
    wheel_dir: Path,
    expected_platform: str,
    expected_cuda_major: int | None = None,
    expected_dependency_class: str = "either",
) -> list[RuntimeWheelInfo]:
    wheels = sorted(wheel_dir.glob("*.whl"))
    expected_count = 2 if expected_platform == "pair" else 1
    if len(wheels) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} runtime wheel(s) in {wheel_dir}, "
            f"found {[wheel.name for wheel in wheels]}"
        )
    infos = [
        inspect_runtime_wheel(
            wheel,
            expected_cuda_major,
            expected_dependency_class,
        )
        for wheel in wheels
    ]
    if expected_platform == "pair":
        platforms = sorted(info.platform for info in infos)
        if platforms != ["manylinux", "windows"]:
            raise RuntimeError(
                "Expected one Windows and one manylinux runtime wheel, "
                f"found {platforms}"
            )
        versions = {info.version for info in infos}
        if len(versions) != 1:
            raise RuntimeError(f"Runtime wheel versions differ: {sorted(versions)}")
        dependency_classes = {info.dependency_class for info in infos}
        if len(dependency_classes) != 1:
            raise RuntimeError(
                "Runtime wheel dependency classes differ: "
                f"{sorted(dependency_classes)}"
            )
        cuda_majors = {info.cuda_major for info in infos}
        if len(cuda_majors) != 1:
            raise RuntimeError(
                "Runtime wheel CUDART majors differ: "
                f"{sorted(str(major) for major in cuda_majors)}"
            )
    elif infos[0].platform != expected_platform:
        raise RuntimeError(
            f"Expected a {expected_platform} runtime wheel, "
            f"found {infos[0].platform}: {infos[0].path.name}"
        )
    return infos


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel-dir", type=Path, required=True)
    parser.add_argument(
        "--platform",
        choices=["windows", "linux", "manylinux", "pair"],
        required=True,
    )
    parser.add_argument("--cuda-major", type=int)
    parser.add_argument(
        "--dependency-class",
        choices=["driver-only", "toolkit-reference", "either"],
        default="either",
        help=(
            "Required CUDA runtime dependency class. Standard release "
            "workflows must use driver-only; either preserves validation of "
            "already-published legacy wheels."
        ),
    )
    args = parser.parse_args()

    try:
        infos = validate_runtime_wheels(
            args.wheel_dir,
            args.platform,
            args.cuda_major,
            args.dependency_class,
        )
    except (OSError, RuntimeError, UnicodeError) as exc:
        raise SystemExit(str(exc)) from exc
    for info in infos:
        suffix = (
            f", bundled CUDART major={info.cuda_major}"
            if info.cuda_major is not None
            else ", bundled CUDART=none"
        )
        print(
            f"Validated {info.path.name}: platform={info.platform}, "
            f"version={info.version}, dependency={info.dependency_class}"
            f"{suffix}"
        )


if __name__ == "__main__":
    main()
